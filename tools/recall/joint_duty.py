"""Optional bounded joint replacement profiles. Acceptance belongs to run_group.

Scores are frozen TRAIN sentence scores. A deleted bank is INTERNAL ONLY.
Source admission conservatively excludes every source in the original bank.
"""
import collections
import itertools
import resource
import time
import numpy as np

FIELDS = {'mode', 'delete_sizes', 'maximum_delete_sets',
          'maximum_profile_candidates', 'maximum_replay_proposals', 'ordering'}

def validate(config):
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError('Unsupported joint_duty fields')
    if config['mode'] != 'joint_duty_v1':
        raise ValueError('Unsupported joint_duty mode')
    sizes = config['delete_sizes']
    if not isinstance(sizes, list) or not sizes or sizes != sorted(set(sizes)) or any(type(k) is not int or k not in (1, 2) for k in sizes):
        raise ValueError('delete_sizes must be sorted unique subset of [1,2]')
    for key, high in [('maximum_delete_sets', 4096), ('maximum_profile_candidates', 1000000), ('maximum_replay_proposals', 96)]:
        if type(config[key]) is not int or not 1 <= config[key] <= high:
            raise ValueError('Out of bounds: ' + key)
    if config['ordering'] not in ('shrink_then_repairs', 'repairs_then_shrink'):
        raise ValueError('Unsupported joint_duty ordering')
    return dict(config)

def admit(rt, bank, candidates, allowed=None):
    bank = set(map(int, bank)); candidates = sorted(set(map(int, candidates)))
    sources = set(rt.groups[list(bank)])
    excluded = set().union(*(set(row['excluded_sourcegroups']) for row in rt.rows))
    allowed = rt.clean if allowed is None else allowed
    if any(j not in rt.clean or j not in allowed or j in bank or rt.groups[j] in sources or rt.groups[j] in excluded for j in candidates):
        raise ValueError('Requires fresh clean TRAIN source outside original bank and all DEV closure')
    return candidates

def prepare_deleted(rt, bank, deleted):
    """Exact internal source ordering; never calls Replay or publishes a state."""
    bank=set(map(int,bank)); deleted=set(map(int,deleted))
    if not deleted or not deleted <= bank or len(deleted)>2:
        raise ValueError('Invalid deletion set')
    ids=np.array(sorted(bank-deleted),dtype=int)
    if not len(ids):raise ValueError('Empty internal bank')
    mat=ids[np.argsort(-rt.S[:,ids],axis=1,kind='stable')]
    alive=rt.eligible[rt.ii[:,None],mat].copy()
    for group,count in collections.Counter(rt.groups[ids]).items():
        if count>1:
            pos=rt.groups[mat]==group
            alive &= ~pos | (np.cumsum(pos & alive,axis=1)<=1)
    cumulative=np.cumsum(alive,axis=1)
    correct=alive & (rt.y[mat]==rt.gold[:,None])
    rank=np.where(correct,cumulative,100000).min(axis=1)
    best=np.where(rank<100000,mat[rt.ii,np.argmax(correct,axis=1)],-1)
    prefix=[mat[i,alive[i]][:50].tolist() for i in rt.ii]
    hit={k:rt.gold_legal & (rank<=k) for k in (15,50)}
    # Cache D-specific boundaries once, rather than rebuilding n-query lists for every j.
    boundaries={k:np.array([p[k-1] if len(p)>=k else -1 for p in prefix],dtype=int) for k in (15,50)}
    return dict(rank=rank,best=best,prefix=prefix,hit=hit,
                best_score=rt.S[rt.ii,np.maximum(best,0)],boundaries=boundaries,
                boundary_scores={k:rt.S[rt.ii,np.maximum(b,0)] for k,b in boundaries.items()},
                union={k:hit[k] | rt.desc_hit for k in (15,50)})

def predict_insert(rt, internal, candidate):
    """Exact four hit vectors including ties, masks, short banks and D5.

    Fresh-source admission is required at the caller. Illegal-label sources still
    consume positions exactly as frozen Replay. No epsilon/label-centroid proxy.
    """
    j=int(candidate); score=rt.S[:,j]; eligible=rt.eligible[:,j]
    same=(rt.y[j]==rt.gold) & rt.gold_legal & eligible
    best=internal['best']; bs=internal['best_score']
    before_best=(best<0) | (score>bs) | ((score==bs)&(j<best))
    hits={}
    for k in (15,50):
        boundary=internal['boundaries'][k]
        boundary_score=internal['boundary_scores'][k]
        enters=(boundary<0) | (score>boundary_score) | ((score==boundary_score)&(j<boundary))
        displaced=eligible & ~same & before_best & (internal['rank']==k)
        hits[k]=rt.gold_legal & ((internal['hit'][k] & ~displaced) | (same & enters))
    return dict(hit=hits,union={k:hits[k] | rt.desc_hit for k in (15,50)})

def deltas(anchor, prediction):
    return {name+str(k):dict(repairs=np.flatnonzero(~anchor[field][k]&prediction[field][k]).tolist(),
                            losses=np.flatnonzero(anchor[field][k]&~prediction[field][k]).tolist())
            for name,field in [('Q','hit'),('U','union')] for k in (15,50)}

def deletion_sets(rt, bank, sizes):
    byclass=collections.defaultdict(list)
    for j in sorted(bank):byclass[int(rt.y[j])].append(int(j))
    # Larger deletion sets first; then round-robin categories prevents a large
    # early class exhausting the entire finite deletion-set budget.
    for size in sorted(sizes,reverse=True):
        queues={c:iter(itertools.combinations(js,size)) for c,js in sorted(byclass.items()) if len(js)>1 and len(js)>=size}
        while queues:
            for c in list(queues):
                pair=next(queues[c],None)
                if pair is None:del queues[c]
                else:yield c,pair

def proposals(rt, anchor, bank, allowed, config, audit=None):
    config=validate(config);start=time.monotonic();cpu=time.process_time()
    bank=set(map(int,bank));sources=set(rt.groups[list(bank)])
    excluded=set().union(*(set(row['excluded_sourcegroups']) for row in rt.rows))
    # All classes with redundant representatives, regardless of Q15 misses.
    byclass=collections.defaultdict(list)
    for j in sorted(allowed):
        if j in rt.clean and j not in bank and rt.groups[j] not in sources and rt.groups[j] not in excluded:
            byclass[int(rt.y[j])].append(int(j))
    all_candidates=sorted(j for js in byclass.values() for j in js)
    sets=list(itertools.islice(deletion_sets(rt,bank,config['delete_sizes']),config['maximum_delete_sets']+1))
    set_truncated=len(sets)>config['maximum_delete_sets'];sets=sets[:config['maximum_delete_sets']]
    rows=[];prepared=0;count=0;truncated=False;pre_cpu=0.;pre_wall=0.;all_safe=0;set_audit=[]
    for label,deleted in sets:
        candidates=all_candidates
        if not candidates:continue
        remaining=config['maximum_profile_candidates']-count
        if remaining<len(candidates):
            # Never silently take an index prefix of TRAIN within a deletion set.
            truncated=True;break
        t=time.monotonic();c=time.process_time();internal=prepare_deleted(rt,bank,deleted)
        pre_cpu+=time.process_time()-c;pre_wall+=time.monotonic()-t;prepared+=1
        duty=deltas(anchor,internal);duty_count=sum(len(z['losses']) for z in duty.values());safe=0
        for j in candidates:
            prediction=predict_insert(rt,internal,j);delta=deltas(anchor,prediction);count+=1
            if any(z['losses'] for z in delta.values()):continue
            safe+=1;all_safe+=1
            q15=len(delta['Q15']['repairs']);q50=len(delta['Q50']['repairs'])
            priority=(-len(deleted),-q15,-q50,j,tuple(deleted)) if config['ordering']=='shrink_then_repairs' else (-q15,-q50,-len(deleted),j,tuple(deleted))
            rows.append((priority,dict(index=j,deleted=list(deleted),targets=delta['Q15']['repairs'],historical=1,prediction=delta,joint_duty_events=duty_count)))
            rows.sort(key=lambda z:z[0]);del rows[config['maximum_replay_proposals']:]
        set_audit.append(dict(label=int(label),deleted=list(deleted),TRAIN_candidates=len(candidates),joint_duty=duty,zero_regression_profiles=safe))
    elapsed=time.monotonic()-start;elapsed_cpu=time.process_time()-cpu
    if audit is not None:
        audit.append(dict(config=config,profiled_candidates=count,candidate_query_comparisons=count*len(rt.rows),profile_CPU_seconds=elapsed_cpu,profile_wall_seconds=elapsed,
            deleted_bank_preparations=prepared,deleted_bank_query_comparisons=prepared*len(rt.rows),
            deleted_bank_CPU_seconds=pre_cpu,deleted_bank_wall_seconds=pre_wall,
            deleted_bank_cost_is_subset_of_profile=True,profile_is_subset_of_total=True,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            eligible_by_class={str(c):len(js) for c,js in sorted(byclass.items())},
            delete_set_truncated=set_truncated,profile_budget_truncated=truncated,
            source_policy='exclude_original_bank_sources',candidate_policy='all_eligible_TRAIN_all_labels_within_each_completed_delete_set',
            zero_regression_profiles=all_safe,selected_count=len(rows),selection=[z[1] for z in rows],deletion_sets=set_audit,
            replay_new_states=0,evaluation_states_charged=0))
    return [z[1] for z in rows]

def submit(rt, bank, pool, test, round_number):
    """The runner's direct path into its ORIGINAL test/evaluate/gate closure."""
    for row in pool:
        j=int(row['index']);deleted=set(map(int,row['deleted']))
        if not deleted<=set(bank) or len(deleted) not in (1,2) or len({int(rt.y[d]) for d in deleted})!=1:
            raise ValueError('Invalid same-label final replacement')
        admit(rt,bank,[j])
        final=(set(bank)-deleted)|{j}
        test(final,'direct_joint_duty',metadata=dict(primary=j,layer=0,action='joint_replace',parent=None,
             path=[dict(action='joint_replace',add=j,delete=sorted(deleted))],
             direct_joint=True,profile_prediction=row.get('prediction'),round=round_number))

def gated_frontier(trials, round_number, bank_size):
    """Audit feasible alternatives without changing the original best selector."""
    feasible=[]
    for row in trials:
        if row['round']!=round_number or not row.get('direct_joint') or not row['feasible']:continue
        gains={k:sum(part['question_repairs'] for part in row['changes'][str(k)].values()) for k in (15,50)}
        feasible.append(dict(bank_sha256=row['bank_sha256'],added=row['added'],removed=row['removed'],
            final_size=bank_size+len(row['added'])-len(row['removed']),Q15_repairs=gains[15],Q50_repairs=gains[50]))
    def dominates(a,b):
        x=(-a['Q15_repairs'],-a['Q50_repairs'],a['final_size']);y=(-b['Q15_repairs'],-b['Q50_repairs'],b['final_size'])
        return all(i<=j for i,j in zip(x,y)) and any(i<j for i,j in zip(x,y))
    front=[a for a in feasible if not any(dominates(b,a) for b in feasible)]
    return dict(round=round_number,feasible_count=len(feasible),nondominated=front,
                smallest_feasible=min(feasible,key=lambda a:(a['final_size'],-a['Q15_repairs'],-a['Q50_repairs'],a['bank_sha256'])) if feasible else None,
                all_feasible=feasible,acceptance_order_unchanged=True)
