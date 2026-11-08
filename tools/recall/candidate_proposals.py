"""Optional deterministic proposal plugin. No Replay evaluations, transport or writes."""
import collections,time
import numpy as np

FIELDS={'generation','selection','tie_break'}
def validate_plugin(config):
    if not isinstance(config,dict) or set(config)-{'joint_duty'}!=FIELDS:raise ValueError('Unsupported proposal_plugin fields')
    if config['generation'] not in ('legacy_topn','all_train'):raise ValueError('Unsupported generation')
    if config['selection'] not in ('similarity','incremental_cover','class_round_robin','risk_aware_cover'):raise ValueError('Unsupported selection')
    if config['tie_break']!='source_diversity_then_index':raise ValueError('Unsupported tie_break')
    if 'joint_duty' in config:
        from joint_duty import validate
        validate(config['joint_duty'])
        if config['generation']!='all_train':raise ValueError('joint_duty requires all_train')
    return dict(config)

def legacy_proposals(rt,cur,ids,allowed,history,p,nearest):
    # Exact legacy body, kept independently testable against the unmodified owner.
    pool={}
    for i0 in np.flatnonzero(~cur['hit'][15]&rt.gold_legal):
        i=int(i0);origins={}
        if p['proposal_order']=='historical_then_similarity':
            for name in ['original8554','old4451','bank731','bank143']:
                b=history[i]['banks'][name];j=b['best_correct']
                if b['15']['question_hit'] and j in allowed and j not in ids:origins[j]=0
        for j in nearest(i,ids,p['candidates_per_error']):origins.setdefault(j,1)
        for j,origin in origins.items():
            row=pool.setdefault(j,dict(index=j,targets=[],historical=origin));row['targets'].append(i);row['historical']=min(row['historical'],origin)
    return sorted(pool.values(),key=lambda z:(z['historical'] if p['proposal_order']=='historical_then_similarity' else 0,-max(rt.S[i,z['index']] for i in z['targets']),z['index']))[:p['maximum_pool_candidates']]

def boundary_profiles(rt,cur,ids,candidates):
    """Exact fresh-source single-insert hit deltas, NOT joint policy results.

    Distinct sources and global DEV closure are admission preconditions. Illegal
    labels still consume source positions, as in the frozen Replay implementation.
    Equal scores are ordered by original TRAIN index, never approximate epsilon.
    """
    candidates=sorted(map(int,candidates));src=set(rt.groups[list(ids)])
    excluded=set().union(*(set(r['excluded_sourcegroups']) for r in rt.rows))
    if any(j in ids or rt.groups[j] in src or rt.groups[j] in excluded or j not in rt.clean for j in candidates):
        raise ValueError('Profiles require fresh-source clean TRAIN outside all DEV sources')
    miss=np.flatnonzero(~cur['hit'][15]&rt.gold_legal)
    boundary=np.array([p[14] if len(p)>=15 else -1 for p in cur['prefix']],dtype=int)
    bs=np.array([rt.S[i,j] if j>=0 else -np.inf for i,j in enumerate(boundary)])
    best=np.asarray(cur['best']);safe=np.maximum(best,0)
    bestscore=rt.S[rt.ii,safe]
    result={}
    for start in range(0,len(candidates),256):
        jj=np.array(candidates[start:start+256],dtype=int)
        scores=np.asarray(rt.S[:,jj])
        before_boundary=(scores>bs[:,None])|((scores==bs[:,None])&(jj[None,:]<boundary[:,None]))|(boundary[:,None]<0)
        same=rt.y[jj][None,:]==rt.gold[:,None]
        support=before_boundary&same&(~cur['hit'][15]&rt.gold_legal)[:,None]
        before_best=(scores>bestscore[:,None])|((scores==bestscore[:,None])&(jj[None,:]<best[:,None]))
        risk={}
        for k in (15,50):
            qrisk=(cur['hit'][k]&(cur['rank']==k))[:,None]&~same&before_best
            risk['Q'+str(k)]=qrisk
            risk['U'+str(k)]=qrisk&~rt.desc_hit[:,None]
        for col,j in enumerate(jj):
            result[int(j)]=dict(index=int(j),support=np.flatnonzero(support[:,col]).tolist(),
                risks={name:np.flatnonzero(mat[:,col]).tolist() for name,mat in risk.items()})
    return result

def generate(rt,cur,ids,allowed,history,p,nearest,config):
    config=validate_plugin(config)
    misses=list(map(int,np.flatnonzero(~cur['hit'][15]&rt.gold_legal)))
    src=set(rt.groups[list(ids)]);eligible=sorted(j for j in allowed if j not in ids and rt.groups[j] not in src)
    targets=collections.defaultdict(list)
    for i in misses:targets[int(rt.gold[i])].append(i)
    if config['generation']=='all_train':
        pool=[dict(index=j,targets=targets[int(rt.y[j])].copy(),historical=1) for j in eligible if int(rt.y[j]) in targets]
    else:
        full=dict(p,maximum_pool_candidates=len(rt.clean))
        pool=legacy_proposals(rt,cur,ids,allowed,history,full,nearest)
        # Explicit plugin admission is stricter for historical-origin same-source keys.
        good=set(eligible);pool=[z for z in pool if z['index'] in good]
    return pool

def select(rt,cur,p,pool,profiles,config):
    config=validate_plugin(config);remaining={z['index']:z for z in pool};out=[];covered=set();sources=set()
    scores={j:max(float(rt.S[i,j]) for i in z['targets']) for j,z in remaining.items()}
    def key(j):
        diversity=int(str(rt.groups[j]) in sources)
        if config['selection']=='risk_aware_cover':
            support=set(profiles[j]['support']);risk=profiles[j]['risks']
            counts=tuple(len(risk[n]) for n in ('Q15','Q50','U15','U50'))
            safe_positive=bool(support) and not any(counts)
            return (not safe_positive,sum(counts),*counts,-len(support-covered),-len(support),-scores[j],diversity,j)
        if config['selection']=='similarity':
            hist=remaining[j]['historical'] if p['proposal_order']=='historical_then_similarity' else 0
            return hist,-scores[j],diversity,j
        return -len(set(profiles[j]['support'])-covered),-scores[j],diversity,j
    def take(j):
        row=remaining.pop(j);out.append(row);covered.update(profiles[j]['support']);sources.add(str(rt.groups[j]))
    cap=p['maximum_pool_candidates']
    if config['selection']=='class_round_robin':
        counts=collections.Counter(int(rt.gold[i]) for i in np.flatnonzero(~cur['hit'][15]&rt.gold_legal))
        classes=sorted({int(rt.y[j]) for j in remaining},key=lambda c:(-counts[c],c))
        byclass={c:set(j for j in remaining if int(rt.y[j])==c) for c in classes}
        while remaining and len(out)<cap:
            for c in classes:
                if byclass[c]:
                    j=min(byclass[c],key=key);take(j);byclass[c].remove(j)
                    if len(out)>=cap:break
    else:
        while remaining and len(out)<cap:take(min(remaining,key=key))
    return out

def proposals(rt,cur,ids,allowed,history,p,nearest,audit=None):
    started=time.monotonic()
    cpu_started=time.process_time()
    config=validate_plugin(p['proposal_plugin'])
    pool=generate(rt,cur,ids,allowed,history,p,nearest,config)
    profile_started=time.monotonic();profile_cpu_started=time.process_time()
    profiles=boundary_profiles(rt,cur,ids,[z['index'] for z in pool])
    profile_wall=time.monotonic()-profile_started;profile_cpu=time.process_time()-profile_cpu_started
    selected=select(rt,cur,p,pool,profiles,config)
    if audit is not None:
        covered=set();selection=[]
        for z in selected:
            j=z['index'];support=set(profiles[j]['support'])
            selection.append(dict(index=j,label=rt.labels[int(rt.y[j])],targets=z['targets'],support=sorted(support),incremental_support=sorted(support-covered),risks=profiles[j]['risks']))
            covered.update(support)
        audit.append(dict(config=config,generated=len(pool),selected_count=len(selected),selection=selection,wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu_started,profiled_candidates=len(profiles),candidate_query_comparisons=len(profiles)*len(rt.rows),profile_CPU_seconds=profile_cpu,profile_wall_seconds=profile_wall,replay_new_states=0,evaluation_states_charged=0,budget_note='Batch single-insertion counterfactual profiles have separate exposure and CPU/wall charges; not free and not equivalent to a Replay-only budget'))
    return selected
