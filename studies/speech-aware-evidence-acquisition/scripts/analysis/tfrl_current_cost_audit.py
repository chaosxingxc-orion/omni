"""Attribute verified real K4 contacts, duplicate answers and latency to each arm.

Read-only: no generation, tokenizer or scoring. The CLI reloads complete source
custody and official score receipts before exposing aggregate diagnostics.
"""
import argparse
from collections import Counter,defaultdict
import hashlib,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis'),str(ROOT/'scripts/flights/nx')]
from tfrl_current_sources import load_half,read,sha
from tfrl_current_half_result import ARMS
import stage3_rollout_score as s3

FIELDS=('calls','prompt_tokens','completion_tokens','audio_seconds','latency_seconds')


def stats(values):
    values=sorted(values)
    return {'n':len(values),'mean':statistics.mean(values),'p50':statistics.median(values),
            'p95':values[int(.95*(len(values)-1))],'sum':sum(values)} if values else None


def summarize(bundle,credit,events):
    """Require exact contact/arm/response joins; never infer arms from order."""
    totals=defaultdict(Counter);measurements=defaultdict(lambda:defaultdict(list))
    duplicates=Counter();patterns=Counter();credit_cost=defaultdict(Counter)
    credits={v['row_id']:v for v in credit}
    ids=[r['sample_id'] for r in bundle.outputs_rows]
    if len(ids)!=len(set(ids)) or len(credits)!=len(credit) or set(credits)!=set(ids):
        raise ValueError('exact unique credit and output rows required')
    timing={}
    for event in events:
        if event['stage']!='request_total' or event.get('sample_id') not in credits:continue
        key=(event['sample_id'],event.get('request_id'))
        if key in timing:raise ValueError('duplicate request timing')
        timing[key]=event
    seen=set();missing=0;abstentions=0
    for row in bundle.outputs_rows:
        sid=row['sample_id'];episode=row['native_episode_receipt']
        visits=episode['role_visits']
        arms=row.get('supply_variants')
        if arms is None:
            from tfrl_seed_request_audit import no_solver_abstention
            if not no_solver_abstention(row):raise ValueError('missing arm custody')
            labels={};abstentions+=1
        else:
            if [a['arm_id'] for a in arms['arms']]!=list(ARMS):raise ValueError('exact K4 arm roster required')
            labels={a['contact_id']:a['arm_id'] for a in arms['arms']}
            if len(labels)!=6:raise ValueError('duplicate arm contact')
        row_labels=set();answers={};row_cost=Counter();row_requests=[]
        for visit in visits:
            if visit['role_id'] not in ('perception','solver'):raise ValueError('unexpected model role')
            for contact in visit['execution_receipt']['contacts']:
                cid=contact['agent_contact_id'];rid=contact['request_id']
                label='perception' if visit['role_id']=='perception' else labels.get(cid)
                if not label or label in row_labels or rid in seen:raise ValueError('ambiguous arm/request identity')
                row_labels.add(label);seen.add(rid);row_requests.append(rid)
                request=bundle.trace_records[rid+'/request'];response=bundle.trace_records[rid+'/response']
                cost=bundle.trace_records[rid+'/cost'];raw=response['text']
                if (request['payload']['sample_id']!=sid or request['payload_sha256']!=contact['request_sha256']
                    or hashlib.sha256(raw.encode()).hexdigest()!=contact['response_text_sha256']
                    or response['response_text_sha256']!=contact['response_text_sha256']):
                    raise ValueError('contact source binding differs')
                for field in FIELDS:
                    value=cost[field]
                    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                        raise ValueError('invalid cost')
                    totals[label][field]+=value;row_cost[field]+=value
                    measurements[label][field].append(value)
                if cost['calls']!=1:raise ValueError('one generation per contact required')
                totals[label]['contacts']+=1
                totals[label]['finish:'+str(cost.get('finish_reason'))]+=1
                totals[label]['forced_close']+=bool(cost.get('forced_close'))
                if cost.get('reasoning_tokens') is not None:
                    value=cost['reasoning_tokens']
                    if type(value) is not int or not 0<=value<=cost['completion_tokens']:raise ValueError('invalid reasoning count')
                    totals[label]['reasoning_tokens']+=value;totals[label]['reasoning_count_observed']+=1
                event=timing.get((sid,rid))
                if event is None:missing+=1
                else:measurements[label]['request_total_seconds'].append(event['elapsed_seconds'])
                if label!='perception':
                    normalized,parsed=s3.normalise_arm_answer(raw,(episode.get('deliver_normaliser') or {}).get('normaliser_id'))
                    answers[label]=(raw,json.dumps(json.loads(normalized),sort_keys=True,separators=(',',':')),parsed)
        expected={'perception',*ARMS} if arms is not None else {'perception'}
        if row_labels!=expected:raise ValueError('incomplete seven-contact row')
        patterns[str(len(row_requests))]+=1
        credit_cost[credits[sid]['reason']].update(row_cost)
        credit_cost[credits[sid]['reason']]['rows']+=1
        if answers:
            for kind,index in (('raw',0),('normalized',1)):
                values=[answers[a][index] for a in ARMS[:4]]
                duplicates[kind+':k4_all_equal']+=len(set(values))==1
                duplicates[kind+':k4_unique_sum']+=len(set(values))
                duplicates[kind+':any_candidate_differs_lock']+=any(v!=answers['lock'][index] for v in values)
                duplicates[kind+':aa_equals_lock']+=answers['aa'][index]==answers['lock'][index]
            duplicates['candidate_parse_failures']+=sum(not answers[a][2] for a in ARMS[:4])
    trace_costs={key[:-5] for key in bundle.trace_records if key.endswith('/cost')}
    if trace_costs!=seen:raise ValueError('unattributed or missing model cost contacts')
    return {'rows':len(ids),'contacts':len(seen),'contacts_per_row_distribution':dict(patterns),
        'no_solver_abstention_rows':abstentions,'groups':{k:dict(v) for k,v in totals.items()},
        'per_contact_distributions':{k:{f:stats(v) for f,v in group.items()} for k,group in measurements.items()},
        'answer_diversity':dict(duplicates),'cost_by_credit_disposition':{k:dict(v) for k,v in credit_cost.items()},
        'missing_request_timing':missing,'new_model_calls':0,'new_tokenizer_calls':0,'new_scoring_jobs':0,
        'limitations':['Seven-contact first-visit K4 only; unavailable arm custody refuses attribution.',
            'Normalized equality preserves list order; it is not semantic equivalence.',
            'Credit disposition partitions spend, not counterfactual value; controls remain necessary evidence.',
            'Contact times overlap across samples and include queue/service/network; they are not wall time.',
            'Costs cover the verified sealed attempt only; FIT and previous failed attempts remain separate.']}


def run(repo,data,source,half,recipe_path,timing,out,iteration=1):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    assert_execution_environment();head=require_clean_study_repo(repo);source_head=require_clean_study_repo(source)
    pin={'relpath':str(recipe_path.relative_to(source)),'sha256':sha(recipe_path)}
    loaded=load_half(source,data,half,pin,iteration=iteration)
    before=sha(timing);events=[]
    with timing.open() as stream:
        for line in stream:
            if not line.endswith('\n'):raise ValueError('sealed timing journal has partial tail')
            events.append(json.loads(line))
    if sha(timing)!=before:raise ValueError('timing journal changed while reading')
    result=summarize(loaded['bundle'],loaded['credit'],events)
    result.update(schema='saea-tfrl-current-cost-audit-v1',source=loaded['source'],source_commit=source_head,
        implementation_commit=head,timing_sha256=before,iteration=iteration,half=half)
    if not out.resolve().is_relative_to(data.resolve()/'work/codex'):raise ValueError('owned audit output required')
    publish_new(out,(json.dumps(result,indent=2)+'\n').encode());return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','source','recipe-path','timing','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--half',choices=['h1','h2'],required=True);p.add_argument('--iteration',type=int,default=1)
    result=run(**vars(p.parse_args()));print(json.dumps({'rows':result['rows'],'contacts':result['contacts']}))
