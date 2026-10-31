"""Audit the complete625 source union without replaying generation or scoring."""
import argparse,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from tfrl_current_sources import read,sha
from tfrl_current_cost_audit import summarize
from tfrl_recovered_consumers import load,POOL
from tfrl_current_pool import pool


def combine(parts):
    expected={'original-checkpoint':3,'repaired310':310,'h2':312}
    if set(parts)!=set(expected):raise ValueError('exact source partition required')
    groups=defaultdict(Counter);credit=defaultdict(Counter);diversity=Counter();patterns=Counter()
    for name,part in parts.items():
        if part['rows']!=expected[name] or part['contacts']!=7*expected[name]:
            raise ValueError('complete seven-contact source required')
        for k,v in part['groups'].items():groups[k].update(v)
        for k,v in part['cost_by_credit_disposition'].items():credit[k].update(v)
        diversity.update(part['answer_diversity']);patterns.update(part['contacts_per_row_distribution'])
    return {'rows':625,'contacts':4375,'groups':{k:dict(v) for k,v in groups.items()},
        'cost_by_credit_disposition':{k:dict(v) for k,v in credit.items()},
        'answer_diversity':dict(diversity),'contacts_per_row_distribution':dict(patterns),
        'missing_request_timing':sum(p['missing_request_timing'] for p in parts.values()),
        'segments':parts,'new_model_calls':0,'new_tokenizer_calls':0,'new_scoring_jobs':0,
        'scope':'Accepted625 source rows only; all failed-attempt and FIT costs remain separate.',
        'limitations':['Segment percentiles are not pooled percentiles.',
            'Normalized answer equality is not semantic equivalence.',
            'Summed contact latency is not wall time; concurrent service time includes waiting.']}


def run(repo,data,source,h1_timing,h2_timing,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    assert_execution_environment();head=require_clean_study_repo(repo)
    source_head=require_clean_study_repo(source)
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned output required')
    _,_,halves=load(source,data)
    verified=pool({v['plan']['half']:(v['plan'],v['results']) for v in halves.values()})
    path=data/'analysis'/POOL/'complete.json';record=read(path)
    if any(record.get(k)!=v for k,v in verified.items()) or record['sources']!={h:v['source'] for h,v in halves.items()}:
        raise ValueError('pool differs from verified sources')
    timings={};pins={}
    for h,p in [('h1',h1_timing),('h2',h2_timing)]:
        before=sha(p);events=[]
        with p.open() as stream:
            for line in stream:
                if not line.endswith('\n'):raise ValueError('partial timing journal')
                events.append(json.loads(line))
        if sha(p)!=before:raise ValueError('timing journal changed')
        timings[h]=events;pins[h]=before
    parts={}
    for segment in halves['h1']['segments']:
        name=segment['namespace'];bundle=segment['rows'];ids={r['sample_id'] for r in bundle.outputs_rows}
        credit=[r for r in halves['h1']['credit'] if r['row_id'] in ids]
        # Original timing was not collected here. Keep the missing21 explicit.
        parts[name]=summarize(bundle,credit,timings['h1'] if name=='repaired310' else [])
    h2=halves['h2'];parts['h2']=summarize(h2['bundle'],h2['credit'],timings['h2'])
    result=combine(parts)
    result.update(schema='saea-tfrl-recovered-cost-audit-v1',implementation_commit=head,
        source_commit=source_head,sources=record['sources'],pool_sha256=sha(path),timing_sha256=pins,
        official_reaggregated=verified['official_reaggregated'],vote_minus_lock_pp=verified['vote_minus_lock_pp'],
        construction_gate_passed=verified['construction_gate_passed'])
    require_clean_study_repo(repo);require_clean_study_repo(source)
    publish_new(out,(json.dumps(result,indent=2)+'\n').encode());return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','source','h1-timing','h2-timing','out'):p.add_argument('--'+k,type=Path,required=True)
    result=run(**vars(p.parse_args()));print(json.dumps({'rows':result['rows'],'contacts':result['contacts']}))
