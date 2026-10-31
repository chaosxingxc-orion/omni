"""Read a real-roster checkpoint snapshot without model calls or runtime mutation.

Completed rows are hash-chain verified and costs are rederived from their traces.
Partial checkpoint costs remain separate; unknown in-flight costs are not zero.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def stats(values):
    v=sorted(values)
    return {'n':len(v),'mean_s':statistics.mean(v),'p50_s':statistics.median(v),'p95_s':v[int(.95*(len(v)-1))]} if v else None


def report(checkpoint, roster, *, state, elapsed_seconds, cards, instances):
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint,summarize_trace_lines
    if state not in {'running','complete','failed','stopped'}:raise ValueError('explicit observed state required')
    if elapsed_seconds<=0 or cards<1 or instances<1 or instances>cards:raise ValueError('invalid observed duration/resources')
    roster_doc=json.loads(roster.read_bytes())
    ids=[x['sample_id'] for x in roster_doc['items']]
    if len(ids)!=len(set(ids)) or roster_doc['count']!=len(ids):raise ValueError('invalid distinct roster')
    doc=read_checkpoint(checkpoint)
    if doc.header.get('split_identity_hash')!=roster_doc['split_identity_hash']:raise ValueError('checkpoint roster identity differs')
    valid=set(ids);complete=[];costs=defaultdict(float);timings=defaultdict(list)
    for row in doc.rows:
        if row.sample_id not in valid:raise ValueError('checkpoint contains foreign sample')
        derived=summarize_trace_lines(row.trace_lines)
        if dict(row.cost)!=derived or row.contacts!=derived['calls']:raise ValueError('checkpoint cost differs from trace')
        for key,value in derived.items():costs[key]+=value
        if row.keep_defect() is not None:continue
        complete.append(row.sample_id)
        sample_t=[]
        for line in row.trace_lines:
            payload=json.loads(line)['payload']
            if 'use_sample_execution_timing' in payload:sample_t.append(payload['use_sample_execution_timing'])
        if len(sample_t)>1:raise ValueError('ambiguous sample duration')
        if sample_t:
            t=sample_t[0];a,b,c=t['batch_started_ns'],t['admitted_ns'],t['finished_ns']
            if not a<=b<=c:raise ValueError('invalid sample timing order')
            timings['admission_wait'].append((b-a)/1e9)
            timings['admitted_to_finished'].append((c-b)/1e9)
    n=len(complete);total=len(ids)
    if state=='complete' and (n!=total or doc.torn_tail_bytes):raise ValueError('completion does not cover intact full roster')
    rate=n/elapsed_seconds if n else None
    eta=(total-n)/rate if rate and state=='running' else (0 if state=='complete' else None)
    return {'schema':'saea-real-engine-progress-v1','state':state,'completed_rows':n,'total_rows':total,
        'checkpoint_rows':len(doc.rows),'unusable_checkpoint_rows':len(doc.rows)-n,'remaining_rows':total-n,
        'checkpoint_sha256':doc.sha256,'split_identity_hash':roster_doc['split_identity_hash'],
        'torn_tail_bytes':doc.torn_tail_bytes,'cards':cards,'instances':instances,'elapsed_seconds':elapsed_seconds,
        'wall_seconds_per_complete_row':elapsed_seconds/n if n else None,'rows_per_second':rate,
        'estimated_remaining_seconds':eta,'eta_basis':'Linear observed whole-task completion rate; startup and tail may change it.' if eta else None,
        'full_roster_comparison_eligible':state=='complete','checkpoint_recorded_cost_lower_bound':dict(costs),
        'uncheckpointed_and_inflight_cost':'unknown; reconcile request journals and final accounting separately',
        'complete_sample_timings':{k:stats(v) for k,v in timings.items()},
        'new_model_calls':0,'limitations':['Duration and state must come from an authoritative run boundary/process observation.',
        'Allocated cards are not measured utilization.','Partial or failed candidates cannot establish a full-roster throughput winner.',
        'Checkpoint costs are a lower bound, not total spend; FIT contacts are accounted separately.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','roster','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--state',required=True);p.add_argument('--elapsed-seconds',type=float,required=True)
    p.add_argument('--cards',type=int,required=True);p.add_argument('--instances',type=int,required=True)
    a=p.parse_args();result=report(a.checkpoint,a.roster,state=a.state,elapsed_seconds=a.elapsed_seconds,cards=a.cards,instances=a.instances)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('state','completed_rows','total_rows','wall_seconds_per_complete_row','estimated_remaining_seconds')}))
