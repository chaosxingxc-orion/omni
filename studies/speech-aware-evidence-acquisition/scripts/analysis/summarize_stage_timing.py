"""Summarize operational timings; inclusive stages overlap and must not be added."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(path, complete_sample_ids=None):
    events=[];incomplete_tail=False
    with path.open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                incomplete_tail=True;break
            event=json.loads(line)
            if event['schema']!='saea-stage-timing-v1' or event['elapsed_seconds']<0:raise ValueError('invalid timing event')
            events.append(event)
    observed_events=len(events)
    if complete_sample_ids is not None:
        complete_sample_ids=set(complete_sample_ids)
        events=[e for e in events if e.get('sample_id') in complete_sample_ids]
    stages=defaultdict(list);sample_stages=defaultdict(lambda:defaultdict(float));requests=defaultdict(dict)
    for e in events:
        stages[e['stage']].append(e)
        if e.get('sample_id'):sample_stages[e['sample_id']][e['stage']]+=e['elapsed_seconds']
        if e.get('request_id') and e['stage'] in ('request_total','transport_total','http_request'):
            key=(e['pid'],e['thread'],e['request_id'])
            if e['stage'] in requests[key]:raise ValueError('duplicate request timing boundary')
            requests[key][e['stage']]=e
    def stats(values):
        values=sorted(values)
        return {'count':len(values),'mean_s':statistics.mean(values),'p50_s':statistics.median(values),'p95_s':values[int(.95*(len(values)-1))],'total_s':sum(values)} if values else None
    result={}
    for stage,group in sorted(stages.items()):
        result[stage]={'invocations':stats([e['elapsed_seconds'] for e in group]),'per_observed_sample':stats([v[stage] for v in sample_stages.values() if stage in v]),'errors':sum(e['status']!='ok' for e in group)}
    overhead=[];wire=[]
    for v in requests.values():
        if 'request_total' in v and 'transport_total' in v:
            a,b=v['request_total'],v['transport_total']
            if a['elapsed_seconds']+1e-6<b['elapsed_seconds']:raise ValueError('nested timing exceeds enclosing request')
            overhead.append(a['elapsed_seconds']-b['elapsed_seconds'])
        if 'http_request' in v:wire.append(v['http_request']['elapsed_seconds'])
    return {'schema':'saea-stage-timing-summary-v1','events':len(events),'total_journal_events':observed_events,'sample_scope':'verified-complete-checkpoint-rows' if complete_sample_ids is not None else 'all-observed-rows-including-incomplete','completed_rows_without_timing':sorted(complete_sample_ids-set(sample_stages)) if complete_sample_ids is not None else None,'observed_sample_count':len(sample_stages),'incomplete_tail':incomplete_tail,'stages':result,'request_validation_accounting_and_response_handling_s':stats(overhead),'http_request_s':stats(wire),'limitations':['Inclusive intervals overlap; do not sum stages as wall time.','Observed samples include incomplete rows; join final checkpoints before estimating complete-task throughput.','HTTP time includes queue, server audio/prefill/decode and network; TTFT unavailable on non-streaming path.','Timing journal uses kernel writes, not per-event fsync; scientific checkpoints remain independently durable.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path);a=p.parse_args()
    complete_ids=None;checkpoint_sha=None
    if a.checkpoint:
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
        from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
        checkpoint=read_checkpoint(a.checkpoint);checkpoint_sha=checkpoint.sha256
        complete_ids={r.sample_id for r in checkpoint.rows if r.keep_defect() is None}
    result=summarize(a.input,complete_ids)
    result['checkpoint_sha256']=checkpoint_sha
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'events':result['events'],'samples':result['observed_sample_count']}))
