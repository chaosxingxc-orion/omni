"""Join completed scientific role contacts with operational timing boundaries.

P/S identities come from canonical episode receipts, never guessed from request
ordering. Gaps include CPU/scheduling/FIT and are not labelled as pure compute.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def stats(v):
    v=sorted(v)
    return {'n':len(v),'mean_s':statistics.mean(v),'p50_s':statistics.median(v),'p95_s':v[int(.95*(len(v)-1))]} if v else None


def roles(output):
    episode=output['native_episode_receipt']
    if 'perception_branches' not in episode:
        result={}
        for role in ('perception','solver'):
            visit,=[v for v in episode['role_visits'] if v['role_id']==role]
            contact,=visit['execution_receipt']['contacts']
            result[role]=contact['request_id']
        return result
    branch,=[b for b in episode['perception_branches']['branches'] if b['branch_id']=='lock']
    contacts={c['agent_contact_id']:c for v in episode['role_visits'] for c in v['execution_receipt']['contacts']}
    return {'perception':contacts[branch['contact_id']]['request_id'],'solver':contacts[branch['solver_contact_id']]['request_id']}


def join(rows,events):
    by_request={};metrics=defaultdict(list);missing=[];seen=set()
    for e in events:
        if e['stage'] not in ('request_total','http_request'):continue
        if not e.get('request_id') or not e.get('sample_id'):continue
        key=(e['sample_id'],e['request_id'],e['stage'])
        if key in by_request:raise ValueError('ambiguous request stage')
        by_request[key]=e
    for row in rows:
        if row.keep_defect() is not None:continue
        output,=row.output_rows
        if row.sample_id in seen:raise ValueError('duplicate sample')
        seen.add(row.sample_id);pairs=roles(output);boundaries={}
        trace={json.loads(line)['record_id']:json.loads(line)['payload'] for line in row.trace_lines}
        for role,rid in pairs.items():
            cost=trace[rid+'/cost'];metrics[role+'/scientific_request'].append(cost['latency_seconds'])
            for stage in ('request_total','http_request'):
                event=by_request.get((row.sample_id,rid,stage))
                if event is None:missing.append({'sample_id':row.sample_id,'role':role,'stage':stage});continue
                metrics[role+'/'+stage].append(event['elapsed_seconds'])
                if stage=='request_total':boundaries[role]=event
        if set(boundaries)=={'perception','solver'}:
            p,s=boundaries['perception'],boundaries['solver']
            if p['pid']!=s['pid']:raise ValueError('incomparable monotonic clocks')
            gap=s['started_monotonic']-(p['started_monotonic']+p['elapsed_seconds'])
            if gap<-.001:raise ValueError('solver began before perception completed')
            metrics['perception_end_to_solver_request_start'].append(max(gap,0))
            metrics['perception_start_to_solver_end'].append(s['started_monotonic']+s['elapsed_seconds']-p['started_monotonic'])
    return {'rows':len(seen),'metrics':{k:stats(v) for k,v in metrics.items()},'missing_timing_entries':len(missing),'missing_sample_count':len({r['sample_id'] for r in missing}),
            'limitations':['Completed subset is descriptive and may be biased toward faster samples.','P→S gap contains preparation/FIT/scheduling; not pure CPU time.','FIT uses prepared request IDs that can be reused before actual generation IDs are assigned; no per-role FIT attribution is inferred. HTTP includes service queue/audio/prefill/decode/network.','Nested intervals overlap; do not sum as whole-task wall time.'],'new_model_calls':0}


def analyze(checkpoint,timing):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    assert_execution_environment();doc=read_checkpoint(checkpoint);events=[];tail=False
    with timing.open() as stream:
        for line in stream:
            if not line.endswith('\n'):tail=True;break
            events.append(json.loads(line))
    result=join(doc.rows,events)
    return {'schema':'saea-real-phase-timing-v1','checkpoint_sha256':doc.sha256,'timing_partial_tail':tail,**result}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('checkpoint','timing','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();result=analyze(a.checkpoint,a.timing)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'rows':result['rows'],'missing_timing_entries':result['missing_timing_entries']}))
