"""Join completed scientific role contacts with operational timing boundaries.

P/S identities come from canonical episode receipts, never guessed from request
ordering. Gaps include CPU/scheduling/FIT and are not labelled as pure compute.
"""
import argparse
from collections import defaultdict
import json
import math
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



def slot_wait_partition(events, *, sample_id, pid, admitted, request_started):
    """Partition pre-P latency using observed slot waits, never inferred CPU time.

    Intersect and union same-process intervals to avoid charging later waits or
    overlapping spans twice. Absence is unknown, not a zero-wait observation.
    """
    spans = []
    for event in events:
        if event.get('stage') != 'dispatch_trajectory_slot_wait' or event.get('sample_id') != sample_id:
            continue
        if event.get('pid') != pid:
            raise ValueError('slot wait belongs to another monotonic clock')
        start, elapsed = event['started_monotonic'], event['elapsed_seconds']
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, elapsed)) or elapsed < 0:
            raise ValueError('invalid slot wait interval')
        end = start + elapsed
        if start > request_started or end < admitted:
            continue
        if event.get('status') != 'ok':
            raise ValueError('pre-perception slot wait did not complete successfully')
        spans.append((max(admitted, start), min(request_started, end)))
    if not spans:
        return None
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    wait = sum(end-start for start, end in merged)
    return wait, max(0, request_started-admitted-wait)


def join(rows,events):
    by_request={};metrics=defaultdict(list);missing=[];seen=set()
    boundary_present = 0
    joined_phase_rows = 0
    slot_wait_rows = 0
    slot_events = defaultdict(list)
    for event in events:
        if event.get("stage") == "dispatch_trajectory_slot_wait":
            slot_events[event.get("sample_id")].append(event)
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
        records = [json.loads(line) for line in row.trace_lines]
        timings = [record['payload']['use_sample_execution_timing'] for record in records
                   if 'use_sample_execution_timing' in record['payload']]
        if len(timings) > 1:
            raise ValueError('ambiguous sample execution timing')
        boundary_present += bool(timings)
        trace = {record['record_id']: record['payload'] for record in records}
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
            if timings:
                timing = timings[0]
                admitted = timing['admitted_ns'] / 1e9
                finished = timing['finished_ns'] / 1e9
                p_start = p['started_monotonic']
                s_end = s['started_monotonic'] + s['elapsed_seconds']
                if not admitted <= p_start <= s_end <= finished + .001:
                    raise ValueError('sample/role time boundaries do not nest')
                metrics['admission_to_perception_request_start'].append(p_start - admitted)
                partition = slot_wait_partition(slot_events[row.sample_id], sample_id=row.sample_id,
                    pid=p['pid'], admitted=admitted, request_started=p_start)
                if partition is not None:
                    slot_wait_rows += 1
                    metrics['pre_perception_observed_slot_wait'].append(partition[0])
                    metrics['pre_perception_other_elapsed'].append(partition[1])
                metrics['solver_request_end_to_sample_finished'].append(max(0, finished - s_end))
                metrics['admission_to_sample_finished'].append(finished - admitted)
                joined_phase_rows += 1

    return {'rows':len(seen),'metrics':{k:stats(v) for k,v in metrics.items()},'missing_timing_entries':len(missing),'missing_sample_count':len({r['sample_id'] for r in missing}),
            # Legacy missing counters describe request/http events only.
            # Boundary presence and successful P/S joins are separate coverage.
            'sample_timing_coverage': {
                'expected_rows': len(seen),
                'boundary_present_rows': boundary_present,
                'boundary_missing_rows': len(seen) - boundary_present,
                'joined_phase_rows': joined_phase_rows,
                'unjoined_phase_rows': len(seen) - joined_phase_rows,
            },
            'pre_perception_slot_coverage': {'joined_rows': slot_wait_rows,
                'rows_without_observed_wait': joined_phase_rows-slot_wait_rows},
            'all_timing_complete': bool(seen) and not missing and joined_phase_rows == len(seen),
            'limitations':['Completed subset is descriptive and may be biased toward faster samples.','P→S gap contains preparation/FIT/scheduling; not pure CPU time.','FIT uses prepared request IDs that can be reused before actual generation IDs are assigned; no per-role FIT attribution is inferred. HTTP includes service queue/audio/prefill/decode/network.','Nested intervals overlap; do not sum as whole-task wall time.',
                'Pre-P slot wait and other elapsed partition only matched sample intervals; other elapsed is not pure CPU time. Missing slot events are unknown.'],'new_model_calls':0}


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
