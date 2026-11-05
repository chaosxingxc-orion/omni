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


def contacts_chain(output):
    """Extract a single ordered chain; branching/multiple contacts stay visible as unsupported."""
    episode = output['native_episode_receipt']
    visits = list(episode.get('role_visits') or [])
    interrupted = episode.get('interrupted_role')
    if interrupted:
        visits.append(interrupted)
    branches = episode.get('perception_branches')
    if branches:
        locks = [b for b in branches.get('branches', []) if b.get('branch_id') == 'lock']
        if len(locks) != 1:
            raise ValueError('unsupported branch selection')
        contacts = {}
        for v in visits:
            for c in v.get('execution_receipt', {}).get('contacts', []):
                cid = c.get('agent_contact_id')
                if not cid or cid in contacts:
                    raise ValueError('ambiguous contact identity')
                contacts[cid] = c
        try:
            chain = [(name, contacts[locks[0][key]]['request_id']) for name, key in
                     (('perception', 'contact_id'), ('solver', 'solver_contact_id'))]
        except KeyError as exc:
            raise ValueError('missing lock contact') from exc
    else:
        chain = []
        for visit in visits:
            name = visit.get('role_id') or visit.get('stage')
            contacts = visit.get('execution_receipt', {}).get('contacts', [])
            if not name or len(contacts) != 1 or not contacts[0].get('request_id'):
                raise ValueError('unsupported contact multiplicity or missing identity')
            chain.append((name, contacts[0]['request_id']))
    if len({r for _, r in chain}) != len(chain) or len({n for n, _ in chain}) != len(chain):
        raise ValueError('ambiguous repeated request or role')
    return chain


def roles(output):
    return dict(contacts_chain(output))


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


def join(rows, events):
    by_request = {}; metrics = defaultdict(list); terminal_metrics = defaultdict(list); missing = []; seen = set()
    counts = defaultdict(int); boundary_present = joined_phase_rows = slot_wait_rows = 0
    slot_events = defaultdict(list)
    for e in events:
        if e.get('stage') == 'dispatch_trajectory_slot_wait':
            slot_events[e.get('sample_id')].append(e)
        if e.get('stage') not in ('request_total', 'http_request') or not e.get('request_id') or not e.get('sample_id'):
            continue
        key = (e['sample_id'], e['request_id'], e['stage'])
        if key in by_request:
            raise ValueError('ambiguous request stage')
        by_request[key] = e
    for row in rows:
        if row.sample_id in seen:
            raise ValueError('duplicate sample')
        seen.add(row.sample_id)
        if row.keep_defect() is not None or len(row.output_rows) != 1:
            counts['unfinished_or_invalid'] += 1
            continue
        output = row.output_rows[0]; episode = output.get('native_episode_receipt') or {}
        if episode.get('perception_branches'):
            counts['lock_branch_projection_rows'] += 1
        reason = (episode.get('terminal_disposition') or {}).get('reason')
        if reason == 'seal_refusal': status = 'refusal'
        elif reason or episode.get('interrupted_role'): status = 'finite_or_other_terminal'
        else: status = 'completed_candidate'
        counts[status] += 1
        try:
            chain = contacts_chain(output)
        except (KeyError, TypeError, ValueError):
            counts['unsupported_chain'] += 1
            continue
        if not chain:
            counts['empty_chain'] += 1
            continue
        # A P/Slot/S trajectory is a candidate only when its solver was accepted.
        names = [name for name, _ in chain]
        if status == 'completed_candidate' and 'slot' in names and not episode.get('accepted_solver_receipt'):
            counts['incomplete_protocol'] += 1
            continue
        active_metrics = metrics if status == 'completed_candidate' else terminal_metrics
        records = [json.loads(line) for line in row.trace_lines]
        timings = [record['payload']['use_sample_execution_timing'] for record in records
                   if 'use_sample_execution_timing' in record['payload']]
        if len(timings) > 1: raise ValueError('ambiguous sample execution timing')
        boundary_present += bool(timings)
        trace = {}
        for record in records:
            rid = record['record_id']
            if rid in trace: raise ValueError('ambiguous trace identity')
            trace[rid] = record['payload']
        bounds = []
        for role, rid in chain:
            cost = trace.get(rid+'/cost')
            if cost is not None:
                latency = cost.get('latency_seconds')
                if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0: raise ValueError('invalid scientific latency')
                active_metrics[role+'/scientific_request'].append(latency)
            else:
                counts['missing_scientific_cost'] += 1
            total = None; http = None
            for stage in ('request_total', 'http_request'):
                event = by_request.get((row.sample_id, rid, stage))
                if event is None:
                    missing.append((row.sample_id, role, stage)); continue
                start, elapsed = event.get('started_monotonic'), event.get('elapsed_seconds')
                if (type(start) not in (int, float) or not math.isfinite(start) or start < 0 or
                    type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0 or event.get('pid') is None):
                    raise ValueError('invalid request timing')
                active_metrics[role+'/'+stage].append(elapsed)
                if stage == 'request_total': total = event
                else: http = event
            if total is not None and http is not None and total['pid'] != http['pid']:
                raise ValueError('incomparable request/http clocks')
            bounds.append(total)
        counts['chain_rows'] += 1
        if any(e is None for e in bounds): continue
        if len({e['pid'] for e in bounds}) != 1: raise ValueError('incomparable monotonic clocks')
        for (left, _), (right, _), prev, curr in zip(chain, chain[1:], bounds, bounds[1:]):
            gap = curr['started_monotonic'] - (prev['started_monotonic'] + prev['elapsed_seconds'])
            if gap < -.001: raise ValueError('next node began before previous completed')
            active_metrics[left+'_end_to_'+right+'_request_start'].append(max(gap, 0))
        first, last = bounds[0], bounds[-1]
        if names == ['perception', 'solver']:
            active_metrics['perception_start_to_solver_end'].append(last['started_monotonic']+last['elapsed_seconds']-first['started_monotonic'])
        if timings:
            timing = timings[0]
            if timing.get('sample_id', row.sample_id) != row.sample_id: raise ValueError('sample timing identity mismatch')
            raw = [timing.get('admitted_ns'), timing.get('finished_ns')]
            if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in raw): raise ValueError('invalid sample timing')
            admitted, finished = [x / 1e9 for x in raw]
            start = first['started_monotonic']; end = last['started_monotonic']+last['elapsed_seconds']
            if not admitted <= start <= end <= finished + .001: raise ValueError('sample/role time boundaries do not nest')
            active_metrics['admission_to_first_node_request_start'].append(start-admitted)
            active_metrics['last_node_request_end_to_sample_finished'].append(max(0,finished-end))
            active_metrics['admission_to_sample_finished'].append(finished-admitted)
            if names[0] == 'perception':
                active_metrics['admission_to_perception_request_start'].append(start-admitted)
                partition = slot_wait_partition(slot_events[row.sample_id], sample_id=row.sample_id, pid=first['pid'], admitted=admitted, request_started=start)
                if partition is not None:
                    slot_wait_rows += 1
                    active_metrics['pre_perception_observed_slot_wait'].append(partition[0]); active_metrics['pre_perception_other_elapsed'].append(partition[1])
            if names[-1] == 'solver': active_metrics['solver_request_end_to_sample_finished'].append(max(0,finished-end))
            joined_phase_rows += 1
    expected = counts['chain_rows']
    return {'rows': expected, 'processed_rows': len(seen), 'row_classification': dict(counts),
            'metrics': {k:stats(v) for k,v in metrics.items()},
            'terminal_metrics': {k:stats(v) for k,v in terminal_metrics.items()}, 'missing_timing_entries':len(missing),
            'missing_sample_count':len({x[0] for x in missing}),
            'sample_timing_coverage': {'expected_rows':expected, 'boundary_present_rows':boundary_present,
                'boundary_missing_rows':expected-boundary_present, 'joined_phase_rows':joined_phase_rows,
                'unjoined_phase_rows':expected-joined_phase_rows},
            'pre_perception_slot_coverage': {'joined_rows':slot_wait_rows, 'rows_without_observed_wait':joined_phase_rows-slot_wait_rows},
            'all_timing_complete':bool(expected) and not missing and joined_phase_rows == expected and not counts['unsupported_chain'],
            'limitations':['Contact chains with multiple contacts, repeated roles or ambiguous identity are counted as unsupported.',
                'Lock branch projection is counted separately; other branch contacts are excluded.',
                'Completed candidates are receipt-based and are not a validated scientific success count.',
                'Gaps include preparation, FIT and scheduling; they are not pure CPU time.',
                'Sample boundary records have no process identity; clock compatibility is assumed after nesting validation.',
                'Missing timing and slot waits are unknown, not zero. Nested intervals overlap.',
                'Processed checkpoint rows need not represent the full workload or successful protocol.'], 'new_model_calls':0}


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
