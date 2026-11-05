"""Join completed scientific role contacts with operational timing boundaries.

P/S identities come from canonical episode receipts, never guessed from request
ordering. Gaps include CPU/scheduling/FIT and are not labelled as pure compute.
"""
import argparse
import hashlib
from collections import Counter
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
                raise ValueError('contact_multiplicity' if len(contacts) != 1 else 'missing_contact_identity')
            chain.append((name, contacts[0]['request_id']))
    if len({r for _, r in chain}) != len(chain) or len({n for n, _ in chain}) != len(chain):
        raise ValueError('repeated_request' if len({r for _, r in chain}) != len(chain) else 'repeated_role')
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


def _join_one(rows, events):
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
        output = row.output_rows[0]
        if not isinstance(output, dict) or not isinstance(output.get('native_episode_receipt'), dict):
            counts['invalid_receipt'] += 1
            continue
        episode = output['native_episode_receipt']
        if episode.get('perception_branches'):
            counts['lock_branch_projection_rows'] += 1
        reason = (episode.get('terminal_disposition') or {}).get('reason')
        if reason == 'seal_refusal': status = 'refusal'
        elif reason or episode.get('interrupted_role'): status = 'finite_or_other_terminal'
        else: status = 'completed_candidate'
        counts[status] += 1
        try:
            chain = contacts_chain(output)
        except (KeyError, TypeError, ValueError) as exc:
            counts['unsupported_chain'] += 1
            counts['unsupported_reason:' + (str(exc) or 'invalid_receipt')] += 1
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
                if event.get('status', 'ok') != 'ok': raise ValueError('request_status_not_ok')
                if (type(start) not in (int, float) or not math.isfinite(start) or start < 0 or
                    type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0 or type(event.get('pid')) is not int or event['pid'] <= 0):
                    raise ValueError('invalid request timing')
                active_metrics[role+'/'+stage].append(elapsed)
                if stage == 'request_total': total = event
                else: http = event
            if total is not None and http is not None:
                if total['pid'] != http['pid']: raise ValueError('request_http_pid_mismatch')
                if http['started_monotonic'] < total['started_monotonic'] - .001 or http['started_monotonic'] + http['elapsed_seconds'] > total['started_monotonic'] + total['elapsed_seconds'] + .001:
                    raise ValueError('http_outside_request')
            bounds.append(total)
        counts['chain_rows'] += 1
        if any(e is None for e in bounds): continue
        if len({e['pid'] for e in bounds}) != 1: raise ValueError('incomparable monotonic clocks')
        for (left, _), (right, _), prev, curr in zip(chain, chain[1:], bounds, bounds[1:]):
            gap = curr['started_monotonic'] - (prev['started_monotonic'] + prev['elapsed_seconds'])
            if gap < 0: raise ValueError('next node began before previous completed')
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
            'all_timing_complete':bool(expected) and not missing and joined_phase_rows == expected and expected == len(seen),
            'limitations':['Contact chains with multiple contacts, repeated roles or ambiguous identity are counted as unsupported.',
                'Lock branch projection is counted separately; other branch contacts are excluded.',
                'Completed candidates are receipt-based and are not a validated scientific success count.',
                'Gaps include preparation, FIT and scheduling; they are not pure CPU time.',
                'Sample boundary records have no process identity; clock compatibility is assumed after nesting validation.',
                'Missing timing and slot waits are unknown, not zero. Nested intervals overlap.',
                'Processed checkpoint rows need not represent the full workload or successful protocol.'], 'new_model_calls':0}


def join(rows, events, *, include_rows=False):
    """Isolate each row so ambiguous or invalid events cannot discard the run."""
    by_sample = defaultdict(list)
    for event in events:
        if isinstance(event, dict) and event.get('sample_id') is not None:
            by_sample[event['sample_id']].append(event)
    frequencies = Counter(row.sample_id for row in rows)
    aggregates = {'metrics': defaultdict(list), 'terminal_metrics': defaultdict(list)}
    counts = Counter(); cover = Counter(); waits = Counter(); missing = 0; missing_rows = 0
    details = []; all_complete = bool(rows); chain_rows = 0
    for index, row in enumerate(rows):
        identifier = hashlib.sha256(str(row.sample_id).encode()).hexdigest()
        detail = {'row_index': index, 'sample_key_sha256': identifier}
        try:
            if frequencies[row.sample_id] != 1: raise ValueError('duplicate_sample')
            result = _join_one([row], by_sample.get(row.sample_id, []))
            detail['classification'] = result['row_classification']
            detail['metrics'] = {group: {key: value['mean_s'] for key, value in result[group].items()}
                                 for group in aggregates}
            detail['sample_timing_coverage'] = result['sample_timing_coverage']
            counts.update(result['row_classification'])
            cover.update(result['sample_timing_coverage']); waits.update(result['pre_perception_slot_coverage'])
            chain_rows += result['rows']; missing += result['missing_timing_entries']; missing_rows += result['missing_sample_count']
            all_complete &= result['all_timing_complete']
            for group in aggregates:
                for key, value in result[group].items(): aggregates[group][key].append(value['mean_s'])
        except (ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) else 'malformed_row_or_event'
            detail.update({'classification': {'unsupported_row': 1}, 'unsupported_reason': reason})
            counts['unsupported_row'] += 1; counts['unsupported_reason:' + reason] += 1
            all_complete = False
        details.append(detail)
    output = {'rows': chain_rows, 'processed_rows':len(rows), 'row_classification':dict(counts),
              **{group:{key:stats(values) for key,values in metrics.items()} for group,metrics in aggregates.items()},
              'missing_timing_entries':missing, 'missing_sample_count':missing_rows,
              'sample_timing_coverage':{k:cover[k] for k in ('expected_rows','boundary_present_rows','boundary_missing_rows','joined_phase_rows','unjoined_phase_rows')},
              'pre_perception_slot_coverage':{k:waits[k] for k in ('joined_rows','rows_without_observed_wait')},
              'all_timing_complete':all_complete, 'all_timing_complete_scope':'all checkpoint rows; single unambiguous chain, all request/http events and sample boundaries',
              'sample_clock_assumption':'Sample boundary has no pid; sample-to-node and slot partitions assume a common monotonic clock after nesting checks.',
              'limitations':['Receipt-complete candidates are not validated scientific success.',
                  'Multi-contact and repeated roles are unsupported; lock branch is a P/S projection excluding other branches.',
                  'Node order is receipt order, not independent evidence of causal dependency.',
                  'Missing values are unknown; gaps include scheduling/FIT and are not pure CPU.',
                  'Sample boundary cross-clock identity is assumed, not proven.'], 'new_model_calls':0}
    if include_rows: output['sample_rows'] = details
    return output


def analyze(checkpoint,timing, *, include_rows=False):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    assert_execution_environment();doc=read_checkpoint(checkpoint);events=[];tail=False
    with timing.open() as stream:
        for line in stream:
            if not line.endswith('\n'):tail=True;break
            events.append(json.loads(line))
    result=join(doc.rows,events,include_rows=include_rows)
    return {'schema':'saea-real-phase-timing-v1','checkpoint_sha256':doc.sha256,'timing_partial_tail':tail,**result}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('checkpoint','timing','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--rows-out', type=Path, help='private per-row JSONL; sample identifiers are SHA256 pseudonyms')
    a=p.parse_args();result=analyze(a.checkpoint,a.timing,include_rows=bool(a.rows_out))
    if a.rows_out:
        rows = result.pop('sample_rows')
        with a.rows_out.open('x') as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'rows':result['rows'],'missing_timing_entries':result['missing_timing_entries']}))
