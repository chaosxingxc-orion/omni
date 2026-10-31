"""Read-only cost/duplication audit of a hash-pinned row checkpoint; no labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def audit(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected or not raw.endswith(b'\n'):
        raise ValueError('checkpoint hash or tail differs')
    groups = defaultdict(Counter)
    latencies = defaultdict(list)
    duplication = Counter()
    patterns = Counter()
    row_cost = Counter()
    previous = '0' * 64
    ids = set()
    for number, line in enumerate(raw.splitlines()):
        record = json.loads(line)
        claimed = record.pop('sha')
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        if record['seq'] != number or record['prev_sha'] != previous or sha(encoded) != claimed:
            raise ValueError('checkpoint hash chain differs')
        previous = claimed
        if number == 0:
            continue
        if not record['complete'] or record['sample_id'] in ids or len(record['output_rows']) != 1:
            raise ValueError('requires unique complete single-output rows')
        ids.add(record['sample_id'])
        if sha(''.join(t + '\n' for t in record['trace']).encode()) != record['trace_sha256']:
            raise ValueError('row trace hash differs')
        receipt = record['output_rows'][0]['native_episode_receipt']
        branches = {b['branch_id']: b for b in receipt['perception_branches']['branches']}
        mapping = {}
        for arm, branch in branches.items():
            sampling = branch.get('sampling') or {}
            for role, key, field in [('perception', 'perception_samples', 'contact_id'),
                                     ('solver', 'solver_samples', 'solver_contact_id')]:
                samples = (sampling.get(key) or {}).get('samples')
                contacts = [s['contact_id'] for s in samples] if samples else [branch[field]]
                for contact in contacts:
                    if contact in mapping:
                        raise ValueError('contact belongs to multiple arms')
                    mapping[contact] = (arm, role)
        requests, costs, answers = {}, {}, {}
        for text in record['trace']:
            event = json.loads(text)
            key = event['record_id'].split('/', 1)[0]
            value = event['payload']
            if event['channel'] == 'MODEL_REQUEST':
                payload = value['payload']
                controls = [h for h in payload['history'] if h['role'] == 'control']
                if key in requests or len(controls) != 1:
                    raise ValueError('duplicate request or ambiguous contact')
                requests[key] = controls[0]['contact_id']
            elif event['channel'] == 'COST':
                if key in costs:
                    raise ValueError('duplicate cost')
                costs[key] = value
            elif event['channel'] == 'MODEL_RESPONSE':
                answers[key] = value['text']
        if set(requests) != set(costs) or set(requests) != set(answers) or set(requests.values()) != set(mapping):
            raise ValueError('incomplete contact-cost-response join')
        by_contact = {contact: answers[key] for key, contact in requests.items()}
        row_totals = Counter()
        for key, contact in requests.items():
            arm, role = mapping[contact]
            cost = costs[key]
            for group in ('all', arm, 'role:' + role):
                target = groups[group]
                target['contacts'] += 1
                for field in ('calls', 'audio_seconds', 'prompt_tokens', 'completion_tokens', 'latency_seconds', 'reasoning_tokens'):
                    value = cost.get(field)
                    if value is not None:
                        target[field] += value
                        target[field + '_observed'] += 1
                target['forced_close'] += bool(cost.get('forced_close'))
                target['finish:' + str(cost.get('finish_reason'))] += 1
                latencies[group].append(cost['latency_seconds'])
            for field in ('calls', 'audio_seconds', 'prompt_tokens', 'completion_tokens', 'latency_seconds'):
                row_totals[field] += cost[field]
        for key, value in row_totals.items():
            if abs(value - record['cost'][key]) > 0.0001:
                raise ValueError('cost trace and checkpoint total differ: ' + key)
        row_cost.update(row_totals)
        patterns[str(len(requests))] += 1
        for arm in ('aa', 'lock-rep2', 'l1-support', 'l2-union'):
            duplication[arm + ':raw_solver_exact_equal_lock'] += branches[arm]['solver_response_text_sha256'] == branches['lock']['solver_response_text_sha256']
        l1 = branches['l1-support']['sampling']['solver_samples']
        hashes = [s['response_text_sha256'] for s in l1['samples']]
        duplication['l1:all_5_exact_same'] += len(set(hashes)) == 1
        duplication['l1:unique_answer_hashes_sum'] += len(set(hashes))
        duplication['l1:gate:' + l1['outcome']['gate']['status']] += 1
        l2 = branches['l2-union']['sampling']['perception_samples']
        sets = [tuple(sorted(s['candidate_intent_ids'])) for s in l2['samples']]
        duplication['l2:all_4_candidate_sets_same'] += len(set(sets)) == 1
        duplication['l2:no_appended_candidates'] += not l2['appended_candidate_ids']
        # Structural equality ignores JSON whitespace/object key order only.
        def normalized(text):
            try:
                return json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False)
            except (ValueError, TypeError):
                return text.strip()
        texts = [normalized(by_contact[s['contact_id']]) for s in l1['samples']]
        duplication['l1:all_5_json_equal'] += len(set(texts)) == 1
        try:
            first = json.loads(by_contact[l1['samples'][0]['contact_id']])
            entity_count = len(first['entities'])
            delivered = l1['outcome']['delivered_entity_count']
            duplication['l1:sample0_parseable'] += 1
            duplication['l1:commit_reduces_entity_count'] += delivered < entity_count
            duplication['l1:commit_preserves_entity_count'] += delivered == entity_count
        except (ValueError, KeyError, TypeError):
            duplication['l1:sample0_not_plain_json'] += 1
        for arm in ('aa', 'lock-rep2', 'l2-union'):
            duplication[arm + ':solver_json_equal_lock'] += normalized(by_contact[branches[arm]['solver_contact_id']]) == normalized(by_contact[branches['lock']['solver_contact_id']])
    for group, values in latencies.items():
        groups[group]['latency_median_s'] = statistics.median(values)
        groups[group]['latency_p95_s'] = sorted(values)[int((len(values)-1)*0.95)]
    return {'schema': 'saea-checkpoint-call-cost-audit-v1', 'checkpoint_sha256': expected,
            'rows': len(ids), 'contacts_per_row_histogram': patterns, 'groups': groups,
            'duplication': duplication, 'row_cost_totals': row_cost,
            'scope': 'complete checkpoint rows only; no gold scoring; latency sums are not wall-clock or NPU-hours',
            'new_model_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint, args.sha256)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(result, indent=2, sort_keys=True))
