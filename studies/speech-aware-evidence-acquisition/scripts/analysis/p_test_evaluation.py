"""Offline accounting for sealed P-only results; never contacts a model or reads audio."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(roster, rows, exceptions):
    ids = roster_ids(roster)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('empty or duplicate roster')
    by_id = {}
    for row in rows:
        sid = row['sample_id']
        if sid in by_id:
            raise ValueError('duplicate result')
        by_id[sid] = row
    if set(by_id) != set(ids):
        raise ValueError('result identities must exactly match roster')
    exc = {}
    for entry in exceptions:
        sid = entry['sample_id']
        if sid in exc or sid not in by_id or entry.get('status') != 'confirmed':
            raise ValueError('invalid exception identity or status')
        for key in ('missing_information', 'checks', 'rationale', 'downstream_evidence'):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError('confirmed exception requires evidence and downstream assessment')
        exc[sid] = entry
    supplied = correct = valid = nonexception_correct = recovered = 0
    for sid in ids:
        r = by_id[sid]
        gold, supply, indices = r['gold'], r['supply'], r['output_indices']
        if not isinstance(gold, str) or not gold or not isinstance(supply, list) or any(not isinstance(x, str) or not x for x in supply) or len(supply) != len(set(supply)):
            raise ValueError('invalid gold or ordered supply')
        # 1-based indices, exactly min(3, supply size), no implicit failure recovery.
        ok = (r['status'] == 'ok' and isinstance(indices, list) and len(indices) == min(3, len(supply)) and bool(supply)
              and all(type(i) is int and 1 <= i <= len(supply) for i in indices) and len(indices) == len(set(indices)))
        has = gold in supply
        hit = ok and gold in [supply[i-1] for i in indices]
        supplied += has
        valid += ok
        correct += hit
        nonexception_correct += hit and sid not in exc
        if sid in exc:
            state = exc[sid].get('downstream_recovered')
            if state is not None and type(state) is not bool:
                raise ValueError('invalid downstream recovery')
            recovered += state is True
    n = len(ids)
    return {'total': n, 'supply': {'correct': supplied, 'total': n},
            'p_top3': {'correct': correct, 'total': n},
            'conditional_retention': {'correct': correct, 'total': supplied},
            'valid_output': {'valid': valid, 'total': n, 'all_valid': valid == n},
            'confirmed_exceptions': len(exc),
            'nonexception_top3': {'correct': nonexception_correct, 'total': n-len(exc)},
            'downstream_recovery': {'recovered': recovered, 'confirmed_exceptions': len(exc),
                                    'assessed': sum(e.get('downstream_recovered') is not None for e in exc.values())},
            'acceptance': valid == n and n > len(exc) and nonexception_correct == n-len(exc)}


def roster_ids(roster):
    return roster['sample_ids'] if 'sample_ids' in roster else [item['sample_id'] for item in roster['items']]


def map_checkpoint(roster, checkpoint, gold_rows):
    """Map the archived ranked-index row format using the original scorer's checks."""
    gold = {}
    for row in gold_rows:
        sid, label = row['sample_id'], row['gold_intent']
        if sid in gold or not isinstance(label, str) or not label:
            raise ValueError('duplicate or invalid gold')
        gold[sid] = label
    ids = roster_ids(roster)
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(gold):
        raise ValueError('gold identities must exactly match unique roster')
    rows = []
    seen = set()
    header_seen = False
    for line in checkpoint.splitlines():
        record = json.loads(line)
        kind = record.get('kind')
        if kind == 'header' and not header_seen and not rows:
            header_seen = True
            continue
        if kind != 'row' or not header_seen:
            raise ValueError('unexpected checkpoint record')
        sid = record['sample_id']
        if sid in seen or sid not in gold:
            raise ValueError('duplicate or unexpected checkpoint identity')
        seen.add(sid)
        outputs = record.get('output_rows')
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise ValueError('cannot recover unique archived output and supply')
        output = outputs[0]
        ranking = output.get('node_outcome', {}).get('ranking_result') or {}
        inputs = ranking.get('input')
        if not isinstance(inputs, list):
            raise ValueError('cannot recover archived supply')
        supply = [item['intent_id'] for item in inputs]
        if any(type(item.get('index')) is not int or item['index'] != i for i, item in enumerate(inputs, 1)):
            raise ValueError('archived supply index mismatch')
        indices = ranking.get('indices')
        try:
            raw = json.loads(output['text'])
        except (ValueError, TypeError, KeyError):
            raw = None
        index_ok = (isinstance(indices, list) and all(type(i) is int and 1 <= i <= len(supply) for i in indices))
        selected = [supply[i-1] for i in indices] if index_ok else None
        consistent = (record.get('complete') is True and ranking.get('status') == 'ok'
                      and output.get('sample_id') == sid and isinstance(raw, list)
                      and all(type(i) is int for i in raw) and raw == indices
                      and selected is not None and ranking.get('candidate_ids') == selected)
        rows.append({'sample_id': sid, 'gold': gold[sid], 'supply': supply,
                     'output_indices': indices, 'status': 'ok' if consistent else 'invalid'})
    if not header_seen or seen != set(ids):
        raise ValueError('checkpoint identities must exactly match roster')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roster', required=True, type=Path)
    p.add_argument('--roster-sha256', required=True)
    p.add_argument('--expected-count', required=True, type=int)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--rows', type=Path, help='private sealed normalized result JSON array')
    source.add_argument('--checkpoint', type=Path, help='sealed ranked-index JSONL checkpoint')
    p.add_argument('--checkpoint-sha256')
    p.add_argument('--gold', type=Path, help='independent private gold array with sample_id and gold_intent')
    p.add_argument('--gold-sha256')
    p.add_argument('--normalized-out', type=Path, help='optional private mapped rows, create only')
    p.add_argument('--exceptions', required=True, type=Path, help='private confirmed exception JSON array')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if digest(a.roster) != a.roster_sha256:
        raise ValueError('roster pin mismatch')
    roster = json.loads(a.roster.read_text())
    if len(roster_ids(roster)) != a.expected_count:
        raise ValueError('roster count mismatch')
    inputs = {'roster_sha256': a.roster_sha256, 'exceptions_sha256': digest(a.exceptions)}
    if a.checkpoint:
        if not a.gold or not a.gold_sha256 or not a.checkpoint_sha256:
            p.error('--checkpoint requires --checkpoint-sha256, --gold and --gold-sha256')
        if digest(a.checkpoint) != a.checkpoint_sha256 or digest(a.gold) != a.gold_sha256:
            raise ValueError('checkpoint or gold pin mismatch')
        rows = map_checkpoint(roster, a.checkpoint.read_text(), json.loads(a.gold.read_text()))
        inputs.update(checkpoint_sha256=a.checkpoint_sha256, gold_sha256=a.gold_sha256)
    else:
        if a.gold or a.gold_sha256 or a.checkpoint_sha256 or a.normalized_out:
            p.error('checkpoint options require --checkpoint')
        rows = json.loads(a.rows.read_text())
        inputs['rows_sha256'] = digest(a.rows)
    result = evaluate(roster, rows, json.loads(a.exceptions.read_text()))
    result['inputs'] = inputs
    if a.normalized_out:
        with a.normalized_out.open('x') as f:
            json.dump(rows, f, ensure_ascii=False)
            f.write('\n')
    with a.out.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')


if __name__ == '__main__':
    main()
