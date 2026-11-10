"""CPU-only paired policy feedback; no sampling, teacher calls or promotion."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arm(spec, allowed, gold=None):
    terminal = read(spec['terminal'])
    require(terminal['status'] == 'COMPLETE', 'incomplete rollout')
    require(terminal['completed'] == terminal['total'] == 1635, 'branch count')
    require(terminal['unknown'] == terminal['absent'] == 0, 'unresolved transport')
    rows = {}
    for line in Path(spec['scores']).read_text().splitlines():
        row = json.loads(line)
        # Report-only partitions never enter returned records or aggregates.
        if row.get('split') != 'learning':
            continue
        sid = row['sample_id']
        require(sid in allowed, 'unknown learning sample')
        require(sid not in rows, 'duplicate learning sample')
        require(row['candidate_id'] == row['gold_intent'], 'not gold intent branch')
        require(type(row['valid']) is bool, 'invalid validity flag')
        for key in ('slot_types', 'gold_types', 'unresolved_types'):
            if not row['valid'] and key != 'gold_types' and row[key] is None:
                continue
            require(type(row[key]) is list and all(type(v) is str for v in row[key]), key)
        require(bool(row['raw_response']['sha256']), 'missing response evidence')
        if row['valid']:
            require(len(row['slot_types']) == len(set(row['slot_types'])), 'duplicate types')
            require(not set(row['slot_types']) & set(row['unresolved_types']), 'overlap')
        if gold is not None:
            require((row['gold_intent'], set(row['gold_types'])) == gold[sid], 'gold drift')
        rows[sid] = row
    require(set(rows) == allowed, 'incomplete learning476')
    return rows


def reward(row):
    g = set(row['gold_types'])
    t = set(row['slot_types']) if row['valid'] else set()
    tp, fp, fn = len(t & g), len(t - g), len(g - t)
    valid = row['valid']
    return dict(exact=int(valid and t == g),
                type_f1=(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0) if valid else 0.0,
                complete_recall=int(valid and g <= t) if g else None,
                negative_fp=-fp, negative_fn=-fn, protocol_valid=int(valid),
                negative_unresolved=-len(set(row['unresolved_types'])) if valid else None)


def aggregate(rows):
    result = dict(samples=len(rows), exact=0, tp=0, fp=0, fn=0,
                  invalid=0, nonempty=0, complete_recall=0, empty_fp_samples=0)
    for row in rows.values():
        g = set(row['gold_types'])
        t = set(row['slot_types']) if row['valid'] else set()
        result['exact'] += int(row['valid'] and g == t)
        result['invalid'] += int(not row['valid'])
        for k, v in [('tp', t & g), ('fp', t - g), ('fn', g - t)]:
            result[k] += len(v)
        result['nonempty'] += int(bool(g))
        result['complete_recall'] += int(bool(g) and row['valid'] and g <= t)
        result['empty_fp_samples'] += int(not g and bool(t))
    return result


def compare(config):
    allowed_list = read(config['allowlist'])['allowed_sample_ids']
    allowed = set(allowed_list)
    require(len(allowed_list) == len(allowed) == 476, 'fixed learning476 required')
    specs = config['arms']
    require(len(specs) >= 2 and len({s['name'] for s in specs}) == len(specs), 'distinct arms')
    parent = config['parent']
    require(parent in {s['name'] for s in specs}, 'missing parent')
    base_spec = next(s for s in specs if s['name'] == parent)
    baseline = load_arm(base_spec, allowed)
    gold = {sid: (r['gold_intent'], set(r['gold_types'])) for sid, r in baseline.items()}
    arms = {s['name']: load_arm(s, allowed, gold) for s in specs}
    signatures = {}
    for name, rows in arms.items():
        signature = tuple(rows[sid]['wire_sha256'] for sid in sorted(allowed))
        require(signature not in signatures,
                'identical policy inputs: reuse results or analyze as repeats, not distinct strategies')
        signatures[signature] = name
    paired = []
    for sid in sorted(allowed):
        rewards = {name: reward(rows[sid]) for name, rows in arms.items()}
        advantages = {name: {} for name in arms}
        for metric in next(iter(rewards.values())):
            vals = {n: r[metric] for n, r in rewards.items()}
            # Undefined complete-recall for empty gold, and invalid uncertainty,
            # cannot enter a candidate comparison as a free reward.
            if any(v is None for v in vals.values()):
                for name in arms:
                    advantages[name][metric] = None
                continue
            mean = sum(vals.values()) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals.values()) / len(vals))
            for name, value in vals.items():
                advantages[name][metric] = (value - mean) / std if std > 1e-12 else 0.0
        paired.append(dict(sample_id=sid, split='learning', rewards=rewards,
                           descriptive_group_advantages=advantages))
    summary = {}
    for name, rows in arms.items():
        repairs = regressions = same_wire_changes = 0
        for sid in allowed:
            b, c = reward(baseline[sid])['exact'], reward(rows[sid])['exact']
            repairs += int(c > b)
            regressions += int(c < b)
            same_wire_changes += int(rows[sid]['wire_sha256'] == baseline[sid]['wire_sha256']
                                     and (rows[sid]['valid'], set(rows[sid]['slot_types'] or [])) !=
                                     (baseline[sid]['valid'], set(baseline[sid]['slot_types'] or [])))
        summary[name] = dict(metrics=aggregate(rows), repairs=repairs, regressions=regressions,
                             same_wire_output_changes=same_wire_changes,
                             by_intent={intent: aggregate({s:r for s,r in rows.items() if r['gold_intent']==intent})
                                        for intent in sorted({r['gold_intent'] for r in rows.values()})})
    return dict(status='COMPLETE_LEARNING476', parent=parent, arms=summary,
                source_pins={s[k]: digest(s[k]) for s in specs for k in ('scores', 'terminal')},
                allowlist_sha256=digest(config['allowlist']),
                algorithm_identity='Descriptive policy-search vector advantages; no parameter GRPO update',
                promotion=False, scalar_reward=None,
                limitations=['Single rollout per arm; correlated samples and nondeterministic outputs remain.',
                             'Group scores depend on included policies; compare raw paired deltas first.',
                             'No causal separation of hardware or multiple simultaneous input changes.',
                             'Undefined metrics excluded; invalid empty predictions are not exact matches.']), paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result, paired = compare(read(args.config))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'SUMMARY.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (out/'PAIRED.private.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in paired))
    print(json.dumps({n:v['metrics'] for n,v in result['arms'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
