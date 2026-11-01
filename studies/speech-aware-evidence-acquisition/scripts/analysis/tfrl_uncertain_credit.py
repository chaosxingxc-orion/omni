"""Learning-side contrast candidates without treating shared-P arms as repeats.

Existing v2 credit and stopped-family decisions are immutable. This read emits
review candidates, never deployable rules or a generation authorization.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis')]
from tfrl_current_half_result import ARMS, score_table
from tfrl_current_sources import read, sha
from tfrl_recovered_consumers import load, POOL
from tfrl_current_pool import pool
import paired_ci


def checked(row):
    paired_ci.row_counts(row)
    intent = row['scenario_action_exact']
    if type(intent) not in (int, float) or intent not in (0, 1):
        raise ValueError('binary intent score required')
    counts = []
    for stat in ('tp', 'fp', 'fn'):
        value = row['slot_f1_exact_' + stat]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('finite nonnegative exact counts required')
        counts.append(value)
    return {'f1': paired_ci.row_f1(row), 'intent_ok': bool(intent),
            'task_error': not intent or counts[1] > 0 or counts[2] > 0}


def propose(row_id, rows):
    """Retain uncertain failures, and require benefit beyond both observed controls.

    The control envelope is descriptive, not a confidence interval. Best-of-K
    evidence needs cross-row validation before a curator may publish a rule.
    """
    if not isinstance(row_id, str) or not row_id or set(rows) != set(ARMS):
        raise ValueError('one named row with exactly six observed arms required')
    values = {arm: checked(row) for arm, row in rows.items()}
    controls = [values[a] for a in ('aa', 'lock')]
    error_seen = any(c['task_error'] for c in controls)
    score_varies = abs(controls[0]['f1'] - controls[1]['f1']) > 1e-9
    outcome_varies = controls[0]['task_error'] != controls[1]['task_error']
    intent_varies = controls[0]['intent_ok'] != controls[1]['intent_ok']
    lower = min(c['f1'] for c in controls)
    upper = max(c['f1'] for c in controls)
    candidates = []
    if error_seen:
        for arm in ARMS[:4]:
            v = values[arm]
            if not v['intent_ok']:
                continue
            # Correcting intent with tied slot quality is useful on empty-slot
            # utterances; a strict slot-F1-only rule would miss it entirely.
            slot_gain = v['f1'] > upper + 1e-9
            intent_gain = not any(c['intent_ok'] for c in controls) and v['f1'] >= upper - 1e-9
            if slot_gain or intent_gain:
                candidates.append({'arm': arm, 'row_f1': v['f1'],
                    'f1_gain_above_control_max': v['f1'] - upper,
                    'intent_repair_above_both_controls': intent_gain})
    candidates.sort(key=lambda c: (-c['f1_gain_above_control_max'],
                                  -int(c['intent_repair_above_both_controls']), c['arm']))
    return {'row_id': row_id, 'learning_candidate': error_seen,
        'confirmed_hard_example': False, 'independent_full_system_repeats': 0,
        'shared_perception_controls': 2,
        'control_score_varies': score_varies, 'control_task_outcome_varies': outcome_varies,
        'control_intent_varies': intent_varies,
        'control_f1_envelope': [lower, upper], 'contrast_candidates': candidates,
        'next_action': ('cross_fold_validate_contrast' if candidates else
                        'diagnose_or_repeat' if error_seen else 'keep_for_regression'),
        'publishable_rule': False}


def audit(halves):
    records = []
    old = Counter()
    for half in sorted(halves):
        part = halves[half]
        ids = [r['row_id'] for r in part['plan']['row_plan']]
        tables = {a: score_table(part['results'][a], ids) for a in ARMS}
        credits = {c['row_id']: c for c in part['credit']}
        if len(credits) != len(part['credit']) or set(credits) != set(ids):
            raise ValueError('credit identity differs')
        for sid in ids:
            item = propose(sid, {a: tables[a][sid] for a in ARMS})
            item.update(half=half, old_reason=credits[sid]['reason'])
            old[item['old_reason']] += 1
            records.append(item)
    if len({r['row_id'] for r in records}) != len(records):
        raise ValueError('overlapping halves')
    result = {'rows': len(records), 'old_credit_reasons': dict(old),
        'learning_candidates': sum(r['learning_candidate'] for r in records),
        'with_contrast_candidates': sum(bool(r['contrast_candidates']) for r in records),
        'old_unstable_retained_for_learning': sum(r['learning_candidate'] and r['old_reason'] == 'baseline-unstable' for r in records),
        'old_unstable_with_contrast': sum(bool(r['contrast_candidates']) and r['old_reason'] == 'baseline-unstable' for r in records),
        'intent_only_contrast_rows': sum(any(c['intent_repair_above_both_controls'] and c['f1_gain_above_control_max'] <= 1e-9 for c in r['contrast_candidates']) for r in records),
        'next_actions': dict(Counter(r['next_action'] for r in records)),
        'confirmed_hard_examples': 0, 'publishable_rules': 0}
    return records, result


def run(repo, data, source, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    assert_execution_environment()
    head = require_clean_study_repo(repo)
    source_head = require_clean_study_repo(source)
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned output directory required')
    _, _, halves = load(source, data)
    verified = pool({p['plan']['half']: (p['plan'], p['results']) for p in halves.values()})
    parent = data/'analysis'/POOL/'complete.json'
    record = read(parent)
    if any(record.get(k) != v for k, v in verified.items()) or record['sources'] != {h:p['source'] for h,p in halves.items()}:
        raise ValueError('pool differs from verified source custody')
    rows, result = audit(halves)
    out.mkdir(parents=True)
    row_path = out/'rows.jsonl'
    publish_new(row_path, b''.join((json.dumps(r, sort_keys=True)+'\n').encode() for r in rows))
    result.update(schema='saea-tfrl-uncertain-credit-read-v1', implementation_commit=head,
        source_commit=source_head, pool_sha256=sha(parent), rows_sha256=sha(row_path),
        sources=record['sources'], new_model_calls=0, new_tokenizer_calls=0, new_scoring_jobs=0,
        generation_authorized=False, system_accepted=False,
        limitations=['Shared-P controls do not estimate independent full-system error probability.',
            'Control envelope is descriptive, not a confidence interval.',
            'Post-hoc best-of-K contrast candidates require independent cross-row validation.',
            'Existing v2 credit, family stop decision and experience banks remain unchanged.',
            'Exact-slot errors are screening signals; annotation/alias ambiguity needs review.'])
    publish_new(out/'complete.json', (json.dumps(result, indent=2)+'\n').encode())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'data', 'source', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    print(json.dumps(run(**vars(parser.parse_args())), sort_keys=True))
