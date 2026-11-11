"""Offline full-denominator S1 scoring; consume actual SlotStages aggregates.

No transcript, entity filler, intent or reference count is used to interpret S1.
Unknown/missing/partial folds retain observed candidates, never earn complete-set
credit. Reference-unmatched candidates describe annotation burden, not semantic
proof of errors. This scorer never calls a model.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

VALID = {'valid_empty', 'valid_nonempty'}


def lines(path):
    with Path(path).open() as f:
        return [json.loads(x) for x in f if x.strip()]


def ratio(n, d):
    return n / d if d else None


def prf(tp, fp, fn):
    return {'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
            'f1': ratio(2 * tp, 2 * tp + fp + fn),
            'true_positive': tp, 'reference_unmatched': fp, 'missing': fn}


def save(path, value, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        if jsonl:
            for row in value:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        else:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fold_map(folds):
    if isinstance(folds, dict):
        folds = folds['folds']
    out = {f['fold_id']: set(f['types']) for f in folds}
    assigned = [t for f in folds for t in f['types']]
    if len(out) != len(folds) or len(assigned) != len(set(assigned)):
        raise ValueError('fold ids/types must be disjoint and unique')
    if not out or any(not ts for ts in out.values()):
        raise ValueError('empty folds')
    return out


def normalize(row, folds):
    """Validate identities/completeness without replacing real after-rails results."""
    errors = []
    branches = row.get('s1')
    if not isinstance(branches, list):
        return {'selected': set(), 'complete': False, 'folds': {}, 'errors': ['missing_s1_aggregate'],
                'states': collections.Counter({'aggregate_protocol_error': 1})}
    grouped = collections.defaultdict(list)
    for b in branches:
        if not isinstance(b, dict) or b.get('branch') not in folds or b.get('stage', 'S1') != 'S1':
            errors.append('unknown_branch_or_stage')
            continue
        grouped[b['branch']].append(b)
    results = {}
    states = collections.Counter()
    for fid, allowed in folds.items():
        bs = grouped.get(fid, [])
        if not bs:
            results[fid] = {'selected': set(), 'valid': False, 'status': 'absent', 'raw': None}
            states['absent'] += 1
            continue
        if len(bs) != 1:
            errors.append('duplicate_branch:' + fid)
            results[fid] = {'selected': set(), 'valid': False, 'status': 'duplicate_branch', 'raw': None}
            states['duplicate_branch'] += 1
            continue
        b = bs[0]
        status = b.get('status', 'missing_status')
        raw = b.get('selected_types', [])
        if not isinstance(raw, list) or any(not isinstance(t, str) for t in raw):
            raw = []
            errors.append('invalid_selected_types:' + fid)
            status = 'protocol_error'
        selected = set(raw) & allowed
        if set(raw) - allowed:
            errors.append('outside_fold:' + fid)
            status = 'protocol_error'
        if b.get('sample_id', row['sample_id']) != row['sample_id']:
            selected = set()
            errors.append('branch_sample_mismatch:' + fid)
            status = 'protocol_error'
        # Unknown/transport failures have no trustworthy typed evidence.
        if status not in VALID | {'partial'}:
            selected = set()
        if status in VALID and ((status == 'valid_empty') != (not selected)):
            status = 'protocol_error'
            errors.append('status_content_mismatch:' + fid)
        if b.get('finish_reason') not in (None, 'stop') and status in VALID:
            status = 'incomplete'
            selected = set()
        valid = status in VALID
        results[fid] = {'selected': selected, 'valid': valid, 'status': status, 'raw': b}
        states[status] += 1
    selected = set().union(*(b['selected'] for b in results.values()))
    top = row.get('selected_types')
    if not isinstance(top, list) or any(not isinstance(t, str) for t in top) or set(top) != selected:
        errors.append('aggregate_union_mismatch')
    complete = all(b['valid'] for b in results.values()) and not errors
    if row.get('complete') is not True:
        complete = False
    return {'selected': selected, 'complete': complete, 'folds': results,
            'errors': errors, 'states': states}


def score(gold_rows, predictions, fold_config, runtime_cost=None):
    folds = fold_map(fold_config)
    universe = set().union(*folds.values())
    refs = {g['sample_id']: set(g['gold_types']) for g in gold_rows}
    if len(refs) != len(gold_rows):
        raise ValueError('duplicate reference sample')
    if any(not g <= universe for g in refs.values()):
        raise ValueError('gold type outside global universe')
    by = {}
    for p in predictions:
        sid = p['sample_id']
        if sid not in refs or sid in by:
            raise ValueError('unknown or duplicate prediction sample: ' + sid)
        by[sid] = p
    per_type = {t: collections.Counter() for t in sorted(universe)}
    per_fold = {f: collections.Counter() for f in folds}
    status = collections.Counter()
    total = collections.Counter()
    usage = collections.Counter()
    raw_call_seconds = 0.0
    stage_timing_seen = 0
    transport_http_seconds = 0.0
    transport_timing_seen = 0
    finish_reasons = collections.Counter()
    usage_missing = 0
    costs_seen = set()
    details = []
    for sid, gold in refs.items():
        p = by.get(sid)
        normalized = normalize(p, folds) if p is not None else {
            'selected': set(), 'complete': False, 'folds': {}, 'errors': [],
            'states': collections.Counter({'absent': len(folds)})}
        pred, complete = normalized['selected'], normalized['complete']
        missing, extra = gold - pred, pred - gold
        total['samples'] += 1
        total['present_samples'] += p is not None
        total['complete_samples'] += complete
        total['positive_samples'] += bool(gold)
        total['empty_gold_samples'] += not gold
        total['type_set_exact'] += complete and pred == gold
        total['count_match'] += complete and len(pred) == len(gold)
        total['complete_positive_coverage'] += complete and bool(gold) and not missing
        total['observed_positive_coverage'] += bool(gold) and not missing
        total['all_sample_complete_coverage'] += complete and not missing
        total['empty_gold_correct'] += complete and not gold and not pred
        total['reference_unmatched'] += len(extra)
        total['missing_types'] += len(missing)
        total['tp'] += len(pred & gold)
        total['fp'] += len(extra)
        total['fn'] += len(missing)
        total['aggregate_protocol_error_samples'] += bool(normalized['errors'])
        total['present_but_incomplete_samples'] += p is not None and not complete
        status.update(normalized['states'])
        for branch in normalized['folds'].values():
            raw = branch['raw']
            if raw is not None:
                finish_reasons[str(raw.get('finish_reason'))] += 1
        for t in universe:
            c = per_type[t]
            c['support'] += t in gold
            c['tp'] += t in gold and t in pred
            c['fp'] += t not in gold and t in pred
            c['fn'] += t in gold and t not in pred
        for fid, allowed in folds.items():
            b = normalized['folds'].get(fid)
            g = gold & allowed
            bp = b['selected'] if b else set()
            valid = b['valid'] if b else False
            c = per_fold[fid]
            c['samples'] += 1
            c['valid_samples'] += valid
            c['exact_samples'] += valid and g == bp
            c['positive_samples'] += bool(g)
            c['positive_complete_coverage'] += valid and bool(g) and g <= bp
            c['tp'] += len(g & bp)
            c['fp'] += len(bp - g)
            c['fn'] += len(g - bp)
            raw = b['raw'] if b else None
            if raw:
                # Unique paid request receipts preserve historical/reused costs once.
                request = raw.get('request_id', sid + ':' + fid)
                if request not in costs_seen:
                    costs_seen.add(request)
                    u = raw.get('usage')
                    if isinstance(u, dict):
                        for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                            if isinstance(u.get(k), (int, float)):
                                usage[k] += u[k]
                    else:
                        usage_missing += 1
                    timing = raw.get('stage_timing') or raw.get('timing') or {}
                    if isinstance(timing.get('call_and_rails_seconds'), (int, float)):
                        raw_call_seconds += timing['call_and_rails_seconds']
                        stage_timing_seen += 1
                    transport = raw.get('transport_timing') or raw.get('timing') or {}
                    http = transport.get('new_http_seconds')
                    if isinstance(http, (int, float)):
                        transport_http_seconds += http
                        transport_timing_seen += 1
        details.append({'sample_id': sid, 'gold_types': sorted(gold), 'selected_types': sorted(pred),
                        'complete': complete, 'type_set_exact': complete and pred == gold,
                        'positive_complete_coverage': complete and bool(gold) and not missing,
                        'observed_positive_coverage': bool(gold) and not missing,
                        'reference_unmatched_types': sorted(extra), 'missing_types': sorted(missing),
                        'fold_states': dict(normalized['states']), 'protocol_errors': normalized['errors']})
    for key in ('samples', 'present_samples', 'complete_samples', 'positive_samples', 'empty_gold_samples',
                'type_set_exact', 'count_match', 'complete_positive_coverage', 'observed_positive_coverage',
                'all_sample_complete_coverage', 'empty_gold_correct', 'reference_unmatched', 'missing_types',
                'tp', 'fp', 'fn', 'aggregate_protocol_error_samples', 'present_but_incomplete_samples'):
        total[key] += 0
    n = total['samples']
    pt = {t: {'support': c['support'], **prf(c['tp'], c['fp'], c['fn'])} for t, c in per_type.items()}
    supported = [v for v in pt.values() if v['support']]
    macro = {k: sum(v[k] or 0.0 for v in supported) / len(supported) if supported else None for k in ('precision', 'recall', 'f1')}
    pf = {fid: {**dict(c), **prf(c['tp'], c['fp'], c['fn']),
                'type_set_exact_accuracy': ratio(c['exact_samples'], n),
                'positive_complete_coverage': ratio(c['positive_complete_coverage'], c['positive_samples'])}
          for fid, c in per_fold.items()}
    summary = {'schema': 's1-type-score-v1', 'status': 'COMPLETE' if total['complete_samples'] == n else 'PARTIAL',
               'evaluation_scope': 'historical P-filtered TRAIN learning cohort; not generalization',
               'denominator': n, 'counts': dict(total), 'global_types': len(universe),
               'metrics': {'type_set_exact_accuracy': ratio(total['type_set_exact'], n),
                           'positive_complete_type_coverage': ratio(total['complete_positive_coverage'], total['positive_samples']),
                           'positive_observed_type_coverage': ratio(total['observed_positive_coverage'], total['positive_samples']),
                           'all_sample_complete_type_coverage': ratio(total['all_sample_complete_coverage'], n),
                           'type_count_match_accuracy': ratio(total['count_match'], n),
                           'mean_reference_unmatched_candidates': ratio(total['reference_unmatched'], n),
                           'mean_missing_types': ratio(total['missing_types'], n),
                           'empty_gold_correct_rate': ratio(total['empty_gold_correct'], total['empty_gold_samples']),
                           'micro_observed': prf(total['tp'], total['fp'], total['fn']),
                           'macro_supported_types_observed': {**macro, 'supported_types': len(supported),
                                                              'zero_prediction_precision_convention': 0}},
               'fold_status_counts': dict(status), 'finish_reason_counts': dict(finish_reasons),
               'truncated_fold_results': finish_reasons['length'],
               'per_type': pt, 'per_fold': pf,
               'cost': {'unique_observed_requests': len(costs_seen), 'usage': dict(usage),
                        'coverage': 'Observed aggregate rows only; full paid/unknown request ledger is runtime_receipt.',
                        'requests_without_usage': usage_missing,
                        'requests_with_stage_timing': stage_timing_seen,
                        'summed_call_and_rails_seconds': raw_call_seconds if stage_timing_seen else None,
                        'requests_with_transport_timing': transport_timing_seen,
                        'summed_observed_new_http_seconds': transport_http_seconds if transport_timing_seen else None,
                        'wall_seconds': None, 'cards_per_instance': 4, 'instances': None,
                        'seconds_per_complete_sample': None, 'remaining_seconds_estimate': None, 'money': None,
                        'runtime_receipt': runtime_cost},
               'interpretation': ['Failed/missing folds remain in full denominator and cannot earn exact/count/complete-coverage credit.',
                                  'Observed micro/macro retain trustworthy candidates from partial folds; missing types include unfinished folds.',
                                  'Reference-unmatched candidates are annotation burden, not established semantic errors.',
                                  'Empty gold contributes only to separately labelled all-sample coverage, never positive coverage.',
                                  'Null/equal-type fillers do not change gold distinct type sets.']}
    return summary, details


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gold', required=True)
    p.add_argument('--predictions', action='append', default=[])
    p.add_argument('--folds', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--runtime-cost')
    args = p.parse_args()
    predictions = [r for file in args.predictions for r in lines(file)]
    summary, details = score(lines(args.gold), predictions, json.loads(Path(args.folds).read_bytes()),
                             json.loads(Path(args.runtime_cost).read_bytes()) if args.runtime_cost else None)
    summary['input_pins'] = {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                             for path in [args.gold, args.folds] + args.predictions}
    save(Path(args.out) / 'SUMMARY.json', summary)
    save(Path(args.out) / 'scores.private.jsonl', details, True)
    print(json.dumps({k: summary[k] for k in ['status', 'denominator', 'counts', 'metrics', 'fold_status_counts']}, indent=2))


if __name__ == '__main__':
    main()
