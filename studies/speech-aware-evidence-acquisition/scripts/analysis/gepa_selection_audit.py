"""Replay pinned paired summaries without authorizing promotion or model calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path


ARMS = ('parent', 'child', 'lock')


def number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError('finite numeric metric required')
    return value


def sign(value, tolerance):
    return 0 if abs(value) <= tolerance else (1 if value > 0 else -1)


def summarize(result, *, tolerance=1e-12):
    """Screen an existing reader artifact, not re-verify its native provenance."""
    if result.get('schema') != 'saea-gepa-pair-result-v1':
        raise ValueError('full paired reader result required; public summary insufficient')
    ids = result['sample_ids']; rows = result['rows']
    if (type(rows) is not int or rows < 1 or not isinstance(ids, list)
            or any(not isinstance(s, str) or not s.startswith('slurp:train:') for s in ids)
            or len(ids) != rows or len(set(ids)) != rows):
        raise ValueError('complete unique TRAIN roster required')
    audit = result['input_audit']
    audited = [r['sample_id'] for r in audit['rows']]
    abstained = audit['abstention_rows']
    if (audit['audited_rows'] != len(audited) or len(set(audited + abstained)) != rows
            or set(audited + abstained) != set(ids) or len(audited + abstained) != rows):
        raise ValueError('complete paired input audit coverage required')
    if set(result['official']) != set(ARMS) or set(result['proxy_scores']) != set(ARMS):
        raise ValueError('exact paired arms required')
    for arm in ARMS:
        scores = result['proxy_scores'][arm]
        if len(scores) != rows or any(not 0 <= number(v) <= 1 for v in scores):
            raise ValueError('complete bounded proxy scores required')
        if not 0 <= number(result['official'][arm]['slu_f1']) <= 1:
            raise ValueError('bounded official SLU required')
        success = result['strict_success'][arm]
        if type(success) is not int or not 0 <= success <= rows:
            raise ValueError('bounded strict success count required')
    pairs = {}
    for first, second in (('child', 'parent'), ('child', 'lock'), ('parent', 'lock')):
        key = first + '_vs_' + second
        comparison = result['comparisons'][key]
        repairs, breaks = (comparison[k] for k in ('strict_repairs', 'strict_breaks'))
        if (any(type(v) is not int or not 0 <= v <= rows for v in (repairs, breaks))
                or repairs + breaks > rows
                or repairs - breaks != result['strict_success'][first] - result['strict_success'][second]
                or repairs > min(result['strict_success'][first], rows-result['strict_success'][second])
                or breaks > min(result['strict_success'][second], rows-result['strict_success'][first])):
            raise ValueError('strict paired counts inconsistent')
        proxy = (sum(result['proxy_scores'][first])-sum(result['proxy_scores'][second]))/rows
        if abs(proxy-number(comparison['proxy_mean_delta'])) > tolerance:
            raise ValueError('proxy aggregate disagrees with per-row scores')
        official = result['official'][first]['slu_f1'] - result['official'][second]['slu_f1']
        pairs[key] = {'official_slu_delta_pp': 100*official, 'proxy_mean_delta': proxy,
            'direction_agreement': sign(official, tolerance) == sign(proxy, tolerance),
            'strict_repairs': repairs, 'strict_breaks': breaks,
            'screen_no_slu_regression': official >= -tolerance,
            'screen_zero_strict_breaks': breaks == 0}
    return {'rows': rows, 'comparisons': pairs,
        'source_manifest_sha256': result['source']['manifest_sha256'],
        'source_config_hash': result['source']['config_hash'],
        'shared_perception_sha256': audit['shared_perception_sha256'],
        'shared_supply_sha256': audit['shared_supply_sha256'],
        'lock_official_slu': result['official']['lock']['slu_f1'],
        'identity_limit': 'full config hash includes candidate/run identities; not a lock-only scientific identity',
        'candidate_accepted': False, 'claims_stable_gain': False}


def run(config_path):
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_bytes())
    if set(cfg) != {'schema', 'results'} or cfg['schema'] != 'saea-gepa-selection-audit-v1':
        raise ValueError('exact selection audit config required')
    if not isinstance(cfg['results'], dict) or not cfg['results']:
        raise ValueError('named pinned results required')
    results, summaries = {}, {}
    for name, pin in cfg['results'].items():
        if set(pin) != {'path', 'sha256'}:
            raise ValueError('exact result pin required')
        path = Path(pin['path'])
        if not path.is_absolute():
            path = config_path.resolve().parent/path
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin['sha256']:
            raise ValueError('paired result changed')
        result = json.loads(raw); results[name] = result
        summaries[name] = {**summarize(result), 'reader_result_sha256': pin['sha256']}
    overlaps = []
    names = list(results)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            a, b = results[first], results[second]
            overlaps.append({'first': first, 'second': second,
                'shared_samples': len(set(a['sample_ids']) & set(b['sample_ids'])),
                'same_ordered_roster': a['sample_ids'] == b['sample_ids'],
                'same_recorded_perception': a['input_audit']['shared_perception_sha256'] == b['input_audit']['shared_perception_sha256'],
                'lock_slu_difference_pp': 100*(b['official']['lock']['slu_f1']-a['official']['lock']['slu_f1']),
                'lock_drift_cause': 'unresolved: compare lock-only config, templates, scorer, and actual input identities'})
    return {'schema': 'saea-gepa-selection-audit-result-v1', 'results': summaries,
        'cross_run_comparisons': overlaps,
        'unique_samples': len({s for d in results.values() for s in d['sample_ids']}),
        'evaluated_sample_occurrences': sum(d['rows'] for d in results.values()),
        'new_model_calls': 0, 'candidate_accepted': False, 'claims_stable_gain': False,
        'scope': 'retrospective screen of pinned reader results; not native bundle revalidation or preregistered promotion',
        'missing_for_promotion': ['pre-registered protocol', 'independent confirmation',
                                  'lock-only scientific identity and scorer reconciliation']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config)
    with args.out.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
