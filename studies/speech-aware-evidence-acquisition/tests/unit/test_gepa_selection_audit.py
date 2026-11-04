import copy
import hashlib
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]/'scripts/analysis'))
import gepa_selection_audit as m


def result():
    return {'schema': 'saea-gepa-pair-result-v1', 'rows': 2,
        'sample_ids': ['slurp:train:1', 'slurp:train:2'],
        'input_audit': {'audited_rows': 2, 'rows': [{'sample_id': 'slurp:train:1'}, {'sample_id': 'slurp:train:2'}],
            'abstention_rows': [], 'shared_perception_sha256': 'a'*64, 'shared_supply_sha256': 'b'*64},
        'official': {a: {'slu_f1': .6 if a == 'child' else .7} for a in m.ARMS},
        'proxy_scores': {a: [.8 if a == 'child' else .7, 1.] for a in m.ARMS},
        'strict_success': dict.fromkeys(m.ARMS, 1),
        'comparisons': {a+'_vs_'+b: {'strict_repairs': 0, 'strict_breaks': 0,
            'proxy_mean_delta': .05 if a == 'child' else 0.} for a, b in [('child','parent'), ('child','lock'), ('parent','lock')]},
        'source': {'manifest_sha256': 'c'*64, 'config_hash': 'd'*64}}


def test_proxy_gain_does_not_hide_official_regression():
    audit = m.summarize(result())
    pair = audit['comparisons']['child_vs_parent']
    assert not pair['direction_agreement'] and not pair['screen_no_slu_regression']
    assert not audit['candidate_accepted']


@pytest.mark.parametrize('fault', ['missing_proxy', 'nan', 'boolean', 'bad_counts', 'aggregate', 'audit_overlap', 'test_sample'])
def test_inconsistent_reader_evidence_rejected(fault):
    r = result()
    if fault == 'missing_proxy': r['proxy_scores']['child'].pop()
    elif fault == 'nan': r['official']['lock']['slu_f1'] = float('nan')
    elif fault == 'boolean': r['strict_success']['lock'] = True
    elif fault == 'bad_counts': r['comparisons']['child_vs_parent']['strict_repairs'] = 2
    elif fault == 'aggregate': r['comparisons']['child_vs_parent']['proxy_mean_delta'] = .2
    elif fault == 'audit_overlap': r['input_audit']['rows'][1] = r['input_audit']['rows'][0]
    else: r['sample_ids'][0] = 'slurp:test:1'
    with pytest.raises(ValueError): m.summarize(r)


def test_repeated_roster_is_not_independent_or_configuration_drift(tmp_path):
    a = result(); b = copy.deepcopy(a)
    b['input_audit']['shared_perception_sha256'] = 'e'*64
    b['official']['lock']['slu_f1'] = .69
    pins = {}
    for name, doc in [('a', a), ('b', b)]:
        path = tmp_path/(name+'.json'); path.write_text(json.dumps(doc))
        pins[name] = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    cfg = tmp_path/'config.json'
    cfg.write_text(json.dumps({'schema': 'saea-gepa-selection-audit-v1', 'results': pins}))
    audit = m.run(cfg)
    assert audit['unique_samples'] == 2 and audit['evaluated_sample_occurrences'] == 4
    pair = audit['cross_run_comparisons'][0]
    assert pair['shared_samples'] == 2 and not pair['same_recorded_perception']
    assert pair['lock_drift_cause'].startswith('unresolved')
    (tmp_path/'a.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'): m.run(cfg)
