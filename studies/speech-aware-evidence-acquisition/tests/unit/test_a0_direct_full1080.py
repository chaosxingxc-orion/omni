import copy
import importlib.util
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('a0_direct', ROOT / 'scripts/analysis/compare_a0_direct_full1080.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def result(tp, fp, fn):
    ids = ['slurp:train:1:audio-a', 'slurp:train:1:audio-b', 'slurp:train:2:audio-c']
    rows = [dict(sample_id=s, **{f'slu_f1_{d}_{c}': v for d in ('word', 'char')
             for c, v in zip(('tp', 'fp', 'fn'), (tp, fp, fn))}) for s in ids]
    return ids, {'per_sample': rows, 'aggregate': {'samples': 3, 'identity_check': {'passed': True},
        'official': {'slu_f1': 2*tp/(2*tp+fp+fn), 'intent_accuracy': .7},
        'evaluator_commit': 'same', 'scorer_version': 'same', 'split_id': 'train',
        'projection': {'applied': False}, 'filler_normalization': {'applied': False}}}


def test_paired_micro_and_cluster_bootstrap_keeps_all_references():
    ids, direct = result(1, 1, 1)
    _, agent = result(3, 1, 1)
    r = m.paired({'direct_lock': direct, 'direct_aa': direct, 'direct_ab': direct, 'a0': agent}, ids, 200)
    assert r['utterance_clusters'] == 2
    assert len(r['comparisons']) == 3
    assert r['comparisons']['a0_minus_direct_lock']['slu_f1_delta_pp'] == pytest.approx(25)
    assert r['comparisons']['a0_minus_direct_lock']['paired_percentile_95_interval_pp'] == pytest.approx([25, 25])


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'identity', 'aggregate', 'nan', 'negative'])
def test_incomplete_or_invalid_statistics_refused(fault):
    ids, r = result(1, 1, 1)
    if fault == 'duplicate': r['per_sample'][1]['sample_id'] = ids[0]
    elif fault == 'missing': r['per_sample'].pop()
    elif fault == 'identity': r['aggregate']['identity_check']['passed'] = False
    elif fault == 'aggregate': r['aggregate']['official']['slu_f1'] = .9
    elif fault == 'nan': r['per_sample'][0]['slu_f1_word_tp'] = float('nan')
    else: r['per_sample'][0]['slu_f1_word_tp'] = -100
    with pytest.raises(ValueError): m.score_matrix(r, ids)


def test_different_scorer_refused():
    ids, r = result(1, 1, 1)
    other = copy.deepcopy(r); other['aggregate']['projection']['applied'] = True
    with pytest.raises(ValueError, match='normalization'):
        m.paired({'direct_lock': r, 'direct_aa': r, 'direct_ab': r, 'a0': other}, ids, 20)


def test_missing_direct_reference_refused():
    ids, r = result(1, 1, 1)
    with pytest.raises(ValueError, match='all three'):
        m.paired({'direct_lock': r, 'a0': r}, ids)


def test_queue_requires_successful_full_source():
    import sys
    host = ROOT.parents[1] / 'scripts/npu'
    if not (host / 'a0_direct_comparison_queue.py').exists(): pytest.skip('public host driver required')
    sys.path.insert(0, str(host))
    import a0_direct_comparison_queue as q
    assert not q.ready({'state': 'waiting_finalized353'}, {'rows': 1080})
    assert q.ready({'state': 'complete'}, {'rows': 1080})
    with pytest.raises(ValueError): q.ready({'state': 'failed'}, None)
    with pytest.raises(ValueError): q.ready({'state': 'complete'}, {'rows': 727})
