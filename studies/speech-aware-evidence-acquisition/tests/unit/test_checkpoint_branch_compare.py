import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('branch_compare', Path(__file__).resolve().parents[2] / 'scripts/analysis/checkpoint_branch_compare.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(root):
    # Two recordings of one utterance; paired bootstrap must use one cluster.
    rows = [{'sample_id': f'slurp:train:000001:audio-{i}',
             **{f'slu_f1_{d}_{c}': float(c == 'tp') for d in ('word', 'char') for c in ('tp', 'fp', 'fn')}} for i in (1, 2)]
    summary = {'rows': 2, 'checkpoint_sha256': 'test', 'columns': {}}
    for name in ('lock', 'l1-support', 'l1-support-sample0'):
        score = {'per_sample': copy.deepcopy(rows), 'aggregate': {'official': {'slu_f1': 1., 'intent_accuracy': 1., 'span_f1': 1.}}}
        if name == 'l1-support': score['per_sample'].reverse()
        raw = json.dumps(score).encode()
        (root / (name + '.scores.json')).write_bytes(raw)
        summary['columns'][name] = {'scores_sha256': hashlib.sha256(raw).hexdigest()}
    (root / 'complete.json').write_text(json.dumps(summary))


def test_pairing_and_utterance_clustering(tmp_path):
    fixture(tmp_path)
    result = MODULE.compare(tmp_path, replicates=100)
    assert result['utterance_clusters'] == 1
    assert result['rows'] == 2
    assert all(v['exploratory_percentile_95_interval_pp'] == [0., 0.] for v in result['paired_comparisons'].values())


def test_rejects_changed_score(tmp_path):
    fixture(tmp_path)
    with (tmp_path / 'lock.scores.json').open('a') as f: f.write(' ')
    with pytest.raises(ValueError, match='hash differs'): MODULE.compare(tmp_path, 100)


def test_rejects_nonofficial_aggregation(tmp_path):
    fixture(tmp_path)
    path = tmp_path / 'lock.scores.json'
    score = json.loads(path.read_text()); score['aggregate']['official']['slu_f1'] = .5
    path.write_text(json.dumps(score))
    summary = json.loads((tmp_path / 'complete.json').read_text())
    summary['columns']['lock']['scores_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / 'complete.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='micro aggregation'): MODULE.compare(tmp_path, 100)
