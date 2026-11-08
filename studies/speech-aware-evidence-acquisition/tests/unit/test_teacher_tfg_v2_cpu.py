import importlib.util
from pathlib import Path
import sys
import numpy as np

scripts = Path(__file__).resolve().parents[2] / 'scripts' / 'analysis'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('tfg', scripts / 'teacher_tfg_v2_cpu.py')
tfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(tfg)


def test_eligibility_negative_bias_and_zero_variance():
    raw = np.array([[.1, .2, -.1, .05]])
    cols = [(np.array([0, 1, 2]), 3), (np.array([3]), 1)]
    score, valid, _ = tfg.aggregate(raw, cols, np.array([.5, .5, -5, .5]))
    assert valid.tolist() == [[True, True]]
    assert score[0, 0] < 0 and score[0, 1] < 0
    assert set(tfg.orders(score, valid)[0]) == {0, 1}
    assert np.array_equal(tfg.zscore(np.ones(4)), np.zeros(4))


def test_swap_only_top_pair_and_exact_tie():
    parent = [[0, 1, 2, 3], [1, 0, 2, 3]]
    scores = np.array([[.51, .5, .4, .3], [.4, .41, .3, .2]])
    med = np.array([[.1, .2, 0, 0], [.2, .2, 0, 0]])
    result = tfg.swaps(parent, scores, med, .02)
    assert result == [[1, 0, 2, 3], [1, 0, 2, 3]]
    assert all(set(x[:3]) == set(y[:3]) for x, y in zip(parent, result))
