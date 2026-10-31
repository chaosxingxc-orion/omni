import importlib.util
from pathlib import Path
import pytest
spec = importlib.util.spec_from_file_location('candidate_runner', Path(__file__).resolve().parents[2] / 'scripts/analysis/run_real_engine_candidate.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_generation_completion_does_not_hide_failed_score():
    result = {'pending': [], 'completed': ['r'], 'states': {'r': {'state': 'failed', 'reason': 'official scoring failed'}}}
    with pytest.raises(RuntimeError, match='finalization failed'):
        m.assert_campaign_success(result, 'r')
    result['states']['r']['state'] = 'scored'
    m.assert_campaign_success(result, 'r')
