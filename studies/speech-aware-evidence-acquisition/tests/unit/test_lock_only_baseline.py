import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('lock_derive', ROOT / 'scripts/analysis/derive_lock_only_baseline.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source():
    return json.loads((ROOT / 'configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json').read_bytes())


def test_two_contacts_preserve_actual_request_inputs():
    before = source()
    original = copy.deepcopy(before)
    result = module.derive(before)
    assert before == original
    assert result['policy'] == 'phase-native-module-v1-c2'
    assert 'sampling_variants' not in result['native_episode']['roles']['perception']
    restored = copy.deepcopy(result)
    restored['policy'] = before['policy']
    restored['native_episode']['consumption_module'] = before['native_episode']['consumption_module']
    restored['native_episode']['roles']['perception']['sampling_variants'] = before['native_episode']['roles']['perception']['sampling_variants']
    assert restored == before


@pytest.mark.parametrize('fault', ['arms', 'override', 'revisions', 'iterations'])
def test_refuse_unreviewed_source(fault):
    value = source()
    native = value['native_episode']
    if fault == 'arms': native['roles']['perception']['sampling_variants']['arms'].pop(0)
    if fault == 'override': native['roles']['perception']['sampling_variants']['arms'][-1]['solver_samples'] = {'n': 5}
    if fault == 'revisions': native['max_revisions'] = 1
    if fault == 'iterations': native['roles']['solver']['max_iterations'] = 2
    with pytest.raises(ValueError): module.derive(value)


def test_budget_candidates_change_only_one_node():
    lock = module.derive(source())
    model = {'reasoning_budget_tokens': 3072, 'answer_reserve_tokens': 1024,
             'enable_thinking': True, 'reasoning_content_guard': 'refuse-think-marker-in-content',
             'decoding_params': {'max_tokens': 4096}}
    candidates = module.budget_candidates(lock, model)
    assert len(candidates) == 7
    assert candidates.pop('control-3072') == lock
    for name, candidate in candidates.items():
        node, budget = name.split('-')
        policy = candidate.pop('thinking_policy')
        assert policy == {node: {'reasoning_budget_tokens': int(budget), 'answer_reserve_tokens': 1024}}
        assert candidate == lock
