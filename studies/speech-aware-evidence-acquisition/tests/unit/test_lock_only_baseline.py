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
    assert restored['native_episode']['prompt_contact_limits'] == 'legacy-c17-lock-v1'
    del restored['native_episode']['prompt_contact_limits']
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


def test_legacy_prompt_limits_do_not_widen_execution_budget():
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config, prompt_remaining_contacts
    from test_native_episode_config import config, program
    values = config()
    for role in ('reviewer', 'planner'):
        values['native_episode']['roles'][role]['max_iterations'] = 0
    values['native_episode']['prompt_contact_limits'] = 'legacy-c17-lock-v1'
    parsed = load_native_episode_config(values, program(ceiling=2))
    assert parsed.max_contacts == 2
    assert prompt_remaining_contacts(parsed, 'perception', 2) == 17
    assert prompt_remaining_contacts(parsed, 'solver', 1) == 9
    with pytest.raises(ValueError): prompt_remaining_contacts(parsed, 'reviewer', 0)
    from dataclasses import replace
    for altered in (replace(parsed, max_contacts=17), replace(parsed, max_revisions=1),
                    replace(parsed, prompt_contact_limits='arbitrary')):
        with pytest.raises(ValueError): prompt_remaining_contacts(altered, 'perception', 2)
    default = replace(parsed, prompt_contact_limits=None)
    assert prompt_remaining_contacts(default, 'perception', 2) == 2
    assert prompt_remaining_contacts(default, 'solver', 1) == 1


def test_derived_candidate_passes_real_config_loader():
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    candidate = module.derive(source())
    parsed = load_native_episode_config({**candidate, 'carrier_lock_key': 'slurp'}, default_phase_program(candidate['policy']))
    assert parsed.max_contacts == 2
    assert parsed.prompt_contact_limits == 'legacy-c17-lock-v1'
