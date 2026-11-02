import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("count", [100, 500])
def test_configuration_replaces_every_population_and_spend_identity(count):
    path = Path(__file__).parents[2]/'scripts/analysis/prepare_concurrency_candidate.py'
    spec = importlib.util.spec_from_file_location('real100_stage', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    old = {'cells': [{'run_id': 'old', 'attempt_id': 'old', 'resume_from': {'old': True},
                      'readiness': {'state': 'ready'}}], 'execution_plan': {}, 'stage3_measurement': {},
           'populations': [{'ids': ['wrong']}], 'population': {'count': 500}}
    base = {'prepared_phase_supply': {'old': True}, 'native_contact_fit': {'run_id': 'old', 'attempt_id': 'old'}}
    view = {'split_name': f'fixed{count}', 'split_identity_hash': 'a'*64, 'audio_seconds_total': 9.2,
            'items': [{'sample_id': f's{i}'} for i in range(count)]}
    name, run, dataset, candidate, exp = module.configuration(old, base, view, level=32, model='m', session='s')
    assert len(exp['populations'][0]['ids']) == count
    assert exp['population']['count'] == count
    assert exp['cells'][0]['dataset_fragment'] == dataset
    assert exp['cells'][0]['run_id'] == run and exp['cells'][0]['readiness']['state'] == 'held'
    assert 'resume_from' not in exp['cells'][0]
    assert candidate['native_contact_fit']['run_id'] == run+'-FIT'
    assert candidate['native_contact_fit']['contact_cap'] == 2*count
    assert 'prepared_phase_supply' not in candidate
    assert exp['execution_plan']['planned_speech_audio_seconds'] == 37
    assert exp['execution_plan']['planned_model_calls'] == 4*count
    assert exp['execution_plan']['planned_gpu_hours'] == (4.0 if count == 100 else 16.0)
    assert exp['stage3_measurement']['rows'] == count
    assert exp['stage3_measurement']['read_roster'] == f'fixed{count}'
    assert f'real{count}' in name and f'REAL{count}' in run
    assert f'real{count}' in exp['cells'][0]['readiness']['reason']
    expected_authority = ('2026-11-02-owner-real100-concurrency.md' if count == 100
                          else '2026-11-02-owner-real500-confirmation.md')
    assert module.owner_authority(count).endswith(expected_authority)
    assert (path.parents[2]/module.owner_authority(count)).is_file()
    assert old['population']['count'] == 500 and 'prepared_phase_supply' in base
    with pytest.raises(ValueError):
        module.configuration(old, base, view, level=True, model='m', session='s')
