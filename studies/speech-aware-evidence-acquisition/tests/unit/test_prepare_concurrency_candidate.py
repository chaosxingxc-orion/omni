import importlib.util
from pathlib import Path

import pytest


def test_configuration_replaces_every_population_and_spend_identity():
    path = Path(__file__).parents[2]/'scripts/analysis/prepare_concurrency_candidate.py'
    spec = importlib.util.spec_from_file_location('real100_stage', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    old = {'cells': [{'run_id': 'old', 'attempt_id': 'old', 'resume_from': {'old': True},
                      'readiness': {'state': 'ready'}}], 'execution_plan': {}, 'stage3_measurement': {},
           'populations': [{'ids': ['wrong']}], 'population': {'count': 500}}
    base = {'prepared_phase_supply': {'old': True}, 'native_contact_fit': {'run_id': 'old', 'attempt_id': 'old'}}
    view = {'split_name': 'fixed100', 'split_identity_hash': 'a'*64, 'audio_seconds_total': 9.2,
            'items': [{'sample_id': f's{i}'} for i in range(100)]}
    name, run, dataset, candidate, exp = module.configuration(old, base, view, level=32, model='m', session='s')
    assert len(exp['populations'][0]['ids']) == 100
    assert exp['population']['count'] == 100
    assert exp['cells'][0]['dataset_fragment'] == dataset
    assert exp['cells'][0]['run_id'] == run and exp['cells'][0]['readiness']['state'] == 'held'
    assert 'resume_from' not in exp['cells'][0]
    assert candidate['native_contact_fit']['run_id'] == run+'-FIT'
    assert candidate['native_contact_fit']['contact_cap'] == 200
    assert 'prepared_phase_supply' not in candidate
    assert exp['execution_plan']['planned_speech_audio_seconds'] == 37
    assert old['population']['count'] == 500 and 'prepared_phase_supply' in base
    with pytest.raises(ValueError):
        module.configuration(old, base, view, level=True, model='m', session='s')
