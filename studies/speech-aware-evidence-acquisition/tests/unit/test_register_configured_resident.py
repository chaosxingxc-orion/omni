import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('registration', ROOT/'scripts/environment/register_configured_resident.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def config(tmp_path):
    observation = tmp_path/'observations.json'; observation.write_text('{}')
    value = {'schema': m.SCHEMA, 'repo': str(tmp_path/'repo'), 'data': str(tmp_path/'data'),
             'model_fragment': 'model', 'session_id': 'session',
             'observations': {'path': str(observation), 'sha256': hashlib.sha256(observation.read_bytes()).hexdigest()}}
    return value


def test_registration_delegates_exact_pinned_inputs(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core import config as cfg
    from speech_aware_evidence_acquisition.core.session import vllm_resident
    value = config(tmp_path); path = tmp_path/'config.json'; path.write_text(json.dumps(value))
    seen = {}
    monkeypatch.setattr(contracts, 'assert_execution_environment', lambda: None)
    monkeypatch.setattr(cfg, 'read_fragment', lambda *args: {'transport': 'vllm'})
    monkeypatch.setattr(cfg, 'resolve_runtime_receipt_path', lambda *args: tmp_path/'receipt.json')
    def register(**kwargs):
        seen.update(kwargs)
        return ['one']
    monkeypatch.setattr(vllm_resident, 'register_deployment', register)
    assert m.execute(path) == {'session_id': 'session', 'files': 1, 'new_model_calls': 0}
    assert seen['model_fragment'] == 'model' and seen['instances'] == {}
    assert seen['data_root'] == tmp_path/'data'


@pytest.mark.parametrize('fault', ['hash', 'identity', 'relative', 'extra'])
def test_invalid_configuration_rejected_before_registration(tmp_path, fault):
    value = config(tmp_path)
    if fault == 'hash': value['observations']['sha256'] = '0'*64
    elif fault == 'identity': value['session_id'] = '../escape'
    elif fault == 'relative': value['repo'] = 'relative'
    else: value['extra'] = True
    path = tmp_path/'config.json'; path.write_text(json.dumps(value))
    with pytest.raises(ValueError): m.load(path)
