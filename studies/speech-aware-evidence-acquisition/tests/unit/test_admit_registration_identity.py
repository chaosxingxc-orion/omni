"""A stale registration must fail before creating a draft or admission receipt."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT/'scripts/analysis'))
spec = importlib.util.spec_from_file_location('admit_identity', ROOT/'scripts/analysis/admit_real_engine_candidate.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize('changed', ['config', 'protocol', 'missing', None])
def test_stale_registration_refused_before_draft(tmp_path, monkeypatch, changed):
    from speech_aware_evidence_acquisition.core import config
    from speech_aware_evidence_acquisition import contracts
    current = 'a'*64
    protocol = config.protocol_hash(current, 'test-arm', 's'*64)
    candidate = {'composed_config_hash': current, 'protocol_hash': protocol,
                 'split_identity_hash': 's'*64}
    if changed == 'config': candidate['composed_config_hash'] = 'b'*64
    elif changed == 'protocol': candidate['protocol_hash'] = 'c'*64
    elif changed == 'missing': candidate.pop('composed_config_hash')
    cell = {'model_fragment': 'model', 'dataset_fragment': 'dataset',
            'baseline_fragment': 'baseline', 'readiness': {'state': 'held'}}
    monkeypatch.setattr(config, 'compose', lambda *args: SimpleNamespace(
        config_hash=current, values={'arm': 'test-arm'}))
    monkeypatch.setattr(contracts, 'assert_execution_environment', lambda: None)
    monkeypatch.setattr(m, 'write', lambda *args: pytest.fail('wrote stale admission'))
    if changed is None:
        m.verify_registration_identity(tmp_path, 'candidate', candidate, cell)
        return
    package = tmp_path/'docs/readiness/candidate'
    package.mkdir(parents=True)
    (package/'candidate.json').write_text(json.dumps(candidate))
    exp = tmp_path/'configs/experiment/candidate.json'
    exp.parent.mkdir(parents=True)
    exp.write_text(json.dumps({'cells': [cell]}))
    with pytest.raises(ValueError, match='rebuild registration'):
        m.admit(tmp_path, tmp_path/'data', 'candidate')
    assert not (package/'root-admission-draft-v1.json').exists()
    assert not (tmp_path/'data').exists()


@pytest.mark.parametrize('label', [
    {'id': 'shared-rollout'},
    {'id': 'unregistered', 'note': 'Declared algorithm'},
    {'id': 'shared-rollout', 'note': 'x', 'native_learning_seal': False},
])
def test_invalid_reader_label_rejected_before_admission_side_effects(tmp_path, monkeypatch, label):
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core.stage3_algorithms import Stage3AlgorithmError
    monkeypatch.setattr(contracts, 'assert_execution_environment', lambda: None)
    monkeypatch.setattr(m, 'verify_registration_identity', lambda *a: None)
    monkeypatch.setattr(m, 'write', lambda *a: pytest.fail('wrote invalid admission'))
    package = tmp_path/'docs/readiness/candidate'
    package.mkdir(parents=True)
    (package/'candidate.json').write_text('{}')
    exp = tmp_path/'configs/experiment/candidate.json'
    exp.parent.mkdir(parents=True)
    exp.write_text(json.dumps({'cells': [{'readiness': {'state': 'held'}}],
                              'stage3_algorithm': label}))
    with pytest.raises(Stage3AlgorithmError):
        m.admit(tmp_path, tmp_path/'data', 'candidate')
    assert not (package/'root-admission-draft-v1.json').exists()
    assert not (tmp_path/'data').exists()
