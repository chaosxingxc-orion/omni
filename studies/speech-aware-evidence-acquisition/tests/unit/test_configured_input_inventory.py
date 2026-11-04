import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).parents[2] / 'scripts/environment'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('input_inventory', SCRIPTS / 'configured_input_inventory.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_reports_all_failures_without_leaking_environment(tmp_path, monkeypatch):
    good = tmp_path / 'good'; good.write_bytes(b'data')
    denied = tmp_path / 'denied'; denied.mkdir()
    config = tmp_path / 'config.json'; config.write_text('{}')
    monkeypatch.setattr(m.launcher, 'load_config', lambda _: {
        'container_name': 'inventory', 'environment': {'SECRET': 'must-not-appear'},
        'mounts': [{'source': str(denied), 'target': '/denied', 'readonly': True},
                   {'source': str(tmp_path / 'absent'), 'target': '/absent', 'readonly': True},
                   {'source': str(good), 'target': '/good', 'readonly': True}],
        'required_inputs': [{'path': str(good), 'sha256': '0' * 64}]})
    real = m.os.scandir
    def scandir(path):
        if path == denied:
            raise PermissionError(13, 'denied')
        return real(path)
    monkeypatch.setattr(m.os, 'scandir', scandir)
    result = m.inventory(config)
    assert result['blocked_checks'] == 3
    assert result['mounts'][0]['errno'] == 13
    assert result['mounts'][1]['error_type'] == 'FileNotFoundError'
    assert result['mounts'][2]['readable']
    assert result['required_inputs'][0]['identity_matches'] is False
    assert 'must-not-appear' not in json.dumps(result)
    assert result['readiness_granted'] is False


def test_pins_hash_full_content_and_directory_is_not_full_manifest(tmp_path):
    file = tmp_path / 'asset'; file.write_bytes(b'x' * (2 * 1024 * 1024) + b'end')
    assert m.probe(file, hashlib.sha256(file.read_bytes()).hexdigest())['identity_matches']
    directory = m.probe(tmp_path)
    assert directory['readable'] and 'descendants not verified' in directory['coverage']
    assert not m.probe(tmp_path, '0' * 64)['readable']


def test_cli_failure_preserves_existing_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'inventory', lambda _: {'blocked_checks': 2})
    out = tmp_path / 'report.json'
    args = ['--config', str(tmp_path / 'c'), '--output', str(out)]
    assert m.main(args) == 1
    first = out.read_bytes()
    with pytest.raises(FileExistsError):
        m.main(args)
    assert out.read_bytes() == first
