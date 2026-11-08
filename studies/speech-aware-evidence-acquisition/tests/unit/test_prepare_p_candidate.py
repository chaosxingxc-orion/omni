import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / 'scripts/environment/prepare_p_candidate.py'
spec = importlib.util.spec_from_file_location('prepare_p_candidate', MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture_config(tmp_path):
    release = tmp_path / 'release'
    parent = tmp_path / 'parent'
    release.mkdir()
    parent.mkdir()
    template = release / 'studies/speech-aware-evidence-acquisition/configs/probes/t.txt'
    template.parent.mkdir(parents=True)
    template.write_text('template')
    digest = hashlib.sha256(template.read_bytes()).hexdigest()
    manifest = release / 'source-manifest.json'
    manifest.write_text(json.dumps({'commit': 'abc', 'files': {str(template.relative_to(release)): digest}}))
    c = {'schema': 'saea-p-candidate-preparation-v1', 'candidate': dict.fromkeys(('id', 'name', 'run_id', 'generation_name', 'admission_name'), 'candidate'), 'parent_root': str(parent), 'target_root': str(tmp_path / 'target'), 'release': {'root': str(release), 'manifest_sha256': m.sha(manifest), 'commit': 'abc'}, 'template': {'relpath': 'configs/probes/t.txt', 'sha256': digest}}
    return c, manifest, template


def test_bad_manifest_rejected_before_any_subprocess(tmp_path, monkeypatch):
    c, manifest, _ = fixture_config(tmp_path)
    manifest.write_text('{}')
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **k: pytest.fail('subprocess called'))
    with pytest.raises(ValueError, match='release manifest SHA mismatch'):
        m.prepare(c, False)
    assert not Path(c['target_root']).exists()


def test_missing_template_rejected_before_any_subprocess(tmp_path, monkeypatch):
    c, _, template = fixture_config(tmp_path)
    template.unlink()
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **k: pytest.fail('subprocess called'))
    with pytest.raises(ValueError, match='template missing'):
        m.prepare(c, False)
    assert not Path(c['target_root']).exists()


def test_template_must_be_in_manifest(tmp_path):
    c, manifest, _ = fixture_config(tmp_path)
    manifest.write_text(json.dumps({'commit': 'abc', 'files': {}}))
    c['release']['manifest_sha256'] = m.sha(manifest)
    with pytest.raises(ValueError, match='template manifest pin mismatch'):
        m.prepare(c, False)


