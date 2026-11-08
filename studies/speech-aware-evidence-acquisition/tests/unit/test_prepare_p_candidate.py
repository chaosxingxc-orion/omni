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


def test_configured_ordered_supply_accepts_dynamic_window_and_checks_pins(tmp_path):
    parent = tmp_path / 'parent'
    (parent / 'workspace').mkdir(parents=True)
    (parent / 'workspace/job-config.private.json').write_text(json.dumps({'sample_ids': ['s']}))
    cards = ['intent-boundaries::intent-boundary:' + str(i) for i in range(8)]
    payload = {key: {'perception': {'s': cards}} for key in ('facets', 'values')}
    path = tmp_path / 'supply.json'
    path.write_text(json.dumps(payload))
    bank = tmp_path / 'bank.json'; bank.write_text('{}')
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{}')
    files = [{'path': str(p), 'name': p.name, 'sha256': m.sha(p)} for p in (path, bank, manifest)]
    c = {'ordered_supply': {'files': files, 'supply': path.name}, 'ranking_window': {'input_k': 25, 'output_k': 3, 'max_tokens': 128, 'manifest': manifest.name, 'bank': bank.name, 'fewshots_per_label': 3}}
    m.validate_ordered_supply(c, parent)
    c['ranking_window']['input_k'] = 5
    with pytest.raises(ValueError, match='invalid supply row'):
        m.validate_ordered_supply(c, parent)
    c['ranking_window']['input_k'] = 25
    path.write_text('{}')
    with pytest.raises(ValueError, match='pin mismatch'):
        m.validate_ordered_supply(c, parent)


def test_map_binding_argv_replaces_exact_serving_and_profile():
    launcher = {'mounts': [{'target': '/data/serving', 'source': '/old'}], 'environment': {'SAEA_DEPLOYMENT_PROFILE': 'old'}}
    binding = {'serving_root': '/new/serving', 'runtime_binding': {'profile': 'new'}}
    result = m.map_binding_argv(['/old:/data/serving:ro', 'SAEA_DEPLOYMENT_PROFILE=old'], launcher, binding)
    assert result == ['/new/serving:/data/serving:ro', 'SAEA_DEPLOYMENT_PROFILE=new']
    assert m.map_binding_argv(['/other-parent-copy:/data/serving:ro'], launcher, binding) == ['/new/serving:/data/serving:ro']
    with pytest.raises(ValueError, match='serving mount'):
        m.map_binding_argv(['SAEA_DEPLOYMENT_PROFILE=old'], launcher, binding)


def test_default_argv_unchanged():
    argv = ['a', 'b']
    assert m.map_binding_argv(argv, {}, None) is argv


def test_serving_mount_exact_match_and_readonly():
    launcher = {'mounts': [{'target': '/data/serving', 'source': '/old'}], 'environment': {'SAEA_DEPLOYMENT_PROFILE': 'old'}}
    binding = {'serving_root': '/new', 'runtime_binding': {'profile': 'new'}}
    for mount in ('/old:/data/serving:rw', 'relative:/data/serving:ro', '/old:/data/serving:ro:extra'):
        with pytest.raises(ValueError, match='serving mount'):
            m.map_binding_argv([mount], launcher, binding)
    assert m.map_binding_argv(['/old:/data/serving:ro', 'x:/data/serving-extra:ro'], launcher, binding)[1] == 'x:/data/serving-extra:ro'
