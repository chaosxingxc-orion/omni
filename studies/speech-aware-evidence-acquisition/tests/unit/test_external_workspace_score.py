"""Authenticated workspace scoring; no model or network requests."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from speech_aware_evidence_acquisition.core.bundle import BundleError
from speech_aware_evidence_acquisition.core.bundle.artifacts import VerifiedBundle
from speech_aware_evidence_acquisition.core.bundle.delivery import assert_raw_parent_scoreable
from speech_aware_evidence_acquisition.core.config import compose


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _world(tmp_path, monkeypatch, *, required=False):
    repo = tmp_path / 'repo'
    workspace = tmp_path / 'workspace'
    repo.mkdir()
    _git(repo, 'init', '-q')
    _git(repo, 'config', 'user.email', 'test@example.org')
    _git(repo, 'config', 'user.name', 'test')
    (repo / 'README').write_text('source')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-qm', 'source')
    commit = _git(repo, 'rev-parse', 'HEAD')
    docs = {'model': {'model_key': 'model'}, 'dataset': {'dataset_key': 'dataset'},
            'baseline': {'baseline_key': 'baseline'}, 'experiment': {'experiment_id': 'experiment'}}
    if required:
        docs['experiment'].update({'cells': [{'cell_id': 'parent', 'model_fragment': 'm',
            'dataset_fragment': 'd', 'baseline_fragment': 'b'}],
            'delivery_derivatives': [{'cell_id': 'delivery', 'parent_cell_id': 'parent',
                'response_postprocessor': 'x', 'response_postprocessor_params': {}, 'source_selector': {}}]})
    names = {'model':'m', 'dataset':'d', 'baseline':'b', 'experiment':'e'}
    files = {}
    for kind, name in names.items():
        rel = f'configs/{kind}/{name}.json'
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(docs[kind]).encode()
        path.write_bytes(raw)
        files[rel] = _sha(raw)
    inventory = {'schema': 'saea-experiment-workspace-v1', 'source_commit': commit,
                 'source_manifest_sha256': None, 'files': files}
    raw = json.dumps(inventory, sort_keys=True).encode()
    inv = workspace / 'manifest.json'
    inv.write_bytes(raw)
    monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE', str(workspace))
    monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_MANIFEST', str(inv))
    monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_SHA256', _sha(raw))
    config = compose(repo, **names)
    bundle = VerifiedBundle(tmp_path/'manifest.json', {'schema':'saea-run-manifest-v1',
        'study_commit':commit, 'config':{'fragments':names, 'config_hash':config.config_hash}},
        tmp_path/'outputs.jsonl', (), {}, frozenset(), 'OPENED')
    return repo, workspace, inv, bundle


def test_workspace_no_delivery_allows_scoring(tmp_path, monkeypatch):
    repo, _, _, bundle = _world(tmp_path, monkeypatch)
    assert_raw_parent_scoreable(bundle, repo)


def test_workspace_required_delivery_is_not_skipped(tmp_path, monkeypatch):
    repo, _, _, bundle = _world(tmp_path, monkeypatch, required=True)
    with pytest.raises(BundleError, match='requires declared delivery'):
        assert_raw_parent_scoreable(bundle, repo)


@pytest.mark.parametrize('damage', ['file', 'missing', 'inventory', 'escape', 'missing_pin', 'config_hash', 'commit'])
def test_workspace_provenance_fail_closed(tmp_path, monkeypatch, damage):
    repo, workspace, inv, bundle = _world(tmp_path, monkeypatch)
    path = workspace/'configs/experiment/e.json'
    if damage == 'file': path.write_text('{}')
    elif damage == 'missing': path.unlink()
    elif damage == 'inventory': inv.write_text('{}')
    elif damage == 'escape':
        out = tmp_path/'outside'; out.write_bytes(path.read_bytes()); path.unlink(); path.symlink_to(out)
    elif damage == 'missing_pin': monkeypatch.delenv('SAEA_EXPERIMENT_WORKSPACE_SHA256')
    elif damage == 'config_hash': bundle.manifest['config']['config_hash'] = '0'*64
    else: bundle.manifest['study_commit'] = '0'*40
    with pytest.raises((BundleError, ValueError, OSError)):
        assert_raw_parent_scoreable(bundle, repo)


def test_git_historical_fragment_remains_authoritative(tmp_path, monkeypatch):
    repo, workspace, inv, bundle = _world(tmp_path, monkeypatch)
    for key in ('SAEA_EXPERIMENT_WORKSPACE','SAEA_EXPERIMENT_WORKSPACE_MANIFEST','SAEA_EXPERIMENT_WORKSPACE_SHA256'):
        monkeypatch.delenv(key)
    path=repo/'configs/experiment/e.json'; path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'cells':[{'cell_id':'p','model_fragment':'m','dataset_fragment':'d','baseline_fragment':'b'}],
        'delivery_derivatives':[{'cell_id':'d','parent_cell_id':'p','response_postprocessor':'x','response_postprocessor_params':{},'source_selector':{}}]}))
    _git(repo,'add','.');_git(repo,'commit','-qm','delivery')
    bundle.manifest['study_commit']=_git(repo,'rev-parse','HEAD')
    path.write_text('{}');_git(repo,'add','.');_git(repo,'commit','-qm','later')
    with pytest.raises(BundleError, match='requires declared delivery'):
        assert_raw_parent_scoreable(bundle, repo)


def test_mounted_release_without_git_uses_verified_release(tmp_path, monkeypatch):
    repo, workspace, inv, bundle = _world(tmp_path, monkeypatch)
    # The production release verifier is exercised in fixed-image tests; this
    # isolates the no-Git branch and verifies the branch never invokes Git.
    monkeypatch.setenv('SAEA_RELEASE_MANIFEST', '/opt/saea-release/source-manifest.json')
    monkeypatch.setenv('SAEA_RELEASE_MANIFEST_SHA256', 'a'*64)
    doc=json.loads(inv.read_bytes());doc['source_manifest_sha256']='a'*64
    raw=json.dumps(doc,sort_keys=True).encode();inv.write_bytes(raw)
    monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_SHA256',_sha(raw))
    monkeypatch.setattr('speech_aware_evidence_acquisition.core.experiment_assets.source_commit',lambda root: bundle.manifest['study_commit'])
    monkeypatch.setattr('speech_aware_evidence_acquisition.core.bundle.git_commit.require_clean_load_bearing_surface',lambda root:(bundle.manifest['study_commit'],[]))
    monkeypatch.setattr('speech_aware_evidence_acquisition.core.bundle.git_commit._git',lambda *a: pytest.fail('Git must not be used'))
    assert_raw_parent_scoreable(bundle,repo)


def test_external_asset_manifest_not_workspace(tmp_path, monkeypatch):
    repo, workspace, inv, bundle = _world(tmp_path, monkeypatch)
    files=json.loads(inv.read_bytes())['files']
    manifest=tmp_path/'assets.json'
    document={'schema':'saea-experiment-assets-v1','files':{k:{'path':str(workspace/k),'sha256':v} for k,v in files.items()}}
    raw=json.dumps(document).encode();manifest.write_bytes(raw)
    for key in ('SAEA_EXPERIMENT_WORKSPACE','SAEA_EXPERIMENT_WORKSPACE_MANIFEST','SAEA_EXPERIMENT_WORKSPACE_SHA256'):monkeypatch.delenv(key)
    monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST',str(manifest));monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST_SHA256',_sha(raw))
    assert_raw_parent_scoreable(bundle,repo)
    (workspace/'configs/experiment/e.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):assert_raw_parent_scoreable(bundle,repo)
