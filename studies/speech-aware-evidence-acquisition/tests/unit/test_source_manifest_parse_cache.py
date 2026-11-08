import concurrent.futures
import hashlib
import json

import pytest

from speech_aware_evidence_acquisition.core import experiment_assets as assets


def pin(monkeypatch, path, raw):
    path.write_bytes(raw)
    monkeypatch.setenv('SAEA_RELEASE_MANIFEST', str(path))
    monkeypatch.setenv('SAEA_RELEASE_MANIFEST_SHA256', hashlib.sha256(raw).hexdigest())


def test_source_content_pin_deletion_and_errors(tmp_path, monkeypatch):
    path = tmp_path / 'source.json'
    pin(monkeypatch, path, b'{"commit":"one"}')
    assert assets.source_commit(tmp_path) == 'one'
    assert assets.source_commit(tmp_path) == 'one'
    path.write_bytes(b'{"commit":"two"}')
    with pytest.raises(ValueError, match='release source inventory differs'):
        assets.source_commit(tmp_path)
    pin(monkeypatch, path, b'{"commit":"two"}')
    assert assets.source_commit(tmp_path) == 'two'
    pin(monkeypatch, path, b'{')
    for _ in range(2):
        with pytest.raises(json.JSONDecodeError):
            assets.source_commit(tmp_path)
    pin(monkeypatch, path, b'{}')
    with pytest.raises(KeyError, match='commit'):
        assets.source_commit(tmp_path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        assets.source_commit(tmp_path)


def test_concurrent_and_mutable_commit_semantics(tmp_path, monkeypatch):
    path = tmp_path / 'source.json'
    pin(monkeypatch, path, b'{"commit":"abc"}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(lambda _: assets.source_commit(tmp_path), range(128))) == ['abc'] * 128
    pin(monkeypatch, path, b'{"commit":[1]}')
    first = assets.source_commit(tmp_path)
    first.append(2)
    assert assets.source_commit(tmp_path) == [1]
    assert assets._source_manifest_commit.cache_info().maxsize == 4


def test_workspace_asset_checks_still_run(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'; repo.mkdir()
    workspace = tmp_path / 'workspace'; (workspace / 'configs').mkdir(parents=True)
    asset = workspace / 'configs' / 'x.json'; asset.write_bytes(b'{}')
    source = tmp_path / 'source.json'; source_raw = b'{"commit":"abc"}'
    pin(monkeypatch, source, source_raw)
    source_sha = hashlib.sha256(source_raw).hexdigest()
    inventory = {'schema':'saea-experiment-workspace-v1', 'source_commit':'abc',
                 'source_manifest_sha256':source_sha, 'files':{'configs/x.json':hashlib.sha256(b'{}').hexdigest()}}
    raw = json.dumps(inventory).encode(); manifest = tmp_path / 'workspace.json'; manifest.write_bytes(raw)
    monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE', str(workspace))
    monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_MANIFEST', str(manifest))
    monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_SHA256', hashlib.sha256(raw).hexdigest())
    assert assets.resolve_asset(repo, 'configs/x.json') == asset
    asset.write_bytes(b'changed')
    with pytest.raises(ValueError, match='workspace asset absent'):
        assets.resolve_asset(repo, 'configs/x.json')
    manifest.write_bytes(b'changed')
    with pytest.raises(ValueError, match='workspace inventory digest differs'):
        assets.resolve_asset(repo, 'configs/x.json')
