"""Archived producer asset layer, independent of today's resolver environment."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from test_producer_workspace_bridge import producer, write
from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import (
    CheckpointCompatibility, CheckpointCompatibilityError, STUDY_PREFIX,
    authenticate_git_capture, authenticated_old_config,
)


def bind_producer_assets(doc, tmp_path, assets, *, run_id='run', attempt_id='old'):
    """Attach producer-shaped captures to either a Git-only or workspace bridge."""
    evidence = doc['evidence']
    workspace = evidence.get('producer_workspace')
    if workspace:
        launch = json.loads(Path(workspace['launch']['path']).read_bytes())
        container = json.loads(Path(workspace['container']['path']).read_bytes())[0]
        source = workspace['source_manifest']
    else:
        files, _ = authenticate_git_capture(doc['predecessor_commit'],
            *(evidence[k]['path'] for k in ('commit_object', 'tree_listing', 'source_archive')))
        source = write(tmp_path / 'producer-source.json', {
            'schema': 'saea-release-source-v1', 'commit': doc['predecessor_commit'],
            'files': {p: hashlib.sha256(blob).hexdigest() for p, blob in files.items()}})
        launch = {'schema': 'saea-configured-job-launch-v1', 'expected_commit': doc['predecessor_commit'],
            'repo_path': '/opt/saea-release/' + STUDY_PREFIX.rstrip('/'), 'image': 'sha256:' + 'a' * 64,
            'source_identity': {'mode': 'mounted-release-v1', 'manifest_sha256': source['sha256']},
            'environment': {'SAEA_RELEASE_MANIFEST': '/opt/saea-release/source-manifest.json',
                'SAEA_RELEASE_MANIFEST_SHA256': source['sha256'], 'SAEA_OBSERVABILITY_RUN_ID': run_id,
                'SAEA_OBSERVABILITY_ATTEMPT_ID': attempt_id},
            'mounts': [{'source': '/host/release', 'target': '/opt/saea-release', 'readonly': True}]}
        container = {'Config': {'Image': launch['image'], 'Labels': {}}, 'Mounts': [
            {'Source': '/host/release', 'Destination': '/opt/saea-release', 'RW': False, 'Type': 'bind'}]}
    root = tmp_path / 'assets-capture'
    inventory = {}
    for name, blob in assets.items():
        original = '/producer/assets/' + name
        target = root / original.lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(blob)
        inventory[name] = {'path': original, 'sha256': hashlib.sha256(blob).hexdigest()}
    manifest = write(tmp_path / 'producer-assets.json', {'schema': 'saea-experiment-assets-v1', 'files': inventory})
    launch['environment'].update(SAEA_EXPERIMENT_MANIFEST='/producer/assets/assets.json',
                                 SAEA_EXPERIMENT_MANIFEST_SHA256=manifest['sha256'])
    launch['mounts'].append({'source': '/host/assets', 'target': '/producer/assets', 'readonly': True})
    container['Mounts'].append({'Source': '/host/assets', 'Destination': '/producer/assets', 'RW': False, 'Type': 'bind'})
    external = {'schema': 'saea-producer-assets-evidence-v1', 'root': str(root),
                'manifest': manifest, 'source_manifest': source}
    evidence['producer_assets'] = external
    seal_launch(doc, tmp_path, launch, container)
    return doc


def seal_launch(doc, tmp_path, launch, container):
    container['Config']['Env'] = [k + '=' + v for k, v in launch['environment'].items()]
    container['Config']['Labels']['saea.launch.config'] = hashlib.sha256(
        json.dumps(launch, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    pins = {'launch': write(tmp_path / 'producer-launch.json', launch),
            'container': write(tmp_path / 'producer-container.json', [container])}
    for name in ('producer_workspace', 'producer_assets'):
        if name in doc['evidence']:
            doc['evidence'][name].update(copy.deepcopy(pins))


@pytest.fixture
def layered(producer, tmp_path):
    doc, header, workspace = producer
    dataset = 'configs/dataset/' + doc['predecessor_fragments']['dataset'] + '.json'
    assets = {dataset: b'{"dataset":"external"}',
              'docs/receipts/runtime.json': b'{"producer":"runtime receipt"}',
              'configs/baseline/' + doc['predecessor_fragments']['baseline'] + '.json': b'{"arm":"external"}'}
    bind_producer_assets(doc, tmp_path, assets)
    return doc, header, workspace, assets


def test_three_layers_and_runtime_receipt(layered, monkeypatch):
    doc, header, workspace, assets = layered
    # A current malicious/missing resolver must not participate in recovery.
    monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST', '/missing/current-assets.json')
    monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE', '/missing/current-workspace')
    bridge = CheckpointCompatibility.from_document(doc)
    files = bridge.old_source(run_id='run', predecessor_attempt_id='old')
    _, restored = authenticated_old_config(files, bridge.predecessor_fragments, header, '/old')
    assert restored.config_hash == header['config_hash']
    for kind in ('dataset', 'baseline'):
        rel = 'configs/' + kind + '/' + doc['predecessor_fragments'][kind] + '.json'
        assert files[STUDY_PREFIX + rel] == (workspace / rel).read_bytes()
        assert files[STUDY_PREFIX + rel] != assets[rel]
    assert files[STUDY_PREFIX + 'docs/receipts/runtime.json'] == assets['docs/receipts/runtime.json']
    del doc['evidence']['producer_workspace']
    files = CheckpointCompatibility.from_document(doc).old_source()
    for name, blob in assets.items():
        assert files[STUDY_PREFIX + name] == blob  # external overrides Git dataset
    assert files[STUDY_PREFIX + 'src/speech_aware_evidence_acquisition/core/flight/runtime.py'] == b'# frozen Git implementation\n'


@pytest.mark.parametrize('fault', ['bytes', 'missing', 'symlink', 'parent-symlink', 'root-symlink',
    'manifest-pin', 'manifest-bytes', 'environment-path', 'environment-sha', 'wrong-path',
    'traversal', 'src', 'asset-path-traversal', 'rw', 'mount-source', 'shadow',
    'launch-shadow', 'manifest-unmounted', 'layer-producer'])
def test_external_layer_rejects_tampering(layered, tmp_path, fault):
    doc, _, _, _ = layered
    external = doc['evidence']['producer_assets']
    root = Path(external['root'])
    manifest = json.loads(Path(external['manifest']['path']).read_bytes())
    receipt = manifest['files']['docs/receipts/runtime.json']
    path = root / receipt['path'].lstrip('/')
    launch = json.loads(Path(external['launch']['path']).read_bytes())
    container = json.loads(Path(external['container']['path']).read_bytes())[0]
    if fault == 'bytes': path.write_bytes(b'changed')
    elif fault == 'missing': path.unlink()
    elif fault == 'symlink':
        other = tmp_path / 'outside.json'; other.write_bytes(path.read_bytes())
        path.unlink(); path.symlink_to(other)
    elif fault == 'parent-symlink':
        moved = tmp_path / 'receipts'; path.parent.rename(moved); path.parent.symlink_to(moved)
    elif fault == 'root-symlink':
        alias = tmp_path / 'alias'; alias.symlink_to(root); external['root'] = str(alias)
    elif fault == 'manifest-pin': external['manifest']['sha256'] = '0' * 64
    elif fault == 'manifest-bytes': Path(external['manifest']['path']).write_bytes(b'{}')
    elif fault in ('wrong-path', 'traversal', 'src', 'asset-path-traversal'):
        if fault == 'wrong-path': receipt['path'] = '/producer/assets/other/runtime.json'
        elif fault == 'asset-path-traversal': receipt['path'] = '/producer/assets/../runtime.json'
        else: manifest['files']['configs/../src/evil.py' if fault == 'traversal' else 'src/evil.py'] = receipt
        external['manifest'] = write(Path(external['manifest']['path']), manifest)
        launch['environment']['SAEA_EXPERIMENT_MANIFEST_SHA256'] = external['manifest']['sha256']
        seal_launch(doc, tmp_path, launch, container)
    elif fault in ('rw', 'mount-source', 'shadow', 'launch-shadow', 'manifest-unmounted'):
        if fault == 'rw': container['Mounts'][-1]['RW'] = True
        elif fault == 'mount-source': container['Mounts'][-1]['Source'] = '/wrong'
        elif fault in ('shadow', 'launch-shadow'):
            shadow = {'Source': '/wrong', 'Destination': '/producer/assets/docs', 'RW': False, 'Type': 'bind'}
            container['Mounts'].append(shadow)
            if fault == 'launch-shadow':
                launch['mounts'].append({'source': '/wrong', 'target': shadow['Destination'], 'readonly': True})
        else: launch['environment']['SAEA_EXPERIMENT_MANIFEST'] = '/unmounted/assets.json'
        seal_launch(doc, tmp_path, launch, container)
    elif fault == 'layer-producer':
        container['Id'] = 'different-producer'
        external['container'] = write(tmp_path / 'other.json', [container])
    else:
        key = 'SAEA_EXPERIMENT_MANIFEST' + ('_SHA256' if fault == 'environment-sha' else '')
        container['Config']['Env'] = [s if not s.startswith(key + '=') else key + '=wrong' for s in container['Config']['Env']]
        pin = write(tmp_path / 'env-container.json', [container])
        for name in ('producer_workspace', 'producer_assets'): doc['evidence'][name]['container'] = pin
    with pytest.raises(CheckpointCompatibilityError):
        CheckpointCompatibility.from_document(doc).old_source(run_id='run', predecessor_attempt_id='old')


def test_unlisted_archive_file_is_never_loaded(layered):
    doc, _, _, _ = layered
    root = Path(doc['evidence']['producer_assets']['root'])
    write(root / 'producer/assets/docs/unlisted.json', {'untrusted': True})
    files = CheckpointCompatibility.from_document(doc).old_source()
    assert STUDY_PREFIX + 'docs/unlisted.json' not in files
