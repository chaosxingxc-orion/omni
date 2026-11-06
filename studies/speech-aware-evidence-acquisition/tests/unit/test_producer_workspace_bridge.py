"""Producer-shaped external fragments plus real Git authentication, no contacts."""
import hashlib
import json
import subprocess

import pytest

from speech_aware_evidence_acquisition.core.config import compose, protocol_hash
from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import (
    CheckpointCompatibility, CheckpointCompatibilityError, IMPLEMENTATION,
    RELEASE_SCHEMA, STUDY_PREFIX, authenticated_old_config,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def producer(tmp_path):
    repo = tmp_path / 'git'
    study = repo / STUDY_PREFIX
    workspace = tmp_path / 'workspace'
    names = dict(model='bf16-thinking-a-tp4-ctx64k-v1',
                 dataset='slurp-speech-native-evo-fold-a-1080',
                 baseline='slurp-fold-a-slot-off-a64k-v1',
                 experiment='slurp-fold-a-slot-off-a64k-v1')
    fragments = dict(model={'model': 'frozen'}, dataset={'dataset': 'frozen'},
                     baseline={'arm': 'test-arm'}, experiment={'cells': [{'attempt_id': 'old'}]})
    for kind, value in fragments.items():
        write(workspace / 'configs' / kind / (names[kind] + '.json'), value)
    write(study / 'configs/dataset' / (names['dataset'] + '.json'), fragments['dataset'])
    for path in IMPLEMENTATION.values():
        target = study / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'# frozen Git implementation\n')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])
    git('init', '-q')
    git('add', '.')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')
    commit = git('rev-parse', 'HEAD').decode().strip()
    pins = {}
    for key, raw in [('commit_object', git('cat-file', 'commit', commit)),
                     ('tree_listing', git('ls-tree', '-r', '--full-tree', commit)),
                     ('source_archive', git('archive', commit))]:
        path = tmp_path / key
        path.write_bytes(raw)
        pins[key] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    source = write(tmp_path / 'source.json', {
        'schema': 'saea-release-source-v1', 'commit': commit,
        'files': {STUDY_PREFIX + p: hashlib.sha256((study / p).read_bytes()).hexdigest()
                  for p in IMPLEMENTATION.values()}})
    write(workspace / 'docs/readiness/protocol.json', {'asset': 'pinned'})
    inventory = {p.relative_to(workspace).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in workspace.rglob('*') if p.is_file()}
    manifest = write(tmp_path / 'manifest.json', {
        'schema': 'saea-experiment-workspace-v1', 'source_commit': commit,
        'source_manifest_sha256': source['sha256'], 'files': inventory})
    target = workspace / 'manifests' / (manifest['sha256'] + '.json')
    target.parent.mkdir(); target.write_bytes((tmp_path / 'manifest.json').read_bytes())
    env = {'SAEA_RELEASE_MANIFEST': '/opt/saea-release/source-manifest.json',
           'SAEA_RELEASE_MANIFEST_SHA256': source['sha256'],
           'SAEA_EXPERIMENT_WORKSPACE': '/producer/workspace',
           'SAEA_EXPERIMENT_WORKSPACE_MANIFEST': '/producer/workspace/manifests/' + manifest['sha256'] + '.json',
           'SAEA_EXPERIMENT_WORKSPACE_SHA256': manifest['sha256'],
           'SAEA_OBSERVABILITY_RUN_ID': 'run', 'SAEA_OBSERVABILITY_ATTEMPT_ID': 'old'}
    launch = {'schema': 'saea-configured-job-launch-v1', 'expected_commit': commit,
              'repo_path': '/opt/saea-release/' + STUDY_PREFIX.rstrip('/'),
              'image': 'sha256:' + 'a' * 64, 'environment': env,
              'source_identity': {'mode': 'mounted-release-v1', 'manifest_sha256': source['sha256']},
              'mounts': [{'source': '/producer/release', 'target': '/opt/saea-release', 'readonly': True},
                         {'source': '/producer/workspace', 'target': '/producer/workspace', 'readonly': True}]}
    container = {'Config': {'Image': launch['image'], 'Env': [k + '=' + v for k, v in env.items()],
                           'Labels': {'saea.launch.config': hashlib.sha256(json.dumps(launch, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}},
                 'Mounts': [{'Source': m['source'], 'Destination': m['target'], 'Type': 'bind', 'RW': False}
                            for m in launch['mounts']]}
    evidence = {'schema': 'saea-producer-workspace-evidence-v1', 'root': str(workspace),
                'manifest': manifest, 'source_manifest': source,
                'launch': write(tmp_path / 'launch.json', launch),
                'container': write(tmp_path / 'container.json', [container])}
    unused = write(tmp_path / 'unused.json', {})
    doc = {'schema': RELEASE_SCHEMA, 'predecessor_commit': commit, 'predecessor_fragments': names,
           'successor_attempt_id': 'new', 'change': 'image-release-operational-v1',
           'evidence': {**pins, 'release_manifest': unused, 'source_review': unused,
                        'reference_pins': unused, 'startup_data_root': '/old', 'producer_workspace': evidence}}
    old = compose(workspace, **names, data_root='/old')
    header = {'config_hash': old.config_hash, 'protocol_hash': protocol_hash(old.config_hash, 'test-arm', 'b' * 64),
              'split_identity_hash': 'b' * 64,
              'implementation': {k: hashlib.sha256((study / p).read_bytes()).hexdigest() for k, p in IMPLEMENTATION.items()}}
    return doc, header, workspace


def test_external_fragments_restore_authenticated_header(producer):
    doc, header, _ = producer
    bridge = CheckpointCompatibility.from_document(doc)
    files = bridge.old_source(run_id='run', predecessor_attempt_id='old')
    docs, composed = authenticated_old_config(files, bridge.predecessor_fragments, header, '/old')
    assert composed.config_hash == header['config_hash']
    assert docs['model']['model'] == 'frozen'
    assert STUDY_PREFIX + 'docs/readiness/protocol.json' in files
    bad = dict(header, implementation={})
    with pytest.raises(CheckpointCompatibilityError, match='implementation'):
        authenticated_old_config(files, bridge.predecessor_fragments, bad, '/old')


def test_git_only_is_unchanged(producer):
    doc, header, _ = producer
    del doc['evidence']['producer_workspace']
    bridge = CheckpointCompatibility.from_document(doc)
    files = bridge.old_source()
    assert all(STUDY_PREFIX + p in files for p in IMPLEMENTATION.values())
    with pytest.raises(CheckpointCompatibilityError, match='fragment missing'):
        authenticated_old_config(files, bridge.predecessor_fragments, header, '/old')


@pytest.mark.parametrize('fault', ['bytes', 'unlisted', 'traversal', 'source-overlay', 'symlink',
    'parent-symlink', 'wrong-source', 'launch-label', 'environment', 'rw', 'mount-source',
    'shadow', 'manifest-pin', 'source-bytes', 'attempt'])
def test_tampering_fails_closed(producer, tmp_path, fault):
    doc, _, workspace = producer
    evidence = doc['evidence']['producer_workspace']
    if fault in ('bytes', 'symlink', 'parent-symlink'):
        path = workspace / 'configs/model' / (doc['predecessor_fragments']['model'] + '.json')
        if fault == 'bytes':
            path.write_bytes(b'changed')
        elif fault == 'symlink':
            other = tmp_path / 'outside.json'; other.write_bytes(path.read_bytes())
            path.unlink(); path.symlink_to(other)
        else:
            directory = path.parent; moved = tmp_path / 'moved'; directory.rename(moved); directory.symlink_to(moved)
    elif fault == 'unlisted':
        write(workspace / 'configs/unlisted.json', {})
    elif fault in ('traversal', 'source-overlay', 'wrong-source'):
        manifest = json.loads((tmp_path / 'manifest.json').read_bytes())
        if fault == 'wrong-source': manifest['source_commit'] = '0' * 40
        else: manifest['files']['configs/../src/evil.py' if fault == 'traversal' else 'src/evil.py'] = 'a' * 64
        evidence['manifest'] = write(tmp_path / 'manifest.json', manifest)
        digest = evidence['manifest']['sha256']
        write(workspace / 'manifests' / (digest + '.json'), manifest)
        launch = json.loads((tmp_path / 'launch.json').read_bytes())
        launch['environment']['SAEA_EXPERIMENT_WORKSPACE_SHA256'] = digest
        launch['environment']['SAEA_EXPERIMENT_WORKSPACE_MANIFEST'] = '/producer/workspace/manifests/' + digest + '.json'
        evidence['launch'] = write(tmp_path / 'launch.json', launch)
        container = json.loads((tmp_path / 'container.json').read_bytes())[0]
        container['Config']['Env'] = [k + '=' + v for k, v in launch['environment'].items()]
        container['Config']['Labels']['saea.launch.config'] = hashlib.sha256(json.dumps(launch, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        evidence['container'] = write(tmp_path / 'container.json', [container])
    elif fault == 'source-bytes':
        (tmp_path / 'source.json').write_bytes(b'changed')
    elif fault == 'attempt':
        with pytest.raises(CheckpointCompatibilityError, match='run/attempt'):
            CheckpointCompatibility.from_document(doc).old_source(run_id='run', predecessor_attempt_id='other')
        return
    else:
        container = json.loads((tmp_path / 'container.json').read_bytes())[0]
        if fault == 'launch-label': container['Config']['Labels']['saea.launch.config'] = '0' * 64
        elif fault == 'environment': container['Config']['Env'][0] += '-changed'
        elif fault == 'rw': container['Mounts'][-1]['RW'] = True
        elif fault == 'mount-source': container['Mounts'][-1]['Source'] = '/wrong'
        elif fault == 'shadow': container['Mounts'].append({'Destination': '/producer/workspace/configs', 'Source': '/wrong', 'Type': 'bind', 'RW': False})
        elif fault == 'manifest-pin': evidence['manifest']['sha256'] = '0' * 64
        evidence['container'] = write(tmp_path / 'container.json', [container])
    with pytest.raises(CheckpointCompatibilityError):
        CheckpointCompatibility.from_document(doc).old_source(run_id='run', predecessor_attempt_id='old')


@pytest.mark.parametrize('field', ['config_hash', 'protocol_hash', 'implementation'])
def test_restored_header_must_still_match(producer, field):
    doc, header, _ = producer
    bridge = CheckpointCompatibility.from_document(doc)
    files = bridge.old_source(run_id='run', predecessor_attempt_id='old')
    header[field] = {} if field == 'implementation' else '0' * 64
    with pytest.raises(CheckpointCompatibilityError):
        authenticated_old_config(files, bridge.predecessor_fragments, header, '/old')
