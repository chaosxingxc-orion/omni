"""Production receipt extensions remain byte-pinned and fail closed."""
import copy
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from _vllm_world import VllmWorld
from speech_aware_evidence_acquisition.contracts import GateClosed
from speech_aware_evidence_acquisition.contracts import gate_runtime_vllm as gate
from speech_aware_evidence_acquisition.core.session import SessionError
from speech_aware_evidence_acquisition.core.session import vllm_identity as identity
from speech_aware_evidence_acquisition.contracts.runtime_identity import runtime_receipt_identity


def validate(world, receipt):
    runtime_receipt_identity('vllm').validate_runtime_receipt(
        world.gate(receipt=receipt), world.lock, full_hash=True, diagnostics={})


def linked_world(tmp_path):
    world = VllmWorld(tmp_path)
    root = world.trees['vllm_ascend']
    (root / 'libalias.so').symlink_to('entrypoints/cli.py')
    receipt = world.receipt()
    for tree in receipt['source_trees']:
        if tree['package'] == 'vllm_ascend':
            tree.update(gate.source_tree_manifest(root))
    receipt['protocol_sha256'] = gate.protocol_sha256(receipt)
    return world, root, receipt


def test_pinned_internal_symlink_and_legacy_plain_trees_validate(tmp_path):
    world, root, receipt = linked_world(tmp_path)
    validate(world, receipt)
    tree = next(t for t in receipt['source_trees'] if 'symlinks' in t)
    link, = tree['symlinks']
    assert link['target_sha256'] == gate.source_tree_manifest(root)['symlinks'][0]['target_sha256']
    assert link['resolves_outside'] is False


@pytest.mark.parametrize('change', ['target-bytes', 'retarget', 'remove', 'add'])
def test_link_identity_changes_fail(tmp_path, change):
    world, root, receipt = linked_world(tmp_path)
    if change == 'target-bytes':
        (root / 'entrypoints/cli.py').write_text('changed')
    elif change == 'retarget':
        (root / 'libalias.so').unlink()
        (root / 'libalias.so').symlink_to('__init__.py')
    elif change == 'remove':
        (root / 'libalias.so').unlink()
    else:
        (root / 'extra-link.so').symlink_to('__init__.py')
    with pytest.raises(GateClosed):
        validate(world, receipt)


@pytest.mark.parametrize('target', ['missing.py', 'loop.so', '../outside.py'])
def test_unresolved_or_external_symlinks_refuse(tmp_path, target):
    root = tmp_path / 'source'
    root.mkdir()
    (root / '__init__.py').write_text('pinned')
    (tmp_path / 'outside.py').write_text('not pinned')
    (root / 'loop.so').symlink_to(target)
    with pytest.raises(GateClosed):
        gate.source_tree_manifest(root)


@pytest.mark.parametrize('mutation', ['extra', 'missing-hash', 'outside', 'duplicate', 'escape', 'negative-size', 'bad-device'])
def test_extended_receipt_shape_is_exact(tmp_path, mutation):
    world, _, receipt = linked_world(tmp_path)
    tree = next(t for t in receipt['source_trees'] if 'symlinks' in t)
    link = tree['symlinks'][0]
    if mutation == 'extra':
        link['unverified'] = True
    elif mutation == 'missing-hash':
        del link['target_sha256']
    elif mutation == 'outside':
        link['resolves_outside'] = True
    elif mutation == 'duplicate':
        tree['symlinks'].append(copy.deepcopy(link))
    elif mutation == 'escape':
        link['path'] = '../escape'
    elif mutation == 'negative-size':
        link['target_bytes'] = -1
    else:
        receipt['deployment']['instances'][0]['device_binding'] = 'ignore-device'
    with pytest.raises(GateClosed):
        gate.validate_vllm_receipt_document(receipt)


def test_device_binding_is_deployment_not_protocol(tmp_path):
    world = VllmWorld(tmp_path)
    receipt = world.receipt()
    original = receipt['protocol_sha256']
    for instance in receipt['deployment']['instances']:
        instance['device_binding'] = 'container-device-nodes'
    gate.validate_vllm_receipt_document(receipt)
    assert gate.protocol_sha256(receipt) == original


def test_container_device_binding_refuses_harness_spawn(tmp_path, monkeypatch):
    world = VllmWorld(tmp_path)
    receipt = world.receipt()
    instance = receipt['deployment']['instances'][0]
    instance['device_binding'] = 'container-device-nodes'
    monkeypatch.setattr(identity, 'assert_execution_environment', lambda: None)

    def unexpected_spawn(*args, **kwargs):
        pytest.fail('container-bound instance must not spawn a harness process')

    with pytest.raises(SessionError, match='must be attached'):
        identity.start_vllm_session(
            runtime_receipt=receipt, host=instance['host'], port=instance['port'],
            session_id='refused-container-spawn', popen=unexpected_spawn,
            extra_argv=receipt['protocol']['server_args'])


def device_world(tmp_path, monkeypatch, ids=(0, 1), filter_env=False):
    world = VllmWorld(tmp_path)
    receipt = world.receipt()
    receipt['deployment']['instances'][0]['device_binding'] = 'container-device-nodes'
    pid = 4312345
    world.spawn('tp2-a', pid=pid, receipt=receipt)
    env_path = world.proc_root / str(pid) / 'environ'
    if not filter_env:
        env_path.write_bytes(b'\0'.join(row for row in env_path.read_bytes().split(b'\0')
            if row and not row.startswith(gate.DEVICE_ENV_KEY.encode() + b'=')) + b'\0')
    devices = world.proc_root / str(pid) / 'root/dev'
    devices.mkdir(parents=True)
    for number in ids:
        (devices / ('davinci' + str(number))).touch()
    original = Path.lstat
    def character_node(path, *args, **kwargs):
        if path.parent == devices and path.name.startswith('davinci'):
            return SimpleNamespace(st_mode=stat.S_IFCHR | 0o600)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'lstat', character_node)
    return world, receipt, pid


def test_live_device_namespace_accepts_exact_physical_set(tmp_path, monkeypatch):
    world, receipt, pid = device_world(tmp_path, monkeypatch)
    identity.assert_vllm_session_alive(world.session('tp2-a', pid, receipt=receipt),
                                      receipt, None, proc_root=world.proc_root)


@pytest.mark.parametrize('ids', [(0,), (0, 1, 2), (2, 3)])
def test_live_device_namespace_rejects_missing_extra_or_foreign_cards(tmp_path, monkeypatch, ids):
    world, receipt, pid = device_world(tmp_path, monkeypatch, ids=ids)
    with pytest.raises(SessionError, match='physical device nodes'):
        identity.assert_vllm_session_alive(world.session('tp2-a', pid, receipt=receipt),
                                          receipt, None, proc_root=world.proc_root)


def test_device_namespace_rejects_added_logical_filter(tmp_path, monkeypatch):
    world, receipt, pid = device_world(tmp_path, monkeypatch, filter_env=True)
    with pytest.raises(SessionError, match='unexpectedly sets'):
        identity.assert_vllm_session_alive(world.session('tp2-a', pid, receipt=receipt),
                                          receipt, None, proc_root=world.proc_root)


@pytest.mark.parametrize('kind', ['regular', 'symlink', 'empty', 'missing'])
def test_device_namespace_noncharacter_or_unobservable_refused(tmp_path, kind):
    dev = tmp_path / '1/root/dev'
    if kind != 'missing':
        dev.mkdir(parents=True)
    if kind == 'regular':
        (dev / 'davinci0').touch()
    elif kind == 'symlink':
        (dev / 'davinci0').symlink_to('/dev/null')
    with pytest.raises(SessionError):
        identity.container_device_ids(tmp_path, 1)
