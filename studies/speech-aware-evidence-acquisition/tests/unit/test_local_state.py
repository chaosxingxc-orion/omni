"""Prevent state migration from bypassing global ownership or losing evidence."""
import fcntl
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('local_state_prepare',
    Path(__file__).resolve().parents[2]/'scripts/environment/local_state/prepare.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


@pytest.fixture
def state(tmp_path):
    data, base = tmp_path/'shared', tmp_path/'local'
    (data/'locks').mkdir(parents=True)
    (data/'locks/generation.guard').touch()
    (data/'locks/flight.lock').write_text('{"old_worker":true}')
    (data/'runs/source').mkdir(parents=True)
    (data/'runs/source/response.json').write_text('{"known_response":1}')
    base.mkdir()
    return data, base


def stage(state, **kwargs):
    data, base = state
    return m.stage(data, base/'reboot-v1', ['runs/source'],
                   m.sha(data/'locks/flight.lock'), base=base,
                   fs_type=kwargs.pop('fs_type', lambda _: 'ext4'), **kwargs)


def test_snapshot_preserves_bytes_and_fences_and_uses_same_global_guard(state):
    data, base = state
    before = m.inventory(data)
    manifest = stage(state)
    assert m.inventory(data) == before
    assert manifest['experiment_authorized'] is False
    assert manifest['shared_archive_current'] is False
    local = base/'reboot-v1'
    assert (local/'state/locks/flight.lock').read_bytes() == (data/'locks/flight.lock').read_bytes()
    args = m.mount_arguments(local)
    assert args[1].endswith(',readonly')
    assert args[-1] == f'type=bind,src={data}/locks/generation.guard,dst={data}/locks/generation.guard'
    with pytest.raises(ValueError, match='fresh'):
        stage(state)


@pytest.mark.parametrize('directory', ['locks', 'models/a', '../runs/a', '/runs/a', 'runs'])
def test_rejects_unsafe_or_overbroad_mounts(directory):
    with pytest.raises(ValueError):
        m.members([directory])


def test_rejects_overlapping_mounts():
    with pytest.raises(ValueError, match='nested'):
        m.members(['runs/a', 'runs/a/b'])


@pytest.mark.parametrize('kind', ['ceph', 'tmpfs', 'nfs', 'overlay'])
def test_rejects_remote_or_ephemeral_state_before_copy(state, kind):
    with pytest.raises(ValueError, match='persistent local'):
        stage(state, fs_type=lambda _: kind)
    assert not (state[1]/'reboot-v1').exists()


def test_active_global_guard_blocks_copy(state):
    with (state[0]/'locks/generation.guard').open('r+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            stage(state)
    assert not (state[1]/'reboot-v1').exists()


def test_symlink_input_rejected(state):
    (state[0]/'runs/source/link').symlink_to('/etc/passwd')
    with pytest.raises(ValueError, match='symlinks'):
        stage(state)
    assert not (state[1]/'reboot-v1').exists()


def test_guard_replacement_and_changed_fence_refuse_mount(state):
    data, base = state
    stage(state)
    guard = data/'locks/generation.guard'
    guard.rename(data/'locks/old.guard')
    guard.touch()
    with pytest.raises(ValueError, match='inode'):
        m.mount_arguments(base/'reboot-v1')
    guard.unlink()
    (data/'locks/old.guard').rename(guard)
    (data/'locks/flight.lock').write_text('new owner')
    with pytest.raises(ValueError, match='fence'):
        m.mount_arguments(base/'reboot-v1')


def test_local_state_remains_independent_after_checkpoint_update(state):
    data, base = state
    stage(state)
    local = base/'reboot-v1'
    (local/'state/runs/source/response.json').write_text('{"known_response":2}')
    assert json.loads((data/'runs/source/response.json').read_text())['known_response'] == 1
    assert m.mount_arguments(local)


def test_fresh_attempt_directory_created_only_locally(state):
    data, base = state
    m.stage(data, base/'fresh', ['runs/new'], m.sha(data/'locks/flight.lock'),
            base=base, fs_type=lambda _: 'ext4')
    assert (base/'fresh/state/runs/new').is_dir()
    assert not (data/'runs/new').exists()


def test_attempt_root_keeps_attempt_mutation_locks_local():
    assert m.members(['attempts']) == [Path('attempts')]
