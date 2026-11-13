from types import SimpleNamespace

import pytest

from tools.experiments.local_runtime_storage import check_paths, mount_for, require_training_storage


MOUNTS = ('1 0 8:1 / / rw - ext4 /dev/root rw\n'
          '2 1 0:2 / /shared rw - ceph mon:/ rw\n'
          '3 1 0:2 /hot /workspace/run rw - ceph mon:/ rw\n'
          '4 1 0:3 / /tmp rw - tmpfs tmpfs rw\n')


def capacity(path):
    return SimpleNamespace(f_bavail=100, f_frsize=1024**3, f_blocks=200,
                           f_files=1000, f_favail=900)


@pytest.mark.parametrize('path', ['/shared/new/run', '/workspace/run/checkpoint', '/tmp/run'])
def test_nonlocal_or_volatile_writes_rejected(path):
    with pytest.raises(ValueError, match='hot writes'):
        check_paths({'output': path}, mountinfo=MOUNTS, statvfs=capacity)


def test_local_output_and_component_boundary():
    rows = check_paths({'output': '/shared-other/new'}, mountinfo=MOUNTS, statvfs=capacity)
    assert rows[0]['filesystem'] == 'ext4'


def test_symlink_cannot_hide_shared_mount(tmp_path):
    (tmp_path / 'alias').symlink_to('/shared')
    assert mount_for(tmp_path / 'alias/new', MOUNTS)['filesystem'] == 'ceph'


def test_escaped_mount_and_read_only():
    mounts = MOUNTS + '5 1 8:2 / /local\\040disk ro - xfs /dev/data ro\n'
    assert mount_for('/local disk/new', mounts)['mount'] == '/local disk'
    with pytest.raises(ValueError, match='read-only'):
        check_paths({'output': '/local disk/new'}, mountinfo=mounts, statvfs=capacity)


@pytest.mark.parametrize('field,value', [('f_bavail', 1), ('f_favail', 1)])
def test_capacity_and_inode_reserve(field, value):
    def low(path):
        row = capacity(path)
        setattr(row, field, value)
        return row
    with pytest.raises(ValueError, match='reserve|inodes'):
        check_paths({'output': '/local/new'}, mountinfo=MOUNTS, statvfs=low)


def test_missing_cache_paths_fail_before_creation(tmp_path):
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='Explicit local cache'):
        require_training_storage(output, {})
    assert not output.exists()
