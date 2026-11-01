"""Stage a hash-verified local state view without changing shared experiment state.

This tool creates no experiment authority and never clears a flight lease. Call
only after the coordinator has reconciled the stopped worker and native contacts.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

LOCAL_BASE = Path('/var/lib/omni-local-state')
LOCAL_FILESYSTEMS = {'ext4', 'xfs', 'btrfs'}
ALLOWED_ROOTS = {'attempts', 'contact-journals', 'runs', 'prepared', 'serving',
                 'coordination', 'work', 'wire-check'}


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError('state snapshot refuses symlinks and special files')
        if stat.S_ISREG(mode):
            result[str(path.relative_to(root))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return result


def members(values):
    paths = [Path(value) for value in values]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('nonempty, unique writable directory list required')
    for path in paths:
        if (path.is_absolute() or '..' in path.parts or not path.parts
                or path.parts[0] not in ALLOWED_ROOTS
                or (len(path.parts) < 2 and str(path) != 'attempts')):
            raise ValueError('use narrow mutable directories (attempts includes its root locks)')
        if any(path != other and path.is_relative_to(other) for other in paths):
            raise ValueError('nested writable mounts are ambiguous')
    return paths


def filesystem(path):
    return subprocess.check_output(['findmnt', '-n', '-o', 'FSTYPE', '-T', str(path)], text=True).strip()


def _sync_tree(root):
    for path in root.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    directories = [root, *(p for p in root.rglob('*') if p.is_dir())]
    for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def stage(data, local, paths, expected_lease_sha256, *, base=LOCAL_BASE, fs_type=filesystem):
    data, local, base = Path(data).resolve(), Path(local).absolute(), Path(base).resolve()
    paths = members(paths)
    if (local.exists() or local.is_symlink() or local.parent != base
            or not base.is_dir() or base.resolve() != base):
        raise ValueError('fresh direct child of the prepared local base required')
    kind = fs_type(base)
    if kind not in LOCAL_FILESYSTEMS:
        raise ValueError('persistent local ext4/xfs/btrfs required')
    guard, lease = data/'locks/generation.guard', data/'locks/flight.lock'
    if guard.is_symlink() or not guard.is_file() or lease.is_symlink():
        raise ValueError('existing regular shared guard and lease required')
    with guard.open('r+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sha(lease) != expected_lease_sha256:
            raise ValueError('shared lease changed; preserve it and reconcile again')
        guard_stat = os.fstat(handle.fileno())
        # Validate all inputs before creating an output directory.
        source = {}
        for rel in paths:
            path = data/rel
            if path.resolve() != path or (path.exists() and not path.is_dir()):
                raise ValueError('nonsymlink source directories required')
            source[str(rel)] = inventory(path) if path.exists() else None
        local.mkdir(mode=0o700)
        for rel in paths:
            if source[str(rel)] is None:
                (local/'state'/rel).mkdir(parents=True)
            else:
                shutil.copytree(data/rel, local/'state'/rel)
            if inventory(local/'state'/rel) != (source[str(rel)] or {}):
                raise ValueError('copied bytes differ; preserve partial output for inspection')
        # Keep the original lease as a fence in BOTH views. The recovery step
        # must archive only the local copy after checking boot/contact evidence.
        (local/'state/locks').mkdir(parents=True)
        shutil.copy2(lease, local/'state/locks/flight.lock')
        (local/'state/locks/generation.guard').touch()
        for rel in paths:
            current = inventory(data/rel) if (data/rel).exists() else None
            if current != source[str(rel)]:
                raise ValueError('source changed during snapshot')
        if sha(lease) != expected_lease_sha256:
            raise ValueError('shared lease changed during snapshot')
        manifest = {'schema': 'saea-local-state-view-v1', 'data_root': str(data),
                    'local_root': str(local), 'filesystem': kind,
                    'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    'shared_guard': {'device': guard_stat.st_dev, 'inode': guard_stat.st_ino},
                    'shared_lease_sha256': expected_lease_sha256,
                    'initial_files': source, 'writable_directories': [str(p) for p in paths],
                    'experiment_authorized': False, 'shared_archive_current': False}
        (local/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
        _sync_tree(local)
        fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return manifest


def mount_arguments(local):
    local = Path(local).resolve()
    manifest = json.loads((local/'manifest.json').read_text())
    if manifest['schema'] != 'saea-local-state-view-v1' or manifest['local_root'] != str(local):
        raise ValueError('local state identity differs')
    data = Path(manifest['data_root'])
    guard = data/'locks/generation.guard'
    current = guard.stat()
    if manifest['shared_guard'] != {'device': current.st_dev, 'inode': current.st_ino}:
        raise ValueError('shared guard inode changed')
    if sha(data/'locks/flight.lock') != manifest['shared_lease_sha256']:
        raise ValueError('shared lease fence changed')
    paths = members(manifest['writable_directories'])
    args = ['--mount', f'type=bind,src={data},dst={data},readonly']
    for rel in [*paths, Path('locks')]:
        path = local/'state'/rel
        if path.resolve() != path or not path.is_dir():
            raise ValueError('local writable directory missing or redirected')
        args += ['--mount', f'type=bind,src={path},dst={data/rel}']
    args += ['--mount', f'type=bind,src={guard},dst={guard}']
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['stage', 'mount-arguments'])
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--directory', action='append', default=[])
    parser.add_argument('--expected-lease-sha256')
    args = parser.parse_args()
    if args.mode == 'stage':
        if args.data is None or args.expected_lease_sha256 is None:
            parser.error('stage requires --data and --expected-lease-sha256')
        result = stage(args.data, args.local, args.directory, args.expected_lease_sha256)
    else:
        result = mount_arguments(args.local)
    print(json.dumps(result, sort_keys=True))
