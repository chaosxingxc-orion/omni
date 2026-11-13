"""Fail closed before training writes hot state to shared or capacity-starved storage.

This checks the calling mount namespace, including bind mounts and symlinks.
It does not certify kernel compatibility, reserve capacity, or validate archives.
"""
import os
from pathlib import Path
import re


LOCAL_FILESYSTEMS = frozenset({'ext4', 'xfs', 'btrfs'})
CACHE_VARIABLES = ('TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME',
                   'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')


def _unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)


def mount_for(path, mountinfo):
    target = Path(path).resolve()
    matches = []
    for line in mountinfo.splitlines():
        before, after = line.split(' - ', 1)
        fields, filesystem = before.split(), after.split()[0]
        mount = Path(_unescape(fields[4]))
        if target.is_relative_to(mount):
            matches.append((len(mount.parts), int(fields[0]), str(mount), filesystem, fields[5]))
    if not matches:
        raise ValueError(f'No mount identity for {target}')
    _, _, mount, filesystem, options = max(matches)
    return {'path': str(target), 'mount': mount, 'filesystem': filesystem, 'options': options}


def check_paths(paths, *, mountinfo=None, statvfs=os.statvfs,
                min_free_bytes=20 * 1024**3, min_free_fraction=0.10):
    """Read-only admission. Minimums are reserves, not a workload sizing estimate."""
    if mountinfo is None:
        mountinfo = Path('/proc/self/mountinfo').read_text()
    records = []
    for name, path in paths.items():
        row = dict(name=name, **mount_for(path, mountinfo))
        if row['filesystem'] not in LOCAL_FILESYSTEMS:
            raise ValueError(f'{name}: hot writes require a verified local ext4/xfs/btrfs '
                             f'mount, found {row["filesystem"]}: {row["path"]}')
        if 'ro' in row['options'].split(','):
            raise ValueError(f'{name}: read-only runtime storage')
        ancestor = Path(row['path'])
        while not ancestor.exists():
            ancestor = ancestor.parent
        stat = statvfs(ancestor)
        free = stat.f_bavail * stat.f_frsize
        total = stat.f_blocks * stat.f_frsize
        fraction = free / total if total else 0
        if free < min_free_bytes or fraction < min_free_fraction:
            raise ValueError(f'{name}: inadequate local reserve: {free} bytes, {fraction:.2%} free')
        if stat.f_files and stat.f_favail / stat.f_files < 0.05:
            raise ValueError(f'{name}: less than 5% free inodes')
        records.append(dict(row, free_bytes=free, free_fraction=fraction))
    return records


def require_training_storage(output, environ=None):
    env = os.environ if environ is None else environ
    missing = [key for key in CACHE_VARIABLES if not env.get(key)]
    if missing:
        raise ValueError('Explicit local cache paths required before training: ' + ', '.join(missing))
    paths = {'output': output, **{key: env[key] for key in CACHE_VARIABLES}}
    # Ascend diagnostics must also avoid a shared HOME or configured work path.
    if not env.get('HOME'):
        raise ValueError('HOME is required to locate default Ascend diagnostics')
    paths['ascend_default_log'] = Path(env['HOME']) / 'ascend/log'
    for key in ('ASCEND_WORK_PATH', 'ASCEND_CACHE_PATH'):
        if env.get(key):
            paths[key] = env[key]
    return check_paths(paths)
