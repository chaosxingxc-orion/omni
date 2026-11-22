"""Local immutable commits; no accelerator imports or alternate runtime."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import uuid

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MODEL = 'qwen3-omni-30b-a3b-instruct'
sys.dont_write_bytecode = True


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pin(path):
    p = Path(path).resolve()
    before = p.stat()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    after = p.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'INPUT_CHANGED')
    return dict(path=str(p), sha256=h.hexdigest(), bytes=after.st_size)


def raw(ref):
    require(pin(ref['path']) == {k: ref[k] for k in ('path', 'sha256', 'bytes')}, 'PIN_DRIFT:' + ref['path'])
    return Path(ref['path']).read_bytes()


def doc(ref):
    return json.loads(raw(ref))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def storage(path, budget=0):
    p = Path(path).resolve()
    while not p.exists():
        p = p.parent
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        a, b = line.split(' - ', 1)
        fields = a.split()
        mount = Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), fields[4]))
        if p == mount or mount in p.parents:
            mounts.append((len(str(mount)), b.split()[0], fields[5]))
    _, fs, options = max(mounts)
    s = os.statvfs(p)
    require(fs in ('ext4', 'xfs', 'btrfs') and 'ro' not in options.split(','), 'LOCAL_PERSISTENT_FS')
    require(s.f_bavail * s.f_frsize >= max(20 * 1024**3, .1 * s.f_blocks * s.f_frsize) + budget,
            'LOCAL_RESERVE_PLUS_WORKLOAD')
    require(s.f_files and s.f_favail / s.f_files >= .05, 'LOCAL_INODES')
    return dict(filesystem=fs, free_bytes=s.f_bavail * s.f_frsize,
                free_fraction=s.f_bavail / s.f_blocks, inode_fraction=s.f_favail / s.f_files)


def save(path, value):
    p = Path(path)
    if p.exists():
        existing=pin(p)
        require(doc(existing) == value, 'IMMUTABLE_COLLISION:' + str(p))
        return existing
    storage(p.parent, len(canonical(value)))
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name('.' + uuid.uuid4().hex)
    with temp.open('xb') as f:
        os.chmod(temp, 0o600)
        f.write(canonical(value))
        f.flush()
        os.fsync(f.fileno())
    # Atomic no-clobber publication, retaining an interrupted commit for recovery.
    os.link(temp, p)
    temp.unlink()
    fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return pin(p)


def environment(root):
    root = Path(root).resolve()
    storage(root, 1024**3)
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for key, leaf in dict(TMPDIR='tmp', XDG_CACHE_HOME='cache', HF_HOME='hf',
                          TORCHINDUCTOR_CACHE_DIR='inductor', TRITON_CACHE_DIR='triton',
                          ASCEND_WORK_PATH='ascend-work', ASCEND_CACHE_PATH='ascend-cache',
                          ASCEND_PROCESS_LOG_PATH='ascend-log').items():
        p = root / leaf
        p.mkdir(exist_ok=True)
        storage(p)
        result[key] = str(p)
    # HOME is not repurposed. Container launcher binds a local directory here.
    storage(Path.home() / 'ascend/log')
    return result


def existing(value):
    if isinstance(value, dict):
        if set(value) == {'existing_path'}:
            return pin(value['existing_path'])
        return {k: existing(v) for k, v in value.items()}
    if isinstance(value, list):
        return [existing(v) for v in value]
    return value


def source_mounts():
    path=os.environ.get('OMNI_ITERATION_STAGE_MANIFEST')
    return doc(pin(path))['source_mounts']+[dict(source=str(Path(path).parent),target=str(Path(path).parent),readonly=True)] if path else [dict(source=str(BASE.parents[1]),target=str(BASE.parents[1]),readonly=True)]


def stage_environment():
    path=os.environ.get('OMNI_ITERATION_STAGE_MANIFEST')
    return {'OMNI_ITERATION_STAGE_MANIFEST':path} if path else {}
