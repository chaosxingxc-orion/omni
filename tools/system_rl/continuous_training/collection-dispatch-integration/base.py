"""Import the sealed collector in place; never copy its DAG or business code."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'collection-scale-successor'
SEALED = PARENT / 'ready-autocontinue/production-next'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SEALED / 'code'))
from common import canonical, digest, require, save, Storage, mkdir, fingerprint, load_module

def read(path):
    return json.loads(Path(path).read_bytes())

def pin(path):
    p = Path(path).resolve()
    raw = p.read_bytes()
    return dict(path=str(p), sha256=digest(raw), bytes=len(raw))

def checked(ref):
    p = Path(ref['path']); before = fingerprint(p); raw = p.read_bytes()
    require(before == fingerprint(p) and digest(raw) == ref['sha256'] and len(raw) == ref.get('bytes', len(raw)), 'PIN_DRIFT:' + str(p))
    return json.loads(raw)

def official_domains(dataset):
    module = PARENT / 'code/train_population.py'
    return load_module(dict(path=str(module), sha256=digest(module.read_bytes())),
                       'collection_train_population').official_domains(dataset)

def process_identity():
    import os
    s = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(s[19]), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())

def alive(identity):
    if identity['boot'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        return False
    p = Path('/proc') / str(identity['pid']) / 'stat'
    try: s = p.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError: return False
    return int(s[19]) == identity['start_ticks'] and s[0] != 'Z'
