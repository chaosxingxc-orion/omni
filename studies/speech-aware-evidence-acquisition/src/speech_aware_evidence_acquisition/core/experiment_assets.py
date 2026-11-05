"""Explicit, hash-bound external scientific inputs; never executable source overlays."""
import hashlib
import json
import os
from pathlib import Path


def resolve_asset(repo_root, relative):
    root=Path(repo_root).resolve();rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('experiment asset must be repository relative')
    manifest=os.environ.get('SAEA_EXPERIMENT_MANIFEST')
    if manifest is None:return root/rel
    raw=Path(manifest).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=os.environ.get('SAEA_EXPERIMENT_MANIFEST_SHA256'):
        raise ValueError('experiment manifest digest differs')
    doc=json.loads(raw)
    if set(doc)!={'schema','files'} or doc['schema']!='saea-experiment-assets-v1' or not isinstance(doc['files'],dict):
        raise ValueError('exact external experiment asset manifest required')
    for name,pin in doc['files'].items():
        rp=Path(name)
        if (rp.is_absolute() or '..' in rp.parts or not name.startswith(('configs/','docs/receipts/','docs/arms/'))
                or rp.suffix not in ('.json','.txt','.md','.jsonl')):
            raise ValueError('external asset cannot replace executable source')
        if not isinstance(pin,dict) or set(pin)!={'path','sha256'} or not Path(pin['path']).is_absolute():
            raise ValueError('exact external asset pin required')
    pin=doc['files'].get(rel.as_posix())
    if pin is None:return root/rel
    path=Path(pin['path'])
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=pin['sha256']:
        raise ValueError('external experiment asset absent or changed: '+str(rel))
    return path
