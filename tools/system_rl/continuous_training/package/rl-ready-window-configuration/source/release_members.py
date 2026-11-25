"""Rebind explicitly inventoried execution manifests in this release only."""
import hashlib,json
from pathlib import Path

def read(path):return json.loads(path.read_bytes())

def pin(path):
    with path.open("rb") as stream:
        h=hashlib.sha256()
        for buf in iter(lambda:stream.read(1048576),b""):h.update(buf)
    return dict(path=str(path.resolve()),sha256=h.hexdigest(),bytes=path.stat().st_size)

EXECUTION_CONFIGS = {
    'parents/collection-dispatch-integration/production-pipeline-next-version-v1/'
    'finite-diagnostic-capture-v1/CONFIG.json',
}


MANIFEST_NAMES={'SOURCE-PINS.json','SOURCEPINS.json','PARENT-PINS.json','DEPENDENCY-PINS.json','DEPENDENCIES.json',
                'PIN.repair-v2.json','SOURCE-CLOSURE.json','CAPTURE-CLOSURE.json','OVERLAY.json','ASSEMBLED-BINDING.json'}


def binding_member(rel):
    return rel.parts[:1]==('parents',) and (rel.name in MANIFEST_NAMES or rel.as_posix() in EXECUTION_CONFIGS)


def rebind_manifests(tree,logical_root,*,members):
    # Only paths from the sealed input inventory or the actual package copy.
    paths=[]
    for rel in sorted(set(map(Path,members))):
        if not binding_member(rel):continue
        p=tree/rel
        if '..' in rel.parts or p.resolve()!=p.absolute() or not p.is_file():
            raise ValueError('invalid or missing manifest: '+str(p))
        if p.stat().st_size:paths.append(p)
    def visit(v):
        if isinstance(v,list):return [visit(x) for x in v]
        if not isinstance(v,dict):return v
        out={k:visit(x) for k,x in v.items()}
        for key,row in list(out.items()):
            if isinstance(key,str) and key.startswith('/') and isinstance(row,dict) and row.get('sha256'):
                path=Path(key)
                if path.is_relative_to(logical_root):
                    actual=tree/'parents'/path.relative_to(logical_root)
                    if not actual.resolve().is_relative_to((tree/'parents').resolve()):raise ValueError('manifest dependency escapes release: '+str(path))
                    if actual.is_file() and actual.stat().st_size:
                        bound=pin(actual);out[key]=dict(row,sha256=bound['sha256'],bytes=bound['bytes'])
        if isinstance(out.get('path'),str) and out.get('sha256'):
            path=Path(out['path'])
            if path.is_relative_to(logical_root):
                actual=tree/'parents'/path.relative_to(logical_root)
                if not actual.resolve().is_relative_to((tree/'parents').resolve()):raise ValueError('manifest dependency escapes release: '+str(path))
                if actual.is_file() and actual.stat().st_size:out.update(dict(pin(actual),path=str(path)))
        return out
    for _ in range(20):
        changed=False
        for p in paths:
            old=read(p);new=visit(old)
            if new!=old:p.write_text(json.dumps(new,indent=2)+'\n');changed=True
        if not changed:return
    raise ValueError('source manifest dependency cycle')
