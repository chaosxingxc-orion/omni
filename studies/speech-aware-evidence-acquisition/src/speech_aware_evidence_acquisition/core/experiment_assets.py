"""Explicit, hash-bound external scientific inputs; never executable source overlays."""
import hashlib
import json
import os
from pathlib import Path


def resolve_asset(repo_root, relative):
    root=Path(repo_root).resolve();rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('experiment asset must be repository relative')
    workspace=workspace_asset(root,rel.as_posix())
    if workspace is not None:return workspace
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
        if (rp.is_absolute() or '..' in rp.parts or not name.startswith(('configs/','docs/'))
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


def workspace_root(repo_root):
    value=os.environ.get('SAEA_EXPERIMENT_WORKSPACE')
    if value is None:return Path(repo_root)
    path=Path(value)
    if not path.is_absolute() or path.resolve().is_relative_to(Path(repo_root).resolve()):
        raise ValueError('experiment workspace must be absolute and outside sealed code')
    return path


def output_asset(repo_root,relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts or not str(rel).startswith(('configs/','docs/')):
        raise ValueError('scientific output must be configuration or documentation')
    return workspace_root(repo_root)/rel


def seal_workspace(repo_root):
    """Publish a full configuration inventory after a successful CPU mutation."""
    root=workspace_root(repo_root)
    if root==Path(repo_root):return None
    from .atomic_file import publish_new
    from .bundle.git_commit import require_clean_study_repo
    files={}
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('workspace symlink forbidden')
        if path.is_file() and path.relative_to(root).parts[0] in ('configs','docs'):
            files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    doc={'schema':'saea-experiment-workspace-v1','source_commit':require_clean_study_repo(repo_root),'source_manifest_sha256':os.environ.get('SAEA_RELEASE_MANIFEST_SHA256'),'files':files}
    raw=(json.dumps(doc,sort_keys=True,separators=(',',':'))+'\n').encode();sha=hashlib.sha256(raw).hexdigest()
    target=root/'manifests'/(sha+'.json')
    if not target.exists():publish_new(target,raw)
    return {'path':str(target),'sha256':sha}


def workspace_asset(repo_root,relative):
    root=workspace_root(repo_root)
    if root==Path(repo_root):return None
    path=root/relative
    manifest=os.environ.get('SAEA_EXPERIMENT_WORKSPACE_MANIFEST');expected=os.environ.get('SAEA_EXPERIMENT_WORKSPACE_SHA256')
    if not manifest or not expected:
        if not path.exists() and not path.is_symlink():return None
        raise ValueError('workspace read requires explicit inventory pin')
    raw=Path(manifest).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('workspace inventory digest differs')
    doc=json.loads(raw)
    if (set(doc)!={'schema','source_commit','source_manifest_sha256','files'} or doc['schema']!='saea-experiment-workspace-v1'
        or doc['source_commit']!=source_commit(repo_root)
        or doc['source_manifest_sha256']!=os.environ.get('SAEA_RELEASE_MANIFEST_SHA256')):
        raise ValueError('workspace source identity differs')
    if relative not in doc['files']:
        if path.exists() or path.is_symlink():raise ValueError('workspace asset absent from pinned inventory or changed')
        return None
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=doc['files'][relative]:
        raise ValueError('workspace asset absent from pinned inventory or changed')
    return path


def source_commit(repo_root):
    manifest=os.environ.get('SAEA_RELEASE_MANIFEST')
    if manifest:
        raw=Path(manifest).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=os.environ.get('SAEA_RELEASE_MANIFEST_SHA256'):raise ValueError('release source inventory differs')
        return json.loads(raw)['commit']
    from .bundle.git_commit import require_clean_study_repo
    return require_clean_study_repo(repo_root)
