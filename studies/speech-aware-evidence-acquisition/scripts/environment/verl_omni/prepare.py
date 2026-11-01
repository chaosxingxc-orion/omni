"""Snapshot NGA's offline inputs into an owned, hash-locked Omni build context."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

NGA_IMAGE = 'sha256:8190b3ec51929baca2388452039ccf05d681c01b7c91cc55dd1e7ac5570a76b7'
OWNED = Path('/mnt/cephfs/xingchao/omni/work/build')
HERE = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def inventory(root):
    out = {}
    for path in sorted(root.rglob('*')):
        if '.git' in path.relative_to(root).parts or '__pycache__' in path.parts:
            continue
        if path.is_symlink():
            raise ValueError('snapshot refuses symlinks: ' + str(path.relative_to(root)))
        if path.is_file():
            out[str(path.relative_to(root))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return out

def verify(context):
    lock = json.loads((context / 'source-lock.json').read_text())
    actual = inventory(context)
    actual.pop('source-lock.json')
    if actual != lock['files']:
        raise ValueError('offline context bytes differ from source-lock.json')
    return {'files': len(actual), 'bytes': sum(v['bytes'] for v in actual.values()), 'verified': True}

def prepare(source, context):
    source = source.resolve(); context = context.resolve()
    if not context.is_relative_to(OWNED) or context.exists():
        raise ValueError('fresh context under Omni work/build required')
    before = inventory(source / 'ctx')
    context.mkdir(parents=True)
    for directory in ('verl_src', 'wheelhouse'):
        shutil.copytree(source/'ctx'/directory, context/directory,
                        ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    shutil.copy2(source/'ctx/raw_sft_dataset.py', context/'raw_sft_dataset.py')
    evidence = context/'nga-evidence'; evidence.mkdir()
    for name in ('image-info.txt', 'packages-image.txt', 'smoke-test.log'):
        shutil.copy2(source/name, evidence/name)
    shutil.copy2(source/'ctx/Dockerfile', evidence/'Dockerfile.nga')
    shutil.copy2(source/'ctx/Dockerfile.cann91', evidence/'Dockerfile.cann91')
    shutil.copy2(source/'proof_run/DONE', evidence/'proof-DONE.txt')
    patch = subprocess.check_output(['git','-C',str(source/'ctx/verl_src'),'diff','--binary'])
    (evidence/'source-vs-upstream.patch').write_bytes(patch)
    commit = subprocess.check_output(['git','-C',str(source/'ctx/verl_src'),'rev-parse','HEAD'],text=True).strip()
    shutil.copytree(HERE,context/'omni-tools',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(HERE/'Dockerfile',context/'Dockerfile')
    shutil.copy2(HERE/'Dockerfile.offline',context/'Dockerfile.offline')
    if inventory(source/'ctx') != before:
        raise ValueError('NGA source changed during snapshot; preserve partial context for inspection')
    lock = {'schema':'omni-verl-offline-context-v1','nga_recorded_image_id':NGA_IMAGE,
            'upstream_verl_commit':commit,'source_is_patched':True,
            'scope':'Offline source snapshot, not proof of built-image equivalence or Omni training support.',
            'files':inventory(context)}
    (context/'source-lock.json').write_text(json.dumps(lock,indent=2,sort_keys=True)+'\n')
    return verify(context)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','verify']);p.add_argument('--context',type=Path,required=True)
    p.add_argument('--nga-image-dir',type=Path)
    a=p.parse_args()
    if a.mode=='prepare' and a.nga_image_dir is None:p.error('--nga-image-dir required')
    print(json.dumps(prepare(a.nga_image_dir,a.context) if a.mode=='prepare' else verify(a.context)))
