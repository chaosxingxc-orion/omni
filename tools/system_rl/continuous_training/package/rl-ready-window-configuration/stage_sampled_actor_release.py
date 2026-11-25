"""Stage changed overlays and the initial policy beside a retained local capture tree."""
import argparse, hashlib, json, os, shlex, subprocess, tarfile, tempfile
from pathlib import Path


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

REMOTE = r"""
import hashlib,json,os,shutil,subprocess,sys,tarfile,tempfile
from pathlib import Path
q=json.loads(sys.argv[1]); old=Path(q['old']); dest=Path(q['dest']); parent=dest.parent
assert old.is_absolute() and old.resolve()==old and dest.resolve()==dest and old.parent==parent and old!=dest
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==q['boot']
assert subprocess.check_output(['findmnt','-T',str(parent),'-n','-o','FSTYPE'],text=True).strip() in ('ext4','btrfs','xfs')
fs=os.statvfs(parent); assert fs.f_bavail*fs.f_frsize>=max((20<<30)+q['bytes'],fs.f_blocks*fs.f_frsize*.1) and fs.f_favail>=fs.f_files*.05
def syncdir(p):
 fd=os.open(p,os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def verify(d):
 assert all(digest(d/name)==v for name,v in q['files'].items())
if dest.exists():
 assert json.loads((dest/'STAGED.json').read_bytes())==q;verify(dest);print('already staged');sys.exit(0)
tmp=Path(tempfile.mkdtemp(prefix=dest.name+'.partial.',dir=parent))
try:
 for name in ('cold','overlays','tools','receipts','policy_assets'):
  subprocess.run(['cp','-al','--',str(old/name),str(tmp/name)],check=True)
 seen=set()
 with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tf:
  for member in tf:
   name=member.name; target=tmp/name
   assert name in q['files'] and name not in seen and member.isfile() and target.resolve()==target and tmp in target.parents
   seen.add(name);target.parent.mkdir(parents=True,exist_ok=True)
   with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as f:
    with tf.extractfile(member) as stream:shutil.copyfileobj(stream,f)
    f.flush();os.fsync(f.fileno())
   os.replace(f.name,target)
 assert seen==set(q['files']);verify(tmp)
 with (tmp/'STAGED.json').open('x') as f:
  json.dump(q,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 for p,ds,fs in os.walk(tmp,topdown=False):syncdir(p)
 os.rename(tmp,dest);syncdir(parent)
except BaseException:
 print('Uncommitted staging retained at '+str(tmp),file=sys.stderr);raise
print(str(dest))
"""

def committed_files(custody,logical,ref):
    """Only the requested commit and the rank manifest's exact members may ship."""
    receipt=custody/logical.relative_to(logical.parents[1])
    if receipt.resolve()!=receipt or receipt.name!='COMMITTED.json' or sha(receipt)!=ref['sha256']:
        raise ValueError('checkpoint custody differs: '+str(receipt))
    commit=json.loads(receipt.read_bytes())
    if not commit['recoverable'] or [r['rank'] for r in commit['ranks']]!=list(range(commit['world_size'])):
        raise ValueError('incomplete checkpoint rank roster')
    members={receipt:ref['sha256']}
    for row in commit['ranks']:
        for name,digest in row['files'].items():
            rel=Path(name)
            if (rel.is_absolute() or '..' in rel.parts or str(rel)!=name or not rel.parts
                    or any(c in name for c in '*?[]')):
                raise ValueError('invalid checkpoint member: '+name)
            path=receipt.parent/rel
            if path==receipt or (path in members and members[path]!=digest):
                raise ValueError('conflicting checkpoint member: '+name)
            members[path]=digest
    for p,h in members.items():
        if p.resolve()!=p or not p.is_file() or sha(p)!=h:raise ValueError('changed or missing checkpoint member: '+str(p))
    return members


def stage(manifest, preview, custody):
    def read(p):return json.loads(Path(p).read_bytes())
    m=read(manifest); actor=read(Path(preview)/'ACTOR.json'); old=read(m['actor_check_ref']['path'])
    plan=read(Path(preview)/'PREVIEW-NOT-FOR-LAUNCH.json')
    if actor.get('initial_actor'):
        if (plan.get('initial_resume_ref') or custody or
                plan.get('branch_initial_ref')!=actor['initial_actor']['branch_ref']):
            raise ValueError('initial weight deployment differs from branch')
    else:
        logical=Path(plan['initial_resume_ref']['path'])
        if not custody or custody.name!=logical.parents[1].name:
            raise ValueError('checkpoint custody differs')
    files={'overlays/'+rel:Path(m['release'])/'tree/parents'/rel for rel in actor['capture_mounts']['overlays']}
    assert all(sha(p)==actor['capture_mounts']['overlays'][n[len('overlays/'):]] for n,p in files.items())
    # Some generated documents carry the original logical source name. Supply
    # the exact verified overlay at that name in this new, immutable actor tree.
    for role in m.get('actor_logical_source_roles', []):
        rel=actor['generation']['runtime']['sources'][role]
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        files['cold/'+rel]=files['overlays/'+rel]
    if custody:
        for p in committed_files(custody,logical,plan['initial_resume_ref']):
            files['policy_assets/'+str(p.relative_to(custody.parent))]=p
    spec=dict(old=old['capture_mounts']['root'],dest=actor['capture_mounts']['root'],boot=actor['expected_boot'],
              files={n:sha(p) for n,p in files.items()},bytes=sum(p.stat().st_size for p in files.values()))
    with tempfile.TemporaryFile(dir=manifest.parent) as archive:
        with tarfile.open(fileobj=archive,mode='w') as tf:
            for n,p in sorted(files.items()):tf.add(p,arcname=n,recursive=False)
        archive.seek(0)
        subprocess.run(['ssh','-oBatchMode=yes',actor['host'],shlex.join(['python3','-B','-c',REMOTE,json.dumps(spec)])],stdin=archive,check=True)
    return spec

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('manifest','preview'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--custody',type=Path)
    a=p.parse_args();q=stage(a.manifest,a.preview,a.custody)
    print(json.dumps(dict(remote_root=q['dest'],verified_files=len(q['files']),bytes=q['bytes'])))
