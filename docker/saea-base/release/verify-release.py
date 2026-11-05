"""Verify a complete immutable release source inventory, including extra files."""
import hashlib,json,pathlib,sys

def verify(manifest,root):
 root=pathlib.Path(root).resolve();doc=json.loads(pathlib.Path(manifest).read_bytes())
 if set(doc)!={'schema','commit','files'} or doc['schema']!='saea-release-source-v1' or not isinstance(doc['files'],dict):raise ValueError('invalid source manifest')
 import re
 if not re.fullmatch('[0-9a-f]{40}',doc['commit']):raise ValueError('invalid source commit')
 actual=set()
 for f in root.rglob('*'):
  if f.is_symlink():raise ValueError('symlinks are not allowed')
  if f.is_file() and f.relative_to(root).as_posix() not in ('source-manifest.json','verify-release.py'):
   actual.add(f.relative_to(root).as_posix())
 if actual!=set(doc['files']):raise ValueError('release file set differs')
 for name,digest in doc['files'].items():
  p=root/name
  if not p.resolve().is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('release digest differs: '+name)
 if hashlib.sha256((root/'verify-release.py').read_bytes()).hexdigest()!=doc['files']['docker/saea-base/release/verify-release.py']:raise ValueError('release verifier differs')
 return doc['commit']
if __name__=='__main__':print(json.dumps({'commit':verify(sys.argv[1],sys.argv[2]),'verified':True}))
