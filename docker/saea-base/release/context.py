"""Create a minimal, reproducible release context from committed Git objects."""
import argparse,hashlib,json,os,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
commit=subprocess.check_output(['git','-C',str(a.repo),'rev-parse',a.commit+'^{commit}'],text=True).strip()
entries=subprocess.check_output(['git','-C',str(a.repo),'ls-tree','-r',commit],text=True).splitlines()
prefixes=('studies/speech-aware-evidence-acquisition/src/','studies/speech-aware-evidence-acquisition/scripts/','studies/speech-aware-evidence-acquisition/configs/','studies/speech-aware-evidence-acquisition/docs/arms/','scripts/npu/','studies/speech-aware-evidence-acquisition/docs/receipts/rosters/')
# The shipped lock-baseline compiler requires this admission record.
exact={'studies/speech-aware-evidence-acquisition/docs/results/2026-09-30-slurp-a2-comp1b-dev2000-read.md','docs/datasets.lock.json','studies/speech-aware-evidence-acquisition/pyproject.toml','studies/speech-aware-evidence-acquisition/LICENSE', 'docker/saea-base/release/verify-release.py', 'docker/saea-base/release/install-source.py', 'docker/saea-base/release/Dockerfile', 'docker/saea-base/release/context.py', 'docker/saea-base/release/build-release.sh'}
selected=[]
for row in entries:
 meta,name=row.split('\t',1);mode,kind,oid=meta.split()
 if name.startswith(prefixes) or name in exact:
  if mode not in ('100644','100755') or kind!='blob':raise ValueError('unsupported release file '+name)
  selected.append((name,oid,mode))
a.out.mkdir(parents=True,exist_ok=False);inventory={}
proc=subprocess.Popen(['git','-C',str(a.repo),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,oid,mode in selected:
 proc.stdin.write((oid+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().decode().split();size=int(header[2]);data=proc.stdout.read(size);assert proc.stdout.read(1)==b'\n';dest=a.out/'umbrella'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);dest.chmod(0o755 if mode=='100755' else 0o644);inventory[name]=hashlib.sha256(data).hexdigest()
proc.stdin.close();assert proc.wait()==0
manifest={'schema':'saea-release-source-v1','commit':commit,'files':inventory};(a.out/'source-manifest.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n')
for name in ('Dockerfile','verify-release.py','install-source.py','build-release.sh'):
 data=subprocess.check_output(['git','-C',str(a.repo),'show',commit+':docker/saea-base/release/'+name]);(a.out/name).write_bytes(data)
print(json.dumps({'commit':commit,'files':len(inventory),'manifest_sha256':hashlib.sha256((a.out/'source-manifest.json').read_bytes()).hexdigest()}))
