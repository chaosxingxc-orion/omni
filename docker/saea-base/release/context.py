"""Create a minimal, reproducible release context from committed Git objects."""
import argparse,hashlib,json,os,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
commit=subprocess.check_output(['git','-C',str(a.repo),'rev-parse',a.commit+'^{commit}'],text=True).strip()
files=subprocess.check_output(['git','-C',str(a.repo),'ls-tree','-r','--name-only',commit],text=True).splitlines()
prefixes=('studies/speech-aware-evidence-acquisition/src/','studies/speech-aware-evidence-acquisition/scripts/','studies/speech-aware-evidence-acquisition/configs/','studies/speech-aware-evidence-acquisition/docs/arms/','scripts/npu/')
exact={'docs/datasets.lock.json','studies/speech-aware-evidence-acquisition/pyproject.toml','studies/speech-aware-evidence-acquisition/LICENSE'}
selected=[f for f in files if f.startswith(prefixes) or f in exact];a.out.mkdir(parents=True,exist_ok=False);inventory={}
for name in selected:
 data=subprocess.check_output(['git','-C',str(a.repo),'show',commit+':'+name]);dest=a.out/'umbrella'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);inventory[name]=hashlib.sha256(data).hexdigest()
manifest={'schema':'saea-release-source-v1','commit':commit,'files':inventory};(a.out/'source-manifest.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n')
for name in ('Dockerfile','verify-release.py'):(a.out/name).write_bytes((pathlib.Path(__file__).parent/name).read_bytes())
print(json.dumps({'commit':commit,'files':len(inventory),'manifest_sha256':hashlib.sha256((a.out/'source-manifest.json').read_bytes()).hexdigest()}))
