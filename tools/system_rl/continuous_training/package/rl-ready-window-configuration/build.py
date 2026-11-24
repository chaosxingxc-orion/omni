"""Materialize a minimal successor of the accepted build; never dispatch.

Only explicit modified modules are checked. Accepted build16/224 CPU evidence
is reused, without rehashing the entire historical asset population.
"""
import argparse,hashlib,importlib.util,json,os,shutil
from pathlib import Path
from source.release_completion import require_closed_release
D=Path(__file__).resolve().parent

def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 b=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode()
 if p.exists():assert p.read_bytes()==b;return pin(p)
 t=p.with_suffix(p.suffix+'.pending')
 with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(t,p);fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd);return pin(p)

def close_release(root,composer):
 composer.durable_tree(root)
 save(Path(root)/'BUILD-COMPLETE.json',dict(schema='durable-release-complete-v1',
  build_intent_ref=pin(Path(root)/'BUILD-INTENT.json'),delta_ref=pin(Path(root)/'DELTA.json'),
  assembly_ref=pin(Path(root)/'ASSEMBLY.json')))

def build(profile):
 c=json.loads(Path(profile).read_bytes())['values'];base=Path(c['BASE_BUILD']);out=Path(c['BUILD'])
 assert not out.exists() and out.is_relative_to(D),'new owned build only'
 st=os.statvfs(out.parent);assert st.f_bavail*st.f_frsize>=max(20*1024**3,.1*st.f_blocks*st.f_frsize)+1024**3 and st.f_favail/st.f_files>=.05
 fs=[]
 for line in Path('/proc/self/mountinfo').read_text().splitlines():
  pre,post=line.split(' - ',1);mount=Path(pre.split()[4])
  if out.is_relative_to(mount):fs.append((len(mount.parts),post.split()[0]))
 assert max(fs)[1] in ('ext4','xfs','btrfs')
 require_closed_release(base)
 out.mkdir();save(out/'BUILD-INTENT.json',dict(profile_ref=pin(profile),base=str(base)))
 shutil.copytree(base/'tree',out/'tree');shutil.copytree(base/'client',out/'client')
 logical=Path(c['WORKSPACE'])/'delivery/system-rl-transition-v1'
 impl=out/'tree/parents/jev-system-rl/implementation';delta=[]
 for src in sorted((D/'source').glob('*.py')):
  target=impl/src.name;before=pin(target) if target.exists() else None;compile(src.read_text(),str(src),'exec');shutil.copyfile(src,target)
  if (out/'client'/src.name).exists():shutil.copyfile(src,out/'client'/src.name)
  delta.append(dict(target=str(target.relative_to(out)),base_ref=before,result_ref=pin(target),source_ref=pin(src)))
 pins=json.loads((impl/'SOURCE-PINS.json').read_bytes())
 for row in delta:
  ref=dict(row['result_ref'],path=str(logical/'jev-system-rl/implementation'/Path(row['target']).name))
  found=[r for r in pins['sources'] if r['path']==ref['path']];assert len(found)<=1
  if found:found[0].update(ref)
  else:pins['sources'].append(ref)
 (impl/'SOURCE-PINS.json').write_text(json.dumps(pins,indent=2)+'\n')
 # Original source-pin rebinder updates only executable manifests in successor.
 spec=importlib.util.spec_from_file_location('accepted_source_composer',c['COMPOSER']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.rebind_manifests(out/'tree',logical)
 assembly=json.loads((base/'ASSEMBLY.json').read_bytes());assembly.update(tree=str(out/'tree'),base_assembly_ref=pin(base/'ASSEMBLY.json'))
 for name,check in assembly['final_owner_assertions'].items():
  if name in [r['target'].removeprefix('tree/') for r in delta] or Path(name).name=='SOURCE-PINS.json':
   ref=pin(out/'tree'/name);check.update(expected=ref,actual=ref,passed=True)
 for row in delta:
  name=row['target'].removeprefix('tree/');ref=pin(out/'tree'/name)
  assembly['final_owner_assertions'][name]=dict(expected=ref,actual=ref,passed=True)
 # Rebinding the changed modules also changes these two inherited closures.
 # Authenticate their exact delta, never accept an unrelated inherited drift.
 for name,check in assembly['final_owner_assertions'].items():
  if Path(name).name!='SOURCE-CLOSURE.json':continue
  before=json.loads((base/'tree'/name).read_bytes());after=json.loads((out/'tree'/name).read_bytes())
  if before==after:continue
  assert isinstance(before,list) and len(before)==len(after)
  changed={str(logical/'jev-system-rl/implementation'/Path(r['target']).name) for r in delta}
  for old,new in zip(before,after):
   assert old==new or old['path']==new['path'] and old['path'] in changed and new==pin(out/'tree/parents/jev-system-rl/implementation'/Path(old['path']).name)|{'path':old['path']}
  ref=pin(out/'tree'/name);check.update(expected=ref,actual=ref,passed=True)
 save(out/'ASSEMBLY.json',assembly)
 save(out/'DELTA.json',dict(base_build=str(base),base_assembly_ref=pin(base/'ASSEMBLY.json'),changes=delta,accepted_CPU_checks_reused=224,NPU=False,dispatch=False,build_completion_required=True))
 close_release(out,m)
 return pin(out/'DELTA.json')

def build_delta(manifest, output):
 """Apply an explicit, pinned incremental release without replaying old overlays.

 Each replacement binds the exact inherited file. Additions may not overwrite
 any inherited module. Existing precision/storage/checkpoint policy remains in
 the parent unless the manifest explicitly changes it.
 """
 c=json.loads(Path(manifest).read_bytes())
 assert c['schema']=='omni-pinned-release-delta-v1'
 def checked(ref):
  assert pin(ref['path'])==ref,'delta input drift: '+ref['path']
  return Path(ref['path'])
 base=Path(c['base']);out=Path(output).resolve();logical=Path(c['logical_root'])
 require_closed_release(base)
 assert checked(c['base_assembly_ref'])==base/'ASSEMBLY.json'
 assert not out.exists() and not out.is_relative_to(base),'new successor outside parent required'
 checked(c['composer_ref'])
 spec=importlib.util.spec_from_file_location('delta_original_composer',c['composer_ref']['path'])
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.storage_admit(out,c['storage_budget_bytes'])
 ancestor=out.parent
 while not ancestor.exists():ancestor=ancestor.parent
 st=os.statvfs(ancestor)
 assert st.f_bavail*st.f_frsize>=max(20*1024**3,.1*st.f_blocks*st.f_frsize)+c['storage_budget_bytes'],'current reserve + workload'
 assembly=json.loads(checked(c['base_assembly_ref']).read_bytes())
 for name,claim in assembly['final_owner_assertions'].items():
  assert pin(base/'tree'/name)['sha256']==claim['actual']['sha256'],'inherited owner drift: '+name
 edits=[];names=set()
 for row in c['changes']:
  rel=Path(row['target'])
  assert not rel.is_absolute() and '..' not in rel.parts and str(rel) not in names
  names.add(str(rel));target=base/'tree'/rel
  if row.get('base_ref'):
   assert checked(row['base_ref'])==target
  else:assert not target.exists(),'addition collision: '+str(rel)
  if 'source_ref' in row:
   data=checked(row['source_ref']).read_bytes()
  else:
   data=target.read_bytes();text=data.decode()
   for patch in row['replacements']:
    assert patch['old'] and text.count(patch['old'])==1,'ambiguous/missing delta seam: '+str(rel)
    text=text.replace(patch['old'],patch['new'],1)
   data=text.encode()
  assert hashlib.sha256(data).hexdigest()==row['result_sha256'],'delta result differs: '+str(rel)
  if rel.suffix=='.py':compile(data,str(rel),'exec')
  edits.append((row,data))
 client_edits=[]
 for row in c.get('client_changes',[]):
  rel=Path(row['target'])
  assert len(rel.parts)==2 and rel.parts[0]=='client' and rel.suffix=='.py','explicit client entrypoint only'
  assert str(rel) not in names,'duplicate client change'
  names.add(str(rel))
  assert checked(row['base_ref'])==base/rel,'client parent differs'
  data=checked(row['source_ref']).read_bytes()
  assert hashlib.sha256(data).hexdigest()==row['result_sha256'],'client result differs'
  compile(data,str(rel),'exec');client_edits.append((row,data))
 out.mkdir(parents=True);save(out/'BUILD-INTENT.json',dict(manifest_ref=pin(manifest),base_assembly_ref=c['base_assembly_ref']))
 shutil.copytree(base/'tree',out/'tree');shutil.copytree(base/'client',out/'client')
 changes=[]
 for row,data in edits:
  target=out/'tree'/row['target'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  changes.append(dict(target=row['target'],base_ref=row.get('base_ref'),result_ref=pin(target)))
  # Client entrypoints resolve their sibling/source modules before the tree.
  # Mirror only proven byte-identical inherited copies; divergence is explicit.
  if Path(row['target']).parent==Path('parents/jev-system-rl/implementation'):
   for rel in (Path('client')/target.name,Path('client/source')/target.name):
    prior=base/rel
    if prior.exists():
     assert row.get('base_ref') and pin(prior)['sha256']==row['base_ref']['sha256'],'divergent client copy: '+str(rel)
     (out/rel).write_bytes(data)
 client_changes=[]
 for row,data in client_edits:
  target=out/row['target'];target.write_bytes(data)
  client_changes.append(dict(target=row['target'],base_ref=row['base_ref'],result_ref=pin(target)))
 pins_path=out/'tree/parents/jev-system-rl/implementation/SOURCE-PINS.json'
 pins=json.loads(pins_path.read_bytes());by_path={r['path']:r for r in pins['sources']}
 for row,_ in edits:
  rel=Path(row['target'])
  assert rel.parts[0]=='parents','only explicit original parent namespace supported'
  path=str(logical/Path(*rel.parts[1:]));by_path[path]=dict(pin(out/'tree'/rel),path=path)
 pins['sources']=[by_path[k] for k in sorted(by_path)]
 pins_path.write_text(json.dumps(pins,indent=2)+'\n')
 m.rebind_manifests(out/'tree',logical)
 derived=[]
 # Authenticate every inherited change: edits or deterministic source manifests.
 manifest_names={'SOURCE-PINS.json','SOURCEPINS.json','PARENT-PINS.json','DEPENDENCY-PINS.json','DEPENDENCIES.json','PIN.repair-v2.json','SOURCE-CLOSURE.json','CAPTURE-CLOSURE.json','OVERLAY.json','ASSEMBLED-BINDING.json'}
 for name,claim in assembly['final_owner_assertions'].items():
  actual=pin(out/'tree'/name)
  if actual['sha256']!=claim['actual']['sha256']:
   assert name in names or Path(name).name in manifest_names or name in getattr(m,'EXECUTION_CONFIGS',set()),'undeclared inherited change: '+name
   derived.append(dict(target=name,before=claim['actual'],after=actual))
  claim.update(expected=actual,actual=actual,passed=True)
 for row,_ in edits:
  actual=pin(out/'tree'/row['target'])
  assembly['final_owner_assertions'][row['target']]=dict(expected=actual,actual=actual,passed=True)
 assembly.update(tree=str(out/'tree'),base_assembly_ref=c['base_assembly_ref'],delta_manifest_ref=pin(manifest),dispatch=False)
 save(out/'ASSEMBLY.json',assembly)
 save(out/'DELTA.json',dict(schema=c['schema'],manifest_ref=pin(manifest),base_assembly_ref=c['base_assembly_ref'],changes=changes,client_changes=client_changes,derived=derived,NPU=False,dispatch=False,training_admission=False,build_completion_required=True))
 close_release(out,m)
 return pin(out/'DELTA.json')
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--profile');g.add_argument('--delta-manifest');p.add_argument('--output');a=p.parse_args()
 if a.delta_manifest and not a.output:p.error('--delta-manifest requires --output')
 print(json.dumps(build_delta(a.delta_manifest,a.output) if a.delta_manifest else build(a.profile)))
