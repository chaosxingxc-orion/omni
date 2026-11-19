"""Resume the same fixed iteration with a pinned read-only result reader."""
import argparse,json,hashlib,subprocess,os
from pathlib import Path
p=argparse.ArgumentParser()
for key in ('handoff','release','output'):p.add_argument('--'+key,required=True)
p.add_argument('--cpu-closure-repair',action='store_true')
p.add_argument('--closed-pull-repair',action='store_true')
p.add_argument('--previous-resume',help='Exact failed recovery container, if any')
p.add_argument('--existing-transport',action='store_true',help='Resume observation of the same already staged request')
a=p.parse_args();handoff=Path(a.handoff);release=Path(a.release);out=Path(a.output)
intent=json.loads((handoff/'EXECUTE-INTENT.json').read_text());old=json.loads((handoff/'EXECUTE-LAUNCH.json').read_text())
obj=json.loads(subprocess.check_output(['docker','inspect',old['cid']]))[0];assert obj['State']['Status']=='exited' and obj['State']['ExitCode']!=0
if a.previous_resume:
 prior_resume=json.loads(subprocess.check_output(['docker','inspect',a.previous_resume]))[0]
 assert prior_resume['Id']==a.previous_resume and prior_resume['State']['Status']=='exited' and prior_resume['State']['ExitCode']!=0
reader=release/'tree/parents/jev-system-rl/implementation/pull_closed.py';digest=hashlib.sha256(reader.read_bytes()).hexdigest()
argv=list(intent['argv']);image=obj['Image'];idx=argv.index(image);argv[idx:idx]=['--mount','type=bind,src='+str(release)+',dst='+str(release)+',readonly','-e','OMNI_CLOSED_READER_PATH='+str(reader),'-e','OMNI_CLOSED_READER_SHA256='+digest]
assert not(a.cpu_closure_repair and a.closed_pull_repair)
if a.cpu_closure_repair or a.closed_pull_repair:
 from importlib.util import spec_from_file_location,module_from_spec
 checker=release/'tree/parents/jev-system-rl/implementation/release_completion.py'
 spec=spec_from_file_location('handoff_release_check',checker);m=module_from_spec(spec);spec.loader.exec_module(m);m.require_closed_release(release)
 delta=json.loads((release/'DELTA.json').read_text())
 prior=intent['identity']['release_ref'];assert delta['base_assembly_ref']==prior
 assert hashlib.sha256(Path(prior['path']).read_bytes()).hexdigest()==prior['sha256']
 assert {r['target'] for r in delta['changes']}=={'parents/jev-system-rl/implementation/'+n for n in (('transport.py','pull_closed.py') if a.closed_pull_repair else ('transport.py','resume_closed.py'))}
 config=handoff/'ITERATION.json';cfg=json.loads(config.read_text())
 cursor=json.loads((Path(cfg['output_root'])/'CURSOR.json').read_text())
 assert cursor['config_ref']['sha256']==hashlib.sha256(config.read_bytes()).hexdigest()
 stage=cursor['pending_stage']['stage']
 assert stage in ('endpoint','capture') and cursor['step']==0
 if stage=='capture':
  assert cursor['pending_stage']['purpose']=='base_parity'
  assert not a.existing_transport or a.closed_pull_repair,'ordinary closure repair is pre-dispatch only'
 if a.closed_pull_repair:assert a.existing_transport and stage=='capture'
 # A failed dependency preparation has not crossed the remote launch boundary.
 mailbox=Path(cfg['output_root'])/'mailbox'/f"{cursor['sequence']:04d}-{stage}"/'transport'
 assert (mailbox/'CLOSURE-CHECK.json').exists()
 if a.existing_transport:
  assert a.previous_resume
  plan=json.loads((mailbox/'TRANSPORT.json').read_text())
  request=plan['job']['task']['request_ref']
  assert hashlib.sha256(Path(request['path']).read_bytes()).hexdigest()==request['sha256']
  assert (mailbox/'REQUEST.json').exists() and (mailbox/'START-RESULT.json').exists()
  if a.closed_pull_repair:
   assert json.loads((mailbox/'START-RESULT.json').read_text())['returncode']==0 and (mailbox/'PULL-FAILURE.json').is_file()
 else:
  assert not (mailbox/'TRANSPORT.json').exists() and not (mailbox/'REQUEST.json').exists()
 original=Path(prior['path']).parent/'tree/parents/jev-system-rl/implementation'
 for i in range(len(argv)-1):
  if argv[i]=='-e' and argv[i+1].startswith('PYTHONPATH='):
   argv[i+1]='PYTHONPATH='+str(original)+':'+argv[i+1].split('=',1)[1]
 idx=argv.index(image)
 argv=argv[:idx+1]+['-B',str(release/'tree/parents/jev-system-rl/implementation/resume_closed.py'),'--config',str(config),'--release',str(release)]
name='omni-handoff-resume-'+hashlib.sha256(str(out).encode()).hexdigest()[:20];argv[argv.index('--name')+1]=name
out.mkdir(parents=True,exist_ok=True)
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
if (out/'INTENT.json').exists():
 assert json.loads((out/'INTENT.json').read_text())['argv']==argv
 actual=json.loads(subprocess.check_output(['docker','inspect',name]))[0];print(actual['Id'])
else:
 save(out/'INTENT.json',dict(argv=argv,original_client=obj['Id'],reader_sha256=digest,unchanged_iteration=str(handoff/'ITERATION.json')))
 cid=subprocess.check_output(argv,text=True).strip();save(out/'LAUNCH.json',dict(container_id=cid));print(cid)
