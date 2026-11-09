"""Wait for all eight known-terminal seed archives; emit actual ACE group feedback."""
import fcntl,hashlib,io,json,os,shlex,subprocess,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parents[3];O=R/'delivery/s-single-intent-execution-v1/rolling-state';P=R/'delivery/s-tfrl-rolling-v1';I='sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026';H='/srv/omni-local-state/s-rolling-12h-v1';M='/workspace/omni'
def save(p,d):
 p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def read(p):return json.loads(p.read_bytes())
def ssh(args,**kw):return subprocess.run(['ssh','-o','ControlMaster=no','-o','ControlPath=none','npu2',shlex.join(args)],check=True,**kw)
def resolve_member(seed,mapping,cfg):
 logical='grpo-fewshot-seed'+str(seed);tid=mapping.get(str(seed),logical)
 assert Path(tid).name==tid,'task id must be a basename'
 path=O/'tasks'/(tid+'.json')
 if not path.exists():return logical,tid,None
 d=read(path);task=d['task'];assert task['seed']==seed,'mapped seed mismatch'
 native=read(R/task['config']);sampling=read(R/native['prepared']/'SAMPLING.json')
 assert sampling['seed']==seed,'sampling seed mismatch'
 assert sampling['temperature']==cfg['temperature'] and sampling['top_p']==cfg['top_p'],'sampling settings changed'
 assert sampling['group_reward']==cfg['reward'],'reward changed'
 if tid!=logical:
  original=read(O/'tasks'/(logical+'.json'))['task']
  for field in ('config','manifest_sha256','parent_version','seed','pins'):
   assert task[field]==original[field],'retry changed '+field
 return logical,tid,d
lock=(O/'group-feedback.guard').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
save(O/'GROUP-COLLECTOR-START.json',dict(pid=os.getpid(),ppid=os.getppid(),at=time.time(),interval_seconds=600,no_model_calls=True))
while True:
 try:
  cfg=read(P/'config/grpo-v1.json');members=[];ready=True
  # A logical seed can have a separately named retry after a confirmed
  # pre-contact launcher failure.  Keep this mapping explicit and local to
  # the runtime collector: the frozen GRPO roster and original failure are
  # never relabelled.  A retry is accepted only if its own terminal/archive
  # evidence passes the same COMPLETE/unknown==0 checks below.
  mapping_path=O/'GROUP-MEMBER-MAP.json'
  mapping=read(mapping_path).get('logical_seed_to_task',{}) if mapping_path.exists() else {}
  for seed in cfg['seeds']:
   logical,tid,d=resolve_member(seed,mapping,cfg)
   if d is None or d['status']!='completed':ready=False;break
   assert d['terminal']['status']=='COMPLETE' and d['terminal']['unknown']==0
   launch=d['launch'];name='omni-s12-score-'+hashlib.sha256(launch['container_id'].encode()).hexdigest()[:12];archive=O/'archives'/(name+'.tar')
   if not archive.with_suffix('.json').exists():ready=False;break
   proof=read(archive.with_suffix('.json'));assert archive.stat().st_size==proof['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==proof['sha256'],'archive integrity'
   evidence=O/'group-inputs'/tid;evidence.mkdir(exist_ok=True,parents=True)
   with tarfile.open(archive) as tf:
    wanted={'run/TERMINAL.json','run/predictions.private.jsonl'}
    for m in tf.getmembers():
     rel=m.name.removeprefix('./')
     if rel in wanted:
      assert m.isfile() and not '..' in Path(rel).parts
      target=evidence/Path(rel).name
      if not target.exists():target.write_bytes(tf.extractfile(m).read())
   task=d['task'];folder=Path(read(R/task['config'])['prepared'])
   members.append(dict(seed=seed,logical_task_id=logical,task_id=tid,config=task['config'],sampling=str(folder/'SAMPLING.json'),terminal=M+'/'+str((evidence/'TERMINAL.json').relative_to(R)),predictions=M+'/'+str((evidence/'predictions.private.jsonl').relative_to(R))))
  if ready:
   assert len(members)==8
   spec=O/'group-inputs/MEMBERS.json';save(spec,dict(members=members,rule_frozen=cfg['reward'],at=time.time()))
   W=H+'/group-feedback-workspace';out=H+'/group-feedback'
   ssh(['sudo','-n','sh','-c','test -d '+shlex.quote(W)+' || cp -al '+shlex.quote(H+'/base-workspace')+' '+shlex.quote(W)]);ssh(['mkdir','-p',out])
   files=set()
   for d in [P/'code',P/'config',R/'delivery/s-typed-normalized-scoring-v1',O/'group-inputs']:
    files.update(p for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'results' not in p.parts)
   for member in members:
    native=read(R/member['config']);folder=R/native['prepared']
    files.update([R/member['config'],R/native['manifest'],R/native['samples'],folder/'SAMPLING.json'])
   archive=O/'group-inputs/GROUP-CPU-PACKAGE.tar'
   with tarfile.open(archive,'w') as tf:
    for p in sorted(files):tf.add(p,arcname=str(p.relative_to(R)),recursive=False)
   with archive.open('rb') as f:ssh(['sudo','-n','tar','--unlink-first','-xf','-','-C',W],stdin=f)
   argv=['docker','run','--pull','never','--name','omni-s12-grpo-feedback-v1','--network','none','--cpus','3','--memory','8g','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','--mount',f'type=bind,src={W},dst={M},readonly','--mount',f'type=bind,src={out},dst=/result','--entrypoint','/usr/local/python3.12.13/bin/python3',I,'-B','-c','import regex,runpy,sys; sys.argv='+repr([M+'/delivery/s-tfrl-rolling-v1/code/group_feedback.py','--members',M+'/'+str(spec.relative_to(R)),'--output','/result'])+'; runpy.run_path('+repr(M+'/delivery/s-tfrl-rolling-v1/code/group_feedback.py')+',run_name="__main__")']
   intent=O/'GROUP-FEEDBACK-INTENT.json'
   if intent.exists():
    prior=read(intent);name=prior['container_name']
    q=ssh(['docker','inspect',name],capture_output=True,text=True);obj=json.loads(q.stdout)[0]
    if obj['State']['Running']:ssh(['docker','wait',name],capture_output=True,text=True);obj=json.loads(ssh(['docker','inspect',name],capture_output=True,text=True).stdout)[0]
    if obj['State']['ExitCode']!=0:
     save(O/'GROUP-FEEDBACK-KNOWN-CPU-FAILURE.json',dict(at=time.time(),container=name,state=obj['State'],cpu_only=True))
     if prior.get('attempt',1)>=2:raise RuntimeError('known CPU failure retained after explicit recovery; inspect logs')
     history=O/'group-feedback-history'/str(time.time_ns());history.mkdir(parents=True);os.rename(intent,history/intent.name)
   if not intent.exists():
    attempt=2 if (O/'GROUP-FEEDBACK-KNOWN-CPU-FAILURE.json').exists() else 1;name='omni-s12-grpo-feedback-v'+str(attempt)
    argv[argv.index('--name')+1]=name;argv.insert(argv.index('--pull'),'-d')
    save(intent,dict(at=time.time(),argv=argv,members=members,attempt=attempt,container_name=name))
    r=ssh(argv,capture_output=True,text=True);save(O/'GROUP-FEEDBACK-LAUNCH.json',dict(at=time.time(),stdout=r.stdout,stderr=r.stderr))
    ssh(['docker','wait',name],capture_output=True,text=True)
    obj=json.loads(ssh(['docker','inspect',name],capture_output=True,text=True).stdout)[0];assert obj['State']['ExitCode']==0,'known CPU failure inspect preserved container'
   r=ssh(['docker','logs',name],capture_output=True,text=True);save(O/'GROUP-FEEDBACK-RECEIPT.json',dict(at=time.time(),returncode=0,container_name=name,stdout=r.stdout,stderr=r.stderr,reconciled=True))
   local=O/'group-feedback';local.mkdir(exist_ok=True)
   with (local/'result.tar').open('wb') as f:ssh(['docker','run','--rm','--pull','never','--network','none','--mount',f'type=bind,src={out},dst=/result,readonly','--entrypoint','tar',I,'-C','/result','-cf','-','.'],stdout=f)
   with tarfile.open(local/'result.tar') as tf:
    for m in tf.getmembers():assert not m.name.startswith('/') and '..' not in Path(m.name).parts and (m.isfile() or m.isdir())
    tf.extractall(local,filter='data')
   save(P/'REQUESTS/GRPO-FEEDBACK-READY.json',dict(at=time.time(),result_directory=str(local.relative_to(R)),reflector_input=str((local/'REFLECTOR-NEXT-INPUT.private.json').relative_to(R)),group_size=8,weights_updated=False))
   break
  save(O/'GROUP-COLLECTOR-WAITING.json',dict(at=time.time(),complete_seed_archives=len(members),required=8,status='waiting complete known group; no unknown resend'))
 except Exception as e:save(O/'GROUP-COLLECTOR-ERROR.json',dict(at=time.time(),error=str(e),no_model_resend=True))
 time.sleep(600)
