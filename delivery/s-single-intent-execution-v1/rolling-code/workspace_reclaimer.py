"""Reclaim only reproducible staged workspaces after durable scored archive custody."""
import fcntl,hashlib,importlib.util,json,os,shlex,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parents[1];O=B/'rolling-state'
sp=importlib.util.spec_from_file_location('queue_custody',B/'rolling-code/scheduler.py')
s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
lock=(O/'workspace-reclaimer.guard').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
s.save(O/'WORKSPACE-RECLAIMER-START.json',dict(pid=os.getpid(),ppid=os.getppid(),at=time.time(),interval=600,evidence_deleted=False))
while True:
 try:
  active={s.read(p)['launch']['workspace'] for p in (O/'lanes').glob('*-CURRENT.json') if s.read(p).get('status')=='running'}
  for path in (O/'tasks').glob('*.json'):
   d=s.read(path)
   if d.get('status')!='completed' or not d.get('launch'):continue
   launch=d['launch'];workspace=launch['workspace'];tid=d['task']['task_id'];lane=d['lane'];host=s.HOSTS[lane]
   marker=O/'reclaimed'/(tid+'.json')
   if marker.exists() or workspace in active:continue
   term=d['terminal']
   assert term['status']=='COMPLETE' and term['unknown']==0
   name='omni-s12-score-'+hashlib.sha256(launch['container_id'].encode()).hexdigest()[:12]
   proof=O/'archives'/(name+'.json');archive=proof.with_suffix('.tar');score=O/'scores'/(name+'.json')
   if not proof.exists() or not archive.exists() or not score.exists() or s.read(score).get('status')!='completed':continue
   receipt=s.read(proof)
   assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
   source=O/'stage'/(tid+'.tar')
   if not source.exists() and d['task'].get('recovery_of'):source=O/'stage'/(d['task']['recovery_of']+'.tar')
   if not source.exists():continue
   target=Path(workspace);root=Path(s.ROOTS[host])/'jobs'
   assert target.is_relative_to(root) and target.name=='workspace' and len(target.relative_to(root).parts)==2
   # Re-read immediately before project-scoped removal; retain all journals, guards,
   # responses, configs, launch receipts, and immutable local preparation archive.
   if any(s.read(p).get('status')=='running' and s.read(p)['launch']['workspace']==workspace for p in (O/'lanes').glob('*-CURRENT.json')):continue
   s.run(host,s.rootcmd(host,['rm','-rf','--',workspace]),capture_output=True)
   s.save(marker,dict(at=time.time(),workspace=workspace,host=host,archive=receipt,source_archive=str(source),source_bytes=source.stat().st_size,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),evidence_deleted=False,terminal=term))
  s.save(O/'WORKSPACE-RECLAIMER-HEARTBEAT.json',dict(at=time.time(),pid=os.getpid(),active_workspaces=sorted(active)))
 except Exception as e:s.save(O/'WORKSPACE-RECLAIMER-ERROR.json',dict(at=time.time(),error=str(e),model_calls=0))
 if time.time()>s.WINDOW['end_epoch']+7200:break
 time.sleep(600)
