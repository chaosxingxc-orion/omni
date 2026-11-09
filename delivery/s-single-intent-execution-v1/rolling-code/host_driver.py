"""Local-disk lane admission, unchanged native runner; invoked by detached owner."""
import argparse,base64,collections,fcntl,hashlib,json,os,subprocess,time,urllib.request
from pathlib import Path
IMAGE='sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026'
M='/workspace/omni';PK='delivery/s-single-intent-fewshot-v1'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
 with tmp.open('w') as f:json.dump(d,f,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def pin(p):return dict(path=str(p),sha256=sha(p))
def inspect(n):return json.loads(subprocess.check_output(['docker','inspect',n]))[0]
def historic(p):
 # Independent fixed-image container reads root-owned historical evidence.
 p=str(p);v=json.loads(subprocess.check_output(['docker','run','--rm','--pull','never','--network','none','--mount',f'type=bind,src={Path(p).parent},dst=/evidence,readonly','--entrypoint','/opt/saea-venv/bin/python',IMAGE,'-c','import json; print(json.dumps(json.load(open('+repr('/evidence/'+Path(p).name)+'))))']))
 return v

def historical_pin(p):
 code='from pathlib import Path; import hashlib,json; p=Path('+repr('/evidence/'+Path(p).name)+'); print(json.dumps(dict(path='+repr(str(p))+',sha256=hashlib.sha256(p.read_bytes()).hexdigest())))'
 return json.loads(subprocess.check_output(['docker','run','--rm','--pull','never','--network','none','--mount',f'type=bind,src={Path(p).parent},dst=/evidence,readonly','--entrypoint','/opt/saea-venv/bin/python',IMAGE,'-c',code]))
def metrics(rt):
 raw=urllib.request.urlopen(rt['endpoint'].removesuffix('/v1')+'/metrics',timeout=10).read().decode()
 return {k:sum(float(l.rsplit(' ',1)[1]) for l in raw.splitlines() if l.startswith('vllm:num_requests_'+k+'{')) for k in ['running','waiting']}
def verify(rt,cards):
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==rt['boot_id'],'boot changed'
 s=inspect(rt['service_name']);assert s['Id']==rt['service_id'] and s['Image']==IMAGE and s['State']['Running'] and s['State']['Pid']==rt['service_pid'],'service changed'
 actual=sorted(int(d['PathOnHost'].rsplit('davinci',1)[1]) for d in s['HostConfig']['Devices'] if d['PathOnHost'].rsplit('davinci',1)[-1].isdigit());assert actual==cards
 assert os.stat(rt['shared_guard']).st_ino==rt['shared_guard_inode'];os.kill(rt['service_pid'],0)
 return s

def reset(a):
 H=Path(a.root);seed=read(H/'AB-runtime-seed.json')[a.lane];state=H/('service-'+a.lane);state.mkdir(exist_ok=True)
 if (state/'READY.json').exists():print((state/'READY.json').read_text());return
 guard=Path(seed['shared_guard'])
 with guard.open('r+b') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  s=verify(seed,list(range(0,4) if a.lane=='A' else range(4,8)));assert metrics(seed)==dict(running=0,waiting=0)
  assert not (guard.parent/'flight.lock').exists() and not (guard.parent/'engine-transition.pending.json').exists()
  pred=read(H/'initial-predecessors.json')[a.lane];p=historic(pred['terminal']);c=inspect(pred['container']);assert p['status']=='COMPLETE' and p['unknown']==0 and not c['State']['Running'] and c['State']['Pid']==0
  session=historic(Path(seed['serving_root'])/seed['session_id']/'session.json')
  save(state/'BEFORE.json',dict(service=s,predecessor=p,predecessor_container=c['State'],runtime=seed,at=time.time(),guard_inode=os.fstat(lock.fileno()).st_ino))
  if (state/'restart-intent.json').exists():raise RuntimeError('restart already attempted; reconcile manually')
  save(state/'restart-intent.json',dict(at=time.time(),service_id=s['Id'],reason='idle same-image reset to align first structured-output backend with C/D xgrammar'))
  r=subprocess.run(['docker','restart','--time','60',s['Id']],capture_output=True,text=True);save(state/'RESTART.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,at=time.time()));assert r.returncode==0
  s=inspect(s['Id']);seed['service_pid']=s['State']['Pid'];seed['session_id']='s-rolling-'+a.lane.lower()+'-'+str(s['State']['Pid']);seed['serving_root']=str(state/'serving');seed['endpoint']=seed['endpoint'].removesuffix('/v1');seed['container_lane']=str(guard.parent.parent)
  session.update(session_id=seed['session_id'],pid=seed['service_pid'],started_at=s['State']['StartedAt']);session['identity'].update(server_argv=s['Config']['Entrypoint']+s['Config']['Cmd'],port=int(seed['endpoint'].rsplit(':',1)[1]));save(Path(seed['serving_root'])/seed['session_id']/'session.json',session)
  save(state/'RESTARTED-RUNTIME.json',seed)
 deadline=time.time()+1800
 while time.time()<deadline:
  try:
   urllib.request.urlopen(seed['endpoint']+'/health',timeout=5).read();assert metrics(seed)==dict(running=0,waiting=0);save(state/'READY.json',seed);print(json.dumps(seed));return
  except Exception:time.sleep(5)
 raise RuntimeError('health timeout')

def launch(a):
 H=Path(a.root);task=read(a.spec);W=Path(task['workspace']);state=H/'jobs'/task['task_id'];state.mkdir(parents=True,exist_ok=True)
 lock=(state/'launcher.guard').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (state/'LAUNCH.json').exists():print((state/'LAUNCH.json').read_text());return
 assert time.time()<task['window_end_epoch'],'window ended'
 assert not (state/'launch-intent.json').exists(),'UNKNOWN_LAUNCH_NO_RESEND'
 rt=read(task['runtime_file']) if task.get('runtime_file') else task['runtime'];rt['container_lane']=str(Path(rt['shared_guard']).parent.parent)
 cfg=read(W/task['config']);original_cfg_pin=pin(W/task['config'])
 if cfg.get('dataset_partition') in ('S_fit','eligible_remainder') and (H/'TRAIN96-VERIFIED.json').exists():
  assert cfg['concurrency']==64 and cfg['dataset_partition'] in ('S_fit','eligible_remainder')
  cfg['concurrency']=96;cfg['attempt_id']+='-c96'
  fixed=read(H/'TRAIN96-VERIFIED.json')
  for rel,pinvalue in fixed['pins'].items():assert sha(H/'execution-code'/rel)==pinvalue,'fixed execution package pin'
  save(state/'EXECUTION-OVERRIDE.json',dict(at=time.time(),original_config=original_cfg_pin,original_concurrency=64,effective_concurrency=96,scope='new TRAIN control/fewshot only; frozen scientific packets unchanged',runner='delivery/s-single-intent-execution-v1/rolling-code/train_runtime.py',fixed_package=read(H/'TRAIN96-VERIFIED.json')))
  task['runner']='delivery/s-single-intent-execution-v1/rolling-code/train_runtime.py';task['bind_script']='delivery/s-single-intent-execution-v1/rolling-code/bind_train.py'
 if task.get('execution_attempt_suffix'):
  suffix=task['execution_attempt_suffix'];assert suffix.startswith('-') and suffix.replace('-','').isalnum();cfg['attempt_id']+=suffix
 budget=W/task['budget'];fit=read(budget);assert fit['status']=='PASS_FIXED_IMAGE_CPU' and fit['rows']==cfg['expected_rows'] and fit['manifest_sha256']==cfg['manifest_sha256']
 pred=task['predecessor'];old=inspect(pred['container']);assert not old['State']['Running'] and old['State']['Pid']==0,'predecessor live'
 term=historic(pred['terminal']);assert term['status']=='COMPLETE' and term['unknown']==0,'predecessor unknown/failed requires explicit reconciliation'
 guard=Path(rt['shared_guard'])
 with guard.open('r+b') as held:
  fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB);verify(rt,task['cards']);assert metrics(rt)==dict(running=0,waiting=0),'engine busy'
  assert not (guard.parent/'flight.lock').exists() and not (guard.parent/'engine-transition.pending.json').exists(),'unreconciled metadata'
  session=historic(Path(rt['serving_root'])/rt['session_id']/'session.json');assert session['session_id']==rt['session_id'] and session['pid']==rt['service_pid']
  save(state/'predecessor.json',dict(**term,container=old['Id'],container_state=old['State'],terminal=historical_pin(pred['terminal'])))
  rt.update(status='OWNER_RECONCILED',cards=4,TP=4,instances=1,context_length=65536,fixed_image_sha256=IMAGE,native_fit_receipt=pin(budget),predecessor_receipt=pin(state/'predecessor.json'))
  save(state/'RUNTIME-CAPTURE.json',dict(status='PASS_RUNTIME_CAPTURE_NOT_STARTED',released=True,unknown=0,**{k:rt[k] for k in ['boot_id','session_id','service_pid','service_id']},lane_root=rt['container_lane'],guard_inode=rt['shared_guard_inode'],predecessor=dict(receipt=rt['predecessor_receipt']),fit_sha256=sha(budget),scheduler=metrics(rt),at=time.time()))
  rt['capture_receipt']=pin(state/'RUNTIME-CAPTURE.json');cfg['runtime']=rt;save(state/'runtime-config.json',cfg)
 mounts=['--mount',f'type=bind,src={W},dst={M},readonly','--mount',f'type=bind,src={W},dst={W},readonly','--mount',f'type=bind,src={H},dst={H}','--mount',f'type=bind,src={rt["container_lane"]},dst={rt["container_lane"]}','--mount',f'type=bind,src={rt["serving_root"]},dst={rt["serving_root"]},readonly']
 if cfg.get('dataset_partition') in ('S_fit','eligible_remainder') and cfg['concurrency']!=64:mounts+=['--mount',f'type=bind,src={H}/execution-code,dst={M}/delivery/s-single-intent-execution-v1/rolling-code,readonly']
 common=['--pull','never','--pid','host','--cpus','8','--memory','16g','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1']+mounts+['--entrypoint','/opt/saea-venv/bin/python',IMAGE,'-B']
 name='omni-s12-'+a.lane.lower()+'-'+hashlib.sha256(task['task_id'].encode()).hexdigest()[:12]
 argv=['docker','run','--name',name+'-bind','--network','none']+common+[M+'/'+task.get('bind_script',PK+'/code/bind_execution.py'),'--config',str(state/'runtime-config.json'),'--state',str(state/'run'),'--output',str(state/'execution-config.json')]
 r=subprocess.run(argv,capture_output=True,text=True);save(state/'BIND.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,at=time.time()));assert r.returncode==0,r.stderr
 assert time.time()<task['window_end_epoch'],'window ended before model admission'
 argv=['docker','run','-d','--name',name,'--network','host']+common+[M+'/'+task.get('runner',PK+'/code/run.py'),'run','--workspace',M,'--config',str(state/'runtime-config.json'),'--output',str(state/'run'),'--execution-config',str(state/'execution-config.json')]
 save(state/'launch-intent.json',dict(at=time.time(),argv=argv,task_id=task['task_id']))
 r=subprocess.run(argv,capture_output=True,text=True);save(state/'LAUNCH-RESULT.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,at=time.time()));assert r.returncode==0,r.stderr
 s=inspect(r.stdout.strip());receipt=dict(at=time.time(),task_id=task['task_id'],lane=a.lane,container_id=s['Id'],pid=s['State']['Pid'],state=str(state),run_id=cfg['run_id'],attempt_id=cfg['attempt_id'],runtime=rt,concurrency=cfg['concurrency'],total=cfg['expected_rows'],cards=task['cards'],workspace=str(W));save(state/'LAUNCH.json',receipt);print(json.dumps(receipt))

def snapshot(a):
 state=Path(a.state);cfg=read(state/'runtime-config.json');j=state/'run/contact-journals'/cfg['run_id']/cfg['attempt_id'];rows=[read(p) for p in (state/'run/rows').glob('*.private.json')];usage=collections.Counter();finish=collections.Counter();errors=collections.Counter();sources=collections.Counter();sample_errors=[]
 for r in rows:
  sources[r['sample_id'].rsplit('#S-',1)[0]]+=1
  if not r['valid']:errors[str(r.get('error',r.get('outcome')))]+=1
  for k,v in r.get('usage',{}).items():
   if isinstance(v,(int,float)):usage[k]+=v
 for p in j.glob('*.response.json'):
  rec=read(p);obj=json.loads(base64.b64decode(rec['response_b64']));c=obj.get('choices',[{}])[0];finish[c.get('finish_reason','HTTP-error')]+=1
  if len(sample_errors)<12 and (c.get('finish_reason')=='length' or not c.get('message',{}).get('content')):sample_errors.append(dict(request_id=rec['request_id'],finish=c.get('finish_reason'),usage=obj.get('usage'),response_sha256=rec['response_sha256']))
 epoch=read(state/'execution-config.json')['original_launch_epoch'];elapsed=time.time()-epoch;term=read(state/'run/TERMINAL.json') if (state/'run/TERMINAL.json').exists() else None
 done=len(rows);n=cfg['expected_rows'];d=dict(at=time.time(),run_id=cfg['run_id'],attempt_id=cfg['attempt_id'],completed_branches=done,total_branches=n,completed_samples=sum(x==3 for x in sources.values()),total_samples=cfg.get('full_denominator',545),invalid=sum(not r['valid'] for r in rows),usage=dict(usage),finish_reasons=dict(finish),invalid_reasons=dict(errors),error_examples=sample_errors,elapsed_seconds=elapsed,seconds_per_branch=elapsed/done if done else None,remaining_seconds_estimate=elapsed/done*(n-done) if done>=64 else None,card_hours=elapsed*4/3600,terminal=term,inflight_or_unknown=len(list(j.glob('*.intent.json')))-len(list(j.glob('*.response.json'))),scheduler=metrics(cfg['runtime']))
 previous=read(state/'LIVE-ROLLING.json') if (state/'LIVE-ROLLING.json').exists() else None
 if previous and previous.get('run_id')==d['run_id'] and previous.get('attempt_id')==d['attempt_id']:
  dt=d['at']-previous['at'];delta=done-previous['completed_branches'];valid_delta=(done-d['invalid'])-(previous['completed_branches']-previous.get('invalid',0))
  if dt>=300 and delta>=0:
   d['interval']=dict(seconds=dt,completed_branches=delta,valid_branches=valid_delta,branches_per_second=delta/dt,valid_branches_per_second=valid_delta/dt,seconds_per_completed_branch=dt/delta if delta else None,prompt_tokens_per_second=(d['usage'].get('prompt_tokens',0)-previous.get('usage',{}).get('prompt_tokens',0))/dt,completion_tokens_per_second=(d['usage'].get('completion_tokens',0)-previous.get('usage',{}).get('completion_tokens',0))/dt,start_epoch=previous['at'],end_epoch=d['at'],note='same run adjacent observations; includes any work in this interval; not causal96 comparison')
 save(state/'LIVE-ROLLING.json',d);print(json.dumps(d))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['reset','launch','snapshot']);p.add_argument('--root');p.add_argument('--lane');p.add_argument('--spec');p.add_argument('--state');a=p.parse_args();globals()[a.mode](a)
