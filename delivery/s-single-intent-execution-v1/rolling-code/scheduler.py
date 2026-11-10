"""Detached four-lane consumer: immutable producer input; local durable state/events."""
import concurrent.futures,ctypes,fcntl,hashlib,json,os,select,shlex,subprocess,sys,tarfile,threading,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[3];O=R/'delivery/s-single-intent-execution-v1/rolling-state';P=R/'delivery/s-tfrl-rolling-v1';WINDOW=json.loads((R/'delivery/s-rolling-12h-v1/WINDOW.json').read_bytes());IMAGE='sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026'
HOSTS={'A':'npu2','B':'npu2','C':'npu','D':'npu'};ROOTS={'npu2':'/srv/omni-local-state/s-rolling-12h-v1','npu':'/srv/omni-local-state/s-rolling-12h-v1'};M='/workspace/omni';mutex=threading.RLock();wake=threading.Event();stage_locks={};prestage_inflight=set();prep_pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-'+str(threading.get_ident()))
 with tmp.open('w') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def read(p):return json.loads(Path(p).read_bytes())
def run(host,args,**kw):
 r=subprocess.run(['ssh','-o','ControlMaster=no','-o','ControlPath=none','-o','BatchMode=yes','-o','ServerAliveInterval=30',host,shlex.join(args)],**kw)
 if r.returncode:raise RuntimeError(str(args[:5])+': '+str(getattr(r,'stderr',None)))
 return r
def rootcmd(host,args,stdin=False):
 if host=='npu2':return ['sudo','-n']+args
 hot=ROOTS[host]
 return ['docker','run','--rm','--pull','never']+(['-i'] if stdin else [])+['--network','none','--mount',f'type=bind,src={hot},dst={hot}','--entrypoint',args[0],IMAGE]+args[1:]
def launch_command(host,unit,hot,lane,remote,spec):
 if host=='npu2':return ['sudo','-n','systemd-run','--unit='+unit,'--property=Type=exec','/usr/bin/python3',hot+'/host_driver.py','launch','--root',hot,'--lane',lane,'--spec',remote]
 rt=spec['runtime'];mounts={hot,rt['container_lane'],rt['serving_root'],str(Path(spec['predecessor']['terminal']).parent),spec['workspace']}
 cmd=['docker','run','-d','--pull','never','--name',unit,'--network','host','--pid','host','--mount','type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock','--mount','type=bind,src=/usr/bin/docker,dst=/usr/bin/docker,readonly']
 for path in sorted(mounts):cmd+=['--mount',f'type=bind,src={path},dst={path}']
 return cmd+['--entrypoint','/opt/saea-venv/bin/python',IMAGE,'-B',hot+'/host_driver.py','launch','--root',hot,'--lane',lane,'--spec',remote]
def remote_json(host,args):return json.loads(run(host,args,capture_output=True,text=True).stdout)
def tasks():
 if not (P/'READY-TASKS.json').exists():return []
 d=read(P/'READY-TASKS.json');return d if isinstance(d,list) else d.get('tasks',[])
def update(t,**kw):
 with mutex:
  s=O/'tasks'/ (t['task_id']+'.json');d=read(s) if s.exists() else dict(task=t,created_at=time.time());d.update(at=time.time(),**kw);save(s,d)
def normalized(t):
 t=dict(t);t['task_id']=t.get('task_id',t.get('id'));t['config']=t.get('config',t.get('native_config'));t['budget']=t.get('budget',t.get('budget_receipt'));return t

def import_ready():
 with mutex:
  for v in tasks():
   t=normalized(v)
   if not t.get('task_id') or not t.get('config') or not t.get('budget'):continue
   if str(t.get('status','ready')).lower() not in ('ready','pass_fixed_image_cpu','ready_for_execution'):continue
   if (O/'tasks'/(t['task_id']+'.json')).exists():continue
   for path,pin in t.get('pins',{}).items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==pin,'producer pin '+path
   cfg=read(R/t['config']);fit=read(R/t['budget']);assert fit['status']=='PASS_FIXED_IMAGE_CPU' and fit['manifest_sha256']==cfg['manifest_sha256'] and fit['rows']==cfg['expected_rows']
   update(t,status='ready',remaining_seconds_estimate=t.get('estimated_seconds',12000))

def lane_admission_open(lane):
 p=O.parent/'lane-policy.json'
 return read(p).get('lanes',{}).get(lane,{}).get('admit_new_tasks',True) if p.exists() else True

def dispatch_rank(task,lane):
 config=read(O.parent/'dispatch-priority.json') if (O.parent/'dispatch-priority.json').exists() else {}
 override=config.get('tasks',{}).get(task['task_id'],{});preferred=override.get('preferred_lane',task.get('preferred_lane'))
 return (preferred!=lane,override.get('priority',config.get('default_priority',10)),task['task_id'])

def prestage_ready():
 for lane in 'ABCD':
  if not lane_admission_open(lane):continue
  host=HOSTS[lane];ready=[read(p)['task'] for p in (O/'tasks').glob('*.json') if read(p)['status']=='ready' and lane in read(p)['task'].get('allowed_lanes',list('ABCD'))]
  if not ready:continue
  t=sorted(ready,key=lambda t:dispatch_rank(t,lane))[0];key=(t['task_id'],host)
  if key in prestage_inflight or (O/'stage'/(t['task_id']+'-'+host+'.json')).exists():continue
  prestage_inflight.add(key)
  def prepare(t=t,host=host,key=key):
   try:stage(t,host)
   except Exception as e:save(O/'stage'/(t['task_id']+'-'+host+'-ERROR.json'),dict(at=time.time(),error=str(e)))
   finally:prestage_inflight.discard(key)
  prep_pool.submit(prepare)

def stage(t,host):
 key=(t['task_id'],host)
 with mutex:lock=stage_locks.setdefault(key,threading.Lock())
 with lock:return stage_once(t,host)
def stage_once(t,host):
 hot=ROOTS[host];job=hot+'/jobs/'+t['task_id'];w=job+'/workspace';receipt=O/'stage'/(t['task_id']+'-'+host+'.json')
 if receipt.exists():return read(receipt)['workspace']
 cfg=read(R/t['config']);prefix=Path('delivery/s-tfrl-rolling-v1');prepared=Path(cfg['prepared']);files=set()
 # Candidate payloads and fixed budget are staged before any lane reservation.
 for d in [prepared,prefix/'code',prefix/'config']:
  if (R/d).exists():files.update(p for p in (R/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 files.add(R/t['config']);files.add(R/t['budget'])
 files.update((R/'delivery/s-single-intent-fewshot-v1/code'/n) for n in ['common.py'])
 files.update((R/'delivery/s-single-intent-execution-v1/rolling-code'/n) for n in ['train_runtime.py','bind_train.py'])
 files.update(p for p in (R/'delivery/s-single-intent-execution-v1/rolling-code/train-runtime-v1').glob('*.py'))
 source=t.get('workspace_files')
 if source:
  entries=read(R/source).get('files',[])
  if isinstance(entries,dict):entries=[dict(path=k,**v) if isinstance(v,dict) else dict(path=k,sha256=v) for k,v in entries.items()]
  for e in entries:
   path=R/e['path'];assert path.resolve().is_relative_to(R.resolve())
   if e.get('sha256'):assert hashlib.sha256(path.read_bytes()).hexdigest()==e['sha256'],'workspace source pin '+e['path']
   files.add(path)
 archive=O/'stage'/(t['task_id']+'.tar');archive.parent.mkdir(exist_ok=True)
 with mutex:
  if not archive.exists():
   with tarfile.open(archive,'w') as tf:
    for p in sorted(files):tf.add(p,arcname=str(p.relative_to(R)),recursive=False)
 base=hot+'/base-workspace'
 run(host,['mkdir','-p',job])
 run(host,rootcmd(host,['cp','-al',base,w]))
 # Unlink each overlaid file before extraction: preserve base and prior job inodes.
 with archive.open('rb') as f:run(host,rootcmd(host,['tar','--unlink-first','-xf','-','-C',w],stdin=True),stdin=f,capture_output=True)
 save(receipt,dict(workspace=w,at=time.time(),files=len(files),archive_bytes=archive.stat().st_size));return w

def snapshot(lane,launch):
 host=HOSTS[lane];hot=ROOTS[host];state=launch['state']
 # Run read-only observation through independent fixed-image container.
 args=['docker','run','--rm','--pull','never','--network','host','--mount',f'type=bind,src={hot},dst={hot}','--mount',f'type=bind,src={state},dst={state}','--entrypoint','/opt/saea-venv/bin/python',IMAGE,'-B',hot+'/host_driver.py','snapshot','--state',state]
 d=remote_json(host,args);save(O/'lanes'/(lane+'-LIVE.json'),d);save(P/'REQUESTS'/('ERROR-SUMMARY-'+lane+'.json'),d);return d

def observation_due(now,last,terminal_event,event_checked):
 return now-last>=600 or (terminal_event and not event_checked)
def wait_container(lane,launch):
 host=HOSTS[lane];cid=launch['container_id'];state=launch['state'];event=threading.Event();result={}
 def waiter():
  try:
   while True:
    try:r=run(host,['docker','wait',cid],capture_output=True,text=True);result['exit_code']=int(r.stdout.strip());break
    except Exception as e:
     save(O/'lanes'/(lane+'-connection-retry.json'),dict(at=time.time(),read_only_retry=True,error=str(e),model_resend=False))
     try:
      obj=remote_json(host,['docker','inspect',cid])[0];boot=run(host,['cat','/proc/sys/kernel/random/boot_id'],capture_output=True,text=True).stdout.strip()
      expected=launch.get('runtime',read(O/'initial-lanes.json')[lane].get('runtime',{})).get('boot_id');assert obj['Id']==cid and (expected is None or expected==boot),'boot/container changed; reconcile required'
      if not obj['State']['Running']:result['exit_code']=obj['State']['ExitCode'];break
     except AssertionError:raise
     except Exception:pass
     time.sleep(30)
  except Exception as e:result['error']=str(e)
  event.set();wake.set()
 threading.Thread(target=waiter,daemon=True).start()
 last=0;event_checked=False
 while True:
  if time.time()>=WINDOW['end_epoch'] and not result.get('stop_sent'):
   code='from pathlib import Path; import os,time; p=Path('+repr(state+'/run/STOP-ADMISSION')+'); f=p.open("a"); f.write(str(time.time())); f.flush(); os.fsync(f.fileno()); f.close()'
   run(host,(['sudo','-n','/usr/bin/python3','-c',code] if host=='npu2' else ['docker','run','--rm','--pull','never','--network','none','--mount',f'type=bind,src={state},dst={state}','--entrypoint','/opt/saea-venv/bin/python',IMAGE,'-c',code]),capture_output=True);result['stop_sent']=True
  if observation_due(time.time(),last,event.is_set(),event_checked):
   try:d=snapshot(lane,launch)
   except Exception as e:save(O/'lanes'/(lane+'-snapshot-error.json'),dict(at=time.time(),error=str(e)));d={}
   last=time.time();event_checked=event.is_set()
  if event.is_set():
   if 'error' in result:raise RuntimeError('UNKNOWN terminal connection: '+result['error'])
   term=d.get('terminal')
   if term is None:
    time.sleep(30);continue
   # A known terminal with unknown contacts is still a terminal observation.
   # Return it to the lane failure branch, which preserves the attempt and
   # selects an independent job. Raising here bypassed that branch and stopped
   # the worker. This does not authorize replay of any unknown request.
   return term,result
  if event.is_set():time.sleep(30)
  else:event.wait(10)

def score(lane,launch,cpu_attempt=1):
 host=HOSTS[lane];hot=ROOTS[host];state=launch['state'];w=launch['workspace'];name='omni-s12-score-'+hashlib.sha256(launch['container_id'].encode()).hexdigest()[:12]
 history_c=str(Path(state).parent/'C') if Path(state).parent.name=='jobs' else None
 try:
  argv=['docker','run','--pull','never','--name',name+('-cpu'+str(cpu_attempt) if cpu_attempt>1 else ''),'--network','none','--cpus','3','--memory','8g','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-e','OMNI_TYPED_SIDECAR=/typed-sidecar','--mount',f'type=bind,src={w},dst={M},readonly','--mount',f'type=bind,src={hot},dst={hot}','--mount',f'type=bind,src={state},dst={state}','--mount',f'type=bind,src={hot}/typed-sidecar,dst=/typed-sidecar,readonly','--entrypoint','/opt/saea-venv/bin/python',IMAGE,'-B',hot+'/score_rolling.py','--state',state,'--workspace',M]
  if history_c:
   pos=argv.index('--entrypoint');argv[pos:pos]=['--mount',f'type=bind,src={history_c},dst={history_c},readonly']
  r=run(host,argv,capture_output=True,text=True);save(O/'scores'/(name+'.json'),dict(at=time.time(),stdout=r.stdout,stderr=r.stderr,status='completed'))
 except Exception as e:save(O/'scores'/(name+'.json'),dict(at=time.time(),error=str(e),status='failed'))
 # Archive is independent of lane reuse, including failed score evidence.
 try:
  out=O/'archives'/(name+'.tar');out.parent.mkdir(exist_ok=True)
  with out.open('wb') as f:run(host,['docker','run','--rm','--pull','never','--network','none','--mount',f'type=bind,src={state},dst=/evidence,readonly','--entrypoint','tar',IMAGE,'-C','/evidence','--exclude=workspace','-cf','-','.'],stdout=f,stderr=subprocess.PIPE);f.flush();os.fsync(f.fileno())
  save(out.with_suffix('.json'),dict(at=time.time(),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
 except Exception as e:save(O/'scores'/(name+'-archive-error.json'),dict(error=str(e),at=time.time()))

def choose(lane):
 if not lane_admission_open(lane):return None
 with mutex:
  states=[read(p) for p in (O/'tasks').glob('*.json')];eligible=[]
  for s in states:
   t=s['task'];pref=t.get('preferred_lane');allowed=t.get('allowed_lanes',list('ABCD'))
   if s['status']!='ready' or lane not in allowed:continue
   if pref in ('A','B') and pref!=lane and not (O/'lanes'/(pref+'-FIRST-LAUNCH.json')).exists():continue
   deps=t.get('dependencies',[])
   if any(not (O/'tasks'/(dep+'.json')).exists() or read(O/'tasks'/(dep+'.json'))['status']!='completed' for dep in deps):continue
   eligible.append((*dispatch_rank(t,lane),t))
  if not eligible:return None
  # Admit only after CPU staging; unstaged fallback is prepared asynchronously.
  staged=[e for e in eligible if (O/'stage'/(e[-1]['task_id']+'-'+HOSTS[lane]+'.json')).exists()]
  if not staged:return None
  t=sorted(staged,key=lambda x:x[:-1])[0][-1];update(t,status='preparing',lane=lane);return t

def failed_handoff(launch,terminal):
 # Only an actual known failed terminal can authorize the host helper.
 if terminal.get('status') not in ('FAILED','PAUSED'):
  raise RuntimeError('BLOCKED_TERMINAL_HANDOFF: no known failed terminal; no resend')
 for key in ('task_id','container_id','state','run_id','attempt_id','pid','runtime'):
  if key not in launch:raise RuntimeError('BLOCKED_TERMINAL_HANDOFF: missing failed launch '+key)
 return dict(launch=launch,terminal=terminal,no_resend=True)


def lane_worker(lane):
 host=HOSTS[lane];hot=ROOTS[host];info=read(O/'initial-lanes.json')[lane];pred=info['predecessor'];launch=info.get('live_launch');runtime=info.get('runtime');runtime_file=info.get('runtime_file');last_task=None;failed_evidence=None
 current=O/'lanes'/(lane+'-CURRENT.json')
 try:
  if current.exists():
   previous=read(current)
   if previous.get('status')=='running':launch=previous['launch']
   elif previous.get('status')=='failed':
    failed_evidence=failed_handoff(previous['launch'],previous['terminal']);launch=None
    failed=failed_evidence['launch'];pred=dict(container=failed['container_id'],terminal=failed['state']+'/run/TERMINAL.json')
    runtime=failed['runtime'];runtime_file=None
   elif previous.get('status')=='completed':
    completed=previous['launch'];launch=None;pred=dict(container=completed['container_id'],terminal=completed['state']+'/run/TERMINAL.json')
    runtime=completed.get('runtime',runtime);runtime_file=None
  if launch:
   last_task=read(O/'tasks'/(launch['task_id']+'.json'))['task'] if launch.get('task_id') else None
   save(O/'lanes'/(lane+'-CURRENT.json'),dict(status='running',launch=launch,preserved=True))
   term,res=wait_container(lane,launch)
   if term['status']=='PAUSED' and time.time()>=WINDOW['end_epoch']:
    save(O/'lanes'/(lane+'-DRAINED.json'),dict(at=time.time(),status='deadline drained',terminal=term));return
   elif term['status']!='COMPLETE' or term.get('unknown',0)!=0:
    if launch.get('task_id'):
     old=read(O/'tasks'/(launch['task_id']+'.json'))['task']
     update(old,status='failed',reason='KNOWN_TERMINAL_FAILURE',terminal=term)
    save(O/'lanes'/(lane+'-FAILED-PRESERVED.json'),dict(at=time.time(),lane=lane,task_id=launch.get('task_id'),reason='KNOWN_TERMINAL_FAILURE',terminal=term,launch=launch,no_resend=True))
    failed_evidence=failed_handoff(launch,term)
    pred=dict(container=launch['container_id'],terminal=launch['state']+'/run/TERMINAL.json');runtime=launch['runtime'];runtime_file=None
    save(O/'lanes'/(lane+'-CURRENT.json'),dict(status='failed',launch=launch,terminal=term,preserved_failure=True,no_resend=True))
    launch=None;last_task=None
   else:
    if res['exit_code']!=0:raise RuntimeError('BLOCKED_TERMINAL_HANDOFF: COMPLETE with failed exit')
    if term['status']=='PAUSED' and time.time()>=WINDOW['end_epoch']:
     save(O/'lanes'/(lane+'-DRAINED.json'),dict(at=time.time(),status='deadline drained',terminal=term));return
    pred=dict(container=launch['container_id'],terminal=launch['state']+'/run/TERMINAL.json')
    if not launch.get('task_id'):
     threading.Thread(target=score,args=(lane,launch),daemon=False).start()
    if launch.get('task_id'):
     t=read(O/'tasks'/(launch['task_id']+'.json'))['task'];update(t,status='completed',terminal=term,exit=res);threading.Thread(target=score,args=(lane,launch),daemon=False).start();runtime=launch['runtime'];runtime_file=None
    save(O/'lanes'/(lane+'-CURRENT.json'),dict(status='completed',launch=launch,terminal=term));last_task=None
  while time.time()<WINDOW['end_epoch']:
   if not lane_admission_open(lane):
    save(O/'lanes'/(lane+'-DRAINED.json'),dict(at=time.time(),status='user_requested_drain',last_launch=launch));return
   if runtime_file:
    try:runtime=remote_json(host,['cat',runtime_file])
    except Exception:time.sleep(5);continue
   t=choose(lane)
   if t is None:time.sleep(2);continue
   last_task=t
   try:
    w=stage(t,host);spec=dict(t,workspace=w,runtime=runtime,cards=list(range(0,4) if lane in 'AC' else range(4,8)),predecessor=pred,window_end_epoch=WINDOW['end_epoch'])
    if failed_evidence:spec['failed_predecessor_evidence']=failed_evidence
    local=O/'specs'/(t['task_id']+'.json');save(local,spec);remote=hot+'/spec-'+t['task_id']+'.json';subprocess.run(['scp',str(local),host+':'+remote],check=True,capture_output=True)
    # Mutation is a remote systemd unit, so an SSH disconnect cannot abort admission.
    unit='omni-s12-launch-'+hashlib.sha256(t['task_id'].encode()).hexdigest()[:12]
    run(host,launch_command(host,unit,hot,lane,remote,spec),capture_output=True,text=True)
    launchfile=hot+'/jobs/'+t['task_id']+'/LAUNCH.json';deadline=time.time()+180
    while True:
     try:launch=remote_json(host,['cat',launchfile]);break
     except Exception:
      state=run(host,(['systemctl','show',unit,'-p','ActiveState','-p','Result'] if host=='npu2' else ['docker','inspect','--format','{{.State.Status}} {{.State.ExitCode}}',unit]),capture_output=True,text=True).stdout
      if 'ActiveState=failed' in state or ('exited' in state and not state.strip().endswith(' 0')) or time.time()>deadline:raise RuntimeError('launch unit failed or unresolved; inspect '+unit+' '+state)
      time.sleep(2)
    update(t,status='running',launch=launch,lane=lane);save(O/'lanes'/(lane+'-CURRENT.json'),dict(status='running',launch=launch));save(O/'lanes'/(lane+'-FIRST-LAUNCH.json'),launch)
    term,res=wait_container(lane,launch);update(t,status='completed' if term['status']=='COMPLETE' and term.get('unknown',0)==0 and res['exit_code']==0 else 'failed',terminal=term,exit=res)
    threading.Thread(target=score,args=(lane,launch),daemon=False).start()
    if term['status']=='PAUSED' and time.time()>=WINDOW['end_epoch']:
     update(t,status='blocked',reason='deadline admission closed; absent contacts were not sent',terminal=term);save(O/'lanes'/(lane+'-DRAINED.json'),dict(at=time.time(),status='deadline drained',terminal=term));return
    if term['status']!='COMPLETE' or term.get('unknown',0)!=0:
     # Keep the failed attempt and move on to a new independent task.  The
     # failed request set is never replayed and its evidence remains durable.
     save(O/'lanes'/(lane+'-FAILED.json'),dict(at=time.time(),lane=lane,task_id=t['task_id'],reason='KNOWN_TERMINAL_FAILURE',terminal=term,launch=launch,no_resend=True))
     failed_evidence=failed_handoff(launch,term)
     pred=dict(container=launch['container_id'],terminal=launch['state']+'/run/TERMINAL.json');runtime=launch['runtime'];runtime_file=None;last_task=None
     save(O/'lanes'/(lane+'-CURRENT.json'),dict(status='failed',launch=launch,terminal=term,no_resend=True))
     continue
    if res['exit_code']!=0:raise RuntimeError('BLOCKED_TERMINAL_HANDOFF: COMPLETE with failed exit')
    failed_evidence=None
    save(O/'lanes'/(lane+'-CURRENT.json'),dict(status='completed',launch=launch,terminal=term))
    pred=dict(container=launch['container_id'],terminal=launch['state']+'/run/TERMINAL.json');runtime=launch['runtime'];runtime_file=None;last_task=None
   except Exception as e:
    update(t,status='blocked',reason=str(e),traceback=traceback.format_exc());save(O/'lanes'/(lane+'-BLOCKED.json'),dict(at=time.time(),error=str(e)));wake.set();return
  save(O/'lanes'/(lane+'-DRAINED.json'),dict(at=time.time(),status='admission_closed_drained',deadline=WINDOW['end_epoch']))
 except Exception as e:save(O/'lanes'/(lane+'-BLOCKED.json'),dict(at=time.time(),error=str(e),traceback=traceback.format_exc()));wake.set()

def reserve():
 states=[read(p) for p in (O/'tasks').glob('*.json')];available=[s for s in states if s['status']=='ready'];running=[s for s in states if s['status']=='running'];left=max(0,WINDOW['end_epoch']-time.time());n=len(available)
 policy=read(O.parent/'dispatch-priority.json') if (O.parent/'dispatch-priority.json').exists() else {}
 def effective_lane(s):
  t=s['task'];return policy.get('tasks',{}).get(t['task_id'],{}).get('preferred_lane',t.get('preferred_lane'))
 plan=read(P/'PRODUCTION-PLAN.json') if (P/'PRODUCTION-PLAN.json').exists() else {'tasks':[]}
 published={t['task_id'] for t in tasks()};cpu=[t for t in plan['tasks'] if t['task_id'] not in published and (R/t['config']).exists() and (R/t['prepared']/'CPU-PREPARED.json').exists() and not (P/'evidence'/t['task_id']/'PRODUCER-FAILED.json').exists() and not (P/'evidence'/t['task_id']/'CONTINUATION-FAILED.json').exists() and t.get('status','') not in ('blocked','rejected','pending_scientific_evidence')]
 unmaterialized=[t['task_id'] for t in plan['tasks'] if t['task_id'] not in published and not (R/t['config']).exists()]
 cpu_by={l:sum(effective_lane({'task':t})==l for t in cpu) for l in 'ABCD'}
 assigned={l:sum(effective_lane(s)==l for s in available) for l in 'ABCD'}
 lanes={};residual=0;adopted=0
 for l in 'ABCD':
  live=read(O/'lanes'/(l+'-LIVE.json')) if (O/'lanes'/(l+'-LIVE.json')).exists() else {}
  current=read(O/'lanes'/(l+'-CURRENT.json')) if (O/'lanes'/(l+'-CURRENT.json')).exists() else {}
  active=current.get('status')=='running' and not live.get('terminal')
  if active and current.get('preserved') and not current.get('launch',{}).get('task_id'):adopted+=1
  eta=live.get('remaining_seconds_estimate')
  if active:residual+=(eta if eta is not None else 11000)
  lanes[l]=dict(ready_reserved=assigned[l],ready_eligible=sum(l in s['task'].get('allowed_lanes',list('ABCD')) for s in available),cpu_preparing=cpu_by[l],running=active,completed_branches=live.get('completed_branches'),total_branches=live.get('total_branches'),seconds_per_branch=live.get('seconds_per_branch'),remaining_seconds_estimate=eta)
 branches=sum(int(s['task'].get('branches',read(R/s['task']['config'])['expected_rows'])) for s in available)
 measured=[x['seconds_per_branch'] for x in lanes.values() if x.get('seconds_per_branch')]
 fast=min(measured) if measured else 3.0;slow=max(measured) if measured else 5.5
 capacity_ready=[s for s in available if s['task'].get('capacity_estimate_eligible',True)]
 estimated_branches=sum(int(s['task'].get('branches',read(R/s['task']['config'])['expected_rows'])) for s in capacity_ready)
 unknown_eta_task_ids=[s['task']['task_id'] for s in available if not s['task'].get('capacity_estimate_eligible',True)]
 pending=estimated_branches*fast/3600
 for l in lanes:
  own=sum(int(s['task'].get('branches',read(R/s['task']['config'])['expected_rows'])) for s in capacity_ready if effective_lane(s)==l)
  lanes[l]['reserved_lane_hours_fast']=own*fast/3600
  lanes[l]['reserved_lane_hours_slow']=own*slow/3600
  lanes[l]['next_staged']=any((O/'stage'/(s['task']['task_id']+'-'+HOSTS[l]+'.json')).exists() for s in capacity_ready if effective_lane(s)==l)
 conditional=read(P/'SVALID-CPU-RESERVE.json').get('tasks',[]) if (P/'SVALID-CPU-RESERVE.json').exists() else []
 conditional=[t for t in conditional if t['task_id'] not in published]
 # A producer plan may contain a future/unmaterialized candidate. It is
 # reported separately and must never make the live reserve pass crash.
 cpu_hours=0.0
 for t in cpu:
  cfg=R/t['config']
  if t.get('estimated_seconds') is not None:
   cpu_hours += float(t['estimated_seconds'])/3600
  elif cfg.exists():
   cpu_hours += float(read(cfg).get('expected_rows',0)*4)/3600
 d=dict(conditional_cpu_verified=len(conditional),conditional_cpu_verified_branches=sum(read(R/t['config'])['expected_rows'] for t in conditional),at=time.time(),window=WINDOW,remaining_window_hours=left/3600,ready_tasks=n,running_tasks=len(running),adopted_running_tasks=adopted,total_running=len(running)+adopted,cpu_preparing=len(cpu),estimated_ready_lane_hours=pending,estimated_running_residual_lane_hours=residual/3600,estimated_queue_wall_hours=(pending+residual/3600)/4,estimated_CPU_pending_lane_hours=cpu_hours,unmaterialized_plan_only=unmaterialized,ready_branches=branches,seconds_per_branch_range=[fast,slow],unknown_eta_task_ids=unknown_eta_task_ids,capacity_estimate_branches=estimated_branches,estimated_ready_lane_hours_range=[estimated_branches*fast/3600,estimated_branches*slow/3600],estimate_basis='ready expected_rows * measured fastest/slowest lane seconds_per_branch; cross-policy estimate not guarantee; running residual measured',lanes=lanes,minimum_simultaneous_ready_reserve=min(assigned.values()),minimum_cpu_reserve=min(cpu_by.values()),required_ready=4,required_cpu=4,coverage_12h='NOT_YET_PROVEN',money=None)
 save(O/'RESERVE.json',d)
 if n<4 or min(assigned.values())<1 or min(cpu_by.values())<1 or pending+residual/3600<left*4/3600:
  save(P/'REQUESTS/QUEUE-DEFICIT.json',dict(**d,request='补齐每lane至少1 ready+1CPU预备；优先未处理已有正确P真实TRAIN与根因策略；既定8seed科学组不扩组填时长。不得重采P或重发unknown。'))
 return d

def inotify():
 libc=ctypes.CDLL(None);fd=libc.inotify_init1(0x800|0x80000)
 libc.inotify_add_watch(fd,os.fsencode(P),0x80|0x100)
 while True:
  if select.select([fd],[],[],5)[0]:
   raw=os.read(fd,65536)
   if b'READY-TASKS.json' in raw:wake.set()

def main():
 O.mkdir(exist_ok=True)
 for d in ['tasks','lanes','stage','specs','scores','archives']:(O/d).mkdir(exist_ok=True)
 lock=(O/'owner.guard').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (O/'DRIVER-START.json').exists():
  old=read(O/'DRIVER-START.json')
  try:os.kill(old['pid'],0)
  except ProcessLookupError:pass
  else:raise RuntimeError('old driver alive')
  history=O/'driver-history'/str(time.time_ns());history.mkdir(parents=True);os.rename(O/'DRIVER-START.json',history/'DRIVER-START.json');save(history/'HANDOVER.json',dict(at=time.time(),old_pid=old['pid'],old_exited=True,adopt_current_containers=True,no_resend=True))
 save(O/'DRIVER-START.json',dict(pid=os.getpid(),ppid=os.getppid(),at=time.time(),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),window=WINDOW,monitor_seconds=600))
 threading.Thread(target=inotify,daemon=True).start()
 workers=[threading.Thread(target=lane_worker,args=(l,),name='lane-'+l) for l in 'ABCD']
 for w in workers:w.start()
 last=0
 while any(w.is_alive() for w in workers):
  try:import_ready();prestage_ready();d=reserve();save(O/'HEARTBEAT.json',dict(at=time.time(),pid=os.getpid(),workers={w.name:w.is_alive() for w in workers},ready=d['ready_tasks']))
  except Exception as e:
   # Keep the scheduler alive on one stale/unmaterialized CPU plan entry;
   # producer state remains evidence and the four NPU workers continue.
   save(O/'DRIVER-ERROR.json',dict(at=time.time(),error=str(e),traceback=traceback.format_exc(),nonfatal=True))
   try:
    save(O/'HEARTBEAT.json',dict(at=time.time(),pid=os.getpid(),workers={w.name:w.is_alive() for w in workers},ready=len([p for p in (O/'tasks').glob('*.json') if read(p).get('status')=='ready']),reserve_error=str(e)))
   except Exception:pass
  wake.wait(600);wake.clear()
 for w in workers:w.join()
 save(O/'DRIVER-DRAINED.json',dict(at=time.time(),pid=os.getpid(),status='workers finished; inspect blocked or drained per lane'))
if __name__=='__main__':main()
