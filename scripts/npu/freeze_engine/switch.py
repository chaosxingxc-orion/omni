"""Recorded engine switch. Plan is read-only; apply/rollback require explicit command.

Only run on npu2. Keep stopped old containers under archival names, allowing exact
Docker-config rollback. Never edits scientific serving/session records.
"""
import argparse,concurrent.futures,fcntl,hashlib,json,os,re,socket,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).parent; D=Path('/mnt/cephfs/xingchao/omni')
RECORDS=D/'work/codex/frozen-engine-deployment'
OLD_PROFILES={f'saea-omni-i{i}':f'qwen3-omni-engine-screen-i{i}-s{seq}-b2048' for i,seq in enumerate((32,48,64,96))}
IMAGE='sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026'
LABEL='saea.deployment.profile'
def call(args):return subprocess.check_output(args,text=True).strip()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def inspect(name):
 r=subprocess.run(['docker','inspect',name],text=True,capture_output=True)
 return json.loads(r.stdout)[0] if r.returncode==0 else None
def validate_original(doc,candidate):
 if not doc or doc['Name']!='/'+candidate['name']:raise ValueError('original name mismatch')
 if doc['Config'].get('Labels',{}).get(LABEL)!=OLD_PROFILES[candidate['name']]:raise ValueError('foreign profile')
 if doc['Image']!=IMAGE:raise ValueError('original image differs')
 if not doc['State']['Running']:raise ValueError('original not running')
 if doc['HostConfig']['RestartPolicy']['Name'] not in ('no',''):raise ValueError('automatic restart configured')
def commands(smi=None):
 nic=next(x['ifname'] for x in json.loads(call(['ip','-j','-4','addr','show','scope','global'])) if x.get('addr_info'))
 env={**os.environ,'TREE':str(D/'publish/omni'),'CPUSET':'32-63','PORT_CONTAINER_USER':'0:0','PORT_IMAGE':IMAGE}
 args=['bash',str(D/'work/scripts/port_py.sh'),str(P/'prepare_commands.py'),'--nic',nic]
 if smi:args+=['--smi',str(smi)]
 return json.loads(subprocess.check_output(args,env=env,text=True))
def quiescent(candidates):
 for c in candidates:
  with urllib.request.urlopen('http://127.0.0.1:%d/metrics'%c['port'],timeout=10) as r:lines=r.read().decode().splitlines()
  for name in ('num_requests_running','num_requests_waiting'):
   values=[float(s.rsplit(' ',1)[1]) for s in lines if s.startswith('vllm:'+name+'{') or s.startswith('vllm:'+name+' ')]
   if not values or any(values):raise ValueError('busy or unreadable endpoint '+c['name'])
def wait_ready(candidates,timeout=1800):
 end=time.monotonic()+timeout
 while True:
  pending=[]
  for c in candidates:
   doc=inspect(c['name'])
   if not doc or not doc['State']['Running'] or doc['RestartCount']:raise RuntimeError('engine died '+c['name'])
   try:
    with urllib.request.urlopen('http://127.0.0.1:%d/health'%c['port'],timeout=5) as r:assert r.status==200
    with urllib.request.urlopen('http://127.0.0.1:%d/v1/models'%c['port'],timeout=5) as r:models=json.load(r)
    assert any(m['id']=='qwen3-omni-30b-a3b-thinking' for m in models['data'])
   except Exception:pending.append(c['name'])
  if not pending:return
  if time.monotonic()>end:raise TimeoutError('readiness '+str(pending))
  time.sleep(600)
def parallel(fn,items):
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(fn,items))
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','apply','rollback']);p.add_argument('--record',type=Path,required=True);a=p.parse_args()
 root=a.record.resolve()
 if not root.is_relative_to(RECORDS.resolve()):raise ValueError('record must stay under owned deployment records')
 if not re.fullmatch('[A-Za-z0-9_-]+',root.name):raise ValueError('plain record directory name required')
 if a.mode=='plan':
  if root.exists():raise ValueError('fresh plan directory required')
  cs=commands()
  if any(c['host']!=socket.gethostname() for c in cs):raise ValueError('wrong host')
  originals=[]
  for c in cs:
   d=inspect(c['name']);validate_original(d,c);originals.append(d)
  root.mkdir(parents=True);write(root/'plan.json',{'schema':'saea-engine-switch-v1','time':time.time(),'host':socket.gethostname(),'candidates':cs,'original_ids':{c['name']:d['Id'] for c,d in zip(cs,originals)},'old_names':{c['name']:c['name']+'-before-'+root.name for c in cs},'rollback':'stop/remove labelled candidates; rename exact archived original containers; docker start originals','scientific_session_records':'unchanged; must re-register after choosing baseline'})
  write(root/'original-docker-inspect.json',originals);print(json.dumps({'planned':str(root),'models_called':0}));return
 plan=read(root/'plan.json');cs=plan['candidates']
 if plan['host']!=socket.gethostname():raise ValueError('wrong host')
 with (D/'locks/generation.guard').open('r+') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (D/'locks/flight.lock').exists():raise ValueError('scientific generation lease exists')
  if a.mode=='apply':
   if (root/'apply.intent.json').exists():raise ValueError('existing apply; inspect and rollback, never replay')
   if commands()!=cs:raise ValueError('candidate profiles changed after plan')
   for c in cs:
    doc=inspect(c['name']);validate_original(doc,c)
    if doc['Id']!=plan['original_ids'][c['name']] or inspect(plan['old_names'][c['name']]):raise ValueError('original identity/archive changed')
   quiescent(cs);write(root/'apply.intent.json',{'time':time.time(),'pid':os.getpid()})
   parallel(lambda c:call(['docker','stop','--time','60',c['name']]),cs)
   for c in cs:call(['docker','rename',c['name'],plan['old_names'][c['name']]])
   smi=root/'cards-after-stop.txt';smi.write_text(call(['npu-smi','info']))
   commands(smi) # shared allocation refuses any remaining card users
   ids=parallel(lambda c:call(c['argv']),cs);write(root/'candidate-container-ids.json',dict(zip([c['name'] for c in cs],ids)))
   wait_ready(cs);write(root/'apply.complete.json',{'time':time.time(),'containers':[inspect(c['name']) for c in cs]})
  else:
   if not (root/'apply.intent.json').exists():raise ValueError('no switch to roll back')
   if (root/'rollback.complete.json').exists():print('rollback already complete');return
   # Validate every original and active candidate before any removal.
   active=[]
   for c in cs:
    old=inspect(plan['old_names'][c['name']]);current=inspect(c['name'])
    if old is None:
     if current and current['Id']==plan['original_ids'][c['name']]:continue
     raise ValueError('original container missing '+c['name'])
    if old['Id']!=plan['original_ids'][c['name']]:raise ValueError('archive identity differs')
    if current:
     if current['Config'].get('Labels',{}).get(LABEL)!=c['profile_id'] or current['Image']!=IMAGE:raise ValueError('foreign replacement '+c['name'])
     active.append(c)
   if active:quiescent([c for c in active if inspect(c['name'])['State']['Running']])
   write(root/('rollback-attempt-'+str(time.time_ns())+'.json'),{'time':time.time(),'active':[c['name'] for c in active]})
   for c in active:
    call(['docker','stop','--time','60',c['name']]);call(['docker','rm',c['name']])
   for c in cs:
    old=inspect(plan['old_names'][c['name']])
    if old:call(['docker','rename',plan['old_names'][c['name']],c['name']])
    if not inspect(c['name'])['State']['Running']:call(['docker','start',c['name']])
   wait_ready(cs);write(root/'rollback.complete.json',{'time':time.time(),'original_ids':plan['original_ids']})
 print(json.dumps({'mode':a.mode,'record':str(root),'ready':True,'scientific_records_unchanged':True}))
if __name__=='__main__':main()
