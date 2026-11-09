"""Developer-host archive owner; exact terminal event +600s, no model dispatch."""
import json,os,sys,time,threading,subprocess,shlex,tarfile,hashlib,fcntl,traceback
LANE='D' if '--D' in sys.argv else 'C-v2'
PREFIX='D-' if LANE=='D' else ''
from pathlib import Path
W=Path(__file__).resolve().parents[3];OUT=W/'delivery/s-single-intent-execution-v1';H='/srv/omni-local-state/s-single-intent-execution-v1';IMAGE='sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026'
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def ssh(args,**kwargs):return subprocess.run(['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=30','npu',shlex.join(args)],check=True,**kwargs)
def capture(label):
 path=OUT/'evidence'/label;path.mkdir(parents=True,exist_ok=True)
 code="""from pathlib import Path
import tarfile,sys
h=Path('"""+H+"""');t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|')
for p in sorted(h.rglob('*')):
 if p.is_file() and not any('workspace' in part for part in p.relative_to(h).parts) and not p.name.endswith('.tmp') and not p.name.endswith('.guard'):
  t.add(p,arcname=str(p.relative_to(h)),recursive=False)
t.close()
"""
 argv=['docker','run','--rm','--log-driver','none','--pull','never','--name','omni-s-single-archive-'+label+'-'+str(time.time_ns()),'--network','none','--cpus','2','--memory','4g','--mount','type=bind,src='+H+',dst='+H+',readonly','--entrypoint','/opt/saea-venv/bin/python',IMAGE,'-B','-c',code]
 with (path/'archive.tar').open('wb') as f:ssh(argv,stdout=f,stderr=subprocess.PIPE);f.flush();os.fsync(f.fileno())
 with tarfile.open(path/'archive.tar') as tf:
  for m in tf.getmembers():assert m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts
  tf.extractall(path)
 pins={str(p.relative_to(path)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in path.rglob('*') if p.is_file() and p.name!='archive.tar'}
 save(path/'DEV-ARCHIVE-RECEIPT.json',dict(at=time.time(),files=pins,tar_sha256=hashlib.sha256((path/'archive.tar').read_bytes()).hexdigest(),remote=H,model_calls=0))
 if (path/'ARCHIVE-PINS.json').exists():
  for rel,pin in json.loads((path/'ARCHIVE-PINS.json').read_bytes()).items():
   # monitor logs are append-only; terminal run files and scorer inputs must match.
   if rel.startswith(('C/','C-v2/','runtime-ready-v2/')):assert hashlib.sha256((path/rel).read_bytes()).hexdigest()==pin['sha256'],rel
 return path

def main():
 guard=(OUT/(PREFIX+'collector.guard')).open('a+b');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(OUT/'receipts'/(PREFIX+'LOCAL-COLLECTOR-START.json'),dict(pid=os.getpid(),ppid=os.getppid(),at=time.time(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),interval=600,event='exact inference container docker wait; terminal scoring polled read-only',model_calls=0))
 launch=None
 while launch is None:
  try:launch=json.loads(ssh(['cat',H+'/'+LANE+'/LAUNCH.json'],capture_output=True).stdout)
  except Exception:time.sleep(5)
 save(OUT/'receipts'/(PREFIX+'LAUNCH-v2.json'),launch);cid=launch['container_id'];event=threading.Event()
 def wait():
  r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=30','npu',shlex.join(['docker','wait',cid])],capture_output=True);save(OUT/'receipts'/(PREFIX+'WAIT-v2.json'),dict(returncode=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode(),at=time.time(),container_id=cid));event.set()
 threading.Thread(target=wait,daemon=True).start();first=False;last=0
 while True:
  try:
   now=time.monotonic()
   if now-last>=600 or event.is_set():
    # Consume the terminal signal once, including failed archive attempts.
    terminal_event=event.is_set();event.clear();last=now
    p=capture(PREFIX+('terminal' if terminal_event else 'snapshot-'+str(time.time_ns())))
    if (p/(PREFIX+'TERMINAL-COLLECTION.json')).exists():
     terminal=json.loads((p/(PREFIX+'TERMINAL-COLLECTION.json')).read_bytes());save(OUT/'receipts'/(PREFIX+'TERMINAL-COLLECTION.json'),terminal)
     if (p/LANE/'score/SUMMARY.json').exists():
      result=json.loads((p/LANE/'score/SUMMARY.json').read_bytes());save(OUT/(PREFIX+'RESULT.json'),result)
      (OUT/(PREFIX+'REPORT.md')).write_text('C no_examples 已完整归档与评分；动态 fewshot 状态另见 READY-FEWSHOT-ADMISSION.json。官方条件槽位指标不是 D/E2E。\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')
     save(OUT/(PREFIX+'READY.json'),dict(status=terminal['status'],archive=str(p.relative_to(W)),collector_pid=os.getpid(),at=time.time(),model_calls=0));return
    # A missing collection marker is retried on the normal 600s cadence.
   if not first:
    try:
     d=json.loads(ssh(['cat',H+('/monitor-D/FIRST-MODEL-RESPONSE.json' if LANE=='D' else '/monitor/FIRST-MODEL-RESPONSE.json')],capture_output=True).stdout);save(OUT/'receipts'/(PREFIX+'FIRST-MODEL-RESPONSE.json'),d);first=True
    except Exception:pass
  except Exception:save(OUT/'receipts'/('COLLECTOR-ERROR-'+str(time.time_ns())+'.json'),dict(at=time.time(),error=traceback.format_exc()))
  event.wait(10 if not first else min(30,max(1,600-(time.monotonic()-last))))
if __name__=='__main__':main()
