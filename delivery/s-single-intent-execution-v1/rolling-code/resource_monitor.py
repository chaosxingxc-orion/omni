"""600-second resource evidence; terminal/arrival scheduler owns dispatch separately."""
import concurrent.futures,json,os,shlex,subprocess,time,sys,fcntl
from pathlib import Path
O=Path(__file__).resolve().parents[1]/'rolling-state';W=json.loads((O.parents[1]/'s-rolling-12h-v1/WINDOW.json').read_bytes())
def save(p,d):
 p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def host(host):
 args=['ssh','-o','ControlMaster=no','-o','ControlPath=none','-o','BatchMode=yes',host]
 r=subprocess.run(args+['npu-smi info'],capture_output=True,text=True,timeout=45)
 metrics={}
 for lane,port in ([('A',8200),('B',8202)] if host=='npu2' else [('C',8204),('D',8206)]):
  code="""import urllib.request,json
s=urllib.request.urlopen('http://127.0.0.1:PORT/metrics',timeout=8).read().decode()
rows={}
for line in s.splitlines():
 if not line or line.startswith('#'):continue
 name,value=line.rsplit(' ',1);base=name.split('{',1)[0]
 try:rows[base]=rows.get(base,0)+float(value)
 except ValueError:pass
families={'prompt_tokens':['prompt_tokens_total'],'generation_tokens':['generation_tokens_total'],'prefix_queries':['prefix_cache_queries_total'],'prefix_hits':['prefix_cache_hits_total'],'queue_time_sum':['request_queue_time_seconds_sum'],'queue_time_count':['request_queue_time_seconds_count'],'prefill_time_sum':['request_prefill_time_seconds_sum'],'prefill_time_count':['request_prefill_time_seconds_count'],'decode_time_sum':['request_decode_time_seconds_sum'],'decode_time_count':['request_decode_time_seconds_count'],'cache_usage':['gpu_cache_usage_perc','kv_cache_usage_perc'],'preemptions':['num_preemptions_total']}
out={k:rows.get('vllm:num_requests_'+k,0) for k in ['running','waiting']}
out['counters']={};out['unsupported']=[]
for label,aliases in families.items():
 values={name:v for name,v in rows.items() if any(name=='vllm:'+alias for alias in aliases)}
 if values:out['counters'][label]=values
 else:out['unsupported'].append(label)
print(json.dumps(out))
""".replace('PORT',str(port))
  q=subprocess.run(args+[shlex.join(['python3','-c',code])],capture_output=True,text=True,timeout=15)
  metrics[lane]=json.loads(q.stdout) if q.returncode==0 else dict(error=q.stderr)
 return dict(host=host,at=time.time(),returncode=r.returncode,npu_smi=r.stdout,stderr=r.stderr,metrics=metrics)
lock=(O/'resource-monitor.guard').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
save(O/'RESOURCE-MONITOR-START.json',dict(pid=os.getpid(),ppid=os.getppid(),at=time.time(),interval=600))
while True:
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:
  fs={p.submit(host,h):h for h in ['npu2','npu']};results={}
  for f,h in fs.items():
   try:results[h]=f.result()
   except Exception as e:results[h]=dict(error=str(e),at=time.time())
 d=dict(at=time.time(),hosts=results,cards=16,instances=4,TP=4,window=W,money=None)
 save(O/'resources'/('snapshot-'+str(time.time_ns())+'.json'),d);save(O/'RESOURCE-LATEST.json',d)
 if time.time()>W['end_epoch']+7200:break
 time.sleep(600)
