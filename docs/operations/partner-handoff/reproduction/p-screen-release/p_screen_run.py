"""Node-scoped consumer of the closed recall queue. Only completed P outputs define strata."""
import collections,concurrent.futures,contextlib,fcntl,json,os,random,shlex,sqlite3,subprocess,sys,time,urllib.request,urllib.error,math
from pathlib import Path
from dataset_encoder import save,put
from p_screen_inputs import Inputs,doc,lines
import hashlib
def url(port,path,wire=None,timeout=20):
    req=urllib.request.Request('http://127.0.0.1:%d/%s'%(port,path),data=wire,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as res:return res.read()
def apport(n,weights):
    raw={k:n*v for k,v in weights.items()}
    q={k:math.floor(v) for k,v in raw.items()}
    for k in sorted(q,key=lambda k:(q[k]-raw[k],k))[:n-sum(q.values())]:q[k]+=1
    return q
def remaining_order(gate,priority,seen,seed):
    ids={'real':[],'synthetic':[]}
    if not seen.issubset(gate):raise ValueError('screen ledger contains a recording outside the frozen gate')
    for rid in sorted(gate):
        domain={'train_real':'real','train_synth':'synthetic'}[rid.split(':',1)[0]]
        ids[domain].append(rid)
    rng=random.Random(seed)
    for domain,vals in ids.items():
        # Replay the original draws so the final roster sampler inherits the same RNG state.
        rng.shuffle(vals);vals.sort(key=priority.__contains__)
        ids[domain]=[rid for rid in vals if rid not in seen]
    return ids,rng

def lane_slots(ports,default,overrides):
    if len(set(ports))!=len(ports) or set(overrides)-set(map(str,ports)):
        raise ValueError('unknown or duplicate screen lane')
    caps={p:overrides.get(str(p),default) for p in ports}
    if not caps or any(type(n) is not int or n<1 for n in caps.values()):
        raise ValueError('invalid screen lane concurrency')
    return [p for n in range(max(caps.values())) for p in ports if n<caps[p]]

def binding_table(db):
    db.execute('create table if not exists screen_bindings(rid text primary key,source_wire text,model text,engine_id text)')
    db.commit()

def transport_payload(packet,binding=None):
    wire=json.dumps(packet['wire'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    assert hashlib.sha256(wire).hexdigest()==packet['wire_sha256']
    if binding is not None:
        if packet['wire']['model']!=binding['base']:raise ValueError('unexpected source model')
        if not binding.get('policy_overlay'):
            wire=json.dumps(dict(packet['wire'],model=binding['alias']),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    return wire

def submit_screen(x,root,timeout,rid,domain,port,*,binding=None,timing=None):
    started=time.monotonic()
    packet=x.packet(rid)
    wire=transport_payload(packet,binding)
    source=packet['wire_sha256']
    actual=hashlib.sha256(wire).hexdigest()
    prepared=time.monotonic()
    ranked=[v['label'] for v in packet['ranking']]
    # Own the unique request before transport. A broken connection stays unknown after restart.
    with sqlite3.connect(root/'p-screen.sqlite',timeout=60) as conn:
        conn.execute('insert into calls(rid,domain,lane,issued,status,wire,ranking) values(?,?,?,?,?,?,?)',(rid,domain,str(port),time.time(),'intent',actual,json.dumps(ranked)))
        if binding is not None:
            conn.execute('insert into screen_bindings values(?,?,?,?)',(rid,source,binding['alias'],binding['engine_id']))
    sent=time.monotonic()
    raw=b'';rank=None;err=None
    try:
        raw=url(port,'v1/chat/completions',wire,timeout)
        received=time.monotonic()
        response=json.loads(raw)
        if binding is not None and response.get('model')!=binding['base' if binding.get('policy_overlay') else 'alias']:
            raise RuntimeError('response model differs from selected serving path')
        choice,=response['choices']
        if choice['finish_reason']!='stop':raise ValueError('incomplete P output')
        selected=x.parse(choice['message']['content'],packet['ranking'],3)
        labels=[v['label'] for v in selected];rank=labels.index(x.gold[rid])+1 if x.gold[rid] in labels else 0
        status='valid'
    except urllib.error.HTTPError as e:status='http_failure';raw=e.read();err=str(e)
    except (ValueError,KeyError) as e:status='invalid';err=str(e)
    except Exception as e:status='unknown';err=repr(e)
    with sqlite3.connect(root/'p-screen.sqlite',timeout=60) as conn:
        conn.execute('update calls set finished=?,status=?,raw=?,rank=?,error=? where rid=?',(time.time(),status,raw,rank,err,rid))
    if timing is not None and status=='valid':
        end=time.monotonic()
        timing.update(prepare=prepared-started,intent=sent-prepared,http=received-sent,parse_and_commit=end-received,total=end-started,prompt=response['usage']['prompt_tokens'],completion=response['usage']['completion_tokens'])
    return status,rank,err,raw[:600] if status=='http_failure' else None

def run(c):
    root=Path(c['output']); lock=(root/'SCREEN.guard').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    db=sqlite3.connect(root/'p-screen.sqlite');db.row_factory=sqlite3.Row
    db.execute('pragma journal_mode=WAL');db.execute('pragma synchronous=FULL')
    db.execute('create table if not exists calls(rid text primary key,domain text,lane text,issued real,finished real,status text,wire text,ranking text,raw blob,rank integer,error text)')
    x=Inputs(c);s=c['sampling'];assert (s['scope'],s['node'],s['metric'])==('node','P','valid_P_top3')
    quotas={(d,o):n for o,total in apport(s['sample_count'],s['error_success']).items() for d,n in apport(total,s['domain_mix']).items()}
    history={}
    sid_to_rid={'slurp:train:'+r['source_id'].zfill(6)+':'+Path(r['recording']).stem:rid for rid,r in x.gate.items()}
    for part in ('MISS','HIT'):
        for row in lines(Path(c['history'])/('exploration-P-'+part+'.private.jsonl')):
            rid=sid_to_rid.get(row['sample_id'])
            if rid is None or row.get('valid',True) is not True:continue
            labels=row['P_top3'];assert len(set(labels))==3
            rank=labels.index(x.gold[rid])+1 if x.gold[rid] in labels else 0
            assert bool(rank)==(part=='HIT')
            history[rid]=dict(reference_id=rid,rank=rank,response_sha256=row['P_response_sha256'],legacy=True)
    # Historical observations have a different checkpoint/supply identity; use them to prioritize new screening.
    save(root/'HISTORY.json',dict(matched=len(history),correct=sum(bool(v['rank']) for v in history.values()),incorrect=sum(not v['rank'] for v in history.values()),use='screen priority only; current observations confirm P class'))
    if db.execute("select 1 from calls where finished is null or status not in ('valid','invalid') limit 1").fetchone():
        raise ValueError('reconcile unresolved screen requests before resuming')
    seen={r[0] for r in db.execute('select rid from calls')}
    ids,rng=remaining_order(x.gate,{rid for rid,r in history.items() if not r['rank']},seen,s['seed'])
    counts=collections.Counter()
    for r in db.execute("select domain,rank,count(*) n from calls where status='valid' group by domain,rank"):
        counts[r['domain'],'success' if r['rank'] else 'error']+=r['n']
    def outstanding(d):return any(counts[d,o]<quotas[d,o] for o in ('success','error'))
    ports=[];leases=[];requests=0;activation_checks={}
    bindings=c.get('model_bindings_by_port',{})
    if bindings:binding_table(db)
    def submit(rid,domain,port):
        return submit_screen(x,root,c['timeout'],rid,domain,port,binding=bindings.get(str(port)))
    with contextlib.ExitStack() as stack:
        for group,h in c['hosts'].items():
            profile=doc(h['profile']);local=h['local_ports'];ports+=local
            cmd=['ssh','-T','-oBatchMode=yes','-oExitOnForwardFailure=yes','-oServerAliveInterval=20','-oServerAliveCountMax=3']
            for port,lane in zip(local,profile['lanes'].values(),strict=True):cmd+=['-L',f'127.0.0.1:{port}:127.0.0.1:{lane["port"]}']
            # Wait for actual service initialization before acquiring the shared guard.
            deadline=time.monotonic()+c['startup_timeout']
            while True:
                remote_ports=[lane['port'] for lane in profile['lanes'].values()]
                cmd_health=' && '.join('curl -fsS --max-time 5 http://127.0.0.1:%d/health >/dev/null'%p for p in remote_ports)
                if subprocess.run(['ssh',h['ssh'],cmd_health],stderr=subprocess.DEVNULL).returncode==0:break
                if time.monotonic()>=deadline:raise TimeoutError('inference startup: '+group)
                time.sleep(10)
            for name in profile['lanes']:
                if h.get('overlay_check'):
                    lane=profile['lanes'][name]
                    port=local[list(profile['lanes']).index(name)]
                    binding=bindings[str(port)]
                    assert binding['policy_overlay']
                    args=['python3','-B',h['overlay_check'],profile['target'],name,profile['attempt'],
                          binding['alias'],binding['engine_id'],h['switch']]
                    activation_checks[port]=['ssh',h['ssh'],shlex.join(args+['active'])]
                    subprocess.run(['ssh',h['ssh'],shlex.join(args+['ready'])],check=True)
                    continue
                check=['python3','-B',h.get('ready_root',profile['target'])+'/lane-'+name+'/launch.py','ready','--attempt',profile['attempt']]
                if 'expected_adapter' in h:check+=['--expected-adapter',h['expected_adapter']]
                subprocess.run(['ssh',h['ssh'],shlex.join(check)],check=True)
            cmd+=[h['ssh'],shlex.join(['flock','-xn',h['guard'],'sh','-c','echo LOCKED; cat >/dev/null'])]
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            stack.callback(lambda p:(p.stdin.close(),p.wait()),proc)
            assert proc.stdout.readline().strip()=='LOCKED','host guard unavailable'
            leases.append(proc)
        slots=lane_slots(ports,c['concurrency_per_lane'],c.get('concurrency_by_port',{}))
        if set(bindings)-set(map(str,ports)):raise ValueError('unregistered screen binding')
        for port,binding in bindings.items():
            if int(port) in activation_checks:
                continue  # Formal overlay activates the pinned adapter on the first real request.
            models=json.loads(url(int(port),'v1/models'))['data']
            if not any(m['id']==binding['alias'] and m.get('parent')==binding['base'] for m in models):
                raise ValueError('adapter alias is unavailable')
        executor=stack.enter_context(concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)))
        grouped=c.get('admission','continuous')=='group_drained'
        if activation_checks and not grouped:raise ValueError('qualified overlay expects fixed batches')
        capacity=collections.Counter(slots); returned=collections.Counter()
        future={};free=collections.deque(slots);di=len(seen);bad=None
        while future or (not bad and any(outstanding(d) and ids[d] for d in ids)):
            while free and not bad:
                avail=[d for d in ids if outstanding(d) and ids[d]]
                if not avail:break
                if any(p.poll() is not None for p in leases):bad='guard connection lost';break
                disk=os.statvfs(root)
                if disk.f_bavail*disk.f_frsize<max(20*1024**3,disk.f_blocks*disk.f_frsize*.1) or disk.f_favail<disk.f_files*.05:bad='disk reserve';break
                d=avail[di%len(avail)];di+=1;port=free.popleft();rid=ids[d].pop()
                future[executor.submit(submit,rid,d,port)]=(rid,d,port)
            if not future:break
            done,_=concurrent.futures.wait(future,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                rid,d,port=future.pop(f)
                if not grouped:free.append(port)
                else:returned[port]+=1
                try:status,rank,err,body=f.result()
                except Exception as e:bad=repr(e);print('prepare failure',rid,bad,flush=True);continue
                requests+=1
                if status=='valid':counts[d,'success' if rank else 'error']+=1
                if status in ('http_failure','unknown'):bad=(rid,status,err,str(body));print('transport failure',bad,flush=True)
                if requests in (1,32,128) or requests%500==0:print(time.time(),requests,dict(counts),flush=True)
            if grouped:
                for port in capacity:
                    if returned[port] and not any(item[2]==port for item in future.values()):
                        if not bad and port in activation_checks:
                            try:subprocess.run(activation_checks.pop(port),check=True,timeout=45)
                            except (OSError,subprocess.SubprocessError) as exc:bad=str(exc)
                        free.extend([port]*returned.pop(port))
        if bad:raise RuntimeError(bad)
        if any(outstanding(d) for d in ids):raise ValueError('P population exhausted before quota')
        roster=[]
        for (domain,outcome),n in quotas.items():
            rows=[dict(r) for r in db.execute("select rid,rank,wire,raw from calls where status='valid' and domain=? and (rank>0)=? order by rid",(domain,outcome=='success'))]
            for row in rng.sample(rows,n):
                rid=row['rid'];roster.append(dict(**x.gate[rid],p_rank=row['rank'],outcome=outcome,wire_sha256=row['wire'],response_sha256=hashlib.sha256(row['raw']).hexdigest(),domain=domain))
        assert len({r['reference_id'] for r in roster})==s['sample_count']
        path=root/'P-SAMPLE.private.jsonl';put(path,b''.join((json.dumps(r,sort_keys=True)+'\n').encode() for r in roster))
        save(root/'COMPLETE.json',dict(finished=time.time(),first_request=db.execute('select min(issued) from calls').fetchone()[0],rows=len(roster),counts={str(k):v for k,v in quotas.items()},roster_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='P screening'))
if __name__=='__main__':run(doc(sys.argv[1]))
