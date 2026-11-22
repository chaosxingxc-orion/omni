"""Close the screening lease at roster publication; reuse the original rollout cutover."""
import argparse,contextlib,fcntl,hashlib,importlib.util,json,os,shlex,subprocess,sys,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def doc(p):return json.loads(Path(p).read_bytes())
def pin(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,v):
    p=Path(p);b=(json.dumps(v,sort_keys=True)+'\n').encode()
    if p.exists():
        if p.read_bytes()!=b:raise ValueError('handoff receipt conflict: '+str(p))
        return
    tmp=p.with_suffix('.partial')
    with tmp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
    with contextlib.ExitStack() as stack:
        fd=os.open(p.parent,os.O_DIRECTORY);stack.callback(os.close,fd);os.fsync(fd)

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def before(h):
    root=Path(h['old']);guards=[];rows=[]
    for name,lane in h['lanes'].items():
        m=load(root/('lane-'+name)/'launch.py','screen_'+name);m.verify()
        assert all(m.C[k]==lane[k] for k in ('port','cards','name'))
        assert m.C['host_boot_id']==h['boot'] and m.C['guard']==h['guard']
        d=m.inspect(m.name(h['old_attempt']))
        assert d['Id']==h['containers'][name] and d['State']['Running']
        assert d['HostConfig']['ReadonlyRootfs'] and not d['HostConfig']['Privileged']
        rows.append((name,lane,m,d['Id']))
    assert sorted(r[1]['cards'] for r in rows)==[[0,1,2,3],[4,5,6,7]]
    return rows
def drain(rows):
    for _,lane,m,_ in rows:
        with urllib.request.urlopen('http://127.0.0.1:%d/metrics'%lane['port'],timeout=8) as res:metrics=res.read().decode()
        for state in ('running','waiting'):
            seen=[float(l.rsplit(' ',1)[1]) for l in metrics.splitlines() if l.startswith('vllm:num_requests_'+state+'{')]
            assert seen and not any(seen),'live vLLM requests'
def host(h,terminal,check_only=False):
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==h['boot']
    root=Path(h['output'])
    if not check_only and (root/'DONE.json').exists():
        prior=doc(root/'DONE.json')
        assert prior['terminal']==terminal and prior['boot']==h['boot']
        return prior
    rows=before(h)
    if h.get('new'):
        for name,lane,old,_ in rows:
            new=load(Path(h['new'])/('lane-'+name)/'launch.py','new_'+name);new.verify()
            assert all(new.C[k]==old.C[k] for k in ('host_boot_id','guard','cards','image','port','host','model'))
            server=doc(new.C['policy_overlay']['server'])
            assert server['release_kind']=='formal' and server['update_inbox']==new.C['policy_overlay']['update_mount']
            assert server['service_identity']['container_name']==new.name(h['new_attempt'])
            assert 'diagnostic_logits' not in server['local_activation']
    if check_only:return dict(verified_current_ids=h['containers'],boot=h['boot'],next_resource=h['role'])
    assert terminal['guards_releasable'] and terminal['unknown_requests']==0 and terminal['screen_ref']['sha256']
    root.mkdir(exist_ok=True,mode=0o700);save(root/'DRAINED.json',terminal)
    if h.get('new'):
        for name,lane,_,_ in rows:
            folder=Path(h['new'])/('lane-'+name)
            receipt=folder/'runs'/h['preflight']/'state/PREFLIGHT.json'
            if not receipt.exists():
                subprocess.run(['python3','-B',str(folder/'launch.py'),'check','--attempt',h['preflight']],check=True,
                    stdout=sys.stderr,timeout=h['timeout'])
        cut=load(Path(__file__).with_name('cutover.py'),'owned_cutover')
        cut.switch(dict(output=str(root/'switch'),guard=h['guard'],terminal=str(root/'DRAINED.json'),
            lanes=[dict(old_code=str(Path(h['old'])/('lane-'+n)),old_attempt=h['old_attempt'],
                new_code=str(Path(h['new'])/('lane-'+n)),new_attempt=h['new_attempt'],preflight=h['preflight']) for n,_,_,_ in rows]))
        deadline=time.monotonic()+h['timeout'];ready={}
        for n,lane,_,_ in rows:
            path=Path(h['new'])/('lane-'+n)/'launch.py'
            while True:
                res=subprocess.run(['python3','-B',str(path),'ready','--attempt',h['new_attempt']],capture_output=True,text=True,timeout=30)
                if not res.returncode:break
                if time.monotonic()>deadline:raise TimeoutError(res.stderr[-2000:])
                time.sleep(10)
            ready[n]=json.loads(res.stdout)
            assert any(m['id']=='checkpoint-sha256:'+h['adapter_sha256'] for m in ready[n]['models'])
        result=dict(ready=ready)
    else:
        assert h['role']=='training' and not (root/'STOP-INTENT.json').exists(),'reconcile previous stop before retry'
        with open(h['guard'],'rb') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            rows=before(h);drain(rows);save(root/'STOP-INTENT.json',dict(containers=h['containers'],terminal=terminal))
            def stop(row):
                n,l,m,cid=row
                subprocess.run(['docker','stop','-t',str(m.C['stop_timeout_seconds']+15),cid],check=True,stdout=sys.stderr,timeout=m.C['stop_timeout_seconds']+45)
                st=m.inspect(m.name(h['old_attempt']))
                assert st['Id']==cid and st['State']['Status']=='exited' and st['State']['ExitCode']==0
                end=Path(m.C['root'])/'runs'/h['old_attempt']/'state/EXIT.json'
                assert doc(end)['exit_code']==0
                m.start_checks()
                return dict(container_id=cid,exit_ref=pin(end))
            with ThreadPoolExecutor(max_workers=len(rows)) as ex:result=dict(stopped=list(ex.map(stop,rows)))
    result.update(terminal=terminal,boot=h['boot'],role=h['role']);save(root/'DONE.json',result);return result

def client(config,check):
    c=doc(config);root=Path(c['output'])
    if not check:
        screen=doc(c['screen']);pool=doc(screen['screen_ref']['path']);m=doc(pool['inputs_ref']['path'])
        assert screen['schema']=='node-screen-data-result-v1' and screen['complete'] and screen['screen_scope']=='node' and screen['node']=='P'
        for ref in [screen['screen_ref'],pool['inputs_ref'],m['complete_ref'],m['selection_ref'],screen['usage_ref']]:assert pin(ref['path'])==ref
        cost=doc(screen['usage_ref']['path']);seal=doc(m['complete_ref']['path'])
        assert m['formal_sample_locked'] and not m['preview'] and 0<m['rows']==seal['rows'] and seal['roster_sha256']==m['selection_ref']['sha256']
        if c.get('consumer_receipt'):
            verified=doc(c['consumer_receipt'])
            assert verified['screen_ref']==pin(c['screen']) and verified['rows']==m['rows']
            assert m['rows']<=c['rows']
        else:
            assert m['rows']==c['rows']
        assert set(cost['statuses'])<= {'valid','invalid'} and sum(cost['statuses'].values())==cost['requests'] and cost['statuses']['valid']>=m['rows']
        terminal=dict(guards_releasable=True,unknown_requests=0,screen_ref=pin(c['screen']),selection_sha256=seal['roster_sha256'],requests=cost['requests'])
    else:terminal={}
    def invoke(item):
        n,h=item;remote=h['remote_script']
        args=['python3','-B',remote,'remote',h['remote_config']]+(['--check'] if check else [])
        res=subprocess.run(['ssh','-oBatchMode=yes',h['ssh'],shlex.join(args)],input=json.dumps(terminal),text=True,stdout=subprocess.PIPE,check=True)
        return n,json.loads(res.stdout)
    with ThreadPoolExecutor(max_workers=len(c['hosts'])) as ex:results=dict(ex.map(invoke,c['hosts'].items()))
    save(root/('PREPARED.json' if check else 'HANDOFF.json'),results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['remote','client']);p.add_argument('config');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.mode=='client':client(a.config,a.check)
    else:
        h=doc(a.config)
        if h.get('io_utils'):sys.path.insert(0,h['io_utils'])
        print(json.dumps(host(h,json.load(sys.stdin),a.check)))
