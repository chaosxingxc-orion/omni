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

def finish_rollout(h,root):
    """Resume only a fully recorded cutover, then activate via the pinned audio probe."""
    switch=root/'switch'
    intent=doc(switch/'INTENT.json')
    assert intent['terminal_ref']==pin(root/'DRAINED.json') and intent['source_boot']==h['boot']
    assert {v['container_id'] for v in intent['predecessors']}==set(h['containers'].values())
    probe_ref=h['startup_probe_ref']
    assert pin(probe_ref['path'])==probe_ref
    probe=doc(probe_ref['path'])
    assert probe['handoff']==str(root/'DONE.json') and probe['guard']==h['guard']
    assert probe['adapter_sha256']==h['adapter_sha256'] and probe['lanes'].keys()==h['lanes'].keys()
    assert pin(probe['wire']['path'])==probe['wire']
    wire=doc(probe['wire']['path'])
    assert wire['chat_template_kwargs']=={'enable_thinking':False} and wire['logprobs'] is True
    assert h['startup_measure_ref']==pin(probe['measure'])
    measure=load(probe['measure'],'activation_probe')
    alias='checkpoint-sha256:'+h['adapter_sha256']
    services={}
    ready={}
    with open(h['guard'],'rb') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for n,lane in h['lanes'].items():
            old=load(Path(h['old'])/('lane-'+n)/'launch.py','old_'+n);old.verify()
            old_state=old.inspect(old.name(h['old_attempt']))
            stopped=doc(switch/f"STOPPED-{lane['port']}.json")
            assert stopped['container_id']==old_state['Id']==h['containers'][n]
            assert old_state['State']['Status']=='exited' and old_state['State']['ExitCode']==0
            assert pin(stopped['service_exit']['path'])==stopped['service_exit']
            assert doc(stopped['service_exit']['path'])['exit_code']==0
            folder=Path(h['new'])/('lane-'+n)
            new=load(folder/'launch.py','new_'+n);new.verify()
            assert all(new.C[k]==old.C[k] for k in ('host_boot_id','guard','cards','image','port','host','model'))
            server=doc(new.C['policy_overlay']['server'])
            container=new.name(h['new_attempt'])
            assert server['release_kind']=='formal' and server['update_inbox']==new.C['policy_overlay']['update_mount']
            assert server['service_identity']['container_name']==container and 'diagnostic_logits' not in server['local_activation']
            profile=doc(server['profile_ref']['path'])
            assert pin(server['profile_ref']['path'])==server['profile_ref']
            assert all(wire.get(k)==v for k,v in profile['observers'].items())
            assert all(k not in wire for k in profile['remove_fields'])
            assert probe['lanes'][n]==dict(container=container,service=pin(folder/'config.json'))
            assert doc(switch/f"START-INTENT-{lane['port']}.json")['container']==container
            started=doc(switch/f"STARTED-{lane['port']}.json")
            current=new.inspect(container)
            assert current['Id']==started['container_id'] and current['Image']==new.C['image']
            assert started['physical_devices']==lane['cards']==new.C['cards']
            endpoint=f"http://127.0.0.1:{lane['port']}"
            services[n]=(new,endpoint,started['container_id'])
        def readiness(new,endpoint):
            with urllib.request.urlopen(endpoint+'/health',timeout=8) as response:
                assert response.status==200
            with urllib.request.urlopen(endpoint+'/v1/models',timeout=8) as response:
                models=json.load(response)['data']
            engine=new.C
            base=[row for row in models if row['id']==wire['model']]
            adapters=[row for row in models if row['id']!=wire['model']]
            assert len(base)==1 and base[0]['max_model_len']==int(
                engine['serve_args'][engine['serve_args'].index('--max-model-len')+1])
            assert not adapters or (len(adapters)==1 and adapters[0]['id']==alias
                                    and adapters[0]['parent']==wire['model'])
            return models
        deadline=time.monotonic()+h['timeout']
        for new,endpoint,cid in services.values():
            live=new.inspect(new.name(h['new_attempt']))
            expected_devices={'/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc'}
            expected_devices.update('/dev/davinci'+str(card) for card in new.C['cards'])
            assert live['Id']==cid and live['State']['Running'] and live['HostConfig']['ReadonlyRootfs']
            assert not live['HostConfig']['Privileged']
            assert {entry['PathOnHost'] for entry in live['HostConfig']['Devices']}==expected_devices
            while True:
                try:
                    readiness(new,endpoint)
                except OSError as e:
                    assert new.inspect(new.name(h['new_attempt']))['State']['Running']
                    if time.monotonic()>=deadline:raise TimeoutError('rollout HTTP health timeout') from e
                    time.sleep(3)
                else:
                    break
        for _,endpoint,_ in services.values():measure.idle(measure.metrics(endpoint))
        def activate(item):
            n,(_,endpoint,_)=item
            path=Path(probe['output'])/n/'call'
            response=path/'RESPONSE.json'
            if not response.exists():measure.contact(path,endpoint,wire,probe['timeout'])
            prior=doc(path/'INTENT.json')
            assert prior['endpoint']==endpoint and prior['api_path']=='/v1/chat/completions'
            assert prior['wire_sha256']==hashlib.sha256(measure.canonical(wire)).hexdigest()
            assert doc(response)['status']==200 and not (path/'UNKNOWN.json').exists()
            measure.idle(measure.metrics(endpoint))
        with ThreadPoolExecutor(max_workers=len(services)) as ex:list(ex.map(activate,services.items()))
        for n,(new,endpoint,cid) in services.items():
            models=readiness(new,endpoint)
            assert any(row['id']==alias for row in models)
            ready[n]=dict(ready=True,container_id=cid,models=models,
                          thinking='OFF; clients send chat_template_kwargs.enable_thinking=false')
            act=json.loads(new.command(['docker','exec',cid,'cat','/runtime/state/policy/ACTIVATION.json']))
            assert act['alias']==act['policy_version']==alias and act['weights_ref']['sha256']==h['adapter_sha256']
            assert act['service_identity']['container_name']==new.name(h['new_attempt']) and act['source_boot']==h['boot']
            assert act['base'] is False and act['fixture'] is False
    return ready

def host(h,terminal,check_only=False):
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==h['boot']
    root=Path(h['output'])
    if not check_only and (root/'DONE.json').exists():
        prior=doc(root/'DONE.json')
        assert prior['terminal']==terminal and prior['boot']==h['boot']
        return prior
    if not check_only and h.get('new') and (root/'switch').exists():
        assert doc(root/'DRAINED.json')==terminal
        result=dict(ready=finish_rollout(h,root),terminal=terminal,boot=h['boot'],role=h['role'])
        save(root/'DONE.json',result)
        return result
    rows=before(h)
    if h.get('new'):
        probe_ref=h['startup_probe_ref']
        assert pin(probe_ref['path'])==probe_ref
        probe=doc(probe_ref['path'])
        assert probe['handoff']==str(root/'DONE.json') and probe['guard']==h['guard']
        assert probe['adapter_sha256']==h['adapter_sha256']
        assert pin(probe['wire']['path'])==probe['wire'] and pin(probe['measure'])==h['startup_measure_ref']
        for name,lane,old,_ in rows:
            new=load(Path(h['new'])/('lane-'+name)/'launch.py','new_'+name);new.verify()
            assert all(new.C[k]==old.C[k] for k in ('host_boot_id','guard','cards','image','port','host','model'))
            server=doc(new.C['policy_overlay']['server'])
            assert server['release_kind']=='formal' and server['update_inbox']==new.C['policy_overlay']['update_mount']
            assert server['service_identity']['container_name']==new.name(h['new_attempt'])
            assert 'diagnostic_logits' not in server['local_activation']
            assert probe['lanes'][name]==dict(container=new.name(h['new_attempt']),
                                            service=pin(new.CODE/'config.json'))
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
        result=dict(ready=finish_rollout(h,root))
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
