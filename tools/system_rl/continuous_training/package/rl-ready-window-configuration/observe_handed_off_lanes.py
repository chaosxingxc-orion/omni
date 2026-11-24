"""Observe the handed-off formal owners using the original HTTP and readback verifiers."""
import argparse,fcntl,hashlib,json,os,shlex,subprocess,sys,tempfile,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from screen_resource_handoff import doc,load,pin,save


def require_idle(endpoint):
    with urllib.request.urlopen(endpoint + '/metrics', timeout=10) as response:
        metrics = response.read().decode()
    occupancy = {}
    for state in ('running', 'waiting'):
        values = [float(line.rsplit(' ', 1)[1]) for line in metrics.splitlines()
                  if line.startswith('vllm:num_requests_' + state + '{')]
        assert values and not any(values), 'live requests on ' + endpoint
        occupancy[state] = sum(values)
    return occupancy


def reconcile_activation_collision(c, stop, failed):
    """Retain the refused requests and isolate each exited process's policy state."""
    root = Path(c['output'])
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    owner = doc(c['handoff'])
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert owner['boot'] == stop['boot'] == boot
    assert owner['role'] == 'rollout' and owner['terminal']['guards_releasable']
    predecessor = doc(failed/'RESTARTED.json')
    assert predecessor.keys() == c['lanes'].keys()
    released = {row['lane']: row for row in stop['stopped']}
    assert released.keys() == c['lanes'].keys()
    target = root/'RECOVERY-STOPPED.json'
    with Path(c['guard']).open('rb') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            closed = doc(target)
            assert closed['failed_restart_ref'] == pin(failed/'RESTARTED.json')
            return closed
        rows = []
        for lane, spec in c['lanes'].items():
            old = released[lane]
            config = spec['service']
            assert pin(config['path']) == config
            launch = load(Path(config['path']).with_name('launch.py'), 'collision_' + lane)
            launch.verify()
            service = launch.inspect(spec['container'])
            assert service['Id'] == old['container_id'] == owner['ready'][lane]['container_id']
            assert launch.C['guard'] == c['guard'] and launch.C['cards'] == old['cards']
            call = failed/lane/'call'
            assert not (call/'UNKNOWN.json').exists()
            response_ref = pin(call/'RESPONSE.json')
            response = doc(call/'RESPONSE.json')
            assert response['status'] == 400
            error = json.loads(response['raw'])['error']['message']
            assert error == 'IMMUTABLE_COLLISION:<path>'
            state = service['State']
            assert state['StartedAt'] == predecessor[lane]['StartedAt']
            runtime, = (m for m in service['Mounts'] if m['Destination'] == '/runtime')
            path = Path(runtime['Source'])/'state'
            assert runtime['RW'] and path.parent.is_relative_to(Path(launch.C['root'])/'runs')
            server = doc(launch.C['policy_overlay']['server'])
            assert server['overlay_root'] == '/runtime/state/policy'
            archived = path/('policy-before-'+pin(failed/'RESTARTED.json')['sha256'][:16])
            intent = root/(lane+'-COLLISION-INTENT.json')
            if state['Running']:
                assert not archived.exists()
                occupancy = require_idle(f"http://127.0.0.1:{launch.C['port']}")
                saved = {}
                for name in ('ACTIVATION.json','PRECISION.json','CACHE-POLICY.json',
                             'ACTIVATION-FAILED.json'):
                    raw = subprocess.check_output(['docker','exec',service['Id'],'cat',
                                                   '/runtime/state/policy/'+name], timeout=15)
                    if name == 'ACTIVATION-FAILED.json':
                        assert 'IMMUTABLE_COLLISION:/runtime/state/policy/ACTIVATION.json' in json.loads(raw)['error']
                    saved[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                st = (path/'policy').stat()
                save(intent, dict(container=service['Id'], started=state['StartedAt'],
                                  response_ref=response_ref, occupancy=occupancy, files=saved,
                                  policy_inode=st.st_ino, archive=str(archived)))
                subprocess.run(['docker','stop','--time',str(launch.C['stop_timeout_seconds']+15),
                                service['Id']], check=True, capture_output=True,
                               timeout=c['timeout'] + launch.C['stop_timeout_seconds'])
                state = launch.inspect(spec['container'])['State']
            record = doc(intent)
            assert record['response_ref'] == response_ref and record['archive'] == str(archived)
            assert record['container'] == service['Id'] and record['started'] == state['StartedAt']
            assert state['Status'] == 'exited' and state['ExitCode'] == 0 and not state['OOMKilled']
            exited = doc(path/'EXIT.json')
            assert exited['exit_code'] == 0 and 'signal:15' in exited['stop_reasons']
            exit_receipt = root/(lane+'-PREDECESSOR-EXIT.json')
            save(exit_receipt, exited)
            if not archived.exists():
                assert (path/'policy').stat().st_ino == record['policy_inode']
                (path/'policy').rename(archived)
                fd = os.open(path, os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            assert archived.stat().st_ino == record['policy_inode'] and not (path/'policy').exists()
            rows.append(dict(old, finished_at=state['FinishedAt'], before=record['occupancy'],
                             archive_intent=pin(intent), service_exit=pin(exit_receipt)))
        closed = dict(stop, stopped=rows, failed_restart_ref=pin(failed/'RESTARTED.json'))
        save(target, closed)
        return closed


def restart_state(c, stop, *, check_only):
    """Restart the exact owners recorded in a clean release, once."""
    root = Path(c['output'])
    if not check_only:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    owner = doc(c['handoff'])
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert owner['boot'] == stop['boot'] == boot
    assert owner['role'] == 'rollout' and owner['terminal']['guards_releasable']
    released = {row['lane']: row for row in stop['stopped']}
    assert released.keys() == c['lanes'].keys()
    services = {}
    states = {}
    intent = root/'RESTART-INTENT.json'
    with open(c['guard'], 'rb') as guard:
        if not check_only:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert os.fstat(guard.fileno()).st_ino == stop['guard_inode']
        expected_intent = dict(boot=boot, released=released)
        started = intent.is_file()
        if started:
            assert doc(intent) == expected_intent, 'restart intent differs from release'
        receipt = root/'RESTARTED.json'
        previous = doc(receipt) if receipt.exists() else None
        assert previous is None or (started and previous.keys() == released.keys())
        for lane, spec in c['lanes'].items():
            old = released[lane]
            config = spec['service']
            assert pin(config['path']) == config
            launch = load(Path(config['path']).with_name('launch.py'), 'restore_' + lane)
            launch.verify()
            d = launch.inspect(spec['container'])
            assert old['container_id'] == owner['ready'][lane]['container_id'] == d['Id']
            assert old['name'].lstrip('/') == spec['container'] == d['Name'].lstrip('/')
            assert old['exit_code'] == 0 and old.get('oom', False) is False and not any(old['before'].values())
            assert old['cards'] == launch.C['cards'] and launch.C['guard'] == c['guard']
            state = d['State']
            if not state['Running']:
                assert previous is None and state['Status'] == 'exited' and state['ExitCode'] == 0
                assert state['FinishedAt'] == old['finished_at'] and not state['OOMKilled'], \
                    'restart outcome requires reconciliation'
                if not check_only:
                    launch.start_checks()
            else:
                assert started and state['StartedAt'] > old['finished_at'] and not state['Restarting']
                if previous is not None:
                    assert state['StartedAt'] == previous[lane]['StartedAt']
            states[lane] = dict(running=state['Running'], cards=old['cards'])
            services[lane] = d
        if check_only:
            return states
        save(intent, expected_intent)
        for d in services.values():
            if not d['State']['Running']:
                subprocess.run(['docker', 'start', d['Id']], capture_output=True,
                               check=True, timeout=c['timeout'])
        current = {
            lane: json.loads(subprocess.check_output(['docker', 'inspect', d['Id']]))[0]['State']
            for lane, d in services.items()
        }
        assert all(v['Running'] and v['StartedAt'] > released[k]['finished_at']
                   for k, v in current.items())
        if previous is not None:
            assert all(current[k]['StartedAt'] == previous[k]['StartedAt'] for k in current)
        else:
            save(receipt, current)


def restore(c, stop):
    restart_state(c, stop, check_only=False)
    end = time.monotonic() + c.get('restart_ready_seconds', c['timeout'])
    for spec in c['lanes'].values():
        cfg = doc(spec['service']['path'])
        endpoint = f"http://127.0.0.1:{cfg['port']}/health"
        while True:
            try:
                with urllib.request.urlopen(endpoint, timeout=5) as response:
                    assert response.status == 200
                break
            except OSError:
                assert time.monotonic() < end, 'restored owner not ready'
                time.sleep(2)


READ_STOPPED_POLICY = """
from pathlib import Path
import hashlib,json,sys
p=Path('/previous-policy')
assert not any(p.glob('**/SWITCH-INTENT.json')), 'stopped policy has an unresolved switch'
files=json.loads(sys.argv[1])
print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in files},sort_keys=True))
"""


def rotate_stopped_policy(c, stop, previous_ready, *, check_only=False):
    """Preserve the previous process receipts before starting the same containers."""
    root = Path(c['output'])
    owner = doc(c['handoff'])
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert owner['boot'] == stop['boot'] == boot
    assert owner['terminal']['guards_releasable']
    released = {row['lane']: row for row in stop['stopped']}
    assert released.keys() == previous_ready.keys() == c['lanes'].keys()
    with Path(c['guard']).open('rb') as guard:
        if not check_only:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert os.fstat(guard.fileno()).st_ino == stop['guard_inode']
        if not check_only:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if (root/'RESTART-INTENT.json').exists():
            assert all((root/(lane+'-POLICY-ROTATED.json')).is_file() for lane in released)
            return
        for lane, spec in c['lanes'].items():
            row = released[lane]
            prior = previous_ready[lane]
            assert row['exit_code'] == 0 and not any(row['before'].values())
            assert prior['container_id'] == row['container_id'] == owner['ready'][lane]['container_id']
            assert prior['guard_inode'] == stop['guard_inode']
            config = spec['service']
            assert pin(config['path']) == config
            runtime = load(Path(config['path']).with_name('launch.py'), 'stopped_' + lane)
            runtime.verify()
            obj = runtime.inspect(spec['container'])
            assert obj['Id'] == row['container_id'] and runtime.C['cards'] == row['cards']
            state = obj['State']
            assert not state['Running'] and not state['OOMKilled'] and state['ExitCode'] == 0
            assert state['FinishedAt'] == row['finished_at']
            mount, = (m for m in obj['Mounts'] if m['Destination'] == '/runtime')
            folder = Path(mount['Source'])/'state'
            assert mount['RW'] and folder.resolve() == folder
            assert folder.parent.is_relative_to(Path(runtime.C['root'])/'runs')
            exit_state = doc(folder/'EXIT.json')
            assert exit_state['exit_code'] == 0 and 'signal:15' in exit_state['stop_reasons']
            policy = folder/'policy'
            suffix = hashlib.sha256(row['finished_at'].encode()).hexdigest()[:16]
            preserved = folder/('policy-stopped-'+suffix)
            if preserved.exists():
                assert not policy.exists()
                active = preserved
            else:
                active = policy
            assert active.is_dir() and active.resolve() == active
            receipts = prior['remote_receipt_sha256']
            assert set(receipts) == {'ACTIVATION.json', 'PRECISION.json', 'CACHE-POLICY.json'}
            # The service writes root-owned mode-0600 files. Read the exact
            # stopped state through its image, with no devices or network.
            observed = subprocess.check_output([
                'docker','run','--rm','--pull=never','--network=none','--read-only',
                '--cap-drop=ALL','--security-opt=no-new-privileges:true','--user=0:0',
                '--mount',f'type=bind,src={active},dst=/previous-policy,readonly',
                '--entrypoint','/usr/local/python3.12.13/bin/python3',runtime.C['image'],
                '-I','-B','-c',READ_STOPPED_POLICY,json.dumps(sorted(receipts)),
            ], timeout=c['timeout'])
            assert json.loads(observed) == receipts, 'stopped policy receipt changed'
            if check_only:
                continue
            if active == policy:
                policy.rename(preserved)
                fd = os.open(folder, os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            save(root/(lane+'-POLICY-ROTATED.json'),
                 dict(container_id=obj['Id'], stopped_at=row['finished_at'], preserved=str(preserved),
                      receipts=prior['remote_receipt_sha256']))


def deployed(c, root):
    ready = doc(root/'READY.json')
    cfg = doc(root/'CONFIG.json')
    assert ready['lanes'].keys() == c['lanes'].keys()
    lanes = {}
    for name, expected in ready['lanes'].items():
        path = root/(name+'-READY.json')
        assert pin(path)['sha256'] == expected
        actual = doc(path)
        version=c.get('policy_version','checkpoint-sha256:'+c['adapter_sha256'])
        assert version in {kind+'-sha256:'+c['adapter_sha256'] for kind in ('initial','checkpoint')}
        assert actual['activation']['version'] == cfg['version'] == version
        assert actual['container_id'] == doc(root/(name+'-STARTED.json'))['container_id']
        assert actual['wire_sha256'] == c['wire']['sha256']
        lanes[name] = actual
    return dict(boot=cfg['boot'],ready=lanes)


def remote(c):
    root=Path(c['output']);root.mkdir(mode=0o700,exist_ok=True)
    if c.get('deployment'):
        owner=deployed(c,Path(c['deployment']))
    else:
        owner=doc(c['handoff']);assert owner['role']=='rollout' and owner['terminal']['guards_releasable']
    assert owner['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    sys.path.insert(0,c['io_utils']);measure=load(c['measure'],'owned_measure')
    verify=load(c['verify'],'owned_readback');wire=doc(c['wire']['path']);assert pin(c['wire']['path'])==c['wire']
    reports={}
    def invoke(item):
        lane,h=item;dst=root/lane;dst.mkdir(exist_ok=True,mode=0o700);cfg=doc(h['service']['path'])
        cid=owner['ready'][lane]['container_id'];assert pin(h['service']['path'])==h['service']
        assert subprocess.check_output(['docker','inspect','-f','{{.Id}}',h['container']],text=True).strip()==cid
        endpoint='http://127.0.0.1:'+str(cfg['port'])
        receipt=dst/'call/RESPONSE.json'
        if not receipt.exists():measure.contact(dst/'call',endpoint,wire,c['timeout'])
        assert doc(receipt)['status']==200
        metrics=measure.metrics(endpoint)
        for state in ('running','waiting'):
            values=[float(l.rsplit(' ',1)[1]) for l in metrics.splitlines() if l.startswith('vllm:num_requests_'+state+'{')]
            assert values and not any(values)
        save(dst/'ROUND.json',dict(requests=[dict(receipt_ref=pin(receipt))]))
        save(dst/'DRAINED.json',dict(guards_releasable=True,unknown_requests=0,terminal=pin(receipt)))
        state='/runtime/state/policy';digests={}
        for name in ('ACTIVATION.json','PRECISION.json','CACHE-POLICY.json'):
            raw=subprocess.check_output(['docker','exec',cid,'cat',state+'/'+name],timeout=15)
            digests[name]=hashlib.sha256(raw).hexdigest()
        return lane,dict(service_config=h['service'],round_result=pin(dst/'ROUND.json'),response_ref=pin(receipt),
            drained=pin(dst/'DRAINED.json'),container=h['container'],model=wire['model'],state_directory=state,
            receipts=digests,adapter_sha256=c['adapter_sha256'],policy_version=c.get('policy_version','checkpoint-sha256:'+c['adapter_sha256']))
    with open(c['guard'],'rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=len(c['lanes'])) as ex:observations=dict(ex.map(invoke,c['lanes'].items()))
    for lane,obs in observations.items():
        rep=verify.observed(obs)
        assert rep['container_id']==owner['ready'][lane]['container_id']
        # Remote verify checks raw hashes and the live service it inspects.
        reports[lane]=dict(report=rep,response=Path(obs['response_ref']['path']).read_text(),
            service=Path(obs['service_config']['path']).read_text())
    return reports

def client(c, *, inspect_restart=False):
    root=Path(c['output']);done=doc(c['handoff'])
    deployed_ids = {lane:doc(Path(c['handoff']).parent/(lane+'-READY.json'))['container_id']
                    for lane,sha in done.get('lanes',{}).items()
                    if pin(Path(c['handoff']).parent/(lane+'-READY.json'))['sha256']==sha}
    if 'lanes' in done: assert deployed_ids.keys()==c['local_ports'].keys()
    argv=['python3','-B',c['remote_script'],c['remote_config'],'--remote']
    if c.get('restart'):
        restart = c['restart']
        assert pin(restart['stop']['path']) == restart['stop']
        argv.extend(['--restart-output', restart['remote_output']])
        if 'ready_seconds' in restart:
            argv.extend(['--restart-ready-seconds', str(restart['ready_seconds'])])
        if restart.get('failed_remote_output'):
            argv.extend(['--failed-restart-output', restart['failed_remote_output']])
        payload = Path(restart['stop']['path']).read_bytes()
        if restart.get('previous_ready'):
            assert not restart.get('failed_remote_output')
            argv.append('--rotate-stopped-policy')
            ready = {lane: doc(ref['path']) for lane, ref in restart['previous_ready'].items()
                     if pin(ref['path']) == ref}
            assert ready.keys() == c['local_ports'].keys()
            payload = json.dumps(dict(stop=json.loads(payload), previous_ready=ready)).encode()
        if inspect_restart:
            argv.append('--inspect-restart')
        reports = subprocess.check_output(['ssh', '-oBatchMode=yes', c['ssh'], shlex.join(argv)],
                                          input=payload,
                                          timeout=min(45, c['timeout']) if inspect_restart else c['timeout'])
    else:
        reports = subprocess.check_output(['ssh','-oBatchMode=yes',c['ssh'],shlex.join(argv)],timeout=c['timeout'])
    reps=json.loads(reports)
    if inspect_restart:
        return reps
    for lane,one in reps.items():
        dst=root/lane;dst.mkdir(mode=0o700,parents=True,exist_ok=True)
        rep=one['report'];assert rep['container_id']==(deployed_ids[lane] if deployed_ids else done['rollout']['ready'][lane]['container_id'])
        def keep(name,raw,sha):
            data=raw.encode();assert hashlib.sha256(data).hexdigest()==sha
            p=dst/name
            if not p.exists():
                with tempfile.NamedTemporaryFile(dir=dst) as f:
                    f.write(data);f.flush();os.fsync(f.fileno());os.link(f.name,p)
                fd=os.open(dst,os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
            result=pin(p);assert result['sha256']==sha
            return result
        state={}
        for name,raw in rep.pop('receipt_bytes').items():state[name]=keep(name,raw,hashlib.sha256(raw.encode()).hexdigest())
        resp=keep('RESPONSE.json',one['response'],rep['response_ref']['sha256'])
        service=keep('SERVICE.json',one['service'],rep['service_config_ref']['sha256'])
        rep.update(local_receipts=state,remote_receipt_sha256={k:v['sha256'] for k,v in state.items()})
        save(dst/'PROVIDER-STARTUP.json',rep)
        save(dst/'LANE.json',dict(lane=lane,activation_ref=state['ACTIVATION.json'],cache_ref=state['CACHE-POLICY.json'],
            precision_ref=state['PRECISION.json'],response_ref=resp))
        source=doc(c['provider_template']);service_cfg=doc(service['path'])
        source.update(lane=lane,service_config=service,service_ready=pin(dst/'PROVIDER-STARTUP.json'),initial_lane_ref=pin(dst/'LANE.json'),
            local_port=c['local_ports'][lane],remote_port=service_cfg['port'],devices=rep['devices'],
            remote_hot_parent=str(Path(service_cfg['root'])/'formal-data'),host=c['ssh'])
        save(dst/'PROVIDER.json',source)
    save(root/'READY.json',dict(handoff_ref=pin(c['handoff']),providers={name:pin(root/name/'PROVIDER.json') for name in reps}))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('config');a.add_argument('--remote',action='store_true')
    a.add_argument('--restart-output')
    a.add_argument('--restart-ready-seconds', type=int)
    a.add_argument('--failed-restart-output')
    a.add_argument('--rotate-stopped-policy', action='store_true')
    a.add_argument('--check-stopped-policy', action='store_true')
    a.add_argument('--inspect-restart', action='store_true')
    v=a.parse_args();c=doc(v.config)
    if v.remote:
        if v.restart_output:
            output = Path(v.restart_output)
            assert output.is_absolute() and output.parent == Path(c['output']).parent
            assert v.restart_ready_seconds is None or v.restart_ready_seconds > 0
            c['output'] = str(output)
            if v.restart_ready_seconds is not None:
                c['restart_ready_seconds'] = v.restart_ready_seconds
            stop = json.load(sys.stdin)
            if v.inspect_restart:
                print(json.dumps(restart_state(c, stop['stop'] if v.rotate_stopped_policy else stop,
                                               check_only=True)))
                sys.exit(0)
            if v.rotate_stopped_policy or v.check_stopped_policy:
                assert not v.failed_restart_output
                rotate_stopped_policy(c, stop['stop'], stop['previous_ready'],
                                      check_only=v.check_stopped_policy)
                if v.check_stopped_policy:
                    print(json.dumps(dict(status='stopped_policy_verified', lanes=sorted(c['lanes']))))
                    sys.exit(0)
                stop = stop['stop']
            if v.failed_restart_output:
                failed = Path(v.failed_restart_output)
                assert failed.is_absolute() and failed.parent == output.parent and failed != output
                stop = reconcile_activation_collision(c, stop, failed)
            restore(c, stop)
        print(json.dumps(remote(c)))
    elif v.inspect_restart:
        print(json.dumps(client(c, inspect_restart=True)))
    else:
        client(c)
