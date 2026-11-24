"""Replace only the experiment's CPU controller during an acknowledged remote update."""
import argparse, fcntl, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path


def read(p):return json.loads(Path(p).read_bytes())
def checked(r):
    p=Path(r['path']);raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=r['sha256'] or len(raw)!=r['bytes']:raise ValueError('changed pinned source')
    return p
def ensure(ok,msg):
    if not ok:raise ValueError(msg)


def switch(spec_path,launcher,evidence):
    from report_training_updates import controller_guard_owner
    spec=read(spec_path); runner=checked(spec['runner_ref']); it=checked(spec['iteration_ref'])
    cfg=read(it); root=Path(cfg['output_root']); out=spec_path.parent
    acceptance=read(evidence)
    ensure(acceptance['config_sha256']==hashlib.sha256(spec_path.read_bytes()).hexdigest()
           and acceptance['passed'] is True,'not the accepted candidate')
    path=out/'HANDOFF.json'
    ensure(not path.exists(),'handoff already recorded; inspect controller')
    def boundary():
        cur=read(root/'CURSOR.json')
        if cur['phase']!='update' or not cur['pending_stage']:return None
        step=cur['step']+1;d=root/f'step-{step:06}'/'update'
        if not (d/'COMMAND-ACK.json').is_file() or (d/'returned').exists() or (d/'closed-return.tar').exists():return None
        ack=read(d/'COMMAND-ACK.json');cmd=read(checked(ack['command_ref']));cap=read(checked(cmd['generation_inventory_ref']))
        ensure(cmd==read(d/'SUBMIT-INTENT.json')['command'] and ack['remote']['sequence']==cmd['sequence']
               and cmd['checkpoint_sha256']==cur['resume_ref']['sha256'] and cap['closed']
               and cmd['generation_inventory_ref']==cur['capture']['generation_inventory_ref'],'unmatched actor ACK')
        drained=read(Path(cap['cohort_root']).parent/'DRAINED.json')
        ensure(drained['terminal_verified'] is True and drained['unknown_requests']==0 and
               drained['generation_closed_ref']==cap['generation_closed_ref'] and checked(cap['generation_closed_ref'])
               ==Path(cap['cohort_root'])/'GENERATION-CLOSED.json' and drained['usage']['unknown']==0 and not (root/'resident-lease/REVOKED.json').exists(),'unsettled capture')
        return dict(step=step,ack=ack,prior=cur['resume_ref'],generation=cmd['generation_inventory_ref'])
    b=boundary()
    if b is None:return False
    pid=controller_guard_owner(root,it);ensure(pid is not None,'iteration holder disappeared')
    cgroup=Path(f'/proc/{pid}/cgroup').read_text()
    import re
    ids=set(re.findall(r'/docker/([0-9a-f]{64})',cgroup));ensure(len(ids)==1,'ambiguous control container')
    cid=ids.pop()
    def state():return json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',cid]))
    ensure(state()['Pid']==pid and state()['Running'],'holder is not this container init')
    fd=os.pidfd_open(pid)
    signal.pidfd_send_signal(fd,signal.SIGSTOP)
    killed=False
    try:
        for _ in range(100):
            if Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]=='T':break
            time.sleep(.01)
        ensure(Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]=='T','could not freeze CPU owner')
        if boundary()!=b:return False
        ensure(controller_guard_owner(root,it)==pid,'guard ownership changed')
        cfglog=dict(**b,previous_pid=pid,container_id=cid,iteration=spec['iteration_ref'],candidate=str(spec_path),
                       acceptance_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
        with path.open('x') as f:json.dump(cfglog,f);f.flush();os.fsync(f.fileno())
        subprocess.run(['docker','kill','--signal','KILL',cid],check=True,stdout=subprocess.DEVNULL)
        killed=True
    finally:
        if not killed:
            try:signal.pidfd_send_signal(fd,signal.SIGCONT)
            except ProcessLookupError:pass
        os.close(fd)
    end=time.monotonic()+30
    while controller_guard_owner(root,it) is not None and time.monotonic()<end:time.sleep(.2)
    ensure(controller_guard_owner(root,it) is None,'original iteration still owns guard')
    with (out/'controller.log').open('ab') as log:
        proc=subprocess.Popen([sys.executable,'-B',str(launcher),str(runner),str(checked(spec['sources']['transition_controller'])),str(spec_path)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    print(json.dumps(dict(controller_launch_pid=proc.pid,adopted_step=b['step'])),flush=True)
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('config','launcher','evidence'):p.add_argument(k,type=Path)
    a=p.parse_args()
    with (a.config.parent/'handoff.guard').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not switch(a.config,a.launcher,a.evidence):time.sleep(4)
