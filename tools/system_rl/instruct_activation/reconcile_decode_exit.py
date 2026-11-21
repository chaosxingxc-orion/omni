"""Close a failed paired measurement whose metrics waiter outlived its engine."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from io_utils import canonical, doc, pin, require, save, storage
from measure_existing_services import grouped_rounds, idle, metrics, service_identity


def reconcile(path, pid):
    cfg = json.loads(path.read_bytes())
    root = Path(cfg['output'])
    storage(root)
    guard, = cfg['guards']
    started = json.loads((root/'STARTED.json').read_bytes())
    require(started['config_ref'] == pin(path) and cfg['source_boot'] ==
            Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'config or boot changed')
    plan = list(grouped_rounds(cfg, cfg['services']))
    expected = {}
    for i, (_, _, roster, _) in enumerate(plan):
        for service in cfg['services']:
            for group, replica, wire in roster[service['name']]:
                expected[root/f'round-{i}'/service['name']/f'{group}-{replica}'] = (service, wire)
    require(set(expected) == {p.parent for p in root.rglob('INTENT.json')} ==
            {p.parent for p in root.rglob('RESPONSE.json')} and
            not list(root.rglob('UNKNOWN.json')), 'incomplete or unknown requests')
    outcomes = []
    for directory, (service, wire) in expected.items():
        intent, response = (doc(pin(directory/n)) for n in ('INTENT.json','RESPONSE.json'))
        require(intent['wire_sha256'] == hashlib.sha256(canonical(wire)).hexdigest() and
                intent['endpoint'] == service['endpoint'] and response['response_sha256'] ==
                hashlib.sha256(response['raw'].encode()).hexdigest(), 'request identity changed')
        outcomes.append(dict(receipt=pin(directory/'RESPONSE.json'),status=response['status']))
    exited = {}
    for service in cfg['services']:
        name = service['container']
        info, = json.loads(subprocess.check_output(['docker','inspect',name]))
        require(info['Name']=='/'+name,'container mismatch')
        if info['State']['Status']=='exited':
            require(info['State']['Pid']==0 and info['RestartCount']==0,'invalid engine exit')
            exited[name]=info['Id']
        else:
            require(info['State']['Pid']==service['pid'],'service replaced')
            service_identity(service)
            idle(metrics(service['endpoint']))
    require(exited and any(r['status']>=400 for r in outcomes),'failed measurement required')
    # Pin the waiter before checking ownership, so PID reuse cannot signal a successor.
    fd = os.pidfd_open(pid)
    try:
        proc = Path('/proc')/str(pid)
        argv = (proc/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        require(len(argv)==2 and Path(os.fsdecode(argv[0])).name=='python3' and
                (Path(os.readlink(proc/'cwd'))/os.fsdecode(argv[1])).resolve()==path.parent/'run.py',
                'unexpected waiter')
        locks = subprocess.check_output(['lslocks','-nr','-o','PID,PATH'],text=True)
        require(f'{pid} {guard}' in locks.splitlines(), 'waiter does not hold this guard')
        signal.pidfd_send_signal(fd,signal.SIGTERM)
        with open(guard,'rb') as lock:
            end=time.monotonic()+10
            while True:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic()>=end:raise
                    time.sleep(.1)
            save(root/'DRAINED.json',dict(guards_releasable=True,passed=False,unknown_requests=0,
                outcomes=outcomes,exited_containers=exited,source_boot=cfg['source_boot'],
                config_ref=pin(path),finished_epoch=time.time()))
    finally:
        os.close(fd)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--waiter',type=int,required=True)
    a=p.parse_args()
    reconcile(a.config.resolve(),a.waiter)
