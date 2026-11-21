"""Replace drained project instances under the original local launch guard."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import urllib.request
from io_utils import pin, save


def load(code, name):
    spec=importlib.util.spec_from_file_location(name,Path(code)/'launch.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify()
    return module


def switch(c):
    root=Path(c['output'])
    root.mkdir(mode=0o700)
    guard=Path(c['guard'])
    with guard.open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        terminal=json.loads(Path(c['terminal']).read_bytes())
        assert terminal['guards_releasable'] and not list(Path(c['terminal']).parent.rglob('UNKNOWN.json'))
        if 'training_terminal' in c:
            training=json.loads(Path(c['training_terminal']).read_bytes())
            assert training['terminal_verified'] is True and training['unknown_requests']==0
        rows=[]
        for n,lane in enumerate(c['lanes']):
            old=load(lane['old_code'],f'old{n}')
            new=load(lane['new_code'],f'new{n}')
            assert all(old.C[k]==new.C[k] for k in ('host_boot_id','guard','cards','image','port','host','model'))
            assert Path(new.C['guard'])==guard
            assert new.C['host']=='127.0.0.1' and new.C['cards'] in ([0,1,2,3],[4,5,6,7])
            before=old.inspect(old.name(lane['old_attempt']))
            assert before['State']['Running']
            assert not (Path(new.C['root'])/'runs'/lane['new_attempt']).exists()
            pre=Path(new.C['root'])/'runs'/lane['preflight']/'state/PREFLIGHT.json'
            assert json.loads(pre.read_bytes())['config_sha256']==hashlib.sha256((new.CODE/'config.json').read_bytes()).hexdigest()
            with urllib.request.urlopen(f"http://127.0.0.1:{new.C['port']}/metrics",timeout=8) as response:
                raw=response.read().decode()
            for state in ('waiting','running'):
                values=[float(row.rsplit(' ',1)[1]) for row in raw.splitlines()
                        if row.startswith('vllm:num_requests_'+state+'{')]
                assert values and all(v==0 for v in values),'active request'
            rows.append((lane,before,new,raw,Path(old.C['root'])/'runs'/lane['old_attempt']/'state/EXIT.json'))
        assert sorted(x[2].C['cards'] for x in rows)==[[0,1,2,3],[4,5,6,7]]
        save(root/'INTENT.json',dict(predecessors=[dict(container_id=x[1]['Id'],
            cards=x[2].C['cards'],metrics=x[3]) for x in rows],terminal_ref=pin(c['terminal']),
            training_terminal_ref=pin(c['training_terminal']) if 'training_terminal' in c else None,
            source_boot=rows[0][2].C['host_boot_id'],started=time.time(),automatic_retry=False))
        for lane,before,new,_,predecessor in rows:
            subprocess.run(['docker','stop','-t',str(new.C['stop_timeout_seconds']+15),before['Id']],
                           check=True,timeout=new.C['stop_timeout_seconds']+25,capture_output=True)
            state=new.command(['docker','inspect','-f','{{json .State}}',before['Id']])
            assert json.loads(state)['Status']=='exited'
            save(root/f"STOPPED-{new.C['port']}.json",dict(container_id=before['Id'],state=json.loads(state),
                service_exit=pin(predecessor)))
            new.start_checks()
            argv=new.argv(lane['new_attempt'],True)
            save(root/f"START-INTENT-{new.C['port']}.json",dict(container=new.name(lane['new_attempt']),automatic_retry=False))
            cid=new.command(argv)
            save(root/f"STARTED-{new.C['port']}.json",dict(container_id=cid,physical_devices=new.C['cards']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    switch(json.loads(p.parse_args().config.read_bytes()))
