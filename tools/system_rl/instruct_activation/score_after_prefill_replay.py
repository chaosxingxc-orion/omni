"""Hand the drained performance lanes to their restored FSDP8 scorer under its guard."""
import concurrent.futures
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from cutover import load
from io_utils import doc, pin, require, save
from measure_existing_services import metrics, idle


def run(c):
    cfg = doc(c['train_config'])
    root = Path(cfg['hot']).parent
    rounds = Path(c['rounds'])
    deadline = time.monotonic()+c['wait_seconds']
    while not (rounds/'DRAINED.json').exists():
        require(time.monotonic()<deadline,'replay not drained')
        time.sleep(5)
    require(doc(pin(rounds/'RESULT.json'))['passed'] and not list(rounds.rglob('UNKNOWN.json')) and
            doc(pin(rounds/'DRAINED.json'))['guards_releasable'],'replay not fully reconciled')
    require(not (root/'hot').exists() and not (root/'STOP-INTENT.json').exists(), 'reconcile preceding handoff')
    lanes=[]
    with open(cfg['existing_guard'],'rb') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for i, spec in enumerate(c['lanes']):
            launcher=load(spec['code'],f'panel_lane_{i}')
            require(cfg['boot_id']==launcher.C['host_boot_id'] and cfg['existing_guard']==launcher.C['guard'],
                    'wrong device owner')
            cid=doc(pin(spec['started']))['container_id']
            state=launcher.inspect(launcher.name(spec['attempt']))
            require(state['Id']==cid and state['State']['Running'], 'service identity changed')
            idle(metrics(f"http://{launcher.C['host']}:{launcher.C['port']}"))
            lanes.append((launcher,cid))
        require(sorted(l.C['cards'] for l,_ in lanes)==[[0,1,2,3],[4,5,6,7]] and
                cfg['devices']==list(range(8)), 'incomplete device handoff')
        save(root/'STOP-INTENT.json',dict(containers=[cid for _,cid in lanes],
             source=pin(rounds/'DRAINED.json'),training=c['train_config']))
        def stop(pair):
            launcher,cid=pair
            launcher.command(['docker','stop','-t',str(launcher.C['stop_timeout_seconds']+15),cid],
                             timeout=launcher.C['stop_timeout_seconds']+45)
            state=launcher.inspect(cid)
            require(state['Id']==cid and state['State']['Status']=='exited','stop not reconciled')
            return dict(cid=cid,exit=state['State']['ExitCode'])
        with concurrent.futures.ThreadPoolExecutor(len(lanes)) as pool:
            stopped=list(pool.map(stop,lanes))
        save(root/'INFERENCE-DRAINED.json',dict(stopped=stopped))
    result=subprocess.run([sys.executable,'-B',str(root/'launch_retained_hf_fsdp.py'),
                           '--config',c['train_config']['path']],cwd=root)
    # A failed worker may still own devices. Restore only after its container exits.
    terminal=doc(pin(root/'hot/EXIT.json'))
    if terminal['container_id']:
        state,=json.loads(subprocess.check_output(['docker','inspect',terminal['container_id']]))
        require(state['Id']==terminal['container_id'] and state['Image']==cfg['image'], 'wrong training container')
        require(state['State']['Status']=='exited','training container still active')
    b=c['restore_control']
    launch=Path(b['code'])/'launch.py'
    subprocess.run([sys.executable,'-B',str(launch),'start','--attempt',b['attempt'],
                    '--preflight-attempt',b['preflight']],check=True)
    control=load(b['code'],'restored_native')
    service=control.inspect(control.name(b['attempt']))
    save(root/'CONTROL-STARTED.json',dict(container_id=service['Id'],physical_devices=control.C['cards']))
    require(result.returncode==0 and terminal['exit_code']==0 and terminal['complete_ranks']==8,
            'score worker failed')


if __name__=='__main__':
    run(doc(pin(sys.argv[1])))
