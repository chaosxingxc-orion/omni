"""Render a retained-input replay into the sealed DP8 guard/watchdog lifecycle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def checked(ref):
    p=Path(ref['path'])
    if hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('source changed')
    return json.loads(p.read_bytes())


def save(p, obj):
    with p.open('x') as f:json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)


def state(cid):
    return json.loads(subprocess.check_output(['docker','inspect',cid]))[0]['State']


def render(spec,root):
    if 'previous_replay' in spec:
        prev=Path(spec['previous_replay'])
        cid=json.loads((prev/'CONTROLLER.json').read_bytes())['container_id']
        subprocess.run(['docker','wait',cid],check=True,stdout=subprocess.DEVNULL)
        prev_control=prev/'hot/new-lease'
        watchdog=json.loads((prev_control/'WATCHDOG.json').read_bytes())['container_id']
        subprocess.run(['docker','wait',watchdog],check=True,stdout=subprocess.DEVNULL)
        assert state(cid)['ExitCode']==0 and state(watchdog)['ExitCode']==0
        assert json.loads((prev_control/'RESTORED.json').read_bytes())['config_exact']
        assert json.loads((prev_control/'NPU-EXIT.json').read_bytes())['ExitCode']!=0
    src=Path(spec['parent_plan']).resolve()
    if hashlib.sha256(src.read_bytes()).hexdigest()!=spec['parent_sha256']:raise ValueError('parent plan changed')
    plan=json.loads(src.read_bytes()); parent=Path(plan['root']); old=parent/'hot/new-lease'
    assert src==parent/'LIFECYCLE-PLAN.json' and root.is_relative_to(Path(plan['storage_scope']['root']))
    restored=json.loads((old/'RESTORED.json').read_bytes())
    assert restored['config_exact'] and len(restored['services'])==2
    for cid in [spec['parent_controller'],json.loads((old/'WATCHDOG.json').read_bytes())['container_id'],
                json.loads((old/'NPU-LAUNCH.json').read_bytes())['container_id']]:
        assert state(cid)['Status']=='exited'
    run=checked(plan['run_ref']); runtime=checked(run['runtime_ref'])
    src_inputs=Path(spec['inputs']).resolve()
    for name,digest in spec['input_files'].items():
        assert Path(name).name==name and hashlib.sha256((src_inputs/name).read_bytes()).hexdigest()==digest
    assert src_inputs.is_relative_to(Path(plan['storage_scope']['root']))
    image=runtime['training']['image_id']
    name='retained-audio-'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    hot=root/'hot'
    root.mkdir(mode=0o700)
    for path in ('tmp','cache','home/ascend/log','ascend-work','ascend-cache','hf','logs','triton','inductor'):
        (hot/path).mkdir(parents=True,exist_ok=True)
    a=plan['npu_argv']; ix=a.index(image); assert a[:2]==['docker','run']
    argv=[s.replace(str(parent/'hot'),str(hot)) for s in a[:ix]]
    argv[argv.index('--name')+1]=name
    argv.extend(['--mount',f'type=bind,src={src_inputs},dst=/e,readonly',
                 '--mount',f'type=bind,src={hot},dst=/hot','--mount',
                 f"type=bind,src={runtime['model']['path']},dst=/model,readonly",image,
                 'timeout','300','python3','-B','/e/retained_audio_hf_boundary.py'])
    now=time.time(); t=spec['compute_seconds']; reserve=spec['restore_seconds']
    assert 600<t<=3600 and 600<=reserve<=3600
    plan.update(root=str(root),phase_names=[name],npu_argv=argv,lease_started=now,compute_seconds=t,
                total_seconds=t+reserve,restore_seconds=reserve,absolute_compute_deadline=now+t,
                absolute_deadline=now+t+reserve)
    # A watchdog's original mounts only covered its restore duties. Both its
    # dispatch sibling and new watchdog need the same physical accounting roots.
    def accounting_mounts(argv):
        dest = {v.split('dst=',1)[1].split(',')[0] for v in argv if v.startswith('type=bind,') and 'dst=' in v}
        return [f'type=bind,src={src},dst={src}' + ('' if src in run['lifecycle']['hot_paths'] else ',readonly') for src in dict.fromkeys([*plan['storage_scope']['auxiliary_roots'],*run['lifecycle']['hot_paths']]) if src not in dest]
    plan['watchdog_mounts'] += accounting_mounts(plan['watchdog_mounts'])
    save(root/'LIFECYCLE-PLAN.json',plan)
    # Reuse the CPU watchdog's exact closed image, binds, user and Docker tools;
    # start its sibling dispatch routine, which creates its OWN detached watchdog.
    w=json.loads((old/'WATCHDOG.json').read_bytes())['argv']
    argv=[s.replace(str(parent/'hot'),str(hot)) for s in w]
    i=argv.index('--name');argv[i+1]=name+'-controller'
    argv[argv.index('--watchdog'):]=['--dispatch-plan',str(root/'LIFECYCLE-PLAN.json')]
    idx=argv.index('--entrypoint')
    argv[idx:idx]=['--mount',f'type=bind,src={root},dst={root}']
    for mount in accounting_mounts(argv):
        argv[idx:idx]=['--mount',mount]
    save(root/'CONTROLLER-INTENT.json',dict(argv=argv,parent_plan_sha256=spec['parent_sha256']))
    cid=subprocess.check_output(argv,text=True).strip()
    save(root/'CONTROLLER.json',dict(container_id=cid));print(cid)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    s=json.loads(args.config.read_bytes());render(s,Path(s['root']).resolve())
