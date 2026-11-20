"""One detached same-host diagnostic: hold the existing guard through worker exit."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dispatch_retained_hf_boundary import save


def main(config):
    cfg = json.loads(config.read_bytes())
    bundle = config.parent
    stage = bundle/'tree'
    hot = Path(cfg['hot'])
    impl = stage/'parents/jev-system-rl/implementation'
    sys.path.insert(0, str(impl))
    from storage import environment, check_running
    devices = cfg['devices']
    world = cfg['training']['world_size']
    if world not in (4, 8) or devices not in ([0,1,2,3],[4,5,6,7],list(range(8))) or len(devices)!=world:
        raise ValueError('expected contiguous visible ranks for native FSDP')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot != cfg['boot_id']:
        raise ValueError('lease boot differs')
    image = cfg['image']
    if subprocess.check_output(['docker','image','inspect','-f','{{.Id}}',image],text=True).strip()!=image:
        raise ValueError('training image not pinned')
    for name in ('retained_hf_fsdp_control.py','worker_dp8.py','forward.py','validate_npu.py'):
        path = dict(
            worker_dp8=impl/'worker_dp8.py',
            forward=stage/'parents/native-model-integration/forward.py',
            validate_npu=stage/'parents/native-training-successor/validate_npu.py',
        ).get(Path(name).stem,bundle/name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=cfg['source_pins'][name]:
            raise ValueError('changed diagnostic source: '+name)
    if 'SOURCE-PINS.json' in cfg['source_pins'] and hashlib.sha256((impl/'SOURCE-PINS.json').read_bytes()).hexdigest()!=cfg['source_pins']['SOURCE-PINS.json']:
        raise ValueError('changed checkpoint source manifest')
    with Path(cfg['existing_guard']).open('r+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        smi = subprocess.check_output(['npu-smi','info'],text=True)
        if any(f'No running processes found in NPU {i} ' not in smi for i in devices):
            raise ValueError('NPU has other processes: decline launch')
        hot.mkdir(mode=0o700)
        paths = environment(hot,cfg['storage_budget_bytes'])
        args = ['docker','run','--detach','--init','--read-only','--network','host',
                '--shm-size','2g','--log-driver','json-file','--log-opt','max-size=50m','--log-opt','max-file=2',
                '--user',f'{os.getuid()}:{os.getgid()}','--name',cfg['name'],'--workdir','/omni-hot']
        for src,dst,ro in [(cfg['shared_root'],cfg['shared_root'],True),
                           ('/usr/local/Ascend/driver','/usr/local/Ascend/driver',True),
                           ('/etc/passwd','/etc/passwd',True),('/etc/group','/etc/group',True),
                           (bundle,'/sealed',True),(hot,'/omni-hot',False)]:
            args += ['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
        if 'restore_checkpoint' in cfg:
            cp=Path(cfg['restore_checkpoint']['directory'])
            if (cp.resolve()!=cp or cp.parents[2]!=bundle.parent or
                    hashlib.sha256((cp/'COMMITTED.json').read_bytes()).hexdigest()!=cfg['restore_checkpoint']['sha256']):
                raise ValueError('closed checkpoint differs from the staged restore source')
            args += ['--mount',f'type=bind,src={cp},dst=/prior-checkpoint,readonly']
        for device in [*(f'davinci{i}' for i in devices),'davinci_manager','devmm_svm','hisi_hdc']:
            args += ['--device','/dev/'+device]
        for key,value in os.environ.items():
            if value==str(hot) or value.startswith(str(hot)+'/'):
                args += ['-e',key+'=/omni-hot'+value[len(str(hot)):]]
        for key in ('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','TORCH_DEVICE_BACKEND_AUTOLOAD',
                    'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TOKENIZERS_PARALLELISM'):
            args += ['-e',key+'='+os.environ[key]]
        args += ['-e','PYTHONPATH=/sealed/tree/parents/jev-system-rl/implementation:/sealed/tree/jev/src:/opt/verl',
                 '-e','TRANSFORMERS_CACHE=/omni-hot/hf/transformers','-e','ASCEND_RT_VISIBLE_DEVICES='+','.join(map(str,range(world)))]
        for key in ('HCCL_SOCKET_IFNAME','GLOO_SOCKET_IFNAME','HCCL_IF_BASE_PORT'):
            args += ['-e',key+'='+str(cfg['communication'][key])]
        args += [image,'python3','-B','-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}',
                 '/sealed/retained_hf_fsdp_control.py','--config','/sealed/'+config.name,'--input','/sealed','--output','/omni-hot']
        save(hot/'INTENT.json',dict(image=image,devices=devices,boot_id=boot,guard_inode=os.fstat(guard.fileno()).st_ino,
                                    storage=paths,config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()))
        cid = subprocess.check_output(args,text=True).strip()
        save(hot/'CONTAINER.json',dict(container_id=cid))
        deadline = time.monotonic()+cfg['timeout_seconds']
        failure = None
        try:
            while True:
                try:
                    code = int(subprocess.check_output(['docker','wait',cid],text=True,timeout=30))
                    break
                except subprocess.TimeoutExpired:
                    check_running(hot,cfg['storage_budget_bytes'])
                    if time.monotonic()>deadline:
                        raise TimeoutError('bounded same-input forward deadline')
        except BaseException as exc:
            failure = repr(exc)
            subprocess.run(['docker','stop','-t','30',cid],check=True)
            code = int(subprocess.check_output(['docker','wait',cid],text=True))
        save(hot/'EXIT.json',dict(container_id=cid,exit_code=code,failure=failure,
                                 complete_ranks=sum((hot/f'rank-{i}'/'RESULT.json').is_file() for i in range(world))))
        if code or failure:
            raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    main(p.parse_args().config.resolve())
