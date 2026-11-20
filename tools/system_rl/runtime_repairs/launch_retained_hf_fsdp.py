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


def capture_mounts(cfg, bundle):
    """Bind only sealed continuation inputs and its local writable index."""
    if 'generation' not in cfg:
        return []
    capture_root = Path(cfg['capture_mounts']['root'])
    if capture_root != capture_root.resolve() or capture_root.parent != bundle.parent:
        raise ValueError('capture inputs must be a sibling local bundle')
    from capture_runtime import capture_runtime
    allowed = capture_runtime(cfg)['targets']
    index = Path(cfg['capture_mounts']['index_host_root'])
    if not index.is_absolute() or index != index.resolve() or not index.is_dir() or (
            index.is_relative_to(capture_root) or capture_root.is_relative_to(index)
            or index.is_relative_to(bundle) or bundle.is_relative_to(index)):
        raise ValueError('capture index must be a separate existing local directory')
    from storage import storage
    storage.check_paths({'capture_index':index},min_free_bytes=20*1024**3+cfg['storage_budget_bytes'])
    args = []
    for name,destination in allowed.items():
        path = index if name == 'index' else capture_root/name
        if path.resolve()!=path or not path.is_dir():
            raise ValueError('missing closed capture inputs: '+name)
        args += ['--mount',f'type=bind,src={path},dst={destination}'+('' if name=='index' else ',readonly')]
    for name,expected in sorted(cfg['capture_mounts']['overlays'].items()):
        relative = Path(name)
        path = capture_root/'overlays'/relative
        if (relative.is_absolute() or '..' in relative.parts or relative.suffix!='.py'
                or path.resolve()!=path or hashlib.sha256(path.read_bytes()).hexdigest()!=expected):
            raise ValueError('scoring source overlay differs: '+name)
    for name,expected in cfg['capture_mounts']['tools'].items():
        path = capture_root/'tools'/name
        if Path(name).name!=name or path.resolve()!=path or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('capture verification tool differs: '+name)
    return args


def verify_staged_inputs(cfg, bundle):
    for name, expected in (cfg['files'] | cfg.get('configuration_sha256', {})).items():
        if name == 'finalized_generation' and 'generation' in cfg:
            if cfg['generation']['generation_inventory_ref']['sha256'] != expected:
                raise ValueError('staged generation inventory drift')
            continue
        if Path(name).name != name:
            raise ValueError(f'invalid staged input name: {name}')
        path = bundle/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'staged input drift: {name}')


def main(config,check_capture=False):
    config_bytes = config.read_bytes()
    cfg = json.loads(config_bytes)
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    bundle = config.parent
    verify_staged_inputs(cfg, bundle)
    preflight_receipt = bundle/'CAPTURE-PREFLIGHT.json'
    if 'generation' in cfg and not check_capture:
        proof = json.loads(preflight_receipt.read_bytes())
        if (proof['train_config_sha256'] != config_sha or
                proof['inventory_sha256'] != cfg['generation']['generation_inventory_ref']['sha256']):
            raise ValueError('capture preflight belongs to a different configuration or inventory')
    stage = bundle/'tree'
    hot = Path(cfg['hot'])
    impl = stage/'parents/jev-system-rl/implementation'
    sys.path.insert(0, str(impl))
    from storage import environment, storage
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
    if 'generation' in cfg:
        sys.path.insert(0,str(stage/'jev/src'))
        from dependencies import source_pins
        source_pins()
    from contextlib import nullcontext
    if check_capture:
        if 'generation' not in cfg:
            raise ValueError('no closed generation to verify')
        hot = bundle/'capture-preflight'
    with (nullcontext() if check_capture else Path(cfg['existing_guard']).open('r+')) as guard:
        if not check_capture:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            smi = subprocess.check_output(['npu-smi','info'],text=True)
            if any(f'No running processes found in NPU {i} ' not in smi for i in devices):
                raise ValueError('NPU has other processes: decline launch')
        hot.mkdir(mode=0o700)
        storage.check_paths({'runtime':hot},min_free_bytes=20*1024**3+cfg['storage_budget_bytes'])
        paths = environment(hot,cfg['storage_budget_bytes'])
        args = ['docker','run',*(['--rm'] if check_capture else ['--detach']),'--init','--read-only','--network','host',
                '--shm-size','2g','--log-driver','json-file','--log-opt','max-size=50m','--log-opt','max-file=2',
                '--user',f'{os.getuid()}:{os.getgid()}','--name',cfg['name']+('-check' if check_capture else ''),'--workdir','/omni-hot']
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
        args += capture_mounts(cfg,bundle)
        if not check_capture:
            for device in [*(f'davinci{i}' for i in devices),'davinci_manager','devmm_svm','hisi_hdc']:
                args += ['--device','/dev/'+device]
        for key,value in os.environ.items():
            if value==str(hot) or value.startswith(str(hot)+'/'):
                args += ['-e',key+'=/omni-hot'+value[len(str(hot)):]]
        for key in ('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','TORCH_DEVICE_BACKEND_AUTOLOAD',
                    'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TOKENIZERS_PARALLELISM'):
            args += ['-e',key+'='+os.environ[key]]
        tools=':'+cfg['capture_mounts']['logical_roots']['tools'] if 'generation' in cfg else ''
        args += ['-e','PYTHONPATH=/sealed/tree/parents/jev-system-rl/implementation:/sealed/tree/jev/src:/opt/verl'+tools,
                 '-e','TRANSFORMERS_CACHE=/omni-hot/hf/transformers','-e','ASCEND_RT_VISIBLE_DEVICES='+','.join(map(str,range(world)))]
        for key in ('HCCL_SOCKET_IFNAME','GLOO_SOCKET_IFNAME','HCCL_IF_BASE_PORT'):
            args += ['-e',key+'='+str(cfg['communication'][key])]
        if check_capture:
            args += [image,'python3','-B','-c','import json,sys;from pathlib import Path;sys.path.insert(0,"/sealed");from bounded_actor_step import capture_preflight;print(json.dumps(capture_preflight(json.load(open(sys.argv[1])),Path("/omni-hot"))))','/sealed/'+config.name]
            result = json.loads(subprocess.check_output(args,text=True,timeout=600).splitlines()[-1])
            if not (result['groups'] > 0 and result['inventory_files'] > 0):
                raise ValueError('capture preflight did not verify a real cohort')
            save(preflight_receipt,dict(result,train_config_sha256=config_sha,
                inventory_sha256=cfg['generation']['generation_inventory_ref']['sha256']))
            print(json.dumps(result))
            return
        args += [image,'python3','-B','-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}',
                 '/sealed/retained_hf_fsdp_control.py','--config','/sealed/'+config.name,'--input','/sealed','--output','/omni-hot']
        save(hot/'INTENT.json',dict(image=image,devices=devices,boot_id=boot,guard_inode=os.fstat(guard.fileno()).st_ino,
                                    storage=paths,config_sha256=config_sha))
        try:
            cid = subprocess.check_output(args,text=True).strip()
        except subprocess.CalledProcessError as exc:
            save(hot/'EXIT.json',dict(container_id=None,exit_code=exc.returncode,
                 failure='container launch failed; no worker started',complete_ranks=0))
            raise
        save(hot/'CONTAINER.json',dict(container_id=cid))
        deadline = time.monotonic()+cfg['timeout_seconds']
        failure = None
        try:
            while True:
                try:
                    code = int(subprocess.check_output(['docker','wait',cid],text=True,timeout=30))
                    break
                except subprocess.TimeoutExpired:
                    storage.check_paths({'runtime':hot},min_free_bytes=20*1024**3+cfg['storage_budget_bytes'])
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
    p.add_argument('--check-capture',action='store_true')
    a=p.parse_args()
    main(a.config.resolve(),a.check_capture)
