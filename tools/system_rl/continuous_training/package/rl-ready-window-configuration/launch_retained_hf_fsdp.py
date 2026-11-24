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


def training_memory_bytes(cfg):
    gib = cfg.get('training_memory_gib')
    if type(gib) is not int or gib <= 1024:
        raise ValueError('training_memory_gib must be an integer above 1024 GiB')
    return gib << 30


def start_container(args, memory_bytes, hot):
    if memory_bytes is None:
        return subprocess.check_output(args, text=True).strip()
    cid = subprocess.check_output(args, text=True).strip()
    try:
        observed = int(subprocess.check_output(
            ['docker', 'inspect', '--format', '{{.HostConfig.Memory}}', cid], text=True))
        if observed != memory_bytes:
            raise ValueError(f'Docker training memory is {observed} bytes; expected {memory_bytes}')
        subprocess.run(['docker', 'start', cid], check=True, stdout=subprocess.DEVNULL)
    except (ValueError, subprocess.SubprocessError, OSError) as exc:
        save(hot/'EXIT.json', dict(container_id=cid, exit_code=1, failure=str(exc), complete_ranks=0))
        raise
    save(hot/'CONTAINER.json', dict(container_id=cid, memory_limit_bytes=observed))
    return cid


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
    formal_roles = cfg['capture_mounts'].get('formal_input_roles', [])
    for name,destination in allowed.items():
        if name == 'index':
            path = index
        elif name in formal_roles:
            key='resident_input_root' if 'continuous_actor' in cfg else 'formal_input_root'
            path = Path(cfg['capture_mounts'][key])/name
            if path.parent.parent != capture_root.parent:
                raise ValueError('formal generation inputs must be a sibling local bundle')
            storage.check_paths({'formal_inputs':path})
        else:
            path = capture_root/name
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
    runtime = capture_runtime(cfg)
    for role in ('consumer_source','consumer_wrapper'):
        if role in runtime:
            args += ['-e','OMNI_CAPTURE_'+role.upper()+'='+runtime[role]]
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


def main(config,check_capture=False,timeout_recovery=None):
    entered = time.monotonic()
    config_bytes = config.read_bytes()
    cfg = json.loads(config_bytes)
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    is_training = ('training_memory_gib' in cfg or
                   ('accepted_fields' in cfg and
                    not any(key in cfg for key in ('updated_http_parity', 'parity_restore_step', 'efficiency'))))
    memory_bytes = training_memory_bytes(cfg) if is_training else None
    if memory_bytes is not None:
        mem_total_kib = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                 if line.startswith('MemTotal:')))
        if memory_bytes > mem_total_kib * 1024:
            raise ValueError('configured training memory exceeds host RAM')
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
        name = cfg['name']
        timeout_seconds = cfg['timeout_seconds']
        if timeout_recovery is None:
            hot.mkdir(mode=0o700)
            if not check_capture and cfg.get('closed_score_replay'):
                reuse_closed_scores(cfg,bundle,hot,guard)
        else:
            from recover_training_timeout import resume_scoring
            storage.check_paths({'runtime':bundle}, min_free_bytes=20*1024**3+cfg['storage_budget_bytes'])
            recovered = resume_scoring(config, cfg, hot, guard, timeout_recovery)
            name += ('-population-recovery' if recovered['schema'] == 'formal-actor-recording-population-recovery-v1'
                     else '-timeout-recovery')
            timeout_seconds = recovered['timeout_seconds']
        storage.check_paths({'runtime':hot},min_free_bytes=20*1024**3+cfg['storage_budget_bytes'])
        paths = environment(hot,cfg['storage_budget_bytes'])
        limit = None if check_capture else memory_bytes
        operation = ['run','--rm'] if check_capture else (
            ['create','--memory',str(limit)] if limit is not None else ['run','--detach'])
        args = ['docker',*operation,'--init','--read-only','--network','host',
                '--shm-size','2g','--log-driver','json-file','--log-opt','max-size=50m','--log-opt','max-file=2',
                '--user',f'{os.getuid()}:{os.getgid()}','--name',name+('-check' if check_capture else ''),'--workdir','/omni-hot']
        for src,dst,ro in [(cfg['shared_root'],cfg['shared_root'],True),
                           ('/usr/local/Ascend/driver','/usr/local/Ascend/driver',True),
                           ('/etc/passwd','/etc/passwd',True),('/etc/group','/etc/group',True),
                           (bundle,'/sealed',True),(hot,'/omni-hot',False)]:
            args += ['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
        if 'continuous_actor' in cfg and not check_capture:
            # Pin new receipts at a globally unique path. The compatibility
            # /omni-hot alias still serves framework caches and working dirs.
            if hot.resolve()!=hot or hot!=bundle/'hot':raise ValueError('uncanonical resident output')
            args+=['--mount',f'type=bind,src={hot},dst={hot}']
        if 'restore_checkpoint' in cfg:
            cp=Path(cfg['restore_checkpoint']['directory'])
            if (cp.resolve()!=cp or cp.parents[2]!=bundle.parent or
                    hashlib.sha256((cp/'COMMITTED.json').read_bytes()).hexdigest()!=cfg['restore_checkpoint']['sha256']):
                raise ValueError('closed checkpoint differs from the staged restore source')
            args += ['--mount',f'type=bind,src={cp},dst=/prior-checkpoint,readonly']
        args += capture_mounts(cfg,bundle)
        if 'batch_probe' in cfg:
            source=Path(cfg['batch_probe']['sealed_cohort'])
            if source.resolve()!=source or not source.is_relative_to(bundle.parent) or source.is_relative_to(bundle):
                raise ValueError('invalid local sealed probe input')
            args+=['--mount',f'type=bind,src={source},dst=/probe-source,readonly']
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
            entry = """
import json, sys, time
from pathlib import Path
start = time.monotonic()
sys.path.insert(0, '/sealed')
from bounded_actor_step import capture_preflight
loaded = time.monotonic()
spec = json.loads(Path(sys.argv[1]).read_bytes())
result = capture_preflight(spec, Path('/omni-hot'))
if spec.get('formal_sampled'):
    from dependencies import ROOT, require
    from fresh_adapter import scope
    from run_package import read
    native = read(spec['generation']['pending_capture_ref'])
    pending = read(native['pending_capture_ref'])
    cohort = Path(pending['cohort_root'])
    with scope():
        from cohort_io import load
        adapter = load(ROOT/'fresh-cohort-successor/consumer.py', 'preflight_consumer').legacy
        origins = adapter.recording_map(read(cohort/'CONSUMER.json'), adapter.Session())
    require(all(row['record']['recording_ref_id'] in origins for row in read(cohort/'SELECTED.json')['draws']),
            'selected TRAIN recording missing from consumer map')
    result['recording_origins'] = len(origins)
print(json.dumps(dict(result=result, import_seconds=loaded-start, audit_seconds=time.monotonic()-loaded)))
"""
            args += [image,'python3','-B','-c',entry,'/sealed/'+config.name]
            prepared = time.monotonic()
            observed = json.loads(subprocess.check_output(args,text=True,timeout=600).splitlines()[-1])
            finished = time.monotonic()
            result = observed['result']
            if not (result['groups'] > 0 and result['inventory_files'] > 0):
                raise ValueError('capture preflight did not verify a real cohort')
            save(preflight_receipt,dict(result,train_config_sha256=config_sha,
                inventory_sha256=cfg['generation']['generation_inventory_ref']['sha256'],
                timing_seconds=dict(host_checks_and_setup=prepared-entered, cpu_container=finished-prepared,
                                    container_import=observed['import_seconds'], capture_audit=observed['audit_seconds'])))
            print(json.dumps(result))
            return
        args += [image,'python3','-B','-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}',
                 '/sealed/retained_hf_fsdp_control.py','--config','/sealed/'+config.name,'--input','/sealed','--output',str(hot) if 'continuous_actor' in cfg else '/omni-hot']
        save(hot/'INTENT.json',dict(image=image,devices=devices,boot_id=boot,guard_inode=os.fstat(guard.fileno()).st_ino,
                                    storage=paths,config_sha256=config_sha,memory_limit_bytes=limit))
        try:
            cid = start_container(args, limit, hot)
        except subprocess.CalledProcessError as exc:
            if not (hot/'EXIT.json').exists():
                save(hot/'EXIT.json',dict(container_id=None,exit_code=exc.returncode,
                     failure='container launch failed; no worker started',complete_ranks=0))
            raise
        if limit is None:
            save(hot/'CONTAINER.json',dict(container_id=cid))
        deadline = (time.monotonic()+max(0,cfg['continuous_actor']['deadline_epoch']-time.time())
                    if 'continuous_actor' in cfg else time.monotonic()+timeout_seconds)
        failure = None
        try:
            while True:
                try:
                    code = int(subprocess.check_output(['docker','wait',cid],text=True,timeout=30))
                    break
                except subprocess.TimeoutExpired:
                    storage.check_paths({'runtime':hot},min_free_bytes=20*1024**3+cfg['storage_budget_bytes'])
                    if time.monotonic()>deadline:
                        raise TimeoutError('bounded formal actor deadline' if cfg.get('formal_sampled')
                                           else 'bounded same-input forward deadline')
        except BaseException as exc:
            failure = repr(exc)
            subprocess.run(['docker','stop','-t','30',cid],check=True)
            code = int(subprocess.check_output(['docker','wait',cid],text=True))
        if 'continuous_actor' in cfg:
            ended=hot/'resident/END.json'
            if code==0 and not failure and not ended.is_file():failure='resident exited before terminal command'
            finished=world if ended.is_file() and not failure and code==0 else 0
        else:finished=sum((hot/f'rank-{i}'/'RESULT.json').is_file() for i in range(world))
        save(hot/'EXIT.json',dict(container_id=cid,exit_code=code,failure=failure,complete_ranks=finished))
        if code or failure:
            raise SystemExit(1)
        if cfg.get('formal_sampled') and 'batch_probe' not in cfg and 'continuous_actor' not in cfg:
            from formal_actor_receipt import close
            close(config, cfg, hot)


def reuse_closed_scores(cfg,bundle,hot,guard):
    """Reuse sealed GROUPs, never a failed attempt's optimizer or mutable model."""
    import os, shutil
    replay=cfg['closed_score_replay']; prior=Path(replay['failed_run'])
    if prior.parent!=bundle.parent or prior.resolve()!=prior or prior==bundle:
        raise ValueError('failed actor must be a separate local sibling')
    read=lambda p:json.loads(p.read_bytes())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    prev=prior/'hot';old=read(prior/'TRAIN.local.json');stop=read(prev/'EXIT.json');intent=read(prev/'INTENT.json')
    state=json.loads(subprocess.check_output(['docker','inspect',stop['container_id']]))[0]
    if not (stop['exit_code']==1 and stop['container_id']==replay['failed_container']==state['Id']
            and state['State']['Status']=='exited' and state['State']['ExitCode']==1
            and intent['boot_id']==old['boot_id']==cfg['boot_id'] and intent['guard_inode']==os.fstat(guard.fileno()).st_ino
            and cfg['existing_guard']==old['existing_guard'] and intent['config_sha256']==sha(prior/'TRAIN.local.json')
            and old['restore_checkpoint']==cfg['restore_checkpoint'] and old['generation']==cfg['generation']
            and old['training']==cfg['training'] and old['lora_row_geometry']==cfg['lora_row_geometry']
            and not any(prev.glob('step-*')) and not (prev/'forward-return').exists()):
        raise ValueError('failed actor is not an unpublished terminal for this generation')
    cohort=prev/cfg['generation']['runtime']['cohort_dir'];digest=sha(cohort/'COMPLETE.json')
    groups=read(cohort/'SELECTED.json')['draws']
    if digest!=replay['cohort_sha256'] or not read(cohort/'COMPLETE.json')['training_ready']:
        raise ValueError('scored cohort differs')
    for rank in cfg['devices']:
        d=prev/f'rank-{rank}';accepted=read(d/'ACCEPTED-COHORT.json');loss=read(d/'LOSS.json')
        if (accepted['dataset']['sha256']!=digest or accepted['checkpoint']['sha256']!=cfg['restore_checkpoint']['sha256']
                or accepted['group_ids']!=[row['group_id'] for row in groups] or not loss['result']['native_optimizer_step']
                or loss['step']!=cfg['continuation']['prior_step']+1 or (d/'RESULT.json').exists()):
            raise ValueError('failed optimizer history differs')
    target=hot/cohort.name
    for row in groups:
        f=cohort/'groups'/str(row['draw_index'])/'GROUP.json';dst=target/f.relative_to(cohort)
        if f.resolve()!=f or read(f)['group_id']!=row['group_id']:raise ValueError('score group identity changed')
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(f,dst)
        if sha(dst)!=sha(f):raise ValueError('score group copy changed')
    save(hot/'REUSED-SCORES.json',dict(failed_container=stop['container_id'],cohort_sha256=digest,groups=len(groups),
                                        predecessor=cfg['restore_checkpoint']['sha256']))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    default=os.environ.get('OMNI_ACTOR_CONFIG')
    p.add_argument('--config',type=Path,default=default,required=not default,help='defaults to OMNI_ACTOR_CONFIG')
    p.add_argument('--check-capture',action='store_true')
    p.add_argument('--recover-scoring-timeout',type=Path)
    a=p.parse_args()
    if a.check_capture and a.recover_scoring_timeout:
        p.error('timeout recovery acquires the original training guard')
    main(a.config.resolve(),a.check_capture,a.recover_scoring_timeout)
