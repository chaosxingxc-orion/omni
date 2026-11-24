"""Prepare two formal resident TP4 services and replace drained predecessors under their guard."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=raw: raise ValueError(f'changed receipt {path}')
        return
    with path.open('xb') as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    fd=os.open(path.parent,os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def call(args, timeout=180):
    return subprocess.check_output(args,timeout=timeout,text=True).strip()


def prepare(root,c):
    base=Path(c['previous'])
    src=root/'inputs';dest=root/'bundle'
    if (root/'PREPARED.json').exists():return
    if not dest.exists():shutil.copytree(src/'bundle',dest)
    graph=read(src/'graph.json')
    old=Path(read(src/'template.json')['policy_overlay']['bundle'])
    for name,sha in c['overlays'].items():
        source=src/'overlays'/name
        if digest(source)!=sha:raise ValueError('source snapshot mismatch '+name)
        shutil.copyfile(source,dest/'parents/jev-system-rl/implementation'/name)
    rows=read(src/'bundle/ROWS.json')
    rows['decode_layer_graph'].update(batch_sizes=graph['batch_sizes'],tiled_attention_batch=graph['tiled_attention_batch'],
                                     source_sha256=c['overlays']['decode_layer_graph.py'])
    rows['decode_attention_sha256']=c['overlays']['decode_tiled_attention.py']
    if 'audio_batch_geometry' in c:
        if c['audio_batch_geometry']!='single_clip_v1':raise ValueError('unsupported audio arithmetic')
        rows['audio_batch_geometry']=dict(mode=c['audio_batch_geometry'],sha256=c['overlays']['audio_batch_geometry.py'])
    def replace(path,val):
        tmp=path.with_suffix('.projecting')
        save(tmp,val); os.replace(tmp,path)
    profile=None
    if c.get('rank_profile_sha256'):
        from rank_profile import validate
        profile=read(src/'rank-profile.json')
        if digest(src/'rank-profile.json')!=c['rank_profile_sha256']:raise ValueError('rank profile changed')
        rank=validate(profile)['lora_rank']
        rows['lora_row_geometry'].update(adapter_rank=rank,source_sha256=c['overlays']['lora_row_precision.py'])
        rows['decode_projection_runtime']=c['overlays']['decode_projection_runtime.py']
        for name in ('local_activation.py','family_precision.py'):
            if name not in c['overlays']:raise ValueError('missing rank-aware service overlay: '+name)
    replace(dest/'ROWS.json',rows)
    def relocate(obj):
        if isinstance(obj,list):return [relocate(v) for v in obj]
        if isinstance(obj,dict):
            original=obj.get('path')
            result={k:relocate(v) for k,v in obj.items()}
            if isinstance(original,str) and Path(original).is_relative_to(old) and 'sha256' in obj:
                f=Path(result['path']);result.update(bytes=f.stat().st_size,sha256=digest(f))
            return result
        if isinstance(obj,str) and Path(obj).is_relative_to(old):return str(dest/Path(obj).relative_to(old))
        return obj
    locations=relocate(read(src/'bundle/LOCATIONS.json'))
    if profile:
        # Parent and successor live at distinct names; the worker authenticates both.
        precision=read(src/'bundle/PRECISION-SPEC.json')
        if digest(src/'bundle/PRECISION-SPEC.json')!=read(src/'bundle/SERVER-a.json')['named_rpc_overlay']['precision_ref']['sha256']:
            raise ValueError('initial precision does not match the original service')
        area=dest/'rank-precision';area.mkdir(exist_ok=True)
        def ref(f):
            return dict(path=str(f),sha256=digest(f),bytes=f.stat().st_size)
        save(area/'parent.json',precision)
        changes={}
        for name in ('projection_family_runtime.py','family_runtime.py'):
            f=src/'rank-sources'/name
            if digest(f)!=c['rank_sources'][name]:raise ValueError('rank implementation changed: '+name)
            shutil.copyfile(f,area/name)
            changes[name]=dict(parent_ref=precision['sources'][name],result_ref=ref(area/name))
        delta=dict(schema='omni-family-adapter-rank-v1',parent_spec_ref=ref(area/'parent.json'),
                   adapter_rank=rank,changes=changes)
        save(area/'delta.json',delta)
        sources=dict(precision['sources']);sources.update({k:v['result_ref'] for k,v in changes.items()})
        replace(dest/'PRECISION-SPEC.json',dict(precision,sources=sources,adapter_rank_delta_ref=ref(area/'delta.json')))
        init=area/'lora_adapter'
        if not init.exists():shutil.copytree(src/'initial-adapter',init)
        parent=Path(c['parent_commit'])
        if digest(init/'ORIGIN.json')!=c['origin_sha256']:raise ValueError('wrong initial publication')
        origin=read(init/'ORIGIN.json')
        if digest(parent)!=origin['inputs']['parent_sha256'] or digest(src/'rank-profile.json')!=origin['inputs']['profile_sha256']:
            raise ValueError('initial policy inputs differ')
        for name,file in [('parent-commit.json',parent),('rank-profile.json',src/'rank-profile.json')]:
            shutil.copyfile(file,area/name)
        assets=[ref(init/n) for n in ('adapter_model.safetensors','adapter_config.json')]
        if {Path(a['path']).name:a['sha256'] for a in assets}!=origin['files']:raise ValueError('wrong adapter assets')
        save(area/'INITIAL.json',dict(schema='expanded-initial-policy-v1',origin_ref=ref(init/'ORIGIN.json'),
             parent_ref=ref(area/'parent-commit.json'),profile_ref=ref(area/'rank-profile.json'),assets=assets))
        version='initial-sha256:'+assets[0]['sha256']
        if c['version']!=version:raise ValueError('wrong configured branch version')
        policy=read(dest/'checkpoint/candidate/POLICY.json')
        save(area/'POLICY.json',dict(schema=policy['schema'],role='current',kind='branch_initial',version=version,
             artifact_version=version,step=0,model=policy['model'],thinking=False,assets=assets,publication_ref=ref(area/'INITIAL.json')))
    replace(dest/'LOCATIONS.json',locations)
    inbox=root/'updates';inbox.mkdir(exist_ok=True)
    for lane,expected in c['old_services'].items():
        orig=base/('lane-'+lane);code=root/('lane-'+lane);code.mkdir(exist_ok=True)
        cfg=read(orig/'config.json')
        if not all(digest(orig/k)==v for k,v in read(orig/'PINS.json').items()):raise ValueError('old service code drift')
        if cfg['host_boot_id']!=c['boot'] or cfg['name']+'-'+c['old_attempt']!=expected:raise ValueError('wrong lane')
        for f in ('launch.py','service.py'):shutil.copyfile(orig/f,code/f)
        cfg['root']=str(code)
        cfg['serve_args'][cfg['serve_args'].index('--max-num-seqs')+1]=str(graph['max_num_seqs'])
        if profile:
            args=cfg['serve_args']
            if args.count('--max-lora-rank')!=1:raise ValueError('expected one explicit service rank limit')
            args[args.index('--max-lora-rank')+1]=str(profile['inference']['max_lora_rank'])
            if c.get('lora_targets_from_publication'):
                modules=read(init/'adapter_config.json')['target_modules']
                suffixes=sorted({name.split('.')[-1] for name in modules})
                # PEFT serializes minimal unique suffixes, including bare o_proj.
                # Verify the actual safetensors header independently of those names.
                import struct
                with (init/'adapter_model.safetensors').open('rb') as weights:
                    size,=struct.unpack('<Q',weights.read(8))
                    if not 0<size<=1<<24:raise ValueError('invalid safetensors header size')
                    tensors=json.loads(weights.read(size))
                tensors.pop('__metadata__',None)
                model_config=read(Path(cfg['model'])/'config.json')
                num_layers=model_config['thinker_config']['text_config']['num_hidden_layers']
                if type(num_layers) is not int or num_layers<=0:raise ValueError('invalid model layer count')
                sites={f'model.layers.{i}.self_attn.{site}' for i in range(num_layers) for site in suffixes}
                matched={s for s in sites if any(s==m or s.endswith('.'+m) for m in modules)}
                if (matched!=sites or any(not any(s==m or s.endswith('.'+m) for s in sites) for m in modules)
                        or set(tensors)!={f'base_model.model.{s}.lora_{part}.weight' for s in sites for part in ('A','B')}
                        or set(suffixes)!={'q_proj','k_proj','v_proj','o_proj'}):raise ValueError('unsupported adapter targets')
                if '--lora-target-modules' in args:
                    import itertools
                    installed=list(itertools.takewhile(lambda v:not v.startswith('--'),args[args.index('--lora-target-modules')+1:]))
                    if args.count('--lora-target-modules')!=1 or sorted(installed)!=suffixes:raise ValueError('conflicting deployment targets')
                else:args.extend(['--lora-target-modules',*suffixes])
        if '--logits-processors' in cfg['serve_args'] or '--enable-prefix-caching' not in cfg['serve_args']:raise ValueError('formal service required')
        cfg['policy_overlay'].update(bundle=str(dest),server=str(dest/f'SERVER-{lane}.json'),
                  update_mount=str(inbox),checkpoint_mount=read(src/'template.json')['policy_overlay']['checkpoint_mount'])
        server=relocate(read(src/'bundle/SERVER-a.json'));server.update(update_inbox=str(inbox),
                endpoint=f"http://{cfg['host']}:{cfg['port']}/v1/chat/completions",continuous_control={'protocol':'apply-local-marker-v1'})
        server['service_identity'].update(container_name=cfg['name']+'-'+c['attempt'],physical_devices=cfg['cards'])
        server['local_activation'].update(source_boot=c['boot'],model=cfg['model'],guard=cfg['guard'])
        if 'reset_audio_on_precision_install' in c:
            flag=c['reset_audio_on_precision_install']
            if type(flag) is not bool:raise ValueError('explicit audio reset flag required')
            server['local_activation']['reset_audio_on_precision_install']=flag
        if server['release_kind']!='formal' or server['policy_ref']['sha256']!=digest(dest/'checkpoint/candidate/POLICY.json'):raise ValueError('wrong initial policy')
        if profile:server['policy_ref']=ref(area/'POLICY.json')
        replace(dest/f'SERVER-{lane}.json',server);save(code/'config.json',cfg)
        save(code/'PINS.json',{n:digest(code/n) for n in ('config.json','launch.py','service.py')})
    save(root/'PREPARED.json',dict(lanes=list(c['old_services']),policy=server['policy_ref']['sha256'],
                                    graph=graph,overlays=c['overlays']))


def launch_module(path):
    spec=importlib.util.spec_from_file_location('resident_'+path.parent.name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.verify()
    return mod


def metric_idle(url):
    with urllib.request.urlopen(url+'/metrics',timeout=10) as response:lines=response.read().decode().splitlines()
    for name in ('num_requests_running','num_requests_waiting'):
        vals=[float(l.rsplit(' ',1)[-1]) for l in lines if l.startswith('vllm:'+name+'{')]
        if not vals or any(vals):raise ValueError('lane is busy or metrics incomplete')


def deploy(root):
    c=read(root/'CONFIG.json')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=c['boot']:raise ValueError('boot changed')
    prepare(root,c)
    mods={k:launch_module(root/('lane-'+k)/'launch.py') for k in c['old_services']}
    guards={m.C['guard'] for m in mods.values()}
    if len(guards)!=1:raise ValueError('shared guard required')
    with open(guards.pop(),'rb') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def preflight(lane):
            m=mods[lane];directory=root/('lane-'+lane)
            if not (directory/'runs'/c['check_attempt']/'state/PREFLIGHT.json').exists():
                call(m.argv(c['check_attempt'],False),c['startup_seconds'])
            receipt=read(directory/'runs'/c['check_attempt']/'state/PREFLIGHT.json')
            if receipt['config_sha256']!=digest(directory/'config.json'):raise ValueError('wrong preflight')
        with concurrent.futures.ThreadPoolExecutor(len(mods)) as pool:list(pool.map(preflight,mods))
        def cutover(lane):
            m=mods[lane];url=f"http://{m.C['host']}:{m.C['port']}"
            old=m.inspect(c['old_services'][lane])
            failed=c.get('failed_startup_predecessor',False)
            if old['Id']!=c['old_ids'][lane]:raise ValueError('old service changed')
            if failed:
                prev=Path(c['previous'])
                first=read(prev/(lane+'-STARTED.json'))
                if (old['State']['Status']!='exited' or old['State']['ExitCode']==0 or first['container_id']!=old['Id']
                        or (prev/(lane+'-READY.json')).exists() or not (prev/('lane-'+lane)/'runs'/c['old_attempt']/'state/EXIT.json').exists()):
                    raise ValueError('predecessor is not the failed startup')
            elif c.get('failed_diagnostic_predecessor'):
                prev=Path(c['previous'])
                receipt=read(prev/(lane+'-READY.json'))
                witness=read(root/'inputs/failed-diagnostic.json')
                if (receipt['container_id']!=old['Id'] or old['State']['Status']!='exited' or
                    witness['container_id']!=old['Id'] or witness['ready_sha256']!=digest(prev/(lane+'-READY.json')) or
                    witness['log_sha256']!=digest(root/'inputs/failed-engine.log') or
                    'EngineDeadError' not in (root/'inputs/failed-engine.log').read_text() or
                    not (prev/'natural-comparison/INTENT.json').exists() or (prev/'natural-comparison/RESULT.json').exists()):
                    raise ValueError('failed diagnostic owner not reconciled')
            elif old['State']['Running']:metric_idle(url)
            else:raise ValueError('old service is not running')
            if (root/(lane+'-INTENT.json')).exists():raise ValueError('existing handover requires reconciliation')
            save(root/(lane+'-INTENT.json'),dict(previous=old['Id'],new=m.name(c['attempt']),automatic_retry=False))
            if old['State']['Running']:call(['docker','stop','-t',str(m.C['stop_timeout_seconds']+15),old['Id']],m.C['stop_timeout_seconds']+30)
            m.start_checks()
            cid=call(m.argv(c['attempt'],True));save(root/(lane+'-STARTED.json'),dict(container_id=cid))
            deadline=time.monotonic()+c['startup_seconds']
            while True:
                if not m.inspect(m.name(c['attempt']))['State']['Running']:raise RuntimeError('service exited')
                try:
                    with urllib.request.urlopen(url+'/health',timeout=8) as response:
                        if response.status==200:break
                except OSError:pass
                if time.monotonic()>deadline:raise TimeoutError('service readiness')
                time.sleep(5)
            m.ready(c['attempt'])
            request=urllib.request.Request(url+'/_omni/activation',method='POST',data=b'')
            with urllib.request.urlopen(request,timeout=600) as response:active=json.load(response)
            if active['version']!=c['version'] or active['service_identity']['physical_devices']!=m.C['cards']:raise ValueError('wrong live adapter or devices')
            request=urllib.request.Request(url+'/v1/chat/completions',method='POST',data=(root/'inputs/wire.json').read_bytes(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=600) as response:body=json.load(response)
            choice,=body['choices']
            if not choice.get('token_ids') or len(choice['logprobs']['content'])!=len(choice['token_ids']):raise ValueError('incomplete natural action response')
            metric_idle(url)
            cache=json.loads(call(['docker','exec',cid,'cat','/runtime/state/policy/CACHE-POLICY.json']))
            prec=json.loads(call(['docker','exec',cid,'cat','/runtime/state/policy/PRECISION.json']))
            if c.get('reset_audio_on_precision_install') and not all(cache.get(k) is True for k in ('audio_processor_reset','audio_encoder_reset')):
                raise ValueError('initial audio cache reset missing')
            if not(cache['prefix_cache'] and cache['kv_cache'] and cache['reset_succeeded'] and len(prec['workers'])==len(m.C['cards'])==int(m.C['serve_args'][m.C['serve_args'].index('--tensor-parallel-size')+1])):raise ValueError('missing worker/cache receipts')
            save(root/(lane+'-READY.json'),dict(container_id=cid,activation=active,model=body['model'],
                     tokens=len(choice['token_ids']),cache=cache,precision=prec,wire_sha256=digest(root/'inputs/wire.json')))
        # Both idle predecessors share the original guard. Keep it until both
        # real requests and activation receipts have been accounted for.
        with concurrent.futures.ThreadPoolExecutor(len(mods)) as pool:list(pool.map(cutover,mods))
        save(root/'READY.json',dict(lanes={k:digest(root/(k+'-READY.json')) for k in mods},profile=read(root/'inputs/graph.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path,nargs='?',default=os.environ.get('OMNI_ROLLOUT_DEPLOYMENT_ROOT') or None,
                   help='prepared bundle directory; defaults to OMNI_ROLLOUT_DEPLOYMENT_ROOT')
    p.add_argument('--config',type=Path,default=os.environ.get('OMNI_ROLLOUT_CONFIG') or None,
                   help='resolved deployment config, installed once in a prepared root')
    args=p.parse_args()
    if args.root is None:p.error('provide root or OMNI_ROLLOUT_DEPLOYMENT_ROOT')
    root=args.root.resolve(strict=True)
    if args.config is not None:
        target=root/'CONFIG.json'
        resolved=read(args.config)
        if target.exists():
            if read(target)!=resolved:p.error('resolved deployment differs from the prepared root')
        else:
            save(target,resolved)
    deploy(root)
