"""Project the original fixed TP4 release onto local policy activation lanes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
D=HERE.parent/'runtime_repairs'
FILES=('retained_action_replay.py','retained_attention_candidate.py',
       'retained_audio_attention_precision.py','retained_audio_row_precision.py',
       'retained_expert_candidate.py','retained_kv_candidate.py','retained_router_candidate.py')


def read(path):
    return json.loads(Path(path).read_bytes())


def pin(path,logical=None):
    payload=path.read_bytes()
    return dict(path=str(logical or path),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')


def checkpoint_dirs(root):
    root=root.resolve()
    publication=read(root/'candidate/POLICY.json')['publication_ref']
    candidate=Path(publication['path'])
    if candidate.parent!=root/'candidate' or pin(candidate)!=publication:
        raise ValueError('publication does not belong to sealed bundle')
    checkpoint=read(candidate)['checkpoint_ref']
    manifest=Path(checkpoint['path'])
    if manifest.parent.parent!=root or manifest.parent==candidate.parent or pin(manifest)!=checkpoint:
        raise ValueError('checkpoint does not belong to sealed bundle')
    return candidate.parent.name,manifest.parent.name


def prepare(config,output):
    runtime=Path(config.get('runtime_sources',D))
    implementation=Path(config.get('implementation_sources',runtime.parent/'implementation'))
    output.mkdir(mode=0o700)
    inputs=Path(config['inputs'])
    target=Path(config['remote_root'])
    bundle=output/'bundle'
    bundle.mkdir()
    parents=Path(config['sealed_parents'])
    replay=read(inputs/'REPLAY.json')
    tp=parents/'jev-system-rl'
    impl=bundle/'parents/jev-system-rl/implementation'
    for rel in ('jev-system-rl/implementation','jev-system-rl/production-family93',
                'policy-iteration-successor','native-training-successor',
                'native-recompute-integration/state-repair/acl-runtime-successor'):
        dest=bundle/'parents'/rel
        dest.mkdir(parents=True)
        for p in (parents/rel).iterdir():
            if p.is_file() and p.suffix in ('.py','.json'):shutil.copyfile(p,dest/p.name)
    rel=Path('jev/src/multimodal_jev/local_runtime_storage.py')
    (bundle/rel).parent.mkdir(parents=True)
    shutil.copyfile(parents.parent/rel,bundle/rel)
    for name in ('service_bootstrap.py','service_overlay.py','local_activation.py'):
        shutil.copyfile(runtime/'source'/name,impl/name)
    if config.get('prefix_trace'):
        for name in ('rollout_logits.py','rollout_logits_rpc.py','rollout_logits_service.py'):
            shutil.copyfile(implementation/name,impl/name)
    if config.get('prefix_trace') in ('q-operands','o-operands','layer1-q-operands'):
        shutil.copyfile(implementation/'q_projection_trace.py',impl/'q_projection_trace.py')
    policy_src=Path(config['checkpoint_bundle'])
    # All source JSON, including nested publication refs, stays byte-identical.
    # The remote launcher mounts this closed checkpoint at its original logical
    # path, retaining the published artifact identity.
    policy=policy_src/'candidate/POLICY.json'
    assert read(policy)['publication_ref']['path']==str(policy_src/'candidate/POLICY-CANDIDATE.json')
    for src in checkpoint_dirs(policy_src):
        dst=bundle/'checkpoint'/src
        shutil.copytree(policy_src/src,dst)
    successor=config.get('successor_checkpoint_bundle')
    inbox=config.get('update_inbox')
    if inbox and (successor or not Path(inbox).is_absolute()):
        raise ValueError('dynamic and static successors have separate configurations')
    if successor:
        successor=Path(successor)
        assert successor.is_absolute() and successor!=policy_src
        for src in checkpoint_dirs(successor):
            shutil.copytree(successor/src,bundle/'successor'/src)
    precision=Path(replay['precision_ref']['path'])
    assert pin(precision)==replay['precision_ref']
    shutil.copyfile(precision,bundle/'PRECISION-SPEC.json')
    shutil.copyfile(inputs/'PROFILE.json',bundle/'PROFILE.json')
    loc=[]
    spec=read(precision)
    for ref in [spec['source_seal_ref'],spec['config_ref'],spec['arithmetic_delta_ref'],
                spec['audio_repair_ref'],*spec['sources'].values()]:
        name=Path(ref['path']).name
        local=next(p for p in (impl/name,bundle/'parents/jev-system-rl/production-family93'/name) if p.is_file())
        observed=pin(local,target/'bundle'/local.relative_to(bundle))
        assert (observed['sha256'],observed['bytes'])==(ref['sha256'],ref['bytes']),name
        loc.append(dict(observed,logical=ref['path']))
    write(bundle/'LOCATIONS.json',loc)
    row_keys=('audio_row_sites','gate_candidate_layers','attention_candidate_layers','down_candidate_layers',
              'audio_attention_layers','router_layers','kv_layers','tiled_layers')
    grouped={}
    for site in ('q','o'):
        name=site+'_base_layers'
        layers=config.get(name,[])
        if layers:
            assert layers==sorted(set(layers)) and all(type(i) is int and 0<=i<48 for i in layers)
            grouped[name]=layers
    if 'lora_row_geometry' in config:
        import importlib.util
        source=runtime/'source/lora_row_precision.py'
        loader=importlib.util.spec_from_file_location('lora_row_precision',source)
        mod=importlib.util.module_from_spec(loader);loader.loader.exec_module(mod)
        mod.validate(config['lora_row_geometry'])
        shutil.copyfile(source,impl/source.name)
        grouped['lora_row_geometry']=dict(config['lora_row_geometry'],source_sha256=pin(source)['sha256'])
    write(bundle/'ROWS.json',{**{name:replay[name] for name in row_keys},**grouped})
    rpc=bundle/'rpc'
    rpc.mkdir()
    arithmetic=bundle/'arithmetic'
    arithmetic.mkdir()
    for name in FILES:
        source=Path(config['arithmetic_sources'])/name
        assert pin(source)['sha256']==config['row_source_sha256'][name],name
        shutil.copyfile(source,arithmetic/name)
    refs={}
    for name,ref in replay['named_rpc_overlay']['files'].items():
        src=tp/'production-rpc'/name
        assert pin(src)['sha256']==ref['sha256'],name
        dst=rpc/name
        shutil.copyfile(src,dst)
        if name=='named_rpc_worker.py':
            with dst.open('a') as stream:
                stream.write("\nfrom local_activation import install_rows\n"
                             "NamedWorkerRPCMixin.omni_local_install_rows=install_rows\n"
                             "import sys as _sys\n"
                             "from rollout_logits_rpc import install_named_extension as _extend\n"
                             "_extend(_sys.modules[__name__])\n")
        refs[name]=pin(dst,target/'bundle/rpc'/name)
    engine=replay['engine']
    assert engine['dtype']=='bfloat16' and engine['lora_dtype']=='float32'
    assert engine['enable_prefix_caching'] and engine['enable_lora'] and engine['enforce_eager'] and not engine['async_scheduling']
    for lane in config['lanes']:
        original=read(inputs/('lane-'+lane+'.json'))
        local=output/('lane-'+lane)
        local.mkdir()
        for name in ('launch.py','service.py'):
            shutil.copyfile(HERE/name,local/name)
        if len(original['cards'])!=engine['tensor_parallel_size']:raise ValueError('lane and sealed engine topology differ')
        serving=['--tensor-parallel-size',str(engine['tensor_parallel_size']),'--enable-lora','--max-loras',str(engine['max_loras']),
            '--lora-dtype',engine['lora_dtype'],'--max-lora-rank',str(engine['max_lora_rank']),
            '--worker-extension-cls','ascend_fp32_worker.FP32LoRAWorkerExtension','--generation-config',
            engine['generation_config'],'--logprobs-mode',engine['logprobs_mode'],'--default-chat-template-kwargs',
            '{"enable_thinking":false}','--max-model-len',str(engine['max_model_len']),
            '--gpu-memory-utilization',str(engine['gpu_memory_utilization']),'--max-num-seqs',str(engine['max_num_seqs']),
            '--enforce-eager','--no-async-scheduling','--enable-prefix-caching','--dtype',engine['dtype'],
            '--limit-mm-per-prompt','{"audio":1,"image":0,"video":0}']
        c=dict(original,name=original['name']+'-rl',root=str(target/('lane-'+lane)),
               policy_overlay=dict(bundle=str(target/'bundle'),server=str(target/f'bundle/SERVER-{lane}.json'),
                                   checkpoint_mount=str(policy_src)),serve_args=serving)
        if successor:c['policy_overlay']['successor_mount']=str(successor)
        if inbox:c['policy_overlay']['update_mount']=inbox
        assert c['model']==engine['model']
        write(local/'config.json',c)
        write(local/'PINS.json',{p.name:pin(p)['sha256'] for p in local.iterdir()})
        write(bundle/f'SERVER-{lane}.json',dict(endpoint=f"http://127.0.0.1:{c['port']}/v1/chat/completions",
            overlay_root='/runtime/state/policy',policy_ref=pin(policy),
            **(dict(successor_ref=pin(successor/'candidate/POLICY.json')) if successor else {}),
            **(dict(update_inbox=inbox) if inbox else {}),
            profile_ref=pin(bundle/'PROFILE.json',target/'bundle/PROFILE.json'),
            service_identity=dict(container_name=c['name']+'-'+config['attempt'],physical_devices=c['cards'],image=c['image']),
            named_rpc_overlay=dict(files=refs,precision_ref=pin(bundle/'PRECISION-SPEC.json',target/'bundle/PRECISION-SPEC.json')),
            local_activation=dict(source_boot=c['host_boot_id'],guard=c['guard'],model=c['model'],bounded_candidate=True,
                locations_ref=pin(bundle/'LOCATIONS.json',target/'bundle/LOCATIONS.json'),
                rows_ref=pin(bundle/'ROWS.json',target/'bundle/ROWS.json'),
                diagnostic_logits=dict(read(inputs/'LOGITS.json'),
                    **({'prefix_trace':config['prefix_trace']} if config.get('prefix_trace') else {}),
                    **({'selection':config['logits_selection']} if 'logits_selection' in config else {})))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(read(a.config),a.output)
