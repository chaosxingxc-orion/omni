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
PRECISION_ONLY_FILES = {
    'abi_probe.py', 'actual_probe.py', 'logits_scope.py', 'logits_selection.py',
    'probe_facts.py', 'probe_installed.py',
}


def read(path):
    return json.loads(Path(path).read_bytes())


def pin(path,logical=None):
    payload=path.read_bytes()
    return dict(path=str(logical or path),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')


def checkpoint_dirs(root, logical=None):
    root=root.resolve()
    logical=root if logical is None else logical
    publication=read(root/'candidate/POLICY.json')['publication_ref']
    candidate=Path(publication['path'])
    if candidate.parent!=logical/'candidate' or pin(root/'candidate'/candidate.name,candidate)!=publication:
        raise ValueError('publication does not belong to sealed bundle')
    checkpoint=read(root/'candidate'/candidate.name)['checkpoint_ref']
    manifest=Path(checkpoint['path'])
    if (manifest.parent.parent!=logical or manifest.parent==candidate.parent or
            pin(root/manifest.parent.name/manifest.name,manifest)!=checkpoint):
        raise ValueError('checkpoint does not belong to sealed bundle')
    return candidate.parent.name,manifest.parent.name


def prepare(config,output):
    release=config.get('release_kind','precision')
    if release not in ('precision','formal','decode_profile','decode_replay'):
        raise ValueError('unknown release kind')
    profiling = release == 'decode_profile'
    profile = config.get('decode_profile')
    if profiling != ('decode_profile' in config) or profiling and (
            not isinstance(profile, dict) or set(profile) != {'rank', 'steps'} or
            type(profile['rank']) is not int or profile['rank'] not in range(4) or
            type(profile['steps']) is not int or not 1 <= profile['steps'] <= 32):
        raise ValueError('bounded decode profiling requires a separate release, rank and step count')
    formal=release in ('formal','decode_profile','decode_replay')
    if formal and any(key in config for key in ('prefix_trace','logits_selection','successor_checkpoint_bundle')):
        raise ValueError('precision collection settings cannot enter a formal bundle')
    engine_limits = config.get('formal_engine_limits', {})
    if (not isinstance(engine_limits, dict)
            or ('formal_engine_limits' in config and not formal)
            or set(engine_limits) - {'max_num_seqs', 'max_num_batched_tokens', 'gpu_memory_utilization'}
            or any(type(v) is not int or v <= 0 for k,v in engine_limits.items() if k!='gpu_memory_utilization')
            or 'gpu_memory_utilization' in engine_limits and not (
                type(engine_limits['gpu_memory_utilization']) is float and 0 < engine_limits['gpu_memory_utilization'] < 1)):
        raise ValueError('formal_engine_limits requires supported positive integer limits in a formal release')
    runtime=Path(config.get('runtime_sources',D))
    implementation=Path(config.get('implementation_sources',runtime.parent/'implementation'))
    candidate = config.get('formal_decode_candidate')
    if 'formal_decode_candidate' in config and (not formal or
            candidate not in ('without_original_controls','skip_empty_experts','pack_active_experts','cache_frozen_projections','decode_expert_owners','direct_projections','owners_direct_projections')):
        raise ValueError('select one supported formal decode candidate')
    if ('empty_expert_sources' in config) != (candidate in ('skip_empty_experts','pack_active_experts')):
        raise ValueError('empty expert source closure is required only for the empty expert candidate')
    packed_limit = config.get('packed_active_limit')
    if ('packed_active_limit' in config) != (candidate == 'pack_active_experts') or (
            candidate == 'pack_active_experts' and (type(packed_limit) is not int or not 1 <= packed_limit <= 32)):
        raise ValueError('packed active experts require a bounded explicit limit')
    direct = candidate in ('direct_projections','owners_direct_projections')
    cache_projections = direct or candidate == 'cache_frozen_projections'
    if ('frozen_projection_cache' in config) != cache_projections:
        raise ValueError('frozen projection cache requires its own candidate and budget')
    expert_owners = candidate in ('decode_expert_owners','owners_direct_projections')
    if ('decode_expert_owners' in config) != expert_owners:
        raise ValueError('expert owner placement requires an explicit cache capacity')
    if 'decode_layer_graph' in config and (candidate != 'owners_direct_projections' or config['decode_expert_owners'].get('static_routes') is not True):
        raise ValueError('decode graph requires static expert owners and verified projections')
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
            if formal and (p.name in PRECISION_ONLY_FILES
                           or p.name.startswith(('diagnostic','rollout_logits','q_projection_trace',
                                                 'causal_diagnostic')) or p.name.endswith('_cpu.py')):
                continue
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
    policy_src=Path(config['checkpoint_bundle']).resolve()
    policy_mount=Path(config.get('checkpoint_mount',str(policy_src)))
    if not policy_mount.is_absolute() or '..' in policy_mount.parts:
        raise ValueError('checkpoint mount must be absolute and canonical')
    # All source JSON, including nested publication refs, stays byte-identical.
    # The remote launcher mounts this closed checkpoint at its original logical
    # path, retaining the published artifact identity.
    policy=policy_src/'candidate/POLICY.json'
    assert read(policy)['publication_ref']['path']==str(policy_mount/'candidate/POLICY-CANDIDATE.json')
    for src in checkpoint_dirs(policy_src,policy_mount):
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
    shutil.copyfile(inputs/'PROFILE.json',bundle/'PROFILE.json')
    loc=[]
    spec=read(precision)
    original_refs=[spec['source_seal_ref'],spec['config_ref'],spec['arithmetic_delta_ref'],
                   spec['audio_repair_ref'],*spec['sources'].values()]
    for ref in original_refs:
        name=Path(ref['path']).name
        local=next(p for p in (impl/name,bundle/'parents/jev-system-rl/production-family93'/name) if p.is_file())
        observed=pin(local,target/'bundle'/local.relative_to(bundle))
        assert (observed['sha256'],observed['bytes'])==(ref['sha256'],ref['bytes']),name
        loc.append(dict(observed,logical=ref['path']))
    if candidate:
        validator=runtime/'source/family_precision.py'
        if not validator.is_file():
            validator=runtime/'family_precision.py'
        shutil.copyfile(validator,impl/'family_precision.py')
    if candidate == 'without_original_controls':
        spec['original_operator_control']=False
    if candidate in ('skip_empty_experts','pack_active_experts'):
        efficiency=bundle/'efficiency'
        efficiency.mkdir()
        approved=read(config['empty_expert_sources'])
        if (approved['schema']!='omni-empty-expert-efficiency-delta-v1' or
                approved['training_admission'] is not False or
                approved['options']!={'skip_empty_expert_chunks':True} or
                set(approved['changes'])!={'family_runtime.py','moe_family_runtime.py',
                                           'expert_gate_repair.py','expert_down_projection.py'}):
            raise ValueError('expected exact sealed empty expert source closure')
        parent=efficiency/'PARENT-SPEC.json'
        shutil.copyfile(precision,parent)
        changes={}
        spec['sources']=dict(spec['sources'])
        for name,row in approved['changes'].items():
            ref=row['result_ref']
            source=Path(ref['path'])
            if row['parent_ref']!=spec['sources'][name] or pin(source)!=ref:
                raise ValueError('empty expert source does not match sealed parent: '+name)
            destination=efficiency/name
            shutil.copyfile(source,destination)
            if candidate == 'pack_active_experts' and name in ('family_runtime.py','moe_family_runtime.py'):
                shutil.copyfile(runtime/'source'/name,destination)
            result=pin(destination,target/'bundle/efficiency'/name)
            changes[name]=dict(parent_ref=row['parent_ref'],result_ref=result)
            spec['sources'][name]=result
        delta=efficiency/'EFFICIENCY-DELTA.json'
        options=dict(approved['options'],**({'packed_active_limit':packed_limit} if packed_limit else {}))
        write(delta,dict(approved,options=options,parent_spec_ref=pin(parent,target/'bundle/efficiency'/parent.name),
                         changes=changes))
        spec['efficiency_delta_ref']=pin(delta,target/'bundle/efficiency'/delta.name)
    if candidate:
        write(bundle/'PRECISION-SPEC.json',spec)
    else:
        shutil.copyfile(precision,bundle/'PRECISION-SPEC.json')
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
    if cache_projections:
        import importlib.util
        cache_source = runtime/'source/frozen_projection_cache.py'
        spec_cache = importlib.util.spec_from_file_location('frozen_projection_cache',cache_source)
        cache_helper = importlib.util.module_from_spec(spec_cache)
        spec_cache.loader.exec_module(cache_helper)
        cache_helper.validate(config['frozen_projection_cache'])
        if not all(grouped.get(site+'_base_layers')==list(range(48)) for site in ('q','o')) or replay['kv_layers']!=list(range(48)):
            raise ValueError('prepacked layout requires complete grouped Q/K/V/O consumers')
        shutil.copyfile(cache_source,impl/cache_source.name)
        grouped['frozen_projection_cache'] = dict(config['frozen_projection_cache'],source_sha256=pin(cache_source)['sha256'])
    if direct:
        geometry = grouped.get('lora_row_geometry',{})
        if geometry.get('site')!='qkvo_lora_a' or geometry.get('algorithm')!='grouped_bf16' or geometry.get('layers')!=list(range(48)):
            raise ValueError('verified grouped Q/K/V/O LoRA A geometry required')
        projection_source=runtime/'source/decode_projection_runtime.py'
        shutil.copyfile(projection_source,impl/projection_source.name)
        grouped['decode_projection_runtime']=pin(projection_source)['sha256']
    if expert_owners:
        import importlib.util
        owner_source=runtime/'source/decode_expert_owners.py'
        spec_owner=importlib.util.spec_from_file_location('decode_expert_owners',owner_source)
        owner_helper=importlib.util.module_from_spec(spec_owner)
        spec_owner.loader.exec_module(owner_helper)
        owner_helper.validate(config['decode_expert_owners'])
        shutil.copyfile(owner_source,impl/owner_source.name)
        grouped['decode_expert_owners']=dict(config['decode_expert_owners'],source_sha256=pin(owner_source)['sha256'])
    if 'decode_layer_graph' in config:
        graph_source=runtime/'source/decode_layer_graph.py'
        graph_spec=importlib.util.spec_from_file_location('decode_layer_graph',graph_source)
        graph_helper=importlib.util.module_from_spec(graph_spec)
        graph_spec.loader.exec_module(graph_helper)
        graph_helper.validate(config['decode_layer_graph'])
        shutil.copyfile(graph_source,impl/graph_source.name)
        if 'tiled_attention_batch' in config['decode_layer_graph']:
            if replay['tiled_layers']!=list(range(48)):
                raise ValueError('packed attention requires all 48 layers on the absolute tile contract')
            helper=runtime/'source/decode_tiled_attention.py'
            shutil.copyfile(helper,impl/helper.name)
            grouped['decode_attention_sha256']=pin(helper)['sha256']
        grouped['decode_layer_graph']=dict(config['decode_layer_graph'],source_sha256=pin(graph_source)['sha256'])
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
            if formal:
                dst.write_text(
                    "from local_activation import install_rows\n\n"
                    "RPC_METHODS = ('omni_sync_worker_readback',)\n\n"
                    "class NamedWorkerRPCMixin:\n"
                    "    omni_local_install_rows = install_rows\n\n"
                    "    def omni_production_precision(self, spec):\n"
                    "        from precision_runtime import install\n"
                    "        return install(self, spec)\n\n"
                    "    def omni_sync_worker_readback(self, *args, **kwargs):\n"
                    "        from engine_bridge import worker_readback\n"
                    "        return worker_readback(self, *args, **kwargs)\n")
            else:
                with dst.open('a') as stream:
                    stream.write("\nfrom local_activation import install_rows\n"
                                 "NamedWorkerRPCMixin.omni_local_install_rows=install_rows\n"
                                 "import sys as _sys\n"
                                 "from rollout_logits_rpc import install_named_extension as _extend\n"
                                 "_extend(_sys.modules[__name__])\n")
            if direct:
                with dst.open('a') as stream:
                    stream.write('\n_verified_readback = NamedWorkerRPCMixin.omni_sync_worker_readback\n'
                                 'def readback_projection_operand(self, *args, **kwargs):\n'
                                 '    receipt = _verified_readback(self, *args, **kwargs)\n'
                                 '    self._projection_adapter_verified(receipt)\n'
                                 '    return receipt\n'
                                 'NamedWorkerRPCMixin.omni_sync_worker_readback = readback_projection_operand\n')
            if profiling:
                with dst.open('a') as stream:
                    stream.write('\n' + (HERE/'decode_profile.py').read_text() + '\n')
                    stream.write('def install_profiled_rows(self, rows):\n'
                                 '    receipt = install_rows(self, rows)\n'
                                 f'    attach_decode_profile(self, {profile!r})\n'
                                 '    return receipt\n'
                                 'NamedWorkerRPCMixin.omni_local_install_rows = install_profiled_rows\n')
        refs[name]=pin(dst,target/'bundle/rpc'/name)
    if release=='decode_replay':
        shutil.copyfile(HERE/'decode_action_processor.py',impl/'decode_action_processor.py')
    engine = {**replay['engine'], **engine_limits}
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
        if release=='decode_replay':
            serving.extend(['--logits-processors','decode_action_processor:ReplayActions'])
        if 'max_num_batched_tokens' in engine_limits:
            serving.extend(['--max-num-batched-tokens', str(engine_limits['max_num_batched_tokens'])])
        c=dict(original,name=original['name']+'-rl',root=str(target/('lane-'+lane)),
               policy_overlay=dict(bundle=str(target/'bundle'),server=str(target/f'bundle/SERVER-{lane}.json'),
                                   checkpoint_mount=str(policy_mount)),serve_args=serving)
        if successor:c['policy_overlay']['successor_mount']=str(successor)
        if inbox:c['policy_overlay']['update_mount']=inbox
        assert c['model']==engine['model']
        write(local/'config.json',c)
        write(local/'PINS.json',{p.name:pin(p)['sha256'] for p in local.iterdir()})
        write(bundle/f'SERVER-{lane}.json',dict(endpoint=f"http://127.0.0.1:{c['port']}/v1/chat/completions",
            release_kind=release,overlay_root='/runtime/state/policy',
            policy_ref=pin(policy,policy_mount/'candidate/POLICY.json'),
            **(dict(successor_ref=pin(successor/'candidate/POLICY.json')) if successor else {}),
            **(dict(update_inbox=inbox) if inbox else {}),
            profile_ref=pin(bundle/'PROFILE.json',target/'bundle/PROFILE.json'),
            service_identity=dict(container_name=c['name']+'-'+config['attempt'],physical_devices=c['cards'],image=c['image']),
            named_rpc_overlay=dict(files=refs,precision_ref=pin(bundle/'PRECISION-SPEC.json',target/'bundle/PRECISION-SPEC.json')),
            local_activation=dict(source_boot=c['host_boot_id'],guard=c['guard'],model=c['model'],bounded_candidate=True,
                locations_ref=pin(bundle/'LOCATIONS.json',target/'bundle/LOCATIONS.json'),
                rows_ref=pin(bundle/'ROWS.json',target/'bundle/ROWS.json'),
                **({} if formal else dict(diagnostic_logits=dict(read(inputs/'LOGITS.json'),
                    **({'prefix_trace':config['prefix_trace']} if config.get('prefix_trace') else {}),
                    **({'selection':config['logits_selection']} if 'logits_selection' in config else {})))))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(read(a.config),a.output)
