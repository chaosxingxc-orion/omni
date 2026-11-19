"""Use the existing rollout owner for a single frozen TP4 boundary replay."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
from dispatch_retained_hf_boundary import save, state


def read(p):return json.loads(p.read_bytes())


def pin(p):
    b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def idle_replay(spec):
    """Run the same sealed diagnostic phase after intentional service release."""
    parent_plan=Path(spec['prepared_plan']).resolve()
    assert pin(parent_plan)['sha256']==spec['prepared_plan_sha256']
    plan=read(parent_plan)
    old=parent_plan.parent
    assert old==Path(plan['root']) and old.parent==Path(spec['output']).resolve().parent
    root=Path(spec['output']).resolve()
    assert root!=old and not root.exists()
    original=read(old/'CONTROLLER.json')['container_id']
    assert state(original)['Status']=='exited' and not (old/'hot/new-lease/NPU-LAUNCH.json').exists()
    assert plan['source_boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert len(plan['guard_paths'])==1
    guard_path=Path(plan['guard_paths'][0]).resolve()
    assert guard_path.is_file()
    inputs=Path(spec['inputs']).resolve()
    for name,digest in spec['input_files'].items():
        assert Path(name).name==name and pin(inputs/name)['sha256']==digest
    # Check the optional callback ABI before binding devices. A self-consistent
    # hash list can still accidentally combine two incompatible revisions.
    if spec.get('moe_origin'):
        import ast
        def keywords(filename, function):
            node, = [n for n in ast.parse((inputs/filename).read_bytes()).body
                     if isinstance(n, ast.FunctionDef) and n.name==function]
            return {a.arg for a in node.args.args + node.args.kwonlyargs}
        if not {'dispatcher_class','operands'}<=keywords('expert_window_vllm_capture.py','attach'):
            raise ValueError('MoE observer bundle has incompatible expert window ABI')
        if not {'moe_origin','expert_operands'}<=keywords('retained_decode_rows.py','begin'):
            raise ValueError('MoE observer bundle has incompatible predictor RPC ABI')
    for name in ('REPLAY.json','EXECUTION.json'):
        assert pin(old/name)['sha256']==spec['prepared_files'][name]
    with guard_path.open('r+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        devices=spec['devices']
        assert devices==list(range(4))
        smi=subprocess.check_output(['npu-smi','info'],text=True)
        assert all(f'No running processes found in NPU {i} ' in smi for i in devices)
        from pathlib import PurePosixPath
        hot=root/'hot'
        new=str(root)
        def relocate(text):
            return text.replace(str(old/'REPLAY.json'),new+'/REPLAY.json').replace(str(old/'EXECUTION.json'),new+'/EXECUTION.json')
        phase=[];oldargv=plan['npu_argv'];i=0;seen_devices=[]
        while i<len(oldargv):
            x=oldargv[i];i+=1
            if x=='--mount':
                m=oldargv[i];i+=1
                fields=m.split(',')
                src=next(k[4:] for k in fields if k.startswith('src='))
                dst=next(k[4:] for k in fields if k.startswith('dst='))
                if src=='/var/run/docker.sock' or dst==str(guard_path):
                    continue
                source=Path(src)
                runtime_target=None
                if dst=='/e':
                    assert 'readonly' in fields
                    fields=[f'src={inputs}' if k.startswith('src=') else k for k in fields]
                if source.is_relative_to(old/'hot'):
                    runtime_target=hot/source.relative_to(old/'hot')
                elif source==old:
                    runtime_target=root
                if runtime_target is not None:
                    fields=[f'src={runtime_target}' if k.startswith('src=') else
                            'dst='+str(PurePosixPath(new)/source.relative_to(old)) if k.startswith('dst=') and dst==src else k for k in fields]
                elif 'readonly' not in fields:
                    fields.append('readonly')
                phase.extend((x,','.join(fields)))
            elif x=='--device':
                dev=oldargv[i];i+=1;seen_devices.append(dev);phase.extend((x,dev))
            elif x=='--name':
                i+=1;phase.extend((x,spec['name']))
            else:
                val=relocate(x).replace(str(old/'hot'),str(hot))
                phase.append(val)
        assert sorted(d for d in seen_devices if d.removeprefix('/dev/davinci').isdigit())==[f'/dev/davinci{i}' for i in devices]
        assert spec['image'] in phase
        assert subprocess.check_output(['docker','image','inspect','-f','{{.Id}}',spec['image']],text=True).strip()==spec['image']
        # Check the actual writable filesystem before creating hot state.
        import os
        stat=os.statvfs(root.parent)
        assert stat.f_bavail*stat.f_frsize>=max(20*1024**3,stat.f_blocks*stat.f_frsize*.10)+spec['storage_budget_bytes']
        assert stat.f_favail>=stat.f_files*.05
        assert subprocess.check_output(['findmnt','-n','-o','FSTYPE','-T',str(root.parent)],text=True).strip() in ('ext4','xfs','btrfs')
        root.mkdir(mode=0o700)
        for d in ('tmp','cache','home/ascend/log','home-ascend-log','ascend-cache','ascend-work','hf','inductor','triton','logs'):
            (hot/d).mkdir(parents=True,exist_ok=True)
        # Keep the original computation, source projection and input pins.
        replay=read(old/'REPLAY.json')
        assert replay['output']==str(old/'hot') and replay['input']=='/e'
        replay['output']=str(hot)
        if 'prefix_scan' in spec:
            scan = spec['prefix_scan']
            assert scan.keys()=={'rows','layers'} and type(scan['rows']) is int and 0<scan['rows']<=8192
            assert type(scan['layers']) is int and 0<scan['layers']<=8
            assert 'retained_prefix_scan.py' in spec['input_files']
            replay['trace_predictor']['prefix_scan']=scan
        if 'attention_layer' in spec:
            layer = spec['attention_layer']
            assert type(layer) is int and 0<=layer<48
            replay['trace_predictor']['attention_layer']=layer
        if 'trace_action_ordinal' in spec:
            ordinal = spec['trace_action_ordinal']
            assert type(ordinal) is int and ordinal>=1
            replay['trace_predictor']['action_ordinal']=ordinal
        if 'attention_shape_shadow' in spec:
            shadow = spec['attention_shape_shadow']
            assert (shadow is True or shadow == 'variants') and 'attention_layer' in spec
            assert 'retained_attention_operands.py' in spec['input_files']
            replay['trace_predictor']['attention_shape_shadow']=shadow
        if 'prefix_origin' in spec:
            origin=spec['prefix_origin']
            assert type(origin) is int and origin>=0
            assert {'retained_decode_rows.py','retained_action_replay.py','retained_prefix_origin.py'}<=spec['input_files'].keys()
            replay['trace_predictor']['prefix_origin']=origin
            if 'origin_layers' in spec:
                assert type(spec['origin_layers']) is int and 1<=spec['origin_layers']<=8
                replay['trace_predictor']['origin_layers']=spec['origin_layers']
        if spec.get('moe_origin'):
            assert 'prefix_origin' in spec and {'retained_moe_origin.py','expert_window_vllm_capture.py'}<=spec['input_files'].keys()
            replay['trace_predictor']['moe_origin']=True
        if 'moe_sites' in spec:
            sites = spec['moe_sites']
            assert not spec.get('moe_origin') and isinstance(sites,list) and 1<=len(sites)<=4
            assert all({'layer','position'}<=s.keys()<={'layer','position','down_operands'}
                       and type(s['layer']) is int and 0<=s['layer']<48
                       and type(s['position']) is int and 0<=s['position']<8192
                       and type(s.get('down_operands',False)) is bool for s in sites)
            assert len({s['layer'] for s in sites})==len(sites)
            assert {'retained_moe_origin.py','expert_window_vllm_capture.py'}<=spec['input_files'].keys()
            if any(s.get('down_operands') for s in sites):
                assert {'retained_expert_operands.py','expert_down_capture.py'}<=spec['input_files'].keys()
            replay['trace_predictor']['moe_sites']=sites
        if spec.get('expert_operands'):
            assert spec.get('moe_origin') and {'retained_expert_operands.py','expert_down_capture.py'}<=spec['input_files'].keys()
            replay['trace_predictor']['expert_operands']=True
        if 'gate_candidate_layers' in spec:
            assert 'retained_expert_candidate.py' in spec['input_files'] and not spec.get('expert_operands')
            replay['gate_candidate_layers']=spec['gate_candidate_layers']
        if 'attention_candidate_layers' in spec:
            assert 'retained_attention_candidate.py' in spec['input_files'] and not spec.get('attention_shape_shadow')
            replay['attention_candidate_layers']=spec['attention_candidate_layers']
        if 'down_candidate_layers' in spec:
            assert 'retained_expert_candidate.py' in spec['input_files']
            assert not spec.get('expert_operands') and not any(s.get('down_operands') for s in spec.get('moe_sites',()))
            replay['down_candidate_layers']=spec['down_candidate_layers']
        if spec.get('native_audio'):
            del replay['audio_control']
        if 'prefix_cache_pair' in spec:
            assert spec['prefix_cache_pair'] == 'P' and spec.get('native_audio') is True
            assert spec.get('trace_predictor') is False
            replay['prefix_cache_pair'] = 'P'
        if 'audio_attention_layers' in spec:
            layers = spec['audio_attention_layers']
            assert spec.get('native_audio') is True
            assert layers and all(type(i) is int and i in range(32) for i in layers)
            assert layers == sorted(set(layers)) and 'retained_audio_attention_precision.py' in spec['input_files']
            replay['audio_attention_layers'] = layers
            if 'inspect_audio_layer' in spec:
                assert spec['inspect_audio_layer'] in layers
                replay['inspect_layer'] = spec['inspect_audio_layer']
        if spec.get('trace_predictor', True) is False:
            # Natural sampling can branch before the retained predictor. Keep
            # the original-action prefill scores and the true matching prefix.
            assert spec.get('native_audio') is True and replay['score_retained_prefill'] is True
            assert not any(k in spec for k in ('prefix_scan', 'attention_layer',
                           'trace_action_ordinal', 'attention_shape_shadow', 'prefix_origin',
                           'origin_layers', 'moe_origin', 'moe_sites', 'expert_operands'))
            del replay['trace_predictor']
        if 'fresh_group' in spec:
            assert spec.get('native_audio') and not spec.get('trace_predictor',True)
            assert 'retained_group_parity.py' in spec['input_files']
            if 'tiled_layers' in spec:
                assert 'kv_layers' in spec and set(spec['tiled_layers'])<=set(spec['attention_candidate_layers'])
                replay['tiled_layers']=spec['tiled_layers']
            if 'kv_layers' in spec:
                assert 'retained_kv_candidate.py' in spec['input_files']
                replay['kv_layers']=spec['kv_layers']
            if 'router_layers' in spec:
                assert 'retained_router_candidate.py' in spec['input_files']
                replay['router_layers']=spec['router_layers']
            if spec.get('isolate_fresh_paths'):
                assert spec['isolate_fresh_paths'] is True
                replay['isolate_fresh_paths']=True
            for flag in ('independent_cache_salts','fresh_prefill_control'):
                if flag in spec:
                    assert spec[flag] is True
                    replay[flag]=True
            if 'fresh_trace' in spec:
                assert (replay.get('isolate_fresh_paths') or (spec['fresh_trace'].get('batch') and replay.get('fresh_prefill_control') and replay.get('independent_cache_salts'))) and 'retained_prefix_scan.py' in spec['input_files']
                replay['fresh_trace']=spec['fresh_trace']
            if spec.get('compare_actor_frame_padding'):
                assert spec['compare_actor_frame_padding'] is True
                replay['compare_actor_frame_padding']=True
            filename=spec['fresh_group']
            replay['fresh_group']=dict(file=filename,sha256=spec['input_files'][filename])
        save(root/'REPLAY.json',replay)
        execution=read(old/'EXECUTION.json')
        execution.update(root=new,control_root=str(hot/'new-lease'),started=time.time(),
                         compute_deadline=time.time()+spec['timeout_seconds'],deadline=time.time()+spec['timeout_seconds']+60)
        save(root/'EXECUTION.json',execution)
        save(root/'IDLE-INTENT.json',dict(plan_sha256=spec['prepared_plan_sha256'],boot_id=plan['source_boot'],
                                         guard_inode=guard_path.stat().st_ino,devices=devices, prior_service_restored=False))
        cid=subprocess.check_output(phase,text=True).strip()
        save(root/'CONTAINER.json',dict(container_id=cid))
        try:
            code=int(subprocess.check_output(['docker','wait',cid],text=True,timeout=spec['timeout_seconds']))
        except subprocess.TimeoutExpired:
            subprocess.run(['docker','stop','-t','30',cid],check=True)
            code=int(subprocess.check_output(['docker','wait',cid],text=True))
        save(root/'EXIT.json',dict(container_id=cid,exit_code=code,
                                   result_exists=(hot/'RESULT.json').is_file()))
        if code or not (hot/'RESULT.json').is_file():
            raise SystemExit(1)


def main(spec):
    if 'previous_replay' in spec:
        prior=Path(spec['previous_replay'])
        cid=read(prior/'CONTROLLER.json')['container_id']
        watchdog=read(prior/'hot/new-lease/WATCHDOG.json')['container_id']
        for previous in (cid,watchdog):
            subprocess.run(['docker','wait',previous],check=True,stdout=subprocess.DEVNULL)
            assert state(previous)['ExitCode']==0
        assert read(prior/'hot/new-lease/RESTORED.json')['config_exact']
        exit_code=read(prior/'hot/new-lease/NPU-EXIT.json')['ExitCode']
        if spec.get('previous_succeeded') is True:
            assert exit_code==0
        else:
            assert exit_code!=0
    parent=Path(spec['parent_plan']).parent
    root=Path(spec['root']).resolve()
    assert pin(parent/'PLAN.json')['sha256']==spec['parent_sha256']
    s=read(parent/'PLAN.json')
    assert root.is_relative_to(parent.parents[2]) and parent not in root.parents
    old=parent/'hot/new-lease'; closed=read(parent/'OWNER-CLOSED.json')
    assert closed['unknown_upper']==0 and read(old/'RESTORED.json')['config_exact']
    watch=read(old/'WATCHDOG.json')
    for name in [watch['container_id'],*s['phase_names'][:2]]:
        assert state(name)['Status']=='exited'
    act=read(parent/'service-overlay/ACTIVATION.json')
    assert act['base'] and act['adapter_id'] is None
    old_service=read(parent/'service-CREATE-INTENT.json')['argv']
    old_watch=watch['argv']
    def origin(ref):
        bind=next(x for x in old_watch if ',dst='+ref['path']+',readonly' in x)
        p=Path(bind.split('src=')[1].split(',')[0]);assert pin(p)['sha256']==ref['sha256']
        return read(p)
    owner=origin(s['owner_config_ref'])
    assert owner['physical_devices']==list(range(4))
    image=owner['image']
    root.mkdir(mode=0o700)
    hot=root/'hot'
    for name in ['tmp','cache','home/ascend/log','home-ascend-log','ascend-cache','ascend-work','hf','inductor','triton','logs']:
        (hot/name).mkdir(parents=True,exist_ok=True)
    inputs=Path(spec['inputs'])
    for k,v in spec['input_files'].items():
        assert Path(k).name==k and pin(inputs/k)['sha256']==v
    request=origin(s['request_ref'])
    request['retained_boundary_replay']=dict(parent=s['request_ref'],base=act['policy_version'],files=spec['input_files'],
                                             diagnostic_root=str(root),new_sampling='retained-actions.json' in spec['input_files'])
    save(root/'REPLAY-REQUEST.json',request);s['request_ref']=pin(root/'REPLAY-REQUEST.json')
    name='retained-tp4-'+s['request_ref']['sha256'][:14]
    pc=read(parent/'service-overlay/PRECISION.json')['specification_ref']
    original=owner['service_args']; model=original[original.index('--model')+1]
    engine=dict(model=model,tensor_parallel_size=4,enforce_eager=True,async_scheduling=False,enable_prefix_caching=True,
                enable_lora=True,max_loras=1,max_lora_rank=int(original[original.index('--max-lora-rank')+1]),
                lora_dtype='float32',dtype='bfloat16',generation_config='vllm',logprobs_mode='raw_logprobs',
                max_model_len=int(original[original.index('--max-model-len')+1]),max_num_seqs=int(original[original.index('--max-num-seqs')+1]),
                gpu_memory_utilization=float(original[original.index('--gpu-memory-utilization')+1]))
    args=dict(input='/e',output=str(hot),named_rpc_overlay=read(parent/'SERVER.json')['named_rpc_overlay'],action_index=spec['action_index'],engine=engine,precision_ref=pc,
              behavior=dict(file='P-BEHAVIOR.json',sha256=spec['input_files']['P-BEHAVIOR.json']),
              features=dict(file='features.json',sha256=spec['input_files']['features.json']))
    if 'inspect_layer' in spec:
        assert type(spec['inspect_layer']) is int and 0<=spec['inspect_layer']<32
        args['inspect_layer']=spec['inspect_layer']
    if 'audio_row_sites' in spec:
        args['audio_row_sites']=spec['audio_row_sites']
    if 'retained-actions.json' in spec['input_files']:
        args['retained_actions']=dict(file='retained-actions.json',sha256=spec['input_files']['retained-actions.json'])
    if 'trace_predictor' in spec:
        trace=spec['trace_predictor']
        assert spec.get('score_retained_prefill') is True and 'retained_decode_rows.py' in spec['input_files']
        assert trace['node']=='P' and 1<=trace['action_ordinal']<=10 and trace['rope_offset']==0
        args['trace_predictor']=trace
    if 'score_retained_prefill' in spec:
        assert spec['score_retained_prefill'] is True and 'retained_actions' in args
        args['score_retained_prefill']=True
    if 'audio_control' in spec:
        control=spec['audio_control']
        assert 'retained_actions' in args and control['file']=='HF.safetensors'
        args['audio_control']=dict(path='/e/HF.safetensors',tensor_file_sha256=spec['input_files'][control['file']],
                                   audio_sha256=control['audio_sha256'])
    save(root/'REPLAY.json',args)
    def base(argv):
        return [x.replace(str(parent/'hot'),str(hot)) for x in argv]
    phase=base(old_service[:old_service.index(image)])
    phase[phase.index('--name')+1]=name
    for i,x in enumerate(phase):
        if x.startswith('PYTHONPATH='):phase[i]=x.replace('PYTHONPATH=','PYTHONPATH=/e:',1)
        if x.startswith('OMNI_ITERATION_EXECUTION_REF='):phase[i]='OMNI_ITERATION_EXECUTION_REF='+str(root/'EXECUTION.json')
    ix=phase.index('--entrypoint')
    phase[ix:ix]=['--mount',f'type=bind,src={root},dst={root}','--mount',f'type=bind,src={inputs},dst=/e,readonly']
    phase += [image,'-B','/e/retained_audio_vllm_boundary.py',str(root/'REPLAY.json')]
    if 'hf_text_image' in spec:
        pinned=spec['hf_text_image']
        assert pinned.startswith('sha256:') and subprocess.check_output(['docker','image','inspect','-f','{{.Id}}',pinned],text=True).strip()==pinned
        assert 'HF-TEXT.json' in spec['input_files'] and 'retained_hf_text_control.py' in spec['input_files']
        phase[phase.index('--entrypoint')+1]='python3'
        phase[-4:]=[pinned,'-B','/e/retained_hf_text_control.py','/e/HF-TEXT.json',model,str(hot/'HF-TEXT-RESULT.json')]
    t=time.time();compute=spec['compute_seconds'];reserve=spec['restore_seconds']
    assert 600<compute<=3600 and 600<=reserve<=3600
    s.update(root=str(root),npu_argv=phase,phase_names=[name],lease_started=t,compute_seconds=compute,
             total_seconds=compute+reserve,restore_seconds=reserve,absolute_compute_deadline=t+compute,absolute_deadline=t+compute+reserve)
    e=read(parent/'EXECUTION.json');e.update(root=str(root),request_ref=s['request_ref'],control_root=str(hot/'new-lease'),
        started=t,compute_deadline=s['absolute_compute_deadline'],deadline=s['absolute_deadline'])
    save(root/'EXECUTION.json',e);save(root/'PLAN.json',s)
    ctrl=base(old_watch);ctrl[ctrl.index('--name')+1]=name+'-owner'
    ctrl[ctrl.index('--watchdog'):]=['--dispatch-plan',str(root/'PLAN.json')]
    ix=ctrl.index('--entrypoint')
    mounted={x.split(',dst=')[1].split(',')[0] for x in ctrl if x.startswith('type=bind,') and ',dst=' in x}
    for m in [str(root),*owner['hot_paths']]:
        if m not in mounted:ctrl[ix:ix]=['--mount',f'type=bind,src={m},dst={m}']
    save(root/'CONTROLLER-INTENT.json',dict(argv=ctrl))
    cid=subprocess.check_output(ctrl,text=True).strip();save(root/'CONTROLLER.json',dict(container_id=cid));print(cid)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path)
    p.add_argument('--idle',action='store_true')
    a=p.parse_args();(idle_replay if a.idle else main)(read(a.config))
