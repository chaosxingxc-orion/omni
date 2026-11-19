"""Use the existing rollout owner for a single frozen TP4 boundary replay."""
import argparse
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
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);a=p.parse_args();main(read(a.config))
