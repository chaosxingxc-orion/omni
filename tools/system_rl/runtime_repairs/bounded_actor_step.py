"""Run one accepted cohort through the original actor, schedule and loss."""
import hashlib
from pathlib import Path


def compact_scores(actor,inputs):
    import torch
    from forward import forward
    from dependencies import require
    mask=inputs['attention_mask'][0].bool(); actions=inputs['response_mask'][0].bool()
    width=len(actions); length=int(actions.sum()); plen=len(mask)-width
    require(len(inputs['input_ids'])==1 and 0<length<=width and torch.equal(mask[-width:],actions)
            and bool(mask[plen-1]) and torch.equal(actions,actions.long().cumprod(0).bool())
            and bool((mask[:plen].long().diff()>=0).all()),'expected left-padded prompt / right-padded actions')
    one=dict(inputs)
    for key in ('input_ids','attention_mask','position_ids'):one[key]=inputs[key][...,mask]
    for key in ('responses','response_mask'):one[key]=inputs[key][...,:length]
    scores=forward(actor,one,temperature=1.,feature_dtype=torch.bfloat16)[1]
    return torch.nn.functional.pad(scores,(0,width-length))


def prepare(spec,sealed,world):
    import torch
    import torch.distributed as dist
    import numpy as np
    from dependencies import ROOT,require,upstream
    from run_package import pin
    from loss_kernel import VerlKernel
    # The sealed FSDP4 implementation predates the multi-topology registry.
    if world==4:profile="omni-bf16-tp4-fsdp4-v1"
    else:
        from actor_arithmetic import PROFILES
        profile=PROFILES[world]
    sealed('events.py')
    from events import run_existing_actor
    import inspect
    require(Path(inspect.getfile(run_existing_actor)).resolve()==sealed('events.py').resolve(),'unexpected actor event implementation')
    from runtime_builder import read
    # Consumer validation happened on CPU, pin-check before deserializing its local handoff.
    packet=torch.load(sealed(spec['accepted_fields']),weights_only=False,map_location='cpu')
    fields=dict(tensors=packet['tensors'],non_tensors={},meta_info=packet['meta'])
    for k,values in packet['non_tensors'].items():
        a=np.empty(len(values),dtype=object);a[:]=values;fields['non_tensors'][k]=a
    report=packet['report'];task=upstream()
    runtime=read(ROOT/'jev-system-rl/implementation/runtime.json')['training']
    require(not packet['meta']['synthetic'] and report['required_fields_validated'] and not packet['clock']['successful_optimizer_steps']
            and not any(g['consumed'] for g in packet['clock']['groups'].values()) and spec['dataset_sha256']==packet['dataset']['sha256'],
            'expected reserved initial TRAIN batch')
    require(spec['image']==runtime['image_id'] and spec['training']['actor_arithmetic_profile']==profile,
            'unexpected actor image/profile')
    kernel=task.bridge.SourceKernel({n:pin(ROOT/'system-objective-cpu'/n) for n in ('objective.py','runtime.py')})
    base=task.bridge.PreparedBatch(fields,report,VerlKernel(kernel,core_sha256=runtime['core_sha256'],clip_ratio=packet['clip']))
    online={tuple(item['identity'][n] for n in ('group_id','episode_id','call_id','node')):item['online_input'] for item in packet['online']}
    plan=task.prepare(base,packet['sidecar'],online)
    require(not any(plan.coefficients.values()),'bounded policy-only step requires zero auxiliary coefficients')
    clock=None
    if spec['training'].get('checkpoint_storage_mode')=='lora_training_state':
        import copy
        from scorer_extension import install
        install(task)
        consumer=copy.deepcopy(packet['consumer'])
        for name,ref in consumer['source_pins'].items():
            folder='system-objective-cpu' if name in ('objective.py','runtime.py') else 'unified-learning'
            ref['path']=str(ROOT/folder/name)  # Same digest, sealed container namespace.
        clock=task.create_clock(packet['learning'],packet['clock'],consumer_config=consumer)
        require(set(clock.state['groups'])==set(packet['selected_groups']),'Clock group roster differs')
        require(all(clock.reserve(g)==base.snapshot for g in packet['selected_groups']),'Clock snapshot differs')
    return packet,fields,plan,runtime,clock


def checkpoint_contract(spec):
    return dict(schema='omni-bounded-checkpoint-v1',lineage=spec.get('lineage',spec['name']),
                training=spec['training'],devices=spec['devices'],image=spec['image'],
                sources={**spec['source_pins'],**{name:spec['files'][name] for name in (
                    'actor_arithmetic.py','bounded_actor_step.py','events.py','policy_identity.py',
                    'retained_attention_candidate.py','retained_expert_candidate.py',
                    'retained_router_candidate.py','retained_kv_candidate.py','BASE-ASSETS.json')}},
                accepted_fields_sha256=spec['files'][spec['accepted_fields']],dataset_sha256=spec['dataset_sha256'])


def recover(worker,spec,batch):
    from dependencies import require
    from checkpoint import restore
    from run_package import pin,read
    packet,*_=batch
    checkpoint=Path('/prior-checkpoint')
    ref=pin(checkpoint/'COMMITTED.json')
    require(ref['sha256']==spec['restore_checkpoint']['sha256'],'restore checkpoint identity differs')
    contract=checkpoint_contract(spec)
    if 'efficiency' in spec or 'updated_http_parity' in spec:
        # A no-update probe adds a separate entrypoint. It may not change any
        # scientific input, actual worker/model source, or saved shard identity.
        previous=read(ref)['contract']
        for n in ('bounded_actor_step.py','retained_hf_fsdp_control.py'):
            require(isinstance(previous['sources'][n],str),'missing original executable pin')
            contract['sources'][n]=previous['sources'][n]
    restored,_=restore(worker,checkpoint,contract)
    require(restored['step']==1 and restored['clock']['successful_optimizer_steps']==1
            and set(restored['clock']['groups'])==set(packet['selected_groups'])
            and all(g['consumed'] for g in restored['clock']['groups'].values()),'restored Clock differs')
    worker.audit_master_optimizer(1)
    return ref,restored


def run(worker,spec,out,rank,world,batch):
    from dependencies import require
    from storage import commit,check_running
    from events import run_existing_actor
    packet,fields,plan,runtime,clock=batch
    recoverable=clock is not None
    contract=checkpoint_contract(spec)
    if spec.get('restore_checkpoint'):
        ref,restored=recover(worker,spec,batch)
        commit(out/f'rank-{rank}/RESULT.json',dict(rank=rank,step=1,restored=True,optimizer_updates=0,
            checkpoint_sha256=ref['sha256'],publication_sha256=restored['resident_identity']['publication']['identity_sha256']))
        return
    if spec.get('deterministic_backward'):
        import torch
        torch.use_deterministic_algorithms(True, warn_only=False)
        require(torch.are_deterministic_algorithms_enabled(), 'strict deterministic control not applied')
    heard=[]; worker.actor.actor_module.train()
    def evaluate(inputs):
        scores=compact_scores(worker.actor,inputs); mask=inputs['response_mask'].bool()
        row=list(fields['non_tensors']['record_identity']).index(inputs['record_identity'][0])
        old=fields['tensors']['old_log_probs'][row:row+1].to(scores)
        delta=(scores.detach()-old)[mask];m=float(delta.abs().max());seq=float(delta.sum().abs())
        require(m<=.01 and seq<=.01,'effective train/behavior parity failed')
        heard.append(dict(row=row,tokens=int(mask.sum()),max_abs=m,sequence_abs=seq))
        return scores
    # Independent, read-only observation of the native optimizer. No extra backward.
    def capture(actor, when):
        import torch
        rows={}
        for name,p in actor.actor_module.named_parameters():
            if not p.requires_grad:continue
            assert 'lora_' in name
            row={'parameter':p.detach().cpu().clone()}
            if when=='before':
                assert p.grad is not None or p.numel()==0
                if p.grad is not None:row['gradient']=p.grad.detach().cpu().clone()
            else:
                state=actor.actor_optimizer.state.get(p,{})
                for key in ('exp_avg','exp_avg_sq','step'):
                    if p.numel():row[key]=state[key].detach().cpu().clone()
            rows[name]=row
        path=out/f'rank-{rank}'/f'UPDATE-{when}.pt'
        torch.save(rows,path)
    def before_step(actor):
        check_running(out,spec['storage_budget_bytes'])
        if spec.get('trace_update'):capture(actor,'before')
        commit(out/f'rank-{rank}/BACKWARDS.json',dict(actions=heard,rank=rank))
    result=run_existing_actor(worker.actor,fields,plan,None,evaluate,actor_sha256=runtime['actor_sha256'],
        world=world,rank=rank,before_step=before_step,forward_api=runtime['forward_api'])
    require(result['native_optimizer_step'],'real optimizer did not update')
    if spec.get('trace_update'):capture(worker.actor,'after')
    audit=worker.audit_master_optimizer(1)
    worker.actor_lr_scheduler.step()
    commit(out/f'rank-{rank}/LOSS.json',dict(rank=rank,audit=audit,result=result,actions=heard,step=1))
    from policy_identity import artifact
    if recoverable:
        from checkpoint import save
        clock.finish(contract['lineage']+':1',packet['selected_groups'],True)
        kernel=packet['learning']['kernel']
        worker.checkpoint_lineage=dict(dataset=packet['dataset'],behavior_policy_vector=kernel['behavior_policy_vector'],
            current_policy_vector=kernel['current_policy_vector'],reference=kernel['reference_policy_vector'],
            annealing_snapshot=plan.snapshot,group_ids=packet['selected_groups'],attempt_id=contract['lineage']+':1')
        check_running(out,spec['storage_budget_bytes'])
        save(worker,out/'step-1',clock,contract,1)
    else:
        worker.export_adapter(out/'step-1')
    # Every rank independently reads the published BF16 delta.
    public=artifact(out/'step-1/lora_adapter/adapter_model.safetensors')
    require(public['nonzero_effective_delta'],'exported empty effective update')
    commit(out/f'rank-{rank}/RESULT.json',dict(rank=rank,step=1,actions=heard,objective=result['objective_observation'],
            metrics=result['metrics'],publication_sha256=public['identity_sha256'],adapter_export=True,
            recoverable_checkpoint=recoverable,arithmetic=worker.close_actor_arithmetic()))
