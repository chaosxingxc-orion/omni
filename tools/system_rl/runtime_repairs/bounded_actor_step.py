"""Run one accepted cohort through the original actor, schedule and loss."""
import hashlib
from contextlib import contextmanager
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


def prepare(spec,sealed,world,packet=None):
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
    if packet is None:
        packet=torch.load(sealed(spec['accepted_fields']),weights_only=False,map_location='cpu')
    fields=dict(tensors=packet['tensors'],non_tensors={},meta_info=packet['meta'])
    for k,values in packet['non_tensors'].items():
        a=np.empty(len(values),dtype=object);a[:]=values;fields['non_tensors'][k]=a
    report=packet['report'];task=upstream()
    runtime=read(ROOT/'jev-system-rl/implementation/runtime.json')['training']
    require(not packet['meta']['synthetic'] and report['required_fields_validated']
            and spec['dataset_sha256']==packet['dataset']['sha256'], 'expected accepted real TRAIN batch')
    prior_step=spec.get('continuation',{}).get('prior_step',0)
    require(packet['clock']['successful_optimizer_steps']==prior_step,'consumer Clock not at requested predecessor step')
    selected=set(packet['selected_groups'])
    require(len(selected)==len(packet['selected_groups']) and selected
            and selected=={gid for gid,g in packet['clock']['groups'].items() if not g['consumed']},
            'consumer reservations must contain exactly the fresh groups')
    if not prior_step:
        require(set(packet['clock']['groups'])==selected,'initial TRAIN batch contains prior history')
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
        require(all(clock.reserve(g)==base.snapshot for g in packet['selected_groups']),'Clock snapshot differs')
    return packet,fields,plan,runtime,clock


def checkpoint_contract(spec):
    return dict(schema='omni-bounded-checkpoint-v1',lineage=spec.get('lineage',spec['name']),
                **(dict(lora_arithmetic=dict(geometry=spec['lora_row_geometry'],source_sha256=spec['files']['lora_row_precision.py']))
                   if 'continuation' in spec and 'lora_row_geometry' in spec else {}),
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


def verify_continuation(spec, prior, packet=None):
    """Authenticate a transition without changing the predecessor manifest."""
    from dependencies import require
    from policy_identity import digest
    c=spec['continuation'];contract=checkpoint_contract(spec);old=prior['contract']
    require(type(c['prior_step']) is int and c['prior_step']>0 and c['prior_step']==prior['step']
            ==prior['clock']['successful_optimizer_steps'] and prior['world_size']==8,
            'checkpoint predecessor step/topology differs')
    require(spec.get('deterministic_backward') is True and spec['devices']==list(range(8)),
            'continuation requires strictly deterministic FSDP8')
    for key in ('schema','lineage','training','devices','image'):
        require(old[key]==contract[key],'checkpoint continuation changed '+key)
    if 'lora_arithmetic' in old:
        require(old['lora_arithmetic']==contract.get('lora_arithmetic'),'restored LoRA arithmetic differs')
    handoff={'bounded_actor_step.py'}
    if 'generation' in spec:
        handoff.add('retained_hf_fsdp_control.py')
        require(old['sources']['retained_hf_fsdp_control.py']==c['predecessor_control_sha256'],
                'predecessor worker entry pin differs')
    require(set(old['sources'])==set(contract['sources']) and all(
        old['sources'][name]==sha for name,sha in contract['sources'].items()
        if name not in handoff),'effective actor implementation changed')
    require(old['sources']['bounded_actor_step.py']==c['predecessor_driver_sha256'],
            'predecessor driver pin differs')
    if 'generation' not in spec or packet is not None:
        require(old['accepted_fields_sha256']!=contract['accepted_fields_sha256']
                and old['dataset_sha256']!=contract['dataset_sha256'],'cannot reuse prior training cohort')
    if packet is None:return
    before=prior['clock'];after=packet['clock'];selected=set(packet['selected_groups'])
    require(all(after[k]==v for k,v in before.items() if k not in ('config_sha256','config_version','groups'))
            and set(after['groups'])==set(before['groups'])|selected and selected.isdisjoint(before['groups'])
            and all(after['groups'][k]==v and v['consumed'] for k,v in before['groups'].items()),
            'consumer did not inherit exact committed history')
    version='checkpoint-sha256:'+prior['ranks'][0]['files']['lora_adapter/adapter_model.safetensors']
    learning=packet['learning'];vectors=learning['kernel']
    require(after['config_sha256']==digest(learning)
            and vectors['current_policy_vector']==vectors['behavior_policy_vector']
            ==dict.fromkeys(('P','S1','S2','D'),version),'consumer policy does not match predecessor export')


def restore_continuation(worker,spec):
    from checkpoint import restore
    from dependencies import require
    from run_package import pin,read
    checkpoint=Path('/prior-checkpoint');ref=pin(checkpoint/'COMMITTED.json')
    require(ref['sha256']==spec['restore_checkpoint']['sha256'],'restore checkpoint identity differs')
    prior=read(ref)
    verify_continuation(spec,prior)
    restored,_=restore(worker,checkpoint,prior['contract'])
    worker.audit_master_optimizer(restored['step'])
    return ref,prior


@contextmanager
def generation_sources(spec):
    """Use the capture release only while scoring and packing the new cohort."""
    import sys
    from dependencies import load
    from capture_runtime import capture_runtime
    sources=capture_runtime(spec)
    names=('publication','bounded_tp4','fresh_adapter','finalize_capture')
    saved={name:sys.modules.pop(name,None) for name in names}
    try:
        for name in names:
            load(sources[name],name)
        yield sources
    finally:
        for name,module in saved.items():
            sys.modules.pop(name,None)
            if module is not None:
                sys.modules[name]=module


def check_rollout_geometry(spec, captured, read):
    witness=read(captured['capture']['endpoint_policy_ref'])
    geometry=spec.get('lora_row_geometry')
    if witness.get('schema')!='omni-bounded-tp4-endpoints-v1':
        if geometry is not None:raise ValueError('LoRA arithmetic requires paired endpoint evidence')
        return
    expected=None if geometry is None else dict(geometry,source_sha256=spec['files']['lora_row_precision.py'])
    lanes=witness['lanes']
    if len(lanes)!=2:raise ValueError('expected both bounded endpoints')
    for name,ref in lanes.items():
        endpoint=read(ref)
        workers=read(endpoint['precision_ref'])['workers']
        if (endpoint['lane']!=name or len(workers)!=4 or {w['rank'] for w in workers}!=set(range(4))
                or any(w.get('lora_row_geometry')!=expected for w in workers)):
            raise ValueError('actor and observed rollout LoRA arithmetic differ: '+name)


def capture_preflight(spec,out):
    """Exercise actual policy, receipts and official labels in the scorer namespace."""
    from dependencies import ROOT,require
    from run_package import pin,read
    config=spec['generation']
    inventory=read(config['generation_inventory_ref'])
    for ref in inventory['files'].values():
        pin(ref)
    native=read(config['pending_capture_ref'])
    pending=read(native['pending_capture_ref']); captured=read(pending['input_ref'])
    check_rollout_geometry(spec,captured,read)
    directory=Path(pending['cohort_root'])
    selected=read(directory/'SELECTED.json')
    consumer=read(directory/'CONSUMER.json')
    local=pin(ROOT/'training-consumer-adapter/adapter.py')
    require(pin(consumer['sealed_adapter_ref'])['sha256']==local['sha256'],'sealed consumer adapter changed')
    with generation_sources(spec) as runtime:
        from fresh_adapter import scope
        with scope():
            from dependencies import load
            load(runtime['cohort_source'],'cohort')
            scorer=load(runtime['capture_source'],'dp8_preflight_scorer')
            from inspect import signature
            if not {'resident','cost_receipt','directory'} <= set(signature(scorer.finalize_on_worker).parameters):
                raise ValueError('unexpected continuation scorer interface')
            from policy import validate
            validate(captured['policy'],read(captured['learning_config_ref']),read(directory/'RESERVED-CLOCK.json'))
            from official import sources,annotation
            from storage import commit
            official=read(captured['capture']['official_source_config_ref'])
            official['index_cache_root']=runtime['index_root']
            path=out/'LOCAL-OFFICIAL-SOURCE.json'; commit(path,official)
            captured['capture']['official_source_config_ref']=pin(path)
            with sources(captured) as (bound,_):
                for row in selected['draws']:
                    annotation(row,bound)
    return dict(groups=len(selected['draws']), inventory_files=len(inventory['files']))


def finalize_and_run(worker,adapted,spec,out,rank,world,sealed):
    """Score sealed generation on the same restored worker that will update."""
    import time
    import torch.distributed as dist
    from dependencies import require
    from run_package import pin
    from storage import commit
    sealed('pack_actor_fields.py')
    from pack_actor_fields import build_packet
    config=spec['generation']
    ref,prior=restore_continuation(worker,spec)
    require(prior['step']==spec['continuation']['prior_step'],'scoring checkpoint step differs')
    started=time.monotonic()
    with generation_sources(spec) as runtime:
        from finalize_capture import restored_worker as finalize
        from fresh_adapter import scope
        finalize(worker,adapted,config,out,rank,sources=runtime)
        with scope():
            packet=build_packet(out/runtime['cohort_dir'])
    # Canonical consumer receipt plus exact sealed rollout bytes identify the
    # in-memory handoff; there is no second deserialization or model restore.
    spec=dict(spec,dataset_sha256=packet['dataset']['sha256'],
              accepted_fields='finalized_generation',files=dict(spec['files']))
    spec['files']['finalized_generation']=pin(config['generation_inventory_ref'])['sha256']
    verify_continuation(spec,prior,packet)
    batch=prepare(spec,sealed,world,packet)
    commit(out/f'rank-{rank}/ACCEPTED-COHORT.json',dict(
        checkpoint=ref,dataset=packet['dataset'],group_ids=packet['selected_groups'],
        episodes=packet['report']['global_episode_denominator'],action_tokens=packet['report']['action_tokens'],
        finalize_seconds=time.monotonic()-started))
    dist.barrier()
    return run(worker,spec,out,rank,world,batch,restored=(ref,prior))


def run(worker,spec,out,rank,world,batch,*,restored=None):
    from dependencies import require
    from storage import commit,check_running
    from events import run_existing_actor
    packet,fields,plan,runtime,clock=batch
    recoverable=clock is not None
    contract=checkpoint_contract(spec)
    step=1
    if 'continuation' in spec:
        require(recoverable and spec.get('restore_checkpoint'),'continuation requires full optimizer state')
        ref,prior=restored if restored is not None else restore_continuation(worker,spec)
        verify_continuation(spec,prior,packet)
        step=prior['step']+1
    elif spec.get('restore_checkpoint'):
        ref,restored=recover(worker,spec,batch)
        commit(out/f'rank-{rank}/RESULT.json',dict(rank=rank,step=1,restored=True,optimizer_updates=0,
            checkpoint_sha256=ref['sha256'],publication_sha256=restored['resident_identity']['publication']['identity_sha256']))
        return
    if spec.get('deterministic_backward'):
        import torch
        torch.use_deterministic_algorithms(True, warn_only=False)
        require(torch.are_deterministic_algorithms_enabled(), 'strict deterministic control not applied')
    # Historical first-step configs predate the shared parity object.
    limits=spec.get('generation',{}).get('parity',spec.get('behavior_parity',
        dict(token_atol=.01,token_rtol=0.,sequence_atol=.01)))
    heard=[]; worker.actor.actor_module.train()
    def evaluate(inputs):
        scores=compact_scores(worker.actor,inputs); mask=inputs['response_mask'].bool()
        row=list(fields['non_tensors']['record_identity']).index(inputs['record_identity'][0])
        old=fields['tensors']['old_log_probs'][row:row+1].to(scores)
        delta=(scores.detach()-old)[mask];m=float(delta.abs().max());seq=float(delta.sum().abs())
        require(bool((delta.abs()<=limits['token_atol']+limits['token_rtol']*old[mask].abs()).all())
                and seq<=limits['sequence_atol'],'effective train/behavior parity failed')
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
    if step>1:
        commit(out/f'rank-{rank}/UPDATE-INTENT.json',dict(step=step,checkpoint_sha256=ref['sha256'],
            dataset_sha256=spec['dataset_sha256'],group_ids=packet['selected_groups']))
    result=run_existing_actor(worker.actor,fields,plan,None,evaluate,actor_sha256=runtime['actor_sha256'],
        world=world,rank=rank,before_step=before_step,forward_api=runtime['forward_api'])
    require(result['native_optimizer_step'],'real optimizer did not update')
    if spec.get('trace_update'):capture(worker.actor,'after')
    audit=worker.audit_master_optimizer(step)
    worker.actor_lr_scheduler.step()
    commit(out/f'rank-{rank}/LOSS.json',dict(rank=rank,audit=audit,result=result,actions=heard,step=step))
    from policy_identity import artifact
    if recoverable:
        from checkpoint import save
        attempt=contract['lineage']+':'+str(step)
        clock.finish(attempt,packet['selected_groups'],True)
        kernel=packet['learning']['kernel']
        worker.checkpoint_lineage=dict(dataset=packet['dataset'],behavior_policy_vector=kernel['behavior_policy_vector'],
            current_policy_vector=kernel['current_policy_vector'],reference=kernel['reference_policy_vector'],
            annealing_snapshot=plan.snapshot,group_ids=packet['selected_groups'],attempt_id=attempt)
        check_running(out,spec['storage_budget_bytes'])
        save(worker,out/f'step-{step}',clock,contract,step)
    else:
        worker.export_adapter(out/'step-1')
    # Every rank independently reads the published BF16 delta.
    public=artifact(out/f'step-{step}/lora_adapter/adapter_model.safetensors')
    require(public['nonzero_effective_delta'],'exported empty effective update')
    commit(out/f'rank-{rank}/RESULT.json',dict(rank=rank,step=step,actions=heard,objective=result['objective_observation'],
            metrics=result['metrics'],publication_sha256=public['identity_sha256'],adapter_export=True,
            recoverable_checkpoint=recoverable,arithmetic=worker.close_actor_arithmetic()))
