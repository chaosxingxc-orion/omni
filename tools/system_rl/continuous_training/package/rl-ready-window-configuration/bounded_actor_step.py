"""Run one accepted cohort through the original actor, schedule and loss."""
import hashlib
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def actor_batch_geometry(adapted, profile):
    """Keep rollout geometry at batch-sensitive sites through checkpoint recomputation."""
    from contextlib import ExitStack
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    if not profile:
        yield
        return
    if profile!='single_clip_compact_qo_v1':
        raise ValueError('unknown actor batch arithmetic')
    def inner(m):
        while type(m).__name__ in ('CheckpointWrapper','FullyShardedDataParallel'):
            m=m.module if type(m).__name__=='FullyShardedDataParallel' else m._checkpoint_wrapped_module
        return m
    body=adapted.get_base_model();tower=inner(body.audio_tower)
    layout=None
    def capture(module,args,kw):
        nonlocal layout
        mask=kw['attention_mask'].bool()
        if mask.shape[0]==1:
            layout=None
            return
        n=mask.shape[1]
        if mask.ndim!=2 or not mask.any(-1).all():
            raise ValueError('invalid actor attention layout')
        starts=mask.int().argmax(-1);widths=mask.long().sum(-1)
        positions=torch.arange(n,device=mask.device)
        if not torch.equal(mask,(positions>=starts[:,None])&(positions<(starts+widths)[:,None])):
            raise ValueError('noncontiguous actor rows')
        layout=(tuple(mask.shape),list(zip(starts.tolist(),(starts+widths).tolist())))
    def replace(stack,module,build):
        prior=vars(module).get('forward')
        fn=build(module.forward)
        module.forward=fn
        if prior is None:stack.callback(delattr,module,'forward')
        else:stack.callback(setattr,module,'forward',prior)
    def audio(orig):
        def per_clip(x,*,feature_lens,return_dict,**kw):
            if feature_lens.numel()==1:return orig(x,feature_lens=feature_lens,return_dict=return_dict,**kw)
            if set(kw)&{'padded_feature','chunk_lengths','valid_indices','cu_seqlens'} or not return_dict or x.ndim!=2 or feature_lens.ndim!=1:
                raise ValueError('unexpected packed audio call')
            parts=[orig(v.contiguous(),feature_lens=feature_lens[i:i+1],return_dict=True,**kw).last_hidden_state
                   for i,v in enumerate(x.split(feature_lens.tolist(),dim=1))]
            return BaseModelOutputWithPooling(last_hidden_state=torch.cat(parts))
        return per_clip
    def projection(orig):
        def forward(x,*args,**kw):
            if x.shape[0]==1:return orig(x,*args,**kw)
            if layout is None or tuple(x.shape[:2])!=layout[0] or args or kw:
                raise ValueError('projection input differs from actor row layout')
            return torch.cat([torch.nn.functional.pad(orig(x[i:i+1,lo:hi].contiguous()),
                                    (0,0,lo,x.shape[1]-hi)) for i,(lo,hi) in enumerate(layout[1])],dim=0)
        return forward
    with ExitStack() as stack:
        stack.callback(body.register_forward_pre_hook(capture,with_kwargs=True).remove)
        replace(stack,tower,audio)
        for layer in body.model.layers:
            attn=inner(inner(layer).self_attn)
            for name in ('q_proj','o_proj'):replace(stack,inner(getattr(attn,name)),projection)
        yield


def compact_scores(actor,inputs):
    import torch
    from forward import forward
    from dependencies import require
    mask=inputs['attention_mask'].bool(); actions=inputs['response_mask'].bool()
    width=actions.shape[1]; plen=mask.shape[1]-width
    require(plen>0 and bool(actions.any(-1).all()) and torch.equal(mask[:,plen:],actions)
            and bool(mask[:,plen-1].all()) and torch.equal(actions,actions.long().cumprod(-1).bool())
            and bool((mask[:,:plen].long().diff(dim=-1)>=0).all()),'expected left-padded prompt / right-padded actions')
    # Remove only columns which are padding for every member of the microbatch.
    # Keep the same right-aligned prompts and first-action predictor per row.
    keep=mask.any(0);length=int(actions.sum(-1).max())
    one=dict(inputs)
    for key in ('input_ids','attention_mask','position_ids'):one[key]=inputs[key][...,keep]
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


def restore_updated_probe(worker,spec):
    from checkpoint import restore
    from dependencies import require
    from run_package import pin,read
    path=Path('/prior-checkpoint')
    ref=pin(path/'COMMITTED.json')
    prior=read(ref)
    contract=checkpoint_contract(spec)
    old=prior['contract']
    require(ref['sha256']==spec['restore_checkpoint']['sha256']
            and prior['step']==prior['clock']['successful_optimizer_steps']==spec['parity_restore_step']
            and old==spec['diagnostic_commit'] and prior['world_size']==8
            and all(old[k]==contract[k] for k in ('lineage','training','devices','image')),
            'wrong diagnostic predecessor')
    ignored={'bounded_actor_step.py','retained_hf_fsdp_control.py','worker_dp8.py','SOURCE-PINS.json'}
    require({k:v for k,v in old['sources'].items() if k not in ignored}
            =={k:v for k,v in contract['sources'].items() if k not in ignored}
            and old['lora_arithmetic']==dict(geometry=spec['lora_row_geometry'],
                source_sha256=spec['files']['lora_row_precision.py']), 'diagnostic arithmetic differs')
    require(old['sources']['worker_dp8.py']==contract['sources']['worker_dp8.py'],
            'diagnostic worker differs')
    restored,_=restore(worker,path,old)
    worker.audit_master_optimizer(restored['step'])
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
    if 'batch_probe' in spec:
        # New batch schedule and multi-row attention are measured candidates,
        # never silently inherited by a formal continuation.
        probe=spec['batch_probe']
        for n in ('events.py','retained_attention_candidate.py'):
            require(old['sources'][n]==probe['predecessor_sources'][n] and
                    contract['sources'][n]==probe['candidate_sources'][n], 'unbound batch trial source')
            handoff.add(n)
    elif 'batch_migration' in spec:
        migrated=spec['batch_migration']
        require(spec.get('formal_sampled') and spec.get('actor_batch') and
                migrated['prior_sha256']==spec['restore_checkpoint']['sha256'],'unbound measured batch transition')
        for name in ('events.py','bounded_actor_step.py','retained_attention_candidate.py','retained_hf_fsdp_control.py'):
            require(old['sources'][name]==migrated['predecessor_sources'][name] and
                    contract['sources'][name]==migrated['candidate_sources'][name],
                    'measured batch transition differs: '+name)
            handoff.add(name)
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
    if spec.get('formal_sampled'):
        from run_package import read
        expected=read(spec['generation']['pending_capture_ref'])['attempt_id']
        require(isinstance(expected,str) and expected and packet.get('attempt_id')==expected
                and expected not in prior['clock']['attempts'],'formal optimizer batch missing or mismatched attempt_id')
    before=prior['clock']
    if c.get('zero_result_refs'):
        from run_package import read
        for receipt_ref in c['zero_result_refs']:
            before=verified_zero_clock(before,read(receipt_ref),spec['restore_checkpoint']['sha256'])
    after=packet['clock'];selected=set(packet['selected_groups'])
    require(all(after[k]==v for k,v in before.items() if k not in ('config_sha256','config_version','groups'))
            and set(after['groups'])==set(before['groups'])|selected and selected.isdisjoint(before['groups'])
            and all(after['groups'][k]==v and v['consumed'] for k,v in before['groups'].items()),
            'consumer did not inherit exact committed history')
    version='checkpoint-sha256:'+prior['ranks'][0]['files']['lora_adapter/adapter_model.safetensors']
    learning=packet['learning'];vectors=learning['kernel']
    require(after['config_sha256']==digest(learning)
            and vectors['current_policy_vector']==vectors['behavior_policy_vector']
            ==dict.fromkeys(('P','S1','S2','D'),version),'consumer policy does not match predecessor export')


def verified_zero_clock(before, receipt, checkpoint_sha256):
    """Verify a terminal outcome chained to this checkpoint and only its new groups."""
    from dependencies import require
    from policy_identity import digest
    after=receipt['clock'];attempt=receipt['attempt_id'];groups=receipt['group_ids']
    require(receipt['status']=='no_gradient' and receipt['optimizer_step'] is False
            and receipt['optimizer_steps']==0 and receipt['checkpoint_sha256']==checkpoint_sha256
            and receipt['prior_clock_sha256']==digest(before),
            'zero-gradient attempt does not extend this recovery anchor')
    require(groups and len(groups)==len(set(groups)) and not set(groups)&set(before['groups'])
            and attempt not in before['attempts'],'zero-gradient attempt reuses consumed history')
    require(after['attempts']==dict(before['attempts'],**{attempt:dict(group_ids=groups,succeeded=False)})
            and set(after['groups'])==set(before['groups'])|set(groups)
            and all(after['groups'][key]==value for key,value in before['groups'].items())
            and all(after['groups'][key]['consumed'] is True and
                    after['groups'][key]['snapshot']['successful_optimizer_steps']==before['successful_optimizer_steps']
                    for key in groups)
            and {k:v for k,v in after.items() if k not in ('groups','attempts','config_version','config_sha256')}
                =={k:v for k,v in before.items() if k not in ('groups','attempts','config_version','config_sha256')},
            'zero-gradient attempt changed committed Clock history')
    return after


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
    if witness.get('schema') not in ('omni-bounded-tp4-endpoints-v1','continuous-training-endpoints-v1'):
        if geometry is not None:raise ValueError('LoRA arithmetic requires paired endpoint evidence')
        return
    expected=None if geometry is None else dict(geometry,source_sha256=spec['files']['lora_row_precision.py'])
    lanes=witness['lanes']
    if spec.get('formal_sampled'):
        expected_lanes = spec['continuation']['rollout_lanes']
        if (not captured.get('sampled_window_ref') or set(lanes) != set(expected_lanes)
                or set(lanes) != set(captured['capture']['lanes'])):
            raise ValueError('endpoint witness does not cover the configured formal lanes')
    elif len(lanes)!=2:raise ValueError('expected both bounded endpoints')
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


def finalize_and_run(worker,adapted,spec,out,rank,world,sealed,*,live=None):
    """Score sealed generation on the same restored worker that will update."""
    import time
    import torch.distributed as dist
    from dependencies import require
    from run_package import pin
    from storage import commit
    sealed('pack_actor_fields.py')
    from pack_actor_fields import build_packet
    config=spec['generation']
    if live is None:
        ref,prior=restore_continuation(worker,spec)
    else:
        ref,prior=live
        require(ref['sha256']==spec['restore_checkpoint']['sha256'],'wrong live optimizer predecessor')
        verify_continuation(spec,prior)
    require(prior['step']==spec['continuation']['prior_step'],'scoring checkpoint step differs')
    started=time.monotonic()
    with generation_sources(spec) as runtime:
        from finalize_capture import restored_worker as finalize
        from fresh_adapter import scope
        root=out/runtime['cohort_dir']
        if runtime.get('actor_fields')=='reuse_accepted_v1':
            from pack_actor_fields import seal_packet,load_packet
            require(live is not None and spec.get('formal_sampled'),'reuse requires a resident sampled actor')
            inventory=config['generation_inventory_ref']
            finalize(worker,adapted,config,out,rank,sources=runtime,
                     accepted_sink=lambda folder,fields,report:seal_packet(folder,fields,report,inventory))
            from resident import collectively
            packet=collectively(lambda:load_packet(root,inventory))
        else:
            finalize(worker,adapted,config,out,rank,sources=runtime)
            with scope():
                packet=build_packet(root)
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
    with actor_batch_geometry(adapted,spec.get('actor_batch',{}).get('arithmetic')):
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
    continuous='continuous_actor' in spec
    if continuous:
        require(restored is not None and spec.get('formal_sampled') and not spec.get('trace_update')
                and spec['generation']['runtime'].get('reference_batch'),
                'continuous update requires a sealed rollout and batched reference')
    heard=[]; worker.actor.actor_module.train()
    def evaluate(inputs):
        scores=compact_scores(worker.actor,inputs)
        if continuous:return scores
        mask=inputs['response_mask'].bool()
        identities=list(fields['non_tensors']['record_identity'])
        for i,identity in enumerate(inputs['record_identity']):
            row=identities.index(identity)
            old=fields['tensors']['old_log_probs'][row].to(scores)
            delta=(scores[i].detach()-old)[mask[i]];m=float(delta.abs().max());seq=float(delta.sum().abs())
            require(bool((delta.abs()<=limits['token_atol']+limits['token_rtol']*old[mask[i]].abs()).all())
                    and seq<=limits['sequence_atol'],'effective train/behavior parity failed: row '+str(row))
            heard.append(dict(row=row,tokens=int(mask[i].sum()),max_abs=m,sequence_abs=seq))
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
        if not continuous:commit(out/f'rank-{rank}/BACKWARDS.json',dict(actions=heard,rank=rank))
    if step>1:
        commit(out/f'rank-{rank}/UPDATE-INTENT.json',dict(step=step,checkpoint_sha256=ref['sha256'],
            dataset_sha256=spec['dataset_sha256'],group_ids=packet['selected_groups']))
    result=run_existing_actor(worker.actor,fields,plan,None,evaluate,actor_sha256=runtime['actor_sha256'],
        world=world,rank=rank,before_step=before_step,forward_api=runtime['forward_api'],
        **({'micro_batch':spec['actor_batch']['microbatch_per_card']} if 'actor_batch' in spec else {}))
    if result.get('status')=='no_gradient' and spec.get('formal_sampled'):
        from policy_identity import digest
        from run_package import read
        require(recoverable and 'continuation' in spec and result['optimizer_step'] is False,
                'formal zero-gradient outcome requires an exact optimizer recovery anchor')
        baseline=prior['clock']
        for receipt_ref in spec['continuation'].get('zero_result_refs',()):
            baseline=verified_zero_clock(baseline,read(receipt_ref),ref['sha256'])
        attempt=packet['attempt_id'];groups=packet['selected_groups']
        clock.finish(attempt,groups,False)
        for gid in groups:
            clock.state['groups'][gid]['consumed']=True
        worker.audit_master_optimizer(step-1)
        receipt=dict(status='no_gradient',optimizer_step=False,optimizer_steps=0,clock=clock.state,
                     prior_clock_sha256=digest(baseline),checkpoint_sha256=ref['sha256'],
                     attempt_id=attempt,group_ids=groups,dataset=packet['dataset'])
        verified_zero_clock(baseline,receipt,ref['sha256'])
        commit(out/f'rank-{rank}/RESULT.json',dict(receipt,rank=rank,step=step-1))
        return ref if continuous else None
    require(result.get('native_optimizer_step') is True,'real optimizer did not update')
    if spec.get('trace_update'):capture(worker.actor,'after')
    audit=worker.audit_master_optimizer(step)
    worker.actor_lr_scheduler.step()
    commit(out/f'rank-{rank}/LOSS.json',dict(rank=rank,audit=audit,result=result,actions=heard,step=step))
    from policy_identity import artifact
    if recoverable:
        from checkpoint import save
        attempt=packet['attempt_id'] if spec.get('formal_sampled') else contract['lineage']+':'+str(step)
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
    if continuous:
        # The checkpoint writer already read the export and proved equality to
        # the resident adapter. Keep that proof; do not load it a second time.
        from run_package import pin,read
        committed=pin(out/f'step-{step}/COMMITTED.json')
        public=read(committed)['resident_identity']['publication']
    else:public=artifact(out/f'step-{step}/lora_adapter/adapter_model.safetensors')
    require(public['nonzero_effective_delta'],'exported empty effective update')
    commit(out/f'rank-{rank}/RESULT.json',dict(rank=rank,step=step,actions=heard,objective=result['objective_observation'],
            metrics=result['metrics'],publication_sha256=public['identity_sha256'],adapter_export=True,
            recoverable_checkpoint=recoverable,**({'arithmetic_scope':'resident'} if continuous else
                {'arithmetic':worker.close_actor_arithmetic()})))
    if continuous:return committed
