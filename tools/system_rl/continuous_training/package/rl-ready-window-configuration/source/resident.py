"""Bounded command window for the SAME constructed FSDP worker and verl actor.

No backward/optimizer implementation lives here. All updates use entry.execute_window.
Only closed command results are visible to transport while the owner is live.
"""
import copy
import json
import time
from pathlib import Path
from dependencies import require
from run_package import read, pin, write
from storage import check_running


def collectively(function):
    """Agree on rank-local admission/IO failures before any actor collective."""
    import torch.distributed as dist
    value=None;failure=None
    try:value=function()
    except Exception as error:failure=f'{type(error).__name__}: {error}'
    statuses=[None]*dist.get_world_size()
    dist.all_gather_object(statuses,dict(rank=dist.get_rank(),error=failure))
    require(all(s['error'] is None for s in statuses),'resident collective preflight failed: '+str(statuses))
    return value


def agree(value):
    """Identical immutable command/config identity on every actual rank."""
    import torch.distributed as dist
    from policy_identity import digest
    identities=[None]*dist.get_world_size()
    dist.all_gather_object(identities,digest(value))
    require(len(set(identities))==1,'resident rank command/config identity drift')


def receive_command(exchange, sequence, config, spec, hot, last, rank):
    """Poll on rank zero; all ranks rendezvous after each bounded poll.

    Never leave nonzero ranks inside an NPU collective while rank zero waits
    for an external CPU controller. That wait can legitimately exceed HCCL's
    operation timeout even though no distributed model operation is running.
    """
    import torch.distributed as dist
    while True:
        def poll():
            if rank != 0:
                return None
            if sequence == 0:
                return dict(action=spec['initial_action'], runtime_ref=pin(config['_path']))
            path=Path(exchange)/'commands'/f'{sequence:06d}.json'
            require(time.time()<config['deadline_epoch'], 'resident absolute compute deadline; independent watchdog restores')
            require(time.monotonic()-last<spec['max_idle_seconds'], 'resident idle timeout; preserve last full checkpoint')
            check_running(hot,config['storage_budget_bytes'])
            return read(path) if path.exists() else None
        command=collectively(poll)
        shared=[command]
        dist.broadcast_object_list(shared,src=0)
        if shared[0] is not None:
            return shared[0]
        # No collective is outstanding during this host-side wait.
        time.sleep(min(spec['poll_seconds'],max(.001,config['deadline_epoch']-time.time())))


def validate(config):
    spec=config['resident']
    require(spec['enabled'] is True and type(spec['max_commands']) is int and spec['max_commands']>0, 'bounded resident window required')
    require(type(spec['checkpoint_interval']) is int and spec['checkpoint_interval']>0, 'checkpoint interval')
    require(spec['checkpoint_on_phase'] is True and spec['checkpoint_on_end'] is True, 'phase/end checkpoints required')
    require(spec['max_idle_seconds']>0 and spec['poll_seconds']>0, 'bounded command wait')
    from storage import validate_checkpoint_budget
    validate_checkpoint_budget(config['storage_budget_bytes'],config.get('checkpoint_storage_budget_ref'),require_scope=True)
    require(config['max_policy_staleness']==0, 'resident staleness0')
    return spec


class Session:
    def __init__(self,worker,config,hot,rank):
        self.worker=worker;self.config=config;self.hot=Path(hot);self.rank=rank
        self.spec=validate(config);self.clock=None;self.contract=None
        self.objects=(id(worker),id(worker.actor),id(worker.actor.actor_module),id(worker.actor.actor_optimizer))
        self.last_checkpoint=config.get('resume_ref');self.last_saved_step=None;self.first_update=True
        self.prior_clock=None;self.resumed=False
        self.last_saved_clock=None
        self.last_saved_accounting=None
        if config.get('resident_resume_state_ref'):
            previous=read(config['resident_resume_state_ref'])
            require(config['prior_logical_tokens_upper']>=previous['charged_tokens_upper'],'resume cannot discard historical/unknown cost')
            self.prior_clock=previous['clock']
            require(self.last_checkpoint is not None or self.prior_clock is None or self.prior_clock['successful_optimizer_steps']==0,
                'metadata-only Clock cannot recover successful Adam steps')
        if self.last_checkpoint:
            from checkpoint import restore
            from entry import scientific_contract
            accounting=read(self.last_checkpoint).get('resident_accounting')
            if accounting:
                require(config['prior_logical_tokens_upper']>=accounting['charged_tokens_upper'],
                    'full restore cannot reset resident logical cost watermark')
            contract=scientific_contract(config,read(config['learning_ref']))
            receipt,_=restore(worker,Path(pin(self.last_checkpoint)['path']).parent,contract)
            self.contract=contract;self.resumed=True
            if self.prior_clock is not None:
                require(self.prior_clock==receipt['clock'],'metadata Clock differs from full authoritative recovery')
            self.prior_clock=receipt['clock']
            self.last_saved_step=receipt['step']
            self.last_saved_clock=copy.deepcopy(receipt['clock'])
            self.last_saved_accounting=copy.deepcopy(receipt.get('resident_accounting'))

    def intact(self):
        require(self.objects==(id(self.worker),id(self.worker.actor),id(self.worker.actor.actor_module),id(self.worker.actor.actor_optimizer)),
                'resident worker/actor/model/optimizer object replaced')

    def bind_clock(self, incoming, contract):
        self.intact()
        require(self.contract is None or self.contract==contract, 'resident scientific contract changed')
        old=self.clock.state if self.clock is not None else self.prior_clock
        if old is not None:
            state=incoming.state
            require(state['successful_optimizer_steps']==old['successful_optimizer_steps'], 'resident Clock step drift')
            require(state['attempts']==old['attempts'], 'resident attempts lost/added')
            require(all(state['validations'].get(k)==v for k,v in old['validations'].items()),'resident validation history lost/changed')
            require(all(state['groups'].get(k)==v for k,v in old['groups'].items()), 'resident group history lost')
            extra=set(state['groups'])-set(old['groups'])
            require(all(not state['groups'][g]['consumed'] for g in extra), 'new group already consumed')
        else:require(incoming.state['successful_optimizer_steps']==0,'fresh resident cannot invent successful steps')
        if self.clock is None:self.clock=incoming
        else:
            # Same Clock object; authenticated original factory validates explicit
            # policy migration/phase validations and reserves this complete group.
            self.clock.config=incoming.config
            self.clock.state=copy.deepcopy(incoming.state)
        self.contract=contract
        return self.clock

    def intent(self,window,config):
        self.intact()
        require(window['physical_requests_drained'] is True and window['unknown_requests']==0, 'publish/update needs drain')
        require(window['attempt_id'] not in self.clock.state['attempts'], 'duplicate optimizer attempt')
        require(all(not self.clock.state['groups'][g]['consumed'] for g in window['group_ids']), 'fresh group reused')

    def checkpoint(self,config,*,reload=False):
        from checkpoint import save,restore,metadata_commit
        self.intact();step=self.clock.state['successful_optimizer_steps']
        accounting=getattr(self.worker,'resident_accounting',None)
        if self.last_saved_step==step and self.last_saved_clock==self.clock.state and self.last_saved_accounting==accounting and not reload:return self.last_checkpoint
        collectively(lambda:check_running(self.hot,config['storage_budget_bytes']))
        dest=self.hot/f'checkpoint-{step:06d}'
        if self.last_saved_step==step:
            from policy_identity import digest
            dest=self.hot/f'checkpoint-{step:06d}-clock-{digest(dict(clock=self.clock.state,accounting=accounting))[:16]}'
        # A preexisting incomplete save is evidence, never overwritten/replayed.
        import torch.distributed as dist
        admission=[not dest.exists() if self.rank==0 else None]
        dist.broadcast_object_list(admission,src=0)
        require(admission[0], 'checkpoint intent already exists; preserve authority')
        if self.last_saved_step==step:
            metadata_commit(self.worker,dest,self.clock,self.contract,self.last_checkpoint)
        else:save(self.worker,dest,self.clock,self.contract,step)
        if reload:
            receipt,_=restore(self.worker,dest,self.contract)
            require(receipt['clock']==self.clock.state, 'save/reload Clock mismatch')
            self.worker.audit_master_optimizer(step)
        self.last_checkpoint=pin(dest/'COMMITTED.json');self.last_saved_step=step
        self.last_saved_clock=copy.deepcopy(self.clock.state)
        self.last_saved_accounting=copy.deepcopy(accounting)
        return self.last_checkpoint

    def updated(self,config,hot,clock,contract,result,audit,window):
        from checkpoint import policy_commit
        step=clock.state['successful_optimizer_steps'];learning=read(config['learning_ref'])
        before=[p for p in learning['phases'] if p['start_step']<=step-1][-1]['id']
        after=[p for p in learning['phases'] if p['start_step']<=step][-1]['id']
        boundary=before!=after
        from lora_training_state import enabled
        full=self.first_update or step%self.spec['checkpoint_interval']==0 or boundary or step>=config['max_optimizer_updates']
        reload=self.first_update and not config.get('newprocess_replay',{}).get('enabled') # replay A stays continuous
        if full:ref=self.checkpoint(config,reload=reload)
        else:ref=policy_commit(self.worker,Path(hot)/'policy',clock,contract,self.last_checkpoint)
        self.first_update=False
        from publication import seal
        policy=collectively(lambda:seal(ref,Path(hot)/'candidate') if self.rank==0 else None)
        import torch.distributed as dist
        shared=[policy];dist.broadcast_object_list(shared,src=0)
        payload=dict(result=result,optimizer=audit,checkpoint_ref=ref,policy_ref=shared[0],optimizer_steps=1,
            checkpoint=str(Path(ref['path']).parent),clock=copy.deepcopy(clock.state),full_checkpoint=full,
            save_reload_verified=reload,latest_recovery_ref=self.last_checkpoint,resident_objects=list(self.objects),
            formal_rl_enabled=config['formal_rl_enabled'],rollout_publication_verified=False)
        if config.get('newprocess_replay',{}).get('enabled'):
            from newprocess_replay import finish
            local_replay=collectively(lambda:finish(self.worker,config,ref,result,self.rank))
            replay_refs=[None]*dist.get_world_size();dist.all_gather_object(replay_refs,local_replay)
            payload['replay_refs']=replay_refs
        write(Path(hot)/f'rank-{self.rank}/RESULT.json',payload)
        return payload


def seal_result(root,payload,config,sequence):
    write(root/'RETURN.json',payload)
    write(root/'CLOSED.json',dict(return_ref=pin(root/'RETURN.json'),immutable=True,kind='resident-command',
        sequence=sequence,source_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        run_id=config['resident']['run_id'],source_commit='source-pins:'+__import__('dependencies').sha(Path(__file__).parent/'SOURCE-PINS.json'),
        watermark=(payload.get('clock') or {}).get('successful_optimizer_steps'),active_directory_snapshot=False))


def serve(worker,adapted,config,hot,task,rank,world,execute):
    import torch.distributed as dist
    agree(config)
    session=Session(worker,config,hot,rank);exchange=Path(hot)/'resident'
    def ready():
        if rank==0:
            require(not (exchange/'READY.json').exists(), 'resident process restart cannot reuse unknown intents')
            write(exchange/'READY.json',dict(run_id=session.spec['run_id'],source_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                deadline_epoch=config['deadline_epoch'],objects=list(session.objects),worker_constructed_once=True,diagnostic_retention_guard="resident-command-guard-v1"))
    collectively(ready)
    dist.barrier();last=time.monotonic()
    for sequence in range(session.spec['max_commands']):
        root=exchange/'results'/f'{sequence:06d}'
        command=receive_command(exchange,sequence,config,session.spec,hot,last,rank)
        def admit():
            require(command.get('run_id',session.spec['run_id'])==session.spec['run_id'], 'resident command owner drift')
            session.intact()
            require(time.time()<config['deadline_epoch'], 'command after compute deadline')
        collectively(admit);action=command['action']
        if action=='end':
            # A completely zero-signal window has no Adam state to serialize;
            # close the terminal Clock separately, retain original recovery.
            has_update=session.clock is not None and session.clock.state['successful_optimizer_steps']>0
            ref=session.checkpoint(config) if has_update else session.last_checkpoint
            def close_end():
                if rank==0:
                    terminal=dict(checkpoint_ref=ref,optimizer_steps=session.clock.state['successful_optimizer_steps'] if session.clock else 0,
                        clock=session.clock.state if session.clock else session.prior_clock,
                        charged_tokens_upper=getattr(session,'charged',config['prior_logical_tokens_upper']))
                    seal_result(root,dict(terminal,ended=True,optimizer_steps=0),config,sequence)
                    write(exchange/'END.json',dict(terminal,command_closed_ref=pin(root/'CLOSED.json')))
            collectively(close_end)
            dist.barrier();return
        def read_config():
            current=read(command['runtime_ref']);current['_path']=pin(command['runtime_ref'])['path']
            from entry import scientific_contract
            incoming_contract=scientific_contract(current,read(current['learning_ref']))
            if session.contract is not None:require(incoming_contract==session.contract,'resident operation changed scientific contract')
            else:session.contract=incoming_contract
            require(action in ('acceptance','finalize','diagnostic'), 'unknown resident operation')
            require(current['resident']==config['resident'], 'resident configuration changed')
            for key in ('model','training','lr','seed','storage_budget_bytes','deadline_epoch','max_optimizer_updates','logical_token_limit'):
                require(current[key]==config[key], 'resident immutable setting drift: '+key)
            return current
        current=collectively(read_config)
        agree(current)
        # Every command reserves its conservative cost before any forward;
        # unknown reservations are never refunded or replayed.
        prior=max(config['prior_logical_tokens_upper'],getattr(session,'charged',0),current['prior_logical_tokens_upper'])
        additional=current['diagnostic_token_upper'] if action=='diagnostic' else current['update_token_upper']
        require(prior+additional<=config['logical_token_limit'], 'resident cumulative token budget')
        current['prior_logical_tokens_upper']=prior;session.charged=prior+additional
        worker.resident_accounting=dict(charged_tokens_upper=session.charged,command_sequence=sequence,
            run_id=session.spec['run_id'],unknown_reservations_retained=True)
        intent=root/f'rank-{rank}/COMMAND-INTENT.json'
        def reserve():
            require(not intent.exists(), 'unknown resident command; reconcile never replay')
            write(intent,dict(command=command,prior_upper=prior,reserved_upper=additional,total_upper=session.charged,status='unknown_until_closed'))
        collectively(reserve)
        if action=='finalize':
            from policy_identity import resident,behavior_binding
            identity=resident(worker)
            collectively(lambda:behavior_binding(identity,read(current['window_ref'])))
        # Rank 0 owns the same host-visible lock used by diagnostic retirement.
        # All ranks agree on admission before any model collective begins.
        import fcntl
        lease=None
        def acquire_command_guard():
            if rank!=0:return
            nonlocal lease
            lease=(exchange/'diagnostic-retention.guard').open('ab')
            fcntl.flock(lease,fcntl.LOCK_EX)
        collectively(acquire_command_guard)
        try:
            payload=execute(worker,adapted,current,root,task,rank,world,diagnostic=action=='diagnostic',finalize=action=='finalize',session=session)
            if payload is None:
                name='DIAGNOSTIC-RESULT.json' if action=='diagnostic' else 'FINALIZE-RESULT.json'
                payload=collectively(lambda:read(root/f'rank-{rank}'/name))
                if action=='diagnostic':
                    payload.update(raw_ref=collectively(lambda:pin(root/'rank-0/DIAGNOSTIC-RAW.json')),runtime_ref=command['runtime_ref'])
            payload=dict(payload,optimizer_steps=payload.get('optimizer_steps',0),charged_tokens_upper=session.charged,
                command_intent_ref=collectively(lambda:pin(root/'rank-0/COMMAND-INTENT.json')))
            dist.barrier()
            collectively(lambda:seal_result(root,payload,config,sequence) if rank==0 else None)
        finally:
            if lease is not None:lease.close()
        dist.barrier();last=time.monotonic()
    raise ValueError('resident max_commands reached without end; preserve state and restore via watchdog')


def formal_window(initial, command, last_ref, prior, zero_refs):
    """Only data and its predecessor can vary while this actor is resident."""
    from bounded_actor_step import verify_continuation
    require(set(command)=={'action','run_id','sequence','previous_close_sha256','checkpoint_sha256',
                           'generation_inventory_ref','pending_capture_ref'} and
            command['action']=='update' and command['checkpoint_sha256']==last_ref['sha256'],
            'unexpected resident training command or stale predecessor')
    inventory=read(command['generation_inventory_ref'])
    require(inventory['closed'] is True,'incomplete business generation')
    # This reference points to the native scorer intent, not to a replacement
    # worker config. Keep the installed code, profile, sampling and resource policy.
    pending=read(command['pending_capture_ref'])
    capture=read(pending['pending_capture_ref'])
    require(Path(capture['cohort_root'])==Path(inventory['cohort_root'])
            and inventory['files']['NATIVE.pending.disabled.json']==command['pending_capture_ref']
            and inventory['files']['PENDING-CAPTURE.json']==pending['pending_capture_ref']
            and inventory['files']['INPUT.json']==capture['input_ref'],'mixed generation and scorer intent')
    gen=dict(initial['generation'],generation_inventory_ref=command['generation_inventory_ref'],
             pending_capture_ref=command['pending_capture_ref'])
    sources=prior['contract']['sources']
    c=dict(initial['continuation'],prior_step=prior['step'],zero_result_refs=zero_refs,
           predecessor_driver_sha256=sources['bounded_actor_step.py'],
           predecessor_control_sha256=sources['retained_hf_fsdp_control.py'])
    spec=dict(initial,generation=gen,continuation=c,restore_checkpoint=dict(
        directory=str(Path(last_ref['path']).parent),sha256=last_ref['sha256']))
    if command['sequence']>0:spec.pop('batch_migration',None)
    verify_continuation(spec,prior)
    return spec


def serve_formal(worker,adapted,initial,hot,rank,world,sealed,execute):
    """Formal FSDP8 callback on the existing resident command rendezvous."""
    import os
    import torch.distributed as dist
    from bounded_actor_step import restore_continuation
    from formal_actor_receipt import close_live
    cfg=initial['continuous_actor'];hot=Path(hot);exchange=hot/'resident'
    require(cfg['protocol']=='formal-resident-v1' and world==8 and initial.get('formal_sampled') is True
            and initial.get('generation',{}).get('runtime',{}).get('reference_batch')
            and cfg['poll_seconds']>0 and cfg['max_idle_seconds']>0 and cfg['max_commands']>=2
            and cfg['deadline_epoch']>time.time() and initial['training_memory_gib']>=1024,
            'invalid resident formal training owner')
    agree(initial)
    def identity():
        return [os.getpid(),id(worker),id(worker.actor),id(worker.actor.actor_module),
                id(worker.actor.actor_optimizer),id(worker.actor_lr_scheduler),id(adapted)]
    objects=identity()
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(boot==initial['boot_id'],'formal worker boot differs')
    peers=[None]*world;dist.all_gather_object(peers,objects)
    def open_owner():
        if rank:return
        require(not exchange.exists(),'resident exchange already exists; reconcile original owner')
        exchange.mkdir()
        write(exchange/'OWNER.json',dict(protocol=cfg['protocol'],run_id=cfg['run_id'],boot_id=boot,
              objects=peers,initial_checkpoint=initial['restore_checkpoint']['sha256']))
    collectively(open_owner)
    # Restore weights, master parameters, Adam and scheduler exactly once.
    last_ref,prior=restore_continuation(worker,initial)
    collectively(lambda:write(exchange/'READY.json',dict(owner_ref=pin(exchange/'OWNER.json'),
        run_id=cfg['run_id'],source_boot=boot,checkpoint_sha256=last_ref['sha256'])) if rank==0 else None)
    last_closed=pin(exchange/'READY.json')
    last=time.monotonic();seen=set()
    zero_refs=list(initial['continuation'].get('zero_result_refs',[]))
    for seq in range(cfg['max_commands']):
        if seq:
            command=receive_command(exchange,seq,dict(deadline_epoch=cfg['deadline_epoch'],
                storage_budget_bytes=initial['storage_budget_bytes']),cfg,hot,last,rank)
        else:
            command=dict(action='update',run_id=cfg['run_id'],sequence=0,previous_close_sha256=last_closed['sha256'],
                checkpoint_sha256=last_ref['sha256'],**{k:initial['generation'][k] for k in
                        ('generation_inventory_ref','pending_capture_ref')})
        root=exchange/'results'/f'{seq:06d}'
        def admit():
            require(command['run_id']==cfg['run_id'] and type(command['sequence']) is int and command['sequence']==seq
                    and command['previous_close_sha256']==last_closed['sha256'] and objects==identity()
                    and cfg['deadline_epoch']>time.time(),'resident command chain, owner or live objects changed')
            check_running(hot,initial['storage_budget_bytes'])
            require(not root.exists(),'unknown resident intent; reconcile before continuing')
        collectively(admit)
        if command['action']=='end':
            require(set(command)=={'run_id','sequence','action','previous_close_sha256'},'unexpected end command')
            collectively(lambda:write(exchange/'END.json',dict(previous=last_closed,checkpoint_ref=last_ref,
                run_id=cfg['run_id'],sequence=seq)) if rank==0 else None)
            return
        require(seq<cfg['max_commands']-1,'reserve final resident command for end')
        current=collectively(lambda:formal_window(initial,command,last_ref,prior,zero_refs))
        gen=current['generation']['generation_inventory_ref']
        require(gen['sha256'] not in seen,'duplicate resident generation')
        seen.add(gen['sha256']);agree(current)
        collectively(lambda:write(root/'COMMAND-INTENT.json',dict(owner_ref=pin(exchange/'OWNER.json'),
            command=command,predecessor=last_ref)) if rank==0 else None)
        result_ref=execute(worker,adapted,current,root,rank,world,sealed,live=(last_ref,prior))
        agree(result_ref)
        def bind_rank():
            require(objects==identity(),'resident optimizer, model or scheduler was replaced during update')
            return write(root/f'rank-{rank}/LIVE.json',dict(rank=rank,objects=objects,
                intent_ref=pin(root/'COMMAND-INTENT.json'),result_ref=pin(root/f'rank-{rank}/RESULT.json')))
        receipt=collectively(bind_rank)
        ranks=[None]*world;dist.all_gather_object(ranks,receipt)
        collectively(lambda:close_live(current,root,last_ref,result_ref,ranks,peers) if rank==0 else None)
        last_closed=pin(root/'CLOSED.json')
        updated=last_ref['sha256']!=result_ref['sha256']
        if updated:
            last_ref=result_ref;prior=read(result_ref);zero_refs=[]
        else:
            zero_refs=[*zero_refs,read(last_closed)['result_ref']]
        last=time.monotonic()
    raise ValueError('formal command limit exhausted')
