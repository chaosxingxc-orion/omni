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
