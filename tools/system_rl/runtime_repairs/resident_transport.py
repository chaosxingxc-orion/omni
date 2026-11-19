"""Existing sealed transport, same leased owner, ordered immutable commands."""
import copy
import os
import time
from pathlib import Path
from dependencies import require,sha
from run_package import read,pin,write


def project_inputs(spec,mappings):
    """Publish exact bytes at future logical paths already mounted RO by actor."""
    from storage import check_running
    attached=spec.get('_resident_attach')
    inventory=read(Path(attached['target'])/'RESIDENT-INPUTS.json') if attached else {}
    initial={}
    for item in mappings:
        target=Path(item['target']);matches=[p for p in spec['resident']['input_projections'] if target.is_relative_to(p['target'])]
        if not matches:
            actual=sha(item['source'])
            require(not attached or inventory.get(str(target))==actual,'resident future input has no configured RO projection: '+str(target))
            initial[str(target)]=actual
            continue # initial immutable file mounts already exist at construction
        projection=max(matches,key=lambda p:len(p['target']))
        destination=Path(projection['source'])/target.relative_to(projection['target'])
        source=Path(item['source'])
        if not destination.exists():check_running(Path(projection['source']),source.stat().st_size)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():require(sha(source)==sha(destination),'resident projected input collision');continue
        pending=destination.with_suffix(destination.suffix+'.incoming')
        # Failed partial is retained; a new task cannot replace these bytes.
        require(not pending.exists(),'partial input projection requires original reconciliation')
        with source.open('rb') as stream,pending.open('xb') as output:
            import shutil
            shutil.copyfileobj(stream,output,1024*1024);output.flush();os.fsync(output.fileno())
        require(sha(pending)==sha(source),'projection byte drift')
        os.rename(pending,destination)
        for folder in [destination.parent,*destination.parent.parents]:
            fd=os.open(folder,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
            if folder==Path(projection['source']):break
    if not attached:write(Path(spec['output_root']).parent/'RESIDENT-INPUTS.json',initial)


def attach(job,spec,run_ref):
    """CPU controller submits once; cannot allocate another actor/lease."""
    owner=spec['_resident_attach'];runtime_ref=read(run_ref)['runtime_ref'];runtime=read(runtime_ref)
    exchange=Path(owner['target'])/'control/package/hot/resident'
    ready=read(exchange/'READY.json')
    require(ready['run_id']==spec['resident']['run_id'] and ready['source_boot']==job['source_boot'],'resident owner/boot differs')
    seq=owner['sequence'];require(seq>0,'attach sequence0 reserved to initial actor')
    require(seq<spec['resident']['max_commands']-1,'resident command bound reserves final end slot')
    require(time.time()<runtime['deadline_epoch']==ready['deadline_epoch'],
        'resident compute deadline expired or differs; preserve original owner')
    command=dict(action=spec['action'],runtime_ref=runtime_ref,run_id=ready['run_id'])
    write(exchange/'commands'/f'{seq:06d}.json',command)
    result=exchange/'results'/f'{seq:06d}'
    while not (result/'CLOSED.json').exists():
        require(time.time()<runtime['deadline_epoch'],'resident result unknown at deadline; preserve original command')
        time.sleep(min(spec['resident']['poll_seconds'],max(.001,runtime['deadline_epoch']-time.time())))
    from storage import check_running
    from pull_closed import collect
    # Collect only the closed command graph, never active owner state.
    collect(result,job['closed_return_max_bytes'],full_checkpoint=runtime.get('checkpoint_storage_mode')=='lora_training_state' or runtime.get('newprocess_replay',{}).get('enabled',False))
    payload=read(result/'RETURN.json')
    if spec['action']=='diagnostic':payload['runtime_ref']=runtime_ref
    return payload


def training(config,spec,directory):
    from transport import execute,prepare
    require(config.get('source_bundle_ref'),'resident commands require one frozen MAIN-merged source bundle')
    require(time.time()<spec['budget']['deadline_epoch']-spec['budget']['restore_seconds'],
        'resident compute deadline expired before command allocation')
    root=Path(config['output_root'])/'resident-client';directory=Path(directory)
    spec=copy.deepcopy(spec);resident=copy.deepcopy(config['resident'])
    resident['initial_action']=resident.get('initial_action','diagnostic')
    require(type(resident['max_commands']) is int and resident['max_commands']>=4,
        'resident needs initial diagnostic, finalize, first acceptance and end slots')
    spec['resident']=resident
    # Include the original cursor's rollout/CPU and unknown reservations, not
    # only the commands executed by the actor. Conservative overlap is charged.
    spec['budget']['prior_logical_tokens_upper']=max(spec['budget'].get('prior_logical_tokens_upper',0),
        config.get('_charged_tokens_upper',config.get('budget',{}).get('prior_tokens_upper',0)))
    # Same startup projection layout for all commands; generated command runtimes
    # and checkpoints under this local target parent are visible without remount.
    spec['input_mounts']=sorted(set(spec.get('input_mounts',[])+resident['future_readonly_roots']))
    call=reserve_call(root,directory,resident,lambda:prepare(config,spec,directory,'npu2','training'))
    if call['sequence']>0:spec['_resident_attach']=call
    else:
        require(spec['action']==resident['initial_action'],'first command must match configured initial action')
    return execute(config,spec,directory,'npu2','training')


def reserve_call(root,directory,resident,prepare):
    """Durable index FIRST, then per-call receipt; repair either crash gap.

    OWNER is the sequence0 index. A reserved index with no call receipt belongs
    to that exact directory forever, so another directory cannot reuse it.
    """
    import fcntl
    root=Path(root);directory=Path(directory);root.mkdir(parents=True,exist_ok=True)
    marker=root/'OWNER.json';intent=directory/'RESIDENT-CALL.json'
    with (root/'allocation.guard').open('ab') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not marker.exists():
            require(not intent.exists(),'orphan resident call requires original owner reconciliation')
            plan=prepare()
            write(marker,dict(directory=str(directory),target=plan['target'],resident=resident))
        owner=read(marker);require(owner['resident']==resident,'resident owner config drift')
        if owner['directory']==str(directory):call=dict(sequence=0,target=owner['target'])
        else:
            indexed=[read(p) for p in sorted((root/'calls').glob('*.json'))] if (root/'calls').exists() else []
            matches=[v['call'] for v in indexed if v['directory']==str(directory)]
            require(len(matches)<=1,'resident directory has duplicate sequence indexes')
            if matches:call=matches[0]
            elif intent.exists():call=read(intent) # repair legacy receipt-before-index gap
            else:
                seq=1
                while (root/'calls'/f'{seq:06d}.json').exists():seq+=1
                call=dict(sequence=seq,target=owner['target'])
            require(call['sequence']>0 and call['target']==owner['target'],'resident call owner drift')
            require(call['sequence']<resident['max_commands']-1,'resident command bound reserves final end slot')
            write(root/'calls'/f"{call['sequence']:06d}.json",dict(directory=str(directory),call=call))
        write(intent,call)
        return call


def end(config):
    """Explicit end under existing transport; only close the current owner."""
    from transport import client,pull_stream,unpack_closed,closed_reader_source
    root=Path(config['output_root'])/'resident-client'
    if not (root/'OWNER.json').exists():return
    owner=read(root/'OWNER.json');seq=1
    while (root/'calls'/f'{seq:06d}.json').exists():seq+=1
    marker=root/'END-CALL.json'
    if marker.exists():seq=read(marker)['sequence']
    else:write(marker,dict(sequence=seq,target=owner['target'],automatic_retry=False))
    directory=Path(owner['directory']);c=client()
    code="""import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from run_package import write
root=Path(sys.argv[2]);seq=int(sys.argv[3]);run_id=sys.argv[4]
write(root/'commands'/f'{seq:06d}.json',dict(action='end',run_id=run_id))
"""
    # Read-only source plus a single immutable local command, never Ceph writes.
    stage=read(directory/'TRANSPORT.json')['stage']
    import fcntl
    with (root/'end.guard').open('ab') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (root/'END-INTENT.json').exists():
            # c.step records intent before the exact idempotent command write.
            # If this transport is unknown, retain it for read-only reconciliation.
            write(root/'REQUEST.json',read(directory/'REQUEST.json'))
            c.step(root,'END',['python3','-B','-c',code,str(Path(stage)/'parents/jev-system-rl/implementation'),
                str(Path(owner['target'])/'control/package/hot/resident'),str(seq),owner['resident']['run_id']],lock.fileno())
        if not (root/'REQUEST.json').exists():write(root/'REQUEST.json',read(directory/'REQUEST.json'))
        # The original controller closes RETURN only after independent watchdog
        # restored both services. END acknowledgement alone is not completion.
        while time.time()<config['deadline_epoch']:
            code=closed_reader_source()
            try:
                result=pull_stream(root,['python3','-B','-c',code,owner['target'],
                    str(config['transports']['npu2']['closed_return_max_bytes'])]+(['--full-checkpoint'] if config.get('checkpoint_storage_mode')=='lora_training_state' or config.get('newprocess_replay',{}).get('enabled') else []),lock.fileno(),config['deadline_epoch'])
            except (OSError,__import__('subprocess').TimeoutExpired):result=None
            if result and result.returncode==0:
                unpack_closed(root/'closed-return.tar',root/'returned')
                plan=copy.deepcopy(read(directory/'TRANSPORT.json'))
                # End archive covers the fully exited owner, including the last
                # complete checkpoint, rather than initial command0 projection.
                plan['job']['task']['resident']['enabled']=False
                plan['task_identity']+='-end'
                if config.get('archive',{}).get('enabled'):
                    from checkpoint_archive import trigger
                    trigger(config,root,plan)
                write(root/'END-COMPLETE.json',dict(return_ref=pin(root/'returned/RETURN.json'),restored=True))
                return
            require(not result or result.returncode in (75,255),'resident end failed; preserve original owner')
            time.sleep(min(900,max(0,config['deadline_epoch']-time.time())))
        raise RuntimeError('resident end unknown at deadline; independent watchdog remains authoritative')
