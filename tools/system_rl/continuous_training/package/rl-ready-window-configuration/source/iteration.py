"""Configuration-driven fixed stage graph around the one DP8 verl entry.

CPU orchestration only. Existing mailbox owner executes 175 endpoint/eval/capture;
181 uses the DP8 fixed lifecycle. Persistent intents never retry unknown work.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from dependencies import HERE,ROOT,require
from run_package import read,pin,write,compile_package


class Pending(RuntimeError):pass


def once(root,name,argv,result):
    root=Path(root);done=root/(name+'-COMPLETE.json');intent=root/(name+'-INTENT.json')
    if done.exists():return read(done)['result_ref']
    if intent.exists():
        if not Path(result).is_file():raise Pending('unknown '+name+': reconcile original intent, never repeat')
    else:
        write(intent,dict(argv=argv,automatic_retry=False))
        process=subprocess.run(argv,check=False)
        write(root/(name+'-EXIT.json'),dict(returncode=process.returncode))
        require(process.returncode==0,'fixed stage failed: '+name)
    require(Path(result).is_file(),'fixed stage has no terminal output: '+name)
    ref=pin(result);write(done,dict(result_ref=ref));return ref


def original_controller():
    from dependencies import load
    saved=sys.modules.pop('io_utils',None);paths=sys.path[:]
    try:
        sys.path.insert(0,str(ROOT/'policy-iteration-successor'))
        controller=load(ROOT/'policy-iteration-successor/controller.py','dp8_iteration_original_controller')
        # Keep original restoration conditions; use the already authenticated
        # logical-to-physical reader for returned receipts.
        def closed_doc(ref):
            require(pin(ref)=={k:ref[k] for k in ('path','sha256','bytes')},'closed receipt identity drift')
            return read(ref)
        controller.doc=closed_doc
        return controller
    finally:
        sys.path[:]=paths;sys.modules.pop('io_utils',None)
        if saved is not None:sys.modules['io_utils']=saved


def stage_budget(config,stage,extra=None):
    purpose=(extra or {}).get('purpose')
    if stage=='capture' and purpose in ('base_parity','updated_parity'):
        if config.get('dataset_profile',{}).get('acquisition'):
            from boundary_acquisition import capture_budget
            budget=capture_budget(config,purpose)
        else:budget=config['diagnostic_capture_budgets'][purpose]
    else:budget=config['stage_budgets'][stage]
    budget=copy.deepcopy(budget)
    if stage in ('capture','dev','test','screen'):
        # The existing owner launches a fresh service and proves that physical
        # endpoint inside the SAME stage/ledger before business requests.
        # Reserving only the business calls rejects the last valid request.
        endpoint=config['stage_budgets']['endpoint']
        require(endpoint['tokens_upper']>0,'active endpoint probe token envelope required')
        budget['tokens_upper']+=endpoint['tokens_upper']
        budget['requests_upper']=budget.get('requests_upper',0)+1
    return budget


def retain_parity_profile(config,state):
    from boundary_acquisition import retain_profile
    # The bounded replay has two explicit policy checks on the same finite
    # source set. Each still captures new actions via the original journal.
    acquisition=copy.deepcopy(config['dataset_profile'].get('acquisition')) if config.get('bounded_restore_compare') else None
    retain_profile(config,state)
    if acquisition:
        config['dataset_profile']['acquisition']=acquisition
        state['accepted_dataset_profile']=copy.deepcopy(config['dataset_profile'])


def rollout_output_root(config,sequence,stage):
    """Pin target-local output before request hashing; never rewrite a live ref."""
    require(stage in ('endpoint','capture','dev','test','screen'),'rollout stage required')
    from program_placement import client_owned, output
    if client_owned(config,stage):
        return output(config,sequence,stage)
    if config.get('formal_rollout') and stage in ('endpoint','capture'):
        return str(Path(config['output_root'])/'rollout'/f'{sequence:04d}-{stage}')
    transport=config['transports']['npu']
    anchor=Path(transport['storage_anchor']);parent=Path(transport['run_parent'])
    run_id=config['run_id']
    require(anchor.is_absolute() and parent.is_absolute() and '..' not in parent.parts and
            parent.is_relative_to(anchor),'rollout storage parent escapes configured anchor')
    require(isinstance(run_id,str) and run_id not in ('','.','..') and Path(run_id).name==run_id,'single run namespace required')
    require(type(sequence) is int and sequence>0,'positive rollout sequence required')
    return str(parent/run_id/'stage-work'/f'{sequence:04d}-{stage}')


def request(config,root,sequence,stage,input_ref,policy_ref,witness_ref=None,extra=None):
    """Same original mailbox request/owner_bridge contract, configured once."""
    directory=Path(root)/'mailbox'/f'{sequence:04d}-{stage}'
    marker=directory/'REQUEST.json'
    # A previous process may already have committed RESPONSE. Restore the
    # authenticated location map before any transitive receipt is consumed.
    if (directory/'transport/returned/CLOSED-COMMIT.json').is_file():
        from transport import restore_locations
        restore_locations(directory/'transport/returned')
    if not marker.exists():
        from rollout_owner import install
        owner=read(read(config['resource_owner_config_ref'])['config_ref']);install(owner)
        before=original_controller().usage(config)
        budget=stage_budget(config,stage,extra)
        require(time.time()<config['deadline_epoch'] and before['unknown']==0,'budget expired or unresolved request')
        require(before['cost_units']+budget['tokens_upper']<=config['budget']['total_tokens'],'cumulative request budget')
        value=dict(schema='policy-iteration-request-v1',config_ref=pin(config['_path']),sequence=sequence,stage=stage,
            fixture=False,input_ref=input_ref,policy_ref=policy_ref,endpoint_witness_ref=witness_ref,
            dataset_ref=config['datasets'][{'dev':'DEV','test':'TEST'}.get(stage,'TRAIN')],
            system_policy_ref=config['system_policy_ref'],authorization_ref=config['authorization_ref'],
            resource_owner_config_ref=config['resource_owner_config_ref'],registry=config['registry'],budget=budget,
            usage_before=before,output_root=rollout_output_root(config,sequence,stage),extra=extra or {},
            constraints=dict(model='qwen3-omni-30b-a3b-instruct',thinking=False,unknown='reconcile_never_resend',progress_check_seconds=900))
        write(marker,value)
    value=read(marker)
    require(value['policy_ref']==policy_ref and value['input_ref']==input_ref and value['endpoint_witness_ref']==witness_ref,'resume stage inputs drift')
    response=directory/'RESPONSE.json'
    if not response.exists():
        # Fixed transport executes same request on175; the owner there owns
        # actual service/watchdog. It is never an unspecified user callback.
        from transport import rollout
        rollout(config,pin(marker),directory)
    if not response.exists():raise Pending('await rollout owner '+str(marker))
    answer=read(response)
    require(answer['request_ref']==pin(marker) and answer['status']=='complete' and answer['fixture'] is False,'owner response not complete')
    if stage in ('dev','test','screen'):
        require(read(answer['program_result_ref'])['status']=='COMPLETE','data program not terminal')
        return answer
    if config.get('formal_rollout'):
        if stage=='capture':
            release=read(answer['resident_release_ref'])
            require(release['status']=='released' and release['terminal']['unknown_requests']==0
                    and release['terminal']['request_ref']==pin(marker)
                    and release['terminal']['generation_closed_ref']==answer['generation_closed_ref'],
                    'formal generation did not release its exact original lease')
        else:
            require(stage=='endpoint' and pin(answer['witness_ref'])==answer['witness_ref'],
                    'missing resident endpoint witness')
        return answer
    receipt=read(answer['owner_receipt_ref'])
    require(receipt['unknown_upper']==0 and receipt['request_ref']==pin(marker),'owner unknown budget')
    original_controller().restored(receipt['original_restore_ref'])
    return answer


def train(config,root,trajectory,witness,parity,learning,resume,iteration,*,diagnostic=False,policy=None):
    directory=Path(root)/('step-%06d'%iteration)/('diagnostic' if diagnostic else 'update')
    spec=copy.deepcopy(read(config['training_package_template']))
    spec.update(execution_mode=config.get('execution_mode','formal'),diagnostic_scope=config.get('diagnostic_scope','strict_full'),diagnostic_source_population=config.get('diagnostic_source_population'))
    for key in ('diagnostic_protocol','diagnostic_target_nodes','engineering_update_sequence'):
        if key in config:spec[key]=copy.deepcopy(config[key])
    if config.get('diagnostic_scope')=='dataset_profile':spec['dataset_profile']=copy.deepcopy(config['dataset_profile'])
    spec.update(trajectory=trajectory,endpoint_witness=witness,parity=parity,learning=learning,resume=resume,
                action='diagnostic' if diagnostic else 'acceptance',update_mode='train' if resume else 'bounded_first_update')
    spec['budget'].update(deadline_epoch=config['deadline_epoch'],max_optimizer_updates=config['max_optimizer_updates'] if resume or config.get('resident',{}).get('enabled') else 1)
    if config.get('diagnostic_index'):spec['diagnostic_index']=config['diagnostic_index']
    if diagnostic and policy and config.get('_parity_capture'):
        spec.update(pending_capture=config['_parity_capture']['pending_capture_ref'],generation_inventory=config['_parity_capture']['generation_inventory_ref'])
        if config['_parity_capture'].get('rollout_logits_index_ref'):
            spec['rollout_logits_index_ref']=config['_parity_capture']['rollout_logits_index_ref']
        spec.pop('diagnostic_index',None)
    if not diagnostic:
        spec.pop('diagnostic_index',None)
    # Target transport carries sealed source+existing refs once; compiler on181
    # generates all runtime/window/learning/lease hashes before actual dispatch.
    if config.get('newprocess_replay',{}).get('enabled'):
        from newprocess_replay import train_spec
        if diagnostic:spec['newprocess_replay']=dict(enabled=True,role='A',root_seed=config['root_seed'])
        else:train_spec(config,spec,source_input=config['_replay_source_input'])
    from transport import training
    result=training(config,spec,directory)
    if not diagnostic:
        from checkpoint_retention import authorize_saved
        authorize_saved(config,result,directory)
    return result


def initial_resume(config, initial):
    """Initialize a new cursor using the original sealed policy migration."""
    if 'initial_resume_ref' not in config:
        require(read(initial['clock_state_ref'])['successful_optimizer_steps']==0,
                'nonzero Clock requires an explicit checkpoint ref')
        return dict(step=0,resume_ref=None,cohort_sequence=0)
    ref=pin(config['initial_resume_ref']);checkpoint=read(ref)
    from publication import validate_publication
    policy=read(initial['policy']['current_ref'])
    publication=validate_publication(policy['publication_ref'],verify_checkpoint_shards=True)
    clock=read(initial['clock_state_ref']);step=checkpoint['step']
    require(step>0 and step==policy['step']==publication['step']==clock['successful_optimizer_steps']
            and ref==publication['checkpoint_ref']==pin(initial['native']['restore_checkpoint_ref']),
            'checkpoint, migration, and current policy must agree')
    require(initial['continuation']==dict(publication_ref=policy['publication_ref'],successful_optimizer_steps=step),
            'original migration receipt required')
    require(all(clock[k]==v for k,v in checkpoint['clock'].items() if k not in ('config_version','config_sha256')),
            'initial cursor must inherit the exact committed Clock history')
    require(step<config['max_optimizer_updates'],'predecessor already reached configured step limit')
    # The resident scientific contract cannot silently ingest a bounded
    # checkpoint. The bounded actor entry authenticates that transition itself.
    from formal_actor_transport import settings as actor_settings
    bounded=checkpoint['contract'].get('schema')=='omni-bounded-checkpoint-v1'
    require(bounded == (actor_settings(config) is not None),
            'bounded checkpoint and formal strict actor must be configured together')
    return dict(step=step,resume_ref=ref,cohort_sequence=step)


def inherited_screen(config, state, witness):
    """Consume the original DATA phase's terminal response without rerunning it."""
    response_ref = pin(config.get('initial_screen_result_ref') or config['initial_screen_result_path'])
    answer = read(response_ref)
    request_ref = pin(answer['request_ref'])
    request = read(request_ref)
    origin = read(request['config_ref'])
    require(answer['status'] == 'complete' and answer['fixture'] is False
            and request['stage'] == 'screen' and request['fixture'] is False
            and request['policy_ref'] == state['policy_ref']
            and request['input_ref'] == state['input_ref']
            and request['endpoint_witness_ref'] == witness
            and request['dataset_ref'] == config['datasets']['TRAIN']
            and request['system_policy_ref'] == config['system_policy_ref']
            and request['authorization_ref'] == config['authorization_ref']
            and origin['program_template_ref'] == config['program_template_ref']
            and origin['program_predecessor_output'] == config['program_predecessor_output'],
            'initial DATA result belongs to a different policy, population or configuration')
    if answer.get('schema')=='node-screen-data-result-v1':
        from node_screen_selection import inherited_cost
        actual_tokens=inherited_cost(answer,request_ref,request)
        upper=request['budget']['tokens_upper']
        require(actual_tokens<=upper and state['charged_tokens_upper']+upper<=config['budget']['total_tokens'],
                'initial P screen exceeds the recorded envelope')
        state['charged_tokens_upper']+=upper
        state['inherited_screen']=dict(response_ref=response_ref,request_ref=request_ref,usage_ref=answer['usage_ref'],
            actual_cost_units=actual_tokens,tokens_upper=upper)
        return answer
    result = read(answer['program_result_ref'])
    pool = read(answer['screen_ref'])
    program = read(result['config'])
    cursor = read(Path(program['output'])/'CURSOR.json')
    require(result['status'] == 'COMPLETE' and result['sample']['status'] == 'SCREEN_SAMPLE_COMPLETE'
            and program['request_ref'] == request_ref and program['stages'] == ['TRAIN']
            and cursor['stage'] == 'DONE' and cursor['output'] == answer['program_predecessor_output']
            and answer['population'] == result['sample']['observed'] == 119881
            and result['sample']['pool'] == answer['screen_ref']
            and pool['population_complete'] and not pool['synthetic_fixture'],
            'initial DATA result requires a closed original full TRAIN screen')
    budget = read(result['budget'])
    reservations = budget['reservations'].values()
    require(all(row['status'] == 'closed' for row in reservations),
            'unreconciled initial DATA reservation')
    actual_tokens = sum(row['cost_units'] for row in reservations)
    upper = request['budget']['tokens_upper']
    require(actual_tokens <= upper and state['charged_tokens_upper']+upper <= config['budget']['total_tokens'],
            'initial DATA result exceeds the recorded envelope')
    state['charged_tokens_upper'] += upper
    state['inherited_screen'] = dict(response_ref=response_ref, request_ref=request_ref,
        program_result_ref=answer['program_result_ref'], actual_cost_units=actual_tokens, tokens_upper=upper)
    state['program_predecessor_output'] = answer['program_predecessor_output']
    return answer


def run(config_path):
    config=read(config_path);config['_path']=str(Path(config_path).resolve());root=Path(config['output_root'])
    require(config['schema']=='omni-dp8-iteration-v1','iteration schema')
    require(config['monitor_seconds']==900 and config['max_optimizer_updates']>0,'bounded configured execution')
    if config.get('newprocess_replay',{}).get('enabled'):
        require(config.get('resident',{}).get('enabled') and config['resident']['checkpoint_interval']==1, 'A requires same resident and full checkpoints')
        require(config['max_optimizer_updates']==2 and config.get('execution_mode','formal')=='formal', 'replay uses original formal two-step schedule')
    from finite_runtime import validate_mode, prepare_finite, parity_accept, freeze_reward
    mode=validate_mode(config)
    finite=mode=='finite_engineering'
    bounded=config.get('bounded_restore_compare',False)
    if bounded:
        require(mode=='formal' and config.get('newprocess_replay',{}).get('enabled') and config['max_optimizer_updates']==2,'bounded mode requires original two-step replay')
    if finite and config.get('diagnostic_release_ref'):
        release=read(config['diagnostic_release_ref'])
        require(release['fixed_selector_repair_present'] and release['dispatch_allowed'],
                'fixed DATA selector repair not sealed; production iteration forbidden')


    from sampled_training import specification as sample_spec, prepare as prepare_samples, complete as samples_complete, bind_window, consume as consume_samples
    from formal_actor_transport import settings as actor_settings
    strict_actor=actor_settings(config)
    if strict_actor:
        from formal_rollout import startup_placement
        startup_placement(config)
    sample_spec(config)
    import fcntl
    from storage import environment,commit
    from storage import configure_scope
    configure_scope(root,config['client_storage_bytes'],config['run_id']+':client',Path('/proc/sys/kernel/random/boot_id').read_text().strip(),noncheckpoint_limit_bytes=config.get('client_noncheckpoint_limit_bytes'))
    environment(root,config['client_storage_bytes'])
    with (root/'iteration.guard').open('ab') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from transport import restore_locations
        for closed in config.get('inherited_closed_returns',[]):
            # Only immutable predecessor returns; retain their original paths
            # and identities rather than copying or replaying model requests.
            restore_locations(closed)
        for closed in sorted(root.glob('mailbox/*/transport/returned/CLOSED-COMMIT.json')):
            restore_locations(closed.parent)
        if strict_actor:
            for closed in sorted(root.glob('step-*/update/returned/CLOSED-COMMIT.json')):
                restore_locations(closed.parent)
        if finite and config.get('runtime_construction'):
            built=read(config['runtime_input_ref'])
            require(pin(config['initial_input'])==built['input_ref'],'sealed runtime input drift')
            require(config['endpoint_probe_packet_ref']==built['endpoint_probe_packet_ref'] and config['endpoint_probe_query_id']==built['endpoint_probe_query_id'],'sealed runtime endpoint probe drift')
        cursor=root/'CURSOR.json'
        if cursor.exists():
            state=read(cursor);require(state['config_ref']==pin(config_path),'iteration config drift')
            require(state.get('execution_mode','formal')==mode,'iteration finite/formal identity drift')
        else:
            initial_ref=pin(config['initial_input']);initial=read(initial_ref)
            resume=initial_resume(config,initial)
            require(not (finite and resume['step']),'finite startup must not reset a restored Clock')
            if resume['step']:
                sample_spec(config,prior_step=resume['step'])
            first_phase='endpoint' if resume['step'] or not finite else 'finite_prepare'
            if strict_actor and config.get('initial_screen_witness_ref'):
                first_phase='initial_screen'
            state=dict(config_ref=pin(config_path),execution_mode=mode,phase=first_phase,sequence=0,
                input_ref=initial_ref,policy_ref=initial['policy']['current_ref'],learning_ref=initial['learning_config_ref'],
                trajectory_ref=pin(config['trajectory']),best=None,stale=0,sampling_round=0,
                screen=None,zero_signal_batches=0,consumed_cohorts=[],program_predecessor_output=config['program_predecessor_output'],
                charged_tokens_upper=config['budget']['prior_tokens_upper'],reservations={},**resume)
            commit(cursor,state)
        if state.get('accepted_dataset_profile'):
            config['dataset_profile']=copy.deepcopy(state['accepted_dataset_profile'])
        def persist():commit(cursor,state,replace=True)
        def allocate(stage,extra=None):
            # Persist exact identity and full reservation BEFORE transport. A
            # crash resumes the same reservation, never gives back unknown work.
            if state.get('pending_stage') is None:
                budget=stage_budget(config,stage,extra)
                require(state['charged_tokens_upper']+budget['tokens_upper']<=config['budget']['total_tokens'],'cumulative total budget exhausted')
                state['sequence']+=1;key=str(state['sequence'])+'-'+stage
                state['charged_tokens_upper']+=budget['tokens_upper']
                state['reservations'][key]=dict(stage=stage,tokens_upper=budget['tokens_upper'],status='intent_unknown')
                state['pending_stage']=dict(key=key,stage=stage,sequence=state['sequence'],purpose=(extra or {}).get('purpose'));persist()
            require(state['pending_stage']['stage']==stage and state['pending_stage'].get('purpose')==(extra or {}).get('purpose'),'cursor stage reservation drift')
            if config.get('resident',{}).get('enabled'):
                config['_charged_tokens_upper']=state['charged_tokens_upper']
            return state['pending_stage']['sequence']
        def settled(payload):
            state['reservations'][state['pending_stage']['key']]['status']='terminal_known'
            state['pending_stage']=None
            if 'charged_tokens_upper' in payload:
                from budget_counts import logical_count
                state['charged_tokens_upper']=max(logical_count(state['charged_tokens_upper']),
                    logical_count(payload['charged_tokens_upper']))
            if payload.get('program_predecessor_output'):state['program_predecessor_output']=payload['program_predecessor_output']
        def call(stage,extra=None,*,input_ref=None):
            sequence=allocate(stage,extra)
            payload=request(config,root,sequence,stage,state['input_ref'] if input_ref is None else input_ref,state['policy_ref'],state.get('witness_ref'),
                dict(extra or {},program_predecessor_output=state['program_predecessor_output']))
            settled(payload);return payload
        while state['phase']!='done':
            require(time.time()<config['deadline_epoch'],'absolute iteration deadline; preserve same cursor')
            from diagnostic_retention import collect_ready
            collect_ready(config,state,root,guard)
            phase=state['phase'];step=state['step']
            if phase=='finite_prepare':
                bound=prepare_finite(config,state['input_ref'],root/'finite-input',state['policy_ref'])
                state['input_ref']=bound['input_ref'];state['learning_ref']=bound['learning_ref']
                state['phase']='endpoint';persist()
            elif phase=='initial_screen':
                from program_placement import resident_provider
                provider=resident_provider(config)
                witness=pin(config['initial_screen_witness_ref'])
                # Dataset screening and fresh rollout can have different owners.
                # The immutable DATA request authenticates the screen witness in
                # inherited_screen. Fresh capture authenticates its live owner
                # independently after handoff via the endpoint/lease checks.
                inherits=config.get('initial_screen_result_path') or config.get('initial_screen_result_ref')
                require(strict_actor and step==read(config['initial_resume_ref'])['step']
                        and provider and (inherits or provider['service_ready']==witness)
                        and read(witness)['schema']=='resident-tp4-startup-observation-v1',
                        'initial screen requires the configured resident startup observation')
                # DATA authenticates the resident adapter under its lease.
                # Same-action training admission still runs at the next phase.
                state['witness_ref']=witness
                waiting=config.get('initial_screen_result_path')
                if waiting and not state.get('inherited_screen') and not Path(waiting).is_file():
                    from storage import check_running
                    check_running(root,config['client_storage_bytes'])
                    time.sleep(config['monitor_seconds'])
                    continue
                screened = (inherited_screen(config, state, witness)
                            if config.get('initial_screen_result_ref') or waiting else call('screen'))
                state['screen']=screened
                state['screen_policy_ref']=state['policy_ref']
                from next_input import bind_phase
                bound=bind_phase(state['input_ref'],screened,root/('phase-binding-%06d'%state['sequence']))
                state['input_ref']=bound['input_ref']
                state['learning_ref']=bound['learning_ref']
                prepare_samples(config,state,root)
                state['witness_ref']=None
                state['phase']='endpoint';persist()
            elif phase=='endpoint':
                # Request identity is persisted; old witness is historical, not
                # a claim that the endpoint is still active in a later window.
                state['witness_ref']=None
                if strict_actor:
                    from formal_rollout import qualify
                    state['policy_ref']=qualify(config,state,root/f'policy-{step:06d}',guard)
                    persist()
                endpoint=call('endpoint');state['witness_ref']=endpoint['witness_ref']
                if strict_actor:
                    if 'baseline_dev_ref' not in state and 'deferred_system_dev_ref' not in state:
                        require(state['resume_ref']==config['initial_resume_ref'],
                                'restored formal run has no DEV baseline for its initial policy')
                        state['phase']='baseline_dev'
                    else:
                        state['phase']='dev' if (state.get('phase_boundary') or samples_complete(config,state)
                            or step%config['selection']['dev_every_optimizer_steps']==0) else 'capture'
                else:
                    state['phase']='parity_capture'
                persist()
            elif phase=='baseline_dev':
                require(strict_actor and state['resume_ref']==config['initial_resume_ref'],
                        'baseline DEV must use the restored initial policy')
                from formal_resume import defer_system_dev
                if defer_system_dev(config, state, root):
                    persist()
                    continue
                dev=call('dev')
                if __import__('os').environ.get('OMNI_FORMAL_RESUME_REF'):
                    from formal_resume import baseline_witness
                    state['witness_ref']=baseline_witness(config,state)
                state.update(baseline_dev_ref=dev['score_ref'], phase_baseline_dev_ref=dev['score_ref'],
                             baseline_policy_ref=state['policy_ref'], baseline_input_ref=state['input_ref'],
                             phase='capture')
                persist()
            elif phase=='parity_capture':
                # Candidate may collect bounded diagnostic evidence before its
                # updated parity promotion, never training on these samples.
                purpose='updated_parity' if state['resume_ref'] else 'base_parity'
                if not state.get('parity_input_ref'):
                    if config.get('dataset_profile',{}).get('acquisition'):
                        from boundary_acquisition import prepare_input
                        prepared=prepare_input(config,state['input_ref'],root/f'parity-input-{step:06d}',state['policy_ref'],state['witness_ref'],purpose)
                        state['parity_input_ref']=prepared['input_ref']
                    elif finite or read(state['input_ref']).get('prepare_mode')=='bounded_train_probe' and state['screen'] is None:
                        from finite_runtime import prepare_diagnostic
                        prepared=prepare_diagnostic(config,state['input_ref'],root/f'parity-input-{step:06d}',
                            state['policy_ref'],state['witness_ref'],purpose)
                        state['parity_input_ref']=prepared['input_ref']
                    else:
                        from next_input import diagnostic_input
                        state['parity_input_ref']=diagnostic_input(state['input_ref'],state['screen'],state['policy_ref'],state['witness_ref'],
                            root/f'parity-input-{step:06d}',root_seed=config['root_seed'],sampling_round=state['sampling_round'],
                            cohort_sequence=state['cohort_sequence'],sampling=config.get('diagnostic_sampling'))
                    persist()
                # Only this evidence-producing request sees the diagnostic
                # input. The pending business phase never gets fake validation.
                # allocate() persists the cursor before transport. Keep the
                # training input durable even if this request crashes; only
                # the request itself receives the diagnostic input override.
                captured=call('capture',dict(purpose=purpose),input_ref=state['parity_input_ref'])
                state['parity_capture']=captured
                state['witness_ref']=captured['endpoint_witness_ref'];state['phase']='diagnostic';persist()
            elif phase=='diagnostic':
                if state.get('parity_capture'):config['_parity_capture']=state['parity_capture']
                receipt=state.get('closed_diagnostic')
                if receipt and receipt['step']==step:
                    numerical=read(receipt['result_ref'])
                else:
                    allocate('diagnostic')
                    numerical=train(config,root,state['trajectory_ref'],state['witness_ref'],None,
                        state['learning_ref'],state['resume_ref'],step,diagnostic=True,policy=state['policy_ref'])
                    settled(numerical)
                    state['closed_diagnostic']=dict(step=step,result_ref=write(
                        root/('DIAGNOSTIC-RETURN-%06d.json'%step),numerical))
                state['raw_ref']=numerical['raw_ref'];state['runtime_ref']=numerical['runtime_ref']
                # A closed command is a known cost even if verification fails.
                # Persist before verification so restart cannot lose its result.
                persist()
                from parity_gate import verify
                observed=read(state['raw_ref'])['resident_identity']
                try:
                    verification=verify(state['raw_ref'],config=read(state['runtime_ref']),policy_version=read(state['policy_ref'])['version'],
                        source_pins_sha256=__import__('dependencies').sha(HERE/'SOURCE-PINS.json'),require_updated=state['resume_ref'] is not None,identity=observed)
                except ValueError as error:
                    if str(error) not in ('real DP8 R/C/P failed; original thresholds retained',
                            'complete actual full-vocabulary output gate failed or missing'):
                        raise
                    state['diagnostic_failure_ref']=write(root/('DIAGNOSTIC-REJECTED-%06d.json'%step),dict(
                        error=str(error),raw_ref=state['raw_ref'],runtime_ref=state['runtime_ref'],
                        charged_tokens_upper=state['charged_tokens_upper'],optimizer_steps=step,
                        numerical_rejection=True,automatic_retry=False))
                    state['phase']='diagnostic_rejected';persist();continue
                from diagnostic_retention import accepted
                accepted(config,state,verification,root)
                state['parity_acceptance']=parity_accept(state,mode)
                if config.get('diagnostic_scope')=='dataset_profile':
                    retain_parity_profile(config,state)
                    persist()
                if config.get('newprocess_replay',{}).get('enabled') and state['step']==1:
                    from newprocess_replay import audit_first_update
                    state['newprocess_replay']['first_update_audit_ref']=audit_first_update(
                        state['newprocess_replay']['A1'],state['raw_ref'],state['runtime_ref'],root/'FIRST-UPDATE-AUDIT.json')

                if state['resume_ref']:
                    from publication import promote
                    state['policy_ref']=promote(state['policy_ref'],state['raw_ref'],state['runtime_ref'],root/f'promoted-{step:06d}')
                # Full DEV is evaluated on configured step intervals and phase
                # boundaries; full TRAIN rescreen only after business promotion.
                state['phase']='dev' if step==0 or step%config['selection']['dev_every_optimizer_steps']==0 or step>=config['max_optimizer_updates'] else 'capture'
                if state.get('phase_boundary'):state['phase']='dev'
                if samples_complete(config,state):state['phase']='dev'
                if finite:state['phase']='finite_complete' if step>=config['max_optimizer_updates'] else 'capture'
                if bounded:state['phase']='capture'
                from diagnostic_retention import release_consumers
                release_consumers(config,state,root)
                persist()
            elif phase=='dev':
                if state.get('deferred_system_dev_ref') and 'baseline_dev_ref' not in state:
                    from formal_resume import accept_deferred_system_dev
                    if not accept_deferred_system_dev(config, state):
                        time.sleep(config['monitor_seconds'])
                        continue
                    persist()
                dev=call('dev');state['current_dev_ref']=dev['score_ref']
                if step==0:
                    state['baseline_dev_ref']=dev['score_ref'];state['baseline_policy_ref']=state['policy_ref'];state['baseline_input_ref']=state['input_ref']
                    state['phase_baseline_dev_ref']=dev['score_ref']
                    state['phase']='screen';persist();continue
                from business_gate import assess
                baseline=state['baseline_dev_ref']
                quality_ref,gate=assess(baseline,dev['score_ref'],config['selection'],root/f'quality-dev-{step:06d}')
                state['last_dev_quality_ref']=quality_ref
                # Monitor the phase that produced these updates. At a boundary
                # the successful-step Clock already names the *next* phase.
                learning=read(state['learning_ref'])
                boundary=state.get('phase_boundary',False)
                phases=[p for p in learning['phases'] if p['start_step']<(step if boundary else step+1)]
                require(phases,'DEV has no producing phase')
                monitored=phases[-1];phase_id=monitored['id']
                scope=monitored['rollout_scope']
                require(scope in ('node','system'),'explicit phase rollout scope')
                if state.get('monitor_phase')!=phase_id:
                    state['monitor_phase']=phase_id;state['stale']=0
                    state['system_stale']=state.get('system_stale',0)
                    state['phase_baseline_dev_ref']=state.get('phase_baseline_dev_ref',baseline)
                    state['node_best']=None
                phase_baseline=state.get('phase_baseline_dev_ref',baseline)
                phase_passed=True
                if scope=='node' or boundary:
                    from business_gate import assess_phase
                    phase_ref,phase_result=assess_phase(phase_baseline,dev['score_ref'],config['selection'],phase_id,root/f'phase-quality-{step:06d}')
                    state['last_phase_quality_ref']=phase_ref
                    phase_passed=phase_result['passed']
                # Final system selection always uses the initial business gate;
                # a node improvement never creates an eligible system best.
                eligible=gate['dev_selection_eligible']
                if eligible and state['best']:
                    ranking_ref,ranking=assess(state['best']['dev_score_ref'],dev['score_ref'],config['selection'],root/f'quality-best-{step:06d}')
                    eligible=read(ranking_ref)['paired']['mean_reward' if 'reward_evaluation' in config['selection'] else 'slu_f1']['delta']>0
                selected=dict(policy_ref=state['policy_ref'],input_ref=state['input_ref'],witness_ref=state['witness_ref'],
                    dev_score_ref=dev['score_ref'],quality_ref=quality_ref,iteration=step)
                if scope=='node':
                    # Preserve the configured scalar and regression guards for
                    # the entire phase, including resumed DEV observations.
                    spec=config['selection']['phase_acceptance'][phase_id]
                    require(spec['target'] in monitored['target_nodes'],'phase target differs from trainable node')
                    identity=dict(phase=phase_id,specification=spec,baseline_ref=phase_baseline)
                    previous=state.get('node_monitor_identity')
                    if previous and previous['phase']==phase_id:
                        require(previous==identity,'node patience target/guards/baseline drift')
                    state['node_monitor_identity']=copy.deepcopy(identity)
                    require(phase_result['specification']==spec,'phase evidence target differs')
                    target=phase_result['target_paired']
                    progress=phase_passed and target['paired']['slu_f1']['delta']>0
                    if progress and state.get('node_best'):
                        rank_ref,rank=assess_phase(state['node_best']['dev_score_ref'],dev['score_ref'],config['selection'],phase_id,root/f'node-best-quality-{step:06d}')
                        # Regression guards are anchored to the fixed phase
                        # baseline above. The same evaluator compares the best.
                        checks=rank['checks']
                        progress=(checks['target_gain'] and checks['target_paired_lower_bound']
                                  and rank['target_paired']['paired']['slu_f1']['delta']>0)
                        state['last_node_ranking_ref']=rank_ref
                    if progress:
                        state['node_best']=dict(selected,quality_ref=phase_ref,target=spec['target'],
                            record_metric_path=spec['record_metric_path'],
                            target_value=target['candidate']['slu_f1'],final_business_promotion=False)
                    state['node_bests']=dict(state.get('node_bests',{}),**{phase_id:state.get('node_best')})
                else:
                    progress=eligible
                state['stale']=0 if progress else state['stale']+1
                if scope=='system':state['system_stale']=state['stale']
                else:state['node_stale']=state['stale']
                transition=False
                if boundary:
                    require(config.get('phase_failure_policy','stop_preserve')=='stop_preserve','explicit phase failure policy')
                    if not phase_passed:
                        state['stop_reason']='PHASE_DEV_GATE_FAILED'
                        state['phase']='phase_stopped' if sample_spec(config) else 'freeze'
                        persist();continue
                    from next_input import phase_validation
                    state['input_ref']=phase_validation(state['input_ref'],dev['score_ref'],phase_ref,True,root/f'validation-{step:06d}')
                    state['phase_baseline_dev_ref']=dev['score_ref']
                    state['monitor_phase']=[p for p in learning['phases'] if p['start_step']<=step][-1]['id']
                    state['node_best']=None;state['stale']=0;state['node_stale']=0;state['system_stale']=0
                    state['phase_boundary']=False;transition=True
                if eligible:
                    state['best']=dict(selected,input_ref=state['input_ref'])
                # Every fresh full-TRAIN screen gets a new round identity,
                # including a node->node/system transition without final gain.
                if (eligible or transition) and not sample_spec(config):
                    state['sampling_round']+=1;state['screen']=None
                if sample_spec(config):
                    state['phase']='freeze' if samples_complete(config,state) else ('phase_screen' if transition else ('screen' if state['screen'] is None else 'capture'))
                elif step>=config['max_optimizer_updates']:
                    state['phase']='freeze'
                elif state['stale']>=config['selection']['patience']:
                    state['stop_reason']='NODE_PATIENCE' if scope=='node' else 'SYSTEM_PATIENCE'
                    state['phase']='freeze'
                else:state['phase']='screen' if state['screen'] is None else 'capture'
                persist()
            elif phase=='phase_screen':
                # Refresh upstream observations under the promoted policy,
                # without resampling/reclassifying this dataset's frozen draws.
                refreshed=call('screen')
                from next_input import bind_phase
                bound=bind_phase(state['input_ref'],refreshed,root/('phase-binding-%06d'%state['sequence']))
                state['input_ref']=bound['input_ref'];state['learning_ref']=bound['learning_ref']
                state.setdefault('phase_upstream_screens',[]).append(dict(
                    screen=refreshed,policy_ref=state['policy_ref'],step=step))
                state['phase']='capture';persist()
            elif phase=='screen':
                state['screen']=call('screen');state['screen_policy_ref']=state['policy_ref']
                from next_input import bind_phase
                bound=bind_phase(state['input_ref'],state['screen'],root/('phase-binding-%06d'%state['sequence']))
                state['input_ref']=bound['input_ref'];state['learning_ref']=bound['learning_ref']
                prepare_samples(config,state,root)
                state['phase']='capture';persist()
            elif phase=='capture':
                if samples_complete(config,state):
                    state['stop_reason']='SAMPLED_TRAIN_COMPLETE';state['phase']='freeze';persist();continue
                if sample_spec(config):
                    require(step<config['max_optimizer_updates'],'sampled TRAIN incomplete at storage/update envelope')
                if step>=config['max_optimizer_updates'] or time.time()>=config['deadline_epoch']-config['final_test_reserve_seconds']:
                    state['phase']='freeze';persist();continue
                if not state.get('capture_pending'):
                    state['cohort_sequence']=state.get('cohort_sequence',0)+1;state['capture_pending']=True;persist()
                from next_input import capture_input
                source_input=state.get('capture_source_input_ref',state['input_ref'])
                state['capture_source_input_ref']=source_input
                if bounded:
                    from next_input import bounded_capture_input
                    state['capture_input_ref']=bounded_capture_input(config,source_input,state['policy_ref'],state['witness_ref'],
                        state['learning_ref'],root/('inputs-%06d'%state['cohort_sequence']),cohort_sequence=state['cohort_sequence'])
                elif finite:
                    if config.get('engineering_update_sequence'):
                        from next_input import bounded_capture_input
                        state['capture_input_ref']=bounded_capture_input(config,source_input,state['policy_ref'],state['witness_ref'],
                            state['learning_ref'],root/('inputs-%06d'%state['cohort_sequence']),cohort_sequence=state['cohort_sequence'])
                    else:
                        require(state['cohort_sequence']==1,'finite mode cannot create successor cohort')
                        state['capture_input_ref']=source_input
                else:
                    state['capture_input_ref']=capture_input(source_input,state['screen'],state['policy_ref'],state['witness_ref'],
                        state['learning_ref'],state['resume_ref'],root/('inputs-%06d'%state['cohort_sequence']),root_seed=config['root_seed'],sampling_round=state['sampling_round'],cohort_sequence=state['cohort_sequence'])
                    state['capture_input_ref']=bind_window(config,state,state['capture_input_ref'],root/('sample-window-%06d'%state['cohort_sequence']))
                previous=state['input_ref'];state['input_ref']=state['capture_input_ref']
                # Every optimizer step gets a new complete G under the CURRENT
                # policy, while retaining the full TRAIN pool for this round.
                captured=call('capture');state['input_ref']=captured['input_ref'];state['capture']=captured;state['phase']='finalize';persist()
            elif phase=='finalize':
                if strict_actor:
                    # The strict actor restores, scores and updates in one
                    # guarded process. No separate legacy training transport.
                    state['phase']='update';persist();continue
                allocate('finalize')
                spec=copy.deepcopy(read(config['training_package_template']))
                spec.pop('diagnostic_index',None)
                spec.update(execution_mode=config.get('execution_mode','formal'),diagnostic_scope=config.get('diagnostic_scope','strict_full'),diagnostic_source_population=config.get('diagnostic_source_population'))
                for key in ('diagnostic_protocol','diagnostic_target_nodes','engineering_update_sequence'):
                    if key in config:spec[key]=copy.deepcopy(config[key])
                if config.get('diagnostic_scope')=='dataset_profile':spec['dataset_profile']=copy.deepcopy(config['dataset_profile'])
                spec.update(action='finalize',pending_capture=state['capture']['pending_capture_ref'],
                    generation_inventory=state['capture']['generation_inventory_ref'],trajectory=state['trajectory_ref'],
                    endpoint_witness=state['witness_ref'],learning=state['learning_ref'],resume=state['resume_ref'])
                spec['budget'].update(deadline_epoch=config['deadline_epoch'],max_optimizer_updates=config['max_optimizer_updates'])
                from transport import training
                finalized=training(config,spec,root/('cohort-%06d'%state['cohort_sequence'])/'finalize')
                settled(finalized);state['trajectory_ref']=finalized['trajectory_ref'];state['phase']='update';persist()
            elif phase=='update':
                if not strict_actor:
                    require(state.get('parity_acceptance')==parity_accept(state,mode),'missing or drifted parity acceptance before update')
                allocate('update')
                if config.get('newprocess_replay',{}).get('enabled'):
                    config['_replay_source_input']=state['capture_input_ref']
                if strict_actor:
                    from formal_actor_transport import train as strict_train
                    result=strict_train(config,state,root/('step-%06d'%state['cohort_sequence'])/'update',guard)
                    state['trajectory_ref']=result['trajectory_ref']
                else:
                    result=train(config,root,state['trajectory_ref'],state['witness_ref'],state['raw_ref'],state['learning_ref'],state['resume_ref'],state['cohort_sequence'])
                settled(result)
                consume_samples(config,state,result)
                state.pop('capture_source_input_ref',None)
                state['consumed_cohorts'].append(dict(sequence=state['cohort_sequence'],trajectory_ref=state['trajectory_ref'],result=result))
                state['capture_pending']=False
                if result['optimizer_steps']==0:
                    # Consumed terminal cohort cannot advance Clock or be replayed.
                    if result.get('result_ref') and read(result['result_ref']).get('clock'):
                        from next_input import retire_zero
                        state['input_ref']=retire_zero(state['input_ref'],result['result_ref'],root/('zero-clock-%06d'%state['cohort_sequence']))
                    if strict_actor:
                        state.setdefault('zero_result_refs',[]).append(result['result_ref'])
                    state['zero_signal_batches']=state.get('zero_signal_batches',0)+1
                    bound=config['max_zero_signal_batches'];require(type(bound) is int and bound>0,'finite zero signal batch budget')
                    state['phase']=(('dev' if state['step'] else 'freeze') if samples_complete(config,state) else 'capture') if sample_spec(config) else ('finite_complete' if finite or bounded else ('freeze' if state['zero_signal_batches']>=bound else 'capture'))
                    if state['phase'] in ('freeze','finite_complete'):state['stop_reason']='NO_LEARNING_SIGNAL'
                    persist();continue
                state['zero_signal_batches']=0
                require(result['optimizer_steps']==1,'only committed one-step window accepted')
                state.pop('parity_capture',None);config.pop('_parity_capture',None)
                state.pop('parity_input_ref',None)
                learning=read(state['learning_ref'])
                before_phase=[p for p in learning['phases'] if p['start_step']<=state['step']][-1]['id']
                next_step=read(result['checkpoint_ref'])['step']
                after_phase=[p for p in learning['phases'] if p['start_step']<=next_step][-1]['id']
                state['phase_boundary']=before_phase!=after_phase
                state['resume_ref']=result['checkpoint_ref'];state['policy_ref']=result['policy_ref']
                state.pop('zero_result_refs',None)
                state['step']=read(state['resume_ref'])['step']
                if config.get('newprocess_replay',{}).get('enabled'):
                    from newprocess_replay import after_committed
                    if after_committed(config,state,result,root):
                        state['phase']='newprocess_replay';persist();continue
                from next_input import migrate
                migration=migrate(state['input_ref'],state['policy_ref'],state['resume_ref'],root/f'migration-{state["step"]:06d}')
                state['input_ref']=migration['input_ref'];state['learning_ref']=migration['learning_ref'];state['phase']='endpoint';persist()
            elif phase=='newprocess_replay':
                # Existing end waits for actual actor exit and original service
                # restoration. Only then original transport creates B's container.
                from resident_transport import end
                end(config)
                from newprocess_replay import replay_second
                ref=replay_second(config,state,root)
                write(root/'OUTCOME.json',dict(status='NEWPROCESS_REPLAY_COMPARED',comparison_ref=ref,
                    achieved=False,dev_gain_claim=False,test_used=False,steps=2))
                state['phase']='done';persist()
            elif phase=='diagnostic_rejected':
                write(root/'OUTCOME.json',dict(status='DIAGNOSTIC_REJECTED',achieved=False,
                    steps=state['step'],failure_ref=state['diagnostic_failure_ref'],
                    dev_gain_claim=False,test_used=False,automatic_retry=False))
                state['phase']='done';persist()
            elif phase=='finite_complete':
                write(root/'OUTCOME.json',dict(status=state.get('stop_reason','FINITE_ENGINEERING_COMPLETE'),achieved=False,
                    execution_mode=mode,bounded_restore_compare=bounded,steps=state['step'],cohorts=len(state['consumed_cohorts']),
                    formal_80_20=False,dev_gain_claim=False,test_used=False,
                    policy_ref=state['policy_ref'],checkpoint_ref=state['resume_ref'],
                    parity_acceptance=state.get('parity_acceptance'),cursor_ref=pin(cursor)))
                state['phase']='done';persist()
            elif phase=='phase_stopped':
                # This is a terminal quality stop, never sample completion.
                # Preserve the exact cursor and failed DEV evidence for review.
                write(root/'OUTCOME.json',dict(status=state['stop_reason'],achieved=False,
                    sampled_dataset_ref=state['sampled_dataset_ref'],sampled_consumed=state['sampled_consumed'],
                    sampled_complete=samples_complete(config,state),steps=state['step'],
                    phase_quality_ref=state['last_phase_quality_ref'],test_used=False))
                state['phase']='done';persist()
            elif phase=='freeze':
                if sample_spec(config):
                    require(samples_complete(config,state),'cannot freeze with unconsumed sampled TRAIN')
                    write(root/'SAMPLED-TRAIN-COMPLETE.json',dict(dataset_ref=state['sampled_dataset_ref'],
                        consumed=state['sampled_consumed'],windows=state['sampled_windows'],
                        optimizer_steps=state['step'],DEV_gain_proven=state['best'] is not None))
                if state['best'] is None:
                    write(root/'OUTCOME.json',dict(status=state.get('stop_reason','NO_DEV_GAIN'),achieved=False,baseline_policy_ref=state['baseline_policy_ref'],
                        test_used=False,steps=state['step'],cursor_ref=pin(cursor)))
                    state['phase']='done';persist();continue
                state['selection_ref']=freeze_reward(config,state,root,config_path)
                if 'reward_evaluation' in config['selection']:
                    state['test_selection']=copy.deepcopy(config['selection'])
                    state['test_selection']['reward_evaluation']['frozen_dev_selection_ref']=state['selection_ref']
                state['selected_policy_ref']=state['best']['policy_ref'];state['selected_input_ref']=state['best']['input_ref']
                state['policy_ref']=state['baseline_policy_ref'];state['input_ref']=state['baseline_input_ref'];state['witness_ref']=None;state['phase']='test_baseline_endpoint';persist()
            elif phase=='test_baseline_endpoint':
                endpoint=call('endpoint');state['witness_ref']=endpoint['witness_ref'];state['phase']='test_baseline';persist()
            elif phase=='test_baseline':
                result=call('test',dict(final_selection_ref=state['selection_ref']));state['baseline_test_ref']=result['score_ref']
                state['policy_ref']=state['selected_policy_ref'];state['input_ref']=state['selected_input_ref'];state['witness_ref']=None;state['phase']='test_candidate_endpoint';persist()
            elif phase=='test_candidate_endpoint':
                endpoint=call('endpoint');state['witness_ref']=endpoint['witness_ref'];state['phase']='test_candidate';persist()
            elif phase=='test_candidate':
                result=call('test',dict(final_selection_ref=state['selection_ref']));state['candidate_test_ref']=result['score_ref']
                from business_gate import assess
                quality_ref,gate=assess(state['baseline_test_ref'],state['candidate_test_ref'],state.get('test_selection',config['selection']),root/'quality-test',split='TEST',frozen_selection=state['selection_ref'])
                achieved=gate['test_consistent_positive_evidence']
                write(root/'OUTCOME.json',dict(status='CONFIRMED_GAIN' if achieved else 'TEST_NO_GAIN_OR_INCONCLUSIVE',achieved=achieved,
                    frozen_selection_ref=state['selection_ref'],paired_test_ref=quality_ref,business_acceptance=gate,steps=state['step'],test_selection_allowed=False))
                state['phase']='done';persist()
            else:raise ValueError('unknown persisted phase: '+phase)
        if (config.get('formal_actor') or {}).get('continuous_actor'):
            from formal_actor_transport import end_resident
            end_resident(config,guard)
        elif config.get('resident',{}).get('enabled'):
            from resident_transport import end
            end(config)
        return read(root/'OUTCOME.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(a.config)
if __name__=='__main__':main()
