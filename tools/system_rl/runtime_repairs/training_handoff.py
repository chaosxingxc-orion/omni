"""Pre-admission successor preparation for the existing iteration controller.

No trainer or scheduler: retain closed capture and original accounting, prepare
a new config/cursor, then launch the same iteration.run via launch_fixed.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from dependencies import HERE, require
from run_package import read, pin, write


def terminal(container, host=None):
    argv=['docker','inspect',container]
    if host:
        import shlex
        argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,shlex.join(argv)]
    obj=json.loads(subprocess.check_output(argv,text=True,timeout=45))[0]
    require(obj['Id']==container and not obj['State']['Running'] and
            obj['State']['Status']=='exited','predecessor container is not terminal')
    return dict(id=obj['Id'],state=obj['State'],image=obj['Image'],
                devices=obj['HostConfig'].get('Devices',[]))


def probe_remote(plan, expected_controller):
    """Read original CPU-controller chain and prove no supervisor was admitted."""
    code='''import json,hashlib,subprocess,sys
from pathlib import Path
r=Path(sys.argv[1]);expected=sys.argv[2]
chain=json.loads((r/'CONTROLLER-RECOVERY-CHAIN.json').read_text())
last=chain[-1];assert last['container_id']==expected
obj=json.loads(subprocess.check_output(['docker','inspect',expected]))[0]
assert not obj['State']['Running'] and obj['State']['Status']=='exited' and obj['State']['ExitCode']!=0
assert not obj['HostConfig'].get('Devices')
p=Path(last['new_failure_path']);raw=p.read_bytes();failure=json.loads(raw)
assert failure['error'].startswith('diagnostic CPU selection failed before model load:'),failure['error']
assert not (r/'control/package/RUN.json').exists()
assert not (r/'control/package/RUN-COMPLETE.json').exists()
assert not (r/'control/package/hot/resident/READY.json').exists()
print(json.dumps(dict(controller=expected,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
state=obj['State'],failure_ref=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)),
failure=failure,supervisor_admitted=False,optimizer_started=False)))
'''
    import shlex
    argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',plan['host'],
          shlex.join(['python3','-B','-c',code,plan['target'],expected_controller])]
    evidence=json.loads(subprocess.check_output(argv,text=True,timeout=45))
    require(evidence['boot']==plan['job']['source_boot'],'predecessor source boot changed')
    return evidence


def successor_state(old, training_ref, config_ref, evidence_ref):
    require(old['phase']=='diagnostic' and old['step']==0 and old['resume_ref'] is None,
            'handoff only before the first admitted optimizer update')
    require(old['pending_stage']['stage']=='diagnostic' and old.get('parity_capture'),
            'complete diagnostic capture and failed diagnostic reservation required')
    require(not old.get('consumed_cohorts') and old.get('cohort_sequence',0)==0,
            'cannot migrate consumed training state through pre-admission handoff')
    s=copy.deepcopy(old)
    key=s['pending_stage']['key']
    require(s['reservations'][key]['status']=='intent_unknown','unexpected reservation terminal')
    s['reservations'][key].update(status='failed_pre_admission',terminal_evidence_ref=evidence_ref)
    s.update(config_ref=config_ref,input_ref=training_ref,pending_stage=None,
             predecessor_handoff_ref=evidence_ref)
    return s


def prepare(config_path):
    cfg=read(config_path)
    if cfg['schema'] in ('undispatched-p-reward-handoff-v1','closed-pre-reservation-capture-handoff-v1'):
        return prepare_undispatched(config_path)
    if cfg['schema']=='rejected-diagnostic-training-handoff-v1':
        return prepare_rejected(config_path)
    require(cfg['schema']=='pre-admission-training-handoff-v1','handoff schema')
    root=Path(cfg['output_root']);old=read(cfg['predecessor_config_ref'])
    old_root=Path(old['output_root']);state=read(cfg['predecessor_cursor_ref'])
    require(state['config_ref']==pin(cfg['predecessor_config_ref']),'predecessor cursor/config differs')
    require(Path(cfg['predecessor_cursor_ref']['path'])==old_root/'CURSOR.json','cursor outside original run')
    # Hold the same durable guard used by iteration while observing and preparing.
    import fcntl
    with (old_root/'iteration.guard').open('ab') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        client=terminal(cfg['predecessor_client'])
        require(client['state']['ExitCode']!=0,'predecessor is not a failed run')
        plan=read(old_root/'step-000000/diagnostic/TRANSPORT.json')
        remote=probe_remote(plan,cfg['predecessor_controller'])
        training_ref=pin(cfg['training_input_ref']);training=read(training_ref)
        require(not training.get('diagnostic_evidence_only') and not training.get('synthetic_fixture'),
                'diagnostic or fixture input cannot train')
        require(training['learning_config_ref']==state['learning_ref'] and
                training['policy']['current_ref']==state['policy_ref'],'training input policy/learning differs')
        require(read(training['clock_state_ref'])['successful_optimizer_steps']==0,'training clock already advanced')
        require(training_ref==pin(old_root/'finite-input/phase/INPUT.json'),
                'handoff must preserve original finite training input')
        from transport import restore_locations
        closed=old_root/'mailbox/0002-capture/transport/returned'
        restore_locations(closed)
        capture=state['parity_capture']
        require(capture['status']=='complete' and not capture['fixture'],'capture not complete/real')
        for key in ('generation_inventory_ref','pending_capture_ref','endpoint_witness_ref'):
            pin(capture[key])
        from registry_custody import facts
        ledger=facts(old['registry']['path'])
        require(all(r[2] in ('known','closed_unobserved') for r in ledger['spend']),
                'unknown ledger requests prohibit handoff')
        from release_completion import require_closed_release
        release=Path(cfg['release']);require_closed_release(release)
        require(HERE==release/'tree/parents/jev-system-rl/implementation',
                'handoff must run from exact successor release')
        run_root=Path(cfg['successor_output_root'])
        require(run_root!=old_root and run_root.parent==old_root.parent,'successor requires sibling local run root')
        from storage import environment
        environment(root,old['client_storage_bytes'])
        evidence_ref=write(root/'PREDECESSOR-TERMINAL.json',dict(
            predecessor_config_ref=pin(cfg['predecessor_config_ref']),
            predecessor_cursor_ref=pin(cfg['predecessor_cursor_ref']),client=client,remote=remote,
            closed_return_ref=pin(closed/'CLOSED-COMMIT.json'),
            registry_facts_sha256=hashlib.sha256(json.dumps(ledger,sort_keys=True).encode()).hexdigest(),
            unknown_requests=0,old_cost_reservations_preserved=True))
        c=copy.deepcopy(old)
        c.update(run_id=cfg['successor_run_id'],output_root=str(run_root),
                 diagnostic_protocol=cfg['diagnostic_protocol'],diagnostic_target_nodes=cfg['diagnostic_target_nodes'],
                 inherited_closed_returns=[str(closed)],predecessor_handoff_ref=evidence_ref)
        require(c['run_id']!=old['run_id'],'successor needs distinct run identity')
        from diagnostic_protocol import target_nodes
        targets=set(target_nodes(c))
        learning=read(state['learning_ref'])
        for phase in learning['phases']:
            require(phase['rollout_scope']=='node' and set(phase['target_nodes'])==targets,
                    'diagnostic targets must cover every configured training phase')
        # Rebind executable closures to this release; numerical precision and
        # original captured actions remain the original immutable references.
        from configure_release import bindings
        b=bindings(root/'bindings',old['precision_policy'].get('persistent_family_ref'))
        for key in ('reward_evaluation','node_evaluation','finite_data_binding','bounded_capture_binding',
                    'finite_capture_binding_ref','finite_diagnostic_binding','finite_diagnostic_config_ref','diagnostic_release_ref'):
            c[key]=b[key]
        c['selection']['reward_evaluation']=c['reward_evaluation']
        c['resident']['run_id']=c['run_id']
        c['checkpoint_cache_root']=str(run_root/'checkpoint-inventory')
        c['archive']['checkpoint_cache_root']=c['checkpoint_cache_root']
        from stage import pack
        bundle=root/'source-release'
        if not (bundle/'BUNDLE.json').exists():pack(bundle,HERE/'runtime.json',source_only=True)
        c['source_bundle_ref']=pin(bundle/'BUNDLE.json')
        from transport import verify_execution_bundle
        verify_execution_bundle(read(c['source_bundle_ref']))
        ref=write(root/'ITERATION.json',c)
        migrated=successor_state(state,training_ref,ref,evidence_ref)
        from storage import configure_scope,commit
        configure_scope(run_root,c['client_storage_bytes'],c['run_id']+':client',
                        Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        noncheckpoint_limit_bytes=c.get('client_noncheckpoint_limit_bytes'))
        cursor=run_root/'CURSOR.json'
        if cursor.exists():require(read(cursor)==migrated,'successor cursor advanced or differs')
        else:commit(cursor,migrated)
        return write(root/'PREPARED.json',dict(config_ref=ref,cursor_ref=pin(cursor),
                    evidence_ref=evidence_ref,model_calls=0,dispatch=False))


def rejected_proof(cfg):
    """Closed failed numeric run is distinct from a pre-load startup failure."""
    proof=read(cfg['terminal_proof_ref'])
    require(proof['all_owners_terminal'] and proof['restored'] and
            proof['optimizer_steps']==0 and proof['training_admission'] is False,
            'failed numerical predecessor must be terminal and restored')
    require(proof['controller']['id']==cfg['predecessor_controller'],'wrong numerical predecessor')
    for role in ('controller','actor','watchdog'):
        actual=terminal(proof[role]['id'],cfg['predecessor_host'])
        require(actual['state']==proof[role]['state'],'closed predecessor state changed')
    terminal(cfg['predecessor_client'])
    return proof


def undispatched_proof(cfg):
    old=read(cfg['predecessor_config_ref']);root=Path(old['output_root'])
    state=read(cfg['predecessor_cursor_ref'])
    require(state['config_ref']==pin(cfg['predecessor_config_ref']) and
            Path(cfg['predecessor_cursor_ref']['path'])==root/'CURSOR.json',
            'predecessor cursor/config differs')
    client=terminal(cfg['predecessor_client'])
    pending=state.get('pending_stage') or {}
    require(client['state']['ExitCode']!=0 and state['phase']=='parity_capture' and
            state['step']==0 and state['resume_ref'] is None and
            pending.get('stage')=='capture' and pending.get('purpose')=='base_parity' and
            not state.get('consumed_cohorts') and state.get('cohort_sequence',0)==0,
            'only failed undispatched base capture may hand off')
    mailbox=root/'mailbox'/('%04d-capture'%state['sequence'])/'transport'
    if cfg['schema']=='closed-pre-reservation-capture-handoff-v1':
        proof=read(cfg['terminal_proof_ref'])
        require(proof['schema']=='failed-capture-terminal-proof-v1' and
                proof['all_owners_terminal'] and proof['restored']['config_exact'] and
                proof['optimizer_updates']==0 and proof['cohort_reserved'] is False and
                proof['cohort_requests']==0,'pre-reservation terminal proof required')
        for role in ('controller','actor','watchdog'):
            actual=terminal(proof[role]['id'],'npu')
            require(actual['state']==proof[role]['state'],'capture terminal state changed')
        from registry_custody import facts
        ledger=facts(old['registry']['path'])
        require(ledger=={k:[tuple(r) for r in v] for k,v in proof['after'].items()},'failed capture accounting not accepted')
        return dict(proof_ref=cfg['terminal_proof_ref'],client=client,remote_dispatched=True,
            previous_cost=state['charged_tokens_upper'],optimizer_updates=0)
    require((mailbox/'CLOSURE-CHECK.json').is_file() and
            not any((mailbox/n).exists() for n in ('TRANSPORT.json','REQUEST.json','START-RESULT.json')),
            'remote dispatch may have occurred; reconcile original request')
    from registry_custody import facts
    ledger=facts(old['registry']['path'])
    require(all(r[2] in ('known','closed_unobserved') for r in ledger['spend']),
            'unknown registry request prohibits handoff')
    return dict(client=client,predecessor_cursor_ref=pin(cfg['predecessor_cursor_ref']),
        closure_failure_ref=pin(mailbox/'CLOSURE-CHECK.json'),remote_dispatched=False,
        previous_cost=state['charged_tokens_upper'],optimizer_updates=0,
        ledger_sha256=hashlib.sha256(json.dumps(ledger,sort_keys=True).encode()).hexdigest())


def prepare_undispatched(config_path):
    """Version the reward and CPU startup fixes before any training update."""
    import fcntl
    from storage import configure_scope,commit,environment
    from release_completion import require_closed_release
    from configure_release import bindings
    from transport import restore_locations,verify_execution_bundle
    from next_input import revise_unstarted_p_reward
    from stage import pack
    cfg=read(config_path);old=read(cfg['predecessor_config_ref'])
    root=Path(cfg['output_root']);old_root=Path(old['output_root'])
    with (old_root/'iteration.guard').open('ab') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof=undispatched_proof(cfg);state=read(cfg['predecessor_cursor_ref'])
        release=Path(cfg['release']);require_closed_release(release)
        require(HERE==release/'tree/parents/jev-system-rl/implementation','exact successor release required')
        for ref in cfg['inherited_closed_returns']:restore_locations(Path(pin(ref)['path']).parent)
        environment(root,old['client_storage_bytes'])
        evidence=write(root/'PREDECESSOR-TERMINAL.json',proof)
        revised=revise_unstarted_p_reward(state['input_ref'],root/'reward-revision')
        run_root=Path(cfg['successor_output_root'])
        require(run_root!=old_root and run_root.parent==old_root.parent,'distinct sibling run required')
        c=copy.deepcopy(old);c.update(run_id=cfg['successor_run_id'],output_root=str(run_root),
            predecessor_handoff_ref=evidence,
            inherited_closed_returns=[str(Path(r['path']).parent) for r in cfg['inherited_closed_returns']])
        require(c['run_id']!=old['run_id'],'distinct run identity required')
        c['budget']['prior_tokens_upper']=state['charged_tokens_upper']
        b=bindings(root/'bindings',c['precision_policy']['persistent_family_ref'])
        for key in ('reward_evaluation','node_evaluation','finite_data_binding','bounded_capture_binding',
                    'finite_capture_binding_ref','finite_diagnostic_binding','finite_diagnostic_config_ref',
                    'diagnostic_release_ref','named_rpc_overlay'):c[key]=b[key]
        c['selection']['reward_evaluation']=c['reward_evaluation']
        c['resident']['run_id']=c['run_id']
        c['checkpoint_cache_root']=str(run_root/'checkpoint-inventory')
        c['archive']['checkpoint_cache_root']=c['checkpoint_cache_root']
        package=copy.deepcopy(read(c['training_package_template']))
        package['learning']=revised['learning_ref']
        if cfg.get('engineering_update_sequence'):
            from checkpoint_budget import calculate
            from dependencies import JEV
            from configure_release import capture_capacity
            from diagnostic_storage import plan as storage_plan
            from unified_config import bind
            workflow=c['unified_workflow']
            workflow.update(max_optimizer_updates=cfg['engineering_update_sequence']['updates'],
                engineering_update_sequence=copy.deepcopy(cfg['engineering_update_sequence']))
            old_budget=read(workflow['checkpoint_budget_ref'])
            budget=copy.deepcopy(read(old_budget['config_ref']))
            updates=workflow['max_optimizer_updates']
            budget.update(workflow_kind='finite_update',max_optimizer_updates=updates,
                max_source_tasks=max(8*updates+12,budget['max_source_tasks']),
                resident_max_commands=max(8*updates+12,budget.get('resident_max_commands',0)))
            capture=c['diagnostic_logits_capture'];capacity=capture_capacity(c,capture)
            payload=storage_plan(capture,capacity,package.get('diagnostic_selection',{}).get('max_records',6),1+updates)
            for role,size in payload['roles'].items():
                budget['noncheckpoint_envelopes'][role]['complete_diagnostic_logits']=size
            br=write(root/'BUDGET-INPUT.json',budget);budget['_config_path']=br['path']
            workflow['checkpoint_budget_ref']=write(root/'BUDGET.json',calculate(budget,HERE/'worker_dp8.py',JEV/'src/multimodal_jev/context_training_checkpoint.py'))
            bind(dict(iteration=c,training_package=package),'finite_engineering',root/'workflow')
            write(root/'DIAGNOSTIC-STORAGE.json',payload)
        c['training_package_template']=write(root/'TRAINING-PACKAGE.json',package)
        bundle=root/'source-release'
        if not (bundle/'BUNDLE.json').exists():pack(bundle,HERE/'runtime.json',source_only=True)
        c['source_bundle_ref']=pin(bundle/'BUNDLE.json');verify_execution_bundle(read(c['source_bundle_ref']))
        ref=write(root/'ITERATION.json',c)
        migrated=copy.deepcopy(state)
        migrated['reservations'][state['pending_stage']['key']].update(
            status='failed_pre_reservation' if proof.get('remote_dispatched') else 'failed_pre_dispatch',terminal_evidence_ref=evidence)
        for key in ('parity_input_ref','witness_ref','parity_capture','closed_diagnostic',
                    'raw_ref','runtime_ref','parity_acceptance'):migrated.pop(key,None)
        migrated.update(config_ref=ref,input_ref=revised['input_ref'],learning_ref=revised['learning_ref'],
            phase='endpoint',pending_stage=None,predecessor_handoff_ref=evidence)
        if cfg.get('reuse_endpoint_witness'):
            # Only unchanged actual policy/profile/precision may reuse the
            # closed HTTP witness. The new owner still checks its live service.
            previous_release=Path(read(cfg['predecessor_launch_intent_ref'])['identity']['release_ref']['path']).parent
            old_precision=read(old['named_rpc_overlay']['precision_ref'])
            require(old_precision==read(c['named_rpc_overlay']['precision_ref']),
                    'endpoint reuse requires unchanged precision overlay')
            for name in ('service_bootstrap.py','rollout_owner.py','provider_session.py','policy_identity.py'):
                require(pin(previous_release/'tree/parents/jev-system-rl/implementation'/name)['sha256']==
                        pin(HERE/name)['sha256'],'endpoint implementation changed: '+name)
            witness=read(state['witness_ref']);new_input=read(revised['input_ref'])
            require(witness['policy_ref']==new_input['policy']['current_ref'] and
                    witness['profile_ref']==new_input['capture']['profile_ref'],
                    'endpoint policy/profile changed')
            migrated.update(phase='parity_capture',witness_ref=pin(state['witness_ref']))
        configure_scope(run_root,c['client_storage_bytes'],c['run_id']+':client',
            Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            noncheckpoint_limit_bytes=c.get('client_noncheckpoint_limit_bytes'))
        cursor=run_root/'CURSOR.json'
        if cursor.exists():require(read(cursor)==migrated,'successor cursor advanced or differs')
        else:commit(cursor,migrated)
        return write(root/'PREPARED.json',dict(config_ref=ref,cursor_ref=pin(cursor),
            evidence_ref=evidence,reward_revision_ref=revised['evidence_ref'],model_calls=0,dispatch=False))


def prepare_rejected(config_path):
    """New precision candidate uses fresh behavior, same original P task/input.

    Reuse CPU assets and source configuration, never reuse old behavior for a
    changed numerical implementation. Original iteration owns all model work.
    """
    import fcntl
    from storage import configure_scope,commit,environment
    from transport import restore_locations,verify_execution_bundle
    from release_completion import require_closed_release
    from configure_release import bindings
    from registry_custody import facts
    from stage import pack
    cfg=read(config_path);old=read(cfg['predecessor_config_ref'])
    old_root=Path(old['output_root']);root=Path(cfg['output_root'])
    with (old_root/'iteration.guard').open('ab') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(cfg['predecessor_cursor_ref']);proof=rejected_proof(cfg)
        require(state['config_ref']==pin(cfg['predecessor_config_ref']) and
                state['phase']=='done' and state['step']==0 and state['pending_stage'] is None and
                state['resume_ref'] is None,'only settled zero-update diagnostic may hand off')
        outcome=read(old_root/'OUTCOME.json')
        require(outcome['status']=='DIAGNOSTIC_REJECTED' and
                state['charged_tokens_upper']==proof['charged_tokens_upper'],'failure or cost differs')
        returns=cfg['inherited_closed_returns']
        for ref in returns:restore_locations(Path(pin(ref)['path']).parent)
        training_ref=pin(cfg['training_input_ref']);training=read(training_ref)
        require(training_ref==state['input_ref'] and not training.get('diagnostic_evidence_only') and
                not training.get('synthetic_fixture'),'preserve actual P training input')
        require(training['learning_config_ref']==state['learning_ref'] and
                training['policy']['current_ref']==state['policy_ref'],'scientific task changed')
        ledger=facts(old['registry']['path'])
        require(all(r[2] in ('known','closed_unobserved') for r in ledger['spend']),'unknown requests prohibit successor')
        release=Path(cfg['release']);require_closed_release(release)
        require(HERE==release/'tree/parents/jev-system-rl/implementation','exact candidate release required')
        run_root=Path(cfg['successor_output_root'])
        require(run_root!=old_root and run_root.parent==old_root.parent,'distinct sibling run required')
        environment(root,old['client_storage_bytes'])
        evidence=write(root/'PREDECESSOR-TERMINAL.json',dict(terminal_proof_ref=cfg['terminal_proof_ref'],
            predecessor_cursor_ref=pin(cfg['predecessor_cursor_ref']),previous_cost=state['charged_tokens_upper'],
            old_behavior_reused=False,model_calls=0))
        c=copy.deepcopy(old)
        c.update(run_id=cfg['successor_run_id'],output_root=str(run_root),
            predecessor_handoff_ref=evidence,inherited_closed_returns=[str(Path(r['path']).parent) for r in returns])
        c['budget']['prior_tokens_upper']=state['charged_tokens_upper']
        c['precision_policy']={'persistent_family_ref':pin(cfg['precision_ref'])}
        b=bindings(root/'bindings',cfg['precision_ref'])
        for key in ('reward_evaluation','node_evaluation','finite_data_binding','bounded_capture_binding',
                    'finite_capture_binding_ref','finite_diagnostic_binding','finite_diagnostic_config_ref',
                    'diagnostic_release_ref','named_rpc_overlay'):
            c[key]=b[key]
        c['selection']['reward_evaluation']=c['reward_evaluation']
        c['resident']['run_id']=c['run_id']
        c['checkpoint_cache_root']=str(run_root/'checkpoint-inventory')
        c['archive']['checkpoint_cache_root']=c['checkpoint_cache_root']
        package=copy.deepcopy(read(c['training_package_template']))
        package['communication']=read(cfg['communication_ref'])
        c['training_package_template']=write(root/'TRAINING-PACKAGE.json',package)
        bundle=root/'source-release'
        if not (bundle/'BUNDLE.json').exists():pack(bundle,HERE/'runtime.json',source_only=True)
        c['source_bundle_ref']=pin(bundle/'BUNDLE.json');verify_execution_bundle(read(c['source_bundle_ref']))
        ref=write(root/'ITERATION.json',c)
        migrated=copy.deepcopy(state)
        for key in ('closed_diagnostic','diagnostic_failure_ref','raw_ref','runtime_ref','parity_capture',
                    'parity_input_ref','parity_acceptance','witness_ref'):
            migrated.pop(key,None)
        migrated.update(config_ref=ref,phase='endpoint',predecessor_handoff_ref=evidence)
        configure_scope(run_root,c['client_storage_bytes'],c['run_id']+':client',
            Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            noncheckpoint_limit_bytes=c.get('client_noncheckpoint_limit_bytes'))
        cursor=run_root/'CURSOR.json'
        if cursor.exists():require(read(cursor)==migrated,'successor advanced or changed')
        else:commit(cursor,migrated)
        return write(root/'PREPARED.json',dict(config_ref=ref,cursor_ref=pin(cursor),
            evidence_ref=evidence,model_calls=0,dispatch=False))


def launch(args):
    """Reuse the fixed launcher's tool/mount recipe; run the same controller."""
    from launch_fixed import require_closed_release, storage_admission, load_profile, save, IMAGE
    cfg=read(args.handoff_config);build=Path(args.build).resolve();out=Path(args.output).resolve()
    require_closed_release(build)
    require(Path(cfg['release']).resolve()==build and Path(cfg['output_root']).resolve()==out,
            'handoff release/output differs from explicit launch')
    profile,_,_=load_profile(args.profile);storage_admission(profile,out.parent)
    old=read(cfg['predecessor_config_ref']);intent=read(cfg['predecessor_launch_intent_ref'])
    terminal(cfg['predecessor_client'])
    if cfg['schema'] in ('undispatched-p-reward-handoff-v1','closed-pre-reservation-capture-handoff-v1'):undispatched_proof(cfg)
    else:
        plan=read(Path(old['output_root'])/'step-000000/diagnostic/TRANSPORT.json')
        if cfg['schema']=='rejected-diagnostic-training-handoff-v1':rejected_proof(cfg)
        else:probe_remote(plan,cfg['predecessor_controller'])
    argv=list(intent['argv']);idx=argv.index(IMAGE);argv=argv[:idx+1]
    name='omni-handoff-client-'+hashlib.sha256((str(out)+str(args.execute)).encode()).hexdigest()[:24]
    argv[argv.index('--name')+1]=name
    identity=intent['identity']
    previous=Path(identity['build']) if 'build' in identity else Path(identity['release_ref']['path']).parent
    namespace='/home/xingchao/workspace/omni/delivery/system-rl-transition-v1'
    require(identity.get('mode',old.get('execution_mode'))==args.mode,'handoff execution mode changed')
    for i,item in enumerate(argv):
        if item=='--mount':
            mount=argv[i+1]
            if mount.startswith('type=bind,src='+str(previous)+'/tree/'):
                argv[i+1]=mount.replace('src='+str(previous)+'/tree/','src='+str(build)+'/tree/',1)
    # Predecessor source/artifacts stay readable. Only the original accounting
    # directory remains writable so its single ledger/guard continues unchanged.
    old_launch=Path(cfg['predecessor_launch_intent_ref']['path']).parent
    for i,item in enumerate(argv):
        if item=='--mount' and argv[i+1]=='type=bind,src='+str(old_launch)+',dst='+str(old_launch):
            argv[i+1]+=',readonly'
    registry_parent=Path(old['registry']['path']).parent
    additions=[]
    for path,readonly in ((build,True),(out,False),(registry_parent,False)):
        additions+=['--mount','type=bind,src='+str(path)+',dst='+str(path)+(',readonly' if readonly else '')]
    cfg_path=Path(args.handoff_config).resolve()
    if not cfg_path.is_relative_to(out):
        additions+=['--mount','type=bind,src='+str(cfg_path)+',dst='+str(cfg_path)+',readonly']
    existing={argv[i+1] for i,v in enumerate(argv) if v=='--mount'}
    additions=[part for i in range(0,len(additions),2)
               if additions[i+1] not in existing for part in additions[i:i+2]]
    idx=argv.index(IMAGE);argv[idx:idx]=additions
    script=str(build/'tree/parents/jev-system-rl/implementation/training_handoff.py')
    argv+=['-B',script,'--config',str(cfg_path)]
    if args.execute:argv.append('--execute')
    receipt=out/('EXECUTE-INTENT.json' if args.execute else 'PREPARE-INTENT.json')
    identity=dict(config_ref=pin(cfg_path),release_ref=pin(build/'ASSEMBLY.json'),execute=args.execute)
    if receipt.exists():
        require(read(receipt)['identity']==identity,'handoff launch identity changed')
        actual=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
        print(actual['Id']);return
    if args.execute:require((out/'PREPARED.json').is_file(),'prepare successor before execution')
    out.mkdir(parents=True,exist_ok=True)
    save(receipt,dict(identity=identity,argv=argv))
    result=subprocess.run(argv,text=True,capture_output=True)
    save(out/('EXECUTE-LAUNCH.json' if args.execute else 'PREPARE-LAUNCH.json'),dict(
        cid=result.stdout.strip(),returncode=result.returncode,stderr=result.stderr))
    require(result.returncode==0,'fixed handoff launch failed; reconcile original intent')
    print(result.stdout.strip())


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args();cfg=read(a.config);root=Path(cfg['output_root'])
    if a.execute:
        ready=read(root/'PREPARED.json')
        require(read(ready['config_ref'])['predecessor_handoff_ref']==ready['evidence_ref'],'prepared lineage drift')
        from iteration import run
        print(json.dumps(run(ready['config_ref']['path'])))
    else:print(json.dumps(prepare(a.config)))


if __name__=='__main__':main()
