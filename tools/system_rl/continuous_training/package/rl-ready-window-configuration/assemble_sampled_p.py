"""Compile the formal Clock continuation; iteration.run remains the only controller."""
import argparse, copy
from pathlib import Path


def read_screen(manifest, output):
    """Exercise the training reader before the screening devices are released."""
    from run_package import pin, read, write
    from dependencies import require
    from next_input import bind_screen_sampling
    from node_screen_selection import inherited_cost, selection
    import hashlib
    import base64
    import wave

    cfg=read(manifest)
    ref=pin(cfg['screen_result_path']);screen=read(ref)
    request_ref=pin(cfg['screen_request_ref']);request=read(request_ref)
    require(screen['request_ref']==request_ref,'screen request differs from the training manifest')
    tokens=inherited_cost(screen,request_ref,request)
    require(tokens<=request['budget']['tokens_upper'],'screen cost exceeds its recorded envelope')
    source=bind_screen_sampling(copy.deepcopy(read(request['input_ref'])),screen)
    pool=read(screen['screen_ref'])
    draws,stats=selection(pool,source['sampling'])
    spec=read(cfg['workflow_ref'])['configuration']['sampled_training']
    require(0<len(draws)<=spec['sample_count'],'screen exceeds the configured training cap')
    for draw in draws:
        record=draw['record'];o=read(draw['recording_origin_ref'])
        packet=read(record['online']['node_input_ref'])
        labels=record['offline_supervision']['labels']
        require(labels['gold_intent_id'] in packet['actual_returned_ranked_intent_ids']
                and o['domain']==record['domain'],'gold label or recording domain differs from served packet')
        audio=pin(o['audio_ref'])
        with wave.open(audio['path'],'rb') as wav:
            require(wav.getnchannels()==1 and wav.getnframes()>0 and wav.getsampwidth()==2
                    and wav.getframerate()==16000,'invalid selected recording')
        part,=[item for msg in packet['wire']['messages'] if isinstance(msg['content'],list)
               for item in msg['content'] if item.get('type')=='input_audio']
        require(hashlib.sha256(base64.b64decode(part['input_audio']['data'],validate=True)).hexdigest()
                ==audio['sha256'],'selected recording differs from the served audio')
    return write(Path(output)/'SCREEN-CONSUMED.json',dict(screen_ref=ref,rows=len(draws),
                 cells=stats['cells'],actual_screen_tokens=tokens))


def assemble(manifest, output, *, bind_live):
    from run_package import pin, read, write
    from config_profile import compile_profile
    from dependencies import require
    from sampled_training import specification, window_capacity
    from formal_actor_transport import settings as actor_settings
    cfg=read(manifest);root=Path(output);root.mkdir(parents=True,exist_ok=True)
    data=read(cfg['screen_request_ref']);base=read(data['config_ref'])
    start=read(data['input_ref']);spec=read(cfg['workflow_ref'])['configuration']['sampled_training']
    screen=read(cfg['screen_config_ref']);check=read(cfg['actor_check_ref'])
    if cfg.get('actor_batch_ref'):
        batch=read(pin(cfg['actor_batch_ref']))
        micro=batch['microbatch_per_card'];devices=check['devices'];group_source=read(cfg['branch_input_ref']) if cfg.get('branch_input_ref') else start
        g=read(group_source['learning_config_ref'])['kernel']['G']
        acc=batch.get('gradient_accumulation_steps',1)
        require(type(acc) is int and acc>0 and not set(batch)-{'microbatch_per_card','arithmetic','gradient_accumulation_steps'} and
                ('arithmetic' not in batch or batch['arithmetic']=='single_clip_compact_qo_v1') and type(micro) is int and micro>0 and
                len(devices)==len(set(devices))>0 and type(g) is int and g>0 and len(devices)*micro*acc%g==0,
                'global actor batch must contain whole rollout groups')
        spec=dict(spec,queries_per_cohort=len(devices)*micro*acc//g)
    initial=pin(start['native']['restore_checkpoint_ref'])
    require(initial['sha256']==check['restore_checkpoint_sha256'] and check['prior_step']==read(initial)['step']
            and check['check_only'] and check['stage_only'] and check['formal_sampled']
            and read(start['policy']['current_ref'])['step']==check['prior_step']
            and start['policy']['current_ref']==data['policy_ref'] and spec['sample_count']==screen['sampling']['sample_count'],
            'the finite actor, DATA request and official sample do not share the initial policy and scope')
    branch=None
    if cfg.get('branch_input_ref'):
        from actor_start import compile_start
        require(not cfg.get('branch_plan_ref') and cfg.get('continuous_deployment_ref') and cfg.get('branch_actor_source'),
                'branch needs its own formal resident configuration')
        start=read(cfg['branch_input_ref'])
        branch=compile_start(start['branch_initial_ref'],cfg['run_id'])
        require(branch['actor_batch']==batch and start['policy']['current_ref']==read(branch['branch_ref'])['policy_ref']
                and read(branch['branch_ref'])['dataset_ref']==read(read(branch['branch_ref'])['parent_cursor_ref'])['sampled_dataset_ref'],
                'branch current policy, batch or dataset changed')
    vals=copy.deepcopy(cfg['values'])
    rollout=copy.deepcopy(cfg['rollout'])
    sealed=read(cfg['screen_result_path']) if bind_live or (cfg.get('sample_size_from_seal')
                and Path(cfg['screen_result_path']).is_file()) else None
    if sealed is not None and cfg.get('sample_size_from_seal'):
        sampling=read(sealed['sampling_config_ref'])
        selected=sampling['sample_count']
        require(type(selected) is int and 0<selected<=spec['sample_count'],
                'sealed count exceeds the configured maximum or is empty')
        from node_screen_selection import inherited_cost
        inherited_cost(sealed,pin(cfg['screen_request_ref']),data)
        spec=dict(spec,sample_count=selected)
    lanes=vals['FORMAL_ROLLOUT_LANES']
    if bind_live:
        ready=read(cfg['live_provider_ready_path'])
        roster=ready['providers']
        handoff=read(ready['handoff_ref'])
        if 'lanes' in handoff:
            ids={}
            for lane,sha in handoff['lanes'].items():
                receipt=pin(Path(ready['handoff_ref']['path']).parent/(lane+'-READY.json'))
                require(receipt['sha256']==sha,'resident deployment changed')
                ids[lane]=read(receipt)['container_id']
        else:
            ids={lane:row['container_id'] for lane,row in handoff['rollout']['ready'].items()}
        require(set(roster)==set(lanes)==set(ids), 'resource handoff lane roster differs')
        for k,p in roster.items():
            provider=read(p)
            require(provider['lane']==k and read(provider['service_ready'])['container_id']==ids[k],
                    'different provider handoff')
        require(sealed['request_ref']==pin(cfg['screen_request_ref']) and sealed['complete']
                and sealed['node']=='P' and sealed['fixture'] is False,'different DATA completion')
    else:
        roster={k:str(Path(cfg['live_provider_ready_path']).parent/k/'PROVIDER.json') for k in lanes}
    rollout['provider_refs']=roster
    train=copy.deepcopy(check)
    if cfg.get('actor_batch_ref'):train['actor_batch']=dict(batch)
    train.update(stage_only=False,check_only=False,rollout_lanes=lanes,training_memory_gib=vals['TRAIN_MEMORY_GIB'])
    if branch:
        train['lora_row_geometry']=dict(train['lora_row_geometry'],adapter_rank=branch['training']['lora_rank'])
        train.update(initial_actor=branch,source_bundle=cfg['branch_actor_source'],lineage=branch['lineage'],prior_step=0)
        for key in ('restore_run','restore_checkpoint_sha256'):train.pop(key,None)
    from launch_retained_hf_fsdp import training_memory_bytes
    training_memory_bytes(train)
    mounts=train['capture_mounts'];mounts['root']=cfg['actor_capture_root']
    mounts['logical_roots'].update(cfg['actor_input_roots']);mounts['formal_input_roles']=list(cfg['actor_input_roots'])
    release=Path(cfg['release'])
    parents=release/'tree/parents'
    for rel in mounts['overlays']:
        mounts['overlays'][rel]=pin(parents/rel)['sha256']
    continuous=None
    if cfg.get('continuous_deployment_ref'):
        deployment=read(cfg['continuous_deployment_ref'])
        from verify_runtime_sources import training_snapshot
        lock=pin(deployment['source_lock_ref']);precision=pin(deployment['precision_lock_ref'])
        snap=training_snapshot(lock,precision['sha256'])
        policy=read(deployment['policy_ref'])
        if 'experiment_sha256' in policy or cfg.get('experiment_ref'):
            from paired_experiment import load, check_actor
            exp = load(pin(cfg['experiment_ref'])['path'], policy['experiment_sha256'])
            check_actor(exp, branch, policy)
            require(train['devices'] == exp['training_devices'] and train['training_memory_gib'] == exp['training_memory_gib']
                    and set(lanes) == set(exp['lanes']), 'training resource assignment differs from experiment')
            if bind_live:
                require(handoff.get('experiment_sha256') == policy['experiment_sha256'],
                        'live inference deployment belongs to a different experiment')
            base['experiment_ref'] = cfg['experiment_ref']
            train['experiment_sha256'] = policy['experiment_sha256']
            if 'publication_norm_atol' in exp:
                train['publication_norm_atol'] = exp['publication_norm_atol']
        require(pin(deployment['policy_ref'])['sha256']==snap['policy_sha256'] and
                read(precision)['accepted_verdict']['passed'] and cfg.get('actor_batch_ref') and
                deployment['protocol']=='formal-resident-v1' and
                spec['queries_per_cohort']==policy['acceptance']['recordings_per_step'] and
                read(start['learning_config_ref'])['kernel']['G']==policy['acceptance']['episodes_per_recording'],
                'continuous source, numerical qualification or formal batch geometry differs')
        continuous=dict(protocol=deployment['protocol'],run_id=cfg['run_id'],source_lock_ref=lock,
            precision_lock_sha256=precision['sha256'],precision_lock_ref=precision,**{k:deployment[k] for k in
            ('max_idle_seconds','poll_seconds','max_commands','input_projection_source_ref','command_source_ref')})
        for key,name in [('command_source_ref','formal_command.py'),('input_projection_source_ref','resident_inputs.py')]:
            require(pin(continuous[key])['sha256']==snap['roles']['controller']['rl-ready-window-configuration/source/'+name],
                    'resident command/input source is not bound by training snapshot')
        parser=policy['screened_p_parser_member']
        require(pin(parents/parser['path'])['sha256']==parser['sha256'],'packaged P parser differs from training policy')
        rollout['screened_p_parser_member']=parser
        rollout.update(policy['rollout_dispatch'])
        rt=train['generation']['runtime'];rt['reference_batch']=policy['reference_batch'];rt['actor_fields']=policy['actor_fields']
        for name in ('row_encoding', 'group_preparation', 'update_timing'):
            if name in policy:
                rt[name] = policy[name]
        rt['sources']['reference_batch_source']=deployment['reference_batch_member']
        for role,rel in rt['sources'].items():
            if role=='consumer_source':
                require(pin(parents/rel)['sha256']==policy['inherited_capture_members'][rel],
                        'inherited consumer differs from qualified training release: '+role)
            else:
                filename=Path(rel).name
                wanted={v for a in snap['roles'].values() for k,v in a.items() if Path(k).name==filename}
                require(wanted=={pin(parents/rel)['sha256']},'capture release differs from training snapshot: '+role)
            mounts['overlays'][rel]=pin(parents/rel)['sha256']
        rollout['continuous_training']=dict(protocol='apply-local-marker-v1',precision_lock_ref=precision)
        if 'activation_concurrency' in policy:
            rollout['continuous_training']['activation_concurrency'] = policy['activation_concurrency']
    train['generation']['update_token_upper']=cfg['actor_update_token_upper']
    vals.update(RESTORED_INPUT_PATH=cfg.get('branch_input_ref',data['input_ref'])['path'],RESTORED_CHECKPOINT_REF=None if branch else initial,
        RESIDENT_STARTUP_REF=data['endpoint_witness_ref'],FORMAL_DATA_PROGRAM_REF=base['program_template_ref'],
        CLOSED_DATA_PREDECESSOR_PATH=base['program_predecessor_output'],SAMPLE_COUNT=spec['sample_count'],
        QUERIES_PER_COHORT=spec['queries_per_cohort'],TRAIN_INITIAL_RUN=train['source_bundle'] if branch else train['restore_run'],
        TRAIN_SOURCE_BOOT=train['expected_boot'],TRAIN_STAGE_TEMPLATE_REF=write(root/'ACTOR.json',train),
        RESIDENT_PROVIDER_REF=roster[cfg['primary_lane']])
    profile=write(root/'STARTUP.local.json',dict(schema='omni-local-profile-v1',values=vals))
    compiled=read(compile_profile(cfg['template_ref']['path'],profile['path'],root/'resolution'))
    require(compiled['template_source_ref']==cfg['template_ref'],'formal template changed')
    base.update(read(compiled['resolved_ref']),formal_rollout=rollout)
    if continuous and 'dev_cadence' in policy:
        require(policy['dev_cadence']=='epoch_end','invalid sampled DEV cadence')
        # This release trains one node for the full sampled population.
        # A multi-phase run must place its promotion gates on epoch boundaries.
        learning=read(start['learning_config_ref'])
        require(len(learning['phases'])==1,'epoch DEV requires an epoch-aligned training phase')
        base['selection']['dev_cadence']=policy['dev_cadence']
        base['selection'].pop('dev_every_optimizer_steps',None)
    if continuous:
        continuous['deadline_epoch']=base['deadline_epoch']
        base['formal_actor']['continuous_actor']=continuous
        if 'transfer_policy' in policy:
            base['formal_actor']['transfer_policy'] = policy['transfer_policy']
        if 'capacity_interval_bytes' in policy:
            base['formal_actor']['capacity_interval_bytes'] = policy['capacity_interval_bytes']
        if branch:base['formal_actor']['initial_actor_run']=train['source_bundle']
        base['continuous_start_step']=train['prior_step']
    if 'heldout_runtime_inputs' in cfg:
        profiles=cfg['heldout_runtime_inputs']
        require(isinstance(profiles,dict) and set(profiles)<=set(('DEV','TEST')),'unknown heldout input split')
        base['program_execution']['heldout_runtime_inputs']={split:pin(ref) for split,ref in profiles.items()}
    if 'heldout_lifetime' in cfg:
        require(cfg['heldout_lifetime'] in ('template','iteration_deadline'),'unknown heldout program lifetime')
        base['program_execution']['heldout_lifetime']=cfg['heldout_lifetime']
    # Rebind only the signed release replacement of the capture driver. Preserve
    # scorer/module/paired/specification and every unaffected source pin.
    binding=copy.deepcopy(base['reward_evaluation'])
    require(binding==base['selection']['reward_evaluation'],'reward binding differs from selection')
    old=read(binding['source_closure']);new=[]
    for ref in old:
        replacement=cfg.get('release_capture_binding')
        if replacement and ref==replacement['prior_ref']:
            delivered=pin(parents/'fresh-cohort-successor/cohort_capture.py')
            require(delivered==replacement['release_ref'] and
                    Path(ref['path']).parts[-2:]==('fresh-cohort-successor','cohort_capture.py'),
                    'unexpected collection source binding')
            new.append(delivered)
        else:new.append(pin(ref))
    if cfg.get('release_capture_binding'):
        require(old.count(cfg['release_capture_binding']['prior_ref'])==1,'missing prior capture driver')
        binding['source_closure']=write(root/'REWARD-SOURCE-PINS.json',new)
        base['reward_evaluation']=binding;base['selection']['reward_evaluation']=binding
    base.update(run_id=cfg['run_id'],output_root=vals['FORMAL_RUN_ROOT'],
        initial_screen_result_path=cfg['screen_result_path'],resident=dict(enabled=False),
        newprocess_replay=dict(enabled=False),execution_mode='formal',formal_rl_enabled=True,
        checkpoint_cache_root=str(Path(vals['FORMAL_RUN_ROOT'])/'checkpoint-inventory'))
    require(base['run_id'].islower() and base['sampled_training']==spec,'formal sampled configuration drift')
    n=spec['queries_per_cohort'];step=train['prior_step']
    total=step+(spec['sample_count']+n-1)//n
    base['max_optimizer_updates']=total;base.pop('engineering_update_sequence',None)
    base['unified_workflow']=dict(read(cfg['workflow_ref'])['configuration'],max_optimizer_updates=total,
                                  sampled_training=spec)
    replicas=read(start['learning_config_ref'])['kernel']['G']
    unit=rollout.get('dispatch_unit','recording')
    require(unit in ('recording','training_batch') and type(rollout['episode_concurrency']) is int and
            1<=rollout['episode_concurrency']<=(replicas*n if unit=='training_batch' else replicas),
            'invalid formal rollout concurrency')
    calls=n*replicas
    from prompt_capacity import prompt_bound
    envelope=calls*(prompt_bound(start['budget'],'P')+rollout['window_capacity']['completion_tokens_upper_per_call'])
    base['stage_budgets']['capture']=dict(requests_upper=calls,tokens_upper=3*envelope)
    base['stage_budgets']['update']['tokens_upper']=cfg['actor_update_token_upper']
    require(base['authorization_ref']==data['authorization_ref'] and base['registry']==start['execution_registry']
            and base['system_policy_ref']==data['system_policy_ref'] and base['datasets']['TRAIN']==data['dataset_ref'],
            'DATA accounting or task identity drift')
    if branch:
        b=read(branch['branch_ref']);parent=read(b['parent_cursor_ref'])
        if cfg.get('dataset_start_mode') is not None:
            require(cfg['dataset_start_mode']=='new_epoch' and not cfg.get('closed_undispatched_ref'),
                    'new epoch requires independent branch initialization')
            base['dataset_start_mode']='new_epoch'
        require(cfg['run_id']!=read(parent['config_ref'])['run_id'] and
                not Path(base['output_root']).resolve().is_relative_to(Path(b['parent_cursor_ref']['path']).parent.resolve())
                and parent['screen']==sealed and read(b['dataset_ref'])['specification']==spec,
                'expanded branch must retain its original screen, dataset and independent run root')
        base.pop('initial_resume_ref',None)
        base.update(initial_input=cfg['branch_input_ref']['path'],branch_initial_ref=branch['branch_ref'],
                    trajectory=parent['trajectory_ref']['path'])
        base['budget']['prior_tokens_upper']=b['prior_tokens_upper']
        if cfg.get('closed_undispatched_ref'):
            closed=read(cfg['closed_undispatched_ref']);retired=read(closed['terminal']['undispatched_ref'])
            failed=read(retired['cursor_ref']);oldrun=read(retired['iteration_ref'])
            require(closed['status']=='released' and retired['business_requests']==retired['unknown_requests']==retired['optimizer_steps']==0
                    and failed['config_ref']==retired['iteration_ref'] and failed['step']==0 and
                    failed['branch_initial_ref']==branch['branch_ref'] and failed['sampled_consumed']==b['next_window_start']
                    and oldrun['output_root']!=base['output_root'] and failed['charged_tokens_upper']==retired['charged_tokens_upper']
                    and b['prior_tokens_upper']<=retired['charged_tokens_upper'],'unrelated failed capture or cost')
            base['budget']['prior_tokens_upper']=retired['charged_tokens_upper']
            base['closed_undispatched_ref']=cfg['closed_undispatched_ref']
            rollout['resource_session']['owner_suffix']='session-'+cfg['closed_undispatched_ref']['sha256'][:16]

    if cfg.get('branch_plan_ref'):
        from formal_actor_transport import recovery_run
        plan=read(cfg['branch_plan_ref']);parent=read(plan['parent_cursor_ref'])
        require(bind_live and continuous and plan['schema']=='independent-latency-input-plan-v1'
                and plan['checkpoint_ref']==parent['resume_ref'] and plan['policy_ref']==parent['policy_ref']
                and read(plan['dataset_ref'])['specification']==spec and parent['screen']==sealed
                and Path(base['output_root']).resolve() != Path(plan['parent_cursor_ref']['path']).parent.resolve()
                and not Path(base['output_root']).resolve().is_relative_to(Path(plan['parent_cursor_ref']['path']).parent.resolve()),
                'invalid independent branch')
        prev=read(parent['config_ref'])
        require(all(prev[k]==base[k] for k in ('datasets','registry','system_policy_ref','authorization_ref')),
                'branch science or authorization changed')
        step=plan['initial_step'];initial=parent['resume_ref']
        prior=str(recovery_run(initial,step))
        train.update(prior_step=step,source_bundle=prior,restore_run=prior,restore_checkpoint_sha256=initial['sha256'])
        base['formal_actor'].update(initial_actor_run=prior,stage_template_ref=write(root/'BRANCH-ACTOR.json',train))
        base.update(initial_input=parent['capture_source_input_ref']['path'],initial_resume_ref=initial,
                    branch_plan_ref=pin(cfg['branch_plan_ref']),continuous_start_step=step,trajectory=parent['trajectory_ref']['path'])
        oldroot=Path(plan['parent_cursor_ref']['path']).parent
        closed=sorted(p.parent for p in oldroot.glob('step-*/update/returned/CLOSED-COMMIT.json'))
        base['inherited_closed_returns']+=list(map(str,closed))
        base['max_optimizer_updates']=base['unified_workflow']['max_optimizer_updates']=step+(spec['sample_count']+n-1)//n
    specification(base,prior_step=step);actor_settings(base)
    probe=copy.deepcopy(start);probe.update(sampling=screen['sampling'],rollout_scope='node')
    window_capacity(base,probe,n)
    if bind_live:
        from formal_rollout import settings
        settings(base)
    return write(root/('ITERATION.json' if bind_live else 'PREVIEW-NOT-FOR-LAUNCH.json'),base)


if __name__=='__main__':
    import json,os,sys
    cli=argparse.ArgumentParser()
    for name,env in [('manifest','OMNI_TRAIN_MANIFEST'),('output','OMNI_TRAIN_OUTPUT')]:
        default=os.environ.get(env)
        cli.add_argument('--'+name,default=default,required=not default,help='defaults to '+env)
    cli.add_argument('--profile',type=Path,default=os.environ.get('OMNI_DEPLOYMENT_PROFILE') or None)
    cli.add_argument('--release',type=Path,default=os.environ.get('OMNI_TRAIN_RELEASE') or None,
                     help='assembled runtime for resolving a manifest template')
    mode=cli.add_mutually_exclusive_group()
    mode.add_argument('--bind-live',action='store_true')
    mode.add_argument('--verify-screen',action='store_true');a=cli.parse_args()
    if a.profile and not a.release:cli.error('template requires --release or OMNI_TRAIN_RELEASE')
    release=a.release or Path(json.loads(Path(a.manifest).read_bytes())['release'])
    sys.path.insert(0,str(release/'tree/parents/jev-system-rl/implementation'))
    if a.profile:
        from source.config_profile import compile_profile
        from run_package import read
        receipt=read(compile_profile(a.manifest,a.profile,Path(a.output)/'manifest-resolution'))
        a.manifest=receipt['resolved_ref']['path']
    if Path(json.loads(Path(a.manifest).read_bytes())['release']).resolve()!=release.resolve():
        cli.error('manifest and selected training runtime differ')
    result=read_screen(a.manifest,a.output) if a.verify_screen else assemble(a.manifest,a.output,bind_live=a.bind_live)
    print(json.dumps(result))
