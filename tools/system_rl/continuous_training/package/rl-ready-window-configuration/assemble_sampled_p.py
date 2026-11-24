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
        micro=batch['microbatch_per_card'];devices=check['devices'];g=read(start['learning_config_ref'])['kernel']['G']
        require(not set(batch)-{'microbatch_per_card','arithmetic'} and
                ('arithmetic' not in batch or batch['arithmetic']=='single_clip_compact_qo_v1') and type(micro) is int and micro>0 and
                len(devices)==len(set(devices))>0 and type(g) is int and g>0 and len(devices)*micro%g==0,
                'global actor batch must contain whole rollout groups')
        spec=dict(spec,queries_per_cohort=len(devices)*micro//g)
    initial=pin(start['native']['restore_checkpoint_ref'])
    require(initial['sha256']==check['restore_checkpoint_sha256'] and check['prior_step']==read(initial)['step']
            and check['check_only'] and check['stage_only'] and check['formal_sampled']
            and read(start['policy']['current_ref'])['step']==check['prior_step']
            and start['policy']['current_ref']==data['policy_ref'] and spec['sample_count']==screen['sampling']['sample_count'],
            'the finite actor, DATA request and official sample do not share the initial policy and scope')
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
        require(set(roster)==set(lanes)==set(handoff['rollout']['ready']), 'resource handoff lane roster differs')
        for k,p in roster.items():
            provider=read(p)
            require(provider['lane']==k and read(provider['service_ready'])['container_id']==
                    handoff['rollout']['ready'][k]['container_id'],'different provider handoff')
        require(sealed['request_ref']==pin(cfg['screen_request_ref']) and sealed['complete']
                and sealed['node']=='P' and sealed['fixture'] is False,'different DATA completion')
    else:
        roster={k:str(Path(cfg['live_provider_ready_path']).parent/k/'PROVIDER.json') for k in lanes}
    rollout['provider_refs']=roster
    train=copy.deepcopy(check)
    if cfg.get('actor_batch_ref'):train['actor_batch']=dict(batch)
    train.update(stage_only=False,check_only=False,rollout_lanes=lanes,training_memory_gib=vals['TRAIN_MEMORY_GIB'])
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
        require(pin(deployment['policy_ref'])['sha256']==snap['policy_sha256'] and
                read(precision)['accepted_verdict']['passed'] and cfg.get('actor_batch_ref') and
                deployment['protocol']=='formal-resident-v1' and
                spec['queries_per_cohort']==policy['acceptance']['recordings_per_step'] and
                read(start['learning_config_ref'])['kernel']['G']==policy['acceptance']['episodes_per_recording'],
                'continuous source, numerical qualification or formal batch geometry differs')
        continuous=dict(protocol=deployment['protocol'],run_id=cfg['run_id'],source_lock_ref=lock,
            precision_lock_sha256=precision['sha256'],**{k:deployment[k] for k in
            ('max_idle_seconds','poll_seconds','max_commands','input_projection_source_ref','command_source_ref')})
        for key,name in [('command_source_ref','formal_command.py'),('input_projection_source_ref','resident_inputs.py')]:
            require(pin(continuous[key])['sha256']==snap['roles']['controller']['rl-ready-window-configuration/source/'+name],
                    'resident command/input source is not bound by training snapshot')
        rt=train['generation']['runtime'];rt['reference_batch']=policy['reference_batch'];rt['actor_fields']=policy['actor_fields']
        rt['sources']['reference_batch_source']=deployment['reference_batch_member']
        for role,rel in rt['sources'].items():
            if role in ('consumer_source','consumer_wrapper'):
                require(pin(parents/rel)['sha256']==policy['inherited_capture_members'][rel],
                        'inherited consumer differs from qualified training release: '+role)
            else:
                filename=Path(rel).name
                wanted={v for a in snap['roles'].values() for k,v in a.items() if Path(k).name==filename}
                require(wanted=={pin(parents/rel)['sha256']},'capture release differs from training snapshot: '+role)
            mounts['overlays'][rel]=pin(parents/rel)['sha256']
        rollout['continuous_training']=dict(protocol='apply-local-marker-v1',precision_lock_ref=precision)
    train['generation']['update_token_upper']=cfg['actor_update_token_upper']
    vals.update(RESTORED_INPUT_PATH=data['input_ref']['path'],RESTORED_CHECKPOINT_REF=initial,
        RESIDENT_STARTUP_REF=data['endpoint_witness_ref'],FORMAL_DATA_PROGRAM_REF=base['program_template_ref'],
        CLOSED_DATA_PREDECESSOR_PATH=base['program_predecessor_output'],SAMPLE_COUNT=spec['sample_count'],
        QUERIES_PER_COHORT=spec['queries_per_cohort'],TRAIN_INITIAL_RUN=train['restore_run'],
        TRAIN_SOURCE_BOOT=train['expected_boot'],TRAIN_STAGE_TEMPLATE_REF=write(root/'ACTOR.json',train),
        RESIDENT_PROVIDER_REF=roster[cfg['primary_lane']])
    profile=write(root/'STARTUP.local.json',dict(schema='omni-local-profile-v1',values=vals))
    compiled=read(compile_profile(cfg['template_ref']['path'],profile['path'],root/'resolution'))
    require(compiled['template_source_ref']==cfg['template_ref'],'formal template changed')
    base.update(read(compiled['resolved_ref']),formal_rollout=rollout)
    if continuous:
        continuous['deadline_epoch']=base['deadline_epoch']
        base['formal_actor']['continuous_actor']=continuous
        base['continuous_start_step']=check['prior_step']
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
    n=spec['queries_per_cohort'];step=check['prior_step']
    total=step+(spec['sample_count']+n-1)//n
    base['max_optimizer_updates']=total;base.pop('engineering_update_sequence',None)
    base['unified_workflow']=dict(read(cfg['workflow_ref'])['configuration'],max_optimizer_updates=total,
                                  sampled_training=spec)
    replicas=read(start['learning_config_ref'])['kernel']['G']
    require(1 <= rollout['episode_concurrency'] <= replicas,'formal episode concurrency exceeds group size')
    calls=n*replicas
    from prompt_capacity import prompt_bound
    envelope=calls*(prompt_bound(start['budget'],'P')+rollout['window_capacity']['completion_tokens_upper_per_call'])
    base['stage_budgets']['capture']=dict(requests_upper=calls,tokens_upper=3*envelope)
    base['stage_budgets']['update']['tokens_upper']=cfg['actor_update_token_upper']
    require(base['authorization_ref']==data['authorization_ref'] and base['registry']==start['execution_registry']
            and base['system_policy_ref']==data['system_policy_ref'] and base['datasets']['TRAIN']==data['dataset_ref'],
            'DATA accounting or task identity drift')
    specification(base,prior_step=step);actor_settings(base)
    probe=copy.deepcopy(start);probe.update(sampling=screen['sampling'],rollout_scope='node')
    window_capacity(base,probe,n)
    if bind_live:
        from formal_rollout import settings
        settings(base)
    return write(root/('ITERATION.json' if bind_live else 'PREVIEW-NOT-FOR-LAUNCH.json'),base)


if __name__=='__main__':
    import json,sys
    cli=argparse.ArgumentParser();cli.add_argument('--manifest',required=True);cli.add_argument('--output',required=True)
    mode=cli.add_mutually_exclusive_group()
    mode.add_argument('--bind-live',action='store_true')
    mode.add_argument('--verify-screen',action='store_true');a=cli.parse_args()
    release=Path(json.loads(Path(a.manifest).read_bytes())['release'])
    sys.path.insert(0,str(release/'tree/parents/jev-system-rl/implementation'))
    result=read_screen(a.manifest,a.output) if a.verify_screen else assemble(a.manifest,a.output,bind_live=a.bind_live)
    print(json.dumps(result))
