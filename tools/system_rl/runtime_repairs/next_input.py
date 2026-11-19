"""Explicit policy/Clock migration, original fresh sampling/capture contracts."""
import copy
from pathlib import Path
from dependencies import require,ROOT,upstream,load
from policy_identity import digest
from run_package import read,pin,write


def revise_unstarted_p_reward(input_ref,output):
    """Explicit zero-update revision; historical rewards and actions stay sealed."""
    prior=read(input_ref);learning=copy.deepcopy(read(prior['learning_config_ref']))
    old_clock=read(prior['clock_state_ref']);consumer=copy.deepcopy(read(prior['consumer_config_ref']))
    require(old_clock['successful_optimizer_steps']==0 and
            all(not old_clock.get(k) for k in ('groups','attempts','validations')),
            'reward revision requires an unreserved zero-update learning Clock')
    require(old_clock['config_sha256']==digest(learning) and old_clock['config_version']==learning['version'],
            'reward revision predecessor config mismatch')
    previous_definition=(learning['scorer']['version'],learning['scorer']['metric_weights']['P'])
    require(previous_definition in (
            ('component-definitions-v2',{'coverage_at3':1.0}),
            ('component-definitions-v3-p-position',{'position_reward_v1':1.0})),
            'unexpected predecessor P reward')
    require(all(p['rollout_scope']=='node' and p['target_nodes']==['P'] and
                p['trainable_mask']=={n:n=='P' for n in ('P','S1','S2','D')}
                for p in learning['phases']),'reward revision is only for P')
    root=Path(output)
    learning['version']+='@p-position-'+pin(ROOT/'unified-learning/scoring.py')['sha256'][:12]
    learning['scorer']['version']='component-definitions-v3-p-position'
    learning['scorer']['metric_weights']['P']={'position_reward_v1':1.0}
    for phase in learning['task_loss']['phases'].values():
        phase['P_ranking']={'kind':'hold','value':0.0}
    # Consumer imports must bind the successor implementation as a complete
    # closure. Other source changes are recorded, never silently accepted.
    pins={n:pin(ROOT/'unified-learning'/n) for n in
          ('kernel.py','learning.py','dataset.py','scoring.py','schedule.py','fixtures.py')}
    pins.update({n:pin(ROOT/'system-objective-cpu'/n) for n in ('objective.py','runtime.py')})
    consumer['source_pins']=pins
    consumer['sealed_adapter_ref']=pin(ROOT/'training-consumer-adapter/adapter.py')
    lr=write(root/'LEARNING.json',learning);consumer['learning_config_ref']=lr
    task=upstream()
    from scorer_extension import install
    install(task)
    clock=task.create_clock(learning,consumer_config=consumer)
    cr=write(root/'CLOCK.json',clock.state);consumer['clock_state_ref']=cr
    consumer_ref=write(root/'CONSUMER.json',consumer)
    nxt=copy.deepcopy(prior)
    nxt.update(learning_config_ref=lr,clock_state_ref=cr,consumer_config_ref=consumer_ref)
    from runtime_builder import bind_business_executables
    business=copy.deepcopy(read(prior['capture']['business_config_ref']))
    bind_business_executables(business,root/'business')
    nxt['capture']['business_config_ref']=write(root/'business/BUSINESS.json',business)
    evidence=write(root/'REWARD-REVISION.json',dict(schema='unstarted-p-reward-revision-v1',
        predecessor_input_ref=pin(input_ref),predecessor_learning_ref=prior['learning_config_ref'],
        predecessor_clock_ref=prior['clock_state_ref'],learning_ref=lr,clock_ref=cr,
        consumer_ref=consumer_ref,business_ref=nxt['capture']['business_config_ref'],
        previous_source_pins=read(prior['consumer_config_ref'])['source_pins'],
        source_pins=pins,reward_by_position=[1.0,.75,.5],absent_reward=0.0,
        model_weights_changed=False,optimizer_updates=0,old_actions_reused=False))
    nxt['reward_revision_ref']=evidence
    return dict(input_ref=write(root/'INPUT.json',nxt),learning_ref=lr,clock_ref=cr,
                consumer_ref=consumer_ref,evidence_ref=evidence)


def bind_screen_sampling(source,screen):
    """Leave the bounded startup selector only on an authenticated full screen."""
    require(screen.get('complete') is True and screen.get('population')==119881,
            'formal sampling needs complete official TRAIN screen')
    pool=read(screen['screen_ref'])
    require(pool.get('population_complete') is True and pool.get('official_train_count')==119881
            and pool.get('synthetic_fixture') is False and pool.get('unknown_count',0)==0,
            'formal sampling requires closed full population')
    sampling=copy.deepcopy(read(screen['sampling_config_ref']))
    require(sampling['schema']=='dataset-sampling-config-v1' and sampling['purpose']=='screen'
            and sampling['split']=='TRAIN' and sampling['scope']=='system'
            and sampling['error_success']==dict(error=.8,success=.2), 'formal historical sampling contract')
    require(sampling['pool_version']==pool['version'] and
            sampling['system_version']==screen['system_version'],'formal sampling version differs')
    require(set(sampling['domain_mix'])=={'real','synthetic'} and
            sampling['format']=='separated-json-v1' and sampling['insufficient']=='fail',
            'explicit source sampling policy required')
    require(source.get('prepare_mode')!='finite_diagnostic_capture' and
            not source.get('diagnostic_evidence_only'),'diagnostic evidence cannot become training')
    transition=source.get('prepare_mode')=='bounded_train_probe'
    if transition:
        source['startup_sampling_lineage']=dict(prepare_mode=source['prepare_mode'],
            sampling=copy.deepcopy(source['sampling']),bounded_train_probe=source.get('bounded_train_probe'),
            phase_binding=source.get('phase_binding'))
        if source.get('formal_startup'):
            source['startup_sampling_lineage']['formal_startup'] = source.pop('formal_startup')
        source.pop('bounded_train_probe',None)
        source.pop('phase_binding',None)
    source['prepare_mode']='full_train_screen'
    difficulty=source.get('sampling',{}).get('node_difficulty')
    if difficulty is not None:
        require('node_difficulty' not in sampling or sampling['node_difficulty']==difficulty,
                'screen cannot silently replace node difficulty')
        sampling['node_difficulty']=copy.deepcopy(difficulty)
    source['sampling']=sampling
    source['screen_ref']=screen['screen_ref']
    source['sampling_config_ref']=pin(screen['sampling_config_ref'])
    return source


def migrate(input_ref,policy_ref,checkpoint_ref,output):
    prior=read(input_ref);checkpoint=read(checkpoint_ref);policy=read(policy_ref)
    learning=copy.deepcopy(read(prior['learning_config_ref']));state=copy.deepcopy(checkpoint['clock'])
    require(state['config_sha256']==digest(learning),'migration must authenticate original learning')
    version=policy['version'];learning['version']+='@'+version
    for key in ('behavior_policy_vector','current_policy_vector'):
        learning['kernel'][key]={n:version for n in ('P','S1','S2','D')}
    previous_step=read(prior['clock_state_ref'])['successful_optimizer_steps']
    before=[p for p in learning['phases'] if p['start_step']<=previous_step][-1]
    after=[p for p in learning['phases'] if p['start_step']<=state['successful_optimizer_steps']][-1]
    if after['rollout_scope']=='system' or before['id']!=after['id']:
        learning['adapter_routing']['frozen_upstream_adapter_vector']=copy.deepcopy(learning['kernel']['current_policy_vector'])
    else:
        require(after['frozen_upstream_release']==before['frozen_upstream_release'],'node frozen release changed inside phase')
    state.update(config_sha256=digest(learning),config_version=learning['version'])
    root=Path(output);learning_ref=write(root/'LEARNING.json',learning);clock_ref=write(root/'CLOCK.json',state)
    consumer=copy.deepcopy(read(prior['consumer_config_ref']));consumer.update(learning_config_ref=learning_ref,clock_state_ref=clock_ref)
    consumer_ref=write(root/'CONSUMER.json',consumer)
    nxt=copy.deepcopy(prior);nxt.update(learning_config_ref=learning_ref,clock_state_ref=clock_ref,consumer_config_ref=consumer_ref)
    nxt['policy']['current_ref']=policy_ref
    nxt['native'].update(restore_checkpoint_ref=checkpoint_ref,restore_learning_config_ref=prior['learning_config_ref'],
                         restore_consumer_config_ref=prior['consumer_config_ref'])
    nxt['continuation']=dict(publication_ref=policy['publication_ref'],successful_optimizer_steps=checkpoint['step'])
    return dict(input_ref=write(root/'INPUT.json',nxt),learning_ref=learning_ref,clock_ref=clock_ref,consumer_ref=consumer_ref)


def capture_input(input_ref,screen,policy,witness,learning,resume,output,*,root_seed=None,sampling_round=0,cohort_sequence=0):
    nxt=copy.deepcopy(read(input_ref));root=Path(output)
    require(screen['population']==119881 and screen['complete'],'full TRAIN source required')
    if screen.get('sampling_config_ref'):bind_screen_sampling(nxt,screen)
    require(nxt.get('prepare_mode') not in ('bounded_train_probe','finite_diagnostic_capture') and
            nxt['sampling'].get('purpose')=='screen','formal capture cannot retain diagnostic selector')
    pool=read(screen['screen_ref']);nxt['screen_ref']=screen['screen_ref']
    nxt['sampling'].update(pool_version=pool['version'],system_version=screen['system_version'],error_success=dict(error=.8,success=.2))
    if root_seed is not None:
        seeds=derive_seeds(root_seed,sampling_round,cohort_sequence)
        nxt['sampling']['seed']=seeds['sampling'];nxt['capture']['seed']=seeds['generation']
        nxt['sampling_lineage']=dict(root_seed=root_seed,sampling_round=sampling_round,cohort_sequence=cohort_sequence,**seeds)
    nxt['policy'].update(current_ref=policy,runtime_evidence_ref=witness)
    nxt['capture'].update(endpoint_policy_ref=witness,transport_config_ref=screen['transport_config_ref'])
    nxt['learning_config_ref']=learning
    # Original consumer owns population/labels/audio/token probabilities. The
    # scope comes from the same successful-step Clock; sampling size stays config.
    task=upstream()
    from scorer_extension import install
    install(task)
    consumer=read(nxt['consumer_config_ref']);clock=task.create_clock(read(learning),read(nxt['clock_state_ref']),consumer_config=consumer)
    phase=clock.phase();nxt['rollout_scope']=phase['rollout_scope']
    if phase['rollout_scope']=='node':
        release=nxt.get('phase_frozen_upstream_ref')
        require(release is not None and pin(release)['sha256']==phase['frozen_upstream_release'],'phase-frozen upstream release missing/drifted')
        # Sampling pool can change after DEV promotion; upstream packet release
        # is independently locked at phase entry and remains immutable.
        nxt['phase_frozen_upstream_ref']=pin(release)
    return write(root/'CAPTURE-INPUT.json',nxt)


def bind_phase(input_ref,screen,output):
    """Lock current upstream packets once at a successful-step phase boundary."""
    nxt=copy.deepcopy(read(input_ref));learning=copy.deepcopy(read(nxt['learning_config_ref']))
    if screen.get('complete'):bind_screen_sampling(nxt,screen)
    state=copy.deepcopy(read(nxt['clock_state_ref']));step=state['successful_optimizer_steps']
    phase=[p for p in learning['phases'] if p['start_step']<=step][-1]
    root=Path(output);release=pin(screen['screen_ref'])
    if phase['rollout_scope']=='node':
        existing=nxt.get('phase_binding')
        if existing and existing['phase_id']==phase['id']:
            require(existing['release_ref']['sha256']==phase['frozen_upstream_release'],'phase upstream release drift')
            return dict(input_ref=pin(input_ref),learning_ref=nxt['learning_config_ref'])
        # Future node phases get a valid initial lineage pin; each is rebound
        # once on its actual entry, never from a caller-written future hash.
        for future in learning['phases']:
            if future['rollout_scope']=='node' and str(future.get('frozen_upstream_release','')).startswith('BIND_'):
                future['frozen_upstream_release']=release['sha256']
        phase['frozen_upstream_release']=release['sha256']
        nxt['phase_frozen_upstream_ref']=release
        nxt['phase_binding']=dict(phase_id=phase['id'],entry_step=step,release_ref=release)
        learning['adapter_routing']['frozen_upstream_adapter_vector']=copy.deepcopy(learning['kernel']['current_policy_vector'])
    learning_ref=write(root/'LEARNING.json',learning)
    state.update(config_sha256=digest(learning),config_version=learning['version'])
    clock=write(root/'CLOCK.json',state);consumer=copy.deepcopy(read(nxt['consumer_config_ref']))
    consumer.update(learning_config_ref=learning_ref,clock_state_ref=clock)
    nxt.update(learning_config_ref=learning_ref,clock_state_ref=clock,consumer_config_ref=write(root/'CONSUMER.json',consumer))
    return dict(input_ref=write(root/'INPUT.json',nxt),learning_ref=learning_ref)


def retire_zero(input_ref,result_ref,output):
    """Persist a terminal zero-signal attempt without advancing optimizer step."""
    nxt=copy.deepcopy(read(input_ref));previous=read(nxt['clock_state_ref']);result=read(result_ref)
    state=result['clock'];require(state['successful_optimizer_steps']==previous['successful_optimizer_steps'],'zero signal advanced Clock')
    require(all(state['attempts'].get(k)==v for k,v in previous['attempts'].items()),'zero result lost attempt history')
    root=Path(output);clock=write(root/'CLOCK.json',state)
    consumer=copy.deepcopy(read(nxt['consumer_config_ref']));consumer['clock_state_ref']=clock
    nxt.update(clock_state_ref=clock,consumer_config_ref=write(root/'CONSUMER.json',consumer))
    return write(root/'INPUT.json',nxt)


def derive_seeds(root_seed,sampling_round,cohort_sequence):
    require(all(type(v) is int and v>=0 for v in (root_seed,sampling_round,cohort_sequence)),'nonnegative explicit random stream coordinates')
    return {name:int(digest(dict(root_seed=root_seed,sampling_round=sampling_round,cohort_sequence=cohort_sequence,stream=name))[:8],16) for name in ('sampling','generation')}


def diagnostic_input(input_ref,screen,policy,witness,output,*,root_seed,sampling_round,cohort_sequence,sampling=None,evidence_nodes=None):
    """Independent scoped evidence collection; never a training reservation.

    A boundary's next training phase deliberately still lacks DEV validation.
    Use a separately identified diagnostic system scope, retaining the exact
    candidate/reference and original checkpoint recovery configuration. Do not
    fabricate validations in the training Clock to collect that evidence.
    """
    nxt=copy.deepcopy(read(input_ref));root=Path(output)
    nxt.pop('sampled_window_ref',None)
    learning=copy.deepcopy(read(nxt['learning_config_ref']))
    training_clock=read(nxt['clock_state_ref']);step=training_clock['successful_optimizer_steps']
    phase=copy.deepcopy([p for p in learning['phases'] if p['start_step']<=step][-1])
    coefficients=copy.deepcopy(learning['task_loss']['phases'][phase['id']])
    phase.update(id='updated-parity-evidence',start_step=0,rollout_scope='system',target_nodes=[],
        trainable_mask={n:True for n in ('P','S1','S2','D')},frozen_upstream_release=None,
        required_validations=[])
    if evidence_nodes is not None:
        require(isinstance(evidence_nodes,list) and evidence_nodes and
                evidence_nodes==[n for n in ('P','S1','S2','D') if n in evidence_nodes],
                'nonempty ordered unique diagnostic scope required')
        frozen=pin(nxt['phase_frozen_upstream_ref'])
        phase.update(rollout_scope='node',target_nodes=list(evidence_nodes),
            trainable_mask={n:n in evidence_nodes for n in ('P','S1','S2','D')},
            frozen_upstream_release=frozen['sha256'])
    # This is a separately identified first phase for evidence generation.
    # The training configuration and its still-pending gates remain immutable.
    learning.update(version=learning['version']+'@updated-parity-evidence',phases=[phase])
    learning['task_loss']['phases']={phase['id']:coefficients}
    lr=write(root/'LEARNING.json',learning)
    state=copy.deepcopy(training_clock);state.update(config_sha256=digest(learning),config_version=learning['version'])
    state['groups']={};state['attempts']={}
    # This derived Clock is only used by the original capture machinery. It has
    # no successful optimizer attempt and cannot be passed to the trainer.
    state_ref=write(root/'CLOCK.json',state)
    consumer=copy.deepcopy(read(nxt['consumer_config_ref']));consumer.update(learning_config_ref=lr,clock_state_ref=state_ref)
    nxt.update(learning_config_ref=lr,clock_state_ref=state_ref,consumer_config_ref=write(root/'CONSUMER.json',consumer),rollout_scope=phase['rollout_scope'])
    nxt['diagnostic_evidence_only']=dict(training_input_ref=pin(input_ref),training_learning_ref=pin(read(input_ref)['learning_config_ref']),
        training_clock_ref=pin(read(input_ref)['clock_state_ref']),purpose='updated_parity',trainable=False)
    nxt['policy'].update(current_ref=policy,runtime_evidence_ref=witness)
    nxt['capture']['endpoint_policy_ref']=witness
    nxt['authorization']=dict(nxt['authorization'],native_training=False)
    nxt['native']=dict(nxt['native'],enabled=False)
    if screen:
        require(screen['complete'] and screen['population']==119881,'diagnostic source needs complete TRAIN pool')
        pool=read(screen['screen_ref']);nxt['screen_ref']=screen['screen_ref']
        nxt['sampling'].update(pool_version=pool['version'],system_version=screen['system_version'])
        nxt['capture']['transport_config_ref']=screen['transport_config_ref']
    if sampling:
        require(set(sampling)<={'sample_count','replacement'},'diagnostic sampling configuration keys')
        nxt['sampling'].update(sampling)
    seeds=derive_seeds(root_seed,sampling_round,cohort_sequence)
    nxt['sampling']['seed']=seeds['sampling'];nxt['capture']['seed']=seeds['generation']
    nxt['sampling_lineage']=dict(root_seed=root_seed,sampling_round=sampling_round,cohort_sequence=cohort_sequence,**seeds)
    return write(root/'INPUT.json',nxt)


def phase_validation(input_ref,dev_ref,quality_ref,passed,output):
    """Attach full DEV business evidence through the original Clock API."""
    require(passed,'phase DEV business gate rejected; configured stop_preserve')
    report=read(dev_ref);require(report['evaluation_complete'] and report['denominator']==report['scored']==8690,'phase validation needs full DEV')
    quality=read(quality_ref)
    require(quality.get('passed') is True if quality.get('schema')=='omni-phase-acceptance-v1'
            else quality['business_acceptance']['dev_selection_eligible'],'paired phase DEV gate failed')
    nxt=copy.deepcopy(read(input_ref));learning=read(nxt['learning_config_ref']);consumer=read(nxt['consumer_config_ref'])
    task=upstream()
    from scorer_extension import install
    install(task)
    clock=task.create_clock(learning,read(nxt['clock_state_ref']),consumer_config=consumer)
    phase=[p for p in learning['phases'] if p['start_step']<=clock.state['successful_optimizer_steps']][-1]
    for name in phase['required_validations']:
        if name not in clock.state['validations']:
            clock.record_validation(name,dict(passed=True,fixture=False,scope='node',config_sha256=clock.state['config_sha256'],
                evidence_sha256=pin(quality_ref)['sha256'],dev_report_ref=pin(dev_ref),paired_quality_ref=pin(quality_ref),
                step=clock.state['successful_optimizer_steps']))
    root=Path(output);ref=write(root/'CLOCK.json',clock.state);consumer.update(clock_state_ref=ref)
    nxt.update(clock_state_ref=ref,consumer_config_ref=write(root/'CONSUMER.json',consumer))
    return write(root/'INPUT.json',nxt)


def bounded_capture_input(config,input_ref,policy,witness,learning,output,*,cohort_sequence):
    """Reuse the existing bounded source/consumer with the committed current policy.

    Two explicitly budgeted cohorts only. The original owner still generates
    real actions, and the original consumer/finalizer alone declares trainability.
    """
    if config.get('engineering_update_sequence'):
        from finite_runtime import engineering_updates
        maximum=engineering_updates(config)
    else:
        require(config.get('bounded_restore_compare') is True and
                config.get('newprocess_replay',{}).get('enabled') and
                config['max_optimizer_updates']==2,'bounded replay configuration required')
        maximum=2
    require(type(cohort_sequence) is int and 1<=cohort_sequence<=maximum,'configured cohort sequence bound')
    nxt=copy.deepcopy(read(input_ref));current=read(policy)
    require(nxt.get('prepare_mode')=='bounded_train_probe' and not nxt.get('cpu_contract_only') and
            not nxt['policy'].get('fixture') and not current.get('fixture'),'real prepared bounded TRAIN input required')
    state=read(nxt['clock_state_ref']);step=state['successful_optimizer_steps']
    require(step==cohort_sequence-1 and current['step']==step,'committed policy/Clock/cohort sequence differs')
    require(nxt['learning_config_ref']==learning and read(nxt['policy']['current_ref'])['version']==current['version'],'migrated learning/current policy differs')
    nxt['policy']['current_ref']=policy
    require(read(nxt['policy']['reference_ref'])['kind']=='base','frozen base reference required')
    if step:
        from publication import validate_publication
        publication=validate_publication(current['publication_ref'])
        require(publication['status']=='published_for_fresh_sampling','updated parity promotion required before cohort2')
    from scorer_extension import install
    task=upstream();install(task)
    clock=task.create_clock(read(learning),state,consumer_config=read(nxt['consumer_config_ref']))
    phase=clock.phase()
    require(phase['rollout_scope']=='node' and phase['target_nodes']==['P'] and
            phase['trainable_mask']=={n:n=='P' for n in ('P','S1','S2','D')},'bounded replay retains original node-P mask')
    require(pin(nxt['phase_frozen_upstream_ref'])['sha256']==phase['frozen_upstream_release'], 'frozen source release differs')
    sampling=nxt['sampling']
    require(sampling['purpose']=='bounded_train_probe' and sampling['split']=='TRAIN' and
            sampling['replacement']=='forbid' and sampling['error_success'] is None and
            nxt['bounded_train_probe']['groups']==2 and nxt['bounded_train_probe']['G']==4,'bounded TRAIN contract differs')
    seeds=derive_seeds(config['root_seed'],0,cohort_sequence)
    sampling['seed']=seeds['sampling'];nxt['capture']['seed']=seeds['generation']
    nxt['sampling_lineage']=dict(root_seed=config['root_seed'],sampling_round=0,cohort_sequence=cohort_sequence,**seeds)
    nxt['policy']['runtime_evidence_ref']=witness;nxt['capture']['endpoint_policy_ref']=witness
    nxt['native']['enabled']=False
    nxt['authorization'].update(model_calls=False,native_training=False)
    return write(Path(output)/'CAPTURE-INPUT.json',nxt)
