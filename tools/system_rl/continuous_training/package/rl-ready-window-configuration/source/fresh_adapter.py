"""Scoped seams for original fresh consumer/policy; DP8 changes no trainer math."""
import contextlib,copy,inspect,os,sys,types
from pathlib import Path
from dependencies import ROOT,require,load
from run_package import read,pin


def consumer(module):
    if getattr(module,'_dp8_scoped',False):return module
    from scorer_extension import extend,configure_clip,configure_frozen
    old=module.unified
    @contextlib.contextmanager
    def unified(config):
        with old(config) as (learning,schedule,kernel):
            extend(sys.modules['scoring']);configure_clip(schedule);configure_frozen(schedule)
            learning.evaluate.__globals__['score_episode']=sys.modules['scoring'].score_episode
            yield learning,schedule,kernel
    module.unified=unified;module._dp8_scoped=True
    return module


def policy(module, *, allow_candidate_parity=False):
    """Keep original validator; adapt only checkpoint.files shape to DP8 commit."""
    if getattr(module,'_dp8_scoped',False):
        module._dp8_allow_candidate_parity=allow_candidate_parity
        return module
    module._dp8_allow_candidate_parity=allow_candidate_parity
    from publication import validate_publication,checkpoint
    historical_scope=module.validate_scope
    def input_scope(config,*,fixture=False):
        if config.get('sampling',{}).get('scope')!='node' or fixture:
            return historical_scope(config,fixture=fixture)
        sampling=config['sampling'];pool=read(config['screen_ref']);window=read(config['sampled_window_ref'])
        dataset=read(window['dataset_ref'])
        require(pool['schema']=='node-screen-selection-v1' and pool['sealed'] and
                sampling['node']==pool['node']=='P' and dataset['screen_ref']==config['screen_ref'] and
                sampling==dataset['sampling'] and pool['version']==sampling['pool_version'],'node screen identity')
        step=read(config['clock_state_ref'])['successful_optimizer_steps']
        phase=[p for p in read(config['learning_config_ref'])['phases'] if p['start_step']<=step][-1]
        from node_screen_selection import phase_inputs
        require(config.get('rollout_scope',sampling['scope'])=='node' and
                phase_inputs(config,phase,pool)==dataset.get('phase_frozen_upstream_ref'),
                'P inputs and active learning phase differ')
    module.validate_scope=input_scope
    original=module.validate
    def validate(spec,learning,state,fixture=False):
        current=read(spec['current_ref'])
        if current['kind']=='branch_initial':
            from bounded_tp4 import validate_routes
            from policy_identity import digest
            admission=read(current['admission_ref']);seal=read(current['asset_manifest_ref'])
            witness=read(spec['runtime_evidence_ref']);ref=read(spec['reference_ref']);kernel=learning['kernel']
            require(not fixture and spec.get('formal_sampled') and current['step']==state['successful_optimizer_steps']==0
                    and current['role']=='current' and current['model']==ref['model']==witness['model']
                    and current['thinking'] is ref['thinking'] is False and current['version']==admission['artifact_version']
                    and seal['status']=='CLOSED_IMMUTABLE' and current['assets']==admission['assets']==seal['assets']
                    and current['publication_ref']==admission['initial_publication_ref']==seal['publication_ref']
                    and witness['asset_manifest_ref']==current['asset_manifest_ref']
                    and ref['frozen'] and ref['kind']=='base' and ref['role']=='reference'
                    and read(ref['asset_manifest_ref'])['assets']==ref['assets'] and ref['version']=='base-sha256:'+digest(ref['assets'])
                    and kernel['reference_policy_vector']==dict.fromkeys(module.NODES,ref['version'])
                    and kernel['current_policy_vector']==kernel['behavior_policy_vector']==dict.fromkeys(module.NODES,current['version'])
                    and learning['phases'][0]['target_nodes']==['P'] and learning['phases'][0]['rollout_scope']=='node',
                    'initial training weights, reference or active Clock differ')
            validate_routes(admission,spec['runtime_evidence_ref'],read(witness['profile_ref']),sampled=True)
            return current,ref
        if current['kind']!='checkpoint':return original(spec,learning,state,fixture=fixture)
        pub=validate_publication(current['publication_ref'],verify_checkpoint_shards=False)
        if pub.get('schema')!='omni-dp8-publication-v1':return original(spec,learning,state,fixture=fixture)
        if pub['status']=='candidate_for_updated_parity' and module._dp8_allow_candidate_parity:
            # Diagnostic collection is the bounded evidence-producing step
            # before promotion; it cannot produce a trainable publication.
            reference=read(spec['reference_ref'])
            require(reference['kind']=='base' and reference['frozen'] and reference['role']=='reference','independent frozen base required')
            require(learning['kernel']['current_policy_vector']==learning['kernel']['behavior_policy_vector']=={n:current['version'] for n in ('P','S1','S2','D')},'candidate parity policy vector differs')
            require(state['successful_optimizer_steps']==current['step'],'candidate parity clock step differs')
            committed,_=checkpoint(pub['checkpoint_ref'],verify_shards=False)
            from policy_identity import behavior_binding
            behavior_binding(committed['resident_identity'],{'endpoint_witness_ref':spec['runtime_evidence_ref']})
            return current,reference
        require(pub['status']=='published_for_fresh_sampling','fresh sampling requires promoted updated parity')
        committed,files=checkpoint(pub['checkpoint_ref'],verify_shards=False)
        def adapted_read(ref):
            from dependencies import local_path
            physical=local_path(ref['path'])
            value=module.read(dict(ref,path=str(physical)))
            return dict(value,files=files) if ref==pub['checkpoint_ref'] else value
        scoped=types.FunctionType(original.__code__,dict(original.__globals__,read=adapted_read,
            validate_publication=lambda ref:validate_publication(ref,verify_checkpoint_shards=False)),original.__name__,original.__defaults__,original.__closure__)
        result=scoped(spec,learning,state,fixture=fixture)
        if 'bounded_tp4_ref' in pub or 'continuous_activation_ref' in pub:
            phase=[p for p in learning['phases'] if p['start_step']<=state['successful_optimizer_steps']][-1]
            require(not fixture and phase['rollout_scope']=='node' and phase['target_nodes']==['P'],
                    'bounded TP4 evidence is only valid for real node-P sampling')
            from bounded_tp4 import validate_routes
            witness=read(spec['runtime_evidence_ref'])
            validate_routes(pub,spec['runtime_evidence_ref'],read(witness['profile_ref']),
                            sampled=spec.get('formal_sampled') is True)
        else:
            from policy_identity import behavior_binding
            behavior_binding(committed['resident_identity'],{'endpoint_witness_ref':spec['runtime_evidence_ref']})
        return result
    # Base policy can continue terminal zero-signal cohorts at step0, with
    # original validator seeing only the unchanged successful-update history.
    base_original=original
    def routed(spec,learning,state,fixture=False):
        if read(spec['current_ref'])['kind']=='base':
            known=copy.deepcopy(state)
            zero_groups={gid for attempt in state.get('attempts',{}).values() if not attempt['succeeded'] for gid in attempt['group_ids']}
            for gid in zero_groups:
                if gid in known['groups']:known['groups'][gid]['consumed']=False
            require(state['successful_optimizer_steps']==0,'base cannot continue updated optimizer')
            return base_original(spec,learning,known,fixture=fixture)
        return validate(spec,learning,state,fixture=fixture)
    module.validate=routed;module._dp8_scoped=True
    return module


@contextlib.contextmanager
def scope():
    """Isolate generic original imports, record exact resolved sources."""
    names=('cohort_io','policy','cohort','consumer','execution','native_update','task_runtime',
           'runtime','common','forward','validate_npu','training_state','sync_runtime','clock','deps','labels','targets','facts','protocol')
    saved={n:sys.modules.get(n) for n in names};paths=sys.path[:]
    try:
        for n in names:sys.modules.pop(n,None)
        sys.path[:0]=[str(ROOT/'fresh-cohort-successor'),str(ROOT/'task-loss-integration'),
            str(ROOT/'native-model-integration'),str(ROOT/'native-training-successor'),
            str(ROOT/'native-recompute-integration/state-repair/acl-runtime-successor')]
        io=load(ROOT/'fresh-cohort-successor/cohort_io.py','cohort_io')
        policy(load(ROOT/'fresh-cohort-successor/policy.py','policy'))
        original_load=io.load
        consumer_path=(ROOT/'fresh-cohort-successor/consumer.py').resolve()
        legacy_path=(ROOT/'training-consumer-successor/adapter.py').resolve()
        overlays={p:os.environ.get('OMNI_CAPTURE_'+role) for p,role in (
            (consumer_path,'CONSUMER_WRAPPER'),(legacy_path,'CONSUMER_SOURCE'))}
        def scoped_load(path,name):
            path=Path(path).resolve()
            if source:=overlays.get(path):
                module=types.ModuleType(name)
                module.__file__=str(path)
                module._capture_source=source
                sys.modules[name]=module
                exec(compile(Path(source).read_bytes(),source,'exec'),module.__dict__)
            else:module=original_load(path,name)
            if path==consumer_path:consumer(module)
            return module
        io.load=scoped_load
        if overlays[consumer_path]:
            scoped_load(consumer_path,'consumer')
        cohort=load(ROOT/'fresh-cohort-successor/cohort.py','cohort')
        def clock_for(config):
            from dependencies import upstream
            task=upstream()
            from scorer_extension import install
            install(task)
            return task.create_clock(read(config['learning_config_ref']),read(config['clock_state_ref']),consumer_config=read(config['consumer_config_ref']))
        cohort.clock_for=clock_for
        original_draw=cohort.draw
        def draw(config,root,**kwargs):
            # Formal and bounded TRAIN selectors defer publication until
            # the final packet-bound plan is ready.
            single_commit=config.get('prepare_mode')!='finite_diagnostic_capture'
            plan=original_draw(config,root,**dict(kwargs,**({'persist':False} if single_commit else {})))
            learning=read(config['learning_config_ref']);state=read(config['clock_state_ref'])
            phase=[p for p in learning['phases'] if p['start_step']<=state['successful_optimizer_steps']][-1]
            if phase['rollout_scope']=='node' and config.get('sampling',{}).get('scope')=='node':
                pool=read(config['screen_ref'])
                from node_screen_selection import phase_inputs
                plan['phase_frozen_upstream_ref']=phase_inputs(config,phase,pool)
                require(phase['target_nodes']==[pool['node']] and
                        all('node_input_ref' in item['record']['online'] for item in plan['draws']),
                        'missing independently bound P inputs')
            elif phase['rollout_scope']=='node':
                release=config['phase_frozen_upstream_ref'];require(pin(release)['sha256']==phase['frozen_upstream_release'],'node packet release drift')
                frozen=None
                if config.get('sampled_window_ref'):
                    window=read(config['sampled_window_ref']);dataset=read(window['dataset_ref'])
                    frozen=dataset.get('phase_frozen_upstream_ref')
                    if frozen:require(frozen==pin(release),'cached node inputs belong to another upstream release')
                if not frozen:
                    from cohort_io import imports
                    from training_references import bind_frozen_inputs
                    with imports([ROOT/'collection-scale-successor/code'],('common','stream_pool')):
                        from stream_pool import Pool
                        pool=read(release);records=Pool(pool)['records'] if 'streams' in pool else pool['records']
                        plan['draws']=bind_frozen_inputs(plan['draws'],records)
                # Packet custody is part of the actual selection identity;
                # rebind draw/group IDs after replacing upstream packet refs.
                plan['sampling_plan_version']=plan['version']
                from policy_identity import digest
                plan['phase_frozen_upstream_ref']=pin(release)
                plan['version']=digest(dict(sampling_plan_version=plan['version'],frozen_release=pin(release),draws=plan['draws']))
                for item in plan['draws']:
                    item['query_draw_id']=plan['version']+':draw:'+str(item['draw_index'])
                    item['group_id']=plan['version']+':group:'+str(item['draw_index'])
                plan['group_ids']=[item['group_id'] for item in plan['draws']]
            # Commit only the final packet-bound selection. Restart recomputes
            # the same identity and immutable save rejects any actual drift.
            if single_commit:
                cohort.save(Path(root)/'SELECTED.json',plan)
            elif phase['rollout_scope']=='node':
                from storage import commit
                commit(Path(root)/'SELECTED.json',plan,replace=True)
            return plan
        cohort.draw=draw
        yield cohort
    finally:
        sys.path[:]=paths
        for n,value in saved.items():
            sys.modules.pop(n,None)
            if value is not None:sys.modules[n]=value


def clone_generation(pending_ref,inventory_ref,destination):
    """Copy CLOSED generation only; new group/consumer outputs are writable local."""
    from dependencies import local_path,sha
    from storage import check_running,commit
    import os
    native=read(pending_ref);pending=read(native['pending_capture_ref']);inventory=read(inventory_ref)
    require(inventory['generation_closed_ref'] and inventory['cohort_root']==pending['cohort_root'],'generation inventory root differs')
    read(inventory['generation_closed_ref'])
    destination=Path(destination)
    require(inventory['closed'] is True,'active generation cannot be cloned')
    check_running(destination.parent,sum(r['bytes'] for r in inventory['files'].values()))
    for name,ref in inventory['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'generation path escapes')
        target=destination/name;raw=local_path(pin(ref)['path']).read_bytes()
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():require(sha(target)==ref['sha256'],'local clone changed')
        else:
            with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            fd=os.open(target.parent,os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
    return native,pending
