"""Compile explicitly selected weights with a new Clock, or replay a historical expanded branch."""
import argparse,copy,os,sys
from pathlib import Path


def base_branch(c,out):
    from run_package import read,pin,write
    from dependencies import require,ROOT,upstream
    from paired_experiment import rank_profile,digest
    from scorer_extension import install
    from initial_policy import validate_initial
    require(set(c)-{'training_input_template'}=={'schema','scope','initialization','base_manifest','system','frozen_upstream','geometry_source','lineage'}
            and isinstance(c['lineage'],str) and c['lineage'],'independent branch config')
    scope=read(c['scope']);task=scope['task'];node=task['node']
    init=c['initialization'];mode=init['mode'];base_ref=pin(c['base_manifest']);base=read(base_ref)
    require(set(init)=={'mode','source_ref'} and mode in ('base_adapter','checkpoint_weights'),
            'select an explicit base adapter or checkpoint weights; resume uses the original run cursor')
    require(isinstance(init['source_ref'],dict) and set(init['source_ref'])=={'path','bytes','sha256'},
            'initialization requires a pinned source manifest')
    src_ref=pin(init['source_ref'])
    require(src_ref==init['source_ref'],'source manifest identity changed')
    origin=read(src_ref)
    require(scope['scope']=='node' and scope['trainable_mask']=={n:n==node for n in ('P','S1','S2','D')},'wrong branch scope')
    if mode=='base_adapter':
        require(origin['schema']=='independent-node-base-adapter-origin-v1'
                and origin['inputs']['base_manifest']['sha256']==base_ref['sha256'],'wrong base adapter')
        folder=Path(src_ref['path']).parent
    else:
        require(origin.get('recoverable') is True and origin['step']==origin['clock']['successful_optimizer_steps']>0
                and origin['resident_identity']['base']['asset_manifest_ref']['sha256']==base_ref['sha256']
                and origin['contract']['lineage']!=c['lineage'],'checkpoint identity or new lineage differs')
        folder=Path(src_ref['path']).parent/'lora_adapter'
    exp=read(scope['inputs']['experiment']);measured=read(c['geometry_source'])
    require(measured['schema']=='resident-actor-initialization-v1' and measured['lineage']!=c['lineage']
            and measured['devices']==exp['training_devices'] and measured['base']['model_path']==exp['model_path']
            and measured['base']['observed_assets']==base['assets'] and measured['base']['asset_manifest_ref']['sha256']==base_ref['sha256']
            and measured['training']['model_path']==exp['model_path'],'actor and independent base differ')
    profile=rank_profile(exp)
    geometry={k:measured[k] for k in ('image','devices','base')}
    geometry.update(schema='base-node-actor-geometry-v1',training=dict(measured['training'],**profile['training']))
    geometry_ref=write(out/'GEOMETRY.json',geometry)
    assets=[pin(folder/name) for name in ('adapter_model.safetensors','adapter_config.json')]
    version='initial-sha256:'+assets[0]['sha256']
    pub=write(out/'INITIAL.json',dict(schema='configured-node-initial-policy-v1',initialization=dict(mode=mode,source_ref=src_ref),
              base_manifest_ref=base_ref,geometry_ref=geometry_ref,profile_ref=write(out/'PROFILE.json',profile),
              assets=assets,node=node,lineage=c['lineage']))
    policy=dict(schema='fresh-policy-version-v1',kind='branch_initial',step=0,role='current',model=base['model'],
                thinking=False,version=version,artifact_version=version,assets=assets,publication_ref=pub)
    validate_initial(policy);policy_ref=write(out/'POLICY.json',policy)
    ref_version='base-sha256:'+digest(base['assets'])
    reference=write(out/'REFERENCE.json',dict(schema='fresh-policy-version-v1',kind='base',role='reference',step=0,
              model=base['model'],thinking=False,version=ref_version,frozen=True,assets=base['assets'],
              asset_manifest_ref=base_ref,initialization='unmodified_base_no_optimizer_update'))
    learning=read(scope['learning_ref']);phase,=learning['phases'];nodes=('P','S1','S2','D')
    require(phase['target_nodes']==[node] and phase['trainable_mask']=={n:n==node for n in nodes},
            'scope and independent learning target differ')
    system=pin(c['system']);frozen=pin(c['frozen_upstream'])
    learning['system_version']=system['sha256'];phase['frozen_upstream_release']=frozen['sha256']
    learning['version']+='@'+c['lineage']+':'+version
    for k,v in [('current',version),('behavior',version),('reference',ref_version)]:
        learning['kernel'][k+'_policy_vector']=dict.fromkeys(nodes,v)
    learning['adapter_routing']['frozen_upstream_adapter_vector']=dict.fromkeys(nodes,ref_version)
    pins={n:pin(ROOT/'unified-learning'/n) for n in ('kernel.py','learning.py','scoring.py','schedule.py','dataset.py','fixtures.py')}
    pins.update({n:pin(ROOT/'system-objective-cpu'/n) for n in ('objective.py','runtime.py')})
    api=upstream();install(api);clock=api.create_clock(learning,consumer_config=dict(source_pins=pins))
    require(clock.phase()==phase and clock.state['successful_optimizer_steps']==0 and
            all(not clock.state[k] for k in ('groups','attempts','validations')),'nonempty initial Clock')
    anchor=write(out/'BRANCH.json',dict(schema='base-node-policy-branch-v1',step=0,optimizer_inherited=False,node=node,
        policy_ref=policy_ref,reference_ref=reference,learning_ref=write(out/'LEARNING.json',learning),
        clock_ref=write(out/'CLOCK.json',clock.state),system_ref=system,frozen_upstream_ref=frozen))
    if 'training_input_template' not in c:return anchor
    # Roster and frozen upstream inputs belong to this task, independently
    # of the checkpoint writer. Never transplant a predecessor's trajectories.
    source=read(c['training_input_template'])
    require(source['sampling']['scope']=='node' and source['sampling']['node']==node
            and source['sampling']['split']=='TRAIN' and source['sampling']['system_version']==system['sha256']
            and pin(source['phase_frozen_upstream_ref'])==frozen and source['rollout_scope']=='node'
            and not set(source)&{'continuation','branch_initial_ref','sampled_window_ref','lease_fence'}
            and not any(k.startswith('restore_') for k in source['native'])
            and not set(source['capture'])&{'endpoint_policy_ref','lanes','lane_transports'}
            and not source['policy'].get('runtime_evidence_ref'),'wrong node roster, upstream or inherited run state')
    branch=read(anchor);consumer=read(source['consumer_config_ref'])
    consumer.update(learning_config_ref=branch['learning_ref'],clock_state_ref=branch['clock_ref'])
    source.update(branch_initial_ref=anchor,clock_state_ref=branch['clock_ref'],learning_config_ref=branch['learning_ref'],
                  consumer_config_ref=write(out/'CONSUMER.json',consumer),
                  phase_binding=dict(phase_id=phase['id'],entry_step=0,release_ref=frozen))
    source['policy'].update(current_ref=policy_ref,reference_ref=reference)
    return write(out/'INPUT.json',source)



def prepare(config,out):
    from run_package import read,write,pin
    from dependencies import require,upstream
    from initial_policy import validate_initial
    from scorer_extension import install
    c=read(config)
    if c['schema']=='base-node-branch-inputs-v1':return base_branch(c,out)
    require(c['schema']=='expanded-policy-branch-inputs-v1','unexpected branch config')
    experiment = None
    if 'package' in c:
        from paired_experiment import load, rank_profile
        package = read(c['package'])
        experiment = load(pin(package['experiment_ref'])['path'], package['experiment_sha256'])
        require(read(package['rank_profile_ref']) == rank_profile(experiment), 'initial profile differs from experiment')
        c['profile'] = pin(package['rank_profile_ref'])['path']
    parent=read(c['cursor']);prior=read(parent['capture_source_input_ref'])
    committed=read(c['parent_commit'])
    require(pin(c['parent_commit'])['sha256']==parent['resume_ref']['sha256'] and
            committed['step']==parent['step'] and parent['consumed_cohorts'][-1]['result']['checkpoint_ref']==parent['resume_ref'],
            'parent policy differs from last successful commit')
    dataset=read(parent['sampled_dataset_ref']);window=read(parent['sampled_pending_window_ref'])
    drained=read(parent['capture']['resident_boundary_ref'])['terminal']
    inventory=read(parent['capture']['generation_inventory_ref'])
    require(parent['phase']=='update' and parent['pending_stage']['stage']=='update' and
            window['start']==parent['sampled_consumed'] and window['end']<=dataset['count'] and
            window['dataset_ref']==parent['sampled_dataset_ref'] and inventory['closed'] and
            drained['terminal_verified'] and drained['unknown_requests']==0 and
            drained['generation_closed_ref']==inventory['generation_closed_ref']==parent['capture']['generation_closed_ref'] and
            read(parent['capture']['input_ref'])['sampled_window_ref']==parent['sampled_pending_window_ref'],
            'unreconciled or unrelated prior collection')
    adapter=Path(c['adapter'])
    assets=[pin(adapter/name) for name in ('adapter_model.safetensors','adapter_config.json')]
    version='initial-sha256:'+assets[0]['sha256']
    publication=write(out/'INITIAL.json',dict(schema='expanded-initial-policy-v1',origin_ref=pin(adapter/'ORIGIN.json'),
            parent_ref=pin(c['parent_commit']),profile_ref=pin(c['profile']),assets=assets))
    old=read(prior['policy']['current_ref'])
    require(old['version']=='checkpoint-sha256:'+committed['ranks'][0]['files']['lora_adapter/adapter_model.safetensors'],
            'source learned from a different parent')
    policy=dict(schema=old['schema'],role='current',kind='branch_initial',step=0,version=version,
                artifact_version=version,model=old['model'],thinking=False,assets=assets,publication_ref=publication)
    validate_initial(policy)
    policy_ref=write(out/'POLICY.json',policy)
    learning=copy.deepcopy(read(prior['learning_config_ref']))
    require(len(learning['phases'])==1 and learning['phases'][0]['target_nodes']==['P']
            and learning['phases'][0]['rollout_scope']=='node' and learning['phases'][0]['start_step']==0
            and not learning['phases'][0]['required_validations'],'expected existing P learning phase')
    if experiment:
        learning['kernel']['G'] = experiment['episodes_per_recording']
    learning['version']+='@'+version
    for name in ('behavior_policy_vector','current_policy_vector'):
        require(all(v==old['version'] for v in learning['kernel'][name].values()),'parent vector drift')
        learning['kernel'][name]=dict.fromkeys(('P','S1','S2','D'),version)
    lr=write(out/'LEARNING.json',learning)
    consumer=copy.deepcopy(read(prior['consumer_config_ref']))
    consumer['learning_config_ref']=lr
    task=upstream();install(task)
    clock=task.create_clock(learning,consumer_config=consumer)
    require(clock.phase()==learning['phases'][0] and clock.state['successful_optimizer_steps']==0
            and all(not clock.state[k] for k in ('groups','attempts','validations')),'invalid initial Clock')
    cr=write(out/'CLOCK.json',clock.state);consumer['clock_state_ref']=cr
    inp=copy.deepcopy(prior)
    inp.update(learning_config_ref=lr,clock_state_ref=cr,consumer_config_ref=write(out/'CONSUMER.json',consumer))
    inp['policy']['current_ref']=policy_ref
    inp['policy'].pop('runtime_evidence_ref',None)
    for key in ('endpoint_policy_ref','lane_transports','transport_config_ref','lanes'):
        inp['capture'].pop(key,None)
    for key in ('restore_checkpoint_ref','restore_learning_config_ref','restore_consumer_config_ref','policy_config_migration_ref'):
        inp['native'].pop(key,None)
    for key in ('sampled_window_ref','lease_fence','continuation'):
        inp.pop(key,None)
    inp['phase_binding']=dict(inp['phase_binding'],entry_step=0)
    anchor=write(out/'BRANCH.json',dict(schema='expanded-policy-branch-v1',step=0,clock_ref=cr,
         policy_ref=policy_ref,learning_ref=lr,parent_cursor_ref=pin(c['cursor']),parent_checkpoint=parent['resume_ref'],
         excluded_window_ref=parent['sampled_pending_window_ref'],closed_generation_ref=drained['generation_closed_ref'],
         dataset_ref=parent['sampled_dataset_ref'],next_window_start=window['end'],prior_tokens_upper=parent['charged_tokens_upper'],
         optimizer_inherited=False))
    inp['branch_initial_ref']=anchor
    return write(out/'INPUT.json',inp)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name, env in [('release','OMNI_TRAIN_RELEASE'),('config','OMNI_BRANCH_CONFIG'),('out','OMNI_BRANCH_OUTPUT')]:
        default=os.environ.get(env)
        p.add_argument('--'+name,type=Path,default=default,required=not default,help=f'default: ${env}')
    p.add_argument('--profile',type=Path,default=os.environ.get('OMNI_DEPLOYMENT_PROFILE') or None,
                   help='resolve branch config as a template with this deployment profile')
    a=p.parse_args();parents=a.release.resolve(strict=True)/'tree/parents'
    impl=parents/'jev-system-rl/implementation'
    if not (impl/'run_package.py').is_file():p.error('expected assembled training release')
    sys.path[:0]=[str(impl),str(parents/'policy-iteration-successor')]
    # Read unchanged dependencies from the supplied release. The compiler ships its
    # branch validator and profile resolver together, even for older parent releases.
    from source import initial_policy,config_profile
    sys.modules['initial_policy']=initial_policy
    if a.profile:
        from run_package import read
        receipt=read(config_profile.compile_profile(a.config,a.profile,a.out/'resolution'))
        config=receipt['resolved_ref']
    else:
        config=a.config
    print(prepare(config,a.out))
