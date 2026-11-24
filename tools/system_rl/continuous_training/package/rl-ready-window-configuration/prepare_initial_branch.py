"""Compile an expanded-weight start with a fresh learning Clock and the existing dataset."""
import argparse,copy,os,sys
from pathlib import Path


def prepare(config,out):
    from run_package import read,write,pin
    from dependencies import require,upstream
    from initial_policy import validate_initial
    from scorer_extension import install
    c=read(config)
    require(c['schema']=='expanded-policy-branch-inputs-v1','unexpected branch config')
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
