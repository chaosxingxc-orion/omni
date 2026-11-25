"""First use of published weights on a new actor lineage. No optimizer restore."""
from pathlib import Path
from dependencies import require
from run_package import pin, read


def compile_start(branch_ref,lineage):
    from rank_profile import validate
    from policy_identity import digest
    branch=read(branch_ref)
    if branch['schema']=='base-node-policy-branch-v1':
        from initial_policy import validate_initial
        policy=read(branch['policy_ref']);pub=validate_initial(policy)
        require(pub['schema']=='configured-node-initial-policy-v1' and pub['lineage']==lineage
                and pub['node']==branch['node'] and branch['step']==0 and branch['optimizer_inherited'] is False
                and not set(branch)&{'parent_checkpoint','parent_cursor_ref','dataset_ref'},'unexpected independent ancestry')
        geometry=read(pub['geometry_ref']);profile=read(pub['profile_ref'])
        clock=read(branch['clock_ref']);learning=read(branch['learning_ref']);reference=read(branch['reference_ref'])
        phase,=learning['phases']
        nodes=('P','S1','S2','D')
        require(phase['rollout_scope']=='node' and phase['target_nodes']==[branch['node']]
                and phase['trainable_mask']=={n:n==branch['node'] for n in nodes}
                and learning['kernel']['current_policy_vector']==learning['kernel']['behavior_policy_vector']==dict.fromkeys(nodes,policy['version'])
                and reference['kind']=='base' and reference['role']=='reference' and reference['frozen'] is True
                and reference['asset_manifest_ref']==pub['base_manifest_ref']
                and reference['version']=='base-sha256:'+digest(geometry['base']['observed_assets'])
                and learning['kernel']['reference_policy_vector']==dict.fromkeys(nodes,reference['version'])
                and clock['successful_optimizer_steps']==0 and all(not clock[k] for k in ('groups','attempts','validations'))
                and clock['config_sha256']==digest(learning),'independent learning or Clock differs')
        return dict(schema='resident-actor-initialization-v1',branch_ref=pin(branch_ref),version=policy['version'],
                    clock_ref=branch['clock_ref'],lineage=lineage,actor_batch=profile['actor_batch'],
                    assets={Path(r['path']).name:r for r in pub['assets']},**{k:geometry[k] for k in ('image','devices','training','base')})
    require(branch['schema']=='expanded-policy-branch-v1' and branch['optimizer_inherited'] is False
            and branch['step']==0,'expected weight-only branch')
    policy=read(branch['policy_ref']);pub=read(policy['publication_ref'])
    profile=read(pub['profile_ref']);training=validate(profile)
    parent=read(pub['parent_ref']);clock=read(branch['clock_ref'])
    assets={Path(r['path']).name:r for r in policy['assets']}
    origin=read(pub['origin_ref'])
    require(policy['kind']=='branch_initial' and policy['role']=='current' and policy['step']==0
            and pub['schema']=='expanded-initial-policy-v1' and policy['assets']==pub['assets']
            and set(assets)=={'adapter_config.json','adapter_model.safetensors'}
            and policy['artifact_version']==policy['version']=='initial-sha256:'+assets['adapter_model.safetensors']['sha256']
            and origin['files']=={n:r['sha256'] for n,r in assets.items()}
            and origin['inputs']['parent_sha256']==pub['parent_ref']['sha256']==branch['parent_checkpoint']['sha256']
            and origin['inputs']['profile_sha256']==pub['profile_ref']['sha256']
            and origin['optimizer_inherited'] is False and origin['rank']==training['lora_rank']
            and lineage and lineage!=parent['contract']['lineage']
            and clock['successful_optimizer_steps']==0 and all(not clock[k] for k in ('groups','attempts','validations'))
            and clock['config_sha256']==digest(read(branch['learning_ref'])),'branch weights, Clock or lineage differ')
    return dict(schema='resident-actor-initialization-v1',branch_ref=pin(branch_ref),
                version=policy['version'],clock_ref=branch['clock_ref'],lineage=lineage,
                image=parent['contract']['image'],devices=parent['contract']['devices'],
                training=dict(parent['contract']['training'],**training),actor_batch=profile['actor_batch'],
                base=parent['resident_identity']['base'],assets=assets)


def check_spec(spec):
    start=spec['initial_actor']
    require(start['schema']=='resident-actor-initialization-v1' and not spec.get('restore_checkpoint')
            and spec.get('formal_sampled') and spec.get('continuous_actor') and spec.get('deterministic_backward')
            and spec['continuation']['prior_step']==0
            and all(spec[k]==start[k] for k in ('training','devices','image','lineage','actor_batch')),
            'initial actor differs from sealed branch or requests optimizer restore')
    return start


def fresh_optimizer(worker):
    optimizer=worker.actor.actor_optimizer
    require(not optimizer.state and worker.actor_lr_scheduler.optimizer is optimizer,
            'branch has old Adam state or a different scheduler')
    return True


def open_start(worker,spec):
    start=check_spec(spec)
    require(worker.base_identity==start['base'] and
            getattr(worker,'initial_policy_version',None)==start['version'],
            'constructed actor did not load the branch weights and base')
    fresh_optimizer(worker)
    ref=start['branch_ref'];branch=read(ref)
    require(branch['schema'] in ('expanded-policy-branch-v1','base-node-policy-branch-v1') and branch['step']==0 and not branch['optimizer_inherited']
            and branch['clock_ref']==start['clock_ref'] and read(branch['policy_ref'])['version']==start['version'],
            'runtime branch differs')
    clock=read(start['clock_ref'])
    require(clock['successful_optimizer_steps']==0 and not clock['groups'] and not clock['attempts'],'used branch Clock')
    return ref,dict(kind='branch_initial',step=0,clock=clock,version=start['version'])


def adapter_directory(spec):
    start=check_spec(spec)
    assets=start['assets']
    directory=Path(assets['adapter_model.safetensors']['path']).parent
    require(all(pin(directory/name)==ref for name,ref in assets.items()),'staged initial adapter differs')
    return str(directory)
