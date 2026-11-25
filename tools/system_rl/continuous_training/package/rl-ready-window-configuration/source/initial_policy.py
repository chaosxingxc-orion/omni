"""Authenticate an expanded initial adapter without claiming an optimizer update."""
from pathlib import Path


def validate_initial(policy):
    from io_utils import MODEL,doc,pin,require
    from rank_profile import validate
    initial=doc(policy['publication_ref'])
    independent=initial.get('schema')=='configured-node-initial-policy-v1'
    fields=({'schema','initialization','geometry_ref','profile_ref','assets','base_manifest_ref','lineage','node'}
            if independent else {'schema','origin_ref','parent_ref','profile_ref','assets'})
    require(set(initial)==fields and (independent or initial['schema']=='expanded-initial-policy-v1'),
            'unknown branch publication')
    if independent:
        init=initial['initialization']
        require(set(init)=={'mode','source_ref'} and init['mode'] in ('base_adapter','checkpoint_weights'), 'invalid initialization config')
        origin_ref=init['source_ref']
    else:origin_ref=initial['origin_ref']
    origin=doc(origin_ref)
    folder=Path(origin_ref['path']).parent
    training=validate(doc(initial['profile_ref']))
    if independent:
        geometry=doc(initial['geometry_ref']);base=doc(initial['base_manifest_ref'])
        require(geometry['schema']=='base-node-actor-geometry-v1' and base['status']=='CLOSED_IMMUTABLE'
                and base['fixture'] is False and base['model']==policy['model']
                and geometry['base']['observed_assets']==base['assets']
                and geometry['base']['asset_manifest_ref']['sha256']==initial['base_manifest_ref']['sha256']
                and geometry['training']['world_size']==len(set(geometry['devices']))==len(geometry['devices'])
                and all(geometry['training'][k]==v for k,v in training.items())
                and initial['node'] in ('P','S1','S2','D') and bool(initial['lineage'])
                and policy['step']==0 and policy['role']=='current','independent base or actor geometry differs')
        if init['mode']=='base_adapter':
            require(origin['schema']=='independent-node-base-adapter-origin-v1' and origin['model']==base['model']
                    and origin['inputs']['base_manifest']['sha256']==initial['base_manifest_ref']['sha256']
                    and origin['actor_max_abs']==0
                    and origin['zero_B']==origin['projections']>0 and origin['loaded_tensors']==2*origin['zero_B']
                    and origin['rank']==training['lora_rank'] and origin['optimizer_inherited'] is False
                    and origin['formal_updates']==0,'wrong base adapter')
            files=origin['files']
        else:
            require(origin.get('recoverable') is True and origin['step']==origin['clock']['successful_optimizer_steps']>0
                    and origin['resident_identity']['base']==geometry['base'] and origin['contract']['lineage']!=initial['lineage']
                    and origin['world_size']==len(origin['ranks']) and origin['ranks'][0]['rank']==0,
                    'wrong checkpoint base, step or ancestry')
            folder=folder/'lora_adapter'
            files={n:origin['ranks'][0]['files']['lora_adapter/'+n] for n in ('adapter_config.json','adapter_model.safetensors')}
    else:
        files=origin['files'];parent=doc(initial['parent_ref'])
        require(origin['schema']=='expanded-lora-initialization-v1' and
                origin['inputs']==dict(parent_sha256=initial['parent_ref']['sha256'],
                     profile_sha256=initial['profile_ref']['sha256'],seed=parent['contract']['training']['seed']) and
                parent['step']==origin['source_step'] and parent['contract']['training']['lora_rank']==origin['parent_rank'] and
                origin['parent_rank']<origin['rank']==training['lora_rank'] and origin['optimizer_inherited'] is False and
                origin['formal_updates']==0 and policy['step']==0,'invalid initialization ancestry')
    assets=initial['assets']
    require(policy['model']==MODEL and policy['thinking'] is False and policy['kind']=='branch_initial'
            and policy['assets']==assets and len(assets)==2,'invalid initial policy')
    actual={Path(r['path']).name:r for r in assets}
    require(actual.keys()==files.keys()=={'adapter_model.safetensors','adapter_config.json'} and
            all(r['sha256']==files[k] and Path(r['path']).parent==folder
                and pin(r['path'])==r for k,r in actual.items()),'initial adapter bytes or location differ')
    config=doc(actual['adapter_config.json'])
    require((config['r'],config['lora_alpha'],config['lora_dropout'])==
            tuple(training[k] for k in ('lora_rank','lora_alpha','lora_dropout')) and
            policy['version']==policy['artifact_version']=='initial-sha256:'+actual['adapter_model.safetensors']['sha256'],
            'wrong configured initial adapter')
    return initial


def verify_first_update(old, new, publication):
    """Accept only the first real optimizer update from the expanded policy."""
    from io_utils import doc,require
    from rank_profile import validate
    initial=validate_initial(old)
    independent=initial['schema']=='configured-node-initial-policy-v1'
    geometry=doc(initial['geometry_ref']) if independent else None
    parent=None if independent else doc(initial['parent_ref'])
    commit=doc(publication['checkpoint_ref'])
    contract=commit['contract']
    science=publication['scientific_contract']
    training=validate(doc(initial['profile_ref']))
    trace=commit['training_lineage']
    vector=dict.fromkeys(('P','S1','S2','D'),old['version'])
    clock=commit['clock']
    successes=[(key,a) for key,a in clock['attempts'].items() if a['succeeded']]
    require(new['kind']=='checkpoint' and
            new['step']==publication['step']==commit['step']==clock['successful_optimizer_steps']==1 and
            commit['world_size']==(geometry['training']['world_size'] if independent else parent['world_size']) and
            (contract['lineage']==initial['lineage'] and all(contract[k]==geometry[k] for k in ('training','devices','image'))
                 if independent else contract['lineage']!=parent['contract']['lineage']) and
            all(science[k]==contract[k] for k in ('lineage','training','devices','image')) and
            commit['resident_identity']['base']==(geometry['base'] if independent else parent['resident_identity']['base']) and
            all(contract['training'][k]==v for k,v in training.items()) and
            trace['current_policy_vector']==trace['behavior_policy_vector']==vector and
            trace['annealing_snapshot']['successful_optimizer_steps']==0 and
            trace['annealing_snapshot']['target_nodes']==[initial['node'] if independent else 'P'] and
            len(successes)==1 and successes[0][0]==trace['attempt_id'] and
            successes[0][1]['group_ids']==trace['group_ids'] and trace['group_ids'] and
            all(clock['groups'][gid]['consumed'] is True for gid in trace['group_ids']) and
            new['assets']==publication['assets'] and
            new['version']==new['artifact_version']==publication['artifact_version'] and
            new['assets'][0]['sha256']!=old['assets'][0]['sha256'],
            'first checkpoint does not belong to the initial policy')
    return True
