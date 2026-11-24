"""Authenticate an expanded initial adapter without claiming an optimizer update."""
from pathlib import Path


def validate_initial(policy):
    from io_utils import MODEL,doc,pin,require
    from rank_profile import validate
    initial=doc(policy['publication_ref'])
    require(set(initial)=={'schema','origin_ref','parent_ref','profile_ref','assets'} and
            initial['schema']=='expanded-initial-policy-v1','unknown branch publication')
    origin=doc(initial['origin_ref']);parent=doc(initial['parent_ref'])
    training=validate(doc(initial['profile_ref']))
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
    require(actual.keys()==origin['files'].keys()=={'adapter_model.safetensors','adapter_config.json'} and
            all(r['sha256']==origin['files'][k] and Path(r['path']).parent==Path(initial['origin_ref']['path']).parent
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
    parent=doc(initial['parent_ref'])
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
            commit['world_size']==parent['world_size'] and
            contract['lineage']!=parent['contract']['lineage'] and
            all(science[k]==contract[k] for k in ('lineage','training','devices','image')) and
            commit['resident_identity']['base']==parent['resident_identity']['base'] and
            all(contract['training'][k]==v for k,v in training.items()) and
            trace['current_policy_vector']==trace['behavior_policy_vector']==vector and
            trace['annealing_snapshot']['successful_optimizer_steps']==0 and
            trace['annealing_snapshot']['target_nodes']==['P'] and
            len(successes)==1 and successes[0][0]==trace['attempt_id'] and
            successes[0][1]['group_ids']==trace['group_ids'] and trace['group_ids'] and
            all(clock['groups'][gid]['consumed'] is True for gid in trace['group_ids']) and
            new['assets']==publication['assets'] and
            new['version']==new['artifact_version']==publication['artifact_version'] and
            new['assets'][0]['sha256']!=old['assets'][0]['sha256'],
            'first checkpoint does not belong to the expanded P policy')
    return True
