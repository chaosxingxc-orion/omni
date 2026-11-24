"""Seal the measured initial policy and the equivalent deployed arithmetic."""
from pathlib import Path
from run_package import pin,read,write
from dependencies import require


def content(ref):
    return {k:ref[k] for k in ('bytes','sha256')}


def normalized(folder,ready):
    p=ready['precision'];root=folder/'bundle'
    def observed(name,ref):
        actual=pin(root/name)
        require(content(actual)==content(ref),'worker precision reference differs: '+name)
        return read(actual)
    rows=observed('ROWS.json',p['rows_ref'])
    spec=observed('PRECISION-SPEC.json',p['specification_ref'])
    delta_ref=spec.pop('adapter_rank_delta_ref')
    delta=observed('rank-precision/delta.json',delta_ref)
    delta['parent_spec_ref']=content(delta['parent_spec_ref'])
    for name,change in delta['changes'].items():
        require(change['result_ref']==spec['sources'][name],'rank precision source differs')
        change['result_ref']=content(change['result_ref'])
        spec['sources'][name]=change['result_ref']
    for w in p['workers']:
        require(w['family'].pop('adapter_rank_delta_ref')==delta_ref,'rank worker source differs')
    require(p['actual_device_execution'] and [w['rank'] for w in p['workers']]==list(range(len(p['workers']))),
            'incomplete precision worker receipt')
    return rows,spec,delta,p['workers']


def seal(cfg,output):
    from initial_action_review import review
    diag=Path(cfg['diagnostic']);formal=Path(cfg['residents'])
    init=read(cfg['actor_start']);prev=read(cfg['previous']);reg=read(formal/'providers/READY.json')
    action=diag/'ACTIONS.json';result=diag/'VERDICT.json'
    review(action,diag/'actor',result);verdict=read(result)
    require(verdict['nonzero']==0 and verdict['max_abs']==0 and verdict['tokens']>=verdict['rows']>0 and
            verdict['policy']==init['version'] and read(diag/'actor/EXIT.json')['complete_ranks']==len(init['devices']),
            'incomplete same-action evaluation')
    reference=read(diag/'a-READY.json')
    measured=normalized(diag,reference)
    runtime=read(formal/'PREPARED.json')
    require(all(runtime[k]==read(diag/'PREPARED.json')[k] for k in ('overlays','graph')) and
            runtime['graph']['max_num_seqs']==verdict['rows'],'different execution profile')
    handoff=read(formal/'READY.json')
    require(reg['handoff_ref']==pin(formal/'READY.json') and set(reg['providers'])==set(handoff['lanes']),
            'provider roster differs from deployment')
    lanes={};cards=set()
    for key,ref in reg['providers'].items():
        deployed=read(formal/(key+'-READY.json'));provider=read(ref)
        lane=read(provider['initial_lane_ref']);act=read(lane['activation_ref']);cache=read(lane['cache_ref'])
        devices=set(act['service_identity']['physical_devices'])
        require(not cards.intersection(devices) and len(devices)==len(measured[-1])
                and pin(formal/(key+'-READY.json'))['sha256']==handoff['lanes'][key]
                and read(provider['service_ready'])['container_id']==deployed['container_id']
                and deployed['activation']['activation_ref']['sha256']==lane['activation_ref']['sha256']
                and act['policy_version']==verdict['policy']==deployed['activation']['version']
                and content(act['weights_ref'])==content(init['assets']['adapter_model.safetensors'])
                and content(act['profile_ref'])==content(prev['assets']['profile_ref'])
                and reference['activation']['service_identity']['image']==act['service_identity']['image']
                and read(lane['precision_ref'])==deployed['precision']
                and all(cache[k] for k in ('prefix_cache','kv_cache','reset_succeeded','audio_processor_reset','audio_encoder_reset'))
                and normalized(formal,deployed)==measured,'formal lane differs from measured policy: '+key)
        cards.update(devices);lanes[key]=lane
    first=next(iter(lanes.values()))
    return write(output,dict(schema='omni-expanded-precision-v1',purpose='precision_alignment',
        accepted_initial_ref=pin(cfg['actor_start']),previous_ref=pin(cfg['previous']),
        assets=dict(profile_ref=prev['assets']['profile_ref'],activation_ref=first['activation_ref'],precision_ref=first['precision_ref'],
                    qualification_ref=pin(result),resident_providers_ref=pin(formal/'providers/READY.json')),
        accepted_verdict=dict(verdict,passed=True),updated_policy_measured=False,
        source_refs={n:pin(Path(__file__).with_name(n)) for n in ('initial_action_review.py','qualify_profile_batch.py','lock_branch_precision.py')},
        performance_claim=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(seal(read(a.config),a.output))
