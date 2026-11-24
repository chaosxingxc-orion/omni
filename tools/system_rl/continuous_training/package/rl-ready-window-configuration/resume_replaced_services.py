"""Activate a committed update on observed replacements under the original run and host guards."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from dependencies import require
from run_package import read, pin, write
from transport import restore_locations
from lock_branch_precision import normalized, content
from paired_experiment import load, check_lane, check_readers, graph
from source.continuous_policy import activate


def bind(iteration, pair, package):
    config=read(iteration)
    config['_path']=str(iteration)
    root=Path(config['output_root'])
    with (root/'iteration.guard').open('ab') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state=read(root/'CURSOR.json')
        grant=read(json.loads(os.environ['OMNI_ROLLOUT_HOST_GRANT_REF']))
        rejected_ref=pin(root/'ACTIVATION-REJECTED.json')
        rejected=read(rejected_ref)
        prior_session=root/'resident-lease'
        release=read(prior_session/'RELEASED.json')
        require(grant['iteration_ref']==state['config_ref']==pin(iteration)==rejected['iteration_ref']
                and rejected['cursor_ref']==pin(root/'CURSOR.json') and rejected['checkpoint_ref']==state['resume_ref']
                and not state['pending_stage'] and state['phase']=='endpoint' and release['status']=='released'
                and release['terminal']['rejected_activation_ref']==rejected_ref and release['terminal']['terminal_verified']
                and release['terminal']['unknown_requests']==rejected['unknown_requests']==0, 'unsettled iteration')
        for directory in config.get('inherited_closed_returns', []):
            restore_locations(directory)
        for marker in sorted(root.glob('step-*/update/returned/CLOSED-COMMIT.json')):
            restore_locations(marker.parent)
        rollout=config['formal_rollout']
        original=rollout['continuous_training']['precision_lock_ref']
        lock=read(original)
        old=Path(lock['assets']['resident_providers_ref']['path']).parent.parent
        reg=read(pair/'providers/READY.json')
        new_ref=pin(pair/'providers/READY.json')
        ready=read(reg['handoff_ref'])
        require(reg['handoff_ref']==pin(pair/'READY.json') and set(reg['providers'])==set(rollout['provider_refs'])==set(ready['lanes']),
                'incomplete replacement roster')
        pkg=read(package)
        exp=load(pin(pkg['experiment_ref'])['path'],pkg['experiment_sha256'])
        prepared=read(pair/'PREPARED.json')
        check_readers(exp,prepared['overlays'])
        old_overlay=read(old/'PREPARED.json')['overlays']
        require(prepared['graph']==graph(exp) and
                {k for k in old_overlay.keys()|prepared['overlays'].keys() if old_overlay.get(k)!=prepared['overlays'].get(k)}
                <= {'policy_identity.py','publication.py'},'changed model arithmetic')
        lanes={}
        for name,ref in reg['providers'].items():
            provider=read(ref)
            require(provider['host'] in grant['hosts'] and provider['host']==rollout['ssh']['target'], 'ungranted rollout host')
            check_lane(exp,name,read(pair/('lane-'+name)/'config.json'))
            lane=read(provider['initial_lane_ref'])
            info=read(pair/(name+'-READY.json'))
            require(ready['lanes'][name]==pin(pair/(name+'-READY.json'))['sha256'] and
                    info['container_id']==read(provider['service_ready'])['container_id'] and
                    content(info['activation']['activation_ref'])==content(lane['activation_ref']) and
                    info['precision']==read(lane['precision_ref']) and
                    normalized(old,read(old/(name+'-READY.json')))==normalized(pair,info), 'changed provider runtime')
            lanes[name]=lane
        primary=read(config['program_execution']['provider_ref'])['lane']
        area=root/'provider-replacements'/new_ref['sha256']
        asset=dict(lock['assets'],resident_providers_ref=new_ref,**{k:lanes[primary][k] for k in ('activation_ref','precision_ref')})
        precision=write(area/'PRECISION.json',dict(lock,assets=asset,equivalent_parent_ref=original))
        config['program_execution']['provider_ref']=reg['providers'][primary]
        rollout.update(provider_refs=reg['providers'])
        rollout['continuous_training']=dict(rollout['continuous_training'],precision_lock_ref=precision)
        # This one-time publication releases its lease with zero generation requests.
        rollout.pop('resource_session')
        binding=dict(iteration_ref=pin(iteration), rejected_ref=rejected_ref, checkpoint_ref=state['resume_ref'],
                     previous_session_ref=pin(prior_session/'REMOTE-LEASE.json'),previous_release_ref=pin(prior_session/'RELEASED.json'),
                     providers=reg['providers'],precision_ref=precision,experiment_ref=pkg['experiment_ref'])
        write(area/'BINDING.json',binding)
        out=area/f"policy-{state['step']:06d}"
        result=activate(config,state,out,replacement=binding)
        return write(area/'SWITCH.json', dict(binding_ref=pin(area/'BINDING.json'),policy_ref=result,
                     released_ref=pin(out/'RELEASED.json'),evidence_ref=pin(out/'ACTIVATIONS.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('iteration','pair','package'):
        p.add_argument(name,type=Path)
    print(json.dumps(bind(**vars(p.parse_args()))))
