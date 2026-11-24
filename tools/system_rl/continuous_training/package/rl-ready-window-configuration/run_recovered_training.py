"""Resume the original iteration after both live switch receipts and a closed resident update."""
import argparse
import json
import os
from pathlib import Path
from run_package import read, pin, write
from dependencies import require
from paired_experiment import load, check_actor, check_readers


def run(iteration, switch, runner, *, check=False):
    setup=read(runner)
    release=Path(setup['release'])
    require(Path(__import__('iteration').__file__).read_bytes()==(release/'tree/parents/jev-system-rl/implementation/iteration.py').read_bytes(),
            'original controller not running from recovery build')
    def placement(config):
        root=Path(config['output_root'])
        result=read(switch);binding=read(result['binding_ref']);area=Path(result['binding_ref']['path']).parent
        require(switch==area/'SWITCH.json' and area.parent==root/'provider-replacements' and
                pin(iteration)==binding['iteration_ref']==read(json.loads(os.environ['OMNI_ROLLOUT_HOST_GRANT_REF']))['iteration_ref']
                and read(result['released_ref'])['terminal']['activation_ref']==result['evidence_ref'],
                'different switch or unreleased activation')
        closed=read(binding['previous_release_ref'])
        failed=read(binding['rejected_ref'])
        require(closed['status']=='released' and closed['terminal']['rejected_activation_ref']==binding['rejected_ref']
                and failed['checkpoint_ref']==binding['checkpoint_ref'] and failed['unknown_requests']==0, 'different prior owner')
        roster=binding['providers'];primary=read(config['program_execution']['provider_ref'])['lane']
        require(set(roster)==set(config['formal_rollout']['provider_refs']), 'different lane roster')
        config['program_execution']['provider_ref']=roster[primary]
        rollout=config['formal_rollout']
        rollout.update(provider_refs=roster)
        rollout['continuous_training']=dict(rollout['continuous_training'],precision_lock_ref=binding['precision_ref'])
        rollout['resource_session']=dict(rollout['resource_session'],owner_suffix='session-'+pin(switch)['sha256'][:12])
        actor=config['formal_actor'];stage=read(actor['stage_template_ref'])
        evidence=read(result['evidence_ref'])
        require(evidence['precision_lock_ref']==binding['precision_ref'] and evidence['checkpoint_ref']==binding['checkpoint_ref'],
                'different active checkpoint')
        oldroot=root/'formal-resident-client'
        completed=[read(p) for p in sorted((oldroot/'completed').glob('*.json'))]
        command=completed[-1]
        require(command['checkpoint_sha256']==binding['checkpoint_ref']['sha256'] and command['owner']==read(oldroot/'OWNER.json')['target']
                and command['source_boot']==actor['source_boot'] and command['sequence']==len(completed)-1,
                'no terminal command for resumed checkpoint')
        command_root=Path(command['owner'])/'hot/resident/results'/f"{command['sequence']:06d}"
        require(Path(binding['checkpoint_ref']['path']).parents[1]==command_root,'wrong resident checkpoint root')
        # Prior publication and step identities remain visible by their original logical names.
        stage['capture_mounts']['logical_roots']['closed_actor_command']=str(command_root)
        stage['capture_mounts']['formal_input_roles'].append('closed_actor_command')
        actor['stage_template_ref']=write(area/'RESTORE-ACTOR.json',stage)
        build=read(release.parent/'TRAINING-BUILD.json');lock=read(build['source_lock_ref'])
        policy=read(setup['policy']);exp=load(pin(binding['experiment_ref'])['path'],policy['experiment_sha256'])
        check_actor(exp,stage['initial_actor'],policy)
        impl=release/'tree/parents/jev-system-rl/implementation'
        check_readers(exp,{n:pin(impl/n)['sha256'] for n in exp['readers']})
        require(pin(setup['policy'])['sha256']==lock['policy_sha256'] and build['precision_lock_ref']==actor['continuous_actor']['precision_lock_ref'],
                'recovery build lost original precision ancestry')
        actor['launcher_ref']=pin(impl.parent/'rl-ready-window-configuration/stage_bounded_actor_step.py')
        actor['continuous_actor']=dict(actor['continuous_actor'],source_lock_ref=build['source_lock_ref'],
            input_projection_source_ref=pin(impl/'resident_inputs.py'),command_source_ref=pin(impl/'formal_command.py'))
        config.update(_session_root=str(area/'session'),_resident_client_root=str(area/'actor-client'),
                      _restored_command=command,_replacement_checkpoint=binding['checkpoint_ref'],_provider_switch=result)
        return config
    if check:
        config=placement(dict(read(iteration),_path=str(iteration)))
        from formal_rollout import settings
        spec,primary=settings(config)
        return dict(ready=True,lanes=list(spec['provider_refs']),checkpoint=config['_replacement_checkpoint'])
    from iteration import run as iterate
    return iterate(iteration,resume_placement=placement)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('iteration','switch','runner'):p.add_argument(k,type=Path)
    p.add_argument('--check',action='store_true')
    print(json.dumps(run(**vars(p.parse_args()))))
