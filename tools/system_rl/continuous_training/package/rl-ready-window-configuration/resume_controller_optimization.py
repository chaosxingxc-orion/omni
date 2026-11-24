"""Resume the original controller around an already acknowledged resident command."""
import argparse,json,os
from pathlib import Path

def run(iteration, switch, parent_runner, runner, package, captured_step, *, check=False):
    from dependencies import load, require
    from run_package import read, pin
    import iteration as engine
    new=read(runner);old=read(parent_runner);pkg=read(package)
    release=Path(new['release']);prior=Path(old['release'])
    build=read(release.parent/'TRAINING-BUILD.json');parent=read(prior.parent/'TRAINING-BUILD.json')
    require(parent['precision_lock_ref']==build['precision_lock_ref'] and
        read(build['source_lock_ref'])['policy_sha256']==pin(new['policy'])['sha256']==pkg['policy_ref']['sha256'],
        'different arithmetic baseline or rollout build')
    package_exp=read(pkg['experiment_ref']);binding=read(read(switch)['binding_ref'])
    original=read(binding['experiment_ref'])
    allowed={'version','group_preparation','activation_concurrency','row_encoding','capacity_interval_bytes','transfer_policy'}
    require({k:v for k,v in original.items() if k not in allowed} ==
            {k:v for k,v in package_exp.items() if k not in allowed}, 'training geometry or reader changed')
    for name in ('iteration.py','resident_inputs.py','formal_command.py','session_lease.py'):
        require(pin(prior/'tree/parents/jev-system-rl/implementation'/name)['sha256']==
                pin(release/'tree/parents/jev-system-rl/implementation'/name)['sha256'], 'changed lifecycle: '+name)
    require(Path(new['grant']).read_bytes()==Path(old['grant']).read_bytes() and
            new['mounts']==old['mounts'] and new['rollout_target']==old['rollout_target'], 'different grant or location')
    require(pin(engine.__file__)['sha256']==pin(release/'tree/parents/jev-system-rl/implementation/iteration.py')['sha256'],
            'controller is outside intended build')
    recovery=load(pin(old['recovery_entry_ref'])['path'],'parent_recovery_entry')
    # Use the parent's placement resolver, but retain the same iteration.run and sole lock.
    def configure(cfg):
        state=read(Path(cfg['output_root'])/'CURSOR.json')
        require(state['config_ref']==pin(iteration) and state['step']>=captured_step-1,'unexpected cursor/iteration')
        stepdir=Path(cfg['output_root'])/f'step-{captured_step:06d}'/'update'
        ack=read(stepdir/'COMMAND-ACK.json')
        intent=read(stepdir/'SUBMIT-INTENT.json')
        require(ack['command_ref']==pin(stepdir/'COMMAND.json') and read(ack['command_ref'])==intent['command'] and
                ack['remote']['source_boot']==cfg['formal_actor']['source_boot'] and
                read(Path(cfg['_resident_client_root'])/'OWNER.json')['target']==intent['target'] and
                ack['remote']['sequence']==intent['sequence'],'no prior resident acknowledgement')
        if state['step']==captured_step-1:
            existing = read(stepdir/'REQUEST.json')['run']
            for key in ('transfer_policy','capacity_interval_bytes'):
                if key in existing:
                    cfg['formal_actor'][key] = existing[key]

            require(state['phase']=='update' and state['pending_stage']['stage']=='update' and
                    intent['command']['checkpoint_sha256']==state['resume_ref']['sha256'] and
                    state['capture']['generation_inventory_ref']==intent['command']['generation_inventory_ref'],
                    'wrong outstanding command')
        policy=read(new['policy'])
        require(all(policy[k]==package_exp[k] for k in ('capacity_interval_bytes','activation_concurrency')),
                'controller settings differ from paired experiment')
        # Keep the original command's immutable REQUEST and trainer inputs. The
        # next command picks up the new checkpoint capacity accounting settings.
        if state['step']>=captured_step:
            if 'transfer_policy' in policy:
                cfg['formal_actor']['transfer_policy'] = policy['transfer_policy']
            cfg['formal_actor']['capacity_interval_bytes']=policy['capacity_interval_bytes']
        cfg['formal_rollout']['continuous_training']['activation_concurrency']=policy['activation_concurrency']
        return cfg
    def placement_call(path, *, resume_placement):
        bound=configure(resume_placement(dict(read(iteration),_path=str(iteration.resolve()))))
        if check:
            return dict(run=bound['run_id'],actor_build=parent['source_lock_ref']['sha256'],
                        controller_build=build['source_lock_ref']['sha256'],retained_command=captured_step,
                        activation_concurrency=bound['formal_rollout']['continuous_training']['activation_concurrency'])
        # Inject the new setting at the first subsequent command, not into the
        # closed or in-flight immutable REQUEST of the retained one.
        import formal_actor_transport as transport
        original_train=transport.train
        def train(config,state,directory,guard):
            if state['step'] >= captured_step and not (directory/'REQUEST.json').exists():
                config=dict(config,formal_actor=dict(config['formal_actor'],
                    capacity_interval_bytes=package_exp['capacity_interval_bytes'],
                    **({'transfer_policy': package_exp['transfer_policy']} if 'transfer_policy' in package_exp else {})))
            return original_train(config,state,directory,guard)
        transport.train=train
        return real_run(path,resume_placement=lambda _:bound)
    real_run=engine.run
    engine.run=placement_call
    try:return recovery.run(iteration,switch,parent_runner)
    finally:engine.run=real_run

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('iteration','switch','parent_runner','runner','package'):p.add_argument(k,type=Path)
    p.add_argument('captured_step',type=int);p.add_argument('--check',action='store_true')
    print(json.dumps(run(**vars(p.parse_args()))))
