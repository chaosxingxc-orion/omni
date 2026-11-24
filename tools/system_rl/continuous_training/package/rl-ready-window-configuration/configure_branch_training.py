"""Bind a closed training build and branch to a local controller deployment."""
import argparse
import json
import os
from pathlib import Path
from build import pin,save


def configure(config,output):
    def read(p):return json.loads(Path(p).read_bytes())
    def checked_path(ref):
        if pin(ref['path'])!=ref:raise ValueError('changed input: '+ref['path'])
        return ref['path']
    def checked(ref):
        return read(checked_path(ref))
    c=read(config);out=Path(output).resolve()
    experiment = None
    if 'package' in c:
        from paired_experiment import load
        package = read(c['package'])
        experiment = load(checked_path(package['experiment_ref']), package['experiment_sha256'])
        c['policy'] = checked_path(package['policy_ref'])
    old=read(c['manifest']);runner=read(c['runner'])
    buildfile=Path(c['build']);build=read(buildfile);release=buildfile.parent/'release'
    lock=checked(build['source_lock_ref']);precision=checked(build['precision_lock_ref'])
    start=checked(precision['accepted_initial_ref']);branch=checked(start['branch_ref'])
    policy = read(c['policy'])
    if 'experiment_sha256' in policy and experiment is None:
        raise ValueError('paired training policy requires its experiment package')
    if experiment is not None:
        from paired_experiment import check_actor, check_readers
        check_actor(experiment, start, policy)
        impl = release / 'tree/parents/jev-system-rl/implementation'
        check_readers(experiment, {name: pin(impl / name)['sha256'] for name in experiment['readers']})
        if checked(branch['policy_ref'])['model'] != experiment['model']:
            raise ValueError('branch model differs from experiment')
    inputs=pin(c['branch_input'])
    if (checked(inputs)['branch_initial_ref']!=start['branch_ref'] or
            build['build_complete_ref']!=pin(release/'BUILD-COMPLETE.json') or
            lock['policy_sha256']!=pin(c['policy'])['sha256'] or
            pin(c['providers'])!=precision['assets']['resident_providers_ref']):
        raise ValueError('training build, providers or branch differ')
    parents=release/'tree/parents';impl=parents/'jev-system-rl/implementation'
    src=impl.parent/'rl-ready-window-configuration'
    # Install the experiment policy itself beside the receipts, never rely on
    # a mutable development path in the controller or its container.
    pol=save(out/'TRAINING-POLICY.json',read(c['policy']))
    if pol['sha256']!=lock['policy_sha256']:
        raise ValueError('noncanonical training policy copy')
    dep=read(runner['deployment'])
    dep.update(source_lock_ref=build['source_lock_ref'],precision_lock_ref=build['precision_lock_ref'],
               policy_ref=pol,input_projection_source_ref=pin(impl/'resident_inputs.py'),command_source_ref=pin(impl/'formal_command.py'))
    manifest=dict(old,run_id=start['lineage'],branch_input_ref=inputs,branch_actor_source=c['actor_source'],
                  actor_capture_root=c['actor_capture_root'],actor_batch_ref=save(out/'ACTOR-BATCH.json',start['actor_batch']),
                  continuous_deployment_ref=save(out/'DEPLOYMENT.json',dep),release=str(release),live_provider_ready_path=c['providers'])
    if experiment is not None:
        manifest['experiment_ref'] = save(out/'EXPERIMENT.json', experiment)
        manifest['values'] = dict(manifest['values'], TRAIN_MEMORY_GIB=experiment['training_memory_gib'],
                                  FORMAL_ROLLOUT_LANES=list(experiment['lanes']))
        runner['lanes'] = list(experiment['lanes'])
    manifest.pop('branch_plan_ref',None)
    if c.get('undispatched_release'):manifest['closed_undispatched_ref']=pin(c['undispatched_release'])
    manifest['values']=dict(manifest['values'],FORMAL_RUN_ROOT=str(out/'run'),TRAIN_LAUNCHER_REF=pin(src/'stage_bounded_actor_step.py'))
    if manifest.get('release_capture_binding'):
        manifest['release_capture_binding']=dict(manifest['release_capture_binding'],release_ref=pin(parents/'fresh-cohort-successor/cohort_capture.py'))
    save(out/'MANIFEST.json',manifest)
    runner.update(deployment=str(out/'DEPLOYMENT.json'),grant=str(out/'HOST-GRANT.json'),policy=pol['path'],
                  compiler=str(src/'assemble_sampled_p.py'),release=str(release),build_log=str(buildfile))
    # The cursor keeps the writer's logical trajectory path. Only that sealed
    # file is needed to seed this run; the old optimizer shards stay offline.
    parent=checked(branch['parent_cursor_ref']);trajectory=parent['trajectory_ref']
    logical=trajectory['path'];prior=checked(branch['policy_ref'])
    commit=checked(prior['publication_ref'])['parent_ref']
    physical=Path(commit['path'])
    oldlogical=Path(branch['parent_checkpoint']['path'])
    if physical.parts[-len(oldlogical.parts[1:]):]!=oldlogical.parts[1:]:
        raise ValueError('parent checkpoint is not a relocated closed return')
    files=physical.parents[len(oldlogical.parts)-2];table=files.parent
    manifest_sha=read(table/'CLOSED-COMMIT.json')
    if (not manifest_sha['immutable'] or
            manifest_sha.get('manifest_sha256',manifest_sha.get('manifest_ref',{}).get('sha256'))!=pin(table/'CLOSED-MANIFEST.json')['sha256']):
        raise ValueError('unsealed predecessor trajectory')
    route=Path('files')/Path(logical).relative_to('/')
    if read(table/'CLOSED-MANIFEST.json')[str(route)]!=trajectory:
        raise ValueError('parent trajectory missing from immutable return')
    source=(table/route).resolve(strict=True);identity=pin(source)
    if not source.is_relative_to(files) or any(identity[k]!=trajectory[k] for k in ('bytes','sha256')):
        raise ValueError('changed predecessor trajectory')
    runner['mounts']=dict(runner['mounts'],**{logical:str(source)})
    return save(out/'RUNNER.json',runner)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key,env in [('config','OMNI_BRANCH_TRAINING_CONFIG'),('output','OMNI_TRAIN_OUTPUT')]:
        default=os.environ.get(env);p.add_argument('--'+key,type=Path,default=default,required=not default)
    p.add_argument('--profile',type=Path,default=os.environ.get('OMNI_DEPLOYMENT_PROFILE'))
    p.add_argument('--release',type=Path,default=os.environ.get('OMNI_TRAIN_RELEASE'))
    a=p.parse_args()
    if a.profile:
        import sys
        if not a.release:p.error('profile resolution needs --release or OMNI_TRAIN_RELEASE')
        sys.path.insert(0,str(a.release/'tree/parents/jev-system-rl/implementation'))
        from source.config_profile import compile_profile
        from run_package import read
        a.config=Path(read(compile_profile(a.config,a.profile,a.output/'resolution'))['resolved_ref']['path'])
    print(json.dumps(configure(a.config,a.output)))
