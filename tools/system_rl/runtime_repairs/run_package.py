"""Config -> sealed DP8 package; derive window/learning/ref/lease wiring once.

This compiler never contacts hosts or launches devices. The run supervisor uses
this same compiler for each sealed window returned by the original data owner.
"""
import argparse
import copy
import json
from pathlib import Path
import shlex
from dependencies import HERE, ROOT, require, sha, local_path
from storage import environment, commit


def pin(path):
    if isinstance(path,dict):
        require(sha(path['path']) == path['sha256'], 'input pin changed')
        path=path['path']
    original=Path(path)
    p=local_path(original).resolve()
    return dict(path=str(original) if str(original) != str(p) and original.is_absolute() else str(p),sha256=sha(p),bytes=p.stat().st_size)


def read(path):
    return json.loads(local_path(pin(path)['path']).read_text())


def write(path,value):
    if Path(path).exists():
        require(json.loads(Path(path).read_text())==value,'immutable receipt changed: '+str(path))
        return pin(path)
    commit(path,value)
    return pin(path)


def compile_package(config_path,output):
    spec=read(config_path); output=Path(output).resolve()
    require(spec['schema']=='omni-dp8-run-package-v1','run package schema')
    if (output/'RUN.json').exists():
        run=read(output/'RUN.json');require(run['config_ref']==pin(config_path),'existing package config changed')
        return pin(output/'RUN.json')
    require(not output.exists(),'partial compilation retained: inspect original output')
    environment(output,spec['budget']['storage_bytes'])
    runtime=read(spec.get('runtime',HERE/'runtime.json'))
    runtime['storage_budget_bytes']=spec['budget']['storage_bytes']
    if spec.get('communication'):runtime['communication']=copy.deepcopy(spec['communication'])
    if spec['lifecycle'].get('docker_storage'):
        from docker_storage import validate
        validate(spec['lifecycle']['docker_storage'])
        runtime['docker_storage']=copy.deepcopy(spec['lifecycle']['docker_storage'])
    runtime['checkpoint_storage_mode']=spec.get('checkpoint_storage_mode','full')
    if spec.get('checkpoint_integrity'):runtime['checkpoint_integrity']=spec['checkpoint_integrity']
    require(runtime['checkpoint_storage_mode'] in ('full','lora_training_state'),'unknown checkpoint storage_mode')
    if spec['budget'].get('checkpoint_storage_budget_ref'):runtime['checkpoint_storage_budget_ref']=pin(spec['budget']['checkpoint_storage_budget_ref'])
    from storage import current_scope
    if spec.get('storage_scope') or current_scope():runtime['storage_scope']=spec.get('storage_scope') or current_scope()
    runtime.update(execution_mode=spec.get('execution_mode','formal'),diagnostic_scope=spec.get('diagnostic_scope','strict_full'),diagnostic_source_population=spec.get('diagnostic_source_population'))
    if spec.get('diagnostic_scope')=='dataset_profile':
        runtime['dataset_profile']=copy.deepcopy(spec['dataset_profile'])
    runtime['action']=spec['action']
    runtime['max_optimizer_updates']=spec['budget']['max_optimizer_updates']
    if spec.get('engineering_update_sequence'):
        runtime['engineering_update_sequence']=copy.deepcopy(spec['engineering_update_sequence'])
    runtime['formal_rl_enabled']=spec.get('formal_rl_enabled',False)
    runtime['update_mode']=spec.get('update_mode','bounded_first_update')
    if spec.get('newprocess_replay'):
        runtime['newprocess_replay']=copy.deepcopy(spec['newprocess_replay'])
        if spec['action']=='acceptance':require(runtime['newprocess_replay']['input_ref'], 'actual replay source input required')
    if spec.get('resident'):
        runtime['resident']=copy.deepcopy(spec['resident'])
    if spec.get('resident_resume_state_ref'):
        runtime['resident_resume_state_ref']=pin(spec['resident_resume_state_ref'])
    hosts=spec.get('host_bindings',dict(npu='npu',npu2='npu2'))
    require(spec['resources']['training_host']==hosts['npu2'] and spec['resources']['training_devices']==list(range(8)) and
            spec['resources']['rollout_host']==hosts['npu'] and spec['resources']['rollout_devices']==list(range(4)), 'fixed resource boundary')
    trajectory=read(spec['trajectory'])
    if 'window_ref' in trajectory: window=read(trajectory['window_ref'])
    elif trajectory.get('schema')=='selected-fresh-dataset-v1':
        require(trajectory['training_ready'] and not trajectory['synthetic_fixture'],'real trainable completed cohort required')
        native=read(trajectory['native_config_ref']);selected=read(native['selected_cohort_ref'])
        consumer=read(trajectory['consumer_config_ref'])
        learning=read(native['learning_config_ref']);version=learning['kernel']['current_policy_vector']
        require(len(set(version.values()))==1,'single shared adapter policy required')
        cohort_root=Path(pin(native['selected_cohort_ref'])['path']).parent
        if local_path(cohort_root/'INPUT.json').exists():
            source_input=read(cohort_root/'INPUT.json')
            if source_input.get('phase_frozen_upstream_ref'):spec.setdefault('frozen_upstream_release',source_input['phase_frozen_upstream_ref'])
        online=read(native['online_by_call_ref'])
        task_online=write(output/'ONLINE.json',[dict(identity=x['identity'],online=x['online_input']) for x in online])
        window=dict(schema='omni-dp8-sealed-window-v1',files=read(trajectory['files_ref']),
            consumer_config_ref=trajectory['consumer_config_ref'],task_sidecar_ref=native['task_sidecar_ref'],
            task_online_ref=task_online,group_ids=selected['group_ids'],attempt_id=native['attempt_id'],
            policy_version=next(iter(version.values())),loaded_rollout_policy_version=next(iter(version.values())),
            actor_policy_version=next(iter(version.values())),physical_requests_drained=True,unknown_requests=0,
            rollout_devices=list(range(4)),max_policy_staleness=0,selected_cohort_ref=native['selected_cohort_ref'],
            trajectory_ref=pin(spec['trajectory']))
        # Terminal physical drain is checked again from lifecycle receipts at
        # admission; a closed dataset alone is never resource ownership.
        trajectory=dict(trajectory,learning_ref=native['learning_config_ref'])
    else: window=copy.deepcopy(trajectory)
    if spec.get('pending_capture'):
        native_capture=read(spec['pending_capture']);pending=read(native_capture['pending_capture_ref'])
        captured=read(pending['input_ref'])
        require(not captured.get('diagnostic_evidence_only') or spec['action']=='diagnostic',
                'diagnostic evidence cannot be finalized or trained')
        if spec['action']=='diagnostic':
            current=read(captured['policy']['current_ref']);version=current.get('artifact_version',current['version'])
            window.update(policy_version=version,actor_policy_version=version,loaded_rollout_policy_version=version)
            spec['endpoint_witness']=captured['capture']['endpoint_policy_ref']
        if captured.get('phase_frozen_upstream_ref'):spec['frozen_upstream_release']=captured['phase_frozen_upstream_ref']
    if spec.get('diagnostic_index'):
        from diagnostic_index import records
        window['diagnostic_records']=records(spec['diagnostic_index'])
    if spec.get('diagnostic'):
        # Already retained real A/B records only; never synthesize tokens or LPs.
        window['diagnostic_records']=copy.deepcopy(spec['diagnostic']['records'])
    if spec.get('diagnostic_scope')=='dataset_profile':
        for key in ('profile_inputs_ref','current_inputs_ref'):
            if key in trajectory:window[key]=pin(trajectory[key])
        if window.get('current_inputs_ref'):runtime['dataset_profile']['current_inputs_ref']=window['current_inputs_ref']
    if spec.get('frozen_upstream_release'):
        window['frozen_upstream_release_ref']=pin(spec['frozen_upstream_release'])
    if spec.get('learning'):
        runtime['learning_ref']=pin(spec['learning'])
    elif 'learning_ref' in trajectory:
        runtime['learning_ref']=pin(trajectory['learning_ref'])
    else: runtime['learning_ref']=None
    runtime['resume_ref']=pin(spec['resume']) if spec.get('resume') else None
    if spec.get('base_assets'):runtime['model']['base_assets_ref']=pin(spec['base_assets'])
    elif spec.get('endpoint_witness'):
        witness=read(spec['endpoint_witness']);policy=read(witness['policy_ref'])
        if policy['kind']=='base':runtime['model']['base_assets_ref']=policy['asset_manifest_ref']
        else:runtime['model']['base_assets_ref']=read(policy['publication_ref'])['resident_identity']['base']['asset_manifest_ref']
    require(runtime['model'].get('base_assets_ref'),'existing immutable base assets required')
    action=spec['action'];require(action in ('diagnostic','acceptance','finalize'), 'fixed actual DP8 action')
    if action=='acceptance':
        if trajectory.get('schema')=='selected-fresh-dataset-v1':
            selected_input=cohort_root/'INPUT.json'
            if selected_input.exists():require(not read(selected_input).get('diagnostic_evidence_only'),'evidence-only cohort cannot train')
        require(runtime['learning_ref'] is not None,'existing learning input required')
        consumer=read(window['consumer_config_ref'])
        require(consumer['learning_config_ref']==runtime['learning_ref'],'consumer/learning drift')
        # Reserve complete existing groups using the one task Clock; do not ask
        # MAIN to construct a future snapshot or sidecar hash.
        import dependencies
        task=dependencies.upstream()
        from scorer_extension import install
        install(task)
        learning=read(runtime['learning_ref'])
        prior=read(consumer['clock_state_ref'])
        clock=task.create_clock(learning,prior,consumer_config=consumer)
        snapshots=[clock.reserve(g) for g in window['group_ids']]
        require(snapshots and all(s==snapshots[0] for s in snapshots),'one successful-step snapshot per window')
        snapshot=snapshots[0]
        consumer['clock_state_ref']=write(output/'CLOCK.json',clock.state)
        window['consumer_config_ref']=write(output/'CONSUMER.json',consumer)
        window.update(policy_step=snapshot['successful_optimizer_steps'],snapshot_sha256=snapshot['snapshot_sha256'])
        if spec.get('parity'):window['updated_rcp_ref' if runtime['update_mode']=='train' else 'base_rcp_ref']=pin(spec['parity'])
        # Source-owned sidecar and actual online input stay immutable. Never fill
        # absent labels or probabilities to make a window trainable.
        for key in ('task_sidecar_ref','task_online_ref'):
            require(key in window, 'source trajectory export missing '+key)
            pin(window[key])
    require(window.get('unknown_requests',0)==0,'unknown window cannot be resubmitted')
    if spec.get('endpoint_witness'):window['endpoint_witness_ref']=pin(spec['endpoint_witness'])
    runtime['window_ref']=write(output/'WINDOW.json',window)
    runtime['package_config_ref']=pin(config_path)
    roots=set(spec.get('input_mounts',[])+[str(output),str(Path(pin(config_path)['path']).parent)])
    def collect(value):
        if isinstance(value,dict):
            if {'path','sha256'}<=value.keys():roots.add(str(Path(value['path']).resolve().parent))
            else:
                for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
    collect(window);collect(spec)
    mapped_targets={str(Path(m['target']).parent) for m in spec.get('input_file_mounts',[])}
    runtime['input_mounts']=sorted(roots-mapped_targets)
    runtime['input_file_mounts']=spec.get('input_file_mounts',[])
    runtime['deadline_epoch']=spec['budget']['deadline_epoch']-spec['budget']['restore_seconds']
    runtime['logical_token_limit']=spec['budget']['logical_tokens']
    from budget_counts import logical_count
    runtime['prior_logical_tokens_upper']=logical_count(spec['budget'].get('prior_logical_tokens_upper',0))
    runtime['diagnostic_token_upper']=spec['budget'].get('diagnostic_token_upper',0)
    runtime['update_token_upper']=spec['budget'].get('update_token_upper',0)
    from budget import check as check_runtime_budget
    check_runtime_budget(runtime)
    require(runtime['diagnostic_token_upper']>0 if action=='diagnostic' else runtime['update_token_upper']>0, 'explicit finite conservative work bound required')
    require(runtime['prior_logical_tokens_upper']+max(runtime['diagnostic_token_upper'],runtime['update_token_upper'])<=runtime['logical_token_limit'],'package exceeds cumulative work budget')
    if spec.get('pending_capture'):
        runtime['pending_capture_ref']=pin(spec['pending_capture'])
        runtime['generation_inventory_ref']=pin(spec['generation_inventory'])
    for key in ('diagnostic_selection','diagnostic_coverage','diagnostic_logits','diagnostic_output_gate','rollout_logits_index_ref','diagnostic_protocol','diagnostic_target_nodes'):
        if key in spec:runtime[key]=copy.deepcopy(spec[key])
    from diagnostic_protocol import enabled, target_nodes
    if enabled(runtime):
        nodes=target_nodes(runtime)
        if nodes!=['P','S1','S2','D']:
            require(runtime['learning_ref'] is not None,'node diagnostic requires learning contract')
            learning=read(runtime['learning_ref'])
            require(all(p['rollout_scope']=='node' and p['target_nodes']==nodes and
                p['trainable_mask']=={n:n in nodes for n in ('P','S1','S2','D')} for p in learning['phases']),
                'narrow diagnostic must match the entire node training contract')
    if runtime.get('diagnostic_logits'):
        from diagnostic_logits import validate
        validate(runtime['diagnostic_logits'])
        require(runtime['diagnostic_logits']['max_bytes']<=runtime['storage_budget_bytes'],
                'diagnostic logits budget exceeds whole-run storage budget')
    if runtime.get('diagnostic_output_gate'):
        from diagnostic_output_gate import validate_config
        validate_config(runtime['diagnostic_output_gate'])
        require(runtime.get('diagnostic_logits') and
                runtime['diagnostic_output_gate']['vocabulary']==runtime['diagnostic_logits']['vocabulary']==152064,
                'Omni full vocabulary capture/gate must agree before dispatch')
    if action=='diagnostic' and spec.get('pending_capture'):
        from diagnostic_index import prepare_generation
        (output/'cpu-diagnostic').mkdir(parents=True,exist_ok=True)
        window['diagnostic_records']=prepare_generation(runtime,output/'cpu-diagnostic')
        if runtime.get('diagnostic_scope')=='dataset_profile':
            from profile_producer import publish_current
            window.update(publish_current(window['diagnostic_records'],runtime,output/'cpu-profile'))
        runtime['diagnostic_prepared_on_cpu']=True
        runtime['window_ref']=write(output/'WINDOW-DIAGNOSTIC.json',window)
        # Selected CPU artifacts must be visible in every fixed worker.
        runtime['input_mounts']=sorted(set(runtime['input_mounts']+[str(output/'cpu-diagnostic')]))
    elif action=='diagnostic' and window.get('diagnostic_records'):
        from diagnostic_index import select_retained
        window['diagnostic_records']=select_retained(window['diagnostic_records'],runtime,output/'DIAGNOSTIC-SELECTION.json')
        if runtime.get('diagnostic_scope')=='dataset_profile':
            from profile_producer import publish_current
            window.update(publish_current(window['diagnostic_records'],runtime,output/'cpu-profile'))
        runtime['window_ref']=write(output/'WINDOW-DIAGNOSTIC.json',window)
    if action=='acceptance' and runtime.get('diagnostic_scope')=='dataset_profile':
        require(window.get('profile_inputs_ref') and window.get('current_inputs_ref'),'original finalized current profile missing')
        read(window['profile_inputs_ref']);read(window['current_inputs_ref'])
    runtime['host_hot_root']=str(output/'hot')
    if spec.get('_resident_attach'):
        runtime['host_hot_root']=str(Path(spec['_resident_attach']['target'])/'control/package/hot')
    runtime_ref=write(output/'RUNTIME.json',runtime)
    request=dict(schema='omni-dp8-fixed-run-v1',config_ref=pin(config_path),runtime_ref=runtime_ref,
                 action=action,resources=spec['resources'],budget=spec['budget'],
                 dataset_ref=pin(spec['dataset']),system_ref=pin(spec['system']),trajectory_ref=pin(spec['trajectory']),
                 lifecycle=spec['lifecycle'],stage_root=spec.get('stage_root'),
                 automatic_optimizer_retry=False,source_sha256={p.name:sha(p) for p in HERE.glob('*.py')})
    ref=write(output/'RUN.json',request)
    launcher=(Path(spec['stage_root'])/'parents/jev-system-rl/implementation') if spec.get('stage_root') else HERE
    command=['python3','-B',str(launcher/'control.py'),'--operation','supervise','--config',ref['path'],'--hot',str(output.parent/'supervision'),'--execute']
    if spec.get('stage_root'):command += ['--stage-root',spec['stage_root']]
    write(output/'COMMAND.json',dict(argv=command,NPU_dispatched=False))
    print(shlex.join(command));return ref


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();compile_package(a.config,a.output)
if __name__=='__main__':main()
