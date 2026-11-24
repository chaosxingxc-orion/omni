"""Storage-only compiler for the original preparation/configure_release entry.

No task execution, host contact or alternate controller. Business configuration
comes exclusively from the caller's original template and explicit local profile.
"""
import argparse,copy,json
from pathlib import Path
from dependencies import HERE,JEV,ROOT,require
from run_package import read,write,pin
from config_profile import compile_profile
from checkpoint_budget import calculate
from unified_config import bind


def configure(template,profile,deployment_template,deployment_profile,budget_config,output):
    root=Path(output)
    # Explicit user policy removes aggregate quotas through the original
    # numeric APIs; no finite diagnostic is relabelled as dataset completion.
    from completion_limits import profile as completion_profile
    initial_profile=read(profile)
    if initial_profile['values'].get('RUN_LIMIT_POLICY') is not None:
        original_profile=profile
        effective=completion_profile(initial_profile)
        write(root/'USER-PROFILE.json',initial_profile)
        profile=write(root/'COMPLETION-PROFILE.json',effective)['path']
        if deployment_profile==original_profile:deployment_profile=profile
        else:
            deployment_profile=write(root/'COMPLETION-DEPLOYMENT-PROFILE.json',
                completion_profile(read(deployment_profile)))['path']
    original_resolution=read(compile_profile(template,profile,root/'original-input'))
    # All later consumers of the same profile see the first frozen environment,
    # including resumed runs after a shell or scheduler change.
    frozen_profile=original_resolution['profile_snapshot_ref']
    shared_profile=frozen_profile if original_resolution['schema']=='omni-profile-resolution-v2' else profile
    deployment_input=shared_profile if deployment_profile==profile else deployment_profile
    deployment_resolution=read(compile_profile(deployment_template,deployment_input,root/'storage-input'))
    value=copy.deepcopy(read(original_resolution['resolved_ref']))
    deployment=read(deployment_resolution['resolved_ref'])
    require(value['schema']=='omni-dp8-iteration-preparation-v1','original preparation schema required')
    require(deployment['schema']=='checkpoint-storage-reuse-deployment-v1','storage deployment schema')
    c=value['iteration'];workflow=c['unified_workflow']
    require(workflow['kind'] in ('finite_update','restore_compare','full_devtest'),'original configured workflow required')
    mode='finite_engineering' if workflow['kind']=='finite_update' else 'formal'
    if mode=='finite_engineering':
        require(not workflow.get('newprocess_replay',False),'finite engineering cannot authorize replay')
        from finite_runtime import engineering_updates
        engineering_updates(dict(execution_mode=mode,formal_rl_enabled=False,
            max_optimizer_updates=workflow['max_optimizer_updates'],
            **({'engineering_update_sequence':workflow['engineering_update_sequence']}
               if workflow.get('engineering_update_sequence') else {})))
    budget_resolution=read(compile_profile(budget_config,shared_profile,root/'budget-input'))
    budget_input=read(budget_resolution['resolved_ref']);budget_input['_config_path']=budget_resolution['resolved_ref']['path']
    budget_input['workflow_kind']=workflow['kind']
    budget_input['max_source_tasks']=max(budget_input['max_source_tasks'],8*workflow['max_optimizer_updates']+12)
    budget_input['resident_max_commands']=budget_input['max_source_tasks']
    require(budget_input['max_optimizer_updates']==workflow['max_optimizer_updates'],'workflow/budget steps differ')
    from configure_release import bind_precision_profile,capture_capacity
    bind_precision_profile(value)
    if mode=='finite_engineering':
        ref=pin(ROOT/'collection-dispatch-integration/production-pipeline-next-version-v1/finite-diagnostic-capture-v1/CONFIG.json')
        require(not c.get('finite_diagnostic_config_ref') or pin(c['finite_diagnostic_config_ref'])==ref,
                'finite diagnostic budget must match original configured binding')
        c['finite_diagnostic_config_ref']=ref
        value['training_package']['diagnostic_selection']=copy.deepcopy(read(ref)['diagnostic_selection'])
    capture=c.get('diagnostic_logits_capture')
    if capture is not None:
        from diagnostic_storage import plan as diagnostic_plan
        workload=dict(c)
        if workflow.get('dataset_profile'):workload['dataset_profile']=workflow['dataset_profile']
        capacity=capture_capacity(workload,capture)
        windows=1+workflow['max_optimizer_updates']+int(workflow['kind']=='restore_compare')
        payload=diagnostic_plan(capture,capacity,
            workflow.get('diagnostic_selection',value['training_package'].get('diagnostic_selection',{})).get('max_records',6),windows)
        for role,size in payload['roles'].items():
            items=budget_input['noncheckpoint_envelopes'][role]
            items['complete_diagnostic_logits']=max(items.get('complete_diagnostic_logits',0),size)
        write(root/'DIAGNOSTIC-STORAGE.json',payload)
    if deployment.get('storage_mode')=='lora_training_state':
        require(budget_input['schema']=='lora-training-state-budget-v1','LoRA storage needs exact small-state budget')
        c['checkpoint_storage_mode']='lora_training_state'
        value['training_package']['checkpoint_storage_mode']='lora_training_state'
        deployment['checkpoint_retention'].update(one_full_source=False,release_archived_A2_before_B=False)
        c['checkpoint_integrity']=deployment['checkpoint_integrity']
        value['training_package']['checkpoint_integrity']=deployment['checkpoint_integrity']

    budget=calculate(budget_input,HERE/'worker_dp8.py',JEV/'src/multimodal_jev/context_training_checkpoint.py')
    docker=c.get('deployment_preparation',{}).get('training_docker_storage')
    if docker:
        from docker_storage import validate
        validate(docker)
        require(docker['max_source_tasks']==budget['roles']['source_181']['max_source_tasks'],
                'Docker and original source task bounds differ')
    budget_ref=write(root/'BUDGET.json',budget)
    paths=deployment['paths']
    old_client=Path(c['output_root']).parent
    if c.get('deployment_preparation'):
        c['resident']=copy.deepcopy(c['deployment_preparation']['resident'])
    for item in c.get('resident',{}).get('input_projections',[]):
        if Path(item['target']).is_relative_to(old_client):item['target']=str(Path(paths['client_hot'])/Path(item['target']).relative_to(old_client))
    c['output_root']=str(Path(paths['client_hot'])/c['run_id'])
    c['checkpoint_storage']=copy.deepcopy(deployment)
    workflow['checkpoint_budget_ref']=budget_ref
    before=dict(selection=copy.deepcopy(c['selection']),precision_policy=copy.deepcopy(c.get('precision_policy')),stage_budgets=copy.deepcopy(c['stage_budgets']))
    bind(value,mode,root/'workflow')
    require(before==dict(selection=c['selection'],precision_policy=c.get('precision_policy'),stage_budgets=c['stage_budgets']),'storage compiler modified business gates')
    # The original configure_release runtime-builder carrier still checks its
    # legacy 128GiB input before bind() installs the certified 72GiB run budget.
    value['training_package']['budget']['storage_bytes']=max(128*1024**3,value['training_package']['budget']['storage_bytes'])
    launch_profile=copy.deepcopy(read(frozen_profile));values=launch_profile['values']
    # The fragment is already the selected numerical policy. Do not reinsert
    # an unrelated legacy macro when the fixed launcher resolves its output.
    if c.get('precision_capture_ref'):
        values['PRECISION_POLICY']=copy.deepcopy(c['precision_policy'])
    values['CLIENT_HOT']=paths['client_hot'];values['CLIENT_STORAGE_BYTES']=c['client_storage_bytes']
    if c.get('checkpoint_storage_mode')=='lora_training_state':
        values['CHECKPOINT_STORAGE_MODE']='lora_training_state';values['CHECKPOINT_BUDGET_REF']=budget_ref
    if values.get('LAUNCHER'):
        # Include generated immutable budget/template and accepted client proof.
        from config_profile import resolve
        launcher=resolve(values['LAUNCHER'],launch_profile)
        extra=[str(root.resolve()),*[str(Path(c['archive'][k]['path']).parent) for k in ('client_source_ref','primitives_source_ref','acceptance_ref')]]
        # The explicit precision fragment may live outside the source namespace.
        # Preserve its pins and expose external referenced files read-only;
        # logical namespace refs remain supplied by the sealed tree overlay.
        if c.get('precision_capture_ref'):
            fragment_ref=pin(c['precision_capture_ref']);extra.append(str(Path(fragment_ref['path']).parent))
            def fragment_mounts(item):
                if isinstance(item,dict):
                    if 'path' in item and 'sha256' in item:
                        ref=pin(item);p=Path(ref['path'])
                        logical=Path(values.get('WORKSPACE',root))/'delivery/system-rl-transition-v1'
                        if not p.is_relative_to(logical):extra.append(str(p.parent))
                    else:
                        for child in item.values():fragment_mounts(child)
                elif isinstance(item,list):
                    for child in item:fragment_mounts(child)
            fragment_mounts(read(fragment_ref))
        launcher['readonly_roots']=list(dict.fromkeys(launcher['readonly_roots']+extra))
        values['LAUNCHER']=launcher
    if c.get('deployment_preparation'):
        d=c['deployment_preparation'];d.update(training_hot=paths['source_anchor'],client_storage_bytes=c['client_storage_bytes'],
            rollout_storage_bytes=budget['roles']['rollout_175']['workload_budget_bytes'])
        d['resident']=copy.deepcopy(c['resident'])
        values['DEPLOYMENT']=copy.deepcopy(d)
    profile_ref=write(root/'LAUNCH-PROFILE.local.json',launch_profile)
    output_ref=write(root/'STORAGE-BOUND-PREPARATION.json',value)
    launch_template=copy.deepcopy(value)
    if c.get('deployment_preparation'):launch_template['iteration']['deployment_preparation']='${DEPLOYMENT}'
    if values.get('PRECISION_POLICY'):launch_template['iteration']['precision_policy']='${PRECISION_POLICY}'
    launch_template_ref=write(root/'STORAGE-LAUNCH.template.json',launch_template)
    return write(root/'STORAGE-CONFIGURATION.json',dict(schema='checkpoint-storage-configuration-v1',original_resolution_ref=pin(root/'original-input/RESOLUTION.json'),
        deployment_resolution_ref=pin(root/'storage-input/RESOLUTION.json'),budget_ref=budget_ref,preparation_ref=output_ref,launch_profile_ref=profile_ref,launch_template_ref=launch_template_ref,
        unchanged_business_fields=before,execution_mode=mode,dispatch=False,accepted_updates=0,next_original_entry='launch_fixed.py --build <composed-build> --template STORAGE-LAUNCH.template.json --profile LAUNCH-PROFILE.local.json --mode '+mode+' --output <new-local-output>'))


def main():
    p=argparse.ArgumentParser()
    for name in ('template','profile','deployment-template','deployment-profile','budget-config','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(configure(a.template,a.profile,a.deployment_template,a.deployment_profile,a.budget_config,a.output)))
if __name__=='__main__':main()
