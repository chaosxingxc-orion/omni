"""Data-only workflow switches on the original preparation and iteration graph."""
import copy
from pathlib import Path


def bind(value,mode,output):
    from run_package import read,write,pin
    from dependencies import require
    c=value['iteration'];package=value['training_package'];spec=c.get('unified_workflow')
    if c.get('checkpoint_storage'):apply_storage(c)
    if not spec:return
    require(spec['schema']=='omni-unified-workflow-v1','workflow schema')
    kind=spec['kind'];require(kind in ('finite_update','restore_compare','full_devtest'),'workflow kind')
    if spec.get('sampled_training'):
        require(kind=='full_devtest' and mode=='formal','sampled completion requires formal full workflow')
        c['sampled_training']=copy.deepcopy(spec['sampled_training'])
    require(spec['population']==dict(TRAIN=119881,DEV=8690,TEST=13078),'official population cannot shrink')
    budget=read(spec['checkpoint_budget_ref'])
    lora=budget['schema']=='omni-lora-training-state-budget-v1'
    v3=budget['schema']=='omni-full-fsdp-storage-budget-v3'
    if lora:
        require(c.get('checkpoint_storage_mode')=='lora_training_state','storage_mode must match budget')
        package['checkpoint_storage_mode']='lora_training_state'
        require(budget['retained_training_states']>0 and budget['retained_full_checkpoints']==0,'exact small-state retention required')
    require(lora or budget['schema'] in ('omni-full-fsdp-storage-budget-v2','omni-full-fsdp-storage-budget-v3') and budget['retained_full_checkpoints']>=3,'recoverable A1/A2/B2 budget required')
    require(budget['reserve_bytes']>=20*1024**3 and budget['reserve_fraction']>=.10 and budget['inode_fraction']>=.05,'absolute and fractional reserve required')
    require(spec.get('max_optimizer_updates',1)<=budget['max_optimizer_updates'],'retention bound must cover every configured step')
    source=budget['roles']['source_181']['workload_budget_bytes']
    client=budget['roles']['client']['workload_budget_bytes']
    if c.get('archive',{}).get('enabled'):
        require(budget['archive_enabled'],'archive staging must be included in client admission')
        client+=budget['roles']['archive_client_additional']['workload_budget_bytes']
    if lora:
        require(kind in ('finite_update','restore_compare','full_devtest'),'configured LoRA workflow required')
        require(spec['max_optimizer_updates']==budget['max_optimizer_updates'],'exact configured update bound')
        c['resident']['checkpoint_interval']=budget['checkpoint_interval']
        c['resident']['max_commands']=budget['resident_max_commands']
        if c.get('deployment_preparation'):
            c['deployment_preparation']['resident']['checkpoint_interval']=budget['checkpoint_interval']
            c['deployment_preparation']['resident']['max_commands']=budget['resident_max_commands']
        require(c.get('archive',{}).get('enabled') and c['archive'].get('full_checkpoint'),'complete exact-state archive required')
        require(c['archive'].get('same_client_mount_and_inventory') is True,'one client CAS required')
        c['checkpoint_cache_root']=str(Path(c['output_root'])/'checkpoint-inventory')
        c['archive'].update(checkpoint_cache_root=c['checkpoint_cache_root'],local_staging_bytes=client)
        c['checkpoint_retention']=dict(budget['retention'],one_full_source=False,release_archived_A2_before_B=False)
        c['client_noncheckpoint_limit_bytes']=budget['roles']['client']['noncheckpoint_limit_bytes']
        host=c['transports']['npu2']
        host.update(storage_bytes=source,noncheckpoint_limit_bytes=budget['roles']['source_181']['noncheckpoint_limit_bytes'],
                    max_full_checkpoints=budget['retained_training_states'],max_scope_tasks=budget['roles']['source_181']['max_source_tasks'])
        package['budget'].update(storage_bytes=source,checkpoint_storage_budget_ref=pin(spec['checkpoint_budget_ref']))
        c['client_storage_bytes']=client
    elif v3:
        require(kind=='restore_compare','v3 peak covers only this bounded two-step replay lifecycle')
        require(c.get('archive',{}).get('enabled') and c['archive'].get('full_checkpoint'),'v3 requires independent full permanent archive')
        require(c.get('checkpoint_retention',{}).get('release_archived_A2_before_B') and c['checkpoint_retention'].get('one_full_source'),'v3 requires one-full-source archive release and rehydration hooks')
        require(c['archive'].get('same_client_mount_and_inventory') is True,'v3 has one configured client mount and inventory')
        c['checkpoint_cache_root']=str(Path(c['output_root'])/'checkpoint-inventory')
        c['archive']['checkpoint_cache_root']=c['checkpoint_cache_root']
        c['archive']['local_staging_bytes']=client
        c['client_noncheckpoint_limit_bytes']=budget['roles']['client']['noncheckpoint_limit_bytes']
        host=c['transports']['npu2']
        require(host.get('noncheckpoint_roots'),'v3 requires explicit stage/projection paths in source mount envelope')
        host['storage_bytes']=source
        host['noncheckpoint_limit_bytes']=budget['roles']['source_181']['noncheckpoint_limit_bytes']
        host['max_full_checkpoints']=1
        host['max_scope_tasks']=budget['roles']['source_181']['max_source_tasks']
        package['budget']['storage_bytes']=source
        package['budget']['checkpoint_storage_budget_ref']=pin(spec['checkpoint_budget_ref'])
        c['client_storage_bytes']=client
    else:require(source>=budget['checkpoint_bytes_upper']*3,'protect all full recovery states')
    package['budget']['storage_bytes']=max(package['budget']['storage_bytes'],source)
    c['client_storage_bytes']=max(c['client_storage_bytes'],client)
    # Remote workload and client staging have different mounts and free checks.
    # Source bundle limit remains the original small bounded source/input TAR.
    for logical_host,host in c['transports'].items():
        host['client_storage_bytes']=client
        if logical_host=='npu' and lora:
            rollout=budget['roles']['rollout_175']
            host['storage_bytes']=max(host.get('storage_bytes',0),rollout['workload_budget_bytes'])
            host['closed_return_max_bytes']=max(host.get('closed_return_max_bytes',0),rollout['workload_budget_bytes'])
            host['noncheckpoint_limit_bytes']=rollout['noncheckpoint_limit_bytes']
        if logical_host=='npu2':
            host['storage_bytes']=max(host['storage_bytes'],source)
            host['closed_return_max_bytes']=max(host.get('closed_return_max_bytes',0),source)
    if kind=='finite_update':
        require(mode=='finite_engineering' and not spec.get('newprocess_replay',False),'finite branch cannot authorize formal replay')
        from finite_runtime import engineering_updates
        probe=dict(execution_mode=mode,formal_rl_enabled=False,max_optimizer_updates=spec['max_optimizer_updates'])
        if spec.get('engineering_update_sequence'):
            probe['engineering_update_sequence']=copy.deepcopy(spec['engineering_update_sequence'])
        updates=engineering_updates(probe)
        require(budget['max_optimizer_updates']==updates,'finite certificate must cover exact update sequence')
        require(not spec.get('dataset_profile') and not c.get('bounded_restore_compare',False),
                'finite update cannot inherit formal profile or restore admission')
        c.update(execution_mode=mode,formal_rl_enabled=False,max_optimizer_updates=updates,
                 max_zero_signal_batches=1,diagnostic_scope='dataset_finite_probe',
                 newprocess_replay=dict(enabled=False))
        package.update(execution_mode=mode,formal_rl_enabled=False,diagnostic_scope='dataset_finite_probe')
        package['budget']['max_optimizer_updates']=updates
        if probe.get('engineering_update_sequence'):
            c['engineering_update_sequence']=copy.deepcopy(probe['engineering_update_sequence'])
            package['engineering_update_sequence']=copy.deepcopy(probe['engineering_update_sequence'])
    else:
        require(mode=='formal' and spec.get('dataset_profile'),'formal registered-profile configuration required')
        require(c.get('execution_mode',mode)==mode,'formal workflow execution mode differs')
        c['execution_mode']=mode
        c['diagnostic_scope']='dataset_profile';c['formal_rl_enabled']=True
        c['dataset_profile']=copy.deepcopy(spec['dataset_profile'])
        package.update(diagnostic_scope='dataset_profile',dataset_profile=copy.deepcopy(spec['dataset_profile']),formal_rl_enabled=True)
        package['diagnostic_selection']=copy.deepcopy(spec['diagnostic_selection'])
        from prompt_capacity import bind_diagnostic_budget
        bind_diagnostic_budget(c,package,spec)
        c['max_optimizer_updates']=spec['max_optimizer_updates']
        package['budget']['max_optimizer_updates']=spec['max_optimizer_updates']
        c['newprocess_replay']=dict(enabled=kind=='restore_compare')
        c['bounded_restore_compare']=spec.get('bounded_train_probe',False)
        if c['bounded_restore_compare']:
            require(kind=='restore_compare' and spec.get('bounded_train_probe') is True,'bounded TRAIN requires restore_compare')
        # This bounded storage plan permits one zero-signal terminal cohort in
        # either formal branch; repeated empty graphs need a separate budget.
        c['max_zero_signal_batches']=1
        if kind=='restore_compare':
            require(spec['max_optimizer_updates']==2,'continuous two steps required')
            require(c['resident']['enabled'] and c['resident']['checkpoint_interval']==1,'original full checkpoint each step')
        else:
            require(spec['max_optimizer_updates']>0,'positive full run budget')
        # Existing stage/evaluator enforce exact official DEV/TEST denominators.
        for stage in ('dev','screen','test'):
            b=c['stage_budgets'][stage]
            require(b['requests_upper']>0 and b['tokens_upper']>0,'full population stage needs explicit measured budget')
    if c.get('sampled_training'):
        from sampled_training import specification
        specification(c)
    write(Path(output)/'UNIFIED-WORKFLOW.json',dict(configuration=spec,checkpoint_budget_ref=pin(spec['checkpoint_budget_ref']),
        independent_gates=['same_policy_forward_and_behavior_raw_RCP','finite_optimizer_publication_and_newprocess_restore','full_DEV_selection_frozen_TEST_business_gain'],
        engineering_group_limit=dict(groups=2,G=4,claim='execution_only_no_business_gain'),dispatch=False))


def apply_storage(c):
    """Reapply storage deployment after original deployment_prepare, before seal."""
    from dependencies import require
    d=c['checkpoint_storage'];paths=d['paths']
    require(d['schema']=='checkpoint-storage-reuse-deployment-v1','storage deployment schema')
    require(d['hosts']['archive_client'] not in (d['hosts']['training'],d['hosts']['rollout']),'independent archive client required')
    require(c['host_bindings']==dict(npu2=d['hosts']['training'],npu=d['hosts']['rollout']),'storage host identity differs')
    require(Path(c['output_root']).is_relative_to(paths['client_hot']),'client run outside configured storage root')
    anchor=Path(paths['source_anchor'])
    require(all(Path(paths[k]).is_relative_to(anchor) for k in ('source_hot','source_stage','source_projection')),'storage roots escape original source anchor')
    source=c['transports']['npu2'];source.update(storage_anchor=str(anchor),run_parent=paths['source_hot'],stage_parent=paths['source_stage'],noncheckpoint_roots=d['source_noncheckpoint_roots'])
    c.setdefault('archive',{}).update(d['archive']);c['archive']['root']=paths['archive_root']
    require(all(k in c['archive'] for k in ('client_source_ref','primitives_source_ref','acceptance_ref','max_bytes','timeout_seconds')),'complete original accepted archive client configuration required')
    c['checkpoint_retention']=d['checkpoint_retention']
    projections=c.get('resident',{}).get('input_projections',[])
    for i,item in enumerate(projections):item['source']=str(Path(paths['source_projection'])/str(i))
    if projections:
        future=c['resident'].setdefault('future_readonly_roots',[])
        if paths['source_hot'] not in future:future.append(paths['source_hot'])
