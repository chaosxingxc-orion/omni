"""Bind a real original preparation config to assembled fixed interfaces, no dispatch."""
import argparse,copy,json,sys
from pathlib import Path
from dependencies import HERE,ROOT,require
from run_package import read,pin,write


def _bindings(output, family_ref=None):
    root=Path(output);reward_dir=ROOT/'jev-system-rl/dev-reward-release'
    sys.path.insert(0,str(reward_dir))
    from configure import configure
    integration=read(configure(root/'reward'))
    reward=integration['reward_evaluation']
    program=ROOT/'collection-dispatch-integration/production-pipeline-next-version-v1'
    closure=write(root/'NODE-SOURCES.json',[pin(program/'node_contributions.py'),*read(reward['source_closure'])])
    entry=HERE/'node_entry.py'
    node=dict(builder_module=pin(program/'node_contributions.py'),metrics_pin=integration['node_metrics_pin'],
        source_closure=write(root/'NODE-CLOSURE.json',[pin(program/'node_contributions.py'),pin(entry),closure,*read(closure)]),
        sealed_entry=pin(entry),sourcepins=closure)
    data=program/'finite-capture-binding-v1'
    sources=write(root/'FINITE-SOURCES.json',[pin(data/n) for n in ('binding.py','prepare_sources.py','bounded_train_probe.py','CONFIG.json')])
    finite=dict(module=pin(data/'binding.py'),source_closure=sources)
    result=dict(reward_evaluation=reward,node_evaluation=node,finite_data_binding=finite,
                bounded_capture_binding=finite,finite_capture_binding_ref=pin(data/'CONFIG.json'))
    diagnostic=program/'finite-diagnostic-capture-v1'
    if (diagnostic/'ASSEMBLED-BINDING.json').is_file():
        seal=read(diagnostic/'ASSEMBLED-BINDING.json')
        for ref in (seal['module_ref'],seal['config_ref'],seal['source_closure_ref']):pin(ref)
        for ref in read(seal['source_closure_ref']):pin(ref)
        result.update(finite_diagnostic_binding=dict(module=seal['module_ref'],source_closure=seal['source_closure_ref']),
            finite_diagnostic_config_ref=seal['config_ref'],diagnostic_release_ref=pin(diagnostic/'ASSEMBLED-BINDING.json'))
    if family_ref is not None:
        from family_precision import validate
        spec=read(family_ref);validate(spec)
        pr=write(root/'PRECISION.family93.json',spec)
    else:
        precision=ROOT/'jev-system-rl/production-precision'
        # Selected strategies only; actual14 RCP FAILED, so the ordinary base gate
        # still decides whether any training can run. No MoE intervention imported.
        from dependencies import load
        r=load(precision/'reduction_policy.py','configured_production_reductions')
        actual=read(ROOT/'native-recompute-integration/state-repair/audio-projection-diagnostic/language-reduction-repair/NEXT-CONFIG.json')['audio_diagnostic']
        pr=write(root/'PRECISION.json',dict(schema='omni-production-row-precision-v1',audio=r.CONFIG,language=r.LANGUAGE_CONFIG,
            sources={n:pin(precision/(n+'.py')) for n in ('projection_trace','reduction_policy')},
            installed_sources=actual['installed_sources'],RCP_required=True,MoE_changed=False))
    rpc=ROOT/'jev-system-rl/production-rpc'
    result['named_rpc_overlay']=dict(files={n:pin(rpc/n) for n in ('engine_api.py','ascend_fp32_worker.py','named_rpc_worker.py','engine_bridge.py','rollout_sync.py')},precision_ref=pr)
    return result


def bindings(output, family_ref=None):
    paths=sys.path[:]
    try:return _bindings(output, family_ref)
    finally:sys.path[:]=paths


def family_selection(iteration, mode):
    """Explicit candidate selection; never silently combine legacy arithmetic."""
    policy=iteration.get('precision_policy',{})
    if 'persistent_family_ref' not in policy:return None
    require(set(policy)=={'persistent_family_ref'},'persistent family conflicts with legacy precision policy')
    require(mode=='finite_engineering','family candidate has no formal runtime acceptance')
    ref=pin(policy['persistent_family_ref'])
    from family_precision import validate
    validate(read(ref))
    return ref


def bind_precision_profile(value):
    """Consume a macro-resolved fragment through the original preparation API."""
    c=value['iteration'];ref=c.get('precision_capture_ref')
    if ref is None:return
    fragment=read(ref)
    require(set(fragment)=={'precision_policy','diagnostic_logits_capture','training_package'},
            'explicit precision/capture configuration fields')
    require(set(fragment['training_package'])=={'diagnostic_logits','diagnostic_output_gate'},
            'precision fragment cannot modify optimizer or task configuration')
    for target,fields in ((c,{k:fragment[k] for k in ('precision_policy','diagnostic_logits_capture')}),
                          (value['training_package'],fragment['training_package'])):
        for key,item in fields.items():
            require(key not in target or target[key]==item,'conflicting precision fragment: '+key)
            target[key]=copy.deepcopy(item)
    c['precision_capture_ref']=pin(ref)


def bind_family_execution(owner, iteration):
    """Compile the selected family's actual worker requirements before drain.

    Only an explicitly selected diagnostic family changes service execution.
    Conflicting options fail rather than relying on parser last-option wins.
    """
    if 'persistent_family_ref' not in iteration.get('precision_policy',{}):return
    args=owner.get('service_args')
    require(isinstance(args,list) and all(isinstance(a,str) for a in args),
            'family service requires explicit argv')
    require(not any(a.split('=',1)[0] in ('--no-enforce-eager','--async-scheduling',
            '--speculative-config') for a in args),'family worker execution conflicts with service argv')
    for flag in ('--enforce-eager','--no-async-scheduling'):
        entries=[a for a in args if a.split('=',1)[0]==flag]
        require(not entries or entries==[flag],'ambiguous family service flag: '+flag)
        if not entries:args.append(flag)


def validate_service_context(owner, iteration):
    """Reject a service that cannot honor configured full request envelopes.

    No truncation, implicit expansion, or change to the response budget. The
    local deployment profile owns max_model_len and the dataset owns bounds.
    """
    envelopes=[]
    for key in ('finite_capture_binding_ref','finite_diagnostic_config_ref'):
        if iteration.get(key):
            b=read(iteration[key])['bounds']
            envelopes.append((key,b['prompt_tokens_upper_per_call'],b['completion_tokens_upper_per_call']))
    acquisition=iteration.get('dataset_profile',{}).get('acquisition')
    if acquisition:
        b=acquisition['bounds']
        envelopes.append(('dataset_profile.acquisition',b['prompt_tokens_upper_per_call'],b['response_tokens_upper_per_call']))
    if not envelopes:return None
    args=owner['service_args'];entries=[]
    for i,arg in enumerate(args):
        if arg=='--max-model-len':
            require(i+1<len(args),'missing max-model-len value')
            entries.append(args[i+1])
        elif arg.startswith('--max-model-len='):entries.append(arg.split('=',1)[1])
    require(len(entries)==1 and entries[0].isdigit() and int(entries[0])>0,
            'one explicit positive service max-model-len required')
    length=int(entries[0])
    for name,prompt,response in envelopes:
        require(type(prompt) is int and type(response) is int and min(prompt,response)>0,
                'positive input/output envelope required: '+name)
        require(length>=prompt+response,'SERVICE_CONTEXT_TOO_SMALL:'+name+':'+str(length)+'<'+str(prompt+response))
    return dict(max_model_len=length,required_tokens=max(p+r for _,p,r in envelopes))


def bind_service_owner(value, root):
    """Carry one explicit capture policy to the actual service owner process."""
    c=value['iteration'];capture=c.get('diagnostic_logits_capture')
    if capture is None:
        require('persistent_family_ref' not in c.get('precision_policy',{}),
                'family candidate requires actual service full-logits capture')
        return
    from rollout_logits_service import validate_config
    validate_config(capture)
    capacity=capture_capacity(c,capture)
    package=value['training_package']
    require(package.get('diagnostic_logits')==capture['policy'],
            'service and training full-logits policy differ')
    from diagnostic_output_gate import validate_config as validate_gate
    gate=package.get('diagnostic_output_gate')
    require(gate is not None,'captured service requires joint training output gate')
    validate_gate(gate)
    require(gate['vocabulary']==capture['policy']['vocabulary'],
            'capture and gate vocabulary differ')
    require(gate['max_bytes']>=capacity['worker_bytes_per_call_required'],
            'output gate cannot read a complete configured response')
    certificate=package.get('budget',{}).get('checkpoint_storage_budget_ref')
    if certificate is not None:
        from diagnostic_storage import plan as storage_plan,validate_envelopes
        # Base, each candidate, and (if enabled) one replay verification.
        windows=1+c['max_optimizer_updates']+int(c.get('newprocess_replay',{}).get('enabled',False))
        payload=storage_plan(capture,capacity,package.get('diagnostic_selection',{}).get('max_records',6),windows)
        validate_envelopes(read(certificate),payload)
        one=payload['per_window']
        require(c['transports']['npu']['closed_return_max_bytes']>=one['rollout_tensor_bytes'],
                'rollout closed-return budget below complete diagnostic payload')
        require(c['transports']['npu2']['closed_return_max_bytes']>=one['training_return_payload_bytes'],
                'training closed-return budget below diagnostic input plus A/B/A payload')
        write(Path(root)/'DIAGNOSTIC-STORAGE.json',payload)
    owner=copy.deepcopy(read(value['resource_owner_config']))
    validate_service_context(owner,c)
    bind_family_execution(owner,c)
    require(capture['storage_budget_bytes']<=owner['storage_budget_bytes'],
            'four-rank logits capture exceeds actual owner storage budget')
    existing=owner.get('diagnostic_logits_capture')
    require(existing is None or existing==capture,'owner capture policy conflicts')
    owner['diagnostic_logits_capture']=copy.deepcopy(capture)
    require(owner.get('diagnostic_logits_scope','parity_only')=='parity_only',
            'new release retains full logits only for explicit parity requests')
    owner['diagnostic_logits_scope']='parity_only'
    write(Path(root)/'LOGITS-CAPACITY.json',capacity)
    value['resource_owner_config']=write(Path(root)/'ROLLOUT-OWNER.with-logits.json',owner)


def capture_capacity(iteration, capture):
    """Charge every retained TP tensor against the actual configured call caps."""
    sources=[]
    for key in ('finite_diagnostic_config_ref',):
        if iteration.get(key):
            ref=pin(iteration[key]);bounds=read(ref)['bounds']
            sources.append(dict(source_ref=ref,requests=bounds['capture_requests_upper'],
                response_tokens=bounds['completion_tokens_upper_per_call']))
    acquisition=iteration.get('dataset_profile',{}).get('acquisition')
    if acquisition:
        bounds=acquisition['bounds']
        sources.append(dict(source='dataset_profile.acquisition',requests=bounds['requests_upper'],
            response_tokens=bounds['response_tokens_upper_per_call']))
    require(sources,'explicit capture workload bounds required before service launch')
    observed_bound=None
    if 'selection' in capture:
        from logits_selection import validate as validate_selection
        observed_bound=validate_selection(capture['selection'])
    for row in sources:
        require(type(row['requests']) is int and row['requests']>0 and
                type(row['response_tokens']) is int and row['response_tokens']>0,
                'positive actual capture call/token bounds')
        row['worker_bytes_per_call']=row['response_tokens']*capture['policy']['vocabulary']*2
        row['observed_requests_upper']=min(row['requests'],observed_bound) if observed_bound else row['requests']
        row['four_rank_tensor_bytes']=4*row['observed_requests_upper']*row['worker_bytes_per_call']
    one=max(r['worker_bytes_per_call'] for r in sources)
    total=max(r['four_rank_tensor_bytes'] for r in sources)
    require(capture['policy']['max_bytes']>=one,
            'logits per-worker budget below configured completion cap: '+str(one))
    require(capture['storage_budget_bytes']>=total,
            'logits total budget below retained four-rank cohort tensors: '+str(total))
    return dict(schema='omni-configured-logits-capacity-v1',sources=sources,
        worker_bytes_per_call_required=one,tensor_bytes_per_cohort_required=total,
        actual_measured=False,metadata_and_other_owner_state_additional=True)


def configure(source,output,mode):
    root=Path(output);value=copy.deepcopy(read(source));require(value['schema']=='omni-dp8-iteration-preparation-v1','original preparation required')
    bind_precision_profile(value)
    from storage import environment
    environment(root,value['iteration']['client_storage_bytes'])
    c=value['iteration'];family_ref=family_selection(c,mode)
    b=bindings(root/'bindings',family_ref);c.update(b,execution_mode=mode)
    for key in ('authority_ref','resources_ref'):
        c['registry'][key]=pin(c['registry'][key])
    c['authorization_ref']=pin(c['authorization_ref'])
    require(c['registry']['authority_ref']==c['authorization_ref'],'registry authority differs')
    if c.get('source_bundle_ref'):c['source_bundle_ref']=pin(c['source_bundle_ref'])
    c['selection']['reward_evaluation']=b['reward_evaluation']
    if c.get('precision_policy',{}).get('moe_final_reduction'):
        from dependencies import load
        module=ROOT/'jev-system-rl/production-moe/moe_reduction_policy.py'
        policy=load(module,'configured_MoE_API')
        sources={p.name:pin(p) for p in module.parent.glob('*.py')}
        cfg=read(module.parent/'NEXT-CONFIG.json')['audio_diagnostic']
        spec=read(b['named_rpc_overlay']['precision_ref'])
        spec['moe']=dict(module_ref=pin(module),sources=sources,configuration=policy.CONFIG,installed_sources=cfg['moe_runtime_sources'],NPU_validated=False)
        c['named_rpc_overlay']['precision_ref']=write(root/'bindings/PRECISION.with-moe.json',spec)
    rows=c.get('precision_policy',{}).get('language_rows')
    if rows is not None:
        from validated_policy import bind
        packaged=pin(rows.get('packaged_ref',ROOT/'jev-system-rl/production-language-rows/PACKAGED-BINDING.json'))
        history=c.get('native_history_refs',[])
        candidate=bind(rows,packaged,history,root/'bindings')
        spec=read(c['named_rpc_overlay']['precision_ref']);spec['language_rows']=candidate
        package=read(packaged)
        if candidate.get('moe_binding'):
            cfg=package['diagnostic_configuration'];mb=candidate['moe_binding']
            require(spec['audio']==cfg['row_parallel_reduction'],'candidate audio policy differs')
            spec['sources']={n:package['worker_sources'][n+'.py'] for n in ('projection_trace','reduction_policy')}
            spec['moe']=dict(binding=mb,module_ref=mb['module'],sources=package['worker_sources'],
                configuration=cfg['moe_final_reduction'],installed_sources=cfg['moe_runtime_sources'],
                NPU_validated=candidate['NPU_validated'])
        if candidate['mode']=='active':
            validation=read(candidate['validation_ref']);cfg=validation['diagnostic_configuration']
            require(spec['audio']==cfg['row_parallel_reduction'],'validated audio policy differs')
            package=read(packaged)
            if not candidate.get('moe_binding'):
                spec['moe']=dict(module_ref=package['worker_sources']['moe_reduction_policy.py'],sources=package['worker_sources'],
                    configuration=cfg['moe_final_reduction'],installed_sources=cfg['moe_runtime_sources'],NPU_validated=True)
        c['named_rpc_overlay']['precision_ref']=write(root/'bindings/PRECISION.with-language-rows.json',spec)
    if mode=='finite_engineering':
        workflow=c.get('unified_workflow',{})
        maximum=workflow.get('max_optimizer_updates',1)
        from finite_runtime import engineering_updates
        check=dict(execution_mode=mode,formal_rl_enabled=False,max_optimizer_updates=maximum)
        if workflow.get('engineering_update_sequence'):
            check['engineering_update_sequence']=copy.deepcopy(workflow['engineering_update_sequence'])
            c['engineering_update_sequence']=copy.deepcopy(workflow['engineering_update_sequence'])
            value['training_package']['engineering_update_sequence']=copy.deepcopy(workflow['engineering_update_sequence'])
        engineering_updates(check)
        c.update(max_optimizer_updates=maximum,max_zero_signal_batches=1,diagnostic_scope='dataset_finite_probe',diagnostic_source_population=19,formal_rl_enabled=False)
        value['training_package']['formal_rl_enabled']=False
        require(c['resident']['enabled'],'original resident config required')
        require(c.get('runtime_construction'),'finite mode constructs real runtime automatically')
        bounds=read(b['finite_capture_binding_ref'])['bounds']
        c['root_seed']=read(b['finite_capture_binding_ref'])['root_seed']
        c['stage_budgets']['capture'].update(tokens_upper=max(c['stage_budgets']['capture']['tokens_upper'],bounds['accounted_capture_tokens_upper']),requests_upper=max(c['stage_budgets']['capture']['requests_upper'],8))
        require('finite_diagnostic_config_ref' in c,'DATA diagnostic seal required for finite release')
        ds=read(c['finite_diagnostic_config_ref']);db=ds['bounds']
        c['diagnostic_capture_budgets']={purpose:dict(tokens_upper=db['accounted_capture_tokens_upper'],requests_upper=db['capture_requests_upper']) for purpose in ('base_parity','updated_parity')}
        value['training_package'].update(diagnostic_coverage=ds['diagnostic_coverage'],diagnostic_selection=ds['diagnostic_selection'])
        from prompt_capacity import bind_diagnostic_budget
        bind_diagnostic_budget(c,value['training_package'],ds)
    if mode=='formal':
        c['diagnostic_scope']='strict_full';value['training_package']['formal_rl_enabled']=True
        require(all(c['stage_budgets'][stage]['requests_upper']>0 and c['stage_budgets'][stage]['tokens_upper']>0 for stage in ('dev','screen','test')),'formal full-population budgets required')
    if c.get('runtime_construction'):
        from runtime_builder import build
        for key in ('transport_config_ref','base_asset_manifest_ref','asset_program_ref'):
            if key in c['runtime_construction']:c['runtime_construction'][key]=pin(c['runtime_construction'][key])
        c['runtime_construction']['output_root']=str(root/'runtime')
        real=build(c,root/'runtime')
        c['runtime_input_ref']=pin(root/'runtime/BUILT.json')
        c['initial_input']=real['input_ref']['path']
        c['endpoint_probe_packet_ref']=real['endpoint_probe_packet_ref'];c['endpoint_probe_query_id']=real['endpoint_probe_query_id']
        # Empty nontrainable window is only a compiler carrier until DATA returns live capture.
        carrier=write(root/'EMPTY-WINDOW.json',dict(schema='finite-empty-untrainable-window-v1',unknown_requests=0,physical_requests_drained=False,training_ready=False))
        c['trajectory']=carrier;c['diagnostic_index']=None;value['training_package'].update(trajectory=carrier,learning=real['learning_ref'],base_assets=c['runtime_construction'].get('base_asset_manifest_ref',str(ROOT/'policy-iteration-successor/BASE-ASSETS.json')))
        for key in ('diagnostic_index','endpoint_witness','parity'):value['training_package'].pop(key,None)
        require(value['training_package']['budget']['storage_bytes']>=128*1024**3,'original full checkpoint budget')
        if mode=='formal':
            from finite_runtime import prepare_formal_startup
            seed=prepare_formal_startup(c,real['input_ref'],root/'formal-diagnostic-seed',read(real['input_ref'])['policy']['current_ref'])
            c['initial_input']=seed['input_ref']['path']
            c['root_seed']=read(b['finite_capture_binding_ref'])['root_seed']
            c['runtime_input_ref']=write(root/'FORMAL-BUILT.json',dict(real,input_ref=seed['input_ref'],learning_ref=seed['learning_ref']))
            ds=read(b['finite_diagnostic_config_ref']);db=ds['bounds']
            c['diagnostic_capture_budgets']={purpose:dict(tokens_upper=db['accounted_capture_tokens_upper'],requests_upper=db['capture_requests_upper']) for purpose in ('base_parity','updated_parity')}
            value['training_package'].update(diagnostic_coverage=ds['diagnostic_coverage'],diagnostic_selection=ds['diagnostic_selection'],learning=seed['learning_ref'])
            from prompt_capacity import bind_diagnostic_budget
            bind_diagnostic_budget(c,value['training_package'],ds)
    if not c.get('source_bundle_ref') and c.get('package_source_bundle',True):
        from dependencies import load
        pack=load(HERE/'stage.py','configured_original_bundle').pack
        seed=write(root/'BUNDLE-INPUT.json',value)
        if (root/'source-release/BUNDLE.json').exists():
            bundle=read(root/'source-release/BUNDLE.json')
            require(pin(root/'source-release/sources.tar')['sha256']==bundle['archive']['sha256'],'source bundle changed on restart')
        else:pack(root/'source-release',seed['path'],source_only=True)
        c['source_bundle_ref']=pin(root/'source-release/BUNDLE.json')
    from unified_config import bind as bind_unified_workflow
    bind_unified_workflow(value,mode,root)
    bind_service_owner(value,root)
    from prepare_iteration import prepare
    source_ref=write(root/'BOUND-PREPARATION.json',value)
    return prepare(source_ref,root/'entry')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--template');p.add_argument('--profile');p.add_argument('--output',required=True);p.add_argument('--mode',choices=['formal','finite_engineering'],required=True);a=p.parse_args()
    require(bool(a.config)!=bool(a.template),'use config or template plus local profile')
    source=a.config
    if a.template:
        require(a.profile,'explicit local profile required')
        from config_profile import compile_profile
        resolved=read(compile_profile(a.template,a.profile,Path(a.output)/'resolved-input'))
        source=resolved['resolved_ref']
    print(json.dumps(configure(source,a.output,a.mode)))


def bind_scoped_diagnostic_sampling(iteration,package):
    """Bind dispatch, observation and budgets to the same selected node roster."""
    spec=read(iteration['finite_diagnostic_config_ref'])
    selection=spec.get('packet_selection',{})
    if selection.get('schema')!='diagnostic-packet-selection-v2':return
    from prompt_capacity import diagnostic_quotas,diagnostic_work
    quotas=diagnostic_quotas(selection);nodes={n for n,q in quotas.items() if q}
    from diagnostic_protocol import target_nodes
    require(set(target_nodes(iteration))<=nodes,'diagnostic sampling omits training targets')
    calls,tokens=diagnostic_work(spec);bounds=spec['bounds']
    require(bounds['capture_requests_upper']==calls and bounds['capture_logical_tokens_upper']==tokens and
            bounds['accounted_capture_tokens_upper']==3*tokens,'scoped capture budget mismatch')
    capture=iteration['diagnostic_logits_capture'];observed=capture['selection']
    capture['selection']=dict(observed,schema='omni-logits-input-selection-v2',
        inputs_per_node={n:observed['inputs_per_node'][n] if n in nodes else 0 for n in quotas})
    from logits_selection import validate
    validate(capture['selection'])
    iteration['diagnostic_capture_budgets']={purpose:dict(tokens_upper=3*tokens,requests_upper=calls)
        for purpose in ('base_parity','updated_parity')}
    package.update(diagnostic_selection=copy.deepcopy(spec['diagnostic_selection']),
        diagnostic_coverage=copy.deepcopy(spec['diagnostic_coverage']))
    iteration['stage_budgets']['diagnostic']['tokens_upper']=spec['diagnostic_token_upper']
    package['budget']['diagnostic_token_upper']=spec['diagnostic_token_upper']
