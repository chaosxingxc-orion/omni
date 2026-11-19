"""Evidence-only extension of the sealed finite selector and original capture."""
import copy
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
IMPL = HERE.parents[2] / 'jev-system-rl/implementation'
sys.path.insert(0, str(IMPL))
from dependencies import require, load
from run_package import read, pin, write
NODES = ('P','S1','S2','D')


def parent(spec):
    for ref in read(spec['parent_closure_ref']): pin(ref)
    return load(spec['parent_module_ref']['path'], 'finite_diagnostic_parent')


def bounds(spec):
    b = copy.deepcopy(spec['bounds'])
    require(type(b['groups']) is int and 2 <= b['groups'] <= 19 and b['G'] == 4, 'finite real19 G4 groups')
    # Original rails: one P, ten S1 folds, at most one S2 per legal type, one D.
    require(b['S1_folds'] == 10 and b['S2_types_upper'] == 55, 'original business topology bound')
    from prompt_capacity import diagnostic_work
    calls,logical=diagnostic_work(spec)
    require(b['capture_requests_upper'] == calls and b['capture_logical_tokens_upper'] == logical and
            b['accounted_capture_tokens_upper'] == 3*logical, 'all-system cost bound differs')
    return b


def prepare_input(input_ref, output, *, config_ref, policy_ref=None, witness_ref=None,
                  purpose='updated_parity', cpu_contract_only=False):
    spec=read(config_ref); b=bounds(spec); root=Path(output)
    require(spec['schema']=='finite-diagnostic-capture-v1', 'diagnostic config schema')
    require(purpose in ('base_parity','updated_parity'), 'explicit diagnostic purpose')
    original=read(input_ref)
    require(original.get('prepare_mode') in ('bounded_train_probe','finite_diagnostic_capture'), 'sealed finite prepared input required')
    fixture=original.get('cpu_contract_only') or original['policy'].get('fixture')
    require(cpu_contract_only == bool(fixture), 'explicit CPU policy boundary required')
    current_ref=policy_ref or original['policy']['current_ref']; current=read(current_ref)
    reference=read(original['policy']['reference_ref'])
    require(current['model']==reference['model']=='qwen3-omni-30b-a3b-instruct' and current['thinking'] is reference['thinking'] is False, 'business baseline differs')
    require((current['kind']=='base' and current['step']==0) if purpose=='base_parity' else
            (current['kind']=='checkpoint' and current['step']>0), 'purpose/current policy mismatch')
    require(reference['kind']=='base' and reference.get('frozen') and reference.get('role')=='reference', 'independent frozen base reference')
    require(cpu_contract_only or not current.get('fixture') and not reference.get('fixture'), 'fixture policy cannot capture')
    learning=read(original['learning_config_ref']); state=read(original['clock_state_ref'])
    require(state['successful_optimizer_steps']==current['step'], 'candidate/input Clock step differs; migrate with original next_input first')
    require(learning['kernel']['reference_policy_vector']=={n:reference['version'] for n in NODES}, 'frozen reference vector differs')
    require(learning['kernel']['current_policy_vector']==learning['kernel']['behavior_policy_vector']=={n:current['version'] for n in NODES}, 'current vector differs')
    from next_input import diagnostic_input
    capture_mode=spec.get('input_mode','system_rollout')
    require(capture_mode in ('system_rollout','frozen_node_inputs'),'explicit diagnostic input mode')
    from prompt_capacity import diagnostic_quotas
    evidence_nodes=([n for n,q in diagnostic_quotas(spec['packet_selection']).items() if q]
                    if spec.get('packet_selection') else list(NODES))
    derived=diagnostic_input(input_ref,None,current_ref,witness_ref,root/'derived',root_seed=spec['root_seed'],sampling_round=0,cohort_sequence=1 if purpose=='base_parity' else 2,
        evidence_nodes=evidence_nodes if capture_mode=='frozen_node_inputs' else None)
    source=read(derived)
    if original.get('diagnostic_evidence_only'):
        source['diagnostic_evidence_only']=copy.deepcopy(original['diagnostic_evidence_only'])
    source['diagnostic_evidence_only'].update(purpose=purpose,trainable=False)
    if capture_mode=='frozen_node_inputs':
        source['diagnostic_evidence_only'].update(input_mode=capture_mode,
            frozen_upstream_ref=pin(original['phase_frozen_upstream_ref']),
            response_policy='fresh_current_policy_sampling_only')
    if spec.get('packet_selection'):
        require(capture_mode=='frozen_node_inputs','diagnostic packet selection requires independent inputs')
        source['diagnostic_packet_selection']=copy.deepcopy(spec['packet_selection'])
    # Keep the original TRAIN selector's source/custody, never old response tokens.
    source['prepare_mode']='finite_diagnostic_capture'
    source['finite_diagnostic_capture']=dict(config_ref=pin(config_ref),purpose=purpose,groups=b['groups'],G=b['G'])
    source['sampling'].update(purpose='finite_diagnostic_capture',sample_count=b['groups'],replacement='forbid',error_success=None)
    source['policy']['runtime_evidence_ref']=witness_ref
    source['capture'].update(endpoint_policy_ref=witness_ref,retry=False,unknown='journal_reconcile_never_resend',runtime_storage_budget_bytes=b['capture_storage_bytes'])
    source['native'].update(enabled=False)
    source['authorization'].update(model_calls=False,native_training=False)
    require(source['budget']['total_tokens'] >= source['budget']['cumulative_prior_tokens_upper'] + b['accounted_capture_tokens_upper'], 'cumulative diagnostic token authority insufficient')
    source['budget'].update(max_requests=b['capture_requests_upper'],prompt_tokens_upper_per_call=b['prompt_tokens_upper_per_call'],capture_logical_tokens_upper=b['capture_logical_tokens_upper'],native_logical_tokens_upper=0)
    if b.get('prompt_tokens_upper_by_node'):
        source['budget']['prompt_tokens_upper_by_node']=copy.deepcopy(b['prompt_tokens_upper_by_node'])
    if b.get('completion_tokens_upper_by_node'):
        source['budget']['completion_tokens_upper_by_node']=copy.deepcopy(b['completion_tokens_upper_by_node'])
        source['budget']['completion_tokens_upper_per_call']=b['completion_tokens_upper_per_call']
    if spec.get('context_limit'):
        context=spec['context_limit'];consumer=read(source['consumer_config_ref'])
        consumer['context_limit']=context
        source['consumer_config_ref']=write(root/'CONSUMER.capacity.json',consumer)
        source['prompt_context_limit']=context
    prepared_ref=write(root/'INPUT.json',source)
    inventory_module=load(pin(spec['inventory_module_ref'])['path'],'finite_diagnostic_inventory')
    inventory_ref=inventory_module.build(prepared_ref,spec,root/'inventory')
    return dict(input_ref=prepared_ref,inventory_ref=inventory_ref,diagnostic_evidence_only=source['diagnostic_evidence_only'],training_ready=False,capture_ready=False,endpoint_witness_required=witness_ref is None,bounds=b,formal_coverage_satisfied=False)


def install(cohort, spec):
    """Reuse the r33 deterministic selector with its real sealed admission audit."""
    import types
    old=cohort.draw
    require('original_draw' in old.__code__.co_freevars, 'original fresh_adapter wrapper required')
    selector_ref=pin(spec['selector_module_ref'])
    original_selector=load(selector_ref['path'],'diagnostic_original_r33')
    formal=old.__closure__[old.__code__.co_freevars.index('original_draw')].cell_contents
    def select(config,root,**kwargs):
        if config.get('prepare_mode')!='finite_diagnostic_capture':return formal(config,root,**kwargs)
        from policy import validate_scope
        validate_scope(config,fixture=False)
        require(config['diagnostic_evidence_only']['trainable'] is False and not config['native']['enabled'] and not config['authorization']['native_training'], 'diagnostic-only isolation required')
        mode=config['finite_diagnostic_capture']; b=bounds(spec)
        require(mode['groups']==b['groups'] and mode['G']==4, 'finite diagnostic bounds drift')
        # Authenticate and select with the unchanged original selector. Its
        # node-P constraints are evaluated on the frozen TRAIN seed config,
        # while actual prepare/reservations use the derived diagnostic Clock.
        seed=copy.deepcopy(config)
        seed.update(cohort_id=config['cohort_id'],native=copy.deepcopy(config['native']),policy=copy.deepcopy(config['policy']))
        seed['capture']=copy.deepcopy(config['capture'])
        seed.update(prepare_mode='bounded_train_probe',rollout_scope='node')
        seed['sampling']['purpose']='bounded_train_probe'
        seed['sampling']['sample_count']=b['groups']
        lr=copy.deepcopy(read(config['learning_config_ref']))
        ph=lr['phases'][0]
        ph.update(rollout_scope='node',target_nodes=['P'],trainable_mask={n:n=='P' for n in NODES},frozen_upstream_release=seed['phase_frozen_upstream_ref']['sha256'])
        seed['learning_config_ref']=write(Path(root)/'SELECTOR-SCOPE.json',lr)
        # Fixed r33 algorithm; only the validated group count is a parameter.
        seed['bounded_train_probe']['groups']=b['groups']
        plan=original_selector.draw(seed,root,fixture=False,groups=b['groups'])
        require(len(plan['draws'])==b['groups'],'original selector did not honor finite group bound')
        from policy_identity import digest
        version=digest(dict(original_selection=plan['version'],diagnostic=config['finite_diagnostic_capture'],policy=config['policy'],learning=config['learning_config_ref'],clock=config['clock_state_ref']))
        plan.update(version=version,sampling=copy.deepcopy(config['sampling']),policy=copy.deepcopy(config['policy']),prepare_mode='finite_diagnostic_capture',diagnostic_evidence_only=copy.deepcopy(config['diagnostic_evidence_only']),training_ready=False)
        for item in plan['draws']:
            item.update(query_draw_id=version+':draw:'+str(item['draw_index']),group_id=version+':diagnostic-group:'+str(item['draw_index']))
        plan['group_ids']=[d['group_id'] for d in plan['draws']]
        plan['statistics']['selected_groups']=b['groups']
        from storage import commit
        commit(Path(root)/'SELECTED.json',plan,replace=True)
        return plan
    cells=list(old.__closure__);cells[old.__code__.co_freevars.index('original_draw')]=types.CellType(select)
    cohort.draw=types.FunctionType(old.__code__,old.__globals__,old.__name__,old.__defaults__,tuple(cells))
    # Original prepare creates disabled native metadata; mark it explicitly.
    prepare=cohort.prepare
    def isolated(config,root,**kwargs):
        result=prepare(config,root,**kwargs)
        if config.get('prepare_mode')=='finite_diagnostic_capture':
            path=Path(root)/'NATIVE.pending.disabled.json'
            if path.exists():
                value=read(path);value.update(diagnostic_evidence_only=config['diagnostic_evidence_only'],training_ready=False,enabled=False)
                from storage import commit
                commit(path,value,replace=True)
        return result
    cohort.prepare=isolated
    return cohort


def generate(request_ref, context):
    require(context['owner_check']() is not False, 'original live owner required')
    request=copy.deepcopy(read(request_ref)); config=read(request['config_ref'])
    require(request['stage']=='capture' and request['fixture'] is False, 'real capture request required')
    purpose=request.get('extra',{}).get('purpose')
    require(purpose in ('base_parity','updated_parity'), 'diagnostic capture purpose required')
    spec_ref=config['finite_diagnostic_config_ref'];spec=read(spec_ref);b=bounds(spec)
    require(request['budget']['requests_upper']>=b['capture_requests_upper'] and request['budget']['tokens_upper']>=b['accounted_capture_tokens_upper'], 'all-system reservation too small')
    root=Path(request['output_root'])/'finite-diagnostic'
    prepared=prepare_input(request['input_ref'],root/'prepared',config_ref=spec_ref,policy_ref=request['policy_ref'],witness_ref=request['endpoint_witness_ref'],purpose=purpose)
    source=read(prepared['input_ref']); p=parent(spec)
    witness=p.validate_witness(prepared['input_ref'],request['endpoint_witness_ref'])
    require(source['execution_registry']==config['registry'], 'original Registry drift')
    authority=read(request['authorization_ref']);require(authority['expires_at']==source['authorization']['expires_at'], 'authority expiry drift')
    source['authorization']=dict(authority,model_calls=False,native_training=False)
    require(witness['endpoint']==context['endpoint'], 'endpoint witness drift')
    transport=read(source['capture']['transport_config_ref']);transport['endpoint']=context['endpoint']
    source['capture']['transport_config_ref']=write(root/'TRANSPORT.json',transport)
    business=read(source['capture']['business_config_ref']);business['index_reuse']=p.index_config(read(spec['parent_config_ref']),root)
    source['capture']['business_config_ref']=write(root/'BUSINESS.json',business)
    request['input_ref']=write(root/'INPUT.json',source);request['finite_parent_request_ref']=pin(request_ref)
    effective=write(root/'REQUEST.json',request)
    from fresh_adapter import scope
    from capture_stage import _generate
    with scope() as cohort:
        install(cohort,spec)
        return _generate(effective,context,cohort)
