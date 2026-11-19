"""Mode identity, DATA binding invocation, and automatic fixed DEV selection.

No sampling, optimizer, network, or alternative controller lives here.
"""
import copy
import sys
from pathlib import Path
from dependencies import require,load,local_path
from run_package import read,pin,write


def engineering_updates(config):
    """Explicit bounded smoke updates never grant formal population acceptance."""
    value=config.get('engineering_update_sequence')
    if value is None:
        require(config['max_optimizer_updates']==1,'legacy engineering mode permits one update')
        return 1
    require(value==dict(schema='engineering-update-sequence-v1',updates=3,
            fresh_actions_each_step=True,retain_optimizer_state=True,business_gain_claim=False),
            'explicit three-update engineering contract required')
    require(config['execution_mode']=='finite_engineering' and config['formal_rl_enabled'] is False and
            config['max_optimizer_updates']==value['updates'],
            'engineering sequence cannot authorize formal training')
    return value['updates']


def validate_mode(config):
    mode=config.get('execution_mode','formal')
    require(mode in ('formal','finite_engineering'),'explicit finite/formal mode')
    binding=config.get('reward_evaluation')
    selected=config.get('selection',{}).get('reward_evaluation')
    if binding or selected:
        require(binding==selected,'production/selection fixed reward binding differs')
        for ref in read(binding['source_closure']):pin(ref)
        pin(binding['module']);pin(binding['paired_module']);pin(binding['specification'])
    from parity_gate import coverage_scope
    coverage_scope(dict(config,formal_rl_enabled=config.get('formal_rl_enabled',False)))
    if mode=='finite_engineering':
        spec=read(config['finite_capture_binding_ref'])
        require(spec['schema']=='finite-capture-binding-v1' and spec['bounds']['groups']==2 and spec['bounds']['G']==4,'finite DATA contract differs')
        engineering_updates(config)
        require(config['max_zero_signal_batches']==1,'finite zero-signal stop policy')
        require(config.get('resident',{}).get('enabled') is True,'finite mode uses original resident')
        require(config['root_seed']==spec['root_seed'],'finite DATA seed drift')
        require(config['finite_data_binding']['module']==config['bounded_capture_binding']['module'],'finite prepare/capture implementation differs')
        require(config['stage_budgets']['capture']['tokens_upper']>=spec['bounds']['accounted_capture_tokens_upper'],'finite capture budget too small')
    return mode


def data_module(config):
    binding=config['finite_data_binding']
    for ref in read(binding['source_closure']):pin(ref)
    path=local_path(pin(binding['module'])['path'])
    sys.path.insert(0,str(path.parent))
    return load(path,'finite_release_DATA_binding')


def prepare_finite(config,input_ref,output,policy_ref):
    source=read(input_ref)
    require(source.get('sampling',{}).get('system_version')!='synthetic-system-v1' and not source.get('synthetic_fixture'),'synthetic runtime input forbidden')
    require(not source.get('cpu_contract_only') and not source.get('policy',{}).get('fixture'),'CPU input cannot enter real finite iteration')
    return data_module(config).prepare_input(input_ref,output,config_ref=config['finite_capture_binding_ref'],policy_ref=policy_ref)


def prepare_formal_startup(config,input_ref,output,policy_ref):
    """Reuse bounded source custody without replacing the formal node Clock.

    This input can produce diagnostic evidence only. Full TRAIN screening must
    replace the bounded sampler before the original formal trainer can use it.
    """
    require(config.get('execution_mode') == 'formal', 'formal startup mode required')
    source = read(input_ref)
    require(not source.get('cpu_contract_only') and not source.get('policy', {}).get('fixture')
            and not source.get('synthetic_fixture'), 'real formal startup required')
    learning = read(source['learning_config_ref']); state = read(source['clock_state_ref'])
    phase = [p for p in learning['phases'] if p['start_step'] <= state['successful_optimizer_steps']][-1]
    task = config.get('runtime_construction', {}).get('node_training_task')
    expected = task['node'] if task else 'P'
    require(phase['target_nodes'] == [expected], 'formal startup target differs from configured task')
    return data_module(config).prepare_input(input_ref,output,
        config_ref=config['finite_capture_binding_ref'],policy_ref=policy_ref,startup_only=True)


def parity_accept(state,mode):
    """Only called after original verify; binds exact raw/runtime/policy/witness."""
    raw=read(state['raw_ref'])
    require(raw['endpoint_witness_ref']==state['witness_ref'],'parity endpoint witness differs from active window')
    require(raw['policy_version']==read(state['policy_ref'])['version'],'parity active policy drift')
    require(raw.get('actual_npu') is True,'CPU fixture parity cannot authorize update')
    return dict(execution_mode=mode,raw_ref=pin(state['raw_ref']),runtime_ref=pin(state['runtime_ref']),
                policy_version=raw['policy_version'],endpoint_witness_ref=pin(state['witness_ref']),diagnostic_scope=raw.get('diagnostic_scope','strict_full'),coverage_acceptance=raw.get('coverage_acceptance'))


def freeze_reward(config,state,root,config_path):
    """Freeze from authenticated full DEV reports, never from TEST or caller text."""
    value=dict(best=state['best'],selection_split='DEV',test_feedback=False,config_ref=pin(config_path))
    binding=config['selection'].get('reward_evaluation')
    if binding:
        baseline_ref=pin(state['baseline_dev_ref']);candidate_ref=pin(state['best']['dev_score_ref'])
        baseline,candidate=read(baseline_ref),read(candidate_ref)
        for report in (baseline,candidate):
            require(report['split']=='DEV' and report['fixture'] is False and report['evaluation_complete'] is True and
                    report['denominator']==8690,'freeze requires real complete DEV')
            require(report.get('policy_treatment') is not None,'freeze DEV policy missing')
            exported=read(report['reward_contributions_ref'])
            require(exported['specification']==binding['specification'] and exported['evaluation_complete'] is True,'freeze fixed reward identity/incomplete')
        quality=read(state['best']['quality_ref'])
        require(quality['baseline_report']==baseline_ref and quality['candidate_report']==candidate_ref,'freeze paired DEV inputs drift')
        require(quality['split']=='DEV' and quality['fixture'] is False and quality['business_acceptance']['dev_selection_eligible'] and
                quality['reward_evaluation']['clear_dev_reward_gain'],'freeze requires original business and fixed reward gates')
        require(baseline['dataset']==candidate['dataset'] and baseline['evaluation_protocol_sha256']==candidate['evaluation_protocol_sha256'],'freeze DEV dataset/protocol drift')
        value.update(frozen=True,reward_specification=binding['specification'],
            baseline_policy=baseline['policy_treatment'],candidate_policy=candidate['policy_treatment'],
            baseline_dev_report_ref=baseline_ref,candidate_dev_report_ref=candidate_ref,paired_dev_ref=state['best']['quality_ref'])
    return write(Path(root)/'FINAL-SELECTION.json',value)


def prepare_diagnostic(config,input_ref,output,policy_ref,witness_ref,purpose):
    require(purpose in ('base_parity','updated_parity'),'finite diagnostic purpose')
    binding=config['finite_diagnostic_binding']
    for ref in read(binding['source_closure']):pin(ref)
    path=local_path(pin(binding['module'])['path']);sys.path.insert(0,str(path.parent))
    module=load(path,'finite_release_DATA_diagnostic')
    result=module.prepare_input(input_ref,output,config_ref=config['finite_diagnostic_config_ref'],
        policy_ref=policy_ref,witness_ref=witness_ref,purpose=purpose)
    value=read(result['input_ref'])
    require(value['diagnostic_evidence_only']['trainable'] is False,'diagnostic may not train')
    require(value['policy']['current_ref']==policy_ref,'diagnostic policy drift')
    return result
