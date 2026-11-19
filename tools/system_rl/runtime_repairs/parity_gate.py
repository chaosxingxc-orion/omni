"""Use the existing numerical R/C/P comparator, never trust passed booleans."""
import json
from pathlib import Path
from dependencies import ROOT, require, load, sha
PARENT = ROOT/'native-recompute-integration/state-repair/acl-runtime-successor'
# Explicit source loading retains upstream thresholds and actual compare code.
import sys
_saved = {n:sys.modules.get(n) for n in ('sync_runtime','artifact_loader')}
try:
    sys.modules['sync_runtime'] = load(PARENT/'sync_runtime.py','dp8_sync_runtime')
    sys.modules['artifact_loader'] = load(PARENT/'artifact_loader.py','dp8_artifact_loader')
    comparator = load(PARENT/'parity.py','dp8_authoritative_rcp')
finally:
    for name,value in _saved.items():
        if value is None: sys.modules.pop(name,None)
        else: sys.modules[name]=value


def coverage_scope(config):
    scope=config.get('diagnostic_scope','strict_full')
    require(scope in ('strict_full','dataset_finite_probe','dataset_profile'),'unknown diagnostic scope')
    if scope=='dataset_profile':
        from domain_admission import config_scope
        config_scope(config)
    if scope=='dataset_finite_probe':
        require(config.get('execution_mode')=='finite_engineering' and config.get('formal_rl_enabled') is False,
                'finite evidence cannot authorize formal training')
        from finite_runtime import engineering_updates
        engineering_updates(config)
    return scope


def coverage_acceptance(records,config):
    scope=coverage_scope(config)
    if scope=='dataset_profile':
        from domain_admission import acceptance
        return acceptance(records,config)
    from diagnostic_protocol import enabled, check_coverage, target_nodes
    split=enabled(config)
    if split:split_coverage=check_coverage(records,config)
    else:require({r['node'] for r in records}=={'P','S1','S2','D'},'complete business node parity coverage required')
    null=any((r['coverage']['observed_null_values'] or 0)>0 for r in records)
    long=any(r['coverage']['long_audio'] for r in records)
    if not split or any(n in target_nodes(config) for n in ('S2','D')):
        require(null,'retained offline null evidence missing')
    if scope=='strict_full':require(long,'retained long audio parity missing')
    return dict(scope=scope,source_population=config.get('diagnostic_source_population'),
        covered=dict({n:any(r['node']==n for r in records) for n in ('P','S1','S2','D')},legal_null=null,long_audio=long),
        **({'diagnostic_roles':split_coverage} if split else {}),
        full_capability_acceptance=scope=='strict_full',formal_training_allowed=scope=='strict_full',
        max_optimizer_updates=config.get('max_optimizer_updates'))


def evaluate(raw, tolerances):
    require(tolerances == comparator.THRESHOLDS, 'frozen R/C/P thresholds changed')
    require(raw['kind'] == 'OMNI_DP8_DIAGNOSTIC_RAW' and raw['actual_npu'] is True,
            'CPU fixture cannot authorize RL')
    require(raw['training_world_size'] == 8 and raw['thinking'] is False, 'RCP topology/model mode differs')
    require(raw['records'], 'nonempty real RCP rows required')
    results=[]
    for row in raw['records']:
        require(row['node'] in ('P','S1','S2','D'), 'RCP unknown business node')
        require(row['same_input_tokens'] and row['same_audio_boundary'] and row['same_positions'],
                'RCP must bind identical real model inputs')
        from diagnostic_protocol import NAME,numerical
        result = (numerical(row,comparator) if raw.get('diagnostic_protocol')==NAME else
                  comparator.diagnostics(row['a1'],row['b'],row['a2'],row['reference'],row['prefix_count']))
        results.append(dict(node=row['node'], R=result['R'], C=result['C'], P=result['P']))
    from diagnostic_protocol import NAME
    passed=all(row['R']['passed'] and row['P']['passed'] and
        (row['C']['passed'] is True or (raw.get('diagnostic_protocol')==NAME and row['C'].get('applicable') is False)) for row in results)
    if raw.get('diagnostic_protocol')==NAME:passed=passed and any(row['C']['passed'] is True for row in results)
    return dict(passed=passed,
                results=results, comparator_sha256=sha(PARENT/'parity.py'))


def verify(raw_ref, *, config, policy_version, source_pins_sha256, require_updated=False, identity=None):
    from entry import pinned
    from policy_identity import behavior_binding, artifact
    raw = json.loads(pinned(raw_ref).read_text())
    from diagnostic_protocol import enabled, roles, tokens_for
    require(raw.get('diagnostic_protocol','legacy_aba')==config.get('diagnostic_protocol','legacy_aba'),
            'diagnostic protocol identity differs')
    if enabled(config):
        require(raw.get('diagnostic_target_nodes')==config.get('diagnostic_target_nodes',['P','S1','S2','D']),
                'diagnostic target identity differs')
    require(raw['model'] == config['model']['name'] and raw['policy_version'] == policy_version, 'RCP policy differs')
    require(raw['image_id'] == config['training']['image_id'] and raw['source_pins_sha256'] == source_pins_sha256, 'RCP source/image differs')
    scope=coverage_scope(config)
    require(raw.get('diagnostic_scope','strict_full')==scope,'raw diagnostic scope differs from requesting runtime')
    if scope=='dataset_profile':
        from domain_admission import verify_execution
        verify_execution(raw,config)
    observed = raw['resident_identity']
    require(identity is not None and observed == identity, 'RCP does not describe this live resident adapter')
    binding = behavior_binding(identity, raw)
    require(binding == raw['behavior_binding'], 'behavior identity replay differs')
    for row in raw['records']:
        behavior=json.loads(pinned(row['behavior_ref']).read_text())
        a=behavior['records'][row['a_index']]
        b=behavior['records'][row['b_index']] if 'causal_control' in roles(row) else a
        tokens_for(row,behavior['records'])
        require(a['prompt_token_ids']==b['prompt_token_ids'] and a['behavior_logprobs']==row['reference'], 'RCP reference custody differs')
        prefix=row['prefix_count']
        if 'causal_control' in roles(row):
            require(a['token_ids'][:prefix]==b['token_ids'][:prefix] and len(a['token_ids'])==len(b['token_ids']) and a['token_ids'][prefix:]!=b['token_ids'][prefix:], 'RCP retained control differs')
        prepared_path=pinned(row['prepared_ref'])
        import torch
        from diagnostic import coverage_facts
        prepared=torch.load(prepared_path,map_location='cpu',weights_only=False)
        if a.get('input_identity') or b.get('input_identity'):
            require(a.get('input_identity')==b.get('input_identity')==prepared.get('input_identity'),
                    'RCP actual multimodal input identity differs')
        require(coverage_facts(prepared,row['coverage'],config)==row['coverage'],'actual coverage replay differs')
        retained_policy=behavior.get('policy_version',a.get('behavior_policy_version'))
        retained_witness=behavior.get('endpoint_witness_ref',a.get('endpoint_witness_ref'))
        require(retained_policy==binding['policy_version'] and retained_witness==raw['endpoint_witness_ref'],'retained behavior belongs to another adapter/witness')
        if not observed['publication']['nonzero_effective_delta']:
            require(row['base_reference'] is not None and comparator.compare(row['a1'],row['base_reference'])['passed'],'base zero adapter versus disabled adapter differs under frozen thresholds')
    scope=coverage_scope(config)
    require(raw.get('diagnostic_scope','strict_full')==scope,'raw diagnostic scope differs from requesting runtime')
    coverage=coverage_acceptance(raw['records'],config)
    if raw.get('coverage_acceptance') is not None:require(raw['coverage_acceptance']==coverage,'raw scope coverage drift')
    result = evaluate(raw, config['parity'])
    require(result['passed'], 'real DP8 R/C/P failed; original thresholds retained')
    from diagnostic_joint_gate import evaluate as joint_evaluate
    joint=joint_evaluate(raw,config)
    require(joint['passed'],'complete actual full-vocabulary output gate failed or missing')
    if require_updated:
        require(observed['publication']['nonzero_effective_delta'], 'updated parity requires effective nonzero BA')
        checkpoint = json.loads(pinned(raw['checkpoint_ref']).read_text())
        require(checkpoint['resident_identity'] == observed, 'updated checkpoint identity differs')
        from dependencies import local_path
        published = artifact(local_path(Path(raw['checkpoint_ref']['path']).parent/'lora_adapter/adapter_model.safetensors'))
        require(published == observed['publication'], 'checkpoint adapter bytes differ')
    verified=dict(raw_ref=raw_ref, binding=binding, recomputed=result,coverage_acceptance=coverage,joint_output=joint)
    if scope=='dataset_profile':
        from domain_admission import certificate
        verified['profile_admission']=certificate(raw,config,verified)
    return verified
