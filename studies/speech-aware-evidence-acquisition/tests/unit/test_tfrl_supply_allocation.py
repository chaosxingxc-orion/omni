"""Real shared runner with fake transport: bank space, custody and controls."""
import copy,json
from dataclasses import replace
from functools import partial
import pytest
from speech_aware_evidence_acquisition.rails import tfrl_supply_allocation as m
from speech_aware_evidence_acquisition.rails.native_episode_receipt import validate_supply_variant_receipt
from speech_aware_evidence_acquisition.rails.tfrl_bank_request_audit import verify_group
from test_llama_server_transport import _spawn_live_matching_process
from test_tfrl_seed_execution import group
from test_decoding_variants_live import _decoding_harness,_requests
from test_tfrl_v2_runner_render import _bank,_block_json,_snapshot,PERCEPTION,ANSWER


def execute(tmp_path,monkeypatch,*,allocated=True,n=55,empty_exemplar=False):
    import test_native_episode_runner as hmodule
    from speech_aware_evidence_acquisition.rails.runner import run_speech_native_agent_loop
    ids=tuple(f'value:v{i+1}' for i in range(n))
    monkeypatch.setattr(hmodule,'run_speech_native_agent_loop',partial(run_speech_native_agent_loop,bootstrap_value_ids=ids))
    _,_,g=group();h=_decoding_harness(tmp_path,g,rows=1);bank=_bank()
    if empty_exemplar:bank=replace(bank,exemplar_bank={'by_scenario':{}})
    block=json.loads(_block_json(bank,bound=55));block['decoding_frame']=block.pop('sampling_frame')
    if allocated:block[m.KEY]=m.declaration(55)
    h.config=replace(h.config,prompt_contact_limits='legacy-c17-lock-v1',consumption_module_json=json.dumps(block))
    h.tfrl_v2_bank=bank;h._output=lambda key:PERCEPTION if key[1]=='perception' else ANSWER
    try:summary=h.run(snapshot=_snapshot(n))
    finally:h.close()
    return summary.results[0],_requests(h),bank,{rid:p for _,rid,p in h.trace.records}


def test_full55_becomes47_only_for_candidates_with_actual_custody(tmp_path,monkeypatch):
    result,requests,bank,trace=execute(tmp_path/'allocated',monkeypatch)
    plain,original,_,_=execute(tmp_path/'plain',monkeypatch,allocated=False)
    assert requests[0]==original[0] and requests[-2:]==original[-2:]
    assert result.supply_variants['schema']==m.RECEIPT_SCHEMA
    assert [a['supplied_item_count'] for a in result.supply_variants['arms']]==[47,47,47,47,55,55]
    validate_supply_variant_receipt(result.supply_variants,arm_ids=m.ARMS,delivered_arm_id='lock',phase_id='solver')
    checked=verify_group(result.sample_id,result.supply_variants['arms'],requests,bank=bank,allocation=m.declaration(55))
    assert checked['bank_candidate_requests']==4
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_supply_variant_trace_binding
    episode=result.native_episode_receipt
    contact=[v for v in episode['role_visits'] if v['stage']=='solver'][-1]['execution_receipt']['final_contact']
    row={'sample_id':result.sample_id,'native_episode_receipt':episode,'supply_variants':result.supply_variants,
        'request_id':contact['request_id'],'request_sha256':contact['request_sha256'],'response_sha256':contact['response_sha256']}
    _verify_supply_variant_trace_binding(row,trace,bank_context={'bank':bank,'allocation':m.declaration(55)})
    import stage3_rollout_score as scorer
    _,_,g=group()
    rendered=scorer.render_supply_source_rows([row],trace,g,{result.sample_id:''},tmp_path/'columns',
        source={'kind':'test-actual-runner'},bank_context={'bank':bank,'allocation':m.declaration(55)})
    assert rendered['rows']==1 and rendered['arm_order']==list(m.ARMS)
    with pytest.raises(scorer.Stage3ScoreError,match='verified bank configuration'):
        scorer.render_supply_source_rows([row],trace,g,{result.sample_id:''},tmp_path/'refuse',source={})
    with pytest.raises(ValueError):verify_group(result.sample_id,result.supply_variants['arms'],requests,bank=bank)
    for fault in ('keep_extra','drop_wrong','provenance','receipt','allocation'):
        rs=copy.deepcopy(requests);arms=copy.deepcopy(result.supply_variants['arms']);allocation=m.declaration(55)
        if fault=='keep_extra':rs[1]['supplied_evidence'].insert(-1,original[1]['supplied_evidence'][-2])
        elif fault=='drop_wrong':rs[1]['supplied_evidence'][0]=copy.deepcopy(original[1]['supplied_evidence'][-2])
        elif fault=='provenance':rs[1]['evidence_provenance'][0]['sha256']='a'*64
        elif fault=='receipt':arms[0]['supplied_value_ids'].reverse()
        else:allocation=m.declaration(54)
        with pytest.raises(ValueError):verify_group(result.sample_id,arms,rs,bank=bank,allocation=allocation)


def test_allocation_preserves_nonvalues_order_and_small_supply():
    parts=[{'knowledge_role':'key','content':'schema'}]+[{'knowledge_role':'value','content':str(i)} for i in range(55)]+[{'content':'extra'}]
    keep=m.keep_indices(parts,m.declaration(55))
    assert keep==list(range(48))+[56]
    assert m.keep_indices(parts[:3],m.declaration(55))==[0,1,2]
    with pytest.raises(ValueError):m.keep_indices(parts+[{'knowledge_role':'value'}],m.declaration(55))


def test_allocation_must_be_in_compiled_module_and_match_control_frame():
    from test_decoding_variants import _decoding_block
    from speech_aware_evidence_acquisition.rails import consumption_module as cm
    from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    import tfrl_v2_sampler as sampler
    plan,_,_=group();block=_decoding_block();block['columns']=sampler.execution_columns(plan)
    block.update(json.loads(_block_json(_bank(),bound=55)))
    block.pop('sampling_frame');block[m.KEY]=m.declaration(55)
    block['prompt_contact_limits']='legacy-c17-lock-v1'
    compiled=cm.compile_module(block,carrier_module_hooks());values=compiled.values()
    values['carrier_lock_key']='slurp';values['native_episode']['consumption_module']=block
    config=load_native_episode_config(values,default_phase_program(values['policy']))
    assert json.loads(config.consumption_module_json)[m.KEY]==m.declaration(55)
    bad=copy.deepcopy(block);bad[m.KEY]['candidate_max_items']=48
    with pytest.raises(cm.ConsumptionModuleConfigError):cm.compile_module(bad,carrier_module_hooks())


@pytest.mark.parametrize('n,empty',[(2,False),(55,True)])
def test_small_supply_and_empty_exemplar_follow_declared_reservation(tmp_path,monkeypatch,n,empty):
    result,requests,bank,_=execute(tmp_path,monkeypatch,n=n,empty_exemplar=empty)
    expected=min(n,47)
    assert [a['supplied_item_count'] for a in result.supply_variants['arms']]==[expected]*4+[n]*2
    verify_group(result.sample_id,result.supply_variants['arms'],requests,bank=bank,allocation=m.declaration(55))


def test_allocated_snapshot_is_bound_to_config(tmp_path):
    from test_tfrl_bank_bundle import setup
    from speech_aware_evidence_acquisition.core.bundle import tfrl_bank
    from speech_aware_evidence_acquisition.core.config import _canonical_hash
    from types import SimpleNamespace
    data,path,manifest,directory,split=setup(tmp_path)
    original=json.loads((directory/'tfrl-composed-config.json').read_text())['values']
    original['native_episode']['consumption_module'].update({m.KEY:m.declaration(55),'decoding_frame':{'max_supplied_items':55}})
    target=directory/'next';target.mkdir()
    cfg=SimpleNamespace(values=original,config_hash=_canonical_hash(original))
    manifest['config']['config_hash']=cfg.config_hash
    manifest['tfrl_bank_config']=tfrl_bank.seal(cfg,data,split,target)
    context=tfrl_bank.load(manifest,target/'manifest.json',split.ids,split.identity_hash)
    assert context['allocation']==m.declaration(55)
