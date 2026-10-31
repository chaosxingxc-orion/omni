"""Actual runner request custody for seed-only candidates consuming a bank."""
import copy,json,sys
from dataclasses import replace
from pathlib import Path
import pytest
pytest.importorskip('openjiuwen')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/analysis'))
from test_llama_server_transport import _spawn_live_matching_process
from test_tfrl_seed_execution import group
from test_decoding_variants_live import _decoding_harness,_requests,_run
from test_tfrl_v2_runner_render import _bank,_block_json,PERCEPTION,BLOCK_TEXT,RULE_TEXT


def actual(tmp_path):
    _,_,g=group();h=_decoding_harness(tmp_path,g,rows=1);bank=_bank()
    h.config=replace(h.config,prompt_contact_limits='legacy-c17-lock-v1',consumption_module_json=_block_json(bank,bound=55))
    h.tfrl_v2_bank=bank;h._output=lambda key:PERCEPTION if key[1]=='perception' else '{"scenario":"alarm","action":"set","entities":[]}'
    summary,close=_run(h)
    return summary.results[0],_requests(h),bank


def test_actual_bank_candidate_custody_shape(tmp_path):
    result,requests,bank=actual(tmp_path)
    assert len(requests)==7
    candidate,lock=requests[1],requests[-1]
    assert RULE_TEXT in candidate['task_instruction'] and RULE_TEXT not in lock['task_instruction']
    assert any(e['content']==BLOCK_TEXT for e in candidate['supplied_evidence'])
    assert not any(e['content']==BLOCK_TEXT for e in lock['supplied_evidence'])
    from tfrl_current_bank_audit import verify_group
    checked=verify_group(result.sample_id,result.supply_variants['arms'],requests,bank=bank)
    assert checked['solver_requests_verified']==6 and checked['bank_candidate_requests']==4


@pytest.mark.parametrize('fault',['rule','exemplar','source','provenance','control','seed','other','scenario'])
def test_actual_bank_audit_refuses_undeclared_context_and_sampling_changes(tmp_path,fault):
    from tfrl_current_bank_audit import verify_group
    result,requests,bank=actual(tmp_path);requests=copy.deepcopy(requests);candidate=requests[1]
    if fault=='rule':candidate['task_instruction']+='changed'
    elif fault=='exemplar':candidate['supplied_evidence'][-1]['content']='changed'
    elif fault=='source':candidate['supplied_evidence'][-1]['source_uri']='wrong-bank'
    elif fault=='provenance':candidate['evidence_provenance'][-1]['sha256']='a'*64
    elif fault=='control':requests[-2]['supplied_evidence']=copy.deepcopy(candidate['supplied_evidence'])
    elif fault=='seed':candidate['decoding_params']['seed']+=1
    elif fault=='other':candidate['audio_seconds']+=1
    elif fault=='scenario':candidate['history'][0]['content']='{"candidates":[{"intent_id":"weather_query"}]}'
    with pytest.raises(ValueError):verify_group(result.sample_id,result.supply_variants['arms'],requests,bank=bank)


def test_reader_requires_declaration_for_later_iteration_before_consumption():
    from tfrl_current_bank_audit import verify_for_values
    with pytest.raises(ValueError,match='lacks declared bank'):
        verify_for_values({'iteration':2,'banks':{'bank_receipt_sha256':'a'*64}},[],{},values={},data=Path('/unused'))


def test_plan_bank_and_payload_hash_are_bound_before_credit(tmp_path):
    import tfrl_v2_sampler as sampler
    from tfrl_current_bank_audit import verify
    from test_tfrl_v2_consumption import make_bank,SCOPE,H1,H2
    from speech_aware_evidence_acquisition.rails import tfrl_v2_consumption as rail
    # Use the production bank receipt/plan constructors; this test performs no model call.
    _,receipt=make_bank(tmp_path,kind='half',sources=(H1,))
    bank=rail.LoadedBank(receipt=receipt,receipt_sha256=rail.receipt_sha256(receipt),exemplar_bank={},rule_bank={})
    roster={'roster_id':sampler.cm.HALF_ROSTER_ID[H2],'split_identity_hash':SCOPE[H2],
            'items':[{'sample_id':'new-row','audio_seconds':2.}]}
    plan=sampler.build_plan(roster,H2,2,bank_receipt=receipt)
    output={'sample_id':'new-row'}
    bad=copy.deepcopy(plan);bad['banks']['bank_receipt_sha256']='a'*64
    with pytest.raises(ValueError,match='bank differs'):verify(bad,[output],{},bank=bank)
    with pytest.raises(ValueError,match='payload hash'):
        verify(plan,[output],{'req/request':{'payload':{},'payload_sha256':'wrong'}},bank=bank)
