import json
import pytest
pytest.importorskip('openjiuwen')
from test_llama_server_transport import _spawn_live_matching_process
from test_tfrl_seed_execution import group
from test_decoding_variants_live import _decoding_harness,_requests,_run,seam
from speech_aware_evidence_acquisition.rails.trajectory_seeds import resolve, verify_recorded_group
from speech_aware_evidence_acquisition.rails.native_episode_receipt import validate_supply_variant_receipt


def test_real_shared_runner_delivers_seed_only_candidates_and_valid_receipt(tmp_path):
    from dataclasses import replace
    p,b,g=group();h=_decoding_harness(tmp_path,g,rows=2)
    h.config=replace(h.config,prompt_contact_limits='legacy-c17-lock-v1')
    summary,close=_run(h)
    assert close['request_count']==14
    requests=_requests(h)
    for result in summary.results:
        row=[r for r in requests if r['sample_id']==result.sample_id]
        assert len(row)==7
        candidates,twin,lock=row[1:5],row[5],row[6]
        for k,request in enumerate(candidates):
            override=resolve(g.arm(f's{k+1}').decoding,result.sample_id)
            assert request['decoding_params']=={**lock['decoding_params'],**override}
            for key in ('speech_feature_parts','speech_packet_sha256','supplied_evidence','task_instruction'):
                assert request[key]==lock[key]
        normalized=[{k:v for k,v in r['decoding_params'].items() if k!='seed'} for r in candidates]
        assert all(x==normalized[0] for x in normalized)
        assert len({r['decoding_params']['seed'] for r in candidates})==4
        assert twin['decoding_params']==lock['decoding_params']
        assert verify_recorded_group(result.sample_id,result.supply_variants['arms'],requests)==6
        validate_supply_variant_receipt(result.supply_variants,arm_ids=g.arm_ids,delivered_arm_id='lock',
            phase_id='solver',retrieved_value_ids=result.final_query_state.retrieved_value_ids,arm_visits='first')


def test_request_seam_resolves_recipe_and_fallback_keeps_seed(seam):
    make,request=seam;p,b,g=group()
    adapter=make(solver_arm_decoding={a.arm_id:a.decoding for a in g.arms if a.decoding},solver_arm_decoding_role_id='solver')
    req=request('solver','s1','seed-c1')
    body=adapter.prepare(req).rendered_body()['decoding_params']
    assert body['seed']==resolve(g.arm('s1').decoding,'s1')['seed']
    assert 'trajectory_index' not in body and 'trajectory_seed_base' not in body


@pytest.mark.parametrize('fault',['seed','evidence','instruction','missing_arm','recipe_on_wire','twin','extra_decoding'])
def test_seed_request_audit_refuses_runtime_mismatch(tmp_path,fault):
    import copy
    from dataclasses import replace
    p,b,g=group();h=_decoding_harness(tmp_path,g,rows=1)
    h.config=replace(h.config,prompt_contact_limits='legacy-c17-lock-v1')
    summary,_=_run(h);result=summary.results[0];requests=copy.deepcopy(_requests(h))
    target=requests[1]
    if fault=='seed':target['decoding_params']['seed']+=1
    elif fault=='evidence':target['supplied_evidence']=[{'changed':True}]
    elif fault=='instruction':target['task_instruction']='changed'
    elif fault=='missing_arm':requests.pop(1)
    elif fault=='recipe_on_wire':target['decoding_params']['trajectory_index']=0
    elif fault=='twin':requests[-2]['decoding_params']['seed']=999
    elif fault=='extra_decoding':target['decoding_params']['max_tokens']+=1
    with pytest.raises(ValueError):verify_recorded_group(result.sample_id,result.supply_variants['arms'],requests)


@pytest.mark.parametrize('fault',[None,'plan_seed','plan_row','receipt_seed'])
def test_real_runner_receipts_bind_back_to_plan(tmp_path,fault):
    import copy
    from dataclasses import replace
    import tfrl_v2_sampler as sampler
    from tfrl_seed_request_audit import verify
    _,_,g=group();h=_decoding_harness(tmp_path,g,rows=1)
    h.config=replace(h.config,prompt_contact_limits='legacy-c17-lock-v1')
    summary,_=_run(h);result=summary.results[0]
    plan=sampler.build_plan({'roster_id':sampler.cm.HALF_ROSTER_ID['R01C-H1'],
        'split_identity_hash':'a'*64,'items':[{'sample_id':result.sample_id,'audio_seconds':2.}]},'R01C-H1',1)
    row={'sample_id':result.sample_id,'supply_variants':copy.deepcopy(result.supply_variants)}
    trace={rid:p for channel,rid,p in h.trace.records}
    if fault=='plan_seed':plan['row_plan'][0]['contacts'][1]['seed']+=1
    elif fault=='plan_row':row['sample_id']='different'
    elif fault=='receipt_seed':row['supply_variants']['arms'][0]['decoding_override']['trajectory_seed_base']+=1
    if fault:
        with pytest.raises(ValueError):verify(plan,[row],trace)
    else:
        audit=verify(plan,[row],trace)
        assert audit['rows']==1 and audit['solver_requests_verified']==6


@pytest.mark.parametrize("measured", [10, 101, None])
def test_registered_fit_count_controls_seed_group_without_byte_heuristic(tmp_path, measured):
    from dataclasses import replace
    from test_e20_context_fit_policy import _budget
    from speech_aware_evidence_acquisition.rails.runner import RailsRunnerError
    _, _, g = group()
    h = _decoding_harness(tmp_path, g, rows=1)
    h.config = replace(h.config, prompt_contact_limits="legacy-c17-lock-v1",
                       context_fit_policy="compact-or-abstain")
    def forbidden_counter(_body):
        raise AssertionError("audio bytes must never enter a heuristic counter")
    h.context_fit_budget = replace(_budget(forbidden_counter, prompt_limit_tokens=100),
                                   measured_by_request_gate=True)
    original = h.request_gate
    gates = []
    def measured_gate(request, *, prompt_limit_tokens):
        assert prompt_limit_tokens == 100
        original(request)
        gates.append(request)
        return measured
    h.request_gate = measured_gate
    if measured is None:
        with pytest.raises(Exception, match="exact prompt count"):
            _run(h)
        assert len(gates) == 1
        return
    summary, close = _run(h)
    row = summary.results[0]
    if measured == 10:
        assert close["request_count"] == 7
        assert len(gates) == 7
        assert len(row.supply_variants["arms"]) == 6
    else:
        assert close["request_count"] == 0
        assert len(gates) == 1
        assert row.final_response_text == ""
        assert row.supply_variants is None
        disposition = row.native_episode_receipt["terminal_disposition"]
        assert disposition["reason"] == "context_compaction"
        assert disposition["tokens_before"] == measured
        assert disposition["limit_tokens"] == 100
        import tfrl_v2_sampler as sampler
        from tfrl_seed_request_audit import verify
        plan = sampler.build_plan({"roster_id": sampler.cm.HALF_ROSTER_ID["R01C-H1"],
            "split_identity_hash": "a" * 64,
            "items": [{"sample_id": row.sample_id, "audio_seconds": 2.}]}, "R01C-H1", 1)
        output = {"sample_id": row.sample_id, "supply_variants": None,
                  "native_episode_receipt": row.native_episode_receipt}
        trace = {rid: p for _, rid, p in h.trace.records}
        checked = verify(plan, [output], trace)
        assert checked["no_solver_abstention_rows"] == 1
        assert checked["solver_requests_verified"] == 0
        from types import SimpleNamespace
        from tfrl_current_distill import sealed_answers
        from tfrl_v2_text import ItemTextResolver
        resolver = ItemTextResolver({"unused": "verified text"}, source={"kind": "test"})
        answers, _, provenance = sealed_answers(SimpleNamespace(outputs_rows=[output],
            trace_records=trace), plan, resolver=resolver)
        assert answers == {}
        assert provenance[0]["availability"] == "no_solver_visit"
        assert provenance[0]["candidate_answers"] == 0
        output["native_episode_receipt"] = {"role_visits": []}
        with pytest.raises(ValueError):
            verify(plan, [output], trace)


@pytest.mark.parametrize('fault',[None,'owner','hash'])
def test_seed_context_accepts_only_owned_trace_provenance_routes(tmp_path,fault):
    import copy
    from dataclasses import replace
    _,_,g=group();h=_decoding_harness(tmp_path,g,rows=1)
    h.config=replace(h.config,prompt_contact_limits='legacy-c17-lock-v1')
    summary,_=_run(h);row=summary.results[0];requests=copy.deepcopy(_requests(h))
    for req in requests:
        req['evidence_provenance']=[{'lock_key':'rails-supply',
            'relpath':f"rails-message/{req['request_id']}/0",'sha256':'a'*64}]
    if fault=='owner':requests[1]['evidence_provenance'][0]['relpath']='rails-message/not-owned/0'
    elif fault=='hash':requests[1]['evidence_provenance'][0]['sha256']='b'*64
    if fault:
        with pytest.raises(ValueError):verify_recorded_group(row.sample_id,row.supply_variants['arms'],requests)
    else:assert verify_recorded_group(row.sample_id,row.supply_variants['arms'],requests)==6
