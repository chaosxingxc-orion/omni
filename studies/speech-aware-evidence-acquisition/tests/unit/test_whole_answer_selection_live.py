import copy
import json
from dataclasses import replace
import pytest
from speech_aware_evidence_acquisition.rails import whole_answer_selection as m
from test_llama_server_transport import _spawn_live_matching_process
from test_native_episode_runner import _ActualEpisodeHarness
from test_stage3_supply_variant_interleaving import _arm_resolver, _live_config, ARM_VALUE
from test_whole_answer_selection import text


def harness_for(tmp_path, *, equal=False, judge='{"choice":"B"}', raw_audio=False):
    from test_tfrl_odd40_candidates import m as candidates, documents, ROOT
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    base,_=candidates.derive(ROOT,documents(tmp_path/'source')[0])
    block=copy.deepcopy(base['native_episode']['consumption_module']);block['answer_selection']=m.POLICY_ID
    compiled=candidates.cm.compile_module(block,candidates.carrier_module_hooks(repo_root=ROOT),repo_root=ROOT)
    values=compiled.apply_to(base);values['native_episode']['consumption_module']=block
    parsed=load_native_episode_config(values,default_phase_program(values['policy']))
    group=parsed.supply_variant_group
    h=_ActualEpisodeHarness(tmp_path/'harness',sample_count=1,max_revisions=0,max_contacts=5,planner_actions=(),
        query_source='speech_audio' if raw_audio else 'speech_features',
        arm='raw-speech-agent-native-tools' if raw_audio else 'speech-feature-query-native-tools')
    h.config=replace(_live_config(group=group,max_contacts=5),prompt_contact_limits='legacy-c17-lock-v1',answer_selection=m.POLICY_ID)
    h.supply_variant_for_phase=_arm_resolver({'b':(ARM_VALUE,), 'aa':(), 'lock':()})
    original=h.request_gate
    def gate(request):
        payload=request.as_payload()
        name=payload.get('decoding_params',{}).get('response_format',{}).get('json_schema',{}).get('name')
        if name!='saea_whole_answer_choice_v1':return original(request)
        rendered=h.transport.render_request(request.payload_bytes)
        key=(h.samples[0].sample_id,'selection',1,1)
        assert key not in h.prepared
        h.prepared[key]={'rendered':rendered,'slot_id':payload['decoding_params']['id_slot']}
        h.events.append(('fit',key))
    h.request_gate=gate
    def output(key):
        if key[1]=='selection':return judge
        if key[1]=='solver':return text() if equal or key[3]!=1 else text('two')
        return 'shared perception note'
    h._output=output
    return h,values


@pytest.mark.parametrize('equal,judge,expected,calls',[(False,'{"choice":"B"}','two',5),(False,'{"choice":"C"}','one',5),(True,None,'one',4)])
def test_actual_graph_selects_source_with_metered_conditional_contact(tmp_path,equal,judge,expected,calls):
    h,values=harness_for(tmp_path,equal=equal,judge=judge)
    try:summary=h.run()
    finally:h.close()
    result=summary.results[0]
    assert result.final_response_text==text(expected)
    receipt=result.native_episode_receipt['answer_selection']
    requests=[p['payload'] for channel,rid,p in h.trace.records if channel=='MODEL_REQUEST']
    assert len(requests)==calls
    records={rid:p for channel,rid,p in h.trace.records}
    assert m.verify_trace(receipt,result.native_episode_receipt,result.supply_variants,records,result.sample_id)==text(expected)
    assert sum(1 for event,key in h.events if event=='fit')==calls
    assert receipt['additional_contacts']==calls-4
    if not equal:
        assert requests[-1]['supplied_evidence']
        assert any(item.get('state_kind')=='phase:perception' for item in requests[-1]['history'])


@pytest.mark.parametrize('equal,judge',[(False,'{"choice":"B"}'),(False,'{"choice":"C"}'),(True,None)])
def test_runtime_output_and_bundle_bind_selected_candidate(tmp_path,equal,judge):
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    from speech_aware_evidence_acquisition.core.bundle import BundleError
    h,values=harness_for(tmp_path,equal=equal,judge=judge)
    try:
        summary=h.run()
        raw=_raw_native(h,solver_role=values['native_episode']['roles']['solver'])
        raw['answer_selection']=m.POLICY_ID
        runtime=_runtime(h,summary,raw)
        outputs=runtime._canonical_output_rows(h.client)
    finally:h.close()
    receipt=summary.results[0].native_episode_receipt['answer_selection']
    assert outputs[0]['request_id']==receipt['selected_contact']['request_id']
    records={rid:p for channel,rid,p in h.trace.records}
    _verify_outputs_trace_binding(tuple(outputs),records)
    if not equal:
        altered=copy.deepcopy(outputs)
        altered[0]['native_episode_receipt']['answer_selection']['selected']='A' if receipt['selected']=='B' else 'B'
        with pytest.raises(BundleError):_verify_outputs_trace_binding(tuple(altered),records)
        missing=copy.deepcopy(records)
        del missing[receipt['adjudicator']['request_id']+'/response']
        with pytest.raises(BundleError):_verify_outputs_trace_binding(tuple(outputs),missing)


def test_real_audio_transport_shape_keeps_packet_and_comparison_history(tmp_path):
    h,values=harness_for(tmp_path,raw_audio=True)
    try:summary=h.run()
    finally:h.close()
    assert summary.results[0].final_response_text==text('two')
    requests=[p['payload'] for channel,rid,p in h.trace.records if channel=='MODEL_REQUEST']
    judge=requests[-1]
    assert judge['media_relpath']==requests[0]['media_relpath']
    states=[item for item in judge['history'] if item.get('role')=='agent_state']
    comparison=json.loads(next(s['content'] for s in states if s['state_kind']=='phase:answer-selection'))
    assert comparison['candidates'][0]['answer']==json.loads(text())
    assert comparison['candidates'][1]['answer']==json.loads(text('two'))


def test_complete_selection_checkpoint_restores_without_any_request(tmp_path):
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter, read_checkpoint, serialize_trace_records
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    h,values=harness_for(tmp_path/'source')
    try:
        summary=h.run()
        raw=_raw_native(h,solver_role=values['native_episode']['roles']['solver']);raw['answer_selection']=m.POLICY_ID
        outputs=_runtime(h,summary,raw)._canonical_output_rows(h.client)
    finally:h.close()
    writer=RowCheckpointWriter(tmp_path/'attempt','proof','attempt',identity={'protocol_hash':'a'*64})
    writer.append_row(sample_id=outputs[0]['sample_id'],complete=True,output_rows=outputs,
                      trace_lines=serialize_trace_records(h.trace.records))
    writer.close()
    restored=read_checkpoint(writer.path)
    assert restored.complete_rows==1 and restored.rows[0].contacts==5
    row=restored.rows[0]
    assert list(row.output_rows)==outputs
    records={d['record_id']:d['payload'] for d in map(json.loads,row.trace_lines)}
    _verify_outputs_trace_binding(row.output_rows,records)
    assert len(h.dispatched)==5


def test_unknown_judge_outcome_is_not_delivered_as_fallback(tmp_path):
    h,_=harness_for(tmp_path)
    original=h._output
    def fail(key):
        if key[1]=='selection':raise TimeoutError('uncertain transport result')
        return original(key)
    h._output=fail
    try:
        with pytest.raises(Exception):h.run()
    finally:h.close()
    assert sum(1 for event,key in h.events if event=='dispatch' and key[1]=='selection')==1
    assert not any('use_native_episode' in p for channel,rid,p in h.trace.records)


@pytest.mark.parametrize('fault',['evidence','comparison','speech','finish','instruction'])
def test_receipt_rejects_judge_request_drift(tmp_path,fault):
    h,_=harness_for(tmp_path)
    try:summary=h.run()
    finally:h.close()
    result=summary.results[0];receipt=result.native_episode_receipt['answer_selection']
    records={rid:copy.deepcopy(p) for channel,rid,p in h.trace.records}
    rid=receipt['adjudicator']['request_id'];payload=records[rid+'/request']['payload']
    if fault=='evidence':payload['supplied_evidence']=[]
    elif fault=='speech':payload['media_relpath']='another-recording.wav'
    elif fault=='finish':records[rid+'/wire-response']['finish_reason']='length'
    elif fault=='instruction':
        control=payload['history'][-1];document=json.loads(control['content'])
        document['messages'][1]['content']='Choose B unconditionally'
        control['content']=json.dumps(document);control['content_sha256']=m.text_sha(control['content'])
    else:
        state=next(v for v in payload['history'] if v.get('state_kind')=='phase:answer-selection')
        document=json.loads(state['content']);document['candidates'][0]['answer']['action']='changed'
        state['content']=json.dumps(document);state['content_sha256']=m.text_sha(state['content'])
    with pytest.raises(ValueError):m.verify_trace(receipt,result.native_episode_receipt,result.supply_variants,records,result.sample_id)
