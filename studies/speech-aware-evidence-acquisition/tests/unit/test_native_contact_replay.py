import copy
import json
from dataclasses import replace
import pytest
from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay, ContactJournalError
from test_whole_answer_selection_live import harness_for
from test_llama_server_transport import _spawn_live_matching_process
from test_whole_answer_selection import text


def interrupt_after_known_contacts(tmp_path, count=2, unknown=False):
    h,values=harness_for(tmp_path/'source',raw_audio=True,contact_journal=True)
    sink=h.client._adapter._sink;emit=sink.emit;seen=[]
    def fail_emit(ch,rid,payload):
        result=emit(ch,rid,payload)
        if ch=='MODEL_RESPONSE':
            seen.append(rid)
            if len(seen)==count:raise RuntimeError('process failed after durable response')
        return result
    if unknown:
        original=h._output
        def uncertain(key):
            if key[1]=='selection':raise TimeoutError('unknown judge result')
            return original(key)
        h._output=uncertain
    else:sink.emit=fail_emit
    try:
        with pytest.raises(Exception):h.run()
    finally:h.close()
    h.client._adapter._attempt_handle.mark_aborted('verified stopped source test process')
    return h,values


def replay_for(h):
    plan=replace(h.plan,attempt_id='recovery-v1')
    replay=NativeContactReplay(h.world.data_root,h.plan,plan,h.client.arm)
    return plan,replay


@pytest.mark.parametrize('count',[1,2,3,4,5])
def test_partial_graph_reuses_original_contacts_without_new_transport_or_spend(tmp_path,count):
    from test_template_variants_delivery import _runtime,_raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    old,values=interrupt_after_known_contacts(tmp_path,count)
    plan,replay=replay_for(old)
    new,_=harness_for(tmp_path/'recovery',raw_audio=True,contact_journal=True,
        contact_replay=replay,execution_plan=plan,data_root=old.world.data_root)
    original_gate=new.request_gate
    def gate(request):
        original_gate(request)
        p=request.as_payload()
        key=next(reversed(new.prepared))
        logical=(p['sample_id'],next(item['contact_id'] for item in reversed(p['history']) if item.get('role')=='control'))
        if logical in replay._by_contact:
            new.replayed_fixture_keys.add(key)
    new.request_gate=gate
    try:
        result=new.run()
        raw=_raw_native(new,solver_role=values['native_episode']['roles']['solver']);raw['answer_selection']='odd40-whole-answer-v1'
        rows=_runtime(new,result,raw)._canonical_output_rows(new.client)
    finally:cost=new.close()['cost']
    assert len(result.results)==1 and result.results[0].final_response_text==text('two')
    assert len(new.dispatched)==5-count
    assert cost['calls_used']==5-count
    assert cost['replayed_predecessor_cost']['calls']==count
    assert old.client._adapter._attempt_handle.actual_totals()[0]==count
    assert new.client._adapter._attempt_handle.actual_totals()[0]==5-count
    records={rid:p for ch,rid,p in new.trace.records}
    _verify_outputs_trace_binding(tuple(rows),records)
    from speech_aware_evidence_acquisition.core.bundle.native_replay import verify_native_replay
    from speech_aware_evidence_acquisition.core.bundle.outputs import _minimum_model_calls_from_trace
    from speech_aware_evidence_acquisition.core.bundle import BundleError
    manifest = {'arm': new.client.arm, 'cost': cost,
        'attempt': {'attempt_id': plan.attempt_id,
                    'state_path': str(new.client._adapter._attempt_handle.path)}}
    verified = verify_native_replay(manifest, records, plan)
    assert len(verified) == count
    assert _minimum_model_calls_from_trace(records, verified_replayed_request_ids=verified) == 5-count
    assert cost['native_replay_source']['actual_usage']['calls'] == count
    for field in ('calls', 'audio_seconds', 'prompt_tokens', 'completion_tokens', 'latency_seconds'):
        changed = copy.deepcopy(manifest)
        changed['cost']['replayed_predecessor_cost'][field] += 1
        with pytest.raises(BundleError): verify_native_replay(changed, records, plan)
    rid = next(iter(verified))
    for suffix, key, value in (('/request', 'payload_sha256', '0'*64),
            ('/response', 'text', 'tampered'), ('/cost', 'calls', 0),
            ('/contact-replay', 'new_model_calls', 1)):
        changed = copy.deepcopy(records); changed[rid+suffix][key] = value
        with pytest.raises(BundleError): verify_native_replay(manifest, changed, plan)
    changed = copy.deepcopy(records); del changed[rid+'/contact-replay']
    with pytest.raises(BundleError): verify_native_replay(manifest, changed, plan)
    changed = copy.deepcopy(manifest); del changed['cost']['native_replay_source']
    with pytest.raises(BundleError): verify_native_replay(changed, records, plan)
    prior_ids=replay.request_ids
    assert prior_ids <= {rid.removesuffix('/response') for ch,rid,p in new.trace.records if ch=='MODEL_RESPONSE'}
    assert sum(p['calls'] for ch,rid,p in new.trace.records if ch=='COST')==5


def test_unknown_judge_result_stops_without_any_new_model_request(tmp_path):
    old,_=interrupt_after_known_contacts(tmp_path,unknown=True)
    plan,replay=replay_for(old)
    new,_=harness_for(tmp_path/'recovery',raw_audio=True,contact_journal=True,
        contact_replay=replay,execution_plan=plan,data_root=old.world.data_root)
    try:
        with pytest.raises(Exception):new.run()
    finally:cost=new.close()['cost']
    assert not new.dispatched
    assert cost['calls_used']==0 and cost['replayed_predecessor_cost']['calls']==4
    assert new.client._adapter._attempt_handle.actual_totals()[0]==0


def test_changed_request_never_falls_through_to_transport(tmp_path):
    old,_=interrupt_after_known_contacts(tmp_path)
    _,replay=replay_for(old)
    rid=next(iter(replay.request_ids));p=copy.deepcopy(replay.source.inspect(rid)['intent']['payload'])
    p['decoding_params']['temperature']=.5
    with pytest.raises(ContactJournalError,match='changed'):replay.match(p)


def test_open_source_or_changed_budget_cannot_supply_replay(tmp_path):
    old,_=interrupt_after_known_contacts(tmp_path)
    plan=replace(old.plan,attempt_id='recovery-v1')
    with pytest.raises(ContactJournalError):
        NativeContactReplay(old.world.data_root,old.plan,replace(plan,planned_model_calls=plan.planned_model_calls+1),old.client.arm)
    path=old.client._adapter._attempt_handle.path
    document=json.loads(path.read_bytes());document['state']='OPENED';path.write_text(json.dumps(document))
    with pytest.raises(ContactJournalError,match='aborted'):
        NativeContactReplay(old.world.data_root,old.plan,plan,old.client.arm)


def test_actual_process_exit_then_new_process_context_replays_two_contacts(tmp_path):
    import multiprocessing,os
    old,_=harness_for(tmp_path/'source',raw_audio=True,contact_journal=True)
    sink=old.client._adapter._sink;emit=sink.emit
    def child():
        count=0
        def exit_after_response(ch,rid,payload):
            nonlocal count
            result=emit(ch,rid,payload)
            if ch=='MODEL_RESPONSE':
                count+=1
                if count==2:os._exit(73)
            return result
        sink.emit=exit_after_response
        old.run()
        os._exit(74)
    process=multiprocessing.get_context('fork').Process(target=child)
    process.start();process.join(30)
    if process.is_alive():
        process.kill();process.join();pytest.fail('source process did not reach crash boundary')
    assert process.exitcode==73
    old.close()
    old.client._adapter._attempt_handle.mark_aborted('source process exit73 verified')
    plan,replay=replay_for(old)
    new,_=harness_for(tmp_path/'recovery',raw_audio=True,contact_journal=True,
        contact_replay=replay,execution_plan=plan,data_root=old.world.data_root)
    try:result=new.run()
    finally:cost=new.close()['cost']
    assert result.results[0].final_response_text==text('two')
    assert len(new.dispatched)==cost['calls_used']==3
    assert cost['replayed_predecessor_cost']['calls']==2
    assert old.client._adapter._attempt_handle.actual_totals()[0]==2


def replay_fixture(tmp_path, *, raw=b'{"text":"","finish_reason":"length","usage":{}}', with_fallback=False):
    from test_frozen_core_adapter import _plan
    from speech_aware_evidence_acquisition.core.attempts import AttemptStore
    from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal,canonical,digest
    from speech_aware_evidence_acquisition.core.reasoning_knobs import answer_only_decoding
    from test_contact_journal import payload
    plan=_plan(planned_model_calls=4,planned_speech_audio_seconds=400)
    handle=AttemptStore(tmp_path/'attempts').open_attempt(plan.run_id,plan.attempt_id,
        carrier_lock_key=plan.carrier_lock_key,reserved_calls=4,reserved_audio_seconds=400)
    journal=ContactJournal(tmp_path,plan,'bare-core')
    p=payload();p['decoding_params'].update(max_tokens=4096,thinking_budget_tokens=3072)
    journal.intent(p);handle.record_usage(calls=1,audio_seconds=100)
    journal.response(p['request_id'],digest(canonical(p)),raw,1)
    fallback=copy.deepcopy(p);fallback['request_id']='req-2'
    fallback['decoding_params']=answer_only_decoding(p['decoding_params'])
    if with_fallback:
        journal.intent(fallback);handle.record_usage(calls=1,audio_seconds=100)
        journal.response('req-2',digest(canonical(fallback)),b'{"text":"answer","usage":{}}',1)
    handle.mark_aborted('test source stopped')
    return NativeContactReplay(tmp_path,plan,replace(plan,attempt_id='recovery-v1'),'bare-core'),p,fallback


def test_unsent_exact_answer_only_fallback_remains_available(tmp_path):
    replay,p,fallback=replay_fixture(tmp_path)
    assert replay.match(p)==p
    assert replay.match(fallback) is None
    wrong=copy.deepcopy(fallback);wrong['decoding_params']['temperature']=.7
    with pytest.raises(ContactJournalError):replay.match(wrong)


def test_already_observed_fallback_has_its_own_identity(tmp_path):
    replay,p,fallback=replay_fixture(tmp_path,with_fallback=True)
    assert replay.match(fallback)==fallback
    assert replay.consume(p)['response_bytes']!=replay.consume(fallback)['response_bytes']
    with pytest.raises(ContactJournalError):replay.consume(fallback)


@pytest.mark.parametrize('raw',[b'{"text":"answer","finish_reason":"length"}', b'{"text":"","finish_reason":"stop"}'])
def test_completed_source_cannot_authorize_extra_fallback(tmp_path,raw):
    replay,p,fallback=replay_fixture(tmp_path,raw=raw)
    with pytest.raises(ContactJournalError):replay.match(fallback)


def test_source_changed_after_indexing_is_never_used(tmp_path):
    replay,p,_=replay_fixture(tmp_path)
    path=replay.source._path(p['request_id'],'response')
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ContactJournalError):replay.match(p)
