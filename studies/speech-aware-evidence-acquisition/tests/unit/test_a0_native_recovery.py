"""Actual two-role raw speech graph recovery, without odd40 selection arms."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
from derive_lock_only_baseline import derive
from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay, ContactJournalError
from speech_aware_evidence_acquisition.core.native_fit_recovery import RecoverableNativeFitGate
from speech_aware_evidence_acquisition.core.preparation_fit import build_native_contact_fit_gate
from test_native_episode_runner import _ActualEpisodeHarness, _review_off_config
from test_native_contact_fit import _raw_runtime, _RawTokenizer
from test_llama_server_transport import _spawn_live_matching_process
ROOT = Path(__file__).resolve().parents[2]


def baseline():
    source=json.loads((ROOT/'configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json').read_text())
    return {**derive(source), 'transport':'vllm', 'carrier_lock_key':'slurp'}


def test_actual_a0_config_enables_capture_without_changing_inputs():
    values=baseline();before=copy.deepcopy(values)
    assert native_contact_custody_enabled(values)
    assert values==before


@pytest.mark.parametrize('change', ['transport','carrier','tool','revision','variant','selection','samples','role-count','bool-iteration'])
def test_other_graphs_are_not_enabled_by_two_contact_ceiling(change):
    values=baseline();native=values['native_episode']
    if change=='transport':values['transport']='llama.cpp'
    elif change=='carrier':values['carrier_lock_key']='earnings21'
    elif change=='tool':values['phase_tools']={'solver':{'allowed_tools':['lookup']}}
    elif change=='revision':native['max_revisions']=1
    elif change=='variant':native['roles']['solver']['supply_variants']={}
    elif change=='selection':native['answer_selection']='another-selector'
    elif change=='samples':native['roles']['solver']['samples']=2
    elif change=='role-count':native['roles']['reviewer']['max_iterations']=1
    else:native['roles']['solver']['max_iterations']=True
    assert not native_contact_custody_enabled(values)


def harness(tmp_path, *, plan=None, data=None, replay=None):
    h=_ActualEpisodeHarness(tmp_path,sample_count=1,max_revisions=0,max_contacts=2,planner_actions=(),
        arm='raw-speech-agent-native-tools',query_source='speech_audio',contact_journal=True,
        execution_plan=plan,data_root=data,contact_replay=replay)
    h.config=replace(_review_off_config(max_contacts=2),prompt_contact_limits='legacy-c17-lock-v1')
    runtime=_raw_runtime(h.world.data_root)
    runtime._samples=h.samples
    runtime._transport=h.transport
    runtime._composed.values['native_contact_fit']['contact_cap']=2
    runtime._adapter=SimpleNamespace(serving_geometry=lambda:{'slot_tokens':24576})
    calls=[]
    class Tokenizer(_RawTokenizer):
        def request_json(self,*args):
            calls.append(args)
            return super().request_json(*args)
    gate=RecoverableNativeFitGate(runtime,h.plan,lambda:build_native_contact_fit_gate(
        runtime,h.program,h.plan,client_factory=Tokenizer),replay=replay)
    original=h.request_gate
    def fit(request):
        original(request)
        result=gate(request)
        if replay is not None and replay.match(request.as_payload()) is not None:
            h.replayed_fixture_keys.add(next(reversed(h.prepared)))
        return result
    h.request_gate=fit
    return h,gate,calls


def stopped(tmp_path, *, unknown=False):
    h,gate,calls=harness(tmp_path/'source')
    if unknown:
        h._output=lambda key: (_ for _ in ()).throw(TimeoutError('unknown perception result'))
    else:
        sink=h.client._adapter._sink;emit=sink.emit
        def fail(ch,rid,payload):
            result=emit(ch,rid,payload)
            if ch=='MODEL_RESPONSE':raise RuntimeError('stop after durable P')
            return result
        sink.emit=fail
    try:
        with pytest.raises(Exception):h.run()
    finally:h.close()
    h.client._adapter._attempt_handle.mark_aborted('known terminal source in graph test')
    return h,gate,calls


def test_actual_ps_recovery_reuses_p_and_fit_then_sends_only_s(tmp_path):
    from test_template_variants_delivery import _runtime,_raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    from speech_aware_evidence_acquisition.core.bundle.native_replay import verify_native_replay
    old,oldfit,oldcalls=stopped(tmp_path)
    assert len(oldcalls)==1
    plan=replace(old.plan,attempt_id='a0-recovery-v1')
    replay=NativeContactReplay(old.world.data_root,old.plan,plan,old.client.arm)
    new,fit,calls=harness(tmp_path/'target',plan=plan,data=old.world.data_root,replay=replay)
    try:
        summary=new.run()
        raw=_raw_native(new,solver_role={'max_iterations':1})
        outputs=_runtime(new,summary,raw)._canonical_output_rows(new.client)
    finally:cost=new.close()['cost']
    assert [v['role_id'] for v in summary.results[0].native_episode_receipt['role_visits']]==['perception','solver']
    assert len(new.dispatched)==1 and next(iter(new.dispatched))[1]=='solver'
    assert cost['calls_used']==1 and cost['replayed_predecessor_cost']['calls']==1
    assert len(calls)==1 and len(fit.receipts)==2
    assert fit.receipts[0]['native_fit_reuse']['new_tokenizer_contacts']==0
    assert fit.receipts[0]['path']==oldfit.receipts[0]['path']
    records={rid:p for channel,rid,p in new.trace.records}
    _verify_outputs_trace_binding(tuple(outputs),records)
    assert verify_native_replay({'arm':new.client.arm,'attempt':{'attempt_id':plan.attempt_id,'state_path':str(new.client._adapter._attempt_handle.path)},'cost':cost},records,plan)
    assert sum(p['calls'] for channel,rid,p in new.trace.records if channel=='COST')==2


def test_unknown_p_refuses_before_any_new_fit_or_generation(tmp_path):
    old,_,_=stopped(tmp_path,unknown=True)
    plan=replace(old.plan,attempt_id='a0-recovery-v1')
    replay=NativeContactReplay(old.world.data_root,old.plan,plan,old.client.arm)
    new,fit,calls=harness(tmp_path/'target',plan=plan,data=old.world.data_root,replay=replay)
    try:
        with pytest.raises(Exception):new.run()
    finally:cost=new.close()['cost']
    assert not calls and not new.dispatched and not fit.receipts
    assert cost['calls_used']==0


def test_altered_p_does_not_fall_through_to_new_request(tmp_path):
    old,_,_=stopped(tmp_path)
    plan=replace(old.plan,attempt_id='a0-recovery-v1')
    replay=NativeContactReplay(old.world.data_root,old.plan,plan,old.client.arm)
    payload=copy.deepcopy(replay.source.inspect(next(iter(replay.request_ids)))['intent']['payload'])
    payload['task_instruction']+=' changed'
    with pytest.raises(ContactJournalError,match='changed'):replay.match(payload)


def test_process_exit_after_p_preserves_response_and_fit_for_new_process_context(tmp_path):
    import multiprocessing,os
    old,_,_=harness(tmp_path/'source')
    sink=old.client._adapter._sink;emit=sink.emit
    def child():
        def exit_after_p(ch,rid,payload):
            result=emit(ch,rid,payload)
            if ch=='MODEL_RESPONSE':os._exit(73)
            return result
        sink.emit=exit_after_p
        old.run();os._exit(74)
    process=multiprocessing.get_context('fork').Process(target=child)
    process.start();process.join(30)
    if process.is_alive():
        process.kill();process.join();pytest.fail('child did not reach durable P boundary')
    assert process.exitcode==73
    old.close()
    old.client._adapter._attempt_handle.mark_aborted('source child exit73 verified')
    plan=replace(old.plan,attempt_id='a0-process-recovery-v1')
    replay=NativeContactReplay(old.world.data_root,old.plan,plan,old.client.arm)
    new,fit,calls=harness(tmp_path/'target',plan=plan,data=old.world.data_root,replay=replay)
    try:summary=new.run()
    finally:cost=new.close()['cost']
    assert summary.results[0].final_response_text=='solver-answer:s1:visit:1'
    assert len(new.dispatched)==1 and len(calls)==1
    assert fit.receipts[0]['native_fit_reuse']['new_tokenizer_contacts']==0
    assert cost['calls_used']==1 and cost['replayed_predecessor_cost']['calls']==1
