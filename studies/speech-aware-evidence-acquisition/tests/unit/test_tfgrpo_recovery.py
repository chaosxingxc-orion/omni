"""Fresh-process learning replays durable responses through the shared adapter."""
import json
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path
import pytest

from test_tfgrpo_dispatch import make_runtime
from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
from speech_aware_evidence_acquisition.core.flight.receipts import FlightReceipts
from speech_aware_evidence_acquisition.rails import tfgrpo_runtime


def interrupted(tmp_path,monkeypatch,*,unknown=False,actual_fit=False):
    runtime,transport,_=make_runtime(tmp_path,monkeypatch)
    # Fixture artifacts and carrier share an owned root, as in production.
    kwargs=runtime.frozen_core_client_kwargs()
    runtime._data_root=kwargs['binding'].data_root
    runtime._explicit_run_dir=runtime._data_root/'runs'/runtime.plan.run_id/runtime.plan.attempt_id
    kwargs['run_dir']=runtime._run_dir()
    runtime.frozen_core_client_kwargs=lambda:kwargs
    fit_calls=[]
    if actual_fit:
        from types import SimpleNamespace
        from speech_aware_evidence_acquisition.core import tfgrpo_fit
        # Undo make_runtime's stub; run the real durable tokenizer/FIT path.
        import importlib
        original_fit=importlib.reload(tfgrpo_fit).build_tfgrpo_fit_gate
        runtime._composed.values.update(transport='vllm',tfgrpo_contact_fit={
            'run_id':'FIT-RECOVERY','attempt_id':'fit-v1','contact_cap':5})
        runtime._session=SimpleNamespace(host='127.0.0.1',port=8200)
        runtime._gate=object()
        runtime._adapter=SimpleNamespace(serving_geometry=lambda:{'slot_tokens':100000})
        runtime._transport=SimpleNamespace(render_request=lambda raw:{'model':'frozen-model',
            'messages':[{'role':'user','content':json.loads(raw)['history'][0]['content']}],
            'max_tokens':100})
        runtime._gate_touch=lambda *a:None
        runtime._require_exposure_row_unconsumed=lambda *a:None
        class Tokenizer:
            def __init__(self,**kwargs):pass
            def request_json(self,method,route,payload):
                fit_calls.append(payload);return {'count':10,'tokens':list(range(10))}
        monkeypatch.setattr(tfgrpo_fit,'build_tfgrpo_fit_gate',lambda r,**kw:
            original_fit(r,client_factory=Tokenizer,**kw))
    source=json.loads((tmp_path/'source.json').read_bytes())
    import shutil
    for name in ('source.json','evidence.json'):
        shutil.copyfile(tmp_path/name,runtime._data_root/name)
    def child():
        original=tfgrpo_runtime.MeteredExperienceCaller.__call__
        def crash(self,request):
            if unknown and request['stage']=='advantage':
                original_transport=self.client._adapter._transport
                def lost(raw):os._exit(73)
                self.client._adapter._transport=lost
            result=original(self,request)
            if not unknown and request['stage']=='advantage':os._exit(73)
            return result
        tfgrpo_runtime.MeteredExperienceCaller.__call__=crash
        runtime.run_phases()
        os._exit(74)
    child_process=multiprocessing.get_context('fork').Process(target=child)
    child_process.start();child_process.join(30)
    if child_process.is_alive():
        child_process.kill();child_process.join();pytest.fail('crash boundary not reached')
    assert child_process.exitcode==73
    from speech_aware_evidence_acquisition.core.attempts import AttemptHandle
    path=runtime._data_root/'attempts'/runtime.plan.run_id/(runtime.plan.attempt_id+'.json')
    handle=AttemptHandle(path=path,run_id=runtime.plan.run_id,attempt_id=runtime.plan.attempt_id)
    handle.mark_aborted('child exit73 authoritatively observed')
    target=replace(runtime.plan,attempt_id='recovery-v1')
    replay=NativeContactReplay(runtime._data_root,runtime.plan,target,'prepared-context-text')
    runtime._plan=target;runtime._attempt_id=target.attempt_id
    runtime._explicit_run_dir=runtime._data_root/'runs'/target.run_id/target.attempt_id
    runtime._native_contact_replay=replay
    runtime.receipts=FlightReceipts(target.run_id,target.attempt_id)
    for stage in ('preflight','preprocess','serve'):runtime.receipts.record(stage,{})
    runtime.frozen_core_client_kwargs=lambda:{**kwargs,'plan':target,'run_dir':runtime._run_dir()}
    runtime.test_fit_calls=fit_calls
    return runtime,transport,replay


def test_process_exit_after_observed_response_resumes_without_duplicate_model_calls(tmp_path,monkeypatch):
    runtime,transport,replay=interrupted(tmp_path,monkeypatch)
    result=runtime.run_phases()
    assert result['contact_count']==5
    assert len(transport.calls)==2
    cost=runtime._client._adapter.cost_summary()
    assert cost['calls_used']==2 and cost['replayed_predecessor_cost']['calls']==3
    assert len(runtime._client.timings)==5
    assert result['learning']['usage']['calls']==5
    rows=runtime._canonical_output_rows(runtime._client)
    runtime._client.close()
    from speech_aware_evidence_acquisition.core.bundle.native_replay import verify_native_replay
    trace={r['record_id']:r['payload'] for r in map(json.loads,
        (runtime._run_dir()/(runtime.plan.run_id+'.trace.jsonl')).read_text().splitlines())}
    manifest={'arm':'prepared-context-text','cost':cost,'attempt':{'attempt_id':runtime.plan.attempt_id,
        'state_path':str(runtime._client._adapter._attempt_handle.path)}}
    assert len(verify_native_replay(manifest,trace,runtime.plan))==3
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    _verify_outputs_trace_binding(tuple(rows),trace)


def test_process_exit_with_unknown_request_holds_without_any_new_transport(tmp_path,monkeypatch):
    runtime,transport,replay=interrupted(tmp_path,monkeypatch,unknown=True)
    with pytest.raises(ValueError,match='outcome unknown'):runtime.run_phases()
    assert not transport.calls
    assert runtime._client._adapter.cost_summary()['calls_used']==0
    runtime._client.close()


def test_recovery_cannot_relax_registration_without_verified_source(tmp_path,monkeypatch):
    runtime,transport,replay=interrupted(tmp_path,monkeypatch)
    runtime._native_contact_replay=None
    with pytest.raises(ValueError,match='exact TRAIN'):runtime.run_phases()
    assert not transport.calls


def test_successor_reuses_original_real_fit_artifacts_without_tokenizer_replay(tmp_path,monkeypatch):
    runtime,transport,replay=interrupted(tmp_path,monkeypatch,actual_fit=True)
    before={p.name:p.read_bytes() for p in (runtime._data_root/'prepared/FIT-RECOVERY/fit-v1').glob('*.request-fit.json')}
    assert len(before)==3
    runtime.run_phases()
    assert len(runtime.test_fit_calls)==2 and len(transport.calls)==2
    for name,blob in before.items():
        assert (runtime._data_root/'prepared/FIT-RECOVERY/fit-v1'/name).read_bytes()==blob
    runtime._client.close()


def test_native_recovery_scope_requires_registered_dynamic_artifact():
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    values={'transport':'vllm','arm':'prepared-context-text','result_kind':'artifact','scorers':[],
        'tfgrpo_learning_registration':{},'tfgrpo_contact_fit':{},
        'prepared_training_feedback':{'schema':'saea-prepared-tfgrpo-dynamic-v1'}}
    assert native_contact_custody_enabled(values)
    for field in ('tfgrpo_learning_registration','tfgrpo_contact_fit','prepared_training_feedback','result_kind'):
        changed=dict(values);changed.pop(field)
        assert not native_contact_custody_enabled(changed)
