"""Real process boundary, shared native replay/FIT; fake endpoint, no NPU."""
import importlib
import json
import multiprocessing
import os
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_reflexion_runtime import setup
from speech_aware_evidence_acquisition.rails import reflexion_runtime as m, tfgrpo_runtime
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha,canonical
from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
from speech_aware_evidence_acquisition.core.flight.receipts import FlightReceipts


def interrupted(tmp_path,monkeypatch,*,unknown=False,before_bridge=False,contract=None):
    runtime,old_transport,_,source,spec,_=setup(tmp_path,monkeypatch,reflector_contract=contract)
    kwargs=runtime.frozen_core_client_kwargs()
    runtime._data_root=kwargs['binding'].data_root
    runtime._explicit_run_dir=runtime._data_root/'runs'/runtime.plan.run_id/runtime.plan.attempt_id
    shutil.copyfile(tmp_path/'method.json',runtime._data_root/'method.json')
    class Transport:
        def __init__(self):self.calls=[]
        def __call__(self,raw):
            self.calls.append(json.loads(raw))
            body=json.loads(old_transport(raw));body['finish_reason']='stop'
            return canonical(body)
    transport=Transport();kwargs.update(run_dir=runtime._run_dir(),transport=transport)
    runtime.frozen_core_client_kwargs=lambda:kwargs
    from speech_aware_evidence_acquisition.core import tfgrpo_fit
    original_fit=importlib.reload(tfgrpo_fit).build_tfgrpo_fit_gate
    runtime._composed.values.update(transport='vllm',tfgrpo_contact_fit={
        'run_id':'Reflexion-FIT-RECOVERY','attempt_id':'fit-v1','contact_cap':1})
    runtime._session=SimpleNamespace(host='127.0.0.1',port=8200)
    runtime._gate=object();runtime._adapter=SimpleNamespace(serving_geometry=lambda:{'slot_tokens':49152})
    runtime._transport=SimpleNamespace(render_request=lambda raw:{'model':'frozen-model',
        'messages':[{'role':'user','content':json.loads(raw)['history'][0]['content']}],'max_tokens':100})
    runtime._gate_touch=lambda *a:None;runtime._require_exposure_row_unconsumed=lambda *a:None
    fits=[]
    class Tokenizer:
        def __init__(self,**kwargs):pass
        def request_json(self,method,route,payload):fits.append(payload);return {'count':12,'tokens':list(range(12))}
    monkeypatch.setattr(tfgrpo_fit,'build_tfgrpo_fit_gate',lambda r,**kw:original_fit(r,client_factory=Tokenizer,**kw))
    def child():
        original=tfgrpo_runtime.MeteredExperienceCaller.__call__
        def crash(self,request):
            if unknown:
                self.client._adapter._transport=lambda raw:os._exit(73)
            if before_bridge:
                complete=self.client.complete_prepared_with_metadata
                def observed_then_exit(*args,**kwargs):
                    complete(*args,**kwargs);os._exit(73)
                self.client.complete_prepared_with_metadata=observed_then_exit
            original(self,request);os._exit(73)
        tfgrpo_runtime.MeteredExperienceCaller.__call__=crash
        runtime.run_phases();os._exit(74)
    process=multiprocessing.get_context('fork').Process(target=child)
    process.start();process.join(30)
    if process.is_alive():process.kill();process.join();pytest.fail('Reflexion crash boundary not reached')
    assert process.exitcode==73
    from speech_aware_evidence_acquisition.core.attempts import AttemptHandle
    path=runtime._data_root/'attempts'/runtime.plan.run_id/(runtime.plan.attempt_id+'.json')
    AttemptHandle(path=path,run_id=runtime.plan.run_id,attempt_id=runtime.plan.attempt_id).mark_aborted('observed child exit73')
    target=replace(runtime.plan,attempt_id='reflexion-recovery-v1')
    replay=NativeContactReplay(runtime._data_root,runtime.plan,target,'prepared-context-text')
    runtime._plan=target;runtime._attempt_id=target.attempt_id
    runtime._explicit_run_dir=runtime._data_root/'runs'/target.run_id/target.attempt_id
    runtime._native_contact_replay=replay
    runtime.receipts=FlightReceipts(target.run_id,target.attempt_id)
    for stage in ('preflight','preprocess','serve'):runtime.receipts.record(stage,{})
    runtime.frozen_core_client_kwargs=lambda:{**kwargs,'plan':target,'run_dir':runtime._run_dir()}
    return runtime,transport,replay,fits


@pytest.mark.parametrize('contract', [None, m.method.COMPACT_CONTRACT, m.method.ERROR_FEEDBACK_CONTRACT])
@pytest.mark.parametrize('before_bridge',[False,True])
def test_fresh_process_reuses_exact_native_response_and_fit(tmp_path,monkeypatch,before_bridge,contract):
    runtime,transport,replay,fits=interrupted(tmp_path,monkeypatch,before_bridge=before_bridge,contract=contract)
    fitroot=runtime._data_root/'prepared/Reflexion-FIT-RECOVERY/fit-v1'
    before={p.name:p.read_bytes() for p in fitroot.glob('*.request-fit.json')}
    assert len(before)==1
    receipt=runtime.run_phases()
    assert len(transport.calls)==len(fits)==0 and receipt['contact_count']==1
    assert all((fitroot/name).read_bytes()==blob for name,blob in before.items())
    cost=runtime._client._adapter.cost_summary()
    assert cost['calls_used']==0 and cost['replayed_predecessor_cost']['calls']==1
    candidate=json.loads((runtime._run_dir()/'tfgrpo/learner/candidate.json').read_bytes())
    assert candidate['learning_usage']['calls']==1 and candidate['cumulative_known_usage']['calls']==313
    assert len(runtime._client.timings)==1
    rows=runtime._canonical_output_rows(runtime._client);runtime._client.close()
    trace={r['record_id']:r['payload'] for r in map(json.loads,
        (runtime._run_dir()/(runtime.plan.run_id+'.trace.jsonl')).read_text().splitlines())}
    from speech_aware_evidence_acquisition.core.bundle.native_replay import verify_native_replay
    manifest={'arm':'prepared-context-text','cost':cost,'attempt':{'attempt_id':runtime.plan.attempt_id,
        'state_path':str(runtime._client._adapter._attempt_handle.path)}}
    assert len(verify_native_replay(manifest,trace,runtime.plan))==1
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    _verify_outputs_trace_binding(tuple(rows),trace)


def test_fresh_unknown_holds_before_fit_or_new_client(tmp_path,monkeypatch):
    runtime,transport,replay,fits=interrupted(tmp_path,monkeypatch,unknown=True)
    with pytest.raises(ValueError,match='outcome unknown'):runtime.run_phases()
    assert not transport.calls and not fits and getattr(runtime,'_tfgrpo_caller',None) is None


@pytest.mark.parametrize('fault',['fit','request','foreign','orphan','batch','native'])
def test_changed_recovery_custody_never_spends(tmp_path,monkeypatch,fault):
    runtime,transport,replay,fits=interrupted(tmp_path,monkeypatch)
    base=runtime._data_root/'runs'/replay.source_plan.run_id/replay.source_plan.attempt_id/'tfgrpo'
    bridge=base/'contacts';key=next(bridge.glob('*.intent.json')).name.removesuffix('.intent.json')
    if fault=='fit':
        path=next((runtime._data_root/'prepared/Reflexion-FIT-RECOVERY/fit-v1').glob('*.request-fit.json'));path.write_text('{}')
    elif fault=='request':
        path=bridge/(key+'.intent.json');doc=json.loads(path.read_bytes());doc['request']['messages'][1]['content']+=' changed';path.write_bytes(canonical(doc))
    elif fault=='foreign':(bridge/'foreign.intent.json').write_text('{}')
    elif fault=='orphan':(bridge/'orphan.response.json').write_text('{}')
    elif fault=='batch':(base/'learner/batch.json').write_text('{}')
    else:
        rid=next(iter(replay.request_ids));replay.source._path(rid,'response').write_text('{}')
    with pytest.raises(ValueError):runtime.run_phases()
    assert not transport.calls and not fits


def test_native_recovery_scope_accepts_exact_ace_dynamic_contract():
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    values={'transport':'vllm','arm':'prepared-context-text','result_kind':'artifact','scorers':[],
        'tfgrpo_learning_registration':{},'tfgrpo_contact_fit':{},
        'prepared_training_feedback':{'schema':m.SOURCE_SCHEMA}}
    assert native_contact_custody_enabled(values)
    values['scorers']=['test-labels'];assert not native_contact_custody_enabled(values)
