"""Shared adapter/dispatcher tests; tokenizer and historical seal are boundary doubles."""
import copy
from dataclasses import replace
import hashlib
import json

import pytest

from speech_aware_evidence_acquisition.rails import grounded_synthesis_runtime as m
from speech_aware_evidence_acquisition.rails import tfgrpo_runtime
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical, sha
from test_tfgrpo_dispatch import make_runtime
from test_grounded_synthesis_plan import source, count
from speech_aware_evidence_acquisition.rails.grounded_synthesis_plan import plan_batches


def setup(tmp_path, monkeypatch):
    runtime, transport, captured = make_runtime(tmp_path,monkeypatch)
    sid = runtime._samples[0].sample_id
    batch, ops = source()
    batch['groups'] = batch['groups'][:1]
    batch['groups'][0]['query_id'] = sid
    batch['model_id'] = 'frozen-model'
    ops = ops[:2]
    for row in ops:
        row['group_contact'] = sha({'stage':'group_update','identity':sid})
    bb,ob = canonical(batch),canonical(ops)
    (tmp_path/'batch.json').write_bytes(bb)
    (tmp_path/'ops.json').write_bytes(ob)
    tokenizer = {'fixture':'local-tokenizer'}
    plan = plan_batches(bb,ob,source_batch_sha256=hashlib.sha256(bb).hexdigest(),
        source_operations_sha256=hashlib.sha256(ob).hexdigest(),count_tokens=count,
        tokenizer_identity=tokenizer,expected_groups=1,expected_operations=2,
        resident_context=500,completion_cap=100,safety_margin=100)
    custody = {'source_native_custody_verified':True,'model_id':'frozen-model','unknown_contacts':0,
        'source_batch_sha256':plan['source_batch_sha256'],
        'source_operations_sha256':plan['source_operations_sha256'],
        'native_source_usage':[{'run_id':'old','attempt_id':'a',
            'usage':{'calls':309,'prompt_tokens':1000,'completion_tokens':500}}],
        'original_total_usage':{'calls':309,'prompt_tokens':1000,'completion_tokens':500}}
    monkeypatch.setattr(m,'load_source',lambda *a,**k:copy.deepcopy(custody))
    monkeypatch.setattr(m,'load_token_counter',lambda p:(count,tokenizer))
    runtime._plan = replace(runtime.plan, planned_model_calls=1)
    original_kwargs = runtime.frozen_core_client_kwargs
    runtime.frozen_core_client_kwargs = lambda:{**original_kwargs(),'plan':runtime.plan}
    registration,_ = m.build_registration(runtime.plan,shard_plan=plan,custody=custody,
        model_id='frozen-model',decoding_params={'max_tokens':100,'temperature':0},
        max_context_bytes=10000,runtime_identity=runtime.session_required_identity(),
        roster_sha256='c'*64,source_batch=batch)
    doc = {'schema':'saea-grounded-runtime-source-v1','plan':plan,
        'source':{'source_batch':{'relpath':'batch.json'},'operations':{'relpath':'ops.json'}},
        'tokenizer_relpath':'tokenizer','max_operations':128,'concurrency':1}
    (tmp_path/'grounded.json').write_bytes(canonical(doc))
    runtime._composed.values.update(tfgrpo_learning_registration=registration,
        prepared_training_feedback={'schema':m.SOURCE_SCHEMA,'source':{'root':'data',
            'relpath':'grounded.json','sha256':sha(doc)}})
    return runtime,transport,captured,doc


def test_shared_dispatch_seal_keeps_intermediate_artifact_and_real_cost(tmp_path,monkeypatch):
    runtime,transport,captured,_ = setup(tmp_path,monkeypatch)
    receipt = runtime.run_phases()
    assert receipt['dispatch']=='registered-grounded-shards'
    assert receipt['candidate_publishable'] is False
    assert receipt['contact_count']==1 and len(transport.calls)==1
    assert receipt['learning']['historical_usage']['calls']==309
    seal = runtime.seal()
    assert seal['cost']['calls_used']==1 and seal['cost']['audio_seconds_used']==0
    assert captured['finalized'] and runtime._client._closed
    assert not (runtime._run_dir()/'tfgrpo/learner/candidate.json').exists()


@pytest.mark.parametrize('fault', ['model', 'negative', 'boolean', 'test_anchor'])
def test_registration_rejects_invalid_source_anchors(tmp_path, monkeypatch, fault):
    runtime, transport, _, doc = setup(tmp_path, monkeypatch)
    batch = json.loads((tmp_path/'batch.json').read_bytes())
    custody = m.load_source(None)
    if fault == 'model':
        batch['model_id'] = 'another-model'
    elif fault == 'test_anchor':
        batch['groups'][0]['query_id'] = 'slurp:test:one'
    else:
        doc['plan']['shards'][0]['group_indices'] = [-1 if fault == 'negative' else False]
    with pytest.raises(ValueError, match='source model|group index|TRAIN anchors'):
        m.build_registration(runtime.plan, shard_plan=doc['plan'], custody=custody,
            model_id='frozen-model', decoding_params={'max_tokens':100, 'temperature':0},
            max_context_bytes=10000, runtime_identity=runtime.session_required_identity(),
            roster_sha256='c'*64, source_batch=batch)
    assert not transport.calls


@pytest.mark.parametrize('fault',['messages','cap','cost','source','roster','old_schema'])
def test_drift_rejected_before_native_contact(tmp_path,monkeypatch,fault):
    runtime,transport,_,_ = setup(tmp_path,monkeypatch)
    reg = runtime._composed.values['tfgrpo_learning_registration']
    if fault=='messages':reg['contacts'][0]['messages_sha256']='a'*64
    elif fault=='cap':reg['completion_cap']=200
    elif fault=='cost':reg['batch_sha256']='a'*64
    elif fault=='source':(tmp_path/'batch.json').write_text('{}')
    elif fault=='roster':reg['contacts'][0]['sample_id']='slurp:train:other'
    else:reg['schema']='saea-tfgrpo-runtime-binding-v1'
    with pytest.raises(ValueError):runtime.run_phases()
    assert not transport.calls


def test_native_fit_rejection_has_no_model_intent(tmp_path,monkeypatch):
    runtime,transport,_,_ = setup(tmp_path,monkeypatch)
    from speech_aware_evidence_acquisition.core import tfgrpo_fit
    def reject(prepared):raise ValueError('wrapped input exceeds context')
    monkeypatch.setattr(tfgrpo_fit,'build_tfgrpo_fit_gate',lambda *a,**k:reject)
    with pytest.raises(ValueError,match='wrapped'):runtime.run_phases()
    assert not transport.calls
    assert not list((runtime._run_dir()/'tfgrpo/learner/contacts').glob('*.intent.json'))
    runtime._client.close()


def test_observed_bridge_handoff_recovers_without_resend(tmp_path,monkeypatch):
    runtime,transport,_,_ = setup(tmp_path,monkeypatch)
    original = tfgrpo_runtime.MeteredExperienceCaller.__call__
    failed = []
    def crash(self,request):
        response=original(self,request)
        if not failed:
            failed.append(True)
            raise OSError('lost learner handoff')
        return response
    monkeypatch.setattr(tfgrpo_runtime.MeteredExperienceCaller,'__call__',crash)
    with pytest.raises(OSError):runtime.run_phases()
    assert len(transport.calls)==1
    result=runtime.run_phases()
    assert result['contact_count']==1 and len(transport.calls)==1
    runtime._client.close()


def test_fresh_client_refuses_existing_intent(tmp_path,monkeypatch):
    runtime,transport,_,_ = setup(tmp_path,monkeypatch)
    runtime.run_phases()
    runtime._client.close()
    runtime._tfgrpo_caller=None
    with pytest.raises(ValueError,match='registered recovery'):m.run_registered_shards(runtime)
    assert len(transport.calls)==1


@pytest.mark.parametrize('unknown',[False,True])
def test_process_exit_native_replay_preserves_known_and_holds_unknown(tmp_path,monkeypatch,unknown):
    import multiprocessing
    import os
    import shutil
    from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
    from speech_aware_evidence_acquisition.core.attempts import AttemptHandle
    from speech_aware_evidence_acquisition.core.flight.receipts import FlightReceipts
    runtime,transport,_,_=setup(tmp_path,monkeypatch)
    kwargs=runtime.frozen_core_client_kwargs()
    runtime._data_root=kwargs['binding'].data_root
    runtime._explicit_run_dir=runtime._data_root/'runs'/runtime.plan.run_id/runtime.plan.attempt_id
    kwargs['run_dir']=runtime._run_dir()
    runtime.frozen_core_client_kwargs=lambda:kwargs
    for name in ('grounded.json','batch.json','ops.json'):
        shutil.copyfile(tmp_path/name,runtime._data_root/name)
    def child():
        original=tfgrpo_runtime.MeteredExperienceCaller.__call__
        def crash(self,request):
            if unknown:
                self.client._adapter._transport=lambda raw:os._exit(73)
            result=original(self,request)
            os._exit(73)
            return result
        tfgrpo_runtime.MeteredExperienceCaller.__call__=crash
        runtime.run_phases()
        os._exit(74)
    process=multiprocessing.get_context('fork').Process(target=child)
    process.start();process.join(30)
    if process.is_alive():
        process.kill();process.join();pytest.fail('crash boundary not reached')
    assert process.exitcode==73
    path=runtime._data_root/'attempts'/runtime.plan.run_id/(runtime.plan.attempt_id+'.json')
    AttemptHandle(path=path,run_id=runtime.plan.run_id,attempt_id=runtime.plan.attempt_id).mark_aborted(
        'child exit73 authoritatively observed')
    target=replace(runtime.plan,attempt_id='grounded-recovery-v1')
    replay=NativeContactReplay(runtime._data_root,runtime.plan,target,'prepared-context-text')
    runtime._plan=target;runtime._attempt_id=target.attempt_id
    runtime._explicit_run_dir=runtime._data_root/'runs'/target.run_id/target.attempt_id
    runtime._native_contact_replay=replay
    runtime.receipts=FlightReceipts(target.run_id,target.attempt_id)
    for stage in ('preflight','preprocess','serve'):runtime.receipts.record(stage,{})
    runtime.frozen_core_client_kwargs=lambda:{**kwargs,'plan':target,'run_dir':runtime._run_dir()}
    if unknown:
        with pytest.raises(ValueError,match='outcome unknown'):runtime.run_phases()
        assert getattr(runtime, '_tfgrpo_caller', None) is None
    else:
        result=runtime.run_phases()
        assert result['contact_count']==1
        assert runtime._client._adapter.cost_summary()['replayed_predecessor_cost']['calls']==1
        assert runtime._client._adapter.cost_summary()['calls_used']==0
        runtime._client.close()
    assert not transport.calls


@pytest.mark.parametrize('oversize',[False,True])
def test_grounded_uses_real_durable_fit_on_wrapped_payload(tmp_path,monkeypatch,oversize):
    from types import SimpleNamespace
    import importlib
    from speech_aware_evidence_acquisition.core import tfgrpo_fit
    runtime,transport,_,_=setup(tmp_path,monkeypatch)
    original_fit=importlib.reload(tfgrpo_fit).build_tfgrpo_fit_gate
    runtime._composed.values.update(transport='vllm',tfgrpo_contact_fit={
        'run_id':'FIT-GROUNDED','attempt_id':'fit-v1','contact_cap':1})
    runtime._session=SimpleNamespace(host='127.0.0.1',port=8200)
    runtime._gate=object()
    runtime._adapter=SimpleNamespace(serving_geometry=lambda:{'slot_tokens':500})
    runtime._transport=SimpleNamespace(render_request=lambda raw:{'model':'frozen-model',
        'messages':[{'role':'user','content':json.loads(raw)['history'][0]['content']}],
        'max_tokens':100})
    runtime._gate_touch=lambda *a:None
    runtime._require_exposure_row_unconsumed=lambda *a:None
    fits=[]
    class Tokenizer:
        def __init__(self,**kwargs):pass
        def request_json(self,method,route,payload):
            fits.append(payload)
            n=450 if oversize else 210
            return {'count':n,'tokens':list(range(n))}
    monkeypatch.setattr(tfgrpo_fit,'build_tfgrpo_fit_gate',lambda r,**kw:
        original_fit(r,client_factory=Tokenizer,**kw))
    if oversize:
        with pytest.raises(ValueError):runtime.run_phases()
        assert not transport.calls
        assert not list((runtime._run_dir()/'tfgrpo/learner/contacts').glob('*.intent.json'))
    else:
        runtime.run_phases()
        assert len(transport.calls)==1
        receipt=next((tmp_path/'prepared/FIT-GROUNDED/fit-v1').glob('*.request-fit.json'))
        assert json.loads(receipt.read_bytes())['output']['rows'][0]['measured_prompt_tokens']==210
    assert len(fits)==1 and 'source_messages' in fits[0]['messages'][0]['content']
    runtime._client.close()


def test_three_shards_global_predecessor_barrier_then_resolved_resume(tmp_path, monkeypatch):
    """Real replay inventory + shard executor; transport/FIT are counted doubles."""
    from types import SimpleNamespace
    from test_grounded_synthesis_execution import fixture
    from test_frozen_core_adapter import _plan
    from test_contact_journal import payload
    from speech_aware_evidence_acquisition.core.attempts import AttemptStore
    from speech_aware_evidence_acquisition.core.contact_journal import (
        ContactJournal, NativeContactReplay, digest,
    )
    f = fixture(tmp_path/'fixture', groups=6, concurrency=2)
    _, plan, bb, ob, args, custody, *_ = f
    assert len(plan['shards']) == 3
    source = _plan(planned_model_calls=3, planned_speech_audio_seconds=300)
    target = replace(source, attempt_id='recovery-v1')
    handle = AttemptStore(tmp_path/'attempts').open_attempt(source.run_id, source.attempt_id,
        carrier_lock_key=source.carrier_lock_key, reserved_calls=3, reserved_audio_seconds=300)
    journal = ContactJournal(tmp_path, source, 'bare-core')
    payloads = {}
    for shard in plan['shards']:
        identity = m.contact_identity(plan, shard)
        p = payload('req-'+str(shard['shard_index']))
        p['history'] = [{'role':'control', 'contact_id':identity}]
        payloads[identity] = p
    # Source order: observed, never-sent, unknown LAST. Concurrent recovery must
    # not dispatch the middle shard before discovering the last unknown.
    first, middle, last = list(payloads)
    for identity in (first, last):
        journal.intent(payloads[identity]); handle.record_usage(calls=1, audio_seconds=100)
    journal.response(payloads[first]['request_id'], digest(canonical(payloads[first])), b'known', 1)
    handle.mark_aborted('source process terminal verified')
    replay = NativeContactReplay(tmp_path, source, target, 'bare-core')
    # Runtime asserts its arm separately; retain real global custody audit while
    # isolating this test from the prepared-text fixture's unrelated validation.
    monkeypatch.setattr(replay, 'assert_target', lambda *a: None)
    binding = {'registration':args['registration'], 'plan':plan}
    loaded = {'binding':binding, 'batch_bytes':bb, 'operation_bytes':ob,
        'source':{'source':{}, 'max_operations':128, 'concurrency':2},
        'count_tokens':args['count_tokens'], 'tokenizer_identity':args['tokenizer_identity']}
    monkeypatch.setattr(m, 'load_runtime_source', lambda *a, **k: (loaded, {}))
    monkeypatch.setattr(m, 'load_source', lambda *a, **k: copy.deepcopy(custody))
    registration = {'construction_roster_sha256':'c'*64, 'learning_config_sha256':'a'*64,
                    'contacts':[{}, {}, {}]}
    runtime = SimpleNamespace(_composed=SimpleNamespace(values={'tfgrpo_learning_registration':registration}),
        plan=target, _repo_root=tmp_path, _data_root=tmp_path, _samples=(),
        _native_contact_replay=replay, _run_dir=lambda:tmp_path/'target')
    fits, sends, reused, binds = [], [], [], []
    class Caller:
        model_id = 'model'
        def preflight(self, request):
            fits.append(request['identity'])
        def reconcile(self, request):
            return None
        def __call__(self, request):
            identity = request['identity']
            if runtime._native_contact_replay.match(payloads[identity]) is None:
                sends.append(identity)
            else:
                reused.append(identity)
            return {'model_id':'model', 'receipt_id':identity,
                'text':'[{"operation":"ADD","id":null,"content":"conditional rule"}]',
                'usage':{'calls':1,'prompt_tokens':30,'completion_tokens':20}}
    def bind(*a, **k):
        binds.append(True)
        caller = Caller(); caller.client = object()
        return caller
    monkeypatch.setattr(m, 'bind_runtime', bind)
    with pytest.raises(ValueError, match='outcome unknown'):
        m.run_registered_shards(runtime)
    assert not fits and not sends and not reused and not binds
    assert not (tmp_path/'target').exists()
    # Authoritative response reconciliation requires a newly verified replay;
    # the old pinned instance must reject the changed custody.
    journal.response(payloads[last]['request_id'], digest(canonical(payloads[last])), b'resolved', 1)
    with pytest.raises(ValueError, match='custody changed'):
        replay.assert_all_contacts_observed()
    replay = NativeContactReplay(tmp_path, source, target, 'bare-core')
    monkeypatch.setattr(replay, 'assert_target', lambda *a: None)
    runtime._native_contact_replay = replay
    result = m.run_registered_shards(runtime)
    assert sends == [middle] and sorted(reused) == sorted([first, last])
    assert len(fits) == 3 and len(binds) == 1
    assert result['historical_usage']['calls'] == 309
    assert result['current_accounting']['known_usage']['calls'] == 3
    assert handle.actual_totals()[0] == 2
