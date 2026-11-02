"""Shared FlightRuntime learning dispatch/seal with real adapter, fake transport."""
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest

from speech_aware_evidence_acquisition.core.flight.runtime import FlightRuntime
from speech_aware_evidence_acquisition.core.flight.receipts import FlightReceipts
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha, LearningError
from speech_aware_evidence_acquisition.rails import tfgrpo_runtime
from test_tfgrpo_source import setup as source_setup
from test_rails_frozen_core_client import _World, _binding, _OpenGate, _CapturingTransport


def make_runtime(tmp_path, monkeypatch):
    values, args, source = source_setup(tmp_path)
    sid = args['sample_ids'][0]
    class World(_World):
        SAMPLE_SPECS = (('sample1', 1.0),)
    world = World(tmp_path)
    world.samples = [replace(s, sample_id=sid) for s in world.samples]
    values.update(arm='prepared-context-text', result_kind='artifact', scorers=[],
                  policy='phase-direct-only-v1')
    def answer(payload):
        stage = json.loads(payload['history'][0]['content'])['stage']
        if stage == 'summary': return 'Action polarity was correct or incorrect.'
        if stage == 'advantage': return '<Experiences>Check polarity.</Experiences>'
        return '[{"operation":"ADD","id":null,"content":"Check polarity."}]'
    transport = _CapturingTransport(answer)
    runtime = object.__new__(FlightRuntime)
    runtime._composed = SimpleNamespace(values=values)
    runtime._plan = args['plan']
    runtime._run_id, runtime._attempt_id = args['plan'].run_id, args['plan'].attempt_id
    runtime._samples = world.samples
    runtime._repo_root = runtime._data_root = tmp_path
    runtime._model, runtime._dataset, runtime._baseline = 'frozen-model', 'dataset', 'baseline'
    runtime._explicit_run_dir = world.run_dir
    runtime._clock = time.monotonic
    runtime._seal_only = False
    runtime._resume = None
    runtime._phase_summaries = ()
    runtime._artifact_result_binding = {'role':'training-feedback-proposal'}
    runtime._artifact_content_validator = None
    runtime._execution_policy = None
    runtime._obs_result = None
    runtime._session_receipt_path = tmp_path/'session.json'
    runtime._serving_session_dir = None
    runtime._session_attach_receipt_path = None
    runtime._split_id = 'train'
    runtime._lock = object()
    runtime.receipts = FlightReceipts(runtime._run_id, runtime._attempt_id)
    for stage in ('preflight','preprocess','serve'): runtime.receipts.record(stage, {})
    runtime.assert_generation_lease = lambda: None
    runtime._supersede_resume_predecessors = lambda: None
    runtime._assert_phase_supply_is_legal_for = lambda p: None
    runtime._assert_phase_review_is_legal_for = lambda p: None
    runtime._phase_planner_specs = lambda p: None
    runtime._phase_tool_specs = lambda p: None
    runtime._prepare_native_phase_runtime = lambda p: pytest.fail('dynamic source reached ordinary phase dispatcher')
    runtime.execution_unit_roster = lambda: None
    runtime._close_live_speech_provider = lambda **k: None
    runtime.serving_monitor_summary = lambda: None
    runtime._default_ledger_cells = lambda p: {c:'test' for c in FlightRuntime.CALLER_LEDGER_COLUMNS}
    runtime.session_required_identity = lambda: {'model_fragment':'frozen-model'}
    runtime.frozen_core_client_kwargs = lambda: {
        'plan':args['plan'], 'gate':_OpenGate(), 'transport':transport,
        'run_dir':world.run_dir, 'binding':replace(_binding(world),carrier_lock_key='slurp'), 'arm':'prepared-context-text'}
    def fit(p):
        return {'artifact_sha256':'d'*64,'every_request_fits':True,'payload_sha256':sha(p.as_payload())}
    from speech_aware_evidence_acquisition.core import tfgrpo_fit
    monkeypatch.setattr(tfgrpo_fit,'build_tfgrpo_fit_gate',lambda *a,**k:fit)
    captured = {}
    def manifest(**kwargs):
        captured.update(kwargs)
        path=world.run_dir/'manifest.json';path.write_text('{}');return path
    runtime._manifest_builder=manifest
    def verify(path):
        from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs, _verify_outputs_trace_binding
        rows=_verify_outputs(captured['result'].outputs_path.read_bytes(),frozenset((sid,)),
                             'slurp','prepared-context-text')
        records={r['record_id']:r['payload'] for r in map(json.loads,
            (world.run_dir/(runtime._run_id+'.trace.jsonl')).read_text().splitlines())}
        _verify_outputs_trace_binding(rows,records)
        captured['verified_rows']=rows
        return SimpleNamespace(attempt_state='FINALIZED' if captured.get('finalized') else 'OPENED',manifest={'scores':None,'scorers':[]})
    runtime._bundle_verifier=verify
    def finalize(*a,**k):
        assert values['scorers']==[]
        captured['finalized']=True
        return SimpleNamespace(scores_path=None,ledger_markdown_path=tmp_path/'ledger.md',attempt_state='FINALIZED')
    runtime._bundle_finalizer=finalize
    return runtime, transport, captured


def test_shared_dispatch_and_scoreless_seal_preserve_all_cost_and_trace(tmp_path,monkeypatch):
    runtime, transport, captured=make_runtime(tmp_path,monkeypatch)
    receipt=runtime.run_phases()
    assert receipt['dispatch']=='registered-tfgrpo-learning'
    assert receipt['contact_count']==5 and receipt['outputs_row_count']==1
    assert len(transport.calls)==5 and not runtime._client._closed
    result=runtime.seal()
    assert runtime._client._closed and captured['finalized']
    assert result['cost']['calls_used']==5
    assert result['cost']['prompt_tokens_total']==15
    assert result['cost']['completion_tokens_total']==10
    assert result['cost']['audio_seconds_used']==0.0  # only text is transmitted
    assert result['finalize']['result_kind']=='artifact'
    assert result['finalize']['scores_path'] is None
    assert len(captured['verified_rows'])==1
    phases=json.loads(Path(result['phases_path']).read_bytes())
    assert sum(p['contacts'] for p in phases['phases'])==5
    pin=phases['phases'][-1]['candidate']
    assert hashlib.sha256((tmp_path/pin['relpath']).read_bytes()).hexdigest()==pin['sha256']
    assert len(runtime._client.timings)==5
    assert runtime.score()['job_count']==0
    assert receipt['learning']['usage']=={'calls':5,'prompt_tokens':15,'completion_tokens':10}


def test_fresh_client_restart_refused_without_duplicate_contact(tmp_path,monkeypatch):
    runtime,transport,_=make_runtime(tmp_path,monkeypatch)
    runtime.run_phases()
    with pytest.raises(LearningError,match='registered recovery'):
        tfgrpo_runtime.run_registered_learning(runtime,
            receipt_root=runtime._run_dir()/'tfgrpo/contacts',learner_root=runtime._run_dir()/'tfgrpo/learner')
    assert len(transport.calls)==5
    runtime._client.close()


def test_source_drift_refused_before_shared_client_open(tmp_path,monkeypatch):
    runtime,transport,_=make_runtime(tmp_path,monkeypatch)
    (tmp_path/'source.json').write_text('{}')
    with pytest.raises(LearningError,match='source (pin )?changed'): runtime.run_phases()
    assert not transport.calls


def test_observed_handoff_recovery_reuses_real_shared_adapter_receipt(tmp_path,monkeypatch):
    runtime,transport,_=make_runtime(tmp_path,monkeypatch)
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import ExperienceLearner
    source=json.loads((tmp_path/'source.json').read_bytes());batch=source['batch']
    root=runtime._run_dir()/'tfgrpo'
    caller=tfgrpo_runtime.bind_runtime(runtime,receipt_root=root/'contacts')
    failed=[False]
    def crash(request):
        response=caller(request)
        if request['stage']=='batch_update' and not failed[0]:
            failed[0]=True
            raise RuntimeError('lost bridge-to-learner handoff')
        return response
    def learner(callback):
        return ExperienceLearner(root/'learner',model_id=batch['model_id'],metered_call=callback,
            learning_config_sha256=batch['learning_config_sha256'],preflight=caller.preflight,
            reconcile=caller.reconcile,**batch['bounds'])
    def run(instance):
        return instance.learn(batch['groups'],batch['parent'],policy_sha256=batch['policy_sha256'],
                              rollout_kind=batch['rollout_kind'],**batch['common'])
    with pytest.raises(RuntimeError,match='handoff'):run(learner(crash))
    assert len(transport.calls)==5
    candidate=run(learner(caller))
    assert len(transport.calls)==5 and candidate['learning_usage']['calls']==5
    assert len(caller.client.timings)==5
    close=caller.client.close()
    assert close['cost']['calls_used']==5


def test_shared_dispatch_recovers_observed_handoff_with_same_live_client(tmp_path,monkeypatch):
    runtime,transport,_=make_runtime(tmp_path,monkeypatch)
    original=tfgrpo_runtime.MeteredExperienceCaller.__call__
    failed=[False]
    def crash(self,request):
        response=original(self,request)
        if request['stage']=='batch_update' and not failed[0]:
            failed[0]=True
            raise RuntimeError('lost handoff')
        return response
    monkeypatch.setattr(tfgrpo_runtime.MeteredExperienceCaller,'__call__',crash)
    with pytest.raises(RuntimeError,match='handoff'):runtime.run_phases()
    assert len(transport.calls)==5 and not runtime.receipts.has('run_phases')
    receipt=runtime.run_phases()
    assert len(transport.calls)==5 and receipt['contact_count']==5
    sealed=runtime.seal()
    assert sealed['cost']['calls_used']==5
