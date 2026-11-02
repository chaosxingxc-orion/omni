"""Real Direct whole-wave runner, shared FIT, two real vLLM renderers/router."""
from dataclasses import replace
import json
import threading

import pytest

from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
from speech_aware_evidence_acquisition.core.model import VllmServerTransport
from speech_aware_evidence_acquisition.core.model.vllm_router_transport import VllmRouterTransport
from speech_aware_evidence_acquisition.core.preparation import digest
from speech_aware_evidence_acquisition.core.preparation_fit import build_live_phase_fit_gate
from speech_aware_evidence_acquisition.core.session.worker import LeaseCheckedTransport
from speech_aware_evidence_acquisition.core.tracing import TraceBuffer
from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
from speech_aware_evidence_acquisition.rails.material import PhaseMaterial
from speech_aware_evidence_acquisition.rails.phase_runner import run_continuous_program
from speech_aware_evidence_acquisition.reproduction.ojw import openjiuwen_available
from test_direct_only_fit import _Runtime, _Tokenizer, FIT_RUN, FIT_ATTEMPT
from test_direct_native_recovery import values
from test_rails_frozen_core_client import _World, _client, _plan
from test_rails_runner import _FakeAdapter, _knowledge_snapshot
from test_vllm_router_transport import world, _Recorder, SERVED, PID_BASE  # noqa: F401


@pytest.mark.skipif(not openjiuwen_available(), reason='OpenJiuwen unavailable')
def test_direct_whole_wave_fits_real_router_then_dispatches_and_recovers(world, tmp_path):
    media = _World(tmp_path / 'source', data_root=world.root)
    receipt = world.receipt_with_instances(2)
    transports, sessions, recorders = [], [], []
    concurrent = threading.Barrier(2)
    for index in range(2):
        iid = f'tp2-{index}'; pid = PID_BASE + index
        world.spawn(iid, pid=pid, receipt=receipt)
        session = world.session(iid, pid, receipt=receipt)
        recorder = _Recorder('direct answer')
        def post(url, raw, recorder=recorder):
            concurrent.wait(timeout=10)
            return recorder(url, raw)
        transports.append(VllmServerTransport(
            f'http://{session.host}:{session.port}', world.root, session, receipt,
            world.root / 'runtime', post=post, served_model_name=SERVED,
            prompt_order='native-raw-speech-v1'))
        sessions.append(session); recorders.append(recorder)
    dispatched = []
    class ObservedRouter(VllmRouterTransport):
        def __call__(self, raw):
            dispatched.append((self._route(raw), json.loads(raw)))
            return super().__call__(raw)
    router = ObservedRouter(transports)
    wrapped = LeaseCheckedTransport(router, lambda: None)
    plan = _plan(planned_model_calls=4)
    program = default_phase_program('phase-direct-only-v1')

    def make_fit(target, replay=None):
        runtime = _Runtime(world.root, cap=2)
        runtime._composed.values.update({k: v for k, v in values().items() if k != 'live_phase_fit'})
        runtime._samples = tuple(media.samples)
        runtime._transport = wrapped
        runtime._session = sessions[0]
        runtime.session_required_identity = lambda: {'sessions': [s.session_id for s in sessions]}
        if replay is not None: runtime._native_contact_replay = replay
        tokenizer = _Tokenizer()
        gate = build_live_phase_fit_gate(runtime, program, target, client_factory=lambda **kw: tokenizer)
        captured = []
        def fit(phase, batch):
            captured.extend(request.as_payload() for request in batch)
            assert all('id_slot' not in p['decoding_params'] for p in captured)
            gate(phase, batch)
        return gate, fit, tokenizer, captured

    def run(client, target, fit):
        return run_continuous_program(program=program, adapter=_FakeAdapter(tuple(media.samples)),
            plan=target, data_root=world.root, core_client=client, split_id='all',
            knowledge_snapshot=_knowledge_snapshot(), speech_packet_bindings={
                s.sample_id: {'packet_id': s.audio_ref.path, 'sha256': s.audio_ref.sha256}
                for s in media.samples}, query_source='speech_audio',
            trace_sink=TraceBuffer(target.run_id),
            decoding_params={'temperature': 0, 'max_tokens': 64},
            material_for_phase=lambda phase, context, sealed: PhaseMaterial(
                instruction='Answer this speech.', response_format=None,
                contact_id='direct-' + context.sample_id, purpose='direct'),
            slot_count=2, queue_depth=2, whole_roster_wave_barrier=True, request_batch_gate=fit)

    old, _, _ = _client(media, tmp_path / 'source', plan=plan, arm='raw-speech-agent-native-tools',
        tool_schemas=(), transport=wrapped, contact_journal=True)
    gate, fit, tokenizer, payloads = make_fit(plan)
    try: result = run(old, plan, fit)
    finally: old.close()
    assert len(result.results) == 2 and len(tokenizer.calls) == 2
    assert all(p['decoding_params']['tool_choice'] == 'none' for p in payloads)
    assert len(dispatched) == 2 and {i for i, payload in dispatched} == {0, 1}
    assert all(i == p['decoding_params']['id_slot'] % 2 for i, p in dispatched)
    artifact = world.root / 'prepared' / FIT_RUN / FIT_ATTEMPT / 'direct.request-fit.json'
    measured = json.loads(artifact.read_bytes())['output']['rows']
    posted = [body for recorder in recorders for url, body in recorder.calls]
    assert sorted(row['request_body_sha256'] for row in measured) == sorted(digest(body) for body in posted)
    assert all('id_slot' not in body for body in posted)
    old._adapter._attempt_handle.mark_aborted('stopped after durable Direct responses')
    target = replace(plan, attempt_id='direct-router-recovery-v1')
    replay = NativeContactReplay(world.root, plan, target, 'raw-speech-agent-native-tools')
    successor = _World(tmp_path / 'successor', data_root=world.root)
    new, _, _ = _client(successor, tmp_path / 'successor', plan=target,
        arm='raw-speech-agent-native-tools', tool_schemas=(), transport=wrapped,
        contact_journal=True, contact_replay=replay)
    reused, fit, tokenizer, _ = make_fit(target, replay)
    try: result = run(new, target, fit)
    finally: cost = new.close()['cost']
    assert len(result.results) == 2 and len(dispatched) == 2
    assert not tokenizer.calls and cost['calls_used'] == 0
    assert cost['replayed_predecessor_cost']['calls'] == 2
    assert reused.receipts[0]['direct_fit_reuse']['new_tokenizer_contacts'] == 0
