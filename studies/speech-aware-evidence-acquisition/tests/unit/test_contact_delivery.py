"""Failure-boundary custody: pre-POST, post-admission and crash recovery."""
import json
from dataclasses import replace

import pytest

from speech_aware_evidence_acquisition.core.contact_journal import (
    ContactJournal, ContactJournalError, NativeContactReplay, canonical, digest,
)
from speech_aware_evidence_acquisition.core.model.transport import LlamaServerTransport
from test_contact_journal import payload
from test_frozen_core_adapter import _adapter, _plan


def transport(render, post):
    # Exercise the production __call__ boundary, with deterministic I/O hooks.
    value = object.__new__(LlamaServerTransport)
    value.render_request_with_receipt = render
    value._post = post
    value._url = 'http://127.0.0.1:1/v1/chat/completions'
    value.assert_session_alive = lambda: None
    value._preserve_wire_responses = False
    return value


def fail(*args):
    raise OSError('injected failure')


@pytest.mark.parametrize('phase,state', [('render', 'not_sent'), ('encode', 'not_sent'),
                                        ('post', 'unknown'), ('unknown_transport', 'unknown')])
def test_adapter_persists_exact_boundary_and_keeps_charge(tmp_path, phase, state):
    adapter, sink = _adapter(tmp_path)
    journal = ContactJournal(tmp_path, _plan(), 'bare-core')
    adapter._contact_journal = journal
    posted = []
    def post(*args):
        posted.append(args)
        return fail()
    render = fail if phase == 'render' else lambda _: ({'x': object()} if phase == 'encode' else {}, None)
    adapter._transport = fail if phase == 'unknown_transport' else transport(render, post)
    with pytest.raises((OSError, TypeError)):
        adapter.request(payload())
    assert journal.inspect('req-1')['state'] == state
    assert len(posted) == (1 if phase == 'post' else 0)
    assert adapter._attempt_handle.actual_totals() == (1, 100.0)
    assert not journal._path('req-1', 'response').exists()


def journal_with_terminal_attempt(tmp_path):
    source = _plan()
    journal = ContactJournal(tmp_path, source, 'bare-core')
    for rid, kind in [('req-1', 'observed'), ('req-2', 'not_sent')]:
        p = payload(rid)
        p['history'][0]['contact_id'] = rid
        journal.intent(p)
        if kind == 'observed':
            journal.delivery_event(rid, digest(canonical(p)), 'dispatch')
            journal.response(rid, digest(canonical(p)), b'{}', .1)
        else:
            journal.delivery_event(rid, digest(canonical(p)), 'not_sent', error_type='OSError')
    path = tmp_path/'attempts'/source.run_id/(source.attempt_id+'.json')
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical({'state': 'ABORTED', 'run_id': source.run_id,
        'attempt_id': source.attempt_id, 'carrier_lock_key': source.carrier_lock_key,
        'actual': {'calls': 2, 'audio_seconds': 200.0}}))
    return journal, source


def test_recovery_preserves_spend_and_requires_fresh_identical_unsent_request(tmp_path):
    journal, source = journal_with_terminal_attempt(tmp_path)
    replay = NativeContactReplay(tmp_path, source, replace(source, attempt_id='recovery-v1'), 'bare-core')
    replay.assert_all_contacts_resolved()
    with pytest.raises(ContactJournalError, match='observed-only'):
        replay.assert_all_contacts_observed()
    assert replay.source_binding()['actual_usage'] == {'calls': 2, 'audio_seconds': 200.0}
    assert len(replay.source_binding()['delivery_custody']) == 2
    p = journal.inspect('req-2')['intent']['payload']
    with pytest.raises(ContactJournalError, match='fresh'):
        replay.match(p)
    p['request_id'] = 'req-3'
    assert replay.match(p) is None and replay.consume(p) is None
    p['task_instruction'] = 'changed'
    with pytest.raises(ContactJournalError, match='changed'):
        replay.match(p)
    known = journal.inspect('req-1')['intent']['payload']
    assert replay.consume(replay.match(known))['response_bytes'] == b'{}'


@pytest.mark.parametrize('fault', ['delete', 'tamper', 'contradiction', 'orphan'])
def test_delivery_mutation_cannot_authorize_recovery(tmp_path, fault):
    journal, source = journal_with_terminal_attempt(tmp_path)
    replay = NativeContactReplay(tmp_path, source, replace(source, attempt_id='recovery-v1'), 'bare-core')
    path = journal._path('req-2', 'not_sent')
    if fault == 'delete': path.unlink()
    elif fault == 'tamper':
        d = json.loads(path.read_bytes()); d['request_sha256'] = '0'*64; path.write_bytes(canonical(d))
    elif fault == 'orphan':
        journal._path('other', 'not_sent').write_bytes(path.read_bytes())
    else:
        journal._path('req-2', 'dispatch').write_bytes(path.read_bytes())
    with pytest.raises(ContactJournalError): replay.assert_all_contacts_resolved()


def test_journal_refuses_response_or_dispatch_after_not_sent(tmp_path):
    journal, _ = journal_with_terminal_attempt(tmp_path)
    sha = journal.inspect('req-2')['intent']['request_sha256']
    with pytest.raises(ContactJournalError): journal.response('req-2', sha, b'{}', 1)
    with pytest.raises(ContactJournalError): journal.delivery_event('req-2', sha, 'dispatch')


def test_failed_boundary_persistence_never_sends(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path)
    journal = ContactJournal(tmp_path, _plan(), 'bare-core')
    adapter._contact_journal = journal
    posted = []
    adapter._transport = transport(lambda _: ({}, None), lambda *args: posted.append(args))
    monkeypatch.setattr(journal, 'delivery_event', fail)
    with pytest.raises(OSError): adapter.request(payload())
    assert posted == [] and journal.inspect('req-1')['state'] == 'unknown'


def test_new_adapter_reuses_observed_and_sends_not_sent_once_with_all_costs(tmp_path):
    from test_frozen_core_adapter import _binding, _OpenGate
    from speech_aware_evidence_acquisition.core.model import FrozenCoreAdapter
    from speech_aware_evidence_acquisition.core.tracing import TraceSink
    binding = _binding(tmp_path)
    plan = _plan(planned_model_calls=4, planned_speech_audio_seconds=500)
    sent = []
    fail_render = [False]
    def render(_):
        if fail_render[0]: raise OSError('identity probe closed before POST')
        return {}, None
    def post(*args):
        sent.append(args)
        return b'{"choices":[{"message":{"content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1}}'
    live = transport(render, post)
    source = FrozenCoreAdapter(gate=_OpenGate(), plan=plan, arm='bare-core',
        transport=live, sink=TraceSink(tmp_path/'source-trace', plan.run_id),
        binding=binding, contact_journal=True)
    first = payload('req-1'); first['history'][0]['contact_id']='first'
    second = payload('req-2'); second['history'][0]['contact_id']='second'
    source.request(first)
    fail_render[0] = True
    with pytest.raises(OSError): source.request(second)
    source._attempt_handle.mark_aborted('test producer stopped')
    target = replace(plan, attempt_id='recovery-v1')
    replay = NativeContactReplay(binding.data_root, plan, target, 'bare-core')
    replay.assert_all_contacts_resolved()
    new = FrozenCoreAdapter(gate=_OpenGate(), plan=target, arm='bare-core',
        transport=live, sink=TraceSink(tmp_path/'target-trace', plan.run_id),
        binding=binding, contact_journal=True, contact_replay=replay)
    fail_render[0] = False
    new.request(first)
    second['request_id']='req-3'
    new.request(second)
    assert len(sent)==2  # one before interruption, one fresh request after recovery
    assert source._attempt_handle.actual_totals()==(2,200.0)
    assert new._attempt_handle.actual_totals()==(1,100.0)
    assert new.cost_summary()['replayed_predecessor_cost']['calls']==1
    assert replay.source_binding()['actual_usage']['calls']==2


def test_concurrent_delivery_scopes_do_not_cross_classify(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from speech_aware_evidence_acquisition.core.contact_delivery import delivery_scope
    journal=ContactJournal(tmp_path,_plan(),'bare-core')
    barrier=Barrier(2)
    def run(rid,should_post):
        p=payload(rid);raw=canonical(p);journal.intent(p)
        def render(_):
            barrier.wait(timeout=5)
            if not should_post: raise OSError('render failure')
            return {},None
        with pytest.raises(OSError):
            with delivery_scope(journal,rid,digest(raw)):
                transport(render,fail)(raw)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,'req-1',False),pool.submit(run,'req-2',True)]
        for future in futures:future.result(timeout=10)
    assert journal.inspect('req-1')['state']=='not_sent'
    assert journal.inspect('req-2')['state']=='unknown'


def _crash_at_delivery(root,phase):
    import os
    from speech_aware_evidence_acquisition.core.contact_delivery import delivery_scope
    journal=ContactJournal(root,_plan(),'bare-core');p=payload();raw=canonical(p);journal.intent(p)
    def render(_):
        if phase=='render':os._exit(73)
        return {},None
    def post(*args):os._exit(73)
    with delivery_scope(journal,'req-1',digest(raw)):transport(render,post)(raw)


@pytest.mark.parametrize('phase',['render','post'])
def test_crash_without_proven_failure_stays_unknown(tmp_path,phase):
    import multiprocessing
    process=multiprocessing.get_context('fork').Process(target=_crash_at_delivery,args=(tmp_path,phase))
    process.start();process.join(10)
    if process.is_alive():process.kill();process.join();pytest.fail('child did not reach crash boundary')
    assert process.exitcode==73
    journal=ContactJournal(tmp_path,_plan(),'bare-core')
    assert journal.inspect('req-1')['state']=='unknown'
    assert journal._path('req-1','dispatch').exists()==(phase=='post')
