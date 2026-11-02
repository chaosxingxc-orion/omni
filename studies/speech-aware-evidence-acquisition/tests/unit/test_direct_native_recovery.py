"""Exact Direct capture, completed-wave FIT reuse, and unknown no-resend."""
import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.core.contact_journal import (
    ContactJournal, ContactJournalError, NativeContactReplay, canonical, digest)
from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
from speech_aware_evidence_acquisition.core.preparation import PreparationError
from test_direct_only_fit import _Runtime, _plan, _build, _batch, FIT_RUN, FIT_ATTEMPT

ROOT = Path(__file__).resolve().parents[2]
ARM = 'raw-speech-agent-native-tools'


def values():
    base = json.loads((ROOT / 'configs/baseline/slurp-direct-only-fold-b-baseline-raw-port-v1.json').read_bytes())
    return {**base, 'transport': 'vllm', 'carrier_lock_key': 'slurp'}


def test_only_exact_direct_graph_enables_custody():
    base = values(); before = copy.deepcopy(base)
    assert native_contact_custody_enabled(base) and base == before
    for key, value in [('policy', 'phase-direct-unknown'), ('arm', 'bare-core'),
        ('transport', 'llama.cpp'), ('carrier_lock_key', 'earnings21'),
        ('native_episode', {}), ('phase_tools', {'direct': 'tool'}),
        ('knowledge_stores', {'test': {}}), ('max_revisions', 1),
        ('whole_roster_wave_barrier', False), ('evidence_source', 'knowledge')]:
        changed = {**base, key: value}
        assert not native_contact_custody_enabled(changed), key


def replay(tmp_path, *, unknown=False, known=False):
    source = _plan(); target = replace(source, attempt_id='direct-recovery-v1')
    journal = ContactJournal(tmp_path, source, ARM)
    payload = _batch()[0].as_payload()
    payload['history'] = [{'role': 'control', 'contact_id': 'c-direct-s1'}]
    calls = int(unknown or known)
    if calls:
        journal.intent(payload)
        if known:
            journal.response(payload['request_id'], digest(canonical(payload)),
                b'{"text":"original answer","usage":{}}', 1)
    path = tmp_path / 'attempts' / source.run_id / (source.attempt_id + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical({'state': 'ABORTED', 'run_id': source.run_id,
        'attempt_id': source.attempt_id, 'carrier_lock_key': source.carrier_lock_key,
        'actual': {'calls': calls, 'audio_seconds': calls * payload['audio_seconds']}}))
    return NativeContactReplay(tmp_path, source, target, ARM), payload, target


def runtime(tmp_path):
    rt = _Runtime(tmp_path)
    rt._composed.values.update({k: v for k, v in values().items() if k != 'live_phase_fit'})
    return rt


def test_direct_known_response_reused_and_unknown_blocks_fresh_work(tmp_path):
    known, payload, _ = replay(tmp_path / 'known', known=True)
    known.assert_all_contacts_observed()
    changed_id = {**payload, 'request_id': 'new-route-id'}
    original = known.match(changed_id)
    assert known.consume(original)['response_bytes'] == b'{"text":"original answer","usage":{}}'
    unknown, _, _ = replay(tmp_path / 'unknown', unknown=True)
    with pytest.raises(ContactJournalError, match='unknown'):
        unknown.assert_all_contacts_observed()
    with pytest.raises(ContactJournalError):
        unknown.match(payload)


@pytest.mark.parametrize('unknown', [False, True])
@pytest.mark.parametrize('mode', ['complete', 'complete_with_metadata', 'prepared', 'prepared_with_metadata'])
def test_real_tool_free_direct_client_capture_and_replay_no_transport(tmp_path, unknown, mode):
    from test_rails_frozen_core_client import _World, _client, _zero_supply_messages, _plan as client_plan
    world = _World(tmp_path / 'source')
    plan = client_plan()
    calls = []
    def source_transport(raw):
        calls.append(raw)
        if unknown:
            raise TimeoutError('unknown server response')
        return b'{"text":"frozen direct answer","usage":{"prompt_tokens":3,"completion_tokens":2}}'
    old, _, _ = _client(world, tmp_path / 'source', plan=plan, arm=ARM,
        tool_schemas=(), transport=source_transport, contact_journal=True)
    arguments = dict(raw_agent_sample_id='s1', agent_state=(),
        agent_contact_id='c-direct-s1', contact_instruction='Answer the speech.',
        tool_names=(), decoding_params={'temperature': 0, 'max_tokens': 64,
            'tool_choice': 'none', 'parallel_tool_calls': False})
    messages = _zero_supply_messages(world.media_relpath('s1'))
    def complete(client):
        if mode.startswith('prepared'):
            prepared = client.prepare_request(messages, **arguments)
            method = (client.complete_prepared_with_metadata if mode.endswith('with_metadata')
                else client.complete_prepared)
            reply = method(prepared, id_slot=0)
        else:
            reply = getattr(client, mode)(messages, **arguments)
        return reply.text if mode.endswith('with_metadata') else reply
    try:
        if unknown:
            with pytest.raises(Exception): complete(old)
        else:
            assert complete(old) == 'frozen direct answer'
    finally:
        old.close()
    old._adapter._attempt_handle.mark_aborted('source test process terminal')
    target = replace(plan, attempt_id='direct-client-recovery')
    recovery = NativeContactReplay(world.data_root, plan, target, ARM)
    successor = _World(tmp_path / 'successor', data_root=world.data_root)
    new_calls = []
    new, _, _ = _client(successor, tmp_path / 'successor', plan=target, arm=ARM,
        tool_schemas=(), contact_journal=True, contact_replay=recovery,
        transport=lambda raw: new_calls.append(raw) or b'{"text":"must never dispatch"}')
    try:
        if unknown:
            with pytest.raises(Exception): complete(new)
        else:
            assert complete(new) == 'frozen direct answer'
    finally:
        result = new.close()
    assert len(calls) == 1 and not new_calls
    assert result['cost']['calls_used'] == 0
    if not unknown:
        assert result['cost']['replayed_predecessor_cost']['calls'] == 1
        original = json.loads(calls[0])
        assert new._timings[0].request_id == original['request_id']
        assert new._timings[0].request_sha256 == digest(canonical(original))


def test_complete_direct_fit_reuses_original_without_tokenizer_or_exposure_reopen(tmp_path):
    old = runtime(tmp_path); gate, tokenizer = _build(old)
    gate('direct', _batch())
    assert len(tokenizer.calls) == 4
    recovery, _, target = replay(tmp_path)
    new = runtime(tmp_path); new._native_contact_replay = recovery
    reused, tokenizer = _build(new, plan=target)
    reused('direct', _batch())
    assert tokenizer.calls == [] and new.touches == [] and new.unconsumed == []
    assert reused.receipts[0]['direct_fit_reuse']['new_tokenizer_contacts'] == 0


@pytest.mark.parametrize('fault', ['missing-result', 'missing-custody', 'journal', 'request', 'service', 'config'])
def test_direct_fit_unknown_or_changed_custody_never_recontacts(tmp_path, fault):
    old = runtime(tmp_path); gate, _ = _build(old); gate('direct', _batch())
    recovery, _, target = replay(tmp_path)
    root = tmp_path / 'prepared' / FIT_RUN / FIT_ATTEMPT
    artifact = root / 'direct.request-fit.json'
    if fault == 'missing-result': artifact.unlink()
    if fault == 'missing-custody': artifact.with_name(artifact.name + '.custody.json').unlink()
    if fault == 'journal': artifact.with_name(artifact.name + '.contacts.jsonl').write_text('')
    new = runtime(tmp_path); new._native_contact_replay = recovery
    if fault == 'service': new.session_required_identity = lambda: {'session': 'different'}
    if fault == 'config': new.request_fit_inputs = lambda p: {'config_sha256': 'changed'}
    reused, tokenizer = _build(new, plan=target)
    with pytest.raises(PreparationError):
        reused('direct', _batch(task_instruction='changed') if fault == 'request' else _batch())
    assert tokenizer.calls == []


def test_unknown_generation_prevents_even_fit_gate_construction(tmp_path):
    recovery, _, target = replay(tmp_path, unknown=True)
    new = runtime(tmp_path); new._native_contact_replay = recovery
    with pytest.raises(ContactJournalError, match='unknown'):
        _build(new, plan=target)
