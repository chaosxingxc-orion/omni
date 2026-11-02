"""Raw Direct recovery keeps one spent source and pinned zero-contact ancestry."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.core.contact_journal import (
    ContactJournal, ContactJournalError, NativeContactReplay, canonical, digest,
)
from speech_aware_evidence_acquisition.core.flight.native_recovery import load_native_recovery
from test_direct_only_fit import _batch, _plan

ARM = 'raw-speech-agent-native-tools'


def abort(data, plan, *, calls=0, unknown=False):
    journal = ContactJournal(data, plan, ARM)
    payloads = []
    for index in range(calls):
        payload = _batch()[index].as_payload()
        payload['request_id'] = f'{plan.attempt_id}-request-{index}'
        payload['history'] = [{'role': 'control', 'contact_id': f'direct-{index}'}]
        journal.intent(payload)
        if not unknown:
            journal.response(payload['request_id'], digest(canonical(payload)),
                canonical({'text': f'answer-{index}', 'usage': {'completion_tokens': 2}}), 1)
        payloads.append(payload)
    path = data / 'attempts' / plan.run_id / (plan.attempt_id + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical({'state': 'ABORTED', 'run_id': plan.run_id,
        'attempt_id': plan.attempt_id, 'carrier_lock_key': plan.carrier_lock_key,
        'actual': {'calls': calls, 'audio_seconds': sum(p['audio_seconds'] for p in payloads)}}))
    return payloads


def declare(data, source, target):
    replay = NativeContactReplay(data, source, target, ARM)
    receipt = data / (target.attempt_id + '.recovery.json')
    target_plan = asdict(target)
    if target.owner_allowance is not None:
        target_plan['owner_allowance'] = asdict(target.owner_allowance)
    receipt.write_bytes(canonical({'schema': 'saea-native-recovery-v1',
        'source': replay.source_binding(), 'target_plan': target_plan}))
    pin = {'path': str(receipt.resolve()), 'sha256': digest(receipt.read_bytes())}
    path = data / 'runs' / target.run_id / target.attempt_id / 'resume-declaration.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical({'schema': 'saea-resume-declaration-v1',
        'run_id': target.run_id, 'attempt_id': target.attempt_id,
        'native_recovery_receipt': pin,
        'predecessors': [{'attempt_id': source.attempt_id, 'mode': 'native-contact-journal',
            'pin_sha256': replay.source_binding()['attempt_sha256']}]}))
    return pin


def chain(data, *, calls=2, unknown=False, original=None, zero_ancestors=2):
    original = original or _plan()
    plans = [replace(original, attempt_id=f'generation-{index}')
             for index in range(zero_ancestors + 2)]
    for index, plan in enumerate(plans[:-1]):
        if index:
            declare(data, plans[index - 1], plan)
        payloads = abort(data, plan, calls=calls if index == zero_ancestors else 0,
            unknown=unknown)
    return plans, payloads


@pytest.mark.parametrize('calls', [0, 2])
def test_zero_ancestors_preserve_original_plan_and_every_spent_contact(tmp_path, calls):
    plans, payloads = chain(tmp_path, calls=calls)
    pin = declare(tmp_path, plans[-2], plans[-1])
    replay = load_native_recovery(pin, data_root=tmp_path, plan=plans[-1], arm=ARM)
    replay.assert_all_contacts_observed()
    assert replay.scientific_source_plan == plans[0]
    assert replay.source_plan == plans[-2]
    assert replay.source_actual_usage == {'calls': calls, 'audio_seconds': calls * 1.5}
    assert replay.request_ids == {p['request_id'] for p in payloads}
    binding = replay.source_binding()
    assert binding['empty_ancestry']['source']['empty_ancestry']['source']['plan'] == asdict(plans[0])
    if payloads:
        original = replay.match({**payloads[0], 'request_id': 'new-route-only'})
        assert original == payloads[0]
        assert json.loads(replay.consume(original)['response_bytes'])['text'] == 'answer-0'
        assert replay.receipt(original['request_id'])['new_model_calls'] == 0
        # The second response remains charged even when it was not reused.
        assert replay.source_binding() == binding
        with pytest.raises(ContactJournalError, match='already replayed'):
            replay.consume(original)


def test_unknown_immediate_source_blocks_all_new_dispatch(tmp_path):
    plans, _ = chain(tmp_path, calls=1, unknown=True)
    replay = NativeContactReplay(tmp_path, plans[-2], plans[-1], ARM)
    with pytest.raises(ContactJournalError, match='unknown'):
        replay.assert_all_contacts_observed()


@pytest.mark.parametrize('unknown', [False, True])
def test_spent_or_unknown_earlier_ancestor_is_not_silently_dropped(tmp_path, unknown):
    original = _plan()
    source = replace(original, attempt_id='middle')
    abort(tmp_path, original, calls=1, unknown=unknown)
    declare(tmp_path, original, source)
    abort(tmp_path, source)
    with pytest.raises(ContactJournalError, match='spent ancestor'):
        NativeContactReplay(tmp_path, source, replace(source, attempt_id='target'), ARM)


@pytest.mark.parametrize('ancestor_index', [0, 1])
@pytest.mark.parametrize('kind', ['intent', 'response'])
@pytest.mark.parametrize('operation', ['dispatch', 'reuse', 'fresh-match', 'fresh-consume'])
def test_new_empty_ancestor_journal_files_invalidate_cached_recovery(
        tmp_path, ancestor_index, kind, operation):
    plans, payloads = chain(tmp_path)
    replay = NativeContactReplay(tmp_path, plans[-2], plans[-1], ARM)
    journal = ContactJournal(tmp_path, plans[ancestor_index], ARM)
    journal.root.mkdir(parents=True, exist_ok=True)
    journal._path('late-unreconciled-contact', kind).write_bytes(b'{}')
    with pytest.raises(ContactJournalError, match='inventory changed'):
        if operation == 'dispatch':
            replay.assert_all_contacts_observed()
        elif operation == 'reuse':
            replay.receipt(payloads[0]['request_id'])
        elif operation == 'fresh-match':
            replay.match({**payloads[0], 'sample_id': 'fresh-sample'})
        else:
            replay.consume({**payloads[0], 'request_id': 'fresh-request'})


@pytest.mark.parametrize('fault', ['receipt-pin', 'target-plan', 'source-plan', 'predecessor', 'cycle'])
def test_changed_ancestor_pin_plan_provenance_or_cycle_refused(tmp_path, fault):
    plans, _ = chain(tmp_path)
    path = tmp_path / 'runs' / plans[-2].run_id / plans[-2].attempt_id / 'resume-declaration.json'
    declaration = json.loads(path.read_bytes())
    receipt_path = Path(declaration['native_recovery_receipt']['path'])
    receipt = json.loads(receipt_path.read_bytes())
    if fault == 'receipt-pin':
        receipt_path.write_bytes(receipt_path.read_bytes() + b' ')
    elif fault == 'predecessor':
        declaration['predecessors'][0]['pin_sha256'] = '0' * 64
    else:
        if fault == 'target-plan':
            receipt['target_plan']['planned_model_calls'] += 1
        elif fault == 'source-plan':
            receipt['source']['plan']['planned_model_calls'] += 1
        else:
            receipt['source']['plan']['attempt_id'] = plans[-1].attempt_id
        receipt_path.write_bytes(canonical(receipt))
        declaration['native_recovery_receipt']['sha256'] = digest(receipt_path.read_bytes())
    path.write_bytes(canonical(declaration))
    with pytest.raises(ContactJournalError):
        NativeContactReplay(tmp_path, plans[-2], plans[-1], ARM)


def test_ancestry_limit_accepts_boundary_and_refuses_deeper_chain(tmp_path, monkeypatch):
    # Exercise the production limit logic with a small bound, without creating
    # a large recursive fixture or weakening request/response validation.
    monkeypatch.setattr(NativeContactReplay, 'MAX_EMPTY_RECOVERY_ANCESTORS', 2)
    plans, _ = chain(tmp_path, calls=0)
    declare(tmp_path, plans[-2], plans[-1])
    abort(tmp_path, plans[-1])
    with pytest.raises(ContactJournalError, match='exceeds supported depth'):
        NativeContactReplay(tmp_path, plans[-1], replace(plans[-1], attempt_id='too-deep'), ARM)


def test_final_report_allowance_survives_repeated_raw_recovery(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.contracts import plan as plan_module
    monkeypatch.setattr(plan_module, '_STUDY_REPO_ROOT', tmp_path)
    original = _plan()
    allowance = {'schema': 'saea-owner-split-allowance-v1',
        'study': 'speech-aware-evidence-acquisition', 'status': 'authorised',
        'dated_owner_decision': 'Fixture decision', 'venue': 'Fixture study',
        'split': {'carrier_lock_key': 'slurp', 'split_identity_hash': original.split_identity_hash},
        'permitted_run_ids': [original.run_id], 'read_run_ids': [original.run_id],
        'preparation_run_ids': []}
    rel = 'docs/receipts/owner/repeated-recovery-fixture.json'
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical(allowance))
    pin = {'path': rel, 'sha256': digest(path.read_bytes())}
    original = replace(original, split_role='final-report', owner_allowance=pin)
    plans, _ = chain(tmp_path, original=original)
    recovery_pin = declare(tmp_path, plans[-2], plans[-1])
    replay = load_native_recovery(recovery_pin, data_root=tmp_path, plan=plans[-1], arm=ARM)
    replay.assert_all_contacts_observed()
    assert replay.scientific_source_plan.owner_allowance == plans[-1].owner_allowance
    assert replay.source_binding()['plan']['owner_allowance'] == pin
    # ExecutionPlan equality excludes this InitVar, so ancestry must compare
    # the allowance explicitly, just like the outer receipt loader does.
    receipt_path = tmp_path / (plans[-2].attempt_id + '.recovery.json')
    receipt = json.loads(receipt_path.read_bytes())
    del receipt['target_plan']['owner_allowance']
    receipt_path.write_bytes(canonical(receipt))
    declaration_path = tmp_path / 'runs' / plans[-2].run_id / plans[-2].attempt_id / 'resume-declaration.json'
    declaration = json.loads(declaration_path.read_bytes())
    declaration['native_recovery_receipt']['sha256'] = digest(receipt_path.read_bytes())
    declaration_path.write_bytes(canonical(declaration))
    with pytest.raises(ContactJournalError, match='ancestor target differs'):
        NativeContactReplay(tmp_path, plans[-2], plans[-1], ARM)


def test_repeated_zero_contact_source_performs_first_fit_then_real_client_generation(tmp_path):
    import base64
    from types import SimpleNamespace
    from test_direct_native_recovery import runtime
    from test_direct_only_fit import _build, _Tokenizer
    from test_rails_frozen_core_client import _World, _client, _zero_supply_messages, _plan as client_plan

    class World(_World):
        SAMPLE_SPECS = tuple((f's{index}', 1.5) for index in range(1, 5))

    world = World(tmp_path / 'source')
    source = client_plan(planned_model_calls=8, planned_speech_audio_seconds=12)
    old, _, _ = _client(world, tmp_path / 'source', plan=source, arm=ARM,
        tool_schemas=(), contact_journal=True)
    old.close()
    old._adapter._attempt_handle.mark_aborted('render failed before FIT or generation')
    middle = replace(source, attempt_id='zero-recovery')
    declare(world.data_root, source, middle)
    abort(world.data_root, middle)
    target = replace(source, attempt_id='working-recovery')
    recovery = NativeContactReplay(world.data_root, middle, target, ARM)
    events = []
    successor = World(tmp_path / 'successor', data_root=world.data_root)

    def transport(raw):
        events.append('generation')
        return canonical({'text': 'new answer', 'usage': {'prompt_tokens': 3, 'completion_tokens': 2}})

    client, _, _ = _client(successor, tmp_path / 'successor', plan=target, arm=ARM,
        tool_schemas=(), contact_journal=True, contact_replay=recovery, transport=transport)
    batch = tuple(client.prepare_request(_zero_supply_messages(world.media_relpath(sample.sample_id)),
        raw_agent_sample_id=sample.sample_id, agent_state=(),
        agent_contact_id=f'direct-{sample.sample_id}', contact_instruction='Answer the speech.',
        tool_names=(), decoding_params={'temperature': 0, 'max_tokens': 64,
            'tool_choice': 'none', 'parallel_tool_calls': False}) for sample in world.samples)
    rt = runtime(world.data_root)
    rt._native_contact_replay = recovery

    def render(raw):
        payload = json.loads(raw)
        audio = (world.data_root / payload['media_relpath']).read_bytes()
        return {'model': 'fixture', 'messages': [{'role': 'user', 'content': [
            {'type': 'input_audio', 'input_audio': {'data': base64.b64encode(audio).decode(), 'format': 'wav'}}]}],
            'max_tokens': payload['decoding_params']['max_tokens']}

    class Tokenizer(_Tokenizer):
        def request_json(self, method, path, payload):
            events.append('fit')
            return super().request_json(method, path, payload)

    rt._transport = SimpleNamespace(render_request=render)
    gate, tokenizer = _build(rt, plan=target, tokenizer=Tokenizer())
    try:
        gate('direct', batch)
        assert events == ['fit'] * 4
        assert len(rt.touches) == 1 and rt.unconsumed == [gate.execution_plan.run_id]
        for prepared in batch:
            assert client.complete_prepared(prepared, id_slot=0) == 'new answer'
    finally:
        result = client.close()
    assert events == ['fit'] * 4 + ['generation'] * 4
    assert len(tokenizer.calls) == 4 and result['cost']['calls_used'] == 4
    assert recovery.scientific_source_plan == source
    assert recovery.source_actual_usage == {'calls': 0, 'audio_seconds': 0.0}
    custody = json.loads(Path(gate.receipts[0]['path'] + '.custody.json').read_bytes())
    assert custody['generation_attempt_id'] == target.attempt_id


@pytest.mark.parametrize('evidence', ['result', 'journal', 'custody'])
def test_zero_generation_with_any_partial_fit_evidence_never_retries(tmp_path, evidence):
    from test_direct_native_recovery import runtime
    from test_direct_only_fit import _build, _batch, FIT_RUN, FIT_ATTEMPT
    from speech_aware_evidence_acquisition.core.preparation import PreparationError
    plans, _ = chain(tmp_path, calls=0)
    root = tmp_path / 'prepared' / FIT_RUN / FIT_ATTEMPT
    root.mkdir(parents=True)
    suffix = {'result': '', 'journal': '.contacts.jsonl', 'custody': '.custody.json'}[evidence]
    (root / ('direct.request-fit.json' + suffix)).write_bytes(b'')
    rt = runtime(tmp_path)
    rt._native_contact_replay = NativeContactReplay(tmp_path, plans[-2], plans[-1], ARM)
    gate, tokenizer = _build(rt, plan=plans[-1])
    with pytest.raises(PreparationError, match='incomplete or unknown'):
        gate('direct', _batch())
    assert not tokenizer.calls and not rt.touches and not rt.unconsumed


def test_spent_generation_without_fit_evidence_is_not_first_fit(tmp_path):
    from test_direct_native_recovery import runtime
    from test_direct_only_fit import _build, _batch
    from speech_aware_evidence_acquisition.core.preparation import PreparationError
    plans, _ = chain(tmp_path, calls=1)
    rt = runtime(tmp_path)
    rt._native_contact_replay = NativeContactReplay(tmp_path, plans[-2], plans[-1], ARM)
    gate, tokenizer = _build(rt, plan=plans[-1])
    with pytest.raises(PreparationError, match='spent Direct predecessor'):
        gate('direct', _batch())
    assert not tokenizer.calls and not rt.touches and not rt.unconsumed


def test_complete_fit_from_pinned_zero_generation_ancestor_reuses_without_contact(tmp_path):
    from test_direct_native_recovery import runtime
    from test_direct_only_fit import _build, _batch
    original = _plan()
    initial = runtime(tmp_path)
    gate, old_tokenizer = _build(initial, plan=original)
    gate('direct', _batch())
    abort(tmp_path, original)
    source = replace(original, attempt_id='failed-before-reuse')
    declare(tmp_path, original, source)
    abort(tmp_path, source)
    target = replace(original, attempt_id='second-recovery')
    rt = runtime(tmp_path)
    rt._native_contact_replay = NativeContactReplay(tmp_path, source, target, ARM)
    gate, tokenizer = _build(rt, plan=target)
    gate('direct', _batch())
    assert len(old_tokenizer.calls) == 4 and not tokenizer.calls
    assert not rt.touches and not rt.unconsumed
    assert gate.receipts[0]['direct_fit_reuse']['new_tokenizer_contacts'] == 0


def test_first_fit_requires_existing_unconsumed_exposure_row(tmp_path):
    from test_direct_native_recovery import runtime
    from test_direct_only_fit import _build, _batch
    plans, _ = chain(tmp_path, calls=0)
    rt = runtime(tmp_path)
    rt._native_contact_replay = NativeContactReplay(tmp_path, plans[-2], plans[-1], ARM)
    def consumed(_):
        raise ValueError('FIT exposure already consumed')
    rt._require_exposure_row_unconsumed = consumed
    gate, tokenizer = _build(rt, plan=plans[-1])
    with pytest.raises(ValueError, match='already consumed'):
        gate('direct', _batch())
    assert not tokenizer.calls


def test_failed_first_tokenizer_contact_is_never_retried_after_another_zero_generation(tmp_path):
    from test_direct_native_recovery import runtime
    from test_direct_only_fit import _build, _batch, _Tokenizer
    from speech_aware_evidence_acquisition.core.preparation import PreparationError
    plans, _ = chain(tmp_path, calls=0)
    # Record the operational receipt exactly as a recovered runtime would.
    declare(tmp_path, plans[-2], plans[-1])
    rt = runtime(tmp_path)
    rt._native_contact_replay = NativeContactReplay(tmp_path, plans[-2], plans[-1], ARM)
    class UnknownTokenizer(_Tokenizer):
        def request_json(self, method, path, payload):
            self.calls.append((method, path, payload))
            raise TimeoutError('tokenizer outcome unknown')
    gate, failed = _build(rt, plan=plans[-1], tokenizer=UnknownTokenizer())
    with pytest.raises(PreparationError, match='FIT_READY refused'):
        gate('direct', _batch())
    assert len(failed.calls) == 1
    abort(tmp_path, plans[-1])
    target = replace(plans[-1], attempt_id='after-unknown-fit')
    next_rt = runtime(tmp_path)
    next_rt._native_contact_replay = NativeContactReplay(tmp_path, plans[-1], target, ARM)
    next_gate, tokenizer = _build(next_rt, plan=target)
    with pytest.raises(PreparationError, match='incomplete or unknown'):
        next_gate('direct', _batch())
    assert not tokenizer.calls and not next_rt.touches and not next_rt.unconsumed
