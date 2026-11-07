import copy
import json
from dataclasses import replace

import pytest

from speech_aware_evidence_acquisition.core.native_fit_recovery import RecoverableNativeFitGate
from speech_aware_evidence_acquisition.core.preparation import PreparationError, canonical_bytes
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal, NativeContactReplay, canonical, digest
from speech_aware_evidence_acquisition.core.attempts import AttemptStore
from speech_aware_evidence_acquisition.core.preparation_fit import build_native_contact_fit_gate
from speech_aware_evidence_acquisition.rails.core_client import PreparedCoreRequest
from test_native_contact_fit import _raw_runtime, _raw_prepared, _program, _plan, _RawTokenizer


def source(tmp_path):
    runtime = _raw_runtime(tmp_path)
    plan = _plan(run_id='SAEA-FIT-RECOVERY-V1')
    tokenizer = _RawTokenizer()
    gate = RecoverableNativeFitGate(runtime, plan, lambda: build_native_contact_fit_gate(
        runtime, _program(), plan, client_factory=lambda **kwargs: tokenizer))
    request = _raw_prepared('s1', 'contact-1', runtime._transport.raw)
    payload = request.as_payload(); payload['audio_seconds'] = 1
    request = PreparedCoreRequest(canonical_bytes(payload), object())
    assert gate(request) == 7
    assert len(tokenizer.calls) == 1
    handle = AttemptStore(tmp_path/'attempts').open_attempt(plan.run_id, plan.attempt_id,
        carrier_lock_key=plan.carrier_lock_key, reserved_calls=4, reserved_audio_seconds=10)
    journal = ContactJournal(tmp_path, plan, runtime._arm())
    journal.intent(payload); handle.record_usage(calls=1, audio_seconds=1)
    journal.response(payload['request_id'], digest(canonical(payload)), b'{"text":"answer","usage":{}}', 1)
    handle.mark_aborted('source process terminal in test')
    recovery = replace(plan, attempt_id='recovery-v1')
    replay = NativeContactReplay(tmp_path, plan, recovery, runtime._arm())
    return runtime, recovery, replay, request, gate


def no_tokenizer():
    pytest.fail('reused FIT must not construct a tokenizer gate')


def test_known_contact_reuses_verified_fit_without_tokenizer_or_duplicate_receipt(tmp_path):
    runtime, plan, replay, request, old = source(tmp_path)
    gate = RecoverableNativeFitGate(runtime, plan, no_tokenizer, replay=replay)
    payload = request.as_payload(); payload['request_id'] = 'recovery-request'; payload['decoding_params']['id_slot']=9
    request = PreparedCoreRequest(canonical_bytes(payload), object())
    assert gate(request) == 7
    assert gate.receipts[0]['path'] == old.receipts[0]['path']
    assert gate.receipts[0]['native_fit_reuse']['new_tokenizer_contacts'] == 0
    with pytest.raises(PreparationError, match='repeated'): gate(request)


def test_fit_reuse_rejects_endpoint_accounting_changed_even_if_journal_hash_rewritten(tmp_path):
    from pathlib import Path
    runtime, plan, replay, request, old = source(tmp_path)
    custody = next((replay.source.root).glob('*.fit.json'))
    document = json.loads(custody.read_bytes())
    artifact = Path(old.receipts[0]['path'])
    journal = artifact.with_name(artifact.name + '.contacts.jsonl')
    lines = [json.loads(line) for line in journal.read_bytes().splitlines()]
    lines[0]['route'] = '/different'
    journal.write_bytes(b''.join(canonical(line) + b'\n' for line in lines))
    document['endpoint_journal_sha256'] = digest(journal.read_bytes())
    custody.write_bytes(canonical(document))
    gate = RecoverableNativeFitGate(runtime, plan, no_tokenizer, replay=replay)
    with pytest.raises(PreparationError, match='endpoint accounting'):
        gate(request)


@pytest.mark.parametrize('change', ['custody', 'artifact', 'journal', 'identity', 'inputs', 'missing', 'limit'])
def test_fit_reuse_rejects_changed_evidence_before_any_tokenizer(tmp_path, change):
    runtime, plan, replay, request, old = source(tmp_path)
    gate = RecoverableNativeFitGate(runtime, plan, no_tokenizer, replay=replay)
    from pathlib import Path
    artifact = Path(old.receipts[0]['path'])
    path = next(iter(gate._pins))
    if change == 'custody': path.write_text('{}')
    elif change == 'artifact': artifact.write_text('{}')
    elif change == 'journal': artifact.with_name(artifact.name+'.contacts.jsonl').write_text('changed')
    elif change == 'identity': runtime.session_required_identity=lambda: {'different':'model'}
    elif change == 'inputs': runtime.request_fit_inputs=lambda payloads: {'different':'config'}
    elif change == 'missing': path.unlink()
    with pytest.raises(PreparationError): gate(request, **({'prompt_limit_tokens':6} if change=='limit' else {}))


def test_new_contact_uses_registered_fit_once_and_captures_receipt(tmp_path):
    runtime, plan, replay, request, old = source(tmp_path)
    tokenizer = _RawTokenizer()
    gate = RecoverableNativeFitGate(runtime, plan, lambda: build_native_contact_fit_gate(
        runtime, _program(), plan, client_factory=lambda **kwargs: tokenizer), replay=replay)
    fresh = _raw_prepared('s2', 'contact-2', runtime._transport.raw)
    assert gate(fresh) == 7
    assert len(tokenizer.calls) == 1
    assert gate.journal.path(fresh.as_payload()).is_file()


def test_operational_recovery_receipt_binds_exact_source_and_target_without_protocol_changes(tmp_path):
    from dataclasses import asdict
    from speech_aware_evidence_acquisition.core.flight.native_recovery import load_native_recovery
    runtime, plan, replay, request, old = source(tmp_path)
    path=tmp_path/'recovery.json'
    document={'schema':'saea-native-recovery-v1', 'source':replay.source_binding(), 'target_plan':asdict(plan)}
    path.write_bytes(canonical(document))
    declaration={'path':str(path),'sha256':digest(path.read_bytes())}
    loaded=load_native_recovery(declaration, data_root=tmp_path,plan=plan,arm=runtime._arm())
    assert loaded.match(request.as_payload()) == request.as_payload()
    with pytest.raises(ValueError):
        load_native_recovery(declaration, data_root=tmp_path,plan=replace(plan,attempt_id='other'),arm=runtime._arm())
    path.write_bytes(b'{}')
    with pytest.raises(ValueError):
        load_native_recovery(declaration, data_root=tmp_path,plan=plan,arm=runtime._arm())


def test_recovery_selection_preserves_registered_fragment_and_refuses_other_run(tmp_path):
    from dataclasses import asdict
    from speech_aware_evidence_acquisition.core.flight.campaign import CampaignCell
    from speech_aware_evidence_acquisition.core.flight.native_recovery import select_recovery_cell
    runtime, plan, replay, request, old = source(tmp_path)
    cell=CampaignCell.from_document({'cell_id':'c','model_fragment':'m','dataset_fragment':'d',
        'baseline_fragment':'b','run_id':plan.run_id,'attempt_id':replay.source_plan.attempt_id,
        'readiness':{'state':'ready'},'allowed_execution_stages':['worker','check']})
    path=tmp_path/'recovery.json'
    path.write_bytes(canonical({'schema':'saea-native-recovery-v1','source':replay.source_binding(),
        'target_plan':asdict(plan)}))
    declaration={'path':str(path),'sha256':digest(path.read_bytes())}
    selected=select_recovery_cell(cell,declaration,data_root=tmp_path)
    assert selected.attempt_id==plan.attempt_id
    assert selected.document=={**cell.document,'attempt_id':plan.attempt_id}
    other=CampaignCell.from_document({**cell.document,'run_id':'SAEA-OTHER'})
    with pytest.raises(ValueError):select_recovery_cell(other,declaration,data_root=tmp_path)


def test_recovery_lease_validates_preflight_before_issuing_target_capability(tmp_path):
    import io
    from dataclasses import asdict
    from speech_aware_evidence_acquisition.core.flight.campaign import CampaignCell
    from speech_aware_evidence_acquisition.core.flight.lease_cli import handoff_worker_lease
    from test_flight_lease_cli import Lease
    runtime, plan, replay, request, old = source(tmp_path)
    cell=CampaignCell.from_document({'cell_id':'c','model_fragment':'m','dataset_fragment':'d',
        'baseline_fragment':'b','run_id':plan.run_id,'attempt_id':replay.source_plan.attempt_id,
        'readiness':{'state':'ready'},'allowed_execution_stages':['worker']})
    path=tmp_path/'recovery.json'
    path.write_bytes(canonical({'schema':'saea-native-recovery-v1','source':replay.source_binding(),
        'target_plan':asdict(plan)}))
    pin={'path':str(path),'sha256':digest(path.read_bytes())}
    events=[]; lease=Lease()
    class Runtime:
        def preflight(self): events.append('preflight')
        def session_required_identity(self): return {'model':'pinned'}
    def factory(*args, **kwargs):
        assert kwargs['attempt_id']==plan.attempt_id
        assert kwargs['native_contact_recovery']==pin
        return Runtime()
    def acquire(*args, **kwargs):
        assert events==['preflight']
        assert kwargs['attempt_id']==plan.attempt_id
        events.append('lease')
        return lease
    assert handoff_worker_lease(repo_root=tmp_path,data_root=tmp_path,runtime_root=tmp_path,
        lock_path=tmp_path/'lock',experiment='e',cell=cell,owner='root',expected_session_id='session',
        token_output=tmp_path/'coordination'/'recovery.token',native_contact_recovery=pin,
        input_stream=io.StringIO('release\n'), output_stream=io.StringIO(),runtime_factory=factory,
        lease_factory=acquire,resident_finder=lambda root:(None,{'session_id':'session','identity':{'model':'pinned'}}),
        study_head_factory=lambda root:'a'*40)==0
    assert lease.released and events==['preflight','lease']


def test_unknown_generation_result_refuses_before_fit_reuse_or_tokenizer(tmp_path):
    from speech_aware_evidence_acquisition.core.contact_journal import ContactJournalError
    runtime, plan, replay, request, old = source(tmp_path)
    rid=request.as_payload()['request_id']
    replay.source._path(rid,'response').unlink()
    replay=NativeContactReplay(tmp_path,replay.source_plan,plan,runtime._arm())
    gate=RecoverableNativeFitGate(runtime,plan,no_tokenizer,replay=replay)
    with pytest.raises(ContactJournalError,match='unknown'):gate(request)
    assert not gate.receipts


def test_recovery_and_inner_fit_lock_waits_are_separately_identified(tmp_path, monkeypatch):
    import os
    from speech_aware_evidence_acquisition.core import stage_timing
    path = tmp_path/'stages.jsonl'
    monkeypatch.setenv('SAEA_STAGE_TIMING_PATH', str(path))
    monkeypatch.setattr(stage_timing, '_fd', None)
    monkeypatch.setattr(stage_timing, '_path', None)
    try:
        source(tmp_path)
        events = [json.loads(line) for line in path.read_text().splitlines()]
        locks = [e for e in events if e['stage'] in {
            'native_fit_lock_wait', 'native_fit_recovery_lock_wait'}]
        assert {e['stage'] for e in locks} == {
            'native_fit_lock_wait', 'native_fit_recovery_lock_wait'}
        assert len(locks) == 2
        assert all(e['sample_id'] == 's1' and e['request_id'] for e in locks)
    finally:
        if stage_timing._fd is not None:
            os.close(stage_timing._fd)


def test_wrapper_allows_parallel_contacts_and_binds_each_custody(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    barrier = threading.Barrier(2)
    class Parallel(_RawTokenizer):
        def request_json(self, method, path, payload):
            barrier.wait(timeout=5)
            return super().request_json(method, path, payload)
    runtime = _raw_runtime(tmp_path)
    plan = _plan(run_id='SAEA-PARALLEL-FIT-V1')
    gate = RecoverableNativeFitGate(runtime, plan, lambda: build_native_contact_fit_gate(
        runtime, _program(), plan, client_factory=Parallel, max_concurrency=2))
    requests = [_raw_prepared(s, s+'-contact', runtime._transport.raw) for s in ('s1', 's2')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(gate, requests)) == [7, 7]
    for req in requests:
        payload = req.as_payload()
        document = json.loads(gate.journal.path(payload).read_text())
        assert document['payload'] == payload
        assert document['receipt']['sample_id'] == payload['sample_id']
    assert len(gate.receipts) == 2


def test_runtime_only_fit_repair_keeps_exact_boundary_and_all_other_sources():
    from speech_aware_evidence_acquisition.core.native_fit_recovery import runtime_only_fit_repair
    old={'config_sha256':'c'*64,'sources':{'boundary_sha256':'d'*64,
         'implementation':{'runtime':'a'*64,'transport':'e'*64},'required_identity':{'model':'same'}}}
    current=copy.deepcopy(old);current['sources']['implementation']['runtime']='b'*64
    proof=runtime_only_fit_repair(current,old)
    assert proof['exact_request_boundary_sha256']=='d'*64
    for field in ('config','boundary','transport','service'):
        bad=copy.deepcopy(current)
        if field=='config':bad['config_sha256']='f'*64
        elif field=='boundary':bad['sources']['boundary_sha256']='f'*64
        elif field=='transport':bad['sources']['implementation']['transport']='f'*64
        else:bad['sources']['required_identity']['model']='other'
        with pytest.raises(PreparationError):runtime_only_fit_repair(bad,old)


def test_actual_fit_gate_reuses_runtime_repair_without_retokenization(tmp_path,monkeypatch):
    from test_native_contact_fit import _Runtime
    from speech_aware_evidence_acquisition.core.preparation_fit import boundary_identity
    original=_Runtime.request_fit_inputs
    runtime_hash={'value':'a'*64}
    def inputs(self,payloads):
        value=original(self,payloads)
        value['sources']['implementation']={'runtime':runtime_hash['value'],'transport':'e'*64}
        value['sources']['boundary_sha256']=boundary_identity(payloads)
        return value
    monkeypatch.setattr(_Runtime,'request_fit_inputs',inputs)
    runtime,plan,replay,request,old=source(tmp_path)
    runtime_hash['value']='b'*64
    gate=RecoverableNativeFitGate(runtime,plan,no_tokenizer,replay=replay)
    assert gate(request)==7
    proof=gate.receipts[0]['native_fit_reuse']['runtime_repair']
    assert proof['previous_runtime_sha256']=='a'*64 and proof['current_runtime_sha256']=='b'*64
    assert gate.receipts[0]['native_fit_reuse']['new_tokenizer_contacts']==0


def test_slow_initialization_does_not_block_reused_contact(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    runtime, plan, replay, request, old = source(tmp_path)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def factory():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return build_native_contact_fit_gate(runtime, _program(), plan,
                                            client_factory=_RawTokenizer, max_concurrency=2)
    gate = RecoverableNativeFitGate(runtime, plan, factory, replay=replay)
    fresh = _raw_prepared('s2', 'contact-2', runtime._transport.raw)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(gate, fresh)
        try:
            assert entered.wait(5)
            assert pool.submit(gate, request).result(timeout=3) == 7
            assert not pending.done()
        finally:
            release.set()
        assert pending.result(timeout=5) == 7
    assert calls == [1] and len(gate.receipts) == 2


@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt])
def test_initialization_failure_is_not_retried_by_waiting_contacts(tmp_path, failure):
    from concurrent.futures import ThreadPoolExecutor
    runtime = _raw_runtime(tmp_path)
    plan = _plan(run_id='SAEA-FIT-FAILED-INITIALIZATION-V1')
    calls = []
    def factory():
        calls.append(1)
        raise failure('partial initialization')
    gate = RecoverableNativeFitGate(runtime, plan, factory)
    requests = [_raw_prepared(s, s+'-contact', runtime._transport.raw) for s in ('s1', 's2')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(gate, r) for r in requests]
        errors = []
        for future in futures:
            try:
                future.result(timeout=5)
            except BaseException as error:
                errors.append(error)
    assert calls == [1]
    assert sum(type(e) is failure for e in errors) == 1
    assert sum(isinstance(e, PreparationError) for e in errors) == 1
    assert not gate.receipts
