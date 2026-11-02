"""Final-report exposure counts one verified native scientific evaluation."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.contracts import plan as plan_module
from speech_aware_evidence_acquisition.contracts.errors import GateClosed
from speech_aware_evidence_acquisition.contracts.gate_exposure import _ExposureAndCarrierMixin
from speech_aware_evidence_acquisition.core.contact_journal import canonical, digest
from test_direct_only_fit import _plan
from test_direct_repeated_recovery import ARM, abort, chain, declare


class Gate(_ExposureAndCarrierMixin):
    def __init__(self, data_root):
        self.data_root = data_root


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(plan_module, '_STUDY_REPO_ROOT', tmp_path)
    original = _plan()
    allowance = {'schema': 'saea-owner-split-allowance-v1',
        'study': 'speech-aware-evidence-acquisition', 'status': 'authorised',
        'dated_owner_decision': 'Fixture final report', 'venue': 'Fixture study',
        'split': {'carrier_lock_key': original.carrier_lock_key,
                  'split_identity_hash': original.split_identity_hash},
        'permitted_run_ids': [original.run_id], 'read_run_ids': [original.run_id],
        'preparation_run_ids': []}
    path = tmp_path / 'docs/receipts/owner/native-gate-fixture.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical(allowance))
    pin = {'path': str(path.relative_to(tmp_path)), 'sha256': digest(path.read_bytes())}
    original = replace(original, split_role='final-report', owner_allowance=pin)
    return tmp_path, original, allowance, Gate(tmp_path)


def declaration_path(data, plan):
    return data / 'runs' / plan.run_id / plan.attempt_id / 'resume-declaration.json'


@pytest.mark.parametrize('calls', [0, 2])
@pytest.mark.parametrize('zero_ancestors', [1, 3])
def test_verified_zero_ancestors_share_one_scientific_read(world, calls, zero_ancestors):
    data, original, allowance, gate = world
    plans, _ = chain(data, original=original, calls=calls, zero_ancestors=zero_ancestors)
    pin = declare(data, plans[-2], plans[-1])
    before = Path(pin['path']).read_bytes()
    gate._require_final_report_attempt_budget(plans[-1], allowance)
    # Admission remains valid once this very recovery has opened its slot.
    own = data / 'attempts' / plans[-1].run_id / (plans[-1].attempt_id + '.json')
    own.write_bytes(canonical({'state': 'OPENED'}))
    gate._require_final_report_attempt_budget(plans[-1], allowance)
    assert Path(pin['path']).read_bytes() == before
    # Gate does not clear or reprice the real spent predecessor.
    source = json.loads((own.parent / (plans[-2].attempt_id + '.json')).read_bytes())
    assert source['actual'] == {'calls': calls, 'audio_seconds': calls * 1.5}


def test_unknown_immediate_contacts_cannot_gain_an_evaluation_slot(world):
    data, original, allowance, gate = world
    plans, _ = chain(data, original=original, calls=1, unknown=True)
    declare(data, plans[-2], plans[-1])
    with pytest.raises(GateClosed, match='unknown'):
        gate._require_final_report_attempt_budget(plans[-1], allowance)


@pytest.mark.parametrize('unknown', [False, True])
def test_spent_earlier_ancestor_is_not_exempted(world, unknown):
    data, original, allowance, gate = world
    source = replace(original, attempt_id='middle')
    target = replace(original, attempt_id='target')
    abort(data, original, calls=1, unknown=unknown)
    declare(data, original, source)
    abort(data, source)
    # A declaration cannot manufacture eligibility for unsupported merging of
    # spent ancestors. The real shared replay validator must reject the chain.
    receipt = data / 'invalid-spent-chain.json'
    source_plan = {**asdict(source), 'owner_allowance': asdict(source.owner_allowance)}
    target_plan = {**asdict(target), 'owner_allowance': asdict(target.owner_allowance)}
    receipt.write_bytes(canonical({'schema': 'saea-native-recovery-v1',
        'source': {'plan': source_plan, 'arm': ARM}, 'target_plan': target_plan}))
    path = declaration_path(data, target)
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical({'schema': 'saea-resume-declaration-v1',
        'run_id': target.run_id, 'attempt_id': target.attempt_id,
        'native_recovery_receipt': {'path': str(receipt), 'sha256': digest(receipt.read_bytes())},
        'predecessors': [{'attempt_id': source.attempt_id, 'mode': 'native-contact-journal',
                         'pin_sha256': '0' * 64}]}))
    with pytest.raises(GateClosed, match='spent ancestor'):
        gate._require_final_report_attempt_budget(target, allowance)


@pytest.mark.parametrize('field,value', [
    ('protocol_hash', 'c' * 64), ('planned_model_calls', 9),
    ('planned_speech_audio_seconds', 13), ('planned_gpu_hours', 0.75),
])
def test_target_scientific_configuration_drift_is_refused(world, field, value):
    data, original, allowance, gate = world
    plans, _ = chain(data, original=original, calls=0)
    declare(data, plans[-2], plans[-1])
    with pytest.raises(GateClosed, match='another target plan'):
        gate._require_final_report_attempt_budget(replace(plans[-1], **{field: value}), allowance)


@pytest.mark.parametrize('fault', ['receipt-bytes', 'predecessor-pin', 'source-config',
                                  'missing-native-pin', 'ancestor-finalized', 'ancestor-response'])
def test_native_provenance_or_ancestor_inventory_drift_is_refused(world, fault):
    data, original, allowance, gate = world
    plans, _ = chain(data, original=original, calls=0)
    pin = declare(data, plans[-2], plans[-1])
    path = declaration_path(data, plans[-1])
    document = json.loads(path.read_bytes())
    if fault == 'receipt-bytes':
        receipt = Path(pin['path'])
        receipt.write_bytes(receipt.read_bytes() + b' ')
    elif fault == 'predecessor-pin':
        document['predecessors'][0]['pin_sha256'] = '0' * 64
    elif fault == 'source-config':
        receipt = Path(pin['path'])
        body = json.loads(receipt.read_bytes())
        body['source']['plan']['protocol_hash'] = 'c' * 64
        receipt.write_bytes(canonical(body))
        document['native_recovery_receipt']['sha256'] = digest(receipt.read_bytes())
    elif fault == 'missing-native-pin':
        del document['native_recovery_receipt']
    elif fault == 'ancestor-finalized':
        attempt = data / 'attempts' / original.run_id / (plans[0].attempt_id + '.json')
        body = json.loads(attempt.read_bytes())
        body['state'] = 'FINALIZED'
        attempt.write_bytes(canonical(body))
    else:
        journal = data / 'contact-journals' / original.run_id / plans[0].attempt_id
        journal.mkdir(parents=True, exist_ok=True)
        (journal / 'orphan.response.json').write_bytes(b'{}')
    path.write_bytes(canonical(document))
    with pytest.raises(GateClosed):
        gate._require_final_report_attempt_budget(plans[-1], allowance)


def test_unrelated_zero_contact_attempt_is_still_counted(world):
    data, original, allowance, gate = world
    plans, _ = chain(data, original=original, calls=0)
    declare(data, plans[-2], plans[-1])
    abort(data, replace(original, attempt_id='unrelated-zero-start'))
    with pytest.raises(GateClosed, match='allows at most 1 attempt'):
        gate._require_final_report_attempt_budget(plans[-1], allowance)


def test_fresh_repeat_without_recovery_declaration_stays_refused(world):
    data, original, allowance, gate = world
    abort(data, original)
    with pytest.raises(GateClosed, match='allows at most 1 attempt'):
        gate._require_final_report_attempt_budget(replace(original, attempt_id='fresh-repeat'), allowance)
