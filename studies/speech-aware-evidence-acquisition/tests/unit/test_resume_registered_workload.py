"""Registered recovery selects exact lineage and releases NPU before CPU scoring."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/analysis'))
import resume_registered_workload as entry
from speech_aware_evidence_acquisition.core.contact_journal import canonical
from test_direct_repeated_recovery import abort, declare
from test_direct_only_fit import _plan


def setup(tmp_path, monkeypatch, *, final_report=False):
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.contracts import plan as plan_module
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    from speech_aware_evidence_acquisition.core.flight import campaign, factory
    from speech_aware_evidence_acquisition.core.flight.receipts import FlightReceipts
    import registered_workload
    import tfgrpo_recovery_queue

    data = tmp_path / 'data'
    repo = tmp_path / 'tree/studies/study'
    candidate = 'fixed-direct'
    package = repo / 'docs/readiness' / candidate
    package.mkdir(parents=True)
    original = _plan()
    pin = None
    if final_report:
        monkeypatch.setattr(plan_module, '_STUDY_REPO_ROOT', repo)
        allowance = {'schema': 'saea-owner-split-allowance-v1',
            'study': 'speech-aware-evidence-acquisition', 'status': 'authorised',
            'dated_owner_decision': 'Fixture decision', 'venue': 'Fixture study',
            'split': {'carrier_lock_key': 'slurp', 'split_identity_hash': original.split_identity_hash},
            'permitted_run_ids': [original.run_id], 'read_run_ids': [original.run_id],
            'preparation_run_ids': []}
        rel = 'docs/receipts/owner/recovery-entry.json'
        path = repo / rel
        path.parent.mkdir(parents=True)
        path.write_bytes(canonical(allowance))
        pin = {'path': rel, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        original = replace(original, split_role='final-report', owner_allowance=pin)
    source = replace(original, attempt_id='generation-v2')
    abort(data, original)
    declare(data, original, source)
    abort(data, source, calls=1)
    spec = {'profile': 'profile', 'authority': 'user continued execution',
        'service_session': 'resident', 'run_id': original.run_id, 'attempt_id': original.attempt_id}
    (package / 'candidate.json').write_bytes(canonical(spec))
    (package / 'generation-plan-registered-v1.json').write_bytes(canonical({
        'execution_plan': asdict(original), 'owner_allowance': pin}))
    baseline = repo / 'configs/baseline' / (candidate + '.json')
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(canonical({'arm': 'raw-speech-agent-native-tools',
        'live_phase_fit': {'run_id': 'FIT', 'attempt_id': 'fit-v1'}}))
    cell = SimpleNamespace(run_id=original.run_id, attempt_id=original.attempt_id,
        document={'model_fragment': 'model', 'dataset_fragment': 'dataset'})
    monkeypatch.setattr(contracts, 'assert_execution_environment', lambda: None)
    monkeypatch.setattr(git_commit, 'require_clean_study_repo', lambda _: 'clean-commit')
    monkeypatch.setattr(registered_workload, 'validate_workload', lambda *a: 4)
    monkeypatch.setattr(registered_workload, 'assert_execution_binding', lambda *a, **kw: None)
    monkeypatch.setattr(campaign, 'load_campaign_cells', lambda *a: [cell])
    monkeypatch.setenv('SAEA_DEPLOYMENT_PROFILE', 'profile')
    captured = {}

    def admit(root, selected, recovery, out, **kwargs):
        # The entry uses the real prepare/loader for source reconstruction and
        # custody. Isolate only external readiness/admission in this fixture.
        raw = json.loads(Path(recovery['path']).read_bytes())
        captured['recovery'] = raw
        target = raw['target_plan']['attempt_id']
        assert selected.attempt_id == original.attempt_id
        captured['admitted_target'] = target
        return SimpleNamespace(run_id=selected.run_id, attempt_id=target, document=selected.document)

    monkeypatch.setattr(tfgrpo_recovery_queue, 'admit', admit)

    def build(*args, **kwargs):
        captured['runtime_kwargs'] = kwargs
        receipts = FlightReceipts(kwargs['run_id'], kwargs['attempt_id'])
        seal = {}
        for key in ('generation_manifest', 'score_ledger_cells', 'score_jobs'):
            path = data / (key + '.json')
            path.write_bytes(b'sealed')
            seal[key + '_path'] = str(path)
            seal[key + '_sha256'] = hashlib.sha256(b'sealed').hexdigest()
        receipts.record('seal', seal)
        return SimpleNamespace(run_id=kwargs['run_id'], attempt_id=kwargs['attempt_id'], receipts=receipts)

    monkeypatch.setattr(factory, 'build_flight_runtime', build)
    out = data / 'work/codex/resumed'

    def execute(runtimes, **kwargs):
        flight, = runtimes
        kwargs['on_generation_released'](flight)
        release = json.loads((out / 'generation-released.json').read_bytes())
        assert release['attempt_id'] == flight.attempt_id
        assert release['generation_lease_released'] and not release['scoring_complete']
        assert not (out / 'campaign.json').exists() and not (out / 'complete.json').exists()
        captured['release'] = release
        return {'pending': [], 'completed': [flight.run_id],
            'states': {flight.run_id: {'state': 'scored'}}}

    monkeypatch.setattr(campaign, 'run_campaign', execute)
    return repo, data, candidate, out, original, source, captured


@pytest.mark.parametrize('final_report', [False, True])
def test_explicit_v2_to_v3_keeps_lineage_costs_allowance_and_release_identity(tmp_path, monkeypatch, final_report):
    repo, data, candidate, out, original, source, captured = setup(tmp_path, monkeypatch, final_report=final_report)
    assert entry.main(repo, data, candidate, out, 'generation-v3', source.attempt_id)['completed']
    recovery = captured['recovery']
    assert recovery['source']['plan']['attempt_id'] == source.attempt_id
    assert recovery['source']['actual_usage']['calls'] == 1
    assert recovery['source']['empty_ancestry']['source']['plan']['attempt_id'] == original.attempt_id
    assert recovery['target_plan']['attempt_id'] == 'generation-v3'
    if final_report:
        assert recovery['source']['plan']['owner_allowance'] == asdict(original.owner_allowance)
        assert recovery['target_plan']['owner_allowance'] == asdict(original.owner_allowance)
    assert captured['release']['attempt_id'] == 'generation-v3'
    assert captured['release']['rows'] == 4 and captured['release']['profile'] == 'profile'
    assert (out / 'complete.json').exists()


def test_omitted_source_keeps_original_registration_default(tmp_path, monkeypatch):
    repo, data, candidate, out, original, _, captured = setup(tmp_path, monkeypatch)
    entry.main(repo, data, candidate, out, 'legacy-successor')
    assert captured['recovery']['source']['plan']['attempt_id'] == original.attempt_id
    assert 'empty_ancestry' not in captured['recovery']['source']


@pytest.mark.parametrize('field', ['candidate', 'source_attempt', 'target_attempt'])
@pytest.mark.parametrize('bad', ['', '../escape', '/absolute', 'a/b', 'a\b', '.hidden'])
def test_unsafe_identifiers_rejected_before_environment_or_paths(tmp_path, monkeypatch, field, bad):
    from speech_aware_evidence_acquisition import contracts
    monkeypatch.setattr(contracts, 'assert_execution_environment', lambda: pytest.fail('must validate IDs first'))
    kwargs = dict(repo=tmp_path / 'missing', data=tmp_path / 'data', candidate='safe',
        out=tmp_path / 'output', target_attempt='target', source_attempt='source')
    kwargs[field] = bad
    with pytest.raises(ValueError, match='safe nonempty identifier'):
        entry.main(**kwargs)
    assert not kwargs['out'].exists()


def test_equal_explicit_attempts_rejected_before_environment(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition import contracts
    monkeypatch.setattr(contracts, 'assert_execution_environment', lambda: pytest.fail('must reject identical IDs first'))
    with pytest.raises(ValueError, match='must differ'):
        entry.main(tmp_path, tmp_path, 'safe', tmp_path / 'out', 'same', 'same')


def test_equal_default_source_rejected_before_output_creation(tmp_path, monkeypatch):
    repo, data, candidate, out, original, _, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='must differ'):
        entry.main(repo, data, candidate, out, original.attempt_id)
    assert not out.exists()


def test_generation_release_rejects_original_attempt_identity_after_recovery(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.core.flight import campaign
    repo, data, candidate, out, original, source, _ = setup(tmp_path, monkeypatch)
    def wrong_release(runtimes, **kwargs):
        flight, = runtimes
        flight.attempt_id = original.attempt_id
        kwargs['on_generation_released'](flight)
        pytest.fail('wrong release identity must fail')
    monkeypatch.setattr(campaign, 'run_campaign', wrong_release)
    with pytest.raises(ValueError, match='release identity differs'):
        entry.main(repo, data, candidate, out, 'generation-v3', source.attempt_id)
    assert not (out / 'generation-released.json').exists()
    assert not (out / 'complete.json').exists()
