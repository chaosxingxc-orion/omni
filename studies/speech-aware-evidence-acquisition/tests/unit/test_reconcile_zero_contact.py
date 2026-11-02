import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).parents[2] / 'scripts/analysis/reconcile_zero_contact.py'
spec = importlib.util.spec_from_file_location('zero_reconcile', path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); self.data = self.root / 'data'; self.data.mkdir()
        self.lock = self.data / 'locks/flight.lock'; self.lock.parent.mkdir()
        self.guard = self.lock.with_name('generation.guard'); self.guard.touch()
        self.source = self.data / 'attempts/RUN/generation-v1.json'
        self.source.parent.mkdir(parents=True)
        self.source.write_text(json.dumps({'state': 'ABORTED', 'run_id': 'RUN', 'attempt_id': 'generation-v1',
                                         'actual': {'calls': 0, 'audio_seconds': 0.0}}))
        lease = {'schema': 'saea-generation-lease-v1', 'run_id': 'RUN', 'attempt_id': 'generation-v1',
                 'pid': 123, 'study_head': 'a' * 40}
        self.lock.write_text(json.dumps(lease)); self.original = self.lock.read_bytes()
        self.plan = {'execution_plan': {'run_id': 'RUN', 'attempt_id': 'generation-v1'}}
        self.started = {'pid': 123, 'study_commit': 'a' * 40, 'time': 1793577601.0}
        self.container = [{'Id': 'b' * 64, 'Name': '/owned', 'State': {'Status': 'exited',
                           'Running': False, 'Pid': 0, 'StartedAt': '2026-11-02T00:00:00Z', 'FinishedAt': '2026-11-02T00:00:10Z'},
                           'Config': {'Env': ['SAEA_OBSERVABILITY_RUN_ID=RUN', 'SAEA_OBSERVABILITY_ATTEMPT_ID=generation-v1']}}]
        self.config = {'schema': m.SCHEMA, 'repo': str(self.root), 'data': str(self.data),
            'out': str(self.data / 'work/codex/reconcile-v1'), 'source_plan': self.pin('plan', self.plan),
            'baseline': self.pin('baseline', {'arm': 'raw-speech-agent-native-tools',
                'live_phase_fit': {'run_id': 'RUN-FIT', 'attempt_id': 'fit-v1'}}),
            'started': self.pin('started', self.started), 'container_inspect': self.pin('container', self.container),
            'container_id': 'b' * 64, 'container_name': 'owned', 'lease_sha256': m.digest(self.original),
            'target_attempt': 'generation-v2', 'arm': 'raw-speech-agent-native-tools',
            'fit': {'run_id': 'RUN-FIT', 'attempt_id': 'fit-v1'}, 'ports': [8200, 8201]}
        self.replays = 0

    def pin(self, name, value):
        p = self.root / (name + '.json'); p.write_text(json.dumps(value))
        return {'path': str(p), 'sha256': m.digest(p.read_bytes())}

    def replay(self, *args):
        self.replays += 1
        return {'actual_usage': {'calls': 0, 'audio_seconds': 0.0}, 'attempt_sha256': m.digest(self.source.read_bytes())}

    def run_reconcile(self, **kwargs):
        return m.reconcile(self.config, observe=kwargs.get('observe', lambda ports: {'queues': 'empty'}),
            replay=kwargs.get('replay', self.replay), pid_exists=kwargs.get('pid_exists', lambda pid: False))

    def assert_preserved(self):
        self.assertEqual(self.lock.read_bytes(), self.original)
        self.assertTrue(self.guard.is_file())

    def test_exact_zero_reconciliation_preserves_lease_and_guard(self):
        inode = self.guard.stat().st_ino
        before = self.source.read_bytes()
        result = self.run_reconcile()
        self.assertFalse(self.lock.exists())
        self.assertEqual(Path(result['preserved_lease_path']).read_bytes(), self.original)
        self.assertEqual(Path(result['preserved_lease_path']).parent, self.lock.parent)
        self.assertEqual(self.guard.stat().st_ino, inode)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(self.replays, 2)
        self.assertFalse((self.data / 'attempts/RUN/generation-v2.json').exists())
        self.assertTrue((Path(self.config['out']) / 'complete.json').is_file())

    def test_live_or_reused_pid_refused(self):
        with self.assertRaisesRegex(ValueError, 'PID exists'):
            self.run_reconcile(pid_exists=lambda pid: True)
        self.assert_preserved()

    def test_running_or_foreign_container_refused(self):
        for field, value in [('Running', True), ('Pid', 77), ('Status', 'running')]:
            container = copy.deepcopy(self.container); container[0]['State'][field] = value
            self.config['container_inspect'] = self.pin('container', container)
            with self.assertRaises(ValueError): self.run_reconcile()
            self.assert_preserved()
        self.config['container_inspect'] = self.pin('container', self.container)
        self.config['container_id'] = 'c' * 64
        with self.assertRaises(ValueError): self.run_reconcile()
        self.assert_preserved()

    def test_nonzero_or_nonaborted_attempt_refused(self):
        for change in ({'state': 'OPENED'}, {'actual': {'calls': 1, 'audio_seconds': 0.0}}):
            doc = {'state': 'ABORTED', 'run_id': 'RUN', 'attempt_id': 'generation-v1',
                   'actual': {'calls': 0, 'audio_seconds': 0.0}, **change}
            self.source.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'exact zero'): self.run_reconcile()
            self.assert_preserved()

    def test_native_fit_artifacts_and_attempt_each_refused(self):
        paths = ['contact-journals/RUN/generation-v1/x.intent.json',
                 'contact-journals/RUN/generation-v1/x.response.json',
                 'prepared/RUN-FIT/fit-v1/x.endpoint.json', 'prepared/RUN-FIT/fit-v1/direct.request-fit.json',
                 'contact-journals/RUN-FIT/fit-v1/x.json', 'attempts/RUN-FIT/fit-v1.json']
        for rel in paths:
            with self.subTest(rel=rel):
                p = self.data / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')
                with self.assertRaises(ValueError): self.run_reconcile()
                self.assert_preserved(); p.unlink()

    def test_foreign_or_changed_lease_refused(self):
        self.config['lease_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'lease bytes'): self.run_reconcile()
        self.assert_preserved()

    def test_quiescence_failure_preserves_all_state(self):
        def busy(ports): raise ValueError('active requests')
        with self.assertRaisesRegex(ValueError, 'active'): self.run_reconcile(observe=busy)
        self.assert_preserved()
        self.assertFalse(Path(self.config['out']).exists())

    def test_native_replay_failure_preserves_lease(self):
        def refuse(*args): raise ValueError('source plan mismatched')
        with self.assertRaisesRegex(ValueError, 'mismatched'): self.run_reconcile(replay=refuse)
        self.assert_preserved()

    def test_mutation_during_quiescence_refused(self):
        def mutate(ports):
            self.source.write_text('{}'); return {}
        with self.assertRaises(ValueError): self.run_reconcile(observe=mutate)
        self.assert_preserved()

    def test_existing_target_and_missing_guard_refused(self):
        target = self.data / 'runs/RUN/generation-v2'; target.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'already exists'): self.run_reconcile()
        target.rmdir(); self.guard.unlink()
        with self.assertRaisesRegex(ValueError, 'stable generation guard'): self.run_reconcile()
        self.assertEqual(self.lock.read_bytes(), self.original)

    def test_config_pin_tampering_refused(self):
        Path(self.config['source_plan']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'evidence changed'): self.run_reconcile()
        self.assert_preserved()

    def test_wrong_fit_or_arm_cannot_hide_custody(self):
        self.config['fit']['run_id'] = 'UNRELATED-FIT'
        with self.assertRaisesRegex(ValueError, 'baseline FIT or arm'): self.run_reconcile()
        self.assert_preserved()

    def test_foreign_terminal_container_run_refused(self):
        self.container[0]['Config']['Env'][0] = 'SAEA_OBSERVABILITY_RUN_ID=FOREIGN'
        self.config['container_inspect'] = self.pin('container', self.container)
        with self.assertRaisesRegex(ValueError, 'exact source run'): self.run_reconcile()
        self.assert_preserved()

    def test_guard_replacement_during_observation_refused(self):
        def replace_guard(ports):
            self.guard.unlink(); self.guard.touch(); return {}
        with self.assertRaisesRegex(ValueError, 'guard was replaced'):
            self.run_reconcile(observe=replace_guard)
        self.assert_preserved()


if __name__ == '__main__': unittest.main()


def test_actual_native_replay_zero_contact_source(tmp_path):
    """Formal-image integration: exercise real plan and journal validation."""
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from dataclasses import asdict
    plan = ExecutionPlan(run_id='SAEA-ZERO-CONTACT-TEST', attempt_id='generation-v1',
        execution_profile='bounded-discovery-probe', carrier_lock_key='slurp', split_role='dev',
        split_identity_hash='a' * 64, protocol_hash='b' * 64,
        planned_model_calls=2, planned_gpu_hours=1.0, planned_speech_audio_seconds=10)
    attempt = tmp_path / 'attempts' / plan.run_id / 'generation-v1.json'
    attempt.parent.mkdir(parents=True)
    attempt.write_text(json.dumps({'state': 'ABORTED', 'run_id': plan.run_id, 'attempt_id': plan.attempt_id,
        'carrier_lock_key': 'slurp', 'actual': {'calls': 0, 'audio_seconds': 0.0}}))
    bound = m.replay_source(path.parents[2], tmp_path, {'execution_plan': asdict(plan)},
                            'generation-v2', 'raw-speech-agent-native-tools')
    assert bound['actual_usage'] == {'calls': 0, 'audio_seconds': 0.0}
    assert bound['attempt_sha256'] == m.digest(attempt.read_bytes())
    assert not (tmp_path / 'runs').exists()


def test_actual_native_replay_final_report_preserves_allowance(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from speech_aware_evidence_acquisition.contracts import plan as plan_module
    from dataclasses import asdict
    monkeypatch.setattr(plan_module, '_STUDY_REPO_ROOT', tmp_path)
    allowance = {'schema': 'saea-owner-split-allowance-v1', 'study': 'speech-aware-evidence-acquisition',
        'status': 'authorised', 'dated_owner_decision': 'Fixture owner decision', 'venue': 'Fixture study',
        'split': {'carrier_lock_key': 'slurp', 'split_identity_hash': 'a' * 64},
        'permitted_run_ids': ['SAEA-ZERO-CONTACT-TEST'], 'read_run_ids': ['SAEA-ZERO-CONTACT-TEST'],
        'preparation_run_ids': []}
    rel = 'docs/receipts/owner/zero-contact-fixture.json'; p = tmp_path / rel
    p.parent.mkdir(parents=True); p.write_text(json.dumps(allowance))
    pin = {'path': rel, 'sha256': m.digest(p.read_bytes())}
    plan = ExecutionPlan(run_id='SAEA-ZERO-CONTACT-TEST', attempt_id='generation-v1',
        execution_profile='bounded-discovery-probe', carrier_lock_key='slurp', split_role='final-report',
        split_identity_hash='a' * 64, protocol_hash='b' * 64,
        planned_model_calls=2, planned_gpu_hours=1.0, planned_speech_audio_seconds=10, owner_allowance=pin)
    attempt = tmp_path / 'attempts' / plan.run_id / 'generation-v1.json'; attempt.parent.mkdir(parents=True)
    attempt.write_text(json.dumps({'state': 'ABORTED', 'run_id': plan.run_id, 'attempt_id': plan.attempt_id,
        'carrier_lock_key': 'slurp', 'actual': {'calls': 0, 'audio_seconds': 0.0}}))
    bound = m.replay_source(tmp_path, tmp_path, {'execution_plan': asdict(plan), 'owner_allowance': pin},
                            'generation-v2', 'raw-speech-agent-native-tools')
    assert bound['plan']['owner_allowance'] == pin
