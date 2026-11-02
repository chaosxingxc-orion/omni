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


def resumed_fixture(tmp_path, monkeypatch, *, final_report=False):
    """Real two-attempt native recovery, with the actual resumed-start schema."""
    from dataclasses import asdict, replace
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from speech_aware_evidence_acquisition.contracts import plan as plan_module
    from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
    monkeypatch.setattr(plan_module, '_STUDY_REPO_ROOT', tmp_path)
    repo = tmp_path; data = tmp_path / 'data'; data.mkdir()
    allowance = None
    if final_report:
        document = {'schema':'saea-owner-split-allowance-v1', 'study':'speech-aware-evidence-acquisition',
            'status':'authorised', 'dated_owner_decision':'Fixture owner decision', 'venue':'Fixture study',
            'split':{'carrier_lock_key':'slurp', 'split_identity_hash':'a' * 64},
            'permitted_run_ids':['SAEA-ZERO-RESUMED'], 'read_run_ids':['SAEA-ZERO-RESUMED'], 'preparation_run_ids':[]}
        p = repo / 'docs/receipts/owner/test.json'; p.parent.mkdir(parents=True); p.write_text(json.dumps(document))
        allowance = {'path':str(p.relative_to(repo)), 'sha256':m.digest(p.read_bytes())}
    original = ExecutionPlan(run_id='SAEA-ZERO-RESUMED', attempt_id='generation-v1',
        execution_profile='bounded-discovery-probe', carrier_lock_key='slurp',
        split_role='final-report' if final_report else 'dev', split_identity_hash='a' * 64,
        protocol_hash='b' * 64, planned_model_calls=2, planned_gpu_hours=1.0,
        planned_speech_audio_seconds=10, owner_allowance=allowance)
    current = replace(original, attempt_id='generation-v2'); arm='raw-speech-agent-native-tools'
    def pin(name, document):
        p = data / (name + '.json'); p.write_text(json.dumps(document))
        return {'path':str(p), 'sha256':m.digest(p.read_bytes())}
    for plan in (original, current):
        p = data / 'attempts' / plan.run_id / (plan.attempt_id + '.json'); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'state':'ABORTED', 'run_id':plan.run_id, 'attempt_id':plan.attempt_id,
            'carrier_lock_key':'slurp', 'actual':{'calls':0, 'audio_seconds':0.0}}))
    replay = NativeContactReplay(data, original, current, arm)
    target = asdict(current)
    if allowance is not None: target['owner_allowance'] = allowance
    recovery = pin('recovery', {'schema':'saea-native-recovery-v1', 'source':replay.source_binding(), 'target_plan':target})
    resume = data / 'runs' / current.run_id / current.attempt_id / 'resume-declaration.json'; resume.parent.mkdir(parents=True)
    resume.write_text(json.dumps({'schema':'saea-resume-declaration-v1', 'run_id':current.run_id,
        'attempt_id':current.attempt_id, 'native_recovery_receipt':recovery,
        'predecessors':[{'attempt_id':original.attempt_id, 'mode':'native-contact-journal',
            'pin_sha256':replay.source_binding()['attempt_sha256']}]}))
    lease = {'schema':'saea-generation-lease-v1', 'run_id':current.run_id, 'attempt_id':current.attempt_id,
        'pid':123, 'study_head':'a' * 40, 'started_at':'2026-11-02T00:00:01.221861+00:00'}
    lock = data / 'locks/flight.lock'; lock.parent.mkdir(); lock.write_text(json.dumps(lease))
    lock.with_name('generation.guard').touch()
    started = {'attempt_id':current.attempt_id, 'recovery':recovery, 'rows':13078, 'study_commit':'a' * 40}
    container = [{'Id':'b' * 64, 'Name':'/owned', 'State':{'Status':'exited', 'Running':False, 'Pid':0,
        'StartedAt':'2026-11-02T00:00:00Z', 'FinishedAt':'2026-11-02T00:00:10Z'},
        'Config':{'Env':['SAEA_OBSERVABILITY_RUN_ID=' + current.run_id,
            'SAEA_OBSERVABILITY_ATTEMPT_ID=' + current.attempt_id]}}]
    registered = {'execution_plan':asdict(original)}
    if allowance is not None: registered['owner_allowance'] = allowance
    config = {'schema':m.SCHEMA, 'repo':str(repo), 'data':str(data), 'out':str(data / 'work/codex/zero-v2'),
        'source_plan':pin('registered', registered), 'baseline':pin('baseline', {'arm':arm,
            'live_phase_fit':{'run_id':current.run_id + '-FIT', 'attempt_id':'fit-v1'}}),
        'started':pin('started', started), 'container_inspect':pin('container', container),
        'container_id':'b' * 64, 'container_name':'owned', 'lease_sha256':m.digest(lock.read_bytes()),
        'target_attempt':'generation-v3', 'arm':arm, 'fit':{'run_id':current.run_id + '-FIT', 'attempt_id':'fit-v1'},
        'ports':[8200,8201]}
    return config, lock, pin, started, recovery, allowance


def test_resumed_reconciliation_uses_actual_lease_without_rewriting_started(tmp_path, monkeypatch):
    config, lock, _, started, _, _ = resumed_fixture(tmp_path, monkeypatch)
    evidence = Path(config['started']['path']); before = evidence.read_bytes(); original = lock.read_bytes()
    pids = []
    result = m.reconcile(config, observe=lambda ports:{'queues':'empty'},
        pid_exists=lambda pid:pids.append(pid) or False)
    assert result['source']['plan']['attempt_id'] == 'generation-v2'
    assert result['source']['empty_ancestry']['source']['plan']['attempt_id'] == 'generation-v1'
    assert result['worker_identity']['identity_source'] == 'generation-lease' and pids == [123,123]
    assert result['worker_identity']['time'] == datetime_timestamp('2026-11-02T00:00:01.221861+00:00')
    assert evidence.read_bytes() == before and 'pid' not in started and 'time' not in started
    assert Path(result['preserved_lease_path']).read_bytes() == original
    assert not (Path(config['data']) / 'attempts/SAEA-ZERO-RESUMED/generation-v3.json').exists()


def datetime_timestamp(value):
    from datetime import datetime
    return datetime.fromisoformat(value).timestamp()


def test_resumed_final_report_keeps_registered_allowance(tmp_path, monkeypatch):
    config, _, _, _, _, allowance = resumed_fixture(tmp_path, monkeypatch, final_report=True)
    result = m.reconcile(config, observe=lambda ports:{}, pid_exists=lambda pid:False)
    assert result['source']['plan']['owner_allowance'] == allowance
    assert result['source']['empty_ancestry']['source']['plan']['owner_allowance'] == allowance


def test_resumed_tampering_and_unknown_custody_preserve_lock(tmp_path, monkeypatch):
    import pytest
    faults = ('pin', 'target', 'protocol', 'ancestry', 'time', 'pid', 'head', 'native', 'fit', 'ancestor-prepared')
    for fault in faults:
        root = tmp_path / fault; root.mkdir()
        config, lock, pin, started, recovery, _ = resumed_fixture(root, monkeypatch)
        if fault in ('pin','target','protocol'):
            doc = json.loads(Path(recovery['path']).read_bytes())
            if fault == 'target':doc['target_plan']['attempt_id'] = 'generation-foreign'
            else:doc['target_plan']['protocol_hash'] = 'c' * 64
            new = pin('recovery', doc)
            if fault != 'pin':
                started['recovery'] = new; config['started'] = pin('started', started)
        elif fault == 'ancestry':
            p = Path(config['data']) / 'runs/SAEA-ZERO-RESUMED/generation-v2/resume-declaration.json'; p.write_text('{}')
        elif fault in ('time','pid','head'):
            lease = json.loads(lock.read_bytes())
            lease.update({'time':{'started_at':'2026-11-02T00:00:20+00:00'},
                'pid':{'pid':0}, 'head':{'study_head':'c' * 40}}[fault])
            lock.write_text(json.dumps(lease)); config['lease_sha256'] = m.digest(lock.read_bytes())
        else:
            rel = {'native':'contact-journals/SAEA-ZERO-RESUMED/generation-v2/unknown.intent.json',
                'fit':'prepared/SAEA-ZERO-RESUMED-FIT/fit-v1/unknown.endpoint.json',
                'ancestor-prepared':'prepared/SAEA-ZERO-RESUMED/generation-v1/unknown.endpoint.json'}[fault]
            p = Path(config['data']) / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')
        before = lock.read_bytes()
        with pytest.raises(ValueError):
            m.reconcile(config, observe=lambda ports:{}, pid_exists=lambda pid:False)
        assert lock.read_bytes() == before and not Path(config['out']).exists()


def test_resumed_evidence_rechecked_after_quiescence(tmp_path, monkeypatch):
    import pytest
    config, lock, _, _, recovery, _ = resumed_fixture(tmp_path, monkeypatch)
    before = lock.read_bytes()
    def observe(ports):
        Path(recovery['path']).write_text('{}'); return {}
    with pytest.raises(ValueError, match='changed'):
        m.reconcile(config, observe=observe, pid_exists=lambda pid:False)
    assert lock.read_bytes() == before and not Path(config['out']).exists()
