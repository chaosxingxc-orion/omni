import copy
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).parents[2] / 'scripts/environment/configured_remote_archive.py'
spec = importlib.util.spec_from_file_location('configured_archive', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'private'
        self.output.mkdir(mode=0o700)
        self.doc = {'schema': m.SCHEMA, 'host': 'npu2', 'container_name': 'saea-example-generation-v1',
            'remote_root': m.REMOTE_BASE + '/example', 'relative_paths': ['control', 'runs'],
            'shared_attempt_paths': [m.REMOTE_BASE + '/shared/attempts/RUN/generation-v1.json',
                                     m.REMOTE_BASE + '/shared/attempts/RUN-FIT/fit-v1.json'],
            'local_output_root': str(self.output), 'interval_seconds': 600}
        self.config = self.root / 'config.json'
        self.observed = {'container_id': 'a' * 64, 'state': 'running', 'running': True,
                         'exit_code': 0, 'finished_at': ''}

    def load(self):
        self.config.write_text(json.dumps(self.doc))
        return m.load_config(self.config)

    def fixture(self, attempt=b'{"state":"FINALIZED"}', missing_member=False):
        spec = m.specification(self.doc)
        included = [spec['paths'][0], spec['attempts'][0]]
        meta = {'included': included, 'absent': [spec['paths'][1], spec['attempts'][1]],
                'attempts_included': [spec['attempts'][0]], 'attempts_absent': [spec['attempts'][1]]}
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as archive:
            directory = tarfile.TarInfo(included[0]); directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
            if not missing_member:
                member = tarfile.TarInfo(included[1]); member.size = len(attempt)
                archive.addfile(member, io.BytesIO(attempt))
        return buffer.getvalue(), m.PREFIX + json.dumps(meta) + chr(10)

    def stream(self, raw, stderr, status):
        class Process:
            def __init__(self): self.stdout = io.BytesIO(raw)
            def wait(self): return status
            def poll(self): return status
            def terminate(self): raise AssertionError('already terminal')
        def popen(argv, stdout, stderr):
            self.assertEqual(argv[:4], ['ssh', '-oBatchMode=yes', '-oConnectTimeout=30', 'npu2'])
            stderr.write(self.stderr_bytes)
            return Process()
        self.stderr_bytes = stderr.encode()
        return patch.object(m.subprocess, 'Popen', side_effect=popen)

    def test_restart_binds_original_container_and_rejects_replacement(self):
        self.load()
        with (patch.object(m, 'inspect', return_value=self.observed),
              patch.object(m, 'snapshot', return_value={'status': 'complete'}) as snapshot,
              patch('builtins.print')):
            m.execute(self.config, 'once')
            m.execute(self.config, 'once')
        self.assertEqual(snapshot.call_count, 2)
        with (patch.object(m, 'inspect', return_value={**self.observed, 'container_id': 'b' * 64}),
              patch.object(m, 'snapshot') as snapshot):
            with self.assertRaisesRegex(ValueError, 'replaced'):
                m.execute(self.config, 'once')
            snapshot.assert_not_called()

    def test_expected_id_and_corrupt_or_drifted_persistent_identity(self):
        self.doc['expected_container_id'] = 'b' * 64
        self.load()
        with self.assertRaisesRegex(ValueError, 'expected ID'):
            m.bind_container(self.doc, self.observed)
        self.doc['expected_container_id'] = 'a' * 64
        m.bind_container(self.load(), self.observed)
        path = self.output / (self.doc['container_name'] + '.identity.json')
        blob = path.read_bytes()
        self.doc['relative_paths'] = ['control']
        with self.assertRaisesRegex(ValueError, 'identity/config'):
            m.bind_container(self.doc, self.observed)
        self.assertEqual(blob, path.read_bytes())
        path.write_text('broken')
        with self.assertRaises(ValueError):
            m.bind_container(self.doc, self.observed)
        self.assertEqual(path.read_text(), 'broken')

    def test_expected_id_schema_and_identity_symlink(self):
        for value in ('short', 'A' * 64, None, 42):
            self.doc['expected_container_id'] = value
            with self.assertRaises(ValueError): self.load()
        self.doc.pop('expected_container_id')
        path = self.output / (self.doc['container_name'] + '.identity.json')
        path.symlink_to(self.root / 'absent')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            m.bind_container(self.load(), self.observed)

    def test_complete_stream_hash_attempts_and_absence(self):
        raw, stderr = self.fixture()
        with self.stream(raw, stderr, 0):
            result = m.snapshot(self.load(), self.observed)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['sha256'], m.hashlib.sha256(raw).hexdigest())
        self.assertEqual(result['bytes'], len(raw))
        self.assertEqual(len(result['attempt_records_verified']), 1)
        self.assertEqual(len(result['attempts_absent']), 1)
        self.assertFalse(result['automatic_restore_authority'])
        self.assertFalse(result['consistent_snapshot'])
        self.assertTrue((Path(result['directory']) / 'intent.json').is_file())

    def test_tar_status_one_preserves_verified_changed_snapshot(self):
        raw, stderr = self.fixture()
        with self.stream(raw, stderr + 'tar: file changed as we read it' + chr(10), 1):
            result = m.snapshot(self.load(), self.observed)
        self.assertEqual(result['status'], 'changed-during-read')
        self.assertTrue(result['inventory_verified'])
        self.assertEqual((Path(result['directory']) / 'state.tar').read_bytes(), raw)

    def test_bad_partial_stream_is_preserved(self):
        raw = b'partial tar bytes'
        with self.stream(raw, 'ssh: connection lost' + chr(10), 255):
            result = m.snapshot(self.load(), self.observed)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failure_kind'], 'transport')
        self.assertFalse(result['inventory_verified'])
        self.assertEqual((Path(result['directory']) / 'state.tar').read_bytes(), raw)
        self.assertTrue((Path(result['directory']) / 'result.json').is_file())

    def test_inspect_transport_retries_same_identity_and_persists_failure(self):
        self.load()
        running = self.observed
        terminal = {**running, 'state': 'exited', 'running': False}
        with (patch.object(m, 'inspect', side_effect=[running, m.TransportError('timeout'), terminal]),
              patch.object(m, 'snapshot', return_value={'status': 'complete'}) as snapshot,
              patch.object(m, 'wait_event'), patch.object(m.time, 'sleep') as sleep,
              patch('builtins.print')):
            m.execute(self.config, 'watch')
        sleep.assert_called_once_with(600)
        self.assertEqual([c.args[1]['container_id'] for c in snapshot.call_args_list], ['a' * 64] * 2)
        event, = self.output.glob('transport-*.json')
        self.assertEqual(json.loads(event.read_text())['container_id'], 'a' * 64)

    def test_inspect_classifies_only_ssh_failures(self):
        from types import SimpleNamespace
        for failure in (subprocess.TimeoutExpired('ssh', 60), SimpleNamespace(returncode=255, stderr='lost')):
            with self.subTest(failure=failure), patch.object(m.subprocess, 'run', side_effect=[failure]):
                with self.assertRaises(m.TransportError): m.inspect(self.doc)
        with patch.object(m.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr='bad')):
            with self.assertRaisesRegex(ValueError, 'inspection failed'): m.inspect(self.doc)
        self.load()
        with patch.object(m, 'inspect', side_effect=m.TransportError('lost')), patch.object(m.time, 'sleep') as sleep:
            with self.assertRaises(m.TransportError): m.execute(self.config, 'once')
        sleep.assert_not_called()

    def test_snapshot_transport_retry_preserves_partial_and_rechecks_identity(self):
        self.load()
        raw, stderr = self.fixture()
        terminal = {**self.observed, 'state': 'exited', 'running': False}
        streams = iter([(b'partial', 'lost', 255), (raw, stderr, 0)])
        real_snapshot = m.snapshot
        def take_snapshot(doc, observed):
            with self.stream(*next(streams)):
                return real_snapshot(doc, observed)
        with (patch.object(m, 'inspect', side_effect=[self.observed, terminal]),
              patch.object(m, 'snapshot', side_effect=take_snapshot),
              patch.object(m.time, 'sleep') as sleep, patch('builtins.print')):
            result = m.execute(self.config, 'watch')
        sleep.assert_called_once_with(600)
        self.assertEqual(result['status'], 'complete')
        records = [json.loads(p.read_text()) for p in self.output.glob('*/result.json')]
        failed, = [r for r in records if r['status'] == 'failed']
        self.assertEqual(failed['failure_kind'], 'transport')
        partial, = [p for p in self.output.glob('*/state.tar') if p.read_bytes() == b'partial']
        self.assertTrue(partial.is_file())

    def test_failed_validation_and_once_transport_do_not_retry(self):
        self.load()
        for mode, record in [('watch', {'status': 'failed'}),
                             ('once', {'status': 'failed', 'failure_kind': 'transport'})]:
            with (self.subTest(mode=mode), patch.object(m, 'inspect', return_value=self.observed),
                  patch.object(m, 'snapshot', return_value=record) as snapshot,
                  patch.object(m.time, 'sleep') as sleep, patch('builtins.print')):
                self.assertEqual(m.execute(self.config, mode), record)
                snapshot.assert_called_once()
                sleep.assert_not_called()

    def test_retry_rejects_replacement_before_second_snapshot(self):
        self.load()
        replacement = {**self.observed, 'container_id': 'b' * 64}
        with (patch.object(m, 'inspect', side_effect=[self.observed, replacement]),
              patch.object(m, 'snapshot', return_value={'status': 'failed', 'failure_kind': 'transport'}) as snapshot,
              patch.object(m.time, 'sleep') as sleep, patch('builtins.print')):
            with self.assertRaisesRegex(ValueError, 'replaced'):
                m.execute(self.config, 'watch')
        sleep.assert_called_once_with(600)
        snapshot.assert_called_once()

    def test_exit_255_never_marks_even_valid_tar_complete(self):
        raw, stderr = self.fixture()
        with self.stream(raw, stderr, 255):
            result = m.snapshot(self.load(), self.observed)
        self.assertEqual((result['status'], result['failure_kind']), ('failed', 'transport'))
        self.assertFalse(result['inventory_verified'])

    def test_snapshot_timeout_is_preserved_and_retryable(self):
        with patch.object(m.subprocess, 'Popen', side_effect=subprocess.TimeoutExpired('ssh', 60)):
            result = m.snapshot(self.load(), self.observed)
        self.assertEqual((result['status'], result['failure_kind']), ('failed', 'transport'))
        self.assertTrue((Path(result['directory']) / 'result.json').is_file())

    def test_bad_attempt_json_and_missing_member_refused(self):
        for raw, stderr in (self.fixture(attempt=b'broken'), self.fixture(missing_member=True)):
            with self.subTest(raw_length=len(raw)), self.stream(raw, stderr, 0):
                result = m.snapshot(self.load(), self.observed)
            self.assertEqual(result['status'], 'failed')

    def test_missing_or_ambiguous_inventory_refused(self):
        raw, stderr = self.fixture()
        for text in ('', stderr + stderr):
            with self.subTest(text=text), self.stream(raw, text, 0):
                self.assertEqual(m.snapshot(self.load(), self.observed)['status'], 'failed')

    def test_repeat_snapshot_uses_distinct_preserved_directories(self):
        raw, stderr = self.fixture()
        with self.stream(raw, stderr, 0):
            first = m.snapshot(self.load(), self.observed)
            second = m.snapshot(self.doc, self.observed)
        self.assertNotEqual(first['directory'], second['directory'])
        self.assertEqual((Path(first['directory']) / 'state.tar').read_bytes(), raw)

    def test_unapproved_host_path_escape_and_interval_refused(self):
        original = copy.deepcopy(self.doc)
        for key, value in [('host', 'npu'), ('container_name', 'name; touch /tmp/x'),
                           ('remote_root', '/var/lib'), ('relative_paths', ['../escape']),
                           ('relative_paths', ['/absolute']), ('relative_paths', ['control', 'control/x']),
                           ('shared_attempt_paths', ['/etc/attempts/a.json']), ('interval_seconds', 30)]:
            with self.subTest(key=key, value=value):
                self.doc = {**original, key: value}
                with self.assertRaises(ValueError): self.load()

    def test_shell_quoting_preserves_literal_paths(self):
        self.doc['relative_paths'] = ["control/a'$(touch ignored); spaced"]
        command = m.archive_command(self.load())
        args = shlex.split(command[-1])
        self.assertEqual(args[:4], ['sudo', '-n', 'python3', '-c'])
        self.assertEqual(json.loads(args[-1]), m.specification(self.doc))

    def test_remote_program_local_tar_inventory_and_read_only_bytes(self):
        base = self.root / 'remote'; (base / 'job/control').mkdir(parents=True)
        (base / 'job/control/result.json').write_text('{"ok": true}')
        attempt = base / 'shared/attempts/RUN/a.json'; attempt.parent.mkdir(parents=True)
        attempt.write_text('{"state": "FINALIZED"}')
        spec = {'base': str(base), 'owned_root': 'job', 'paths': ['job/control', 'job/absent', 'shared/attempts/RUN/a.json',
                                            'shared/attempts/RUN/missing.json'],
                'attempts': ['shared/attempts/RUN/a.json', 'shared/attempts/RUN/missing.json']}
        before = attempt.read_bytes()
        result = subprocess.run([sys.executable, '-c', m.REMOTE, json.dumps(spec)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        archive = self.root / 'local.tar'; archive.write_bytes(result.stdout)
        verified = m.verify_inventory(archive, result.stderr.decode(), spec)
        self.assertEqual(verified['attempt_records_verified'], [spec['attempts'][0]])
        self.assertEqual(attempt.read_bytes(), before)

    def test_remote_program_refuses_root_symlink_escape(self):
        base = self.root / 'remote'; base.mkdir()
        (base / 'control').symlink_to(self.root)
        result = subprocess.run([sys.executable, '-c', m.REMOTE, json.dumps({
            'base': str(base), 'owned_root': 'control', 'paths': ['control'], 'attempts': []})], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'')

    def test_remote_program_refuses_owned_path_alias_to_other_job(self):
        base = self.root / 'remote'; (base / 'job').mkdir(parents=True)
        (base / 'other/control').mkdir(parents=True)
        (base / 'job/alias').symlink_to(base / 'other')
        result = subprocess.run([sys.executable, '-c', m.REMOTE, json.dumps({
            'base': str(base), 'owned_root': 'job', 'paths': ['job/alias/control'], 'attempts': []})], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'')

    def test_wait_is_remote_bounded_and_handles_terminal_or_timeout(self):
        from types import SimpleNamespace
        for status in (0, 124):
            with (patch.object(m.subprocess, 'run', return_value=SimpleNamespace(returncode=status, stderr='')) as run,
                  patch.object(m.time, 'sleep'), patch.object(m.time, 'monotonic', return_value=0)):
                m.wait_event(self.doc)
            args = shlex.split(run.call_args.args[0][-1])
            self.assertEqual(args, ['sudo', '-n', 'timeout', '--signal=TERM', '--kill-after=5s', '600', 'docker', 'wait', self.doc['container_name']])

    def test_wait_transport_failure_obeys_cadence_without_claiming_terminal(self):
        from types import SimpleNamespace
        for status in (137, 255):
            with (self.subTest(status=status),
                  patch.object(m.subprocess, 'run', return_value=SimpleNamespace(returncode=status, stderr='lost')),
                  patch.object(m.time, 'monotonic', side_effect=[10, 20]),
                  patch.object(m.time, 'sleep') as sleep, patch('builtins.print') as output):
                m.wait_event(self.doc)
                sleep.assert_called_once_with(590)
                self.assertFalse(json.loads(output.call_args.args[0])['terminal_verified'])

    def test_local_wait_timeout_is_observation_failure_and_rechecks_identity(self):
        self.load()
        with (patch.object(m, 'inspect', side_effect=[self.observed, {**self.observed, 'container_id': 'b' * 64}]),
              patch.object(m, 'snapshot', return_value={'status': 'complete'}) as snapshot,
              patch.object(m.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ssh', 630)),
              patch.object(m.time, 'monotonic', side_effect=[0, 630]),
              patch.object(m.time, 'sleep') as sleep, patch('builtins.print')):
            with self.assertRaisesRegex(ValueError, 'replaced'):
                m.execute(self.config, 'watch')
        self.assertEqual(snapshot.call_count, 1)
        sleep.assert_not_called()

    def test_unexpected_wait_failure_remains_fatal(self):
        from types import SimpleNamespace
        with (patch.object(m.subprocess, 'run', return_value=SimpleNamespace(returncode=125, stderr='invalid command')),
              patch.object(m.time, 'sleep') as sleep):
            with self.assertRaisesRegex(ValueError, 'wait failed'):
                m.wait_event(self.doc)
        sleep.assert_not_called()

    def test_inventory_cannot_add_unconfigured_or_false_absence(self):
        raw, stderr = self.fixture()
        meta = json.loads(stderr[len(m.PREFIX):])
        meta['absent'].append('unconfigured/path')
        with self.stream(raw, m.PREFIX + json.dumps(meta), 0):
            self.assertEqual(m.snapshot(self.load(), self.observed)['status'], 'failed')

    def test_watch_archives_running_then_terminal_without_sleep_poll(self):
        self.load()
        terminal = {**self.observed, 'state': 'exited', 'running': False, 'exit_code': 1}
        with (patch.object(m, 'inspect', side_effect=[self.observed, terminal]),
              patch.object(m, 'snapshot', return_value={'status': 'complete'}) as snapshot,
              patch.object(m, 'wait_event') as wait, patch('builtins.print')):
            m.execute(self.config, 'watch')
        self.assertEqual(snapshot.call_count, 2)
        wait.assert_called_once_with(self.doc)

    def test_watch_refuses_replaced_container(self):
        self.load()
        with (patch.object(m, 'inspect', side_effect=[self.observed, {**self.observed, 'container_id': 'b' * 64}]),
              patch.object(m, 'snapshot', return_value={'status': 'complete'}) as snapshot,
              patch.object(m, 'wait_event'), patch('builtins.print')):
            with self.assertRaisesRegex(ValueError, 'replaced'): m.execute(self.config, 'watch')
        self.assertEqual(snapshot.call_count, 1)

    def test_private_output_required_before_remote_action(self):
        self.load(); self.output.chmod(0o755)
        with patch.object(m, 'inspect') as inspect:
            with self.assertRaisesRegex(ValueError, 'private'): m.execute(self.config, 'once')
        inspect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
