import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).parents[2] / 'scripts/environment/configured_preflight_binding.py'
spec = importlib.util.spec_from_file_location('preflight_binding', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class BindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'owned'
        self.root.mkdir()
        self.code = self.root / 'code'
        self.code.mkdir()
        shutil.copyfile(path.with_name('configured_environment_preflight.py'), self.code / 'configured_environment_preflight.py')
        self.config = self.code / 'probe.json'
        self.config.write_text(json.dumps({'schema': m.probe.SCHEMA, 'timeout_seconds': 5,
            'required_environment': {}, 'interpreters': [{'name': 'business', 'python': sys.executable,
            'environment_key': None, 'imports': ['json'], 'distributions': {}}]}))
        self.out = self.root / 'out'
        self.out.mkdir()
        m.probe.execute(self.config, self.out / 'attempt')
        self.ready = self.out / 'attempt/ready.json'
        self.control = self.root / 'control'
        self.control.mkdir()
        self.source_path = self.root / 'source.json'
        self.successor_path = self.root / 'successor.json'
        self.binding = self.root / 'binding.json'
        self.source = {'schema': m.launcher.SCHEMA, 'container_name': 'cpu-preflight', 'image': 'sha256:' + 'a' * 64,
            'repo_path': str(self.code), 'expected_commit': 'b' * 40, 'receipt_dir': str(self.control),
            'argv': [sys.executable, '/code/configured_environment_preflight.py', '--config', '/code/probe.json', '--output', '/preflight/attempt'],
            'working_directory': '/tmp', 'network': 'none', 'pid_host': False, 'cpuset_cpus': '0-1',
            'environment': {}, 'mounts': [{'source': str(self.code), 'target': '/code', 'readonly': True},
                                        {'source': str(self.out), 'target': '/preflight', 'readonly': False}],
            'create_directories': [str(self.control)], 'allowed_create_roots': [str(self.root)],
            'required_inputs': [{'path': str(self.config), 'sha256': m.sha(self.config)}], 'monitor_interval_seconds': 600}
        self.successor = copy.deepcopy(self.source)
        self.successor.update(container_name='business', argv=[sys.executable, '/code/business.py'], network='host',
                              receipt_dir=str(self.root / 'next-control'))
        self.successor['mounts'] = [copy.deepcopy(self.source['mounts'][0]), {'source': str(self.out), 'target': '/results', 'readonly': False}]
        self.successor['required_inputs'] = [m.pin(self.ready)]
        self.write()
        self.status = patch.object(m.launcher, 'status', return_value={'state': 'exited', 'running': False, 'exit_code': 0, 'container_id': 'owned-cpu-container'})
        self.status_mock = self.status.start()
        self.addCleanup(self.status.stop)

    def write(self):
        self.source_path.write_text(json.dumps(self.source))
        self.successor_path.write_text(json.dumps(self.successor))
        (self.control / 'launch-intent.json').write_text(json.dumps({'config_sha256': m.probe.digest(self.source),
                                                                   'runner_sha256': m.sha(m.launcher.__file__)}))

    def bind(self):
        self.write()
        return m.bind(self.source_path, self.ready, self.successor_path, self.binding)

    def test_compatible_binding_allows_business_network_and_output_difference(self):
        self.bind()
        self.assertEqual(m.verify(self.binding, self.successor_path), self.successor)
        with patch.object(m.launcher, 'execute', return_value={'state': 'unresolved-launch-intent', 'automatic_retry': False}) as execute:
            result = m.execute(self.binding, self.successor_path, 'launch')
            self.assertEqual(result['state'], 'unresolved-launch-intent')
            execute.assert_called_once_with(self.successor_path, 'launch')

    def test_environment_mismatch(self):
        self.successor['environment']['PYTHONPATH'] = '/other'
        with self.assertRaisesRegex(ValueError, 'environment drift'):
            self.bind()

    def test_image_mismatch(self):
        self.successor['image'] = 'sha256:' + 'c' * 64
        with self.assertRaisesRegex(ValueError, 'image drift'):
            self.bind()

    def test_unprobed_interpreter(self):
        self.successor['argv'][0] = '/another/python'
        with self.assertRaisesRegex(ValueError, 'not probed'):
            self.bind()

    def test_dependency_source_drift(self):
        self.successor['mounts'][0]['source'] = str(self.out)
        with self.assertRaisesRegex(ValueError, 'mount drift'):
            self.bind()

    def test_dependency_readonly_drift(self):
        self.successor['mounts'][0]['readonly'] = False
        with self.assertRaisesRegex(ValueError, 'mount drift'):
            self.bind()

    def test_new_nested_dependency_mount(self):
        self.successor['mounts'].append({'source': str(self.out), 'target': '/code/package', 'readonly': True})
        with self.assertRaisesRegex(ValueError, 'shadows dependency'):
            self.bind()

    def test_new_image_interpreter_mount(self):
        self.successor['mounts'].append({'source': str(self.out), 'target': '/usr/lib', 'readonly': True})
        with self.assertRaisesRegex(ValueError, 'shadows dependency'):
            self.bind()

    def test_absent_readiness(self):
        self.ready.unlink()
        with self.assertRaisesRegex(ValueError, 'readiness is absent'):
            self.bind()

    def test_result_tampering(self):
        self.ready.with_name('result.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'result absent or changed'):
            self.bind()

    def test_source_configuration_not_pinned(self):
        self.source['required_inputs'] = []
        with self.assertRaisesRegex(ValueError, 'must pin probe'):
            self.bind()

    def test_successor_missing_ready_pin(self):
        self.successor['required_inputs'] = []
        with self.assertRaisesRegex(ValueError, 'must pin successful readiness'):
            self.bind()

    def test_source_not_terminal(self):
        self.status_mock.return_value = {'state': 'running', 'running': True, 'exit_code': 0}
        with self.assertRaisesRegex(ValueError, 'not successfully terminal'):
            self.bind()

    def test_launch_intent_mismatch(self):
        (self.control / 'launch-intent.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'launch identity differs'):
            m.bind(self.source_path, self.ready, self.successor_path, self.binding)

    def test_ready_from_another_output_rejected(self):
        foreign = self.root / 'foreign-ready.json'
        shutil.copyfile(self.ready, foreign)
        with self.assertRaisesRegex(ValueError, 'does not belong'):
            m.bind(self.source_path, foreign, self.successor_path, self.binding)

    def test_successor_drift_after_binding_never_invokes_launcher(self):
        self.bind()
        self.successor['image'] = 'sha256:' + 'c' * 64
        self.successor_path.write_text(json.dumps(self.successor))
        with patch.object(m.launcher, 'execute') as execute:
            with self.assertRaisesRegex(ValueError, 'absent or changed'):
                m.execute(self.binding, self.successor_path, 'launch')
            execute.assert_not_called()

    def test_repeated_binding_preserves_existing_bytes(self):
        self.bind()
        before = self.binding.read_bytes()
        with self.assertRaises(FileExistsError):
            self.bind()
        self.assertEqual(self.binding.read_bytes(), before)

    def test_probe_config_drift_after_binding(self):
        self.bind()
        self.config.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'absent or changed'):
            m.verify(self.binding, self.successor_path)

    def test_working_directory_drift(self):
        self.successor['working_directory'] = '/other'
        with self.assertRaisesRegex(ValueError, 'working directory drift'):
            self.bind()

    def test_execution_commit_drift(self):
        self.successor['expected_commit'] = 'c' * 40
        with self.assertRaisesRegex(ValueError, 'repository drift'):
            self.bind()

    def test_failed_readiness_cannot_bind_even_with_matching_result_hash(self):
        result_path = self.ready.with_name('result.json')
        result = json.loads(result_path.read_bytes())
        result['ok'] = False
        result_path.write_text(json.dumps(result))
        ready = json.loads(self.ready.read_bytes())
        ready['result_sha256'] = m.sha(result_path)
        self.ready.write_text(json.dumps(ready))
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            self.bind()

    def test_source_file_change_after_binding(self):
        self.bind()
        self.source_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'absent or changed'):
            m.verify(self.binding, self.successor_path)

    def test_library_module_identity_drift(self):
        self.bind()
        binding = json.loads(self.binding.read_bytes())
        binding['probe_module_sha256'] = '0' * 64
        self.binding.write_text(json.dumps(binding))
        with self.assertRaisesRegex(ValueError, 'implementation identity differs'):
            m.verify(self.binding, self.successor_path)


if __name__ == '__main__':
    unittest.main()
