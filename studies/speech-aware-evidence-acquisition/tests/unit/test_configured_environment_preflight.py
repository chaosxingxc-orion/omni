import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

path = Path(__file__).parents[2] / 'scripts/environment/configured_environment_preflight.py'
spec = importlib.util.spec_from_file_location('environment_preflight', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class EnvironmentPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'config.json'
        self.output = self.root / 'receipt'
        self.doc = {'schema': m.SCHEMA, 'timeout_seconds': 5, 'required_environment': {},
                    'interpreters': [{'name': 'cpu', 'python': sys.executable, 'environment_key': None,
                                      'imports': ['json'], 'distributions': {}}]}

    def execute(self):
        self.path.write_text(json.dumps(self.doc))
        return m.execute(self.path, self.output)

    def assert_failed(self):
        result = self.execute()
        self.assertFalse(result['ok'])
        self.assertEqual(result['new_model_calls'], 0)
        self.assertTrue((self.output / 'result.json').exists())
        self.assertFalse((self.output / 'ready.json').exists())
        return result

    def test_success_receipt_binds_config_and_result(self):
        result = self.execute()
        self.assertTrue(result['ok'])
        self.assertEqual(result['checks'][0]['executable'], sys.executable)
        ready = json.loads((self.output / 'ready.json').read_text())
        self.assertEqual(ready['config_sha256'], m.digest(self.doc))
        self.assertEqual(ready['result_sha256'], m.hashlib.sha256((self.output / 'result.json').read_bytes()).hexdigest())

    def test_repeated_output_preserves_existing_receipt_without_probe(self):
        self.execute()
        before = (self.output / 'result.json').read_bytes()
        with patch.object(m.subprocess, 'run') as run:
            with self.assertRaises(FileExistsError):
                self.execute()
            run.assert_not_called()
        self.assertEqual((self.output / 'result.json').read_bytes(), before)

    def test_missing_python(self):
        self.doc['interpreters'][0]['python'] = str(self.root / 'missing-python')
        self.assert_failed()

    def test_required_file_checks_actual_bytes(self):
        source = self.root / 'source.json'
        source.write_bytes(b'frozen knowledge')
        self.doc['required_files'] = [{'path': str(source), 'sha256': m.hashlib.sha256(source.read_bytes()).hexdigest()}]
        result = self.execute()
        self.assertTrue(result['ok'])
        self.assertEqual(result['checks'][0]['actual_sha256'], self.doc['required_files'][0]['sha256'])

    def test_missing_required_file_blocks_imports_and_ready(self):
        self.doc['required_files'] = [{'path': str(self.root / 'absent'), 'sha256': 'a' * 64}]
        with patch.object(m.subprocess, 'run') as run:
            self.assert_failed()
            run.assert_not_called()

    def test_changed_required_file_blocks_ready(self):
        source = self.root / 'changed'
        source.write_bytes(b'changed')
        self.doc['required_files'] = [{'path': str(source), 'sha256': 'a' * 64}]
        self.assert_failed()

    def test_invalid_required_file_declarations(self):
        valid = {'path': str(self.root / 'source'), 'sha256': 'a' * 64}
        for files in ([valid, valid], [{'path': 'relative', 'sha256': 'a' * 64}],
                      [{'path': '/tmp/../source', 'sha256': 'a' * 64}],
                      [{'path': '/source', 'sha256': 'A' * 64}], {'path': '/source'}):
            with self.subTest(files=files):
                self.doc['required_files'] = files
                with self.assertRaises(ValueError):
                    self.execute()
                self.assertFalse(self.output.exists())

    def test_missing_module(self):
        self.doc['interpreters'][0]['imports'] = ['saea_nonexistent_dependency_for_test']
        self.assert_failed()

    def test_exact_distribution_version_mismatch(self):
        packages = self.root / 'packages'
        metadata = packages / 'saea_probe_fixture-1.0.dist-info'
        metadata.mkdir(parents=True)
        (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: saea-probe-fixture\nVersion: 1.0\n')
        self.doc['interpreters'][0]['distributions'] = {'saea-probe-fixture': '2.0'}
        with patch.dict(os.environ, {'PYTHONPATH': str(packages)}):
            result = self.assert_failed()
        check = result['checks'][0]['checks'][-1]
        self.assertEqual(check['actual'], '1.0')
        self.assertEqual(check['expected'], '2.0')

    def test_missing_environment_selection_stops_before_probes(self):
        self.doc['required_environment'] = {'SAEA_SLURP_OFFICIAL_PYTHON': sys.executable}
        self.doc['interpreters'][0]['environment_key'] = 'SAEA_SLURP_OFFICIAL_PYTHON'
        with patch.dict(os.environ, {}, clear=True), patch.object(m.subprocess, 'run') as run:
            self.assert_failed()
            run.assert_not_called()

    def test_matching_environment_selection(self):
        self.doc['required_environment'] = {'SAEA_SLURP_OFFICIAL_PYTHON': sys.executable}
        self.doc['interpreters'][0]['environment_key'] = 'SAEA_SLURP_OFFICIAL_PYTHON'
        with patch.dict(os.environ, {'SAEA_SLURP_OFFICIAL_PYTHON': sys.executable}):
            self.assertTrue(self.execute()['ok'])

    def test_config_cannot_probe_different_python_from_selection(self):
        self.doc['required_environment'] = {'SAEA_SLURP_OFFICIAL_PYTHON': '/another/python'}
        self.doc['interpreters'][0]['environment_key'] = 'SAEA_SLURP_OFFICIAL_PYTHON'
        with self.assertRaisesRegex(ValueError, 'selection must match'):
            self.execute()
        self.assertFalse(self.output.exists())

    def test_timeout_does_not_retry(self):
        with patch.object(m.subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 5)) as run:
            self.assert_failed()
            self.assertEqual(run.call_count, 1)

    def test_invalid_relative_interpreter_creates_no_receipt(self):
        self.doc['interpreters'][0]['python'] = 'python'
        with self.assertRaisesRegex(ValueError, 'absolute interpreter'):
            self.execute()
        self.assertFalse(self.output.exists())

    def test_cli_failure_exit_code_and_durable_evidence(self):
        self.doc['interpreters'][0]['imports'] = ['saea_nonexistent_dependency_for_test']
        self.path.write_text(json.dumps(self.doc))
        result = subprocess.run([sys.executable, str(path), '--config', str(self.path),
                                 '--output', str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)['ok'])
        self.assertTrue((self.output / 'result.json').exists())
        self.assertFalse((self.output / 'ready.json').exists())

    def test_transitive_import_failure_is_detected(self):
        modules = self.root / 'modules'
        modules.mkdir()
        (modules / 'probe_dependent.py').write_text('import saea_nonexistent_transitive_dependency\n')
        self.doc['interpreters'][0]['imports'] = ['probe_dependent']
        with patch.dict(os.environ, {'PYTHONPATH': str(modules)}):
            result = self.assert_failed()
        self.assertEqual(result['checks'][0]['checks'][0]['error_type'], 'ModuleNotFoundError')
        self.assertFalse((modules / '__pycache__').exists())

    def test_existing_launcher_input_gate_blocks_absent_and_changed_receipts(self):
        launcher_path = path.with_name('configured_job_launch.py')
        launcher_spec = importlib.util.spec_from_file_location('existing_launcher', launcher_path)
        launcher = importlib.util.module_from_spec(launcher_spec)
        launcher_spec.loader.exec_module(launcher)
        self.execute()
        ready = self.output / 'ready.json'
        pin = {'path': str(ready), 'sha256': launcher.digest(ready)}
        doc = {'repo_path': str(self.root), 'expected_commit': 'b' * 40,
               'required_inputs': [pin], 'mounts': []}
        def git(argv, **kwargs):
            return SimpleNamespace(returncode=0, stdout='b' * 40 if argv[-1] == 'HEAD' else '', stderr='')
        with patch.object(launcher, 'run', side_effect=git):
            launcher.preflight(doc)
            original = ready.read_bytes()
            ready.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'absent or changed'):
                launcher.preflight(doc)
            ready.unlink()
            with self.assertRaisesRegex(ValueError, 'absent or changed'):
                launcher.preflight(doc)
            ready.write_bytes(original)
            launcher.preflight(doc)


if __name__ == '__main__':
    unittest.main()
