"""Explicit output semantics retain dependency protection and full config pins."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).parents[2] / 'scripts/environment/configured_preflight_binding.py'
spec = importlib.util.spec_from_file_location('runtime_path_binding', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RuntimePathTests(unittest.TestCase):
    def setUp(self):
        self.examples = path.parents[2] / 'configs/launcher'
        self.source = json.loads((self.examples / 'direct-preflight-job.example.json').read_text())
        self.successor = json.loads((self.examples / 'direct-successor-job.example.json').read_text())
        self.config = m.probe.load_config(self.examples / 'direct-environment-preflight.example.json')
        self.output = next(x for x in self.source['mounts'] if x['target'] == '/preflight')

    def check(self):
        return m.compatibility(self.source, self.successor, self.config, self.output)

    def test_direct_pair_records_exact_runtime_semantics(self):
        result = self.check()
        self.assertEqual(result['runtime_paths'], self.config['runtime_paths'])
        self.assertEqual(result['environment'], self.source['environment'])

    def test_production_direct_layout_with_separate_telemetry_mount(self):
        # Same geometry as the actual Direct pair: code and read-only leaves below
        # the data root, separately mounted writable state, telemetry outside it.
        root = '/shared/omni'
        code = root + '/work/dev/direct/studies/speech-aware-evidence-acquisition'
        for doc in (self.source, self.successor):
            doc['environment']['SPEECHRL_DATA_DIR'] = root
            doc['environment']['SAEA_STAGE_TIMING_PATH'] = '/telemetry/timing-generation-v1.jsonl'
            doc['environment']['PYTHONPATH'] = code + '/src'
            doc['environment']['GIT_CONFIG_VALUE_0'] = code
            for mount in doc['mounts']:
                target = mount['target']
                if target == '/data/work':
                    mount['target'] = root + '/work/codex/direct'
                elif target.startswith('/data/'):
                    mount['target'] = root + target[len('/data'):]
                elif target == '/workspace/studies/speech-aware-evidence-acquisition':
                    mount['target'] = code
        self.successor['mounts'].append({'source': '/srv/omni/jobs/direct-example/telemetry',
                                         'target': '/telemetry', 'readonly': False})
        self.config['required_environment'] = dict(self.source['environment'])
        self.config['runtime_paths'] = {'SPEECHRL_DATA_DIR': {'kind': 'data-root', 'path': root},
            'SAEA_STAGE_TIMING_PATH': {'kind': 'timing-file',
                'path': '/telemetry/timing-generation-v1.jsonl', 'mount': '/telemetry'}}
        self.check()

    def test_missing_timing_mount_refused(self):
        self.successor['mounts'] = [m for m in self.successor['mounts'] if m['target'] != '/data/work']
        with self.assertRaisesRegex(ValueError, 'explicitly mounted'):
            self.check()

    def test_undeclared_runtime_variable_refused(self):
        del self.config['runtime_paths']
        with self.assertRaisesRegex(ValueError, 'explicitly declared'):
            self.check()

    def test_environment_drift_refused(self):
        self.successor['environment']['SAEA_STAGE_TIMING_PATH'] = '/data/work/other.jsonl'
        with self.assertRaisesRegex(ValueError, 'environment drift'):
            self.check()

    def test_dependency_shadow_refused(self):
        self.successor['mounts'].append({'source': '/srv/output', 'target': '/data/models/weights', 'readonly': False})
        with self.assertRaisesRegex(ValueError, 'shadows dependency'):
            self.check()

    def test_data_root_cannot_exempt_import_paths(self):
        for doc in (self.source, self.successor):
            doc['environment']['PYTHONPATH'] = '/data/work/library'
        self.config['required_environment']['PYTHONPATH'] = '/data/work/library'
        with self.assertRaisesRegex(ValueError, 'shadows dependency'):
            self.check()

    def test_timing_mount_must_be_declared_and_writable(self):
        mount = next(x for x in self.successor['mounts'] if x['target'] == '/data/work')
        mount['readonly'] = True
        with self.assertRaisesRegex(ValueError, 'declared writable'):
            self.check()
        mount['readonly'] = False
        self.config['runtime_paths']['SAEA_STAGE_TIMING_PATH']['mount'] = '/data'
        with self.assertRaisesRegex(ValueError, 'declared writable'):
            self.check()

    def test_host_alias_of_dependency_refused(self):
        mount = next(x for x in self.successor['mounts'] if x['target'] == '/data/work')
        mount['source'] = self.source['mounts'][0]['source']
        with self.assertRaisesRegex(ValueError, 'aliases dependency'):
            self.check()

    def test_host_parent_of_dependency_refused(self):
        mount = next(x for x in self.successor['mounts'] if x['target'] == '/data/work')
        mount['source'] = '/srv/omni/jobs/direct-example'
        with self.assertRaisesRegex(ValueError, 'aliases dependency'):
            self.check()

    def test_host_symlink_escape_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'state'; root.mkdir()
            (root / 'timing-generation-v1.jsonl').symlink_to(Path(tmp) / 'outside.jsonl')
            next(x for x in self.successor['mounts'] if x['target'] == '/data/work')['source'] = str(root)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                self.check()

    def test_timing_sink_and_mount_types_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            mount = next(x for x in self.successor['mounts'] if x['target'] == '/data/work')
            source = Path(tmp) / 'source'
            source.touch()
            mount['source'] = str(source)
            with self.assertRaisesRegex(ValueError, 'must be a directory'):
                self.check()
            source.unlink(); source.mkdir()
            (source / 'timing-generation-v1.jsonl').mkdir()
            with self.assertRaisesRegex(ValueError, 'must be a file'):
                self.check()

    def test_timing_cannot_share_preflight_evidence(self):
        next(x for x in self.successor['mounts'] if x['target'] == '/data/work')['source'] = self.output['source']
        with self.assertRaisesRegex(ValueError, 'aliases dependency or preflight'):
            self.check()

    def test_unknown_runtime_variable_cannot_exempt_dependencies(self):
        self.config['runtime_paths']['PYTHONPATH'] = {'kind': 'data-root', 'path': '/code'}
        with self.assertRaisesRegex(ValueError, 'only known'):
            self.check()

    def test_path_escape_and_noncanonical_paths_refused(self):
        original = copy.deepcopy(self.config)
        for value in ('/data/work/../models/x', '/data//work/x', '//data/work/x', '/data/work/./x', '/data/work:x', '/', 'relative'):
            with self.subTest(value=value):
                self.config = copy.deepcopy(original)
                self.config['runtime_paths']['SAEA_STAGE_TIMING_PATH']['path'] = value
                self.config['required_environment']['SAEA_STAGE_TIMING_PATH'] = value
                with self.assertRaises(ValueError):
                    m.probe.runtime_paths(self.config)

    def test_runtime_semantics_are_config_hash_bound(self):
        before = m.probe.digest(self.config)
        self.config['runtime_paths']['SAEA_STAGE_TIMING_PATH']['mount'] = '/data'
        self.assertNotEqual(before, m.probe.digest(self.config))

    def test_cpu_probe_does_not_open_runtime_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = copy.deepcopy(self.config)
            import sys
            config['interpreters'] = [{'name': 'stdlib', 'python': sys.executable,
                'environment_key': None, 'imports': ['json'], 'distributions': {}}]
            config_path = Path(tmp) / 'probe.json'; config_path.write_text(json.dumps(config))
            with patch.dict(os.environ, config['required_environment']):
                result = m.probe.execute(config_path, Path(tmp) / 'output')
            self.assertTrue(result['ok'])
            self.assertEqual(result['new_model_calls'], 0)


if __name__ == '__main__':
    unittest.main()
