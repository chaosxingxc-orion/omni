"""Small dependency regression only; never runs TRAIN bind or native coroutine."""
import importlib.metadata
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fixed_cpu_suite as suite

REQUIRED = ('numpy', 'tiktoken', 'pydantic', 'openjiuwen')


class DependencyTests(unittest.TestCase):
    def probe(self, spec):
        imported = []
        def load(name):
            self.assertIn(name, REQUIRED, 'optional torch must never be imported')
            imported.append(name)
            return SimpleNamespace(__file__='/fixture/'+name+'.py', __version__='fixture')
        with patch.object(suite.importlib, 'import_module', side_effect=load), \
             patch.object(suite.importlib.metadata, 'version', side_effect=importlib.metadata.PackageNotFoundError), \
             patch.object(suite.importlib.util, 'find_spec', return_value=spec) as probe:
            deps = suite.dependency_packages()
        self.assertEqual(imported, list(REQUIRED))
        probe.assert_called_once_with('torch')
        self.assertTrue(all(deps[name]['required'] and deps[name]['available'] for name in REQUIRED))
        return deps

    def test_torch_absent_is_recorded_without_import(self):
        self.assertEqual(self.probe(None)['torch'],
                         dict(required=False, available=False, path=None, probe='find_spec'))

    def test_torch_present_is_recorded_without_import(self):
        self.assertEqual(self.probe(SimpleNamespace(origin='/fixture/torch/__init__.py'))['torch'],
                         dict(required=False, available=True, path='/fixture/torch/__init__.py', probe='find_spec'))

    def test_required_dependency_failure_is_not_suppressed(self):
        with patch.object(suite.importlib, 'import_module', side_effect=ModuleNotFoundError('numpy')), \
             patch.object(suite.importlib.util, 'find_spec') as probe:
            with self.assertRaises(ModuleNotFoundError):
                suite.dependency_packages()
            probe.assert_not_called()

    def test_dependency_contract_matches_main_probe(self):
        contract = suite.h.read(suite.OWNER/'FIXED-CPU-DEPENDENCIES.json')
        actual = suite.h.read(suite.OWNER/'evidence/MAIN-FIXED-IMAGE-IMPORT-PROBE.json')
        self.assertEqual(contract['required_python_packages'], list(REQUIRED))
        self.assertEqual(contract['optional_python_packages'], ['torch'])
        self.assertFalse(actual['actual_find_spec']['torch'])
        self.assertTrue(all(actual['actual_find_spec'][name] for name in REQUIRED))
        self.assertEqual(actual['image'], suite.IMAGE)
        self.assertEqual(actual['python'], suite.PYTHON)


if __name__ == '__main__':
    unittest.main(verbosity=2)
