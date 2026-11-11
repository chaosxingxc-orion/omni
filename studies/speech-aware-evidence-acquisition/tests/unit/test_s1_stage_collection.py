import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

CODE = Path(__file__).resolve().parents[2] / 'scripts/analysis/s1_stage_evaluation'
sys.path.insert(0, str(CODE))
spec = importlib.util.spec_from_file_location('s1_collect', CODE / 'collect.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class CollectorTests(unittest.TestCase):
    def test_real_row_json_and_jsonl_replicas(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            obj = {'sample_id': 'p', 's1': [], 'selected_types': [], 'complete': False}
            (root / 'p.private.json').write_text(json.dumps(obj))
            (root / 'replica.private.json').write_text(json.dumps(obj, indent=2))
            pred, custody = mod.collect([root])
            self.assertEqual(len(pred), 1)
            self.assertEqual(custody['identical_replicas_reused'], 1)
            (root / 'rows.jsonl').write_text(json.dumps(obj) + '\n')
            pred, custody = mod.collect([root, root / 'rows.jsonl'])
            self.assertEqual(len(pred), 1)
            self.assertEqual(custody['identical_replicas_reused'], 2)

    def test_conflicting_paid_response_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'one.json').write_text(json.dumps({'sample_id': 'p', 'selected_types': []}))
            (root / 'two.json').write_text(json.dumps({'sample_id': 'p', 'selected_types': ['time']}))
            with self.assertRaises(ValueError):
                mod.collect([root])

    def test_empty_archive_is_explicitly_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            pred, custody = mod.collect([temp])
            self.assertEqual(pred, [])
            self.assertEqual(custody['identical_replicas_reused'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
