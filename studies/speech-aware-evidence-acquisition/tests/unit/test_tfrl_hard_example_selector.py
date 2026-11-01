import unittest

from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "tfrl_hard_example_selector",
    ROOT / "scripts" / "analysis" / "tfrl_hard_example_selector.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HardExampleSelectorTests(unittest.TestCase):
    def test_filters_easy_rows_and_covers_observed_relations(self):
        rows = [
            {"sample_id": "a", "intent": "music_play", "slot_types": ["song"], "attempt_count": 10, "failure_count": 9},
            {"sample_id": "b", "intent": "music_play", "slot_types": ["artist"], "attempt_count": 10, "failure_count": 6},
            {"sample_id": "easy", "intent": "music_play", "slot_types": ["song"], "attempt_count": 10, "failure_count": 1},
        ]
        result = MODULE.select_hard_examples(rows, limit=2)
        self.assertEqual([r["sample_id"] for r in result["selected_rows"]], ["a", "b"])
        self.assertEqual(result["covered_relations"], 2)
        self.assertFalse(result["test_labels_read"])

    def test_does_not_invent_cartesian_relations(self):
        rows = [{"sample_id": "a", "intent": "x", "slot_types": ["s"], "attempt_count": 2, "failure_count": 1}]
        result = MODULE.select_hard_examples(rows, limit=1)
        self.assertEqual(result["observed_relations"], 1)

    def test_duplicate_and_invalid_counts_fail_closed(self):
        row = {"sample_id": "a", "intent": "x", "slot_types": ["s"], "attempt_count": 1, "failure_count": 2}
        with self.assertRaises(MODULE.SelectionError):
            MODULE.select_hard_examples([row], limit=1)
        with self.assertRaises(MODULE.SelectionError):
            MODULE.select_hard_examples([row | {"failure_count": 1}, row | {"failure_count": 1}], limit=1)


if __name__ == "__main__":
    unittest.main()
