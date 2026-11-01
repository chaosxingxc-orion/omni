import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts/checks/final_test_admission.py"
spec = importlib.util.spec_from_file_location("final_test_admission", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(mod)


def _seed(tmp_path):
    roster = {
        "roster_id": mod.ROSTER_ID,
        "split_role": "final-report",
        "status": "FROZEN",
        "counts": {"recordings": 13078},
        "contacts": 0,
    }
    (tmp_path / "roster.json").write_text(json.dumps(roster))
    methods = {}
    for name in mod.EXPECTED_METHODS:
        paths = {}
        for field in ("config", "runtime", "session", "budget", "owner_allowance", "exposure_rows", "queue"):
            rel = f"records/{name}-{field}.json"
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
            paths[field] = rel
        methods[name] = {**paths, "run_id": f"run-{name}", "checkpointed": True, "test_labels_in_runtime": False}
    registration = {
        "schema": "omni-final-test-registration-v1",
        "test_tuning": False,
        "roster": "roster.json",
        "roster_sha256": __import__("hashlib").sha256((tmp_path / "roster.json").read_bytes()).hexdigest(),
        "methods": methods,
    }
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration))
    return path


class FinalTestAdmissionTests(unittest.TestCase):
    def test_complete_registration_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            result = mod.evaluate(tmp_path, _seed(tmp_path))
            self.assertTrue(result["passed"], result)


    def test_missing_owner_allowance_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            path = _seed(tmp_path)
            document = json.loads(path.read_text())
            document["methods"]["best_t"].pop("owner_allowance")
            path.write_text(json.dumps(document))
            result = mod.evaluate(tmp_path, path)
            self.assertFalse(result["passed"])
            self.assertTrue(any("best_t: owner_allowance" in failure for failure in result["failures"]))


    def test_test_labels_or_uncheckpointed_method_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            path = _seed(tmp_path)
            document = json.loads(path.read_text())
            document["methods"]["agent"]["test_labels_in_runtime"] = True
            document["methods"]["direct"]["checkpointed"] = False
            path.write_text(json.dumps(document))
            result = mod.evaluate(tmp_path, path)
            self.assertFalse(result["passed"])
            self.assertTrue(any("agent: test_labels_in_runtime" in failure for failure in result["failures"]))
            self.assertTrue(any("direct: checkpointed" in failure for failure in result["failures"]))
