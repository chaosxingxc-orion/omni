import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tfrl_cpu_preparation_bridge import validate


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.receipt_path = self.root / "preparation" / "receipt.json"
        self.content_path = self.root / "prepared" / "vectors.json"
        self.receipt_path.parent.mkdir()
        self.content_path.parent.mkdir()
        self.receipt_path.write_text(json.dumps({
            "run_id": "SAEA-SLURP-R01C-SPEECH-QUERY-CPU-ISOLATED-V1",
            "protocol_sha256": "a" * 64,
            "generation_admitted": False,
        }, sort_keys=True) + "\n")
        self.content_path.write_text("{\"rows\":625}\n")
        self.closure_path = self.root / "closure.json"
        self.closure_path.write_text(json.dumps({
            "schema": "saea-r01c-preparation-closure-v1", "generation_admitted": False, "rows": 625,
            "protocol_sha256": "a" * 64,
            "pins": {"preparation-runs/SAEA-SLURP-R01C-SPEECH-QUERY-CPU-ISOLATED-V1/encode-v1/receipt.json": hashlib.sha256(self.receipt_path.read_bytes()).hexdigest()},
        }, sort_keys=True) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def pin(self, path):
        return {"relpath": str(path.relative_to(self.root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def bridge(self):
        source_run = "SAEA-SLURP-R01C-SPEECH-QUERY-CPU-ISOLATED-V1"
        return {
            "schema": "saea-tfrl-cpu-preparation-bridge-v1",
            "source": {"run_id": source_run, "protocol_sha256": "a" * 64, "receipt": self.pin(self.receipt_path), "closure": self.pin(self.closure_path), "content": [self.pin(self.content_path)]},
            "target": {"run_id": "SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-SMOKE-4-NPU-PORT-V1", "source_run_id": source_run, "source_protocol_sha256": "a" * 64},
            "generation_admitted": False,
        }

    def test_accepts_distinct_bound_identities(self):
        result = validate(self.bridge(), data_root=self.root)
        self.assertTrue(result["ready"])
        self.assertFalse(result["generation_admitted"])

    def test_rejects_renamed_cpu_receipt(self):
        bridge = self.bridge()
        bridge["source"]["run_id"] = bridge["target"]["run_id"]
        with self.assertRaisesRegex(ValueError, "CPU-isolated"):
            validate(bridge, data_root=self.root)

    def test_rejects_generation_admission(self):
        bridge = self.bridge()
        bridge["generation_admitted"] = True
        with self.assertRaisesRegex(ValueError, "cannot admit"):
            validate(bridge, data_root=self.root)

    def test_rejects_changed_content(self):
        bridge = self.bridge()
        self.content_path.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate(bridge, data_root=self.root)


if __name__ == "__main__":
    unittest.main()
