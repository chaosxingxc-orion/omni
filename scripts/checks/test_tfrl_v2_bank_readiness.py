import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tfrl_v2_bank_readiness as m


def release_doc():
    return {"iterations": [{"releases": {"union": {
        "bank_receipt": {"sha256": "0" * 64, "relpath": "missing.json"},
        "measure": {"observed": None},
        "exemplar_bank_content_sha256": "0" * 64,
        "rule_bank_content_sha256": "0" * 64,
        "token_counts": {"exemplar_block_token_cap": 2000, "rule_bank_token_cap": 400,
                         "exemplar_max_block_tokens": None, "rule_bank_rendered_tokens": None},
        "source_halves": ["R01C-H1", "R01C-H2"], "consuming_roster": "FOLD-B",
    }}}]}


class BankReadinessTests(unittest.TestCase):
    def test_pending_release_is_explained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank-release.json"
            path.write_text(json.dumps(release_doc()))
            result = m.audit(path)
        self.assertFalse(result["ready"])
        reasons = result["reports"][0]["reasons"]
        self.assertIn("receipt hash is zero, malformed, or pending", reasons)
        self.assertIn("token measure is None, expected 'vllm-tokenize'", reasons)
        self.assertIn("exemplar token count is not measured", reasons)


    def test_measured_release_can_pass_without_model_contact(self):
        doc = release_doc()
        item = doc["iterations"][0]["releases"]["union"]
        item["bank_receipt"]["sha256"] = "a" * 64
        item["measure"]["observed"] = "vllm-tokenize"
        item["exemplar_bank_content_sha256"] = "b" * 64
        item["rule_bank_content_sha256"] = "c" * 64
        item["token_counts"]["exemplar_max_block_tokens"] = 1200
        item["token_counts"]["rule_bank_rendered_tokens"] = 100
        item["bank_receipt"].pop("relpath")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank-release.json"
            path.write_text(json.dumps(doc))
            self.assertTrue(m.audit(path)["ready"])


if __name__ == "__main__":
    unittest.main()
