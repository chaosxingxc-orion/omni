import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tfrl_v2_smoke_admission as m


class SmokeAdmissionTests(unittest.TestCase):
    def test_draft_and_pending_rails_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "root-admission-draft-v1.json").write_text(json.dumps({
                "status": "DRAFT-NOT-MINTED", "service": {"deployment_profile": "PORT_DEPLOYMENT_PROFILE (pending)"}}))
            (root / "foldb-smoke-generation-plan-registered-v1.json").write_text(json.dumps({"rails_pending": ["bank-store-roles", "exemplar-block"]}))
            (root / "signoff-conditions-fold-b-v1.json").write_text(json.dumps({"table_relpath": "analysis/smoke.json"}))
            result = m.audit(root, bank_ready=True)
        self.assertFalse(result["ready"])
        self.assertTrue(any("deployment profile" in item for item in result["failures"]))
        self.assertTrue(any("rails_pending" in item for item in result["failures"]))

    def test_bank_failure_is_refused_without_model_contact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "root-admission-draft-v1.json").write_text(json.dumps({"status": "DECIDED", "service": {"deployment_profile": "tp4"}}))
            (root / "signoff-conditions-fold-b-v1.json").write_text(json.dumps({"table_relpath": "analysis/smoke.json"}))
            result = m.audit(root, bank_ready=False)
        self.assertFalse(result["ready"])
        self.assertIn("bank readiness audit did not pass", result["failures"])
        self.assertEqual(result["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
