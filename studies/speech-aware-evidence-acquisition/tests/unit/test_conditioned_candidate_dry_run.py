import hashlib
import json

import pytest

from speech_aware_evidence_acquisition.rails.structured_experience_rules import (
    RULE_SCHEMA,
    RULE_SET_SCHEMA,
)
from scripts.analysis.validate_conditioned_candidate_dry_run import run


def _fixture(tmp_path):
    operation = {"operation": "ADD", "id": None, "content": "Keep the spoken person form."}
    payload = json.dumps(operation, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    source_sha = hashlib.sha256(payload).hexdigest()
    operations = [{**operation, "operation_sha256": source_sha,
                   "group_contact": "g", "operation_index": 0}]
    labels = {"rows": {"s": {"intent": "calendar_set", "slots": ["person"]}}}
    rule = {
        "schema": RULE_SCHEMA, "rule_id": "r1", "kind": "conditional-rule",
        "scope": {"task_ids": ["calendar_set"], "phases": ["knowledge-consumption"],
                  "components": ["person"]},
        "trigger": ["person cue"], "action": ["preserve person"], "exceptions": [],
        "source": {"ref_id": f"h1-operation:{source_sha}", "sha256": source_sha},
        "conflict": {"state": "none", "with_rule_ids": [], "resolution": None},
    }
    candidate = {
        "status": "cpu-validated-not-admitted",
        "rules": {"schema": RULE_SET_SCHEMA, "rules": [rule]},
        "source_index": [{"ref_id": f"h1-operation:{source_sha}",
                           "operation_sha256": source_sha}],
        "validation": {"new_model_calls": 0, "new_scoring_jobs": 0,
                        "h2_read": False, "test_read": False,
                        "cartesian_intent_slot_expansion": False},
    }
    cp = tmp_path / "candidate.json"; cp.write_text(json.dumps(candidate))
    op = tmp_path / "operations.json"; op.write_text(json.dumps(operations))
    lb = tmp_path / "labels.json"; lb.write_text(json.dumps(labels))
    return cp, op, lb


def test_dry_run_is_read_only_and_source_bound(tmp_path):
    cp, op, lb = _fixture(tmp_path)
    result = run(cp, op, lb)
    assert result["status"] == "rendered-compatible-not-admitted"
    assert result["source_payloads_verified"] == 1
    assert result["runtime_configuration_mutated"] is False
    assert result["solver_prompt_changed"] is False


def test_dry_run_rejects_h2_or_test_evidence(tmp_path):
    cp, op, lb = _fixture(tmp_path)
    value = json.loads(cp.read_text())
    value["validation"]["h2_read"] = True
    cp.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="disallowed evaluation evidence"):
        run(cp, op, lb)
