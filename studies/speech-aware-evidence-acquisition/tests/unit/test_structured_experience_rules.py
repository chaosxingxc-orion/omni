import hashlib
import json
import pytest

from speech_aware_evidence_acquisition.rails.structured_experience_rules import (
    RULE_SCHEMA, RULE_SET_SCHEMA, StructuredRuleError, canonical_rule_set,
    validate_rule_set,
)


def make_rule(*, kind="conditional-rule", source=b"source", conflict=None, rid="r1"):
    conflict = conflict or {"state": "none", "with_rule_ids": [], "resolution": None}
    return {"schema": RULE_SCHEMA, "rule_id": rid, "kind": kind,
            "scope": {"task_ids": ["slurp"], "phases": ["use"], "components": ["knowledge"]},
            "trigger": ["signal:ambiguous"], "action": ["inspect evidence"],
            "exceptions": ["state:verified"],
            "source": {"ref_id": "src", "sha256": hashlib.sha256(source).hexdigest()},
            "conflict": conflict}


def test_validates_source_bound_typed_rule_set():
    row = {"schema": RULE_SET_SCHEMA, "rules": [make_rule()]}
    rules = validate_rule_set(row, source_payloads={"src": b"source"})
    assert rules[0].rule_id == "r1"
    assert json.loads(canonical_rule_set(rules))["rules"][0]["source"]["ref_id"] == "src"


def test_rejects_mixed_rule_kinds():
    row = {"schema": RULE_SET_SCHEMA,
           "rules": [make_rule(), make_rule(kind="corrective-experience", rid="r2")]}
    with pytest.raises(StructuredRuleError, match="mixes"):
        validate_rule_set(row, source_payloads={"src": b"source"})


def test_rejects_wrong_source_hash():
    rule = make_rule()
    rule["source"]["sha256"] = "0" * 64
    with pytest.raises(StructuredRuleError, match="hash mismatch"):
        validate_rule_set({"schema": RULE_SET_SCHEMA, "rules": [rule]}, source_payloads={"src": b"source"})


def test_rejects_unresolved_conflict():
    rule = make_rule(conflict={"state": "unresolved", "with_rule_ids": ["r2"], "resolution": None})
    with pytest.raises(StructuredRuleError, match="unresolved"):
        validate_rule_set({"schema": RULE_SET_SCHEMA, "rules": [rule]}, source_payloads={"src": b"source"})


def test_rejects_missing_source():
    with pytest.raises(StructuredRuleError, match="unavailable"):
        validate_rule_set({"schema": RULE_SET_SCHEMA, "rules": [make_rule()]}, source_payloads={})
