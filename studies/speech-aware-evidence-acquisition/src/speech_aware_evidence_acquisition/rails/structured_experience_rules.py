"""Fail-closed validation for source-bound structured experience rules.

Rules are CPU-side knowledge-construction artifacts.  They are deliberately
separate from runtime ``ExperienceRecord`` objects: a rule must carry an
explicit scope, trigger, action, exceptions, source pin and conflict state
before a later admission step can translate it into runtime guidance.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

RULE_SCHEMA = "saea-structured-experience-rule-v1"
RULE_SET_SCHEMA = "saea-structured-experience-rule-set-v1"
RULE_KINDS = frozenset({"conditional-rule", "corrective-experience"})
CONFLICT_STATES = frozenset({"none", "resolved", "unresolved"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StructuredRuleError(ValueError):
    """A structured rule set is not source-bound, typed, or conflict-free."""


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuredRuleError(f"{field} must be non-empty text")
    return value.strip()


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise StructuredRuleError(f"{field} must be a lowercase SHA-256")
    return value


def _texts(value: object, field: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StructuredRuleError(f"{field} must be a list of strings")
    result = tuple(_text(item, f"{field}[{index}]") for index, item in enumerate(value))
    if required and not result:
        raise StructuredRuleError(f"{field} must not be empty")
    if len(result) != len(set(result)):
        raise StructuredRuleError(f"{field} must contain unique values")
    return result


def _exact(value: object, field: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise StructuredRuleError(f"{field} must have exact keys {sorted(keys)}")
    return value


@dataclass(frozen=True)
class RuleScope:
    task_ids: tuple[str, ...]
    phases: tuple[str, ...]
    components: tuple[str, ...]

    @classmethod
    def from_json(cls, value: object) -> "RuleScope":
        row = _exact(value, "scope", {"task_ids", "phases", "components"})
        return cls(_texts(row["task_ids"], "scope.task_ids"),
                   _texts(row["phases"], "scope.phases"),
                   _texts(row["components"], "scope.components"))

    def as_json(self) -> dict[str, object]:
        return {"task_ids": list(self.task_ids), "phases": list(self.phases),
                "components": list(self.components)}


@dataclass(frozen=True)
class RuleSource:
    ref_id: str
    sha256: str

    @classmethod
    def from_json(cls, value: object) -> "RuleSource":
        row = _exact(value, "source", {"ref_id", "sha256"})
        return cls(_text(row["ref_id"], "source.ref_id"), _sha(row["sha256"], "source.sha256"))

    def as_json(self) -> dict[str, str]:
        return {"ref_id": self.ref_id, "sha256": self.sha256}


@dataclass(frozen=True)
class RuleConflict:
    state: str
    with_rule_ids: tuple[str, ...]
    resolution: str | None

    @classmethod
    def from_json(cls, value: object) -> "RuleConflict":
        row = _exact(value, "conflict", {"state", "with_rule_ids", "resolution"})
        state = _text(row["state"], "conflict.state")
        if state not in CONFLICT_STATES:
            raise StructuredRuleError(f"conflict.state must be one of {sorted(CONFLICT_STATES)}")
        ids = _texts(row["with_rule_ids"], "conflict.with_rule_ids", required=False)
        resolution = row["resolution"]
        if resolution is not None:
            resolution = _text(resolution, "conflict.resolution")
        if state == "unresolved":
            raise StructuredRuleError("unresolved rule conflict is not admissible")
        if state == "resolved":
            if not ids or resolution is None:
                raise StructuredRuleError("resolved conflict requires related rules and resolution")
        elif ids or resolution is not None:
            raise StructuredRuleError("conflict state none cannot carry conflict details")
        return cls(state, ids, resolution)

    def as_json(self) -> dict[str, object]:
        return {"state": self.state, "with_rule_ids": list(self.with_rule_ids),
                "resolution": self.resolution}


@dataclass(frozen=True)
class StructuredExperienceRule:
    rule_id: str
    kind: str
    scope: RuleScope
    trigger: tuple[str, ...]
    action: tuple[str, ...]
    exceptions: tuple[str, ...]
    source: RuleSource
    conflict: RuleConflict

    @classmethod
    def from_json(cls, value: object) -> "StructuredExperienceRule":
        row = _exact(value, "rule", {"schema", "rule_id", "kind", "scope", "trigger", "action", "exceptions", "source", "conflict"})
        if row["schema"] != RULE_SCHEMA:
            raise StructuredRuleError("rule schema mismatch")
        rule_id = _text(row["rule_id"], "rule_id")
        kind = _text(row["kind"], "kind")
        if kind not in RULE_KINDS:
            raise StructuredRuleError(f"kind must be one of {sorted(RULE_KINDS)}")
        trigger = _texts(row["trigger"], "trigger")
        action = _texts(row["action"], "action")
        exceptions = _texts(row["exceptions"], "exceptions", required=False)
        if set(trigger) & set(exceptions):
            raise StructuredRuleError("trigger and exceptions must not overlap")
        return cls(rule_id, kind, RuleScope.from_json(row["scope"]), trigger, action,
                   exceptions, RuleSource.from_json(row["source"]),
                   RuleConflict.from_json(row["conflict"]))

    def as_json(self) -> dict[str, object]:
        return {"schema": RULE_SCHEMA, "rule_id": self.rule_id, "kind": self.kind,
                "scope": self.scope.as_json(), "trigger": list(self.trigger),
                "action": list(self.action), "exceptions": list(self.exceptions),
                "source": self.source.as_json(), "conflict": self.conflict.as_json()}


def validate_rule_set(document: object, *, source_payloads: Mapping[str, bytes]) -> tuple[StructuredExperienceRule, ...]:
    """Validate and return canonical rules, refusing unbound or mixed inputs.

    ``source_payloads`` maps each source ``ref_id`` to the exact immutable bytes
    used to build the rule.  Hashes are checked here rather than treated as
    descriptive metadata.
    """
    row = _exact(document, "rule set", {"schema", "rules"})
    if row["schema"] != RULE_SET_SCHEMA:
        raise StructuredRuleError("rule-set schema mismatch")
    raw_rules = row["rules"]
    if isinstance(raw_rules, (str, bytes)) or not isinstance(raw_rules, Sequence) or not raw_rules:
        raise StructuredRuleError("rules must be a non-empty list")
    rules = tuple(StructuredExperienceRule.from_json(item) for item in raw_rules)
    ids = [rule.rule_id for rule in rules]
    if len(ids) != len(set(ids)):
        raise StructuredRuleError("rule IDs must be unique")
    kinds = {rule.kind for rule in rules}
    if len(kinds) != 1:
        raise StructuredRuleError("rule set mixes conditional-rule and corrective-experience types")
    for rule in rules:
        payload = source_payloads.get(rule.source.ref_id)
        if not isinstance(payload, bytes):
            raise StructuredRuleError(f"source payload is unavailable for {rule.rule_id}")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != rule.source.sha256:
            raise StructuredRuleError(f"source hash mismatch for {rule.rule_id}")
        for target in rule.conflict.with_rule_ids:
            if target == rule.rule_id or target not in ids:
                raise StructuredRuleError(f"conflict reference for {rule.rule_id} is invalid")
    return rules


def canonical_rule_set(rules: Sequence[StructuredExperienceRule]) -> str:
    """Serialize already-validated rules deterministically for hashing/storage."""
    if not rules:
        raise StructuredRuleError("cannot serialize an empty rule set")
    payload = {"schema": RULE_SET_SCHEMA, "rules": [rule.as_json() for rule in rules]}
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
