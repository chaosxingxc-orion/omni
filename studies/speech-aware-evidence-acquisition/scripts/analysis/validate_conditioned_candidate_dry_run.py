"""Read-only fixed-consumer compatibility check for a conditioned candidate.

This command validates source custody and renders the candidate into a bounded
consumer projection.  It never edits runtime configuration and never calls a
model, scorer, tokenizer, or serving endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from speech_aware_evidence_acquisition.rails.structured_experience_rules import (
    StructuredRuleError,
    validate_rule_set,
)


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def run(candidate_path: Path, operations_path: Path, labels_path: Path) -> dict[str, object]:
    candidate = json.loads(candidate_path.read_bytes())
    operations = json.loads(operations_path.read_bytes())
    labels = json.loads(labels_path.read_bytes())
    if candidate.get("status") != "cpu-validated-not-admitted":
        raise ValueError("candidate must remain CPU-only and not admitted")
    validation = candidate.get("validation")
    if not isinstance(validation, dict) or any(validation.get(k) for k in (
        "new_model_calls", "new_scoring_jobs", "h2_read", "test_read",
        "cartesian_intent_slot_expansion",
    )):
        raise ValueError("candidate contains disallowed evaluation evidence")
    by_sha = {row["operation_sha256"]: row for row in operations}
    payloads: dict[str, bytes] = {}
    for row in candidate["source_index"]:
        source_sha = row["operation_sha256"]
        operation = by_sha.get(source_sha)
        if operation is None:
            raise ValueError("candidate source operation is missing")
        payload = canonical({k: operation[k] for k in ("operation", "id", "content")})
        if hashlib.sha256(payload).hexdigest() != source_sha:
            raise ValueError("candidate source operation bytes changed")
        payloads[row["ref_id"]] = payload
    try:
        rules = validate_rule_set(candidate["rules"], source_payloads=payloads)
    except StructuredRuleError as error:
        raise ValueError(f"structured candidate rejected: {error}") from error
    relations = {(row["intent"], slot)
                 for row in labels["rows"].values() for slot in row["slots"]}
    projection = []
    for rule in rules:
        for task_id in rule.scope.task_ids:
            if not any((task_id, component) in relations for component in rule.scope.components):
                raise ValueError(f"consumer scope lacks fixed TRAIN relation: {rule.rule_id}")
        projection.append({
            "rule_id": rule.rule_id,
            "task_ids": list(rule.scope.task_ids),
            "components": list(rule.scope.components),
            "trigger": list(rule.trigger),
            "action": list(rule.action),
            "exceptions": list(rule.exceptions),
        })
    rendered = json.dumps(projection, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
    if len(rendered) > 16000:
        raise ValueError("consumer projection exceeds fixed memory bound")
    return {
        "schema": "saea-conditioned-candidate-consumer-dry-run-v1",
        "status": "rendered-compatible-not-admitted",
        "rules": len(rules),
        "rendered_characters": len(rendered),
        "projection_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "new_model_calls": 0,
        "new_scoring_jobs": 0,
        "runtime_configuration_mutated": False,
        "solver_prompt_changed": False,
        "source_payloads_verified": len(payloads),
        "h2_read": False,
        "test_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.candidate, args.operations, args.labels)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
