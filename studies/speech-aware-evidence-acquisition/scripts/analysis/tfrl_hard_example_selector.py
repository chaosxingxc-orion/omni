"""Deterministic CPU selector for a training-free-RL development roster.

The selector consumes *development* observations only.  It admits rows with a
measured failure rate, then maximizes coverage of relations that actually
occur in the source data.  It never constructs an intent/slot Cartesian
product and it has no model, tokenizer, scorer, or test-label access.
"""
from __future__ import annotations

from typing import Iterable, Mapping


class SelectionError(ValueError):
    pass


def _relation(row: Mapping[str, object]) -> set[tuple[str, str]]:
    intent = row.get("intent")
    slots = row.get("slot_types")
    if not isinstance(intent, str) or not intent:
        raise SelectionError("each row needs a non-empty intent")
    if not isinstance(slots, (list, tuple, set)) or not all(isinstance(s, str) and s for s in slots):
        raise SelectionError("each row needs observed slot_types")
    return {(intent, slot) for slot in set(slots)}


def select_hard_examples(
    rows: Iterable[Mapping[str, object]],
    *,
    limit: int,
    min_error_rate: float = 0.5,
) -> dict[str, object]:
    """Select hard rows while preserving observed relation coverage.

    A row is eligible only when ``attempt_count`` is positive and its measured
    ``failure_count / attempt_count`` reaches ``min_error_rate``.  Greedy
    coverage chooses the highest-error row for each still-uncovered observed
    relation, then fills remaining capacity by error rate.  Ties use the
    stable ``sample_id`` string so reruns are byte-for-byte reproducible.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise SelectionError("limit must be a positive integer")
    if not isinstance(min_error_rate, (int, float)) or isinstance(min_error_rate, bool) or not 0 <= min_error_rate <= 1:
        raise SelectionError("min_error_rate must be between zero and one")
    materialized = list(rows)
    seen: set[str] = set()
    eligible: list[dict[str, object]] = []
    for raw in materialized:
        if not isinstance(raw, Mapping):
            raise SelectionError("rows must be mappings")
        sample_id = raw.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise SelectionError("each row needs sample_id")
        if sample_id in seen:
            raise SelectionError("duplicate sample_id: " + sample_id)
        seen.add(sample_id)
        relations = _relation(raw)
        attempts, failures = raw.get("attempt_count"), raw.get("failure_count")
        if (isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0 or
                isinstance(failures, bool) or not isinstance(failures, int) or failures < 0 or failures > attempts):
            raise SelectionError("invalid attempt_count/failure_count for " + sample_id)
        rate = failures / attempts
        if rate >= min_error_rate:
            eligible.append({"sample_id": sample_id, "intent": raw["intent"],
                             "slot_types": sorted(set(raw["slot_types"])),
                             "error_rate": rate, "attempt_count": attempts,
                             "failure_count": failures, "relations": relations})
    eligible.sort(key=lambda r: (-float(r["error_rate"]), str(r["sample_id"])))
    selected: list[dict[str, object]] = []
    covered: set[tuple[str, str]] = set()
    remaining = list(eligible)
    while remaining and len(selected) < limit:
        candidate = sorted(
            remaining,
            key=lambda r: (-len(set(r["relations"]) - covered),
                           -float(r["error_rate"]), str(r["sample_id"])),
        )[0]
        remaining.remove(candidate)
        selected.append(candidate)
        covered.update(candidate["relations"])
        # Preserve the stable error-rate ordering for the fill pass.
        if not any(set(r["relations"]) - covered for r in remaining):
            selected.extend(remaining[: max(0, limit - len(selected))])
            break
    for row in selected:
        row.pop("relations", None)
    all_relations = set().union(*(_relation(r) for r in materialized)) if materialized else set()
    return {
        "schema": "saea-tfrl-hard-example-selection-v1",
        "limit": limit,
        "min_error_rate": float(min_error_rate),
        "eligible_rows": len(eligible),
        "selected_rows": selected,
        "observed_relations": len(all_relations),
        "covered_relations": len(covered),
        "selection_scope": "development observations; original observed relations only",
        "new_model_calls": 0,
        "test_labels_read": False,
    }


__all__ = ["SelectionError", "select_hard_examples"]
