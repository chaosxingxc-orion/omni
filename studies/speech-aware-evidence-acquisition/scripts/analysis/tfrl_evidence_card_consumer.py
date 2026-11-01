"""Deterministic CPU-only consumer for source-bound TFRL evidence cards.

This module is deliberately independent of the model runner.  It consumes
already validated cards from the development workflow and either returns a
stable, relation-matched evidence view or the unchanged lock answer.  Missing
evidence, conflicting cards, malformed relations, and cards from another task
fail closed to the lock answer.  No test labels, tokenizer, scorer, or model
client is imported here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


class EvidenceCardError(ValueError):
    """Raised when the consumer input violates the CPU contract."""


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceCardError(f"{field} must be non-empty text")
    return value.strip()


def _slots(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvidenceCardError(f"{field} must be a sequence of slot names")
    result = tuple(sorted({_text(item, f"{field}[]") for item in value}))
    if not result:
        raise EvidenceCardError(f"{field} must not be empty")
    return result


def _relation(value: object, field: str) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise EvidenceCardError(f"{field} must contain intent and slots")
    return _text(value.get("intent"), f"{field}.intent"), _slots(
        value.get("slots"), f"{field}.slots"
    )


def _card_key(card: Mapping[str, object]) -> tuple[str, str, str, tuple[str, ...]]:
    source = _text(card.get("source"), "card.source")
    intent = _text(card.get("intent"), "card.intent")
    slot = _text(card.get("slot"), "card.slot")
    relation = _relation(card.get("relation"), "card.relation")
    if relation[0] != intent or slot not in relation[1]:
        raise EvidenceCardError("card relation does not match intent/slot")
    return source, intent, slot, relation[1]


def consume_evidence_cards(
    cards: Sequence[Mapping[str, object]],
    *,
    request_relation: Mapping[str, object],
    lock_answer: Mapping[str, object],
) -> dict[str, Any]:
    """Build a deterministic evidence view, or return an explicit lock fallback.

    ``request_relation`` is the relation present in the source dataset.  The
    consumer matches that relation exactly; it never creates missing
    intent/slot combinations.  A card is considered usable only when its
    ``evidence`` field is non-empty and its answer payload is present.  If two
    cards address the same source/intent/slot/relation key with different
    answer payloads, the whole request falls back to ``lock_answer``.
    """

    if isinstance(cards, (str, bytes)) or not isinstance(cards, Sequence):
        raise EvidenceCardError("cards must be a sequence of mappings")
    wanted_intent, wanted_slots = _relation(request_relation, "request_relation")
    if not isinstance(lock_answer, Mapping):
        raise EvidenceCardError("lock_answer must be a mapping")

    usable: list[tuple[tuple[str, str, str, tuple[str, ...]], dict[str, Any]]] = []
    for raw in cards:
        if not isinstance(raw, Mapping):
            raise EvidenceCardError("each card must be a mapping")
        key = _card_key(raw)
        if key[1] != wanted_intent or key[3] != wanted_slots:
            continue
        evidence = raw.get("evidence")
        answer = raw.get("answer")
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)) or not evidence:
            return _fallback(lock_answer, "missing-evidence")
        if not isinstance(answer, Mapping):
            return _fallback(lock_answer, "missing-answer")
        row = deepcopy(dict(raw))
        row["relation"] = {"intent": key[1], "slots": list(key[3])}
        row["evidence"] = deepcopy(list(evidence))
        usable.append((key, row))

    if not usable:
        return _fallback(lock_answer, "no-matching-evidence")
    usable.sort(key=lambda item: item[0])
    by_key: dict[tuple[str, str, str, tuple[str, ...]], dict[str, Any]] = {}
    for key, row in usable:
        previous = by_key.get(key)
        if previous is not None and previous["answer"] != row["answer"]:
            return _fallback(lock_answer, "conflicting-evidence")
        by_key[key] = row
    ordered = [by_key[key] for key in sorted(by_key)]
    return {
        "schema": "saea-tfrl-evidence-card-consumption-v1",
        "mode": "conditioned",
        "answer": deepcopy(ordered[0]["answer"]),
        "evidence_cards": ordered,
        "fallback_reason": None,
        "new_model_calls": 0,
        "test_labels_read": False,
    }


def _fallback(lock_answer: Mapping[str, object], reason: str) -> dict[str, Any]:
    return {
        "schema": "saea-tfrl-evidence-card-consumption-v1",
        "mode": "lock",
        "answer": deepcopy(dict(lock_answer)),
        "evidence_cards": [],
        "fallback_reason": reason,
        "new_model_calls": 0,
        "test_labels_read": False,
    }


__all__ = ["EvidenceCardError", "consume_evidence_cards"]
