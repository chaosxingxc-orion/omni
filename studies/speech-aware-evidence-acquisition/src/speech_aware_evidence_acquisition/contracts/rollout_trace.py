"""``saea-rollout-trace-v1``: the schema of the per-flight, per-contact rollout trace (plan T1 / fix M4).

One JSONL file per flight, one line per event, three line kinds:

``contact``  one model contact (one dispatched request that returned): who asked (run, attempt, row, arm, node, contact index
             in the program, the slot and the vLLM instance it routed to), what was supplied (item ids with retrieval scores,
             the supply variant), what it cost (prompt, thinking and completion tokens, the truncation flag), what came back
             (the parsed answer fields as emitted) and how long it took.
``score``    the per-row official score of one arm's delivered answer, emitted by the scorer once it is known, keyed by
             (row, arm).
``tfrl``     one sampled training-free-RL trajectory: its id, its reward and whether it was selected.

The schema is closed. Every line carries exactly the fields of its kind (names fixed here, versioned by ``SCHEMA``); a line with
a missing, extra or mistyped field, or of an unknown kind, is refused by :func:`validate_line`. Fields that a transport cannot
always fill are present with an explicit ``null`` (and, for thinking tokens, a ``thinking_tokens_basis`` that says why), never
silently absent.

This module is pure: no I/O, no dependency beyond the standard library. The writer lives in ``core.rollout_trace``.

Truncation (``truncated``) means the contact hit its thinking budget or its completion cap: the server stopped on ``length``,
or the pinned forced-close message ended the reasoning channel, or the reported thinking tokens reached the declared budget.
The trace changes no hash, no prompt byte and no decision of the flight; it only records what the flight did.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Mapping

SCHEMA = "saea-rollout-trace-v1"

KIND_CONTACT = "contact"
KIND_SCORE = "score"
KIND_TFRL = "tfrl"
KINDS = (KIND_CONTACT, KIND_SCORE, KIND_TFRL)

#: The graph node a contact belongs to. ``perception`` and ``solver`` are the two nodes the three-way comparison reads; the
#: other native roles are kept apart so they never pollute those two rates.
NODES = ("perception", "solver", "reviewer", "planner", "acquisition", "other")

OUTCOMES = ("ok", "refused", "error")

#: Where ``thinking_tokens`` came from.
THINKING_BASES = ("server-usage-details", "not-reported-by-server")

#: A UTC timestamp, second resolution or finer, always ``Z``.
_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RolloutTraceSchemaError(ValueError):
    """A rollout-trace line violates ``saea-rollout-trace-v1``."""


# ----------------------------------------------------------------------------------------------------------------------
# Field checkers. Each returns None when the value is acceptable, else a short reason.
# ----------------------------------------------------------------------------------------------------------------------

Checker = Callable[[Any], "str | None"]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _text(value: object) -> str | None:
    return None if isinstance(value, str) and value != "" else "must be a non-empty string"


def _text_or_null(value: object) -> str | None:
    return None if value is None or (isinstance(value, str) and value != "") else "must be a non-empty string or null"


def _any_string(value: object) -> str | None:
    return None if isinstance(value, str) else "must be a string"


def _bool(value: object) -> str | None:
    return None if isinstance(value, bool) else "must be a boolean"


def _bool_or_null(value: object) -> str | None:
    return None if value is None or isinstance(value, bool) else "must be a boolean or null"


def _count(value: object) -> str | None:
    return None if _is_int(value) and value >= 0 else "must be a non-negative integer"


def _count_or_null(value: object) -> str | None:
    return None if value is None or (_is_int(value) and value >= 0) else "must be a non-negative integer or null"


def _number(value: object) -> str | None:
    return None if _is_number(value) else "must be a finite number"


def _number_or_null(value: object) -> str | None:
    return None if value is None or _is_number(value) else "must be a finite number or null"


def _seconds(value: object) -> str | None:
    return None if value is None or (_is_number(value) and float(value) >= 0.0) else "must be a non-negative number or null"


def _sha256_or_null(value: object) -> str | None:
    return None if value is None or (isinstance(value, str) and _SHA256.fullmatch(value)) else "must be a lowercase sha256 or null"


def _one_of(allowed: tuple[str, ...]) -> Checker:
    def check(value: object) -> str | None:
        return None if isinstance(value, str) and value in allowed else f"must be one of {list(allowed)}"

    return check


def _timestamp(value: object) -> str | None:
    return None if isinstance(value, str) and _TS.fullmatch(value) else "must be a UTC timestamp ending in Z"


def _literal(expected: str) -> Checker:
    def check(value: object) -> str | None:
        return None if value == expected else f"must be {expected!r}"

    return check


def _supply_items(value: object) -> str | None:
    """``[{"item_id": str, "rank": int >= 1, "score": number | null}, ...]`` in supplied order; ranks are 1..n."""

    if not isinstance(value, list):
        return "must be a list"
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"item_id", "rank", "score"}:
            return f"item {index} must have exactly the keys item_id, rank, score"
        if not isinstance(item["item_id"], str) or item["item_id"] == "":
            return f"item {index} item_id must be a non-empty string"
        if not _is_int(item["rank"]) or item["rank"] != index + 1:
            return f"item {index} rank must be {index + 1}"
        if item["score"] is not None and not _is_number(item["score"]):
            return f"item {index} score must be a finite number or null"
    ids = [item["item_id"] for item in value]
    if len(ids) != len(set(ids)):
        return "item ids must be unique"
    return None


def _answer(value: object) -> str | None:
    """``null`` (no answer parsed at this contact) or the answer fields as emitted."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"parsed", "scenario", "action", "entities"}:
        return "must be null or have exactly the keys parsed, scenario, action, entities"
    if not isinstance(value["parsed"], bool):
        return "parsed must be a boolean"
    for key in ("scenario", "action"):
        if not isinstance(value[key], str):
            return f"{key} must be a string"
    entities = value["entities"]
    if not isinstance(entities, list):
        return "entities must be a list"
    for index, entity in enumerate(entities):
        if not isinstance(entity, Mapping) or set(entity) != {"type", "filler"}:
            return f"entity {index} must have exactly the keys type, filler"
        if not isinstance(entity["type"], str) or not isinstance(entity["filler"], str):
            return f"entity {index} type and filler must be strings"
    return None


_COMMON: dict[str, Checker] = {
    "schema": _literal(SCHEMA),
    "kind": _one_of(KINDS),
    "ts": _timestamp,
    "run_id": _text,
    "attempt_id": _text,
    "row_id": _text,
    "arm": _text_or_null,
}

_CONTACT: dict[str, Checker] = {
    **_COMMON,
    "node": _one_of(NODES),
    "role_id": _text_or_null,
    "contact_id": _text,
    "contact_index": _count_or_null,
    "sample_index": _count_or_null,
    "request_id": _text_or_null,
    "outcome": _one_of(OUTCOMES),
    "error_type": _text_or_null,
    "id_slot": _count_or_null,
    "instance": _count_or_null,
    "supply_variant_id": _text_or_null,
    "supply_items": _supply_items,
    "prompt_tokens": _count_or_null,
    "completion_tokens": _count_or_null,
    "thinking_tokens": _count_or_null,
    "thinking_tokens_basis": _one_of(THINKING_BASES),
    "reasoning_chars": _count_or_null,
    "thinking_budget": _count_or_null,
    "max_tokens": _count_or_null,
    "truncated": _bool,
    "finish_reason": _text_or_null,
    "forced_close": _bool_or_null,
    "answer_only_reask": _bool,
    "answer": _answer,
    "response_text_sha256": _sha256_or_null,
    "latency_seconds": _seconds,
}

_SCORE: dict[str, Checker] = {
    **_COMMON,
    "scorer_id": _text,
    "scenario_action_exact": _bool,
    "slu_f1": _number,
}

_TFRL: dict[str, Checker] = {
    **_COMMON,
    "iteration": _count_or_null,
    "trajectory_id": _text,
    "reward": _number,
    "selected": _bool,
    "reason": _text_or_null,
}

FIELDS: dict[str, dict[str, Checker]] = {KIND_CONTACT: _CONTACT, KIND_SCORE: _SCORE, KIND_TFRL: _TFRL}


def field_names(kind: str) -> tuple[str, ...]:
    """The fixed, ordered field names of one line kind."""

    if kind not in FIELDS:
        raise RolloutTraceSchemaError(f"unknown rollout-trace kind {kind!r}; known: {list(KINDS)}")
    return tuple(FIELDS[kind])


def validate_line(line: object) -> dict[str, Any]:
    """Return ``line`` (as a dict) when it is a valid ``saea-rollout-trace-v1`` line, else raise :class:`RolloutTraceSchemaError`.

    Refuses: a non-mapping, an unknown or missing ``kind``, a wrong ``schema``, any missing field, any extra field, and any field
    whose value does not satisfy its checker. Unknown kinds are refused before any field is looked at.
    """

    if not isinstance(line, Mapping):
        raise RolloutTraceSchemaError("a rollout-trace line must be a JSON object")
    kind = line.get("kind")
    if not isinstance(kind, str) or kind not in FIELDS:
        raise RolloutTraceSchemaError(f"unknown rollout-trace kind {kind!r}; known: {list(KINDS)}")
    spec = FIELDS[kind]
    missing = [name for name in spec if name not in line]
    if missing:
        raise RolloutTraceSchemaError(f"{kind} line is missing fields {missing}")
    extra = sorted(set(line) - set(spec))
    if extra:
        raise RolloutTraceSchemaError(f"{kind} line carries unknown fields {extra}")
    for name, check in spec.items():
        reason = check(line[name])
        if reason is not None:
            raise RolloutTraceSchemaError(f"{kind} line field {name!r} {reason}")
    if kind == KIND_CONTACT:
        _validate_contact_coherence(line)
    return dict(line)


def _validate_contact_coherence(line: Mapping[str, Any]) -> None:
    """Cross-field rules of a contact line."""

    if (line["thinking_tokens"] is None) != (line["thinking_tokens_basis"] == "not-reported-by-server"):
        raise RolloutTraceSchemaError(
            "contact line: thinking_tokens is null exactly when thinking_tokens_basis is 'not-reported-by-server'"
        )
    if line["outcome"] == "ok" and line["error_type"] is not None:
        raise RolloutTraceSchemaError("contact line: an 'ok' outcome carries no error_type")
    if line["outcome"] != "ok" and line["error_type"] is None:
        raise RolloutTraceSchemaError("contact line: a non-'ok' outcome names its error_type")


def dumps_line(line: Mapping[str, Any]) -> str:
    """Validate and serialize one line: compact, key-sorted, UTF-8, newline-terminated."""

    validate_line(line)
    return json.dumps(line, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def loads_line(text: str) -> dict[str, Any]:
    """Parse and validate one serialized line."""

    try:
        document = json.loads(text)
    except ValueError as error:
        raise RolloutTraceSchemaError(f"not JSON: {error}") from error
    return validate_line(document)


def is_truncated(
    *, finish_reason: object, forced_close: object, thinking_tokens: object, thinking_budget: object,
) -> bool:
    """The truncation flag: the contact hit its thinking budget or its completion cap (module docstring)."""

    if finish_reason == "length" or forced_close is True:
        return True
    return bool(
        _is_int(thinking_tokens) and _is_int(thinking_budget) and thinking_budget > 0 and thinking_tokens >= thinking_budget
    )
