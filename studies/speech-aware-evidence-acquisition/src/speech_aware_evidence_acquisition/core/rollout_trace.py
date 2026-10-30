"""Writer and helpers of the per-flight rollout trace (``contracts.rollout_trace``, schema ``saea-rollout-trace-v1``; plan T1 / M4).

The trace is OBSERVATION ONLY. It changes no hash, no prompt byte and no decision of a flight:

* it is read-only with respect to everything the runner does; the seams call it AFTER a contact has been dispatched and its result
  received, passing values the runner already holds;
* a failure of any kind (an unwritable path, a full disk, a malformed field, a bug in a helper) is logged and swallowed -- the
  writer never raises into a contact, so a broken trace can cost a line but never a row;
* it writes nothing into the sealed run directory, the trace sink or any manifest: its only output is one append-only JSONL file
  under ``$SPEECHRL_DATA_DIR/analysis/<READ id>/rollout-trace.jsonl``.

One file per flight: every writer opened for the same path in one process shares one file descriptor and one lock, lines are
appended with a single ``os.write`` on an ``O_APPEND`` descriptor and ``fsync``-ed one by one, so a crash loses at most the line
being written.

Where the path comes from: the flight launcher names the READ id its analysis directory carries
(``SAEA_ROLLOUT_TRACE_READ_ID``, set beside ``SPEECHRL_DATA_DIR`` by the launch script); with it unset, tracing is off and every
call here is a no-op. The READ id is not a field of the hashed execution plan, so it cannot come from there.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from ..contracts import RUN_ID_PATTERN
from ..contracts import rollout_trace as schema

_LOG = logging.getLogger(__name__)

#: The launcher's switch: the READ id whose analysis directory receives the flight's trace.
READ_ID_ENV = "SAEA_ROLLOUT_TRACE_READ_ID"
DATA_DIR_ENV = "SPEECHRL_DATA_DIR"
TRACE_FILENAME = "rollout-trace.jsonl"


def trace_path(data_root: str | os.PathLike[str], read_id: str) -> Path:
    """``<data root>/analysis/<READ id>/rollout-trace.jsonl``; the READ id must be a safe single path segment."""

    if not isinstance(read_id, str) or RUN_ID_PATTERN.fullmatch(read_id) is None:
        raise ValueError(f"READ id {read_id!r} is not a safe path segment")
    return Path(data_root) / "analysis" / read_id / TRACE_FILENAME


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class _Appender:
    """One shared append-only descriptor per path (per process)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.fd: int | None = None

    def append(self, data: bytes) -> None:
        with self.lock:
            if self.fd is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            view = memoryview(data)
            while view:
                written = os.write(self.fd, view)
                view = view[written:]
            os.fsync(self.fd)


_APPENDERS: dict[str, _Appender] = {}
_APPENDERS_LOCK = threading.Lock()


def _appender(path: Path) -> _Appender:
    key = os.path.abspath(path)
    with _APPENDERS_LOCK:
        found = _APPENDERS.get(key)
        if found is None:
            found = _APPENDERS[key] = _Appender(Path(key))
        return found


class RolloutTraceWriter:
    """Append validated ``saea-rollout-trace-v1`` lines for one flight; never raises (module docstring)."""

    def __init__(
        self, path: str | os.PathLike[str], *, run_id: str, attempt_id: str,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        self._path = Path(path)
        self._run_id = run_id
        self._attempt_id = attempt_id
        self._clock = clock
        self._appender = _appender(self._path)
        self._counter_lock = threading.Lock()
        self.written = 0
        self.failures = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    def emit(self, kind: str, **fields: object) -> bool:
        """Validate and append one line; True when it reached the file. Any failure is logged, counted and swallowed."""

        try:
            line = {
                "schema": schema.SCHEMA, "kind": kind, "ts": self._clock(),
                "run_id": self._run_id, "attempt_id": self._attempt_id, **fields,
            }
            data = schema.dumps_line(line).encode("utf-8")
            self._appender.append(data)
        except Exception as error:  # noqa: BLE001 -- the trace must never fail a contact
            with self._counter_lock:
                self.failures += 1
            _LOG.warning("rollout trace line dropped (%s: %s) path=%s", type(error).__name__, error, self._path)
            return False
        with self._counter_lock:
            self.written += 1
        return True

    def score(self, *, row_id: str, arm: str | None, scorer_id: str, scenario_action_exact: bool, slu_f1: float) -> bool:
        return self.emit(
            schema.KIND_SCORE, row_id=row_id, arm=arm, scorer_id=scorer_id,
            scenario_action_exact=bool(scenario_action_exact), slu_f1=float(slu_f1),
        )

    def tfrl(
        self, *, row_id: str, arm: str | None, trajectory_id: str, reward: float, selected: bool,
        iteration: int | None = None, reason: str | None = None,
    ) -> bool:
        return self.emit(
            schema.KIND_TFRL, row_id=row_id, arm=arm, iteration=iteration, trajectory_id=trajectory_id,
            reward=float(reward), selected=bool(selected), reason=reason,
        )

    def contact(self, **fields: object) -> bool:
        return self.emit(schema.KIND_CONTACT, **fields)

    def score_keys(self) -> set[tuple[str, str | None, str]]:
        """The ``(row, arm, scorer)`` keys of the score lines already in the file (so a re-run of a scorer appends nothing twice)."""

        keys: set[tuple[str, str | None, str]] = set()
        try:
            with open(self._path, "rb") as handle:
                for raw in handle:
                    try:
                        line = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(line, Mapping) and line.get("kind") == schema.KIND_SCORE:
                        keys.add((line.get("row_id"), line.get("arm"), line.get("scorer_id")))
        except OSError:
            pass
        return keys


def emit_arm_scores(
    writer: RolloutTraceWriter | None, *, arm: str | None, scorer_id: str, rows: Iterable[Mapping[str, object]],
) -> int:
    """Append one ``score`` line per scored row of one arm's column (rows as the official scorer returns them); returns how many.

    A row already scored for the same (row, arm, scorer) in the file is skipped, so a scorer re-run is idempotent. Never raises.
    """

    if writer is None:
        return 0
    try:
        existing = writer.score_keys()
        written = 0
        for row in rows:
            row_id = str(row["sample_id"])
            key = (row_id, arm, scorer_id)
            if key in existing:
                continue
            if writer.score(
                row_id=row_id, arm=arm, scorer_id=scorer_id,
                scenario_action_exact=float(row["scenario_action_exact"]) >= 1.0, slu_f1=float(row["slu_f1_f1"]),
            ):
                existing.add(key)
                written += 1
        return written
    except Exception as error:  # noqa: BLE001 -- the trace never fails a scoring
        _LOG.warning("rollout trace scores skipped (%s: %s)", type(error).__name__, error)
        return 0


def open_writer(
    *, data_root: str | os.PathLike[str] | None, run_id: str, attempt_id: str,
    environ: Mapping[str, str] | None = None,
) -> RolloutTraceWriter | None:
    """The flight's writer, or None when tracing is off (``SAEA_ROLLOUT_TRACE_READ_ID`` unset). Never raises."""

    try:
        env = os.environ if environ is None else environ
        read_id = env.get(READ_ID_ENV)
        if not read_id:
            return None
        root = data_root if data_root is not None else env.get(DATA_DIR_ENV)
        if not root:
            _LOG.warning("rollout trace is requested (%s) but no data root is known; tracing is off", READ_ID_ENV)
            return None
        return RolloutTraceWriter(trace_path(root, read_id), run_id=run_id, attempt_id=attempt_id)
    except Exception as error:  # noqa: BLE001
        _LOG.warning("rollout trace could not be opened (%s: %s); tracing is off", type(error).__name__, error)
        return None


# ----------------------------------------------------------------------------------------------------------------------
# Value extraction. Pure helpers over what the runner already holds.
# ----------------------------------------------------------------------------------------------------------------------

def cost_records(trace_buffer: object) -> list[Mapping[str, object]]:
    """Every ``COST`` payload of a per-contact trace buffer, in emission order (duck-typed ``iter_records``)."""

    out: list[Mapping[str, object]] = []
    iterate = getattr(trace_buffer, "iter_records", None)
    if not callable(iterate):
        return out
    for channel, record_id, payload in iterate():
        if channel == "COST" and isinstance(record_id, str) and record_id.endswith("/cost") and isinstance(payload, Mapping):
            out.append(payload)
    return out


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def cost_facts(
    costs: Sequence[Mapping[str, object]], *, thinking_budget: object = None,
    finish_reason_fallback: object = None,
) -> dict[str, object]:
    """The token, truncation and latency fields of a contact line from its ``COST`` payloads.

    The FIRST record is the contact's thinking request; a second record is the single answer-only re-ask of decision 278. Prompt
    tokens, thinking tokens, finish reason and forced close are the first request's; completion tokens and latency are summed over
    the records. ``thinking_tokens`` is the server's own ``usage.completion_tokens_details.reasoning_tokens`` as the transport
    stored it; when the server does not report it, it is ``None`` with basis ``not-reported-by-server`` and
    ``reasoning_chars`` (the length of the dropped reasoning channel, when one came back) is the only proxy.
    """

    first = costs[0] if costs else {}
    thinking = _count(first.get("reasoning_tokens"))
    finish = first.get("finish_reason") if isinstance(first.get("finish_reason"), str) else None
    if finish is None and isinstance(finish_reason_fallback, str):
        finish = finish_reason_fallback
    forced = first.get("forced_close") if isinstance(first.get("forced_close"), bool) else None
    budget = _count(thinking_budget)
    completion_total = None
    latency_total = None
    for record in costs:
        tokens = _count(record.get("completion_tokens"))
        if tokens is not None:
            completion_total = (completion_total or 0) + tokens
        seconds = record.get("latency_seconds")
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and seconds >= 0:
            latency_total = (latency_total or 0.0) + float(seconds)
    return {
        "prompt_tokens": _count(first.get("prompt_tokens")),
        "completion_tokens": completion_total,
        "thinking_tokens": thinking,
        "thinking_tokens_basis": "server-usage-details" if thinking is not None else "not-reported-by-server",
        "reasoning_chars": _count(first.get("reasoning_chars")),
        "thinking_budget": budget,
        "finish_reason": finish,
        "forced_close": forced,
        "truncated": schema.is_truncated(
            finish_reason=finish, forced_close=forced, thinking_tokens=thinking, thinking_budget=budget,
        ),
        "answer_only_reask": len(costs) > 1,
        "latency_seconds": latency_total,
    }


def parse_answer(text: object, parser: Callable[[str], tuple] | None = None) -> dict[str, object] | None:
    """The answer fields as emitted: ``{"parsed", "scenario", "action", "entities"}``, or None for a non-text answer.

    ``parser`` is ``reproduction.slurp_scoring.parse_slu_response`` (the governed parser) unless a test injects another. A refusal
    to parse yields ``parsed: false`` with empty fields -- the trace records that nothing usable came back, never a guess.
    """

    if not isinstance(text, str):
        return None
    if parser is None:
        from ..reproduction.slurp_scoring import parse_slu_response  # noqa: PLC0415

        parser = parse_slu_response
    scenario, action, entities, ok = parser(text)
    if not ok:
        return {"parsed": False, "scenario": "", "action": "", "entities": []}
    return {
        "parsed": True,
        "scenario": scenario or "",
        "action": action or "",
        "entities": [{"type": str(entity.type), "filler": str(entity.filler)} for entity in entities],
    }


def supply_items(
    item_ids: Iterable[str], scores: Mapping[str, float] | None = None, used: Mapping[str, bool | None] | None = None,
    bases: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    """``supply_items`` of a contact line: supplied ids in supplied order, 1-based rank, retrieval score, usage and its basis.

    ``used_basis`` is ``json-form`` or ``content-pipe-form`` when ``used`` is a boolean (the item form the judgement read) and ``none``
    when ``used`` is null; ``bases`` supplies the form per item. A judged item without a form raises (the contact line is then dropped
    and logged by :func:`record_contact`).
    """

    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for item_id in item_ids:
        if not isinstance(item_id, str) or item_id == "" or item_id in seen:
            continue
        seen.add(item_id)
        score = None if scores is None else scores.get(item_id)
        flag = None if used is None else used.get(item_id)
        flag = flag if isinstance(flag, bool) else None
        basis = schema.USED_BASIS_NONE if flag is None else (None if bases is None else bases.get(item_id))
        if basis is None:
            raise ValueError(f"supply item {item_id!r} is judged but names no basis")
        out.append({
            "item_id": item_id, "rank": len(out) + 1,
            "score": float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
            "used": flag, "used_basis": basis,
        })
    return out


def _fold(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


_PIPE_TOKEN = re.compile(r"\S+\Z")


def _json_item_strings(content: str) -> tuple[str | None, list[str]]:
    try:
        document = json.loads(content.strip())
    except ValueError:
        return None, []
    if not isinstance(document, Mapping):
        return None, []
    intent = document.get("observed_intent")
    examples = document.get("observed_slot_examples")
    strings: list[str] = []
    if isinstance(examples, Mapping):
        for values in examples.values():
            if isinstance(values, list):
                strings.extend(value for value in values if isinstance(value, str) and value.strip())
    return (intent if isinstance(intent, str) and intent else None), strings


def _pipe_item_strings(content: str) -> tuple[str | None, list[str]]:
    """``<utterance> | <intent> | <slot>=<filler>; <slot>=<filler>`` (the real ``train-exemplars`` store form): ``(intent, fillers)``.

    The intent must be a single token (no whitespace), so prose that merely contains two pipes is not mistaken for an exemplar. A third
    part without any ``slot=filler`` pair is a valid exemplar with no slots.
    """

    parts = [part.strip() for part in content.split("|")]
    if len(parts) != 3 or not parts[1] or _PIPE_TOKEN.fullmatch(parts[1]) is None:
        return None, []
    strings: list[str] = []
    for chunk in parts[2].split(";"):
        _, separator, filler = chunk.partition("=")
        if separator and filler.strip():
            strings.append(filler.strip())
    return parts[1], strings


def item_form(content: object) -> tuple[str, str | None, list[str]]:
    """``(basis, intent, strings)`` of one supplied item's content.

    ``basis`` is ``json-form`` (an ``observed_intent`` / ``observed_slot_examples`` document), ``content-pipe-form`` (the pipe form of
    :func:`_pipe_item_strings`) or ``none`` (the content carries neither, so nothing can be compared). The JSON form is tried first.
    """

    if not isinstance(content, str):
        return schema.USED_BASIS_NONE, None, []
    intent, strings = _json_item_strings(content)
    if intent is not None or strings:
        return schema.USED_BASIS_JSON, intent, strings
    intent, strings = _pipe_item_strings(content)
    if intent is not None:
        return schema.USED_BASIS_PIPE, intent, strings
    return schema.USED_BASIS_NONE, None, []


def item_strings(content: object) -> tuple[str | None, list[str]]:
    """``(observed intent, slot example strings)`` of one supplied item's content, or ``(None, [])`` when it carries neither.

    Two forms are read: the JSON exemplar form ``{"observed_intent": "<scenario>_<action>", "observed_slot_examples": {"<type>": ["<string>", ...]}}``
    and the pipe form of the real ``train-exemplars`` store, ``<utterance> | <scenario>_<action> | <slot>=<filler>; ...``. Any other
    content (cards, prose) carries nothing this check can compare.
    """

    _, intent, strings = item_form(content)
    return intent, strings


def usage_flags(texts: Mapping[str, str], answer: Mapping[str, object] | None) -> dict[str, bool | None]:
    """For each supplied item, whether a string it carries appears in the parsed answer (None when not checkable).

    An item is used when its observed intent equals the answer's ``<scenario>_<action>``, or one of its slot example strings equals
    or occurs (on word boundaries, case- and punctuation-folded) in one of the answer's entity fillers. See :func:`usage_judgements`
    for the same flags together with the item form each judgement read.
    """

    return {item_id: flag for item_id, (flag, _) in usage_judgements(texts, answer).items()}


def usage_judgements(
    texts: Mapping[str, str], answer: Mapping[str, object] | None,
) -> dict[str, tuple[bool | None, str]]:
    """``{item id: (used, used_basis)}``; ``used`` is None exactly when ``used_basis`` is ``none``."""

    if not isinstance(answer, Mapping) or answer.get("parsed") is not True:
        return {item_id: (None, schema.USED_BASIS_NONE) for item_id in texts}
    label = f"{answer.get('scenario', '')}_{answer.get('action', '')}"
    fillers = [_fold(entity["filler"]) for entity in answer.get("entities", []) if isinstance(entity, Mapping)]
    flags: dict[str, tuple[bool | None, str]] = {}
    for item_id, content in texts.items():
        basis, intent, strings = item_form(content)
        if basis == schema.USED_BASIS_NONE:
            flags[item_id] = (None, schema.USED_BASIS_NONE)
            continue
        hit = intent == label
        for string in strings:
            needle = _fold(string)
            if needle and any(f" {needle} " in f" {filler} " for filler in fillers):
                hit = True
                break
        flags[item_id] = (hit, basis)
    return flags


_REQUEST_DERIVED_ID = re.compile(r".+-supply-\d+\Z")


def supplied_values_from_payload(payload: Mapping[str, object]) -> list[tuple[str, str]]:
    """``(value id, rendered content)`` of each value a request rendered as supplied knowledge, in rendered order.

    Read from ``supplied_evidence`` of the request payload. Key-facet parts (``knowledge_role`` ``key`` or an ``evidence_id``
    starting ``key:``), candidate proposals and request-derived ids (``<request id>-supply-<n>``, which name a position, not a
    value) are not supplied VALUES and are skipped.
    """

    out: list[tuple[str, str]] = []
    entries = payload.get("supplied_evidence") if isinstance(payload, Mapping) else None
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return out
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        evidence_id = entry.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            continue
        # ``tfrl-v2/``: the TFRL-v2 exemplar block (``rails.tfrl_v2_consumption.EXEMPLAR_EVIDENCE_ID``) is a worked-example text part, not a supplied value.
        if entry.get("knowledge_role") == "key" or evidence_id.startswith(("key:", "candidate-proposal:", "tfrl-v2/")):
            continue
        if _REQUEST_DERIVED_ID.fullmatch(evidence_id):
            continue
        content = entry.get("content")
        out.append((evidence_id, content if isinstance(content, str) else ""))
    return out


_NODE_BY_KIND = {
    "perception": "perception", "solver": "solver", "reviewer": "reviewer", "planner": "planner", "acquisition": "acquisition",
}


def node_of(role_kind: object) -> str:
    """The schema node of a native role kind; anything unrecognised is ``other``."""

    return _NODE_BY_KIND.get(role_kind, "other") if isinstance(role_kind, str) else "other"


def instance_for_slot(transport: object, id_slot: object) -> int | None:
    """The router's instance index for a slot (``id_slot mod N``); 0 for a single transport; None when unknown."""

    if isinstance(id_slot, bool) or not isinstance(id_slot, int) or id_slot < 0:
        return None
    count = getattr(transport, "instance_count", None)
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        return id_slot % count
    return 0


class RowSupplyScores:
    """Per-flight memory of the retrieval score of each supplied value, by (row, variant key, value id).

    Filled from the ``RankResult`` / ``FusionDecision`` pair a supply step already computed
    (``rails.knowledge_index.value_retrieval_ranks``: the best supporting facet's score); read when a contact line is written.
    The key ``None`` is the delivering / ordinary supply; a supply-variant arm or perception branch uses its own id.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scores: dict[tuple[str, str | None], dict[str, float]] = {}

    def note(
        self, row_id: str, key: str | None, *, rank_result: object, fusion_decision: object, values: Iterable[object],
    ) -> None:
        try:
            from ..rails.knowledge_index import value_retrieval_ranks  # noqa: PLC0415

            ranks = value_retrieval_ranks(rank_result=rank_result, fusion_decision=fusion_decision, values=tuple(values))
            found = {
                value_id: float(info["score"]) for value_id, info in ranks.items()
                if isinstance(info, Mapping) and isinstance(info.get("score"), (int, float))
            }
        except Exception as error:  # noqa: BLE001 -- observation only
            _LOG.warning("rollout trace supply scores skipped (%s: %s)", type(error).__name__, error)
            return
        with self._lock:
            self._scores.setdefault((row_id, key), {}).update(found)

    def lookup(self, row_id: str, key: str | None) -> dict[str, float]:
        """Scores for this key; a variant key falls back to the ordinary supply's scores for value ids it does not hold."""

        with self._lock:
            merged = dict(self._scores.get((row_id, None), {}))
            if key is not None:
                merged.update(self._scores.get((row_id, key), {}))
            return merged


def _id_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def record_contact(
    writer: RolloutTraceWriter | None, *, row_id: object, arm: object, node: str, role_id: object, contact_id: object,
    contact_index: object, sample_index: object, request_id: object, outcome: str, error_type: str | None,
    id_slot: object, instance: object, supply_variant_id: object, supply_values: Sequence[tuple[str, str]],
    supply_scores: Mapping[str, float] | None, costs: Sequence[Mapping[str, object]], decoding: Mapping[str, object] | None,
    finish_reason_fallback: object, answer_text: object, response_text_sha256: object, bank_receipt: object = None,
) -> bool:
    """Assemble and write one ``contact`` line. A no-op without a writer; never raises.

    ``bank_receipt`` is the TFRL-v2 bank receipt hash a Solver contact consumed (``rails.runner``); ``None`` -- written as ``null`` -- for every contact that consumed none."""

    if writer is None:
        return False
    try:
        decoding = decoding or {}
        facts = cost_facts(
            costs, thinking_budget=decoding.get("thinking_budget_tokens"), finish_reason_fallback=finish_reason_fallback,
        )
        max_tokens = _count(decoding.get("max_tokens"))
        answer = parse_answer(answer_text) if node == "solver" and outcome != "error" else None
        texts = {item_id: content for item_id, content in supply_values}
        judged = usage_judgements(texts, answer)
        used = {item_id: flag for item_id, (flag, _) in judged.items()}
        bases = {item_id: basis for item_id, (_, basis) in judged.items()}
        return writer.contact(
            row_id=_id_or_none(row_id), arm=_id_or_none(arm), node=node, role_id=_id_or_none(role_id),
            contact_id=_id_or_none(contact_id), contact_index=_count(contact_index), sample_index=_count(sample_index),
            request_id=_id_or_none(request_id), outcome=outcome, error_type=error_type,
            id_slot=_count(id_slot), instance=_count(instance), supply_variant_id=_id_or_none(supply_variant_id),
            supply_items=supply_items(list(texts), supply_scores, used, bases), bank_receipt=bank_receipt if isinstance(bank_receipt, str) else None,
            max_tokens=max_tokens, answer=answer,
            response_text_sha256=response_text_sha256 if isinstance(response_text_sha256, str) else None, **facts,
        )
    except Exception as error:  # noqa: BLE001
        _LOG.warning("rollout trace contact skipped (%s: %s)", type(error).__name__, error)
        return False


#: Re-exported for callers that only need the clock injection point in tests.
now_utc = _utc_now
