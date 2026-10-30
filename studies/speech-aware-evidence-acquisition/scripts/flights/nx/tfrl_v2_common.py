#!/usr/bin/env python3
"""tfrl_v2_common.py -- shared pins, hashing, token measure, seeds and the cross-fit guard of the TFRL-v2 split / sampler / distiller (task T3).

Model-free and import-only. No network client, no model call, no card. Every function is pure over its arguments except the write-once file
helper.

What lives here (and why it is shared):

* the R01C pins (roster path, file sha256, split identity, the two fixed halves and their row counts), so the split script, the sampler and the
  distiller refuse the same moved roster;
* canonical JSON bytes and sha256 (the bank content hashes are over canonical compact JSON; file hashes are over the written bytes);
* the offline token measure: ``ceil(chars / CHARS_PER_TOKEN)`` with the conservative 3.5 chars per token of the L338 cost packages (no tokenizer
  contact). The real tokenizer lives behind the model server; the registered FIT path re-checks the exact prompt length at flight time;
* the K decoding seeds (a pure function of base seed, iteration, row id and k);
* the cross-fit guard: a bank distilled from half X may not be consumed by half X (plan section 9, ruling 3).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LOOP_ID = "SAEA-SLURP-TFRL-V2-NPU-PORT-V1"
DEFAULT_K = 4
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_BASE_SEED = 20261030

# ---- the construction column R01C and its two fixed halves (plan section 9, ruling 3) --------------------------------------------------------
ROSTER_RELPATH = "docs/receipts/rosters/slurp-stage3-construction-r01c.json"
ROSTER_ID = "SLURP-STAGE3-CONSTRUCTION-R01C"
ROSTER_FILE_SHA256 = "e8abc65ba4626cccb4aae9d2eb1bd73f077a6e9a407b3dba9c18bbe4c03eeddd"
ROSTER_SPLIT_IDENTITY = "f56bc386776421aec2d8001d8697611b01c9b0562885a462759b15531e2f4515"
ROSTER_ROWS = 625
HALF_H1 = "R01C-H1"
HALF_H2 = "R01C-H2"
HALVES = (HALF_H1, HALF_H2)
HALF_ROWS = {HALF_H1: 313, HALF_H2: 312}
HALF_ROSTER_ID = {h: f"{ROSTER_ID}-{h.split('-')[1]}" for h in HALVES}
HALF_ROSTER_RELPATH = {
    HALF_H1: "docs/receipts/rosters/slurp-stage3-construction-r01c-h1.json",
    HALF_H2: "docs/receipts/rosters/slurp-stage3-construction-r01c-h2.json",
}
SPLIT_RECEIPT_RELPATH = "docs/receipts/rosters/slurp-stage3-construction-r01c-halves-split-receipt.json"

# ---- caps (plan section 9, ruling 4) and the vote (plan section 4) --------------------------------------------------------------------------
EXEMPLAR_BLOCK_TOKEN_CAP = 2000
RULE_BANK_TOKEN_CAP = 400
VOTE_MIN_ROWS = 3
MAX_EXEMPLARS_PER_SCENARIO = 8
MAX_EVIDENCE_ITEMS_PER_EXEMPLAR = 6
CHARS_PER_TOKEN = 3.5

# ---- token measures and exemplar text modes (task T3b); the measure classes live in tfrl_v2_text ---------------------------------------------
MEASURE_ESTIMATE = "estimate"             # ceil(chars / 3.5): offline estimate; a bank counted with it is refused by the plan builder
MEASURE_VLLM = "vllm-tokenize"            # the FIT gate's vLLM tokenize measure
EXEMPLAR_TEXT_RESOLVED = "resolved"       # exemplars quote the knowledge-store text of the supplied items
EXEMPLAR_TEXT_IDS_ONLY = "ids-only"       # legacy: exemplars name item ids only
FOLD_B_CONSUMER = "FOLD-B"                # the consumer id of the union bank (never a construction half)

# ---- budget constants given by the planner ---------------------------------------------------------------------------------------------------
MEDIAN_COMPLETION_TOKENS = 4372
HOST_OUTPUT_TOK_S = 1422.59          # host output tokens per second at 32 sequences per instance (npu-port synthetic speed baseline)
HOST_CARDS = 8
MARGIN = 1.3


class CommonError(ValueError):
    """A refusal of the shared helpers (a write-once clash, a malformed id)."""


class BankNotMeasuredError(CommonError):
    """A bank receipt whose counts are estimates, or whose exemplars do not carry the resolved item text: it cannot enter a flight."""


class CrossFitError(CommonError):
    """The consuming half equals, or overlaps, the half the bank was distilled from."""


def other_half(half: str) -> str:
    if half not in HALVES:
        raise CommonError(f"unknown half {half!r}; known: {list(HALVES)}")
    return HALF_H2 if half == HALF_H1 else HALF_H1


# ---- bytes and hashes ------------------------------------------------------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Compact, key-sorted, UTF-8 JSON text (no trailing newline): the form a content hash is taken over."""

    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def file_bytes(obj: Any) -> bytes:
    """The bytes a JSON document is written as: indented, key-sorted, newline-terminated (the roster writer's form)."""

    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(lines: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(line) + "\n" for line in lines).encode("utf-8")


def write_once(path: Path, data: bytes) -> str:
    """Write ``data`` to ``path`` unless it exists; an existing file must hold exactly these bytes.

    Returns ``"written"`` or ``"unchanged"``; raises :class:`CommonError` when the file exists with other bytes (never overwrites).
    """

    target = Path(path)
    if target.exists():
        if target.read_bytes() == data:
            return "unchanged"
        raise CommonError(f"refusing to overwrite {target}: it exists with different bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return "written"


# ---- tokens ----------------------------------------------------------------------------------------------------------------------------------


def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Upper-bound token estimate of ``text``: ``ceil(len / chars_per_token)``; 0 for the empty string."""

    if chars_per_token <= 0:
        raise CommonError("chars_per_token must be positive")
    return int(math.ceil(len(text) / chars_per_token)) if text else 0


# ---- seeds -----------------------------------------------------------------------------------------------------------------------------------


def iteration_seed(base_seed: int, iteration: int) -> int:
    """The base seed of one iteration: a pure function of the loop's base seed and the iteration number."""

    return _seed_from(f"tfrl-v2-iteration|{int(base_seed)}|{int(iteration)}")


def derive_seed(base_seed: int, row_id: str, k: int) -> int:
    """The decoding seed of trajectory ``k`` of ``row_id``: a pure function of (base seed, row id, k), in ``[1, 2**31 - 2]``."""

    return _seed_from(f"tfrl-v2-seed|{int(base_seed)}|{row_id}|{int(k)}")


def _seed_from(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 2) + 1


# ---- cross-fit guard -------------------------------------------------------------------------------------------------------------------------


def check_cross_fit(
    bank_receipt: Mapping[str, Any], consuming_half: str, *,
    consuming_row_ids: Sequence[str] | None = None, consuming_split_identity: str | None = None,
) -> None:
    """Refuse when the bank was distilled from (or contains rows of) the half that would consume it.

    The bank receipt names ``source_halves`` (the halves whose positives built the bank), their split identities and the sorted
    ``source_row_ids`` that contributed. Raises :class:`CrossFitError` when

    * ``consuming_half`` is one of the source halves, or
    * ``consuming_split_identity`` equals a source split identity, or
    * ``consuming_row_ids`` intersects ``source_row_ids``.

    The union bank (both halves as sources) is consumed only by Fold B / Fold A flights, never by a construction half.
    """

    sources = bank_receipt.get("source_halves")
    if not isinstance(sources, list) or not sources:
        raise CrossFitError("the bank receipt names no source_halves")
    if consuming_half in sources:
        raise CrossFitError(f"cross-fit guard: the bank was distilled from {sources}, which includes the consuming half {consuming_half}")
    identities = bank_receipt.get("source_split_identities") or {}
    if consuming_split_identity is not None and consuming_split_identity in set(identities.values()):
        raise CrossFitError("cross-fit guard: the consuming split identity equals a source split identity of the bank")
    if consuming_row_ids is not None:
        overlap = sorted(set(consuming_row_ids) & set(bank_receipt.get("source_row_ids") or []))
        if overlap:
            raise CrossFitError(f"cross-fit guard: {len(overlap)} consuming rows are source rows of the bank (first: {overlap[0]})")


def check_bank_measured(bank_receipt: Mapping[str, Any]) -> None:
    """Refuse a bank receipt that a flight may not consume: its token counts must come from the vLLM tokenize measure and its exemplars must quote
    the resolved supplied-item text (``measure == "vllm-tokenize"`` and ``exemplar_text == "resolved"``)."""

    measure = bank_receipt.get("measure")
    if measure != MEASURE_VLLM:
        raise BankNotMeasuredError(f"the bank's token counts were produced by measure {measure!r}, not {MEASURE_VLLM!r}: re-measure it with the vLLM tokenize measure")
    if bank_receipt.get("exemplar_text") != EXEMPLAR_TEXT_RESOLVED:
        raise BankNotMeasuredError("the bank's exemplars do not carry the resolved supplied-item text (exemplar_text must be 'resolved')")


def check_union_for_fold_b(
    bank_receipt: Mapping[str, Any], fold_b_row_ids: Sequence[str], *, half_row_ids: Mapping[str, Sequence[str]] | None = None,
    fold_b_split_identity: str | None = None, require_measured: bool = True,
) -> None:
    """The Fold B flight consumes the UNION bank only: both halves as sources, Fold B rows disjoint from every source row (and, when the full row
    lists of the halves are given, from every row of either half). A construction half is refused by :func:`check_cross_fit` (it is a source)."""

    if bank_receipt.get("kind") != "union" or sorted(bank_receipt.get("source_halves") or []) != sorted(HALVES):
        raise CrossFitError("the Fold B flight consumes the union bank of both halves; this receipt is not one")
    check_cross_fit(bank_receipt, FOLD_B_CONSUMER, consuming_row_ids=fold_b_row_ids, consuming_split_identity=fold_b_split_identity)
    if half_row_ids is not None:
        every = {r for half in HALVES for r in half_row_ids.get(half, ())}
        overlap = sorted(set(fold_b_row_ids) & every)
        if overlap:
            raise CrossFitError(f"cross-fit guard: {len(overlap)} Fold B rows are rows of a construction half (first: {overlap[0]})")
    if require_measured:
        check_bank_measured(bank_receipt)
