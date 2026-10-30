#!/usr/bin/env python3
"""tfrl_v2_split.py -- the deterministic half-split of the construction column R01C (TFRL-v2, plan section 9 ruling 3, task T3).

R01C (625 rows) is split into two FIXED halves by roster order. Mode ``contiguous`` (the DEFAULT, the ruling as written): ``R01C-H1`` is the first
313 items of the roster file, ``R01C-H2`` the remaining 312. Mode ``interleave`` (an alternative offered for the planner's decision, never the
default): H1 = even positions (313 items), H2 = odd positions (312 items). No draw, no seed, no re-ranking: every item is the parent's item verbatim.

FINDING (measured on the pinned roster): the roster is sorted by ``sample_id`` and SLURP train ids are scenario-clustered, so the contiguous halves
are scenario-disjoint almost everywhere (H1 has 25 alarm rows and H2 has 1; H2 has 31 cooking rows and H1 has none). A scenario-keyed exemplar bank
distilled from the other half therefore has little to offer a half. Both modes report ``cross_fit_stratum_coverage`` (the share of a half's rows
whose stratum has at least 3 rows in the other half) in the split receipt so the planner can read it before the first flight. The split is pinned: the parent roster's file sha256 and split
identity are constants (``tfrl_v2_common``), and the script REFUSES when the roster on disk is not those bytes.

Outputs (write-once; a re-run with the same inputs is a no-op, any difference is a refusal):

* two roster receipts in the same format as R01C (``saea-roster-receipt-v1``; ``core.roster.verify_roster`` accepts them), each with its own
  ``split_identity_hash`` and a ``parent`` block;
* one split receipt with the counts, the file sha256 of every output, the item-id hashes and the disjoint-union proof.

MODEL-FREE: reads one roster file, writes three JSON files. No model, tokenizer, encoder, network or card contact; no utterance text, label or
score is read (only ids, strata, audio seconds and media fields already in the roster).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _path in (str(REPO_ROOT / "src"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from speech_aware_evidence_acquisition.core import roster as core_roster  # noqa: E402
import tfrl_v2_common as cm  # noqa: E402

SPLIT_SCHEMA = "saea-tfrl-v2-r01c-halves-split-receipt-v1"
MODES = ("contiguous", "interleave")
RULES = {
    "contiguous": "H1 = items[0:313] of the R01C roster file in file order, H2 = items[313:625]; no draw, no seed, no re-ranking",
    "interleave": "H1 = items[0::2] (even positions, 313 items), H2 = items[1::2] (odd positions, 312 items) of the R01C roster file; no draw, no seed",
}
MIN_STRATUM_ROWS = cm.VOTE_MIN_ROWS


class SplitError(ValueError):
    """The roster moved, is malformed, or an output clashes with an existing file."""


def load_parent(path: Path, *, expected_sha256: str = cm.ROSTER_FILE_SHA256, expected_split_identity: str = cm.ROSTER_SPLIT_IDENTITY,
                expected_rows: int = cm.ROSTER_ROWS) -> tuple[dict[str, Any], str]:
    """Read the R01C roster, refuse unless its file sha256 and split identity are the pinned ones. Returns ``(document, file_sha256)``."""

    raw = Path(path).read_bytes()
    measured = cm.sha256_bytes(raw)
    if measured != expected_sha256:
        raise SplitError(f"the roster hash moved: pinned {expected_sha256}, found {measured} at {path}")
    document = json.loads(raw.decode("utf-8"))
    try:
        identity = core_roster.verify_roster(document)
    except core_roster.RosterError as error:
        raise SplitError(f"the roster does not verify: {error}") from error
    if identity != expected_split_identity:
        raise SplitError(f"the split identity moved: pinned {expected_split_identity}, found {identity}")
    items = document["items"]
    if len(items) != expected_rows:
        raise SplitError(f"the roster has {len(items)} items, expected {expected_rows}")
    return document, measured


def _strata(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[str(item["stratum"])] = counts.get(str(item["stratum"]), 0) + 1
    return dict(sorted(counts.items()))


def split_items(items: Sequence[Mapping[str, Any]], h1_rows: int = cm.HALF_ROWS[cm.HALF_H1], mode: str = "contiguous") -> tuple[list[dict], list[dict]]:
    """The two halves of ``items`` by file order (verbatim item copies); ``mode`` is ``contiguous`` or ``interleave``."""

    if mode not in MODES:
        raise SplitError(f"unknown split mode {mode!r}; known: {list(MODES)}")
    if mode == "interleave":
        return [dict(item) for item in items[0::2]], [dict(item) for item in items[1::2]]
    if not 0 < h1_rows < len(items):
        raise SplitError("h1_rows must leave both halves non-empty")
    return [dict(item) for item in items[:h1_rows]], [dict(item) for item in items[h1_rows:]]


def stratum_coverage(own: Mapping[str, int], other: Mapping[str, int], minimum: int = MIN_STRATUM_ROWS) -> float:
    """The share of a half's rows whose stratum has at least ``minimum`` rows in the other half (what a cross-fitted bank can cover)."""

    total = sum(own.values())
    covered = sum(n for stratum, n in own.items() if other.get(stratum, 0) >= minimum)
    return round(covered / total, 6) if total else 0.0


def _half_receipt(parent: Mapping[str, Any], parent_sha: str, half: str, items: list[dict], sibling: Mapping[str, Any], mode: str) -> dict[str, Any]:
    ids = [str(item["sample_id"]) for item in items]
    identity = core_roster.split_identity_hash(core_roster.prefixed(str(parent["carrier_lock_key"]), ids))
    strata = _strata(items)
    receipt: dict[str, Any] = {
        "schema": parent["schema"],
        "roster_id": cm.HALF_ROSTER_ID[half],
        "carrier_lock_key": parent["carrier_lock_key"],
        "split_role": parent["split_role"],
        "roster_role": parent["roster_role"],
        "seed": parent["seed"],
        "per_stratum": dict(strata),
        "strata_counts": dict(strata),
        "items": items,
        "excluded_count": 0,
        "split_identity_hash": identity,
        "half": half,
        "authority": ("TFRL-v2 plan 2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md section 9 ruling 3: the construction column is split into two "
                      "fixed halves by roster order so that the bank a half consumes is distilled from the OTHER half's positives (cross-fit)."),
        "draw_method": ("No draw. " + RULES[mode] + ". The seed is the parent's and is carried for provenance only."),
        "selection_rule": f"{RULES[mode]} (this receipt: {half})",
        "split_mode": mode,
        "parent": {
            "roster_id": parent["roster_id"], "relpath": cm.ROSTER_RELPATH, "file_sha256": parent_sha,
            "split_identity_hash": parent["split_identity_hash"], "count": len(parent["items"]),
        },
        "sibling": dict(sibling),
        "d1_supplement": parent["d1_supplement"],
    }
    core_roster.verify_roster(receipt)
    return receipt


def build_halves(parent: Mapping[str, Any], parent_sha: str, h1_rows: int = cm.HALF_ROWS[cm.HALF_H1], mode: str = "contiguous") -> tuple[dict, dict, dict]:
    """``(h1_receipt, h2_receipt, split_receipt)`` as plain documents; nothing is written."""

    carrier = str(parent["carrier_lock_key"])
    items1, items2 = split_items(parent["items"], h1_rows, mode)
    ids1 = [str(i["sample_id"]) for i in items1]
    ids2 = [str(i["sample_id"]) for i in items2]
    if set(ids1) & set(ids2):
        raise SplitError("the halves are not disjoint")
    identity1 = core_roster.split_identity_hash(core_roster.prefixed(carrier, ids1))
    identity2 = core_roster.split_identity_hash(core_roster.prefixed(carrier, ids2))
    r1 = _half_receipt(parent, parent_sha, cm.HALF_H1, items1, {"roster_id": cm.HALF_ROSTER_ID[cm.HALF_H2], "split_identity_hash": identity2, "count": len(items2)}, mode)
    r2 = _half_receipt(parent, parent_sha, cm.HALF_H2, items2, {"roster_id": cm.HALF_ROSTER_ID[cm.HALF_H1], "split_identity_hash": identity1, "count": len(items1)}, mode)
    file1, file2 = core_roster.canonical_receipt_bytes(r1), core_roster.canonical_receipt_bytes(r2)
    parent_ids = [str(i["sample_id"]) for i in parent["items"]]
    summary = {
        "schema": SPLIT_SCHEMA,
        "loop_id": cm.LOOP_ID,
        "split_mode": mode,
        "rule": RULES[mode],
        "parent": {"roster_id": parent["roster_id"], "relpath": cm.ROSTER_RELPATH, "file_sha256": parent_sha,
                   "split_identity_hash": parent["split_identity_hash"], "count": len(parent_ids),
                   "item_ids_sha256": cm.content_sha256(parent_ids)},
        "halves": {
            cm.HALF_H1: {"roster_id": r1["roster_id"], "relpath": cm.HALF_ROSTER_RELPATH[cm.HALF_H1], "count": len(ids1),
                         "split_identity_hash": identity1, "file_sha256": cm.sha256_bytes(file1), "item_ids_sha256": cm.content_sha256(ids1),
                         "first_sample_id": ids1[0], "last_sample_id": ids1[-1], "strata_counts": r1["strata_counts"],
                         "cross_fit_stratum_coverage": stratum_coverage(r1["strata_counts"], r2["strata_counts"])},
            cm.HALF_H2: {"roster_id": r2["roster_id"], "relpath": cm.HALF_ROSTER_RELPATH[cm.HALF_H2], "count": len(ids2),
                         "split_identity_hash": identity2, "file_sha256": cm.sha256_bytes(file2), "item_ids_sha256": cm.content_sha256(ids2),
                         "first_sample_id": ids2[0], "last_sample_id": ids2[-1], "strata_counts": r2["strata_counts"],
                         "cross_fit_stratum_coverage": stratum_coverage(r2["strata_counts"], r1["strata_counts"])},
        },
        "proof": {"disjoint": True, "union_equals_parent": sorted(ids1 + ids2) == sorted(parent_ids), "counts_sum": len(ids1) + len(ids2) == len(parent_ids),
                  "halves_keep_parent_order": [i for i in parent_ids if i in set(ids1)] == ids1 and [i for i in parent_ids if i in set(ids2)] == ids2,
                  "model_calls": 0},
    }
    if not (summary["proof"]["union_equals_parent"] and summary["proof"]["halves_keep_parent_order"]):
        raise SplitError("the union of the halves is not the parent, or a half does not keep the parent's order")
    return r1, r2, summary


def run_split(roster_path: Path, out_dir: Path, *, expected_sha256: str = cm.ROSTER_FILE_SHA256,
              expected_split_identity: str = cm.ROSTER_SPLIT_IDENTITY, expected_rows: int = cm.ROSTER_ROWS,
              h1_rows: int = cm.HALF_ROWS[cm.HALF_H1], mode: str = "contiguous", dry_run: bool = False) -> dict[str, Any]:
    """Split and write (write-once). Returns the split receipt; with ``dry_run`` nothing is written."""

    parent, parent_sha = load_parent(roster_path, expected_sha256=expected_sha256, expected_split_identity=expected_split_identity,
                                     expected_rows=expected_rows)
    r1, r2, summary = build_halves(parent, parent_sha, h1_rows, mode)
    if dry_run:
        return summary
    out = Path(out_dir)
    targets = {
        out / Path(cm.HALF_ROSTER_RELPATH[cm.HALF_H1]).name: core_roster.canonical_receipt_bytes(r1),
        out / Path(cm.HALF_ROSTER_RELPATH[cm.HALF_H2]).name: core_roster.canonical_receipt_bytes(r2),
        out / Path(cm.SPLIT_RECEIPT_RELPATH).name: cm.file_bytes(summary),
    }
    for path, data in targets.items():       # refuse before writing anything if any target clashes
        if path.exists() and path.read_bytes() != data:
            raise SplitError(f"refusing to overwrite {path}: it exists with different bytes")
    for path, data in targets.items():
        cm.write_once(path, data)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--roster", default=str(REPO_ROOT / cm.ROSTER_RELPATH))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "docs" / "receipts" / "rosters"))
    parser.add_argument("--mode", choices=MODES, default="contiguous", help="contiguous is the ruling; interleave is an alternative for the planner")
    parser.add_argument("--dry-run", action="store_true", help="compute and print the split receipt; write nothing")
    args = parser.parse_args(argv)
    try:
        summary = run_split(Path(args.roster), Path(args.out_dir), mode=args.mode, dry_run=args.dry_run)
    except (SplitError, cm.CommonError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
