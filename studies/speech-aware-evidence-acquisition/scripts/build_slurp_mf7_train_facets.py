#!/usr/bin/env python3
"""CLI: rebuild MF-7's TRAIN key roster ``vectors/slurp-mf7/train-facets.jsonl``.

MODEL-FREE. Profile ``model-free-check``: zero frozen-core, tool, embedding,
GPU or network contacts. Only the governed TRAIN metadata is read; no DEV or
test row is named.

The roster is the input every MF-7b key pass reads through
``analysis.slurp_mf7b_vectors.train_facets`` (the Omni-Embed-Nemotron
``keys`` set behind ``vectors/slurp-kus3/facet-vectors.jsonl``). MF-7 left it
on the author's data disk and no repository script wrote it, so it is
re-derived here from what the study does pin:

* the keys are "the 11,514 TRAIN content facets of snapshot ``e27b2745...``
  -- one per TRAIN utterance, the utterance text as the key representation --
  each carrying its TRAIN gold intent (``scenario_action``) and its gold
  ``{type, filler}`` slots" (``docs/readiness/2026-09-06-slurp-retrieval-key-
  audit-v1.md`` section 3);
* that snapshot is the KB-BUILD v1 ``size-full`` build
  (``docs/receipts/knowledge/slurp-kb-size-full.json``). It is rebuilt in
  memory with the same ``rails.knowledge_build`` call and refused unless its
  content hash equals the pin, so the facet ids and their order are the ones
  the sealed ``vectors-content-facets.npz`` ids carried;
* the default ``exemplar`` view renders a content facet as the utterance
  verbatim, "what the MF-7b key vectors embedded"
  (``rails.knowledge_build._content_facet_representation``);
* the gold slots come from ``data.slurp.load_slurp_train_material``, the same
  lowercased token join the official scorer and the DEV gold side use, one
  entry per annotation in source order.

One JSON object per line, in snapshot content-facet order::

    {"facet_id": "facet:content:<slurp_id>", "text": "<utterance>",
     "intent": "<scenario>_<action>", "entities": [{"type": ..., "filler": ...}]}

Create-only: the output and its receipt (``train-facets.receipt.json`` beside
it) are refused if either exists. ``--check`` rebuilds and compares against
the file on disk, writing nothing.

Usage::

    python scripts/build_slurp_mf7_train_facets.py
    python scripts/build_slurp_mf7_train_facets.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from speech_aware_evidence_acquisition.data.lock import (  # noqa: E402
    data_root,
    load_lock,
    umbrella_lock_path,
)
from speech_aware_evidence_acquisition.data.slurp import load_slurp_train_material  # noqa: E402
from speech_aware_evidence_acquisition.e0.artifacts import write_json_artifact  # noqa: E402
from speech_aware_evidence_acquisition.rails.knowledge_build import (  # noqa: E402
    KnowledgeBuildSpec,
    build_knowledge_snapshot,
    governed_train_jsonl_relpath,
    slurp_kb_rows,
)

SEED = 20260823
SNAPSHOT_CONTENT_SHA256 = "e27b274516604ffc0e21319758aea99dc027f459d480a32a74a715b023a51b57"
EXPECTED_ROWS = 11514
OUTPUT_RELPATH = "vectors/slurp-mf7/train-facets.jsonl"
RECEIPT_NAME = "train-facets.receipt.json"
RECEIPT_SCHEMA = "saea-mf7-train-facets-receipt-v1"
DIMENSION = "content"


class TrainFacetsError(RuntimeError):
    """The rebuilt roster does not match what MF-7's keys are pinned to be."""


def facet_rows(snapshot, materials: Sequence[object]) -> list[dict[str, object]]:
    """One roster row per ``content`` facet, in snapshot order.

    ``materials`` are ``SlurpTrainMaterial`` records (``utterance_id``,
    ``sentence``, ``scenario``, ``action``, ``entities``). The facet's own
    representation must equal the utterance verbatim, so a snapshot rendered
    under any non-exemplar view is refused rather than silently re-keyed.
    """

    by_utterance: Mapping[str, object] = {str(m.utterance_id): m for m in materials}
    if len(by_utterance) != len(materials):
        raise TrainFacetsError("TRAIN material repeats an utterance id")
    prefix = f"facet:{DIMENSION}:"
    rows: list[dict[str, object]] = []
    for facet in snapshot.facets:
        if facet.dimension != DIMENSION:
            continue
        if not facet.facet_id.startswith(prefix):
            raise TrainFacetsError(f"content facet id {facet.facet_id!r} is not {prefix}<slurp_id>")
        material = by_utterance.get(facet.facet_id[len(prefix):])
        if material is None:
            raise TrainFacetsError(f"{facet.facet_id} has no TRAIN utterance")
        if facet.representation != material.sentence:
            raise TrainFacetsError(f"{facet.facet_id} is not rendered as its utterance verbatim")
        rows.append({
            "facet_id": facet.facet_id,
            "text": material.sentence,
            "intent": f"{material.scenario}_{material.action}",
            "entities": [{"type": entity.type, "filler": entity.filler} for entity in material.entities],
        })
    return rows


def render_jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def build(lock_path: Path | None = None) -> tuple[object, list[dict[str, object]], str]:
    lock = load_lock(lock_path if lock_path is not None else umbrella_lock_path(REPO_ROOT))
    root = data_root()
    started = time.monotonic()
    snapshot = build_knowledge_snapshot(
        KnowledgeBuildSpec(carrier_lock_key="slurp", datastore_size=None, seed=SEED).validate(),
        lock=lock, root=root, rows=slurp_kb_rows(lock, root), repo_root=REPO_ROOT,
    )
    if snapshot.content_hash() != SNAPSHOT_CONTENT_SHA256:
        raise TrainFacetsError(
            f"size-full snapshot rebuilds to {snapshot.content_hash()}, not the pinned {SNAPSHOT_CONTENT_SHA256}"
        )
    rows = facet_rows(snapshot, load_slurp_train_material(lock, root))
    if len(rows) != EXPECTED_ROWS:
        raise TrainFacetsError(f"expected {EXPECTED_ROWS:,} TRAIN facets, got {len(rows):,}")
    print(f"rebuilt snapshot {snapshot.snapshot_id} (content {SNAPSHOT_CONTENT_SHA256[:16]}... VERIFIED) and "
          f"{len(rows):,} facet rows in {time.monotonic() - started:.1f}s", flush=True)
    return snapshot, rows, governed_train_jsonl_relpath(lock, root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare a rebuild against the file on disk")
    parser.add_argument("--lock-path", type=Path, help="explicit umbrella datasets lock")
    args = parser.parse_args()

    root = data_root()
    output = root / OUTPUT_RELPATH
    receipt_path = output.parent / RECEIPT_NAME
    if not args.check and (output.exists() or receipt_path.exists()):
        raise SystemExit(f"refusing to overwrite {output} or {receipt_path}")
    snapshot, rows, train_relpath = build(args.lock_path)
    payload = render_jsonl(rows)
    digest = hashlib.sha256(payload).hexdigest()
    if args.check:
        on_disk = hashlib.sha256(output.read_bytes()).hexdigest()
        print(f"rebuild {digest}  on disk {on_disk}  match {digest == on_disk}")
        return 0 if digest == on_disk else 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(payload)
    train_path = root / train_relpath
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "generator": "scripts/build_slurp_mf7_train_facets.py",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "profile": "model-free-check",
        "accounting": {"frozen_core_contacts": 0, "tool_contacts": 0, "embedding_contacts": 0, "gpu_hours": 0.0},
        "snapshot": {
            "snapshot_id": snapshot.snapshot_id,
            "content_sha256": SNAPSHOT_CONTENT_SHA256,
            "build_receipt": "docs/receipts/knowledge/slurp-kb-size-full.json",
        },
        "source": {"relpath": train_relpath, "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()},
        "output": {
            "relpath": OUTPUT_RELPATH,
            "sha256": digest,
            "rows": len(rows),
            "entities": sum(len(row["entities"]) for row in rows),
        },
    }
    receipt_digest = write_json_artifact(receipt_path, receipt)
    print(f"wrote {output}  sha256={digest}  rows={len(rows):,}")
    print(f"wrote {receipt_path}  sha256={receipt_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
