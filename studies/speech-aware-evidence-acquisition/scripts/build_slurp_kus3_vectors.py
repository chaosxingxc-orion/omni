#!/usr/bin/env python3
"""CLI: convert the sealed MF-7b Omni-Embed-Nemotron vectors into the
``sealed-vectors`` schema cell KU-S3 flies, and build the cell's form twin.

MODEL-FREE. Profile ``model-free-check``: zero frozen-core contacts, zero tool
contacts, zero embedding contacts, zero GPU, zero network, zero paid spend. No
new embedding pass happens here and none may: every number this writes is a
re-encoding of bytes the registered pass
``SAEA-SLURP-MF7B-NEMOTRON-EMBEDDING-V1`` already produced on 2026-09-06, and
each source file is verified against that pass's own receipt before a line of
it is read.

Why a conversion is needed at all
---------------------------------
``rails.phase_supply.load_sealed_vectors`` reads a JSONL whose every line is
``{"id": ..., "vectors": {"<dimension>": [...]}}`` -- one file may carry more
than one KB dimension, and it refuses any other shape. The MF-7b audit wrote
``{"id": ..., "vector": [...]}``, because its own consumer
(``analysis.slurp_retrieval_key_audit``) is dimension-free. Same floats, same
order, same ids; one key renamed and one nesting level added.

What is written (all under ``$SPEECHRL_DATA_DIR/vectors/slurp-kus3/``)
---------------------------------------------------------------------
``facet-vectors.jsonl``
    11,514 TRAIN utterance keys under dimension ``content``, ids
    ``facet:content:<slurp_id>`` -- already exactly the content facet ids of
    the KB snapshot, which is why ``kb-facet-v1`` needs no new provider.
``query-vectors-speech.jsonl``
    the 1,000 DEV-1000 speech queries (key (a), ``q-speech-nemotron``).
``query-vectors-speech-state.jsonl``
    the same 1,000 rows fused with their sealed KU-S1c draft text inside the
    encoder (key (b), ``q-speech-nemotron-fused``).
``query-vectors-speech-twin.jsonl``
    THE FORM TWIN. Each DEV row is given ANOTHER DEV row's speech vector,
    drawn by a deterministic derangement inside the bucket of rows the sealed
    closed-list baseline predicted the SAME scenario for. The retrieved
    neighbours are therefore real TRAIN utterances, of the row's own predicted
    scenario, rendered by the same template in the same number -- and aligned
    to a different recording. One factor moves: whether the speech that keyed
    the retrieval is this row's speech. No gold is read: the bucket comes from
    the sealed prediction, which is the same information the phase's schema
    block already conditions on.

Usage::

    python scripts/build_slurp_kus3_vectors.py
    python scripts/build_slurp_kus3_vectors.py --check   # verify, write nothing
    python scripts/build_slurp_kus3_vectors.py --facets-only --receipt-dir <dir>

``--facets-only`` converts the TRAIN key set alone into ``facet-vectors.jsonl``. That is the only kus3 file
a cell whose queries come from elsewhere binds; Fold B's solver ``facet_source`` is an example. The DEV-1000
query sets, the form twin and the sealed closed-list read are skipped. The file's bytes depend only on the
MF-7b ``nemotron-train-text`` set, so a narrowed run writes the same ``facet-vectors.jsonl`` as a full run
over the same vectors. Its receipt must go to its own ``--receipt-dir``, never over the historical full
receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from speech_aware_evidence_acquisition.data.lock import data_root  # noqa: E402
from speech_aware_evidence_acquisition.e0.artifacts import write_json_artifact  # noqa: E402
from speech_aware_evidence_acquisition.rails.phase_supply import (  # noqa: E402
    load_sealed_vectors,
)

MF7B_DIR_RELPATH = "vectors/slurp-mf7b"
MF7B_RECEIPT_NAME = "nemotron.receipt.json"
MF7B_LEDGER_ROW = "SAEA-SLURP-MF7B-NEMOTRON-EMBEDDING-V1"
OUT_DIR_RELPATH = "vectors/slurp-kus3"
DIMENSION = "content"
LEDGER_ROW = "SAEA-SLURP-KU-S3-VECTORS-V1"
RECEIPT_DIR = (
    REPO_ROOT
    / "docs"
    / "checks"
    / "speech-aware-evidence-acquisition"
    / "2026-09-07-slurp-kus3-vectors"
)
SEALED_CLOSED_LIST_OUTPUTS_RELPATH = (
    "runs/SAEA-SLURP-DEV1000-CLOSED-LIST-DIRECT-V1/"
    "attempt-20260905T060424Z-2c5596/"
    "SAEA-SLURP-DEV1000-CLOSED-LIST-DIRECT-V1.outputs.jsonl"
)
UNPARSED_BUCKET = "(unparsed)"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_rows(path: Path) -> Iterator[tuple[str, list[float]]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            document = json.loads(line)
            if set(document) != {"id", "vector"}:
                raise SystemExit(
                    f"{path} line {number}: expected exactly 'id' and 'vector', got "
                    f"{sorted(document)}"
                )
            yield str(document["id"]), list(document["vector"])


def _write_sealed(path: Path, rows: Sequence[tuple[str, Sequence[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry_id, vector in rows:
            handle.write(
                json.dumps(
                    {"id": entry_id, "vectors": {DIMENSION: list(vector)}},
                    ensure_ascii=False,
                    separators=(", ", ": "),
                )
            )
            handle.write("\n")


def _l2_stats(rows: Sequence[tuple[str, Sequence[float]]]) -> dict[str, float]:
    norms = [math.sqrt(sum(float(v) * float(v) for v in vector)) for _id, vector in rows]
    return {"min": min(norms), "max": max(norms), "mean": sum(norms) / len(norms)}


def _predicted_scenarios(root: Path) -> dict[str, str]:
    """``{sample_id: predicted scenario}`` from the SEALED closed-list bytes.

    The prediction, never gold: the same sealed response the phase's own
    ``sealed-manifest`` predecessor binding appends, read here only to bucket
    the twin's permutation. A response that does not parse, or carries no
    scenario, goes to its own bucket rather than being guessed at.
    """

    path = root / SEALED_CLOSED_LIST_OUTPUTS_RELPATH
    buckets: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scenario = UNPARSED_BUCKET
        try:
            parsed = json.loads(row["text"])
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, Mapping):
            value = parsed.get("scenario")
            if isinstance(value, str) and value.strip():
                scenario = value.strip().casefold()
        buckets[str(row["sample_id"])] = scenario
    return buckets


def _derangement(buckets: Mapping[str, str]) -> dict[str, str]:
    """``{sample_id: the sample whose speech vector it receives}``.

    Deterministic: inside each predicted-scenario bucket the ids are sorted
    and rotated by one, which is a derangement for every bucket of two or
    more. Buckets of exactly one are pooled into a residual list, sorted and
    rotated the same way, so no row can ever be given its own vector; the
    residual is reported so a reader knows how many rows crossed a bucket.
    """

    by_scenario: dict[str, list[str]] = {}
    for sample_id, scenario in buckets.items():
        by_scenario.setdefault(scenario, []).append(sample_id)
    mapping: dict[str, str] = {}
    residual: list[str] = []
    for scenario in sorted(by_scenario):
        members = sorted(by_scenario[scenario])
        if len(members) < 2:
            residual.extend(members)
            continue
        for index, sample_id in enumerate(members):
            mapping[sample_id] = members[(index + 1) % len(members)]
    residual.sort()
    if len(residual) == 1:
        # One orphan cannot be deranged against itself: give it the first id of
        # the largest bucket, and take that bucket's own donor for it.
        largest = max(sorted(by_scenario), key=lambda name: (len(by_scenario[name]), name))
        mapping[residual[0]] = sorted(by_scenario[largest])[0]
    elif residual:
        for index, sample_id in enumerate(residual):
            mapping[sample_id] = residual[(index + 1) % len(residual)]
    self_maps = sorted(s for s, donor in mapping.items() if s == donor)
    if self_maps:
        raise SystemExit(f"twin permutation maps {len(self_maps)} row(s) to themselves")
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the sources and the already-written outputs; write nothing new",
    )
    parser.add_argument(
        "--facets-only",
        action="store_true",
        help="convert only the TRAIN key set into facet-vectors.jsonl (no DEV query set, no twin)",
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=RECEIPT_DIR,
        help="where vectors.json is written (a --facets-only run must not use the historical one)",
    )
    parser.add_argument(
        "--mf7b-row",
        default=MF7B_LEDGER_ROW,
        help="the exposure row the MF-7b receipt must name (a re-run of MF-7b registers its own)",
    )
    args = parser.parse_args()
    receipt_dir = args.receipt_dir if args.receipt_dir.is_absolute() else REPO_ROOT / args.receipt_dir
    if args.facets_only and receipt_dir.resolve() == RECEIPT_DIR.resolve():
        parser.error("--facets-only must write its receipt to its own --receipt-dir")

    root = data_root()
    source_dir = root / MF7B_DIR_RELPATH
    out_dir = root / OUT_DIR_RELPATH
    receipt = json.loads((source_dir / MF7B_RECEIPT_NAME).read_text(encoding="utf-8"))
    if receipt.get("exposure_ledger_row") != args.mf7b_row:
        raise SystemExit(
            f"{MF7B_RECEIPT_NAME} names exposure row "
            f"{receipt.get('exposure_ledger_row')!r}, not {args.mf7b_row!r}"
        )

    plan = (
        ("nemotron-train-text", "facet-vectors.jsonl", "facet"),
        ("q-speech-nemotron", "query-vectors-speech.jsonl", "query"),
        ("q-speech-nemotron-fused", "query-vectors-speech-state.jsonl", "query"),
    )
    if args.facets_only:
        plan = plan[:1]
    sets: dict[str, object] = {}
    speech_rows: list[tuple[str, list[float]]] = []
    started = time.monotonic()
    for set_name, out_name, role in plan:
        declared = receipt["sets"][set_name]
        source = source_dir / declared["file"]
        measured = _sha256(source)
        if measured != declared["sha256"]:
            raise SystemExit(
                f"{source} hashes to {measured}, not the MF-7b receipt's "
                f"{declared['sha256']}"
            )
        rows = list(_iter_rows(source))
        if len(rows) != int(declared["count"]):
            raise SystemExit(
                f"{source} carries {len(rows)} rows, not the receipt's {declared['count']}"
            )
        if set_name == "q-speech-nemotron":
            speech_rows = rows
        out_path = out_dir / out_name
        if not args.check:
            _write_sealed(out_path, rows)
        sets[out_name] = {
            "role": role,
            "source_set": set_name,
            "source_file": declared["file"],
            "source_sha256": declared["sha256"],
            "count": len(rows),
            "dims": len(rows[0][1]),
            "dimension": DIMENSION,
            "l2_norms": _l2_stats(rows),
            "sha256": _sha256(out_path),
            "bytes": out_path.stat().st_size,
        }
        print(f"{out_name}: {len(rows)} rows, sha256 {sets[out_name]['sha256'][:16]}...")

    twin_rows: list[tuple[str, list[float]]] = []
    if not args.facets_only:
        _form_twin(root, out_dir, receipt, speech_rows, sets, twin_rows, check=args.check)

    readback = _readback(out_dir, source_dir, sets, twin_rows)
    document = _receipt_document(receipt, sets, readback, started, facets_only=args.facets_only)
    receipt_sha256 = write_json_artifact(receipt_dir / "vectors.json", document)
    try:
        shown = receipt_dir.relative_to(REPO_ROOT)
    except ValueError:
        shown = receipt_dir
    print(f"wrote {shown}/vectors.json sha256={receipt_sha256[:16]}...")
    return 0


def _form_twin(root, out_dir, receipt, speech_rows, sets, twin_rows, *, check) -> None:
    # ---- the form twin ----------------------------------------------------
    buckets = _predicted_scenarios(root)
    missing = sorted({sample for sample, _v in speech_rows} - set(buckets))
    if missing:
        raise SystemExit(f"{len(missing)} query rows have no sealed closed-list row")
    mapping = _derangement({sample: buckets[sample] for sample, _v in speech_rows})
    by_id = {sample: vector for sample, vector in speech_rows}
    twin_rows.extend((sample, by_id[mapping[sample]]) for sample, _v in speech_rows)
    twin_path = out_dir / "query-vectors-speech-twin.jsonl"
    if not check:
        _write_sealed(twin_path, twin_rows)
    bucket_sizes = {}
    for scenario in sorted(set(buckets.values())):
        bucket_sizes[scenario] = sum(1 for v in buckets.values() if v == scenario)
    sets["query-vectors-speech-twin.jsonl"] = {
        "role": "query",
        "source_set": "q-speech-nemotron",
        "source_file": receipt["sets"]["q-speech-nemotron"]["file"],
        "source_sha256": receipt["sets"]["q-speech-nemotron"]["sha256"],
        "count": len(twin_rows),
        "dims": len(twin_rows[0][1]),
        "dimension": DIMENSION,
        "l2_norms": _l2_stats(twin_rows),
        "sha256": _sha256(twin_path),
        "bytes": twin_path.stat().st_size,
        "permutation": {
            "rule": "saea-kus3-within-predicted-scenario-derangement-v1",
            "bucket_key": "the scenario the SEALED closed-list response names; "
            "an unparsed or scenario-free response gets its own bucket",
            "bucket_source_relpath": SEALED_CLOSED_LIST_OUTPUTS_RELPATH,
            "bucket_sizes": bucket_sizes,
            "self_maps": 0,
            "cross_bucket_rows": sum(
                1 for sample, donor in mapping.items() if buckets[sample] != buckets[donor]
            ),
            "gold_read": False,
        },
    }
    print(
        f"query-vectors-speech-twin.jsonl: {len(twin_rows)} rows, "
        f"sha256 {sets['query-vectors-speech-twin.jsonl']['sha256'][:16]}..."
    )



def _readback(out_dir, source_dir, sets, twin_rows) -> dict[str, object]:
    # ---- read every written file back through the flight's own reader -----
    readback: dict[str, object] = {}
    for name in sets:
        table = load_sealed_vectors(
            out_dir / name,
            expected_sha256=str(sets[name]["sha256"]),
            dimensions=(DIMENSION,),
        )
        deltas = 0.0
        source_rows = twin_rows if name.endswith("twin.jsonl") else None
        if source_rows is None:
            source_rows = [
                (entry_id, vector)
                for entry_id, vector in _iter_rows(
                    source_dir / str(sets[name]["source_file"])
                )
            ]
        for entry_id, vector in source_rows:
            got = table[entry_id][DIMENSION]
            deltas = max(deltas, max(abs(a - b) for a, b in zip(vector, got)))
        readback[name] = {"rows": len(table), "max_abs_delta": deltas}
        print(f"read back {name}: {len(table)} rows, max |delta| {deltas}")
    return readback


def _receipt_document(receipt, sets, readback, started, *, facets_only) -> dict[str, object]:
    document = {
        "schema": "saea-kus3-vectors-receipt-v1",
        "generator": "scripts/build_slurp_kus3_vectors.py",
        "exposure_ledger_row": LEDGER_ROW,
        "execution_profile": "model-free-check",
        "accounting": {
            "frozen_core_contacts": 0,
            "tool_contacts": 0,
            "embedding_contacts": 0,
            "gpu_hours": 0.0,
            "paid_spend": 0,
            "wall_seconds": round(time.monotonic() - started, 3),
        },
        "produced_by": {
            "exposure_ledger_row": receipt["exposure_ledger_row"],
            "receipt_relpath": f"{MF7B_DIR_RELPATH}/{MF7B_RECEIPT_NAME}",
            "encoder_id": receipt["embedder"]["encoder_id"],
            "dims": receipt["embedder"]["dims"],
            "pooling": receipt["embedder"]["pooling"],
            "prompts": receipt["embedder"]["prompts"],
        },
        "out_dir_relpath": OUT_DIR_RELPATH,
        "sets": sets,
        "readback": readback,
        "boundary": {
            "new_embedding_pass": False,
            "devel_sentence_text_read": False,
            "gold_read": False,
            "held_out_test_split_named": False,
        },
    }
    if facets_only:
        document["narrowed_to"] = {
            "sets": ["facet-vectors.jsonl"],
            "skipped": ["query-vectors-speech.jsonl", "query-vectors-speech-state.jsonl",
                        "query-vectors-speech-twin.jsonl"],
            "reason": "the binding cell reads only the TRAIN key facets; its queries come from elsewhere",
        }
    return document


if __name__ == "__main__":
    raise SystemExit(main())
