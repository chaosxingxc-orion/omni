#!/usr/bin/env python3
"""Derive the Fold A twins of the Fold B port speech-query ENCODING admission trio (DRAFT; nothing is contacted).

The Fold B port encoding (run ``SAEA-SLURP-TFRL-FOLD-B-SPEECH-QUERY-ENCODING-NPU-PORT-V1``) was admitted by three committed artefacts: a protocol fragment, a registration JSON and a
countersigned exposure-ledger row. The Fold A confirmation cell has none of them. This writer derives all three Fold A twins from the Fold B ones, deterministically (same inputs, same bytes):

1. ``configs/preparation/slurp-tfrl-fold-a-speech-query-encoding-protocol-npu-port-v1.json`` -- the Fold B fragment with exactly these leaves moved: ``roster_sha256`` (the Fold A roster
   receipt's pinned hash), ``output_relpath`` and ``query_rows_relpath`` (the Fold A run id), and ``sources.path`` / ``sources.sha256`` (the Fold A source inventory, which the runtime
   binds to the roster hash: ``sources["roster_sha256"] == protocol["roster_sha256"]``, so the Fold B inventory cannot stand in). The encoder pin, the forced-causal geometry, fp32 CPU
   pool, pooling, windowing and output layout are byte-identical. The writer prints the diff and refuses any other moved leaf.
2. ``docs/readiness/2026-10-30-slurp-tfrl-fold-a-speech-query-encoding-registration-npu-port-v1.json`` -- the registration, same shape as Fold B's. It pins the LIVE datasets lock
   (the 2026-10-28 amendment, ``docs/readiness/2026-10-28-npu-port-lock-entries-v1/``); the Fold B registration pins the pre-amendment lock, which the runtime now refuses.
3. ``docs/readiness/2026-10-30-slurp-l338-fold-a-confirmation-npu-port-v1/encoding-ledger-row-draft-v1.md`` -- the DRAFT exposure-ledger row text. It is NEVER appended to
   ``docs/exposure-ledger.md`` by this writer; the owner countersigns the text first, and the main thread appends it afterwards. Admission rule: no contact before the countersign.

Model-free and card-free: the writer reads committed files, the Fold A source inventory (``coordination/``, built by ``scripts/build_speech_query_source_inventory.py``, header reads
only) and the live lock bytes. ``--dry-run`` writes nothing; ``--check`` verifies the committed files equal the derivation; a plain run writes create-only (an existing file with
different bytes is refused).

    python -B scripts/flights/nx/fold_a_encoding_write_registration.py [--repo-root DIR] [--data-root DIR] [--lock-path FILE] [--dry-run | --check]
"""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REPO_ROOT = HERE.parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("SPEECHRL_DATA_DIR") or "/mnt/cephfs/xingchao/omni")

FOLD_B_RUN = "SAEA-SLURP-TFRL-FOLD-B-SPEECH-QUERY-ENCODING-NPU-PORT-V1"
FOLD_A_RUN = "SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-NPU-PORT-V1"
HISTORICAL_FOLD_A_RUN = "SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-V1"
FOLD_A_SESSION = "local-nemotron-cpu-slurp-tfrl-fold-a-query-npu-port-v1"

FOLD_B_FRAGMENT = "configs/preparation/slurp-tfrl-fold-b-speech-query-encoding-protocol-npu-port-v1.json"
FOLD_B_FRAGMENT_SHA256 = "ebd88049b8086d7c0f37463a913c68f52cef013b79b8040824b7ed82d693efd6"
FOLD_B_REGISTRATION = "docs/readiness/2026-10-28-slurp-tfrl-fold-b-speech-query-encoding-registration-npu-port-v1.json"
FOLD_A_FRAGMENT = "configs/preparation/slurp-tfrl-fold-a-speech-query-encoding-protocol-npu-port-v1.json"
FOLD_A_REGISTRATION = "docs/readiness/2026-10-30-slurp-tfrl-fold-a-speech-query-encoding-registration-npu-port-v1.json"
FOLD_A_LEDGER_ROW = "docs/readiness/2026-10-30-slurp-l338-fold-a-confirmation-npu-port-v1/encoding-ledger-row-draft-v1.md"
HISTORICAL_REGISTRATION = "docs/readiness/2026-09-12-slurp-training-query-embedding-registration-v1.json"

ROSTER_RECEIPT = "docs/receipts/rosters/slurp-evo-fold-a-v2.json"
ROSTER_SHA256 = "1fe7908c31396774d7535fdf37bd791762215c324999d66ed7beb9e7b2f9033d"
SPLIT_IDENTITY_HASH = "61b0f06254975873bea6afb287ab54823dc71b57e5398697b9bef817e25398b6"
DATASET_CONFIG = "configs/dataset/slurp-speech-native-evo-fold-a-1080.json"
INVENTORY_RELPATH = "coordination/slurp-tfrl-fold-a-speech-query-source-inventory-npu-port-v1.json"
INVENTORY_PATH_LEAF = "${SAEA_DATA_ROOT}/" + INVENTORY_RELPATH
#: The live datasets lock after the 2026-10-28 port amendment (docs/readiness/2026-10-28-npu-port-lock-entries-v1/README.md, "Lock identity").
LIVE_LOCK_SHA256 = "61f3d111b7bed7dfc191358f2ce557efd824795cc3cdbb28f32761d8b2ee20df"
LIVE_LOCK_RELPATH = "work/dev/umbrella/docs/datasets.lock.json"

REGISTERED_AT = "2026-10-30"
ROWS = 1080
PLANNED_GPU_HOURS = 2.0
LEDGER_COLUMNS = ("date", "run/experiment id", "execution profile", "scope", "model/tool", "speech carrier + split", "split role", "split identity hash",
                  "consumed", "calls", "speech audio seconds", "gpu hours", "inherited exposure", "artifacts", "notes")
#: The only leaves of the Fold B fragment the Fold A twin may move (dotted paths).
EXPECTED_MOVED_LEAVES = ("output_relpath", "query_rows_relpath", "roster_sha256", "sources.path", "sources.sha256")
FOLD_B_TOKEN = re.compile(r"fold[-_ ]b", re.IGNORECASE)


class DerivationError(SystemExit):
    pass


def canonical(document) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def leaves(node, prefix: str = ""):
    """Flatten a JSON document to {dotted path: scalar}; lists index as ``[i]``."""

    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from leaves(value, f"{prefix}[{index}]")
    else:
        yield prefix, node


def moved_leaves(old, new) -> list[str]:
    a, b = dict(leaves(old)), dict(leaves(new))
    return sorted(path for path in set(a) | set(b) if a.get(path, KeyError) != b.get(path, KeyError))


def fold_b_token_survivors(raw_text: str) -> list[str]:
    return sorted(set(FOLD_B_TOKEN.findall(raw_text)))


def _read_json(path: Path):
    return json.loads(path.read_bytes())


def load_inventory(data_root: Path) -> tuple[dict, str]:
    path = data_root / INVENTORY_RELPATH
    if not path.is_file():
        raise DerivationError(f"refusing: the Fold A source inventory is absent ({path}); build it with scripts/build_speech_query_source_inventory.py first")
    raw = path.read_bytes()
    inventory = json.loads(raw)
    if inventory.get("roster_sha256") != ROSTER_SHA256 or inventory.get("split_identity_hash") != SPLIT_IDENTITY_HASH:
        raise DerivationError("refusing: the Fold A source inventory does not carry the Fold A roster hash and split identity")
    if inventory.get("count") != ROWS or len(inventory.get("rows", ())) != ROWS:
        raise DerivationError(f"refusing: the Fold A source inventory does not have {ROWS} rows")
    for key in ("model_contacts", "tokenizer_contacts", "network_contacts", "audio_decodes"):
        if inventory.get(key) != 0:
            raise DerivationError(f"refusing: the Fold A source inventory records {key} != 0")
    return inventory, sha256_bytes(raw)


def derive_fragment(repo_root: Path, inventory_sha256: str) -> tuple[dict, dict]:
    raw = (repo_root / FOLD_B_FRAGMENT).read_bytes()
    if sha256_bytes(raw) != FOLD_B_FRAGMENT_SHA256:
        raise DerivationError("refusing: the Fold B fragment bytes differ from the pinned ones")
    fold_b = json.loads(raw)
    roster = repo_root / ROSTER_RECEIPT
    if sha256_bytes(roster.read_bytes()) != ROSTER_SHA256:
        raise DerivationError("refusing: the Fold A roster receipt bytes differ from the pinned hash")
    fold_a = copy.deepcopy(fold_b)
    for key in ("output_relpath", "query_rows_relpath"):
        if FOLD_B_RUN not in fold_b[key]:
            raise DerivationError(f"refusing: {key} does not carry the Fold B run id")
        fold_a[key] = fold_b[key].replace(FOLD_B_RUN, FOLD_A_RUN)
    fold_a["roster_sha256"] = ROSTER_SHA256
    fold_a["sources"] = {"path": INVENTORY_PATH_LEAF, "sha256": inventory_sha256}
    moved = moved_leaves(fold_b, fold_a)
    if moved != sorted(EXPECTED_MOVED_LEAVES):
        raise DerivationError(f"refusing: the Fold A fragment moved {moved}, expected exactly {sorted(EXPECTED_MOVED_LEAVES)}")
    survivors = fold_b_token_survivors(canonical(fold_a).decode("utf-8"))
    if survivors:
        raise DerivationError(f"refusing: a Fold B token survives in the Fold A fragment: {survivors}")
    return fold_b, fold_a


def derive_registration(repo_root: Path, data_root: Path, lock_path: Path, inventory: dict, inventory_sha256: str, fragment_sha256: str) -> dict:
    lock_sha = sha256_bytes(lock_path.read_bytes()) if lock_path.is_file() else None
    if lock_sha != LIVE_LOCK_SHA256:
        raise DerivationError(f"refusing: the live datasets lock at {lock_path} has sha256 {lock_sha}, expected {LIVE_LOCK_SHA256}; re-derive after reviewing the lock amendment")
    fold_b = _read_json(repo_root / FOLD_B_REGISTRATION)
    historical = _read_json(repo_root / HISTORICAL_REGISTRATION)
    twin = [p for p in historical["plans"] if p.get("fold") == "fold-a" and p["execution_plan"]["run_id"] == HISTORICAL_FOLD_A_RUN]
    if len(twin) != 1:
        raise DerivationError("refusing: the historical registration does not carry exactly one Fold A plan")
    twin_plan = twin[0]["execution_plan"]
    dataset_sha = sha256_bytes((repo_root / DATASET_CONFIG).read_bytes())
    if inventory["dataset_config"]["sha256"] != dataset_sha or inventory["dataset_config"]["relpath"] != DATASET_CONFIG:
        raise DerivationError("refusing: the inventory's dataset config pin differs from the committed Fold A dataset fragment")
    seconds = inventory["audio_seconds_total"]
    planned_seconds = -int(-seconds // 1)  # ceiling, the ledger convention (Fold B: 3154.797 -> 3155)
    if planned_seconds != twin_plan["planned_speech_audio_seconds"]:
        raise DerivationError(f"refusing: planned seconds {planned_seconds} differ from the historical Fold A plan {twin_plan['planned_speech_audio_seconds']}")
    if twin_plan["planned_model_calls"] != ROWS:
        raise DerivationError("refusing: the historical Fold A plan is not 1,080 calls")

    entry_b = fold_b["plans"][0]
    plan = copy.deepcopy(entry_b["execution_plan"])
    plan.update({"run_id": FOLD_A_RUN, "protocol_hash": fragment_sha256, "split_identity_hash": SPLIT_IDENTITY_HASH,
                 "planned_model_calls": ROWS, "planned_speech_audio_seconds": planned_seconds, "planned_gpu_hours": PLANNED_GPU_HOURS})
    cells = ledger_row_cells(fragment_sha256, planned_seconds)
    lock_ref = {"path": str(lock_path), "sha256": LIVE_LOCK_SHA256}
    entry = {
        "actual_source_audio_seconds": seconds,
        "dataset_config": {"relpath": DATASET_CONFIG, "sha256": dataset_sha},
        "execution_plan": plan,
        "fold": "fold-a",
        "ledger_row_cells": cells,
        "output_relpath": f"prepared/{FOLD_A_RUN}/index.json",
        "protocol": {"relpath": FOLD_A_FRAGMENT, "sha256": fragment_sha256},
        "query_rows_relpath": f"prepared/{FOLD_A_RUN}/queries.jsonl",
        "roster_receipt": {"relpath": ROSTER_RECEIPT, "sha256": ROSTER_SHA256},
        "runtime_config_shape": {
            "plan": copy.deepcopy(plan),
            "protocol": "the protocol above, by absolute path",
            "session_id": FOLD_A_SESSION,
            "study_head": "SET_TO_THE_FINAL_CLEAN_REGISTRATION_COMMIT",
            "umbrella_lock": lock_ref,
        },
        "session_id": FOLD_A_SESSION,
        "source_inventory": {"path": INVENTORY_PATH_LEAF, "sha256": inventory_sha256},
    }
    registration = copy.deepcopy(fold_b)
    registration.update({
        "historical_twin": {"registration": HISTORICAL_REGISTRATION, "run_id": HISTORICAL_FOLD_A_RUN},
        "plans": [entry],
        "registered_at_utc": REGISTERED_AT,
        "status": "DRAFT_PENDING_COUNTERSIGN",
        "umbrella_lock": lock_ref,
        "registration_requirements": [
            "Do not contact the encoder before the owner countersigns the ledger row text (encoding-ledger-row-draft-v1.md).",
            "Commit the protocol, this registration and the countersigned exposure row before any contact.",
            "Write the external runtime config with study_head equal to that clean commit.",
            "Keep consumed=no until the attempt is admitted; never replay an interrupted contact.",
        ],
        "root_authority": "Owner 2026-10-28: experiments run in the saea-base Docker container; WSL2 constraints removed; generation runs in the background. The Fold A confirmation encoding additionally needs the owner's countersign of the ledger row (docs/owner-decisions-pending.md, 2026-10-30).",
    })
    survivors = fold_b_token_survivors(canonical(registration).decode("utf-8"))
    if survivors:
        raise DerivationError(f"refusing: a Fold B token survives in the Fold A registration: {survivors}")
    return registration


def ledger_row_cells(fragment_sha256: str, planned_seconds: int) -> list[str]:
    declared = ("CPU float32 retrieval encoder (declared difference 8.7, design section 6.4): pinned pool of 4 workers x 16 threads on the cores recorded in the encoder inventory "
                "(first_core 128, cores 128-191 of the Ascend host), batch-one forwards; planned_gpu_hours is the attempt's wall-clock ceiling and the actual GPU/NPU hours are 0. "
                "Attention geometry FORCED CAUSAL on every text layer and measured on the loaded weights at worker start (declared difference 8.4; as shipped the tower is "
                "bidirectional at batch one and causal in a padded batch, design section 6.4)")
    notes = (f"DRAFT (owner countersignature pending); NO CONTACT BEFORE THE COUNTERSIGN. Admission rule: the runtime admits this plan only against the committed protocol fragment, "
             f"the committed registration and this row once countersigned and appended to the ledger; the encoder is not contacted before then. {declared}. "
             f"Protocol SHA256 {fragment_sha256}. One encode-v1 attempt; no retry, fallback, scorer, answer read, or cross-fold feedback. "
             "The runtime config study_head binds the clean registration commit.")
    return [
        REGISTERED_AT,
        FOLD_A_RUN,
        "bounded-discovery-probe",
        f"Port twin of {HISTORICAL_FOLD_A_RUN}: complete-speech retrieval-query encoding for TRAIN fold A; no answers or scoring",
        "Frozen local Omni-Embed-Nemotron-3B retrieval tool on the CPU float32 worker pool; no answering core",
        "slurp SLURP-EVO-FOLD-A-V2 TRAIN discovery",
        "discovery",
        SPLIT_IDENTITY_HASH,
        "no",
        str(ROWS),
        str(planned_seconds),
        str(PLANNED_GPU_HOURS),
        "The historical Fold A roster, packet and query-encoding exposures carried forward; this row adds only the port's pinned retrieval-encoder contacts",
        FOLD_A_REGISTRATION,
        notes,
    ]


def render_ledger_row(cells: list[str], fragment_sha256: str, inventory_sha256: str, seconds_exact: float) -> str:
    for cell in cells:
        if "|" in cell or "\n" in cell:
            raise DerivationError("refusing: a ledger cell contains a pipe or a newline")
    header = "| " + " | ".join(LEDGER_COLUMNS) + " |\n| " + " | ".join("---" for _ in LEDGER_COLUMNS) + " |\n"
    row = "| " + " | ".join(cells) + " |\n"
    return (
        "# Fold A speech-query encoding: DRAFT exposure-ledger row (owner countersign pending)\n\n"
        "**DRAFT TEXT. NOT APPENDED to `docs/exposure-ledger.md`. NO CONTACT BEFORE THE COUNTERSIGN.** Generated by "
        "`scripts/flights/nx/fold_a_encoding_write_registration.py`; do not edit by hand (re-run the writer; `--check` verifies it).\n\n"
        "## What the row admits\n\n"
        f"- Run id `{FOLD_A_RUN}`: {ROWS:,} retrieval-encoder contacts (one per Fold A recording, `SLURP-EVO-FOLD-A-V2`), {cells[10]} planned speech-audio seconds "
        f"(roster total {seconds_exact:.6f} s, ceiling), {cells[11]} h wall-clock ceiling, 0 NPU hours. One `encode-v1` attempt, no retry.\n"
        f"- Protocol fragment `{FOLD_A_FRAGMENT}`, sha256 `{fragment_sha256}` (the Fold B port fragment `{FOLD_B_FRAGMENT}`, sha256 `{FOLD_B_FRAGMENT_SHA256}`, "
        "with only the roster, run-id and source-inventory leaves moved).\n"
        f"- Roster receipt `{ROSTER_RECEIPT}`, sha256 `{ROSTER_SHA256}`; split identity hash `{SPLIT_IDENTITY_HASH}`.\n"
        f"- Source inventory `{INVENTORY_RELPATH}` (data root), sha256 `{inventory_sha256}` (model-free, header reads only).\n"
        f"- Registration `{FOLD_A_REGISTRATION}` (named in the row's artifacts cell).\n\n"
        "## Admission rule\n\n"
        "No contact before the countersign. The step order is: (1) the owner countersigns the row text below; (2) the main thread appends the row to `docs/exposure-ledger.md` and commits the "
        "fragment, the registration and the ledger together; (3) the runtime config is written against that clean HEAD; (4) the detached CPU chain starts. The runtime itself refuses a plan "
        "that has no matching unconsumed ledger row, a fragment whose bytes differ from the registration, or a lock whose bytes differ from the pinned lock.\n\n"
        "## The row (append verbatim after the countersign)\n\n" + header + row
    )


def derive_all(repo_root: Path, data_root: Path, lock_path: Path) -> dict[str, bytes]:
    inventory, inventory_sha256 = load_inventory(data_root)
    _, fragment = derive_fragment(repo_root, inventory_sha256)
    fragment_raw = canonical(fragment)
    fragment_sha256 = sha256_bytes(fragment_raw)
    registration = derive_registration(repo_root, data_root, lock_path, inventory, inventory_sha256, fragment_sha256)
    cells = registration["plans"][0]["ledger_row_cells"]
    ledger = render_ledger_row(cells, fragment_sha256, inventory_sha256, inventory["audio_seconds_total"]).encode("utf-8")
    return {FOLD_A_FRAGMENT: fragment_raw, FOLD_A_REGISTRATION: canonical(registration), FOLD_A_LEDGER_ROW: ledger}


def fragment_diff(repo_root: Path, data_root: Path) -> tuple[str, list[str]]:
    _, inventory_sha256 = load_inventory(data_root)
    fold_b, fold_a = derive_fragment(repo_root, inventory_sha256)
    diff = "".join(difflib.unified_diff(canonical(fold_b).decode().splitlines(True), canonical(fold_a).decode().splitlines(True),
                                        fromfile=FOLD_B_FRAGMENT, tofile=FOLD_A_FRAGMENT))
    return diff, moved_leaves(fold_b, fold_a)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--lock-path", type=Path, default=None, help=f"live datasets lock (default <data-root>/{LIVE_LOCK_RELPATH})")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the diff and hashes; write nothing")
    mode.add_argument("--check", action="store_true", help="verify the committed files equal the derivation")
    args = parser.parse_args(argv)
    repo_root, data_root = args.repo_root.resolve(), args.data_root.resolve()
    lock_path = args.lock_path or data_root / LIVE_LOCK_RELPATH
    diff, moved = fragment_diff(repo_root, data_root)
    sys.stdout.write(diff)
    print(f"moved leaves: {moved}")
    outputs = derive_all(repo_root, data_root, lock_path)
    for relpath, raw in outputs.items():
        print(f"{sha256_bytes(raw)}  {relpath}")
    if args.dry_run:
        return 0
    status = 0
    for relpath, raw in outputs.items():
        path = repo_root / relpath
        if args.check:
            if not path.is_file() or path.read_bytes() != raw:
                print(f"MISMATCH {relpath}", file=sys.stderr)
                status = 1
            continue
        if path.exists():
            if path.read_bytes() != raw:
                raise DerivationError(f"refusing: {relpath} exists with different bytes")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)
        print(f"wrote {relpath}")
    return status


if __name__ == "__main__":
    sys.exit(main())
