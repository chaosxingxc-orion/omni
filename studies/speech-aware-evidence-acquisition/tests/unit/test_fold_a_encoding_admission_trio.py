"""The Fold A twins of the Fold B port speech-query encoding admission trio (DRAFT; the encoder is never contacted).

Pins: the Fold A fragment bytes, the Fold B fragment it is derived from, the diff-only-roster property (exactly the roster, run-id and source-inventory leaves move), the registration
shape against the Fold B one, the DRAFT ledger row text (parsed with the gate's own ledger parser and cross-checked against the registered plan the way the gate does), and the writer's
determinism. Nothing here opens an audio file, a model or an encoder.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import fold_a_encoding_write_registration as w  # noqa: E402

from speech_aware_evidence_acquisition.contracts.ledger import _EXPOSURE_CARRIER_COLUMN, _EXPOSURE_RUN_COLUMN, _parse_ledger_tables  # noqa: E402

CJK = re.compile("[\\u4e00-\\u9fff]")
#: The Fold A fragment's sha256 (the plan's protocol_hash). It moves only if the Fold B fragment, the Fold A roster receipt or the Fold A source inventory moves.
FOLD_A_FRAGMENT_SHA256 = "5099b95d1cb431d7ddf05c43260db8f4f0d6cb09700a59cfac7fb01fc8e5faad"
FOLD_A_INVENTORY_SHA256 = "c81c631f0db5d8715cce54a9bf272eceee71d10023e5eeb332eb86496b46f4fb"
DATA_ROOT = Path(os.environ.get("SPEECHRL_DATA_DIR") or "/mnt/cephfs/xingchao/omni")
LOCK_PATH = DATA_ROOT / w.LIVE_LOCK_RELPATH
needs_data = pytest.mark.skipif(
    not ((DATA_ROOT / w.INVENTORY_RELPATH).is_file() and LOCK_PATH.is_file()),
    reason="the Fold A source inventory or the live lock is not on this machine",
)


def _json(relpath: str):
    return json.loads((REPO / relpath).read_bytes())


def _sha(relpath: str) -> str:
    return hashlib.sha256((REPO / relpath).read_bytes()).hexdigest()


def test_fragment_hash_is_pinned_and_the_fold_b_source_is_unchanged():
    assert _sha(w.FOLD_A_FRAGMENT) == FOLD_A_FRAGMENT_SHA256
    assert _sha(w.FOLD_B_FRAGMENT) == w.FOLD_B_FRAGMENT_SHA256
    raw = (REPO / w.FOLD_A_FRAGMENT).read_bytes()
    assert raw == w.canonical(json.loads(raw))


def test_fragment_differs_from_fold_b_only_in_roster_run_id_and_source_inventory_leaves():
    fold_b, fold_a = _json(w.FOLD_B_FRAGMENT), _json(w.FOLD_A_FRAGMENT)
    assert w.moved_leaves(fold_b, fold_a) == sorted(w.EXPECTED_MOVED_LEAVES)
    # the unchanged contract: encoder pin, geometry-bearing policy text, pool size, pooling prompt, windows, layout leaves
    for key in ("encoder", "contact_policy", "cpu_workers", "dimension", "query_prompt", "boundary_policy", "cache_relpath", "max_audio_samples", "max_seconds",
                "max_sequence_tokens", "overlap_seconds", "expected_unique_spans", "input_order", "isolation", "schema", "answer_authority", "monitor_interval_seconds"):
        assert fold_a[key] == fold_b[key], key
    assert fold_a["roster_sha256"] == w.ROSTER_SHA256 == _sha(w.ROSTER_RECEIPT)
    assert fold_a["output_relpath"] == f"prepared/{w.FOLD_A_RUN}/index.json"
    assert fold_a["query_rows_relpath"] == f"prepared/{w.FOLD_A_RUN}/queries.jsonl"
    assert w.fold_b_token_survivors(json.dumps(fold_a)) == []
    # a leaf outside the declared set is detected by the same helper the writer refuses on
    tampered = copy.deepcopy(fold_a)
    tampered["cpu_workers"] = 2
    assert "cpu_workers" in w.moved_leaves(fold_b, tampered)


def test_registration_has_the_fold_b_shape_with_fold_a_identities():
    fold_b, fold_a = _json(w.FOLD_B_REGISTRATION), _json(w.FOLD_A_REGISTRATION)
    assert set(fold_a) == set(fold_b)
    entry_b, entry_a = fold_b["plans"][0], fold_a["plans"][0]
    assert set(entry_a) == set(entry_b)
    assert set(entry_a["execution_plan"]) == set(entry_b["execution_plan"])
    plan = entry_a["execution_plan"]
    assert plan["run_id"] == w.FOLD_A_RUN
    assert plan["protocol_hash"] == FOLD_A_FRAGMENT_SHA256 == entry_a["protocol"]["sha256"]
    assert plan["split_identity_hash"] == w.SPLIT_IDENTITY_HASH == _json(w.ROSTER_RECEIPT)["split_identity_hash"]
    assert (plan["planned_model_calls"], plan["planned_speech_audio_seconds"], plan["planned_gpu_hours"]) == (1080, 3085, 2.0)
    assert plan["attempt_id"] == "encode-v1" and plan["execution_profile"] == "bounded-discovery-probe" and plan["split_role"] == "discovery"
    assert entry_a["protocol"]["relpath"] == w.FOLD_A_FRAGMENT
    assert entry_a["roster_receipt"] == {"relpath": w.ROSTER_RECEIPT, "sha256": w.ROSTER_SHA256}
    assert entry_a["source_inventory"] == {"path": w.INVENTORY_PATH_LEAF, "sha256": FOLD_A_INVENTORY_SHA256}
    assert entry_a["runtime_config_shape"]["plan"] == plan
    assert entry_a["dataset_config"]["sha256"] == _sha(w.DATASET_CONFIG)
    assert fold_a["historical_twin"]["run_id"] == w.HISTORICAL_FOLD_A_RUN
    assert fold_a["encoder"] == fold_b["encoder"]
    assert fold_a["status"] == "DRAFT_PENDING_COUNTERSIGN"
    # the live (amended) lock, not the pre-amendment lock the Fold B registration still pins
    assert fold_a["umbrella_lock"]["sha256"] == w.LIVE_LOCK_SHA256 != fold_b["umbrella_lock"]["sha256"]
    assert w.fold_b_token_survivors(json.dumps(fold_a)) == []


def test_ledger_row_draft_parses_and_matches_the_plan_like_the_gate_checks():
    text = (REPO / w.FOLD_A_LEDGER_ROW).read_text(encoding="utf-8")
    rows = _parse_ledger_tables(text)
    assert len(rows) == 1
    row = rows[0]
    registration = _json(w.FOLD_A_REGISTRATION)
    entry = registration["plans"][0]
    plan = entry["execution_plan"]
    assert row[_EXPOSURE_RUN_COLUMN] == plan["run_id"]
    assert [row[c] for c in ("execution profile", "split role", "split identity hash")] == [plan["execution_profile"], plan["split_role"], plan["split_identity_hash"]]
    assert row[_EXPOSURE_CARRIER_COLUMN].split()[0] == plan["carrier_lock_key"] == "slurp"
    assert row["consumed"] == "no"
    assert int(row["calls"]) >= plan["planned_model_calls"] == 1080
    assert int(row["speech audio seconds"]) >= plan["planned_speech_audio_seconds"] == 3085
    assert float(row["gpu hours"]) >= plan["planned_gpu_hours"]
    assert row["artifacts"] == w.FOLD_A_REGISTRATION
    assert list(row.values()) == entry["ledger_row_cells"]
    assert FOLD_A_FRAGMENT_SHA256 in row["notes"] and "NO CONTACT BEFORE THE COUNTERSIGN" in row["notes"]
    assert "DRAFT" in text and "No contact before the countersign" in text and FOLD_A_FRAGMENT_SHA256 in text


def test_the_draft_row_is_not_in_the_exposure_ledger():
    ledger = (REPO / "docs" / "exposure-ledger.md").read_text(encoding="utf-8")
    assert w.FOLD_A_RUN not in ledger


def test_new_files_carry_no_cjk():
    for relpath in (w.FOLD_A_FRAGMENT, w.FOLD_A_REGISTRATION, w.FOLD_A_LEDGER_ROW, "scripts/flights/nx/fold_a_encoding_write_registration.py", Path(__file__).relative_to(REPO).as_posix()):
        assert not CJK.search((REPO / relpath).read_text(encoding="utf-8")), relpath


@needs_data
def test_writer_is_deterministic_and_reproduces_the_committed_files():
    first = w.derive_all(REPO, DATA_ROOT, LOCK_PATH)
    second = w.derive_all(REPO, DATA_ROOT, LOCK_PATH)
    assert first == second
    assert set(first) == {w.FOLD_A_FRAGMENT, w.FOLD_A_REGISTRATION, w.FOLD_A_LEDGER_ROW}
    for relpath, raw in first.items():
        assert (REPO / relpath).read_bytes() == raw, relpath
    assert hashlib.sha256(first[w.FOLD_A_FRAGMENT]).hexdigest() == FOLD_A_FRAGMENT_SHA256


@needs_data
def test_writer_refuses_a_wrong_lock_and_an_overwrite(tmp_path):
    wrong = tmp_path / "datasets.lock.json"
    wrong.write_bytes(b"{}")
    with pytest.raises(SystemExit, match="live datasets lock"):
        w.derive_all(REPO, DATA_ROOT, wrong)
    # create-only: a differing file in a throwaway repo root is refused, an identical one accepted
    root = tmp_path / "repo"
    for relpath in (w.FOLD_B_FRAGMENT, w.FOLD_B_REGISTRATION, w.ROSTER_RECEIPT, w.DATASET_CONFIG, w.HISTORICAL_REGISTRATION):
        (root / relpath).parent.mkdir(parents=True, exist_ok=True)
        (root / relpath).write_bytes((REPO / relpath).read_bytes())
    target = root / w.FOLD_A_FRAGMENT
    target.write_bytes(b"{}\n")
    with pytest.raises(SystemExit, match="exists with different bytes"):
        w.main(["--repo-root", str(root), "--data-root", str(DATA_ROOT)])
    target.unlink()
    assert w.main(["--repo-root", str(root), "--data-root", str(DATA_ROOT)]) == 0
    assert (root / w.FOLD_A_FRAGMENT).read_bytes() == (REPO / w.FOLD_A_FRAGMENT).read_bytes()
    assert w.main(["--repo-root", str(root), "--data-root", str(DATA_ROOT), "--check"]) == 0
