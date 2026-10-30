"""TFRL-v2 half-split of the construction column R01C (``scripts/flights/nx/tfrl_v2_split.py``, task T3): model-free, no data root, no card.

Proved here: the split is deterministic (same input, same bytes), the counts are 313 / 312, the halves are disjoint and cover the parent in
order, each half receipt verifies under ``core.roster.verify_roster`` and carries its own split identity, a moved roster hash is refused, outputs
are write-once, the ``interleave`` alternative keeps the pinned counts, and on the REAL committed roster the pinned hashes and counts hold.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NX = REPO / "scripts" / "flights" / "nx"
for _p in (str(REPO / "src"), str(NX)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from speech_aware_evidence_acquisition.core import roster as core_roster  # noqa: E402
import tfrl_v2_common as cm  # noqa: E402
import tfrl_v2_split as sp  # noqa: E402

SCENARIOS = ["alarm", "music", "news", "cooking"]


def fake_parent(rows: int = 9) -> dict:
    items = [{"audio_seconds": 1.0 + i, "draw_rank": i, "headset": bool(i % 2), "min_cosine_to_selected": 0.5,
              "sample_id": f"slurp:train:{i:06d}:audio-{i}", "stratum": SCENARIOS[i // 3 % len(SCENARIOS)], "utterance_id": f"slurp:train:{i:06d}"}
             for i in range(rows)]
    ids = [i["sample_id"] for i in items]
    return {"schema": "saea-roster-receipt-v1", "roster_id": "SLURP-STAGE3-CONSTRUCTION-R01C", "carrier_lock_key": "slurp", "split_role": "discovery",
            "roster_role": "construction", "seed": 7, "per_stratum": {}, "strata_counts": {}, "items": items, "excluded_count": 3,
            "d1_supplement": {"file_sha256": "0" * 64},
            "split_identity_hash": core_roster.split_identity_hash(core_roster.prefixed("slurp", ids))}


def write_parent(tmp_path: Path, rows: int = 9) -> tuple[Path, str, dict]:
    doc = fake_parent(rows)
    path = tmp_path / "parent.json"
    raw = json.dumps(doc, sort_keys=True, indent=2).encode("utf-8")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), doc


def run(tmp_path: Path, out: str = "out", **kw):
    path, sha, doc = write_parent(tmp_path)
    summary = sp.run_split(path, tmp_path / out, expected_sha256=sha, expected_split_identity=doc["split_identity_hash"], expected_rows=9, h1_rows=5, **kw)
    return summary, doc, sha


def test_split_counts_disjoint_union_and_verifies(tmp_path):
    summary, doc, _ = run(tmp_path)
    h1 = core_roster.read_roster(tmp_path / "out" / "slurp-stage3-construction-r01c-h1.json")
    h2 = core_roster.read_roster(tmp_path / "out" / "slurp-stage3-construction-r01c-h2.json")
    assert (len(h1["items"]), len(h2["items"])) == (5, 4)
    ids1, ids2 = [i["sample_id"] for i in h1["items"]], [i["sample_id"] for i in h2["items"]]
    assert not set(ids1) & set(ids2)
    assert ids1 + ids2 == [i["sample_id"] for i in doc["items"]]          # contiguous halves keep the parent's order
    assert h1["roster_id"].endswith("-H1") and h2["roster_id"].endswith("-H2")
    assert h1["split_identity_hash"] != h2["split_identity_hash"] != doc["split_identity_hash"]
    assert h1["parent"]["file_sha256"] == summary["parent"]["file_sha256"]
    assert summary["proof"]["union_equals_parent"] and summary["proof"]["halves_keep_parent_order"] and summary["proof"]["model_calls"] == 0
    assert summary["halves"][cm.HALF_H1]["count"] == 5


def test_items_are_verbatim_parent_items(tmp_path):
    _, doc, _ = run(tmp_path)
    h1 = core_roster.read_roster(tmp_path / "out" / "slurp-stage3-construction-r01c-h1.json")
    assert h1["items"] == doc["items"][:5]


def test_deterministic_same_input_same_bytes_and_rerun_is_a_noop(tmp_path):
    s1, _, _ = run(tmp_path, "a")
    s2, _, _ = run(tmp_path, "b")
    assert s1 == s2
    for name in ("slurp-stage3-construction-r01c-h1.json", "slurp-stage3-construction-r01c-h2.json", "slurp-stage3-construction-r01c-halves-split-receipt.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    before = {p: p.stat().st_mtime_ns for p in (tmp_path / "a").iterdir()}
    run(tmp_path, "a")                                                     # same inputs again: nothing is rewritten
    assert {p: p.stat().st_mtime_ns for p in (tmp_path / "a").iterdir()} == before


def test_refuses_when_the_roster_hash_moved(tmp_path):
    path, _, doc = write_parent(tmp_path)
    with pytest.raises(sp.SplitError, match="roster hash moved"):
        sp.run_split(path, tmp_path / "out", expected_sha256="0" * 64, expected_split_identity=doc["split_identity_hash"], expected_rows=9, h1_rows=5)
    assert not (tmp_path / "out").exists()


def test_refuses_when_the_split_identity_or_row_count_moved(tmp_path):
    path, sha, doc = write_parent(tmp_path)
    with pytest.raises(sp.SplitError, match="split identity moved"):
        sp.run_split(path, tmp_path / "o1", expected_sha256=sha, expected_split_identity="1" * 64, expected_rows=9, h1_rows=5)
    with pytest.raises(sp.SplitError, match="expected 625"):
        sp.run_split(path, tmp_path / "o2", expected_sha256=sha, expected_split_identity=doc["split_identity_hash"], h1_rows=5)


def test_refuses_a_tampered_item_even_with_a_matching_pin(tmp_path):
    doc = fake_parent()
    doc["items"][0]["sample_id"] = "slurp:train:999999:audio-x"            # identity no longer matches the items
    path = tmp_path / "bad.json"
    raw = json.dumps(doc, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    with pytest.raises(sp.SplitError, match="does not verify"):
        sp.load_parent(path, expected_sha256=hashlib.sha256(raw).hexdigest(), expected_split_identity=doc["split_identity_hash"], expected_rows=9)


def test_outputs_are_write_once(tmp_path):
    run(tmp_path)
    target = tmp_path / "out" / "slurp-stage3-construction-r01c-h1.json"
    target.write_bytes(b"changed\n")
    with pytest.raises(sp.SplitError, match="refusing to overwrite"):
        run(tmp_path)
    assert target.read_bytes() == b"changed\n"


def test_dry_run_writes_nothing(tmp_path):
    summary, _, _ = run(tmp_path, dry_run=True)
    assert summary["halves"][cm.HALF_H2]["count"] == 4
    assert not (tmp_path / "out").exists()


def test_interleave_mode_keeps_counts_and_covers_strata(tmp_path):
    summary, doc, _ = run(tmp_path, mode="interleave")
    ids = [i["sample_id"] for i in doc["items"]]
    h1 = core_roster.read_roster(tmp_path / "out" / "slurp-stage3-construction-r01c-h1.json")
    assert [i["sample_id"] for i in h1["items"]] == ids[0::2]
    assert summary["split_mode"] == "interleave"
    assert summary["proof"]["halves_keep_parent_order"]
    with pytest.raises(sp.SplitError, match="unknown split mode"):
        sp.split_items(doc["items"], 5, "random")


def test_stratum_coverage_arithmetic():
    assert sp.stratum_coverage({"a": 10, "b": 5}, {"a": 3, "b": 2}) == round(10 / 15, 6)
    assert sp.stratum_coverage({}, {"a": 9}) == 0.0


def test_pinned_constants_are_consistent():
    assert cm.HALF_ROWS[cm.HALF_H1] + cm.HALF_ROWS[cm.HALF_H2] == cm.ROSTER_ROWS == 625
    assert (cm.HALF_ROWS[cm.HALF_H1], cm.HALF_ROWS[cm.HALF_H2]) == (313, 312)
    assert cm.other_half(cm.HALF_H1) == cm.HALF_H2 and cm.other_half(cm.HALF_H2) == cm.HALF_H1


REAL = REPO / cm.ROSTER_RELPATH


@pytest.mark.skipif(not REAL.is_file(), reason="the committed R01C roster is not in this tree")
def test_real_r01c_roster_split_counts_and_pins():
    summary = sp.run_split(REAL, Path("/nonexistent-never-written"), dry_run=True)
    assert summary["parent"]["file_sha256"] == cm.ROSTER_FILE_SHA256
    assert summary["parent"]["split_identity_hash"] == cm.ROSTER_SPLIT_IDENTITY
    assert (summary["halves"][cm.HALF_H1]["count"], summary["halves"][cm.HALF_H2]["count"]) == (313, 312)
    again = sp.run_split(REAL, Path("/nonexistent-never-written"), dry_run=True)
    assert again == summary


def test_no_non_ascii_text_in_the_split_files():
    for path in (NX / "tfrl_v2_split.py", NX / "tfrl_v2_common.py", Path(__file__)):
        assert all(ord(ch) < 128 for ch in path.read_text(encoding="utf-8")), path
