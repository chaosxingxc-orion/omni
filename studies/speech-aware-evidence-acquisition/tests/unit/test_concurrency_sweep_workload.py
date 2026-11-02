import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from concurrency_sweep_workload import WorkloadError, build_level_spec, validate_manifest


def _write(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _fixture(tmp_path):
    source_items = []
    for i in range(2):
        source_items.append({
            "sample_id": f"slurp:train:{i:06d}:audio-{i}",
            "packet_id": f"slurp/train/{i:06d}/audio-{i}",
            "media_relpath": f"datasets/slurp/slurp_real/audio-{i}.flac",
            "audio_sha256": f"{i+1:064x}",
            "audio_seconds": 1.5 + i,
            "sample_rate_hz": 16000,
        })
    roster = {"schema": "saea-real-engine-roster-v1", "items": source_items}
    roster_path = _write(tmp_path, "roster.json", roster)
    source_sha = hashlib.sha256(roster_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "saea-fixed-real-subset-manifest-v1",
        "status": "PINNED_CPU_ONLY_NOT_ADMITTED",
        "source_roster": "real-engine-500-v1",
        "source_roster_sha256": source_sha,
        "count": 2,
        "items": source_items,
        "unique_sample_ids": 2,
        "unique_audio_hashes": 2,
        "test_labels_read": False,
        "model_calls": 0,
    }
    return _write(tmp_path, "manifest.json", manifest), roster_path


def test_validate_and_build_level(tmp_path):
    manifest, roster = _fixture(tmp_path)
    # expected_count is configurable for the small fixture; production is 100.
    checked = validate_manifest(manifest, source_roster_path=roster, expected_profile_id="p", expected_profile_sha256="a" * 64, expected_count=2)
    level = build_level_spec(checked, level_id="C32", concurrency_per_instance=32)
    assert level["status"] == "PLAN_ONLY_NOT_ADMITTED"
    assert level["model_calls"] == 0
    assert level["test_labels_read"] is False


def test_rejects_source_hash_drift(tmp_path):
    manifest, roster = _fixture(tmp_path)
    roster.write_text(roster.read_text() + "\n", encoding="utf-8")
    try:
        validate_manifest(manifest, source_roster_path=roster, expected_profile_id="p", expected_profile_sha256="a" * 64, expected_count=2)
    except WorkloadError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("source hash drift was accepted")


def test_rejects_test_or_model_touch(tmp_path):
    manifest, roster = _fixture(tmp_path)
    doc = json.loads(manifest.read_text())
    doc["test_labels_read"] = True
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    try:
        validate_manifest(manifest, source_roster_path=roster, expected_profile_id="p", expected_profile_sha256="a" * 64, expected_count=2)
    except WorkloadError as exc:
        assert "test labels" in str(exc)
    else:
        raise AssertionError("test label access was accepted")


def _full_fixture(tmp_path):
    rows = [{"sample_id": f"slurp:train:{i}", "packet_id": f"p{i}",
             "media_relpath": f"datasets/slurp/slurp_real/{i}.flac", "audio_sha256": f"{i+1:064x}",
             "audio_seconds": 1.0, "sample_rate_hz": 16000} for i in range(500)]
    identity = hashlib.sha256("\n".join(sorted("slurp/" + r["sample_id"] for r in rows)).encode()).hexdigest()
    roster = {"schema": "saea-real-engine-roster-v1", "count": 500, "items": rows,
              "split_identity_hash": identity}
    path = _write(tmp_path, "roster500.json", roster)
    parent = {"count": 500, "items": rows, "split_identity_hash": identity,
              "source_binding_sha256": _digest(roster), "audio_seconds_total": 500.0,
              "split_name": "SLURP-REAL-ENGINE-500-V1", "loader": "preserved"}
    parent["runtime_view_sha256"] = _digest(parent)
    return path, parent


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_full(manifest, roster):
    return validate_manifest(manifest, source_roster_path=roster, expected_profile_id="p",
                             expected_profile_sha256="a"*64, expected_count=500)


def test_full500_reuses_original_bytes_and_complete_parent(tmp_path):
    from concurrency_sweep_workload import build_execution_view
    roster, parent = _full_fixture(tmp_path)
    original = roster.read_bytes()
    copy = tmp_path / "exact-copy.json"
    copy.write_bytes(original)
    checked = _validate_full(copy, roster)
    view = build_execution_view(checked, parent)
    assert view == parent and view is not parent and view["items"] is not parent["items"]
    assert roster.read_bytes() == original
    assert checked["manifest_sha256"] == hashlib.sha256(original).hexdigest()
    assert checked["rows"] == build_level_spec(checked, level_id="C64", concurrency_per_instance=64)["rows"] == 500


def test_full500_rejects_reserialized_or_reordered_roster(tmp_path):
    import pytest
    roster, _ = _full_fixture(tmp_path)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(roster.read_bytes() + b"\n")
    with pytest.raises(WorkloadError, match="original roster bytes"):
        _validate_full(replacement, roster)
    doc = json.loads(roster.read_bytes())
    doc["items"].reverse()
    replacement.write_text(json.dumps(doc))
    with pytest.raises(WorkloadError, match="original roster bytes"):
        _validate_full(replacement, roster)


def test_full500_rejects_parent_tampering_even_with_new_hash(tmp_path):
    import copy
    import pytest
    from concurrency_sweep_workload import build_execution_view
    roster, parent = _full_fixture(tmp_path)
    checked = _validate_full(roster, roster)
    for field in ("order", "row", "identity", "source"):
        changed = copy.deepcopy(parent)
        if field == "order":
            changed["items"].reverse()
        elif field == "row":
            changed["items"][0]["audio_seconds"] = 99.0
        elif field == "identity":
            changed["split_identity_hash"] = "f"*64
        else:
            changed["source_binding_sha256"] = "f"*64
        changed["runtime_view_sha256"] = _digest({k: v for k, v in changed.items() if k != "runtime_view_sha256"})
        with pytest.raises(WorkloadError):
            build_execution_view(checked, changed)
