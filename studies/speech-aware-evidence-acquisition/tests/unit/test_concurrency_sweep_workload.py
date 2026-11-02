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
