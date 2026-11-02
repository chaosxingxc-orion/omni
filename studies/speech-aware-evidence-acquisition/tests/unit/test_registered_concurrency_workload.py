import importlib.util
import json
from pathlib import Path


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_registered_workload_dispatches_fixed_manifest(tmp_path):
    root = Path(__file__).parents[2]
    w = _load(root / "scripts/analysis/concurrency_sweep_workload.py", "workload")
    r = _load(root / "scripts/analysis/registered_workload.py", "registered")
    rows = [{"sample_id": f"s{i}", "packet_id": f"p{i}", "media_relpath": f"datasets/slurp/slurp_real/{i}.flac", "audio_sha256": f"{i+1:064x}", "audio_seconds": 1.0, "sample_rate_hz": 16000} for i in range(100)]
    roster = tmp_path / "roster.json"; roster.write_text(json.dumps({"schema": "saea-real-engine-roster-v1", "items": rows}, sort_keys=True))
    import hashlib
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"schema": w.MANIFEST_SCHEMA, "status": "PINNED_CPU_ONLY_NOT_ADMITTED", "source_roster": "real-engine-500-v1", "source_roster_sha256": hashlib.sha256(roster.read_bytes()).hexdigest(), "count": 100, "items": rows, "unique_sample_ids": 100, "unique_audio_hashes": 100, "test_labels_read": False, "model_calls": 0}, sort_keys=True))
    spec = {"cell": "x", "experiment": "x", "workload_kind": "concurrency-sweep-real100", "manifest_path": str(manifest), "source_roster_path": str(roster), "profile": "p", "profile_sha256": "a" * 64, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "rows": 100, "new_model_calls": 0}
    assert r.validate_workload(tmp_path, tmp_path, "x", spec) == 100
