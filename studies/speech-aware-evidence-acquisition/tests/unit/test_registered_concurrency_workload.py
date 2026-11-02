import importlib.util
import json
import hashlib
from pathlib import Path

import pytest


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def registered(tmp_path):
    root = Path(__file__).parents[2]
    w = _load(root / "scripts/analysis/concurrency_sweep_workload.py", "workload")
    r = _load(root / "scripts/analysis/registered_workload.py", "registered")
    rows = [{"sample_id": f"s{i}", "packet_id": f"p{i}", "media_relpath": f"datasets/slurp/slurp_real/{i}.flac", "audio_sha256": f"{i+1:064x}", "audio_seconds": 1.0, "sample_rate_hz": 16000} for i in range(100)]
    roster = tmp_path / "roster.json"; roster.write_text(json.dumps({"schema": "saea-real-engine-roster-v1", "items": rows}, sort_keys=True))
    import hashlib
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"schema": w.MANIFEST_SCHEMA, "status": "PINNED_CPU_ONLY_NOT_ADMITTED", "source_roster": "real-engine-500-v1", "source_roster_sha256": hashlib.sha256(roster.read_bytes()).hexdigest(), "count": 100, "items": rows, "unique_sample_ids": 100, "unique_audio_hashes": 100, "test_labels_read": False, "model_calls": 0}, sort_keys=True))
    spec = {"cell": "x", "experiment": "x", "workload_kind": "concurrency-sweep-real100", "manifest_path": str(manifest), "source_roster_path": str(roster), "profile": "p", "profile_sha256": "a" * 64, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "rows": 100, "new_model_calls": 0}
    identity = hashlib.sha256("\n".join(sorted("slurp/" + row["sample_id"] for row in rows)).encode()).hexdigest()
    spec.update(run_id="real100", attempt_id="generation-v1", split_identity_hash=identity)
    exp = {"cells": [{"run_id": spec["run_id"], "attempt_id": spec["attempt_id"], "dataset_fragment": "real100"}]}
    view = {"count": 100, "items": rows, "split_identity_hash": identity}
    view["runtime_view_sha256"] = hashlib.sha256(json.dumps(view, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    exp_path = tmp_path / "configs/experiment/x.json"
    view_path = tmp_path / "configs/dataset/real100.json"
    for path, doc in ((exp_path, exp), (view_path, view)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
    return r, spec, exp_path, view_path


def test_registered_workload_dispatches_fixed_manifest(tmp_path, registered):
    r, spec, _, _ = registered
    assert r.validate_workload(tmp_path, tmp_path, "x", spec) == 100


@pytest.mark.parametrize("mutation", ["500_rows", "different_audio", "reordered", "stale_hash", "wrong_split"])
def test_rejects_execution_view_drift(tmp_path, registered, mutation):
    r, spec, _, path = registered
    view = json.loads(path.read_bytes())
    if mutation == "500_rows":
        view["items"] *= 5
        view["count"] = 500
    elif mutation == "different_audio":
        view["items"][0]["audio_sha256"] = "f" * 64
    elif mutation == "reordered":
        view["items"].reverse()
    elif mutation == "stale_hash":
        view["runtime_view_sha256"] = "f" * 64
    else:
        view["split_identity_hash"] = "f" * 64
    path.write_text(json.dumps(view))
    with pytest.raises(ValueError, match="concurrency"):
        r.validate_workload(tmp_path, tmp_path, "x", spec)


@pytest.mark.parametrize("field", ["run_id", "attempt_id"])
def test_rejects_execution_attempt_drift(tmp_path, registered, field):
    r, spec, path, _ = registered
    exp = json.loads(path.read_bytes())
    exp["cells"][0][field] = "wrong"
    path.write_text(json.dumps(exp))
    with pytest.raises(ValueError, match="registered attempt differs"):
        r.validate_workload(tmp_path, tmp_path, "x", spec)


def test_execution_view_derives_exact_subset_and_rejects_parent_drift():
    root = Path(__file__).parents[2]
    w = _load(root / "scripts/analysis/concurrency_sweep_workload.py", "view_workload")
    rows = [{"sample_id": f"slurp:train:{i}", "audio_seconds": 1.0} for i in range(500)]
    parent = {"count": 500, "items": rows, "loader": "unchanged"}
    parent["runtime_view_sha256"] = hashlib.sha256(json.dumps(parent, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    checked = {"status": "HELD_NOT_ADMITTED", "items": rows[:100], "rows": 100, "manifest_sha256": "a"*64}
    view = w.build_execution_view(checked, parent)
    assert view["items"] == rows[:100]
    assert view["loader"] == "unchanged"
    assert view["audio_seconds_total"] == 100.0
    parent["items"][200]["audio_seconds"] = 2.0
    with pytest.raises(ValueError, match="parent runtime view hash differs"):
        w.build_execution_view(checked, parent)
