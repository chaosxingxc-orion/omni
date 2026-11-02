import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

path = Path(__file__).parents[2] / "scripts/analysis/finalized_generation_release.py"
spec = importlib.util.spec_from_file_location("finalized_release", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def setup(tmp_path, monkeypatch):
    manifest = tmp_path / "runs/r/a/r.run-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    locks = tmp_path / "locks"
    locks.mkdir()
    (locks / "generation.guard").touch()
    out = tmp_path / "work/codex/job/release.json"
    out.parent.mkdir(parents=True)
    bundle = SimpleNamespace(attempt_state="FINALIZED", manifest={"plan": {"run_id": "r", "attempt_id": "a"}})
    monkeypatch.setattr(m, "verify_bundle", lambda p: bundle)
    return manifest, out, bundle


def test_verified_terminal_event_is_idempotent_and_time_is_not_invented(tmp_path, monkeypatch):
    manifest, out, _ = setup(tmp_path, monkeypatch)
    doc = m.release(tmp_path, manifest, out, "r")
    assert doc["scoring_complete"] and doc["original_release_time_unknown"]
    assert "campaign_seconds_to_release" not in doc
    assert m.release(tmp_path, manifest, out, "r") == doc
    manifest.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="differs"):
        m.release(tmp_path, manifest, out, "r")


@pytest.mark.parametrize("fault", ["lease", "state", "run", "verify"])
def test_refuses_unverified_or_active_predecessor(tmp_path, monkeypatch, fault):
    manifest, out, bundle = setup(tmp_path, monkeypatch)
    if fault == "lease": (tmp_path / "locks/flight.lock").write_text("active")
    elif fault == "state": bundle.attempt_state = "OPENED"
    elif fault == "run": bundle.manifest["plan"]["run_id"] = "other"
    else:
        def fail(p): raise ValueError("corrupt bundle")
        monkeypatch.setattr(m, "verify_bundle", fail)
    with pytest.raises(ValueError): m.release(tmp_path, manifest, out, "r")
    assert not out.exists()
