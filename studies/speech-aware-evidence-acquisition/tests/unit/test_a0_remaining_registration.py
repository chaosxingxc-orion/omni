import copy
import json
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from register_a0_remaining import derive_documents, NAME
from registered_workload import validate_workload


def test_continuation_budget_population_and_method_are_preserved():
    exp = {"cells": [{"cell_id": "old"}], "execution_plan": {"planned_model_calls": 2000},
           "stage3_measurement": {"rows": 500, "read_roster": "engineering500"}}
    base = {"native_contact_fit": {"run_id": "old", "attempt_id": "old", "contact_cap": 1000},
            "prepared_phase_supply": {"relpath": "old", "sha256": "old"},
            "unchanged_method": {"roles": ["perception", "solver"], "thinking_budget": 3072}}
    original = copy.deepcopy(base)
    view = {"count": 353, "items": [{"sample_id": str(i)} for i in range(353)],
        "audio_seconds_total": 1087.9, "split_name": "remaining", "split_identity_hash": "a" * 64}
    candidate = {"model_fragment": "model", "service_session": "session"}
    e, b = derive_documents(exp, base, view, candidate)
    assert base == original
    assert e["execution_plan"]["planned_model_calls"] == 1412
    assert e["execution_plan"]["planned_speech_audio_seconds"] == 4352
    assert e["stage3_measurement"]["read_roster"] == "remaining"
    assert e["population"]["count"] == 353
    assert e["cells"][0]["readiness"]["state"] == "held"
    assert b["native_contact_fit"]["contact_cap"] == 706
    b.pop("native_contact_fit"); original.pop("native_contact_fit"); original.pop("prepared_phase_supply")
    assert b == original
    view["items"][-1] = view["items"][0]
    with pytest.raises(ValueError, match="distinct"):
        derive_documents(exp, base, view, candidate)


def test_workload_rejects_mislabeled_count_before_admission(tmp_path):
    for directory in ("experiment", "dataset"):
        (tmp_path / "configs" / directory).mkdir(parents=True)
    (tmp_path / "configs/experiment/x.json").write_text(json.dumps({"cells": [{"run_id": "r", "attempt_id": "a", "dataset_fragment": "x"}]}))
    (tmp_path / "configs/dataset/x.json").write_text(json.dumps({"count": 500, "items": [{"sample_id": "a"}], "split_identity_hash": "hash"}))
    spec = {"cell": "x", "experiment": "x", "run_id": "r", "attempt_id": "a", "rows": 500, "split_identity_hash": "hash"}
    with pytest.raises(ValueError, match="population"):
        validate_workload(tmp_path, tmp_path, "x", spec)


def test_selection_refuses_partial_or_missing_comparisons(tmp_path, monkeypatch):
    import bind_a0_selected_profile as binding
    import speech_aware_evidence_acquisition.core.bundle as bundle_module
    from types import SimpleNamespace
    doc = {"schema": "saea-real500-profile-selection-v1", "complete_roster_comparison": True,
        "reason": "whole roster throughput and quality", "selected_run_id": "r", "candidates": []}
    with pytest.raises(ValueError, match="comparison missing"):
        binding.verify_selection(tmp_path, doc)
    monkeypatch.setattr(binding, "pin", lambda root, value: tmp_path / "manifest.json")
    monkeypatch.setattr(bundle_module, "verify_bundle", lambda path: SimpleNamespace(
        manifest={"run_id":"r"}, attempt_state="FINALIZED", outputs_rows=[{"sample_id":"x"}]))
    doc["candidates"] = [{"manifest": {}}]
    with pytest.raises(ValueError, match="incomplete"):
        binding.verify_selection(tmp_path, doc)


def test_selection_refuses_fabricated_wall_timing(tmp_path, monkeypatch):
    import bind_a0_selected_profile as binding
    import speech_aware_evidence_acquisition.core.bundle as bundle_module
    from types import SimpleNamespace
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    start=tmp_path/'start.json';start.write_text(json.dumps({'time':manifest.stat().st_mtime-20}))
    seal=tmp_path/'seal.json';seal.write_text(json.dumps({'run_id':'r','status':'COMPLETED'}))
    monkeypatch.setattr(binding,'pin',lambda root,value: root/value['relpath'])
    monkeypatch.setattr(bundle_module,'verify_bundle',lambda path:SimpleNamespace(
        manifest={'run_id':'r'},attempt_state='FINALIZED',outputs_rows=[{'sample_id':str(i)} for i in range(500)]))
    doc={'schema':'saea-real500-profile-selection-v1','complete_roster_comparison':True,
         'reason':'compare full rosters','selected_run_id':'r','candidates':[{
            'manifest':{'relpath':'manifest.json'},'started':{'relpath':'start.json'},
            'generation_manifest':{'relpath':'seal.json'},'generation_elapsed_seconds':1}]}
    with pytest.raises(ValueError,match='timing disagrees'):
        binding.verify_selection(tmp_path,doc)
