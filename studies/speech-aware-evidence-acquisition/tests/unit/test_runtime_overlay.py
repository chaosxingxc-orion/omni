"""Runtime controls cannot silently miss the native scheduler or alter prompts."""
import copy
import hashlib
import json

import pytest

from speech_aware_evidence_acquisition.core.flight.factory import load_external_runtime_config
from speech_aware_evidence_acquisition.core.flight.runtime_overlay import apply_runtime_overlay


DISPATCH = {
    "schema": "saea-dispatch-slot-fill-v1", "role_input_offload": True,
    "ready_request": True, "prepare_workers": 2, "validation_workers": 2,
    "model_inflight": 1, "admitted_trajectories": 4,
}


def native_overlay(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({
        "schema": "saea-native-efficiency-v1", "dispatch": DISPATCH,
        "audio_cache": {"enabled": False, "max_entries": 0,
                        "max_bytes": 0, "preload_workers": 1},
    }))
    return path


def test_native_config_pins_and_copies_controls_without_mutating_science(tmp_path, monkeypatch):
    path = native_overlay(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setenv("SAEA_RUNTIME_CONFIG_SHA256", digest)
    overlay = load_external_runtime_config(path)
    values = {"native_episode": {}, "policy": "native", "transport": "vllm",
              "task_instruction": "Keep this exact instruction."}
    before = copy.deepcopy(values)
    result = apply_runtime_overlay(values, overlay)
    assert values == before and result["task_instruction"] == before["task_instruction"]
    assert result["dispatch"] == DISPATCH
    assert result["runtime_config_identity"]["sha256"] == digest
    overlay["dispatch"]["model_inflight"] = 99
    assert result["dispatch"]["model_inflight"] == 1


@pytest.mark.parametrize("change,match", [
    ({"dispatch": {}}, "collides"),
    ({"native_node_batch": {}}, "another scheduler"),
    ({"native_node_pipeline": {}}, "another scheduler"),
    ({"native_prefix_slots": {}}, "another scheduler"),
    ({"request_pipeline": {"enabled": True}}, "another scheduler"),
    ({"policy": "fixed"}, "native role"),
    ({"transport": "llama-server"}, "vllm"),
    ({"native_episode": None}, "native role"),
])
def test_native_overlay_refuses_ineffective_or_conflicting_configuration(tmp_path, change, match):
    overlay = load_external_runtime_config(native_overlay(tmp_path))
    values = {"native_episode": {}, "policy": "native", "transport": "vllm", **change}
    with pytest.raises(ValueError, match=match):
        apply_runtime_overlay(values, overlay)


def test_direct_pipeline_cannot_claim_to_optimize_native_episodes():
    with pytest.raises(ValueError, match="native episodes require dispatch"):
        apply_runtime_overlay({"native_episode": {}}, {"request_pipeline": {"enabled": True}})


@pytest.mark.parametrize("change", [
    {"prepare_workers": 0}, {"model_inflight": True},
    {"task_instruction": "change"}, {"ready_request": True, "role_input_offload": False},
])
def test_native_loader_refuses_invalid_controls_before_factory(tmp_path, change):
    path = native_overlay(tmp_path)
    doc = json.loads(path.read_text())
    doc["dispatch"].update(change)
    path.write_text(json.dumps(doc))
    with pytest.raises(Exception):
        load_external_runtime_config(path)


def test_native_loader_rejects_mixed_schedulers_and_scientific_fields(tmp_path):
    path = native_overlay(tmp_path)
    original = json.loads(path.read_text())
    for key in ("request_pipeline", "thinking_policy", "max_tokens"):
        path.write_text(json.dumps({**original, key: {}}))
        with pytest.raises(Exception, match="unexpected fields"):
            load_external_runtime_config(path)


def test_native_queue_cannot_be_smaller_than_model_window():
    with pytest.raises(ValueError, match="must cover"):
        apply_runtime_overlay(
            {"native_episode": {}, "policy": "native", "transport": "vllm"},
            {"dispatch": {**DISPATCH, "model_inflight": 8}},
        )


def test_native_fit_overlay_binds_only_raw_vllm_native_dispatch():
    values = {"native_episode": {}, "native_contact_fit": {},
              "arm": "raw-speech-agent-native-tools", "transport": "vllm"}
    overlay = {"dispatch": DISPATCH, "native_fit": {"max_concurrency": 16}}
    result = apply_runtime_overlay(values, overlay)
    assert result["native_fit"]["max_concurrency"] == 16

    with pytest.raises(ValueError, match="requires raw-speech"):
        apply_runtime_overlay({"native_episode": {}, "native_contact_fit": {},
                               "arm": "raw-speech-agent-native-tools",
                               "transport": "llama-server"}, overlay)
