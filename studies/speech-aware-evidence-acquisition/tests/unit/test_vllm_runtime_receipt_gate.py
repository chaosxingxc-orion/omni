"""The vLLM runtime receipt at GATE time: what it must say, and that every word of it is re-verified from bytes.

Owner decision 9.3 (2026-10-28, "可以，但是最好用adapter模式兼容" -- approved, but preferably compatible via an adapter
pattern): a vLLM runtime-receipt schema sits BESIDE the llama.cpp one without loosening it, reached through one
runtime-identity seam (``contracts.runtime_identity``) that the gate, the transport and the session lifecycle call
instead of branching on the backend. These tests cover the receipt half of that seam:

* the seam selects by the model fragment's ``transport``, and an unregistered transport closes the gate;
* a llama.cpp receipt is not a vLLM receipt and a vLLM receipt is not a llama.cpp receipt -- neither validator has
  been taught to tolerate the other's document;
* every pinned byte is re-measured: the interpreter, its runtime library, the import-path files that decide what
  ``import vllm`` resolves to, both editable source trees, each package's METADATA/RECORD, and the served weights;
* the protocol/deployment split holds: a deployment change (one more instance, a wider instance, a different port)
  leaves ``protocol_sha256`` untouched, while anything that changes what the model computes moves it.

No NPU, no server, no network: see ``tests/unit/_vllm_world.py``.
"""

from __future__ import annotations

import copy
import json

import pytest

from _vllm_world import MODEL_LOCK_KEY, SUBDIR, VllmWorld, sha
from speech_aware_evidence_acquisition.contracts import GateClosed
from speech_aware_evidence_acquisition.contracts.gate_runtime_vllm import protocol_sha256
from speech_aware_evidence_acquisition.contracts.runtime_identity import (
    LLAMA_SERVER_TRANSPORT,
    VLLM_TRANSPORT,
    runtime_receipt_identity,
)


@pytest.fixture()
def world(tmp_path):
    return VllmWorld(tmp_path)


def _validate(world, receipt=None, **gate_overrides):
    gate = world.gate(receipt=receipt, **gate_overrides)
    diagnostics: dict = {}
    runtime_receipt_identity(gate.transport).validate_runtime_receipt(
        gate, world.lock, full_hash=True, diagnostics=diagnostics
    )
    return diagnostics


# ---------------------------------------------------------------------------
# The seam.
# ---------------------------------------------------------------------------


def test_the_seam_selects_an_adapter_by_transport_and_refuses_an_unknown_one():
    assert runtime_receipt_identity(LLAMA_SERVER_TRANSPORT).transport == LLAMA_SERVER_TRANSPORT
    assert runtime_receipt_identity(VLLM_TRANSPORT).transport == VLLM_TRANSPORT
    with pytest.raises(GateClosed) as error:
        runtime_receipt_identity("sglang")
    assert "no runtime identity is registered" in str(error.value)
    with pytest.raises(GateClosed):
        runtime_receipt_identity(None)


def test_a_valid_world_opens_and_reports_its_protocol_and_instances(world):
    diagnostics = _validate(world)
    assert diagnostics["vllm_protocol_sha256"] == world.receipt()["protocol_sha256"]
    assert diagnostics["vllm_instances"] == ["tp2-a", "tp2-b"]
    # The llama path's own per-file loop did the weights, unchanged and fully re-hashed.
    assert diagnostics["gguf_files_total"] == 4
    assert diagnostics["gguf_files_full_hashed"] == 4


def test_the_vllm_validator_is_reached_through_assert_model_touch_allowed(world, monkeypatch):
    """The gate calls the seam, not a backend branch: the vLLM validator is what a ``transport='vllm'`` gate runs."""

    gate = world.gate()
    seen: dict = {}
    monkeypatch.setattr(
        type(gate), "_validate_vllm_runtime", lambda self, lock=None, **kwargs: seen.update(called=True)
    )
    monkeypatch.setattr(type(gate), "_validate_runtime", lambda self, lock=None, **kwargs: seen.update(llama=True))
    runtime_receipt_identity(gate.transport).validate_runtime_receipt(gate, world.lock)
    assert seen == {"called": True}


def test_a_llama_receipt_is_refused_under_the_vllm_transport(world):
    llama = {
        "schema": "saea-runtime-receipt-v2",
        "study": "speech-aware-evidence-acquisition",
        "model_lock_key": MODEL_LOCK_KEY,
        "llama_cpp": {"binary_path": "llama.cpp/build/bin/llama-server", "binary_sha256": "a" * 64},
        "gguf_files": [],
    }
    with pytest.raises(GateClosed) as error:
        _validate(world, receipt=llama)
    assert "schema must be" in str(error.value)


def test_a_vllm_receipt_is_refused_under_the_llama_transport(world):
    """The llama validator was not loosened: it still demands its own schema and its own binary."""

    with pytest.raises(GateClosed):
        _validate(world, transport=LLAMA_SERVER_TRANSPORT)


# ---------------------------------------------------------------------------
# Re-verification from bytes.
# ---------------------------------------------------------------------------


def test_the_interpreter_is_rehashed(world):
    world.interpreter.write_bytes(b"ELF-fake-python-stub" * 8 + b"\0")
    with pytest.raises(GateClosed) as error:
        _validate(world)
    assert "interpreter" in str(error.value)


def test_a_swapped_runtime_library_is_refused(world):
    world.libpython.write_bytes(b"ELF-other-libpython" * 64)
    with pytest.raises(GateClosed) as error:
        _validate(world)
    assert "runtime_libraries[0]" in str(error.value)


def test_an_unlisted_import_path_file_is_refused(world):
    """A new ``.pth`` can silently redirect ``import vllm``; an exact set, not a subset, is what is pinned."""

    (world.site_packages / "zz-operator-added.pth").write_bytes(b"/tmp/elsewhere\n")
    with pytest.raises(GateClosed) as error:
        _validate(world)
    assert "import-path files drifted" in str(error.value)


def test_an_edited_source_tree_file_is_refused(world):
    (world.trees["vllm"] / "entrypoints" / "cli.py").write_bytes(b"def main():\n    return 1\n")
    with pytest.raises(GateClosed) as error:
        _validate(world)
    assert "source tree vllm" in str(error.value)


def test_an_added_source_tree_file_is_refused(world):
    (world.trees["vllm_ascend"] / "patch.py").write_bytes(b"# hotfix\n")
    with pytest.raises(GateClosed) as error:
        _validate(world)
    assert "source tree vllm_ascend" in str(error.value)


def test_bytecode_is_not_part_of_source_tree_identity(world):
    (world.trees["vllm"] / "__pycache__" / "more.pyc").write_bytes(b"still not identity")
    _validate(world)


def test_a_source_tree_symlink_is_refused(world):
    (world.trees["vllm"] / "link.py").symlink_to(world.trees["vllm"] / "__init__.py")
    with pytest.raises(GateClosed) as error:
        _validate(world)
    assert "symlink" in str(error.value)


def test_a_package_downgrade_is_refused(world):
    dist_info, _version = world.packages["vllm"]
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: vllm\nVersion: 0.29.0\n\nfake\n", encoding="utf-8"
    )
    with pytest.raises(GateClosed) as error:
        _validate(world)
    assert "vllm" in str(error.value)


def test_a_missing_package_record_is_refused(world):
    dist_info, _version = world.packages["vllm_ascend"]
    (dist_info / "RECORD").unlink()
    with pytest.raises(GateClosed) as error:
        _validate(world)
    assert "RECORD" in str(error.value)


def test_an_edited_weight_file_is_refused(world):
    (world.model_dir / "config.json").write_bytes(b'{"model_type": "something_else"}\n')
    with pytest.raises(GateClosed) as error:
        _validate(world)
    assert "SHA-256 drift" in str(error.value) or "size drift" in str(error.value)


def test_an_extra_file_in_the_served_model_directory_is_refused(world):
    (world.model_dir / "adapter_model.safetensors").write_bytes(b"lora")
    with pytest.raises(GateClosed) as error:
        _validate(world)
    assert "served model directory files drifted" in str(error.value)


def test_a_symlinked_weight_is_refused(world):
    target = world.root / "elsewhere.safetensors"
    target.write_bytes(world.model_files["model-00001-of-00002.safetensors"])
    (world.model_dir / "model-00001-of-00002.safetensors").unlink()
    (world.model_dir / "model-00001-of-00002.safetensors").symlink_to(target)
    with pytest.raises(GateClosed) as error:
        _validate(world)
    assert "symlink" in str(error.value)


def test_the_lock_file_count_must_agree(world):
    lock = world.lock
    lock["models"][0]["files"] = 3
    gate = world.gate()
    with pytest.raises(GateClosed) as error:
        runtime_receipt_identity(gate.transport).validate_runtime_receipt(gate, lock)
    assert "model files but the lock freezes" in str(error.value)


def test_the_receipt_must_name_this_runs_model(world):
    with pytest.raises(GateClosed) as error:
        _validate(world, expected_model_lock_key="some-other-model")
    assert "does not match the model this run declares" in str(error.value)


def test_the_served_directory_must_be_the_locks_directory(world):
    receipt = world.receipt()
    receipt["model"]["model_dir"] = str(world.root / "cephfs" / "models" / "other")
    receipt["protocol_sha256"] = protocol_sha256(receipt)
    with pytest.raises(GateClosed) as error:
        _validate(world, receipt=receipt)
    assert "is not the lock's directory" in str(error.value)


# ---------------------------------------------------------------------------
# Document structure: what a receipt may and may not say.
# ---------------------------------------------------------------------------


def test_a_tampered_protocol_hash_is_refused(world):
    receipt = world.receipt()
    receipt["protocol"]["server_args"] = ["--max-model-len", "32768"]
    with pytest.raises(GateClosed) as error:
        _validate(world, receipt=receipt)
    assert "protocol_sha256" in str(error.value)


def test_deployment_flags_may_not_hide_in_the_protocol_args(world):
    receipt = world.receipt(server_args=["--max-model-len", "24576", "--port", "8299"])
    with pytest.raises(GateClosed) as error:
        _validate(world, receipt=receipt)
    assert "protocol.server_args must hold only protocol flag groups" in str(error.value)


def test_a_flag_that_loads_unpinned_code_is_refused(world):
    receipt = world.receipt(server_args=["--max-model-len", "24576", "--trust-remote-code"])
    with pytest.raises(GateClosed) as error:
        _validate(world, receipt=receipt)
    assert "--trust-remote-code" in str(error.value)


def test_the_user_site_directory_must_be_off(world):
    env = dict(world.receipt()["protocol"]["env"])
    del env["PYTHONNOUSERSITE"]
    receipt = world.receipt(env=env)
    with pytest.raises(GateClosed) as error:
        _validate(world, receipt=receipt)
    assert "PYTHONNOUSERSITE" in str(error.value)


def test_unpinned_environment_variables_may_not_ride_in_the_receipt(world):
    env = dict(world.receipt()["protocol"]["env"])
    env["HOSTNAME"] = "npu-181"
    receipt = world.receipt(env=env)
    with pytest.raises(GateClosed) as error:
        _validate(world, receipt=receipt)
    assert "pinned behaviour-relevant variables" in str(error.value)


def test_the_image_id_must_be_a_digest(world):
    receipt = world.receipt()
    receipt["container"]["image_id"] = "latest"
    receipt["protocol_sha256"] = protocol_sha256(receipt)
    with pytest.raises(GateClosed) as error:
        _validate(world, receipt=receipt)
    assert "image_id" in str(error.value)


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda instances: instances[1].update(port=8200), "reuses the endpoint"),
        (lambda instances: instances[1].update(device_ids=[1, 2]), "shares devices"),
        (lambda instances: instances[1].update(device_ids=[2, 3, 4]), "devices for tensor-parallel size"),
        (lambda instances: instances[1].update(instance_id="tp2-a"), "unique identifier"),
        (lambda instances: instances[1].update(host="10.0.0.5"), "must be a loopback literal"),
        (lambda instances: instances[1].update(max_num_seqs=0), "max_num_seqs"),
        (lambda instances: instances[1].update(gpu_memory_utilization=1.5), "gpu_memory_utilization"),
    ],
)
def test_the_deployment_section_is_checked_per_instance(world, mutate, expected):
    receipt = world.receipt()
    mutate(receipt["deployment"]["instances"])
    with pytest.raises(GateClosed) as error:
        _validate(world, receipt=receipt)
    assert expected in str(error.value)


def test_a_deployment_change_is_not_a_protocol_change(world):
    """The coordinator's requirement (2026-10-28): adding or resizing an instance must not move ``protocol_sha256``."""

    receipt = world.receipt()
    before = receipt["protocol_sha256"]
    instances = receipt["deployment"]["instances"]
    instances.append(
        {
            "instance_id": "tp2-c",
            "host": "127.0.0.1",
            "port": 8202,
            "device_ids": [4, 5],
            "tensor_parallel_size": 2,
            "max_num_seqs": 16,
            "gpu_memory_utilization": 0.85,
            "max_num_batched_tokens": 8192,
        }
    )
    instances[0]["max_num_seqs"] = 32
    instances[0]["gpu_memory_utilization"] = 0.8
    receipt["protocol_sha256"] = protocol_sha256(receipt)
    assert receipt["protocol_sha256"] == before
    diagnostics = _validate(world, receipt=receipt)
    assert diagnostics["vllm_instances"] == ["tp2-a", "tp2-b", "tp2-c"]


def test_a_protocol_change_does_move_the_protocol_hash(world):
    changed = world.receipt(server_args=["--max-model-len", "32768"])
    assert changed["protocol_sha256"] != world.receipt()["protocol_sha256"]


def test_an_unexpected_receipt_key_is_refused(world):
    receipt = world.receipt()
    receipt["extra"] = "whatever"
    with pytest.raises(GateClosed) as error:
        _validate(world, receipt=receipt)
    assert "must have exact keys" in str(error.value)


def test_a_relative_model_file_path_outside_the_lock_subdir_is_refused(world):
    receipt = world.receipt()
    receipt["model"]["files"][0]["path"] = "models/other/config.json"
    receipt["protocol_sha256"] = protocol_sha256(receipt)
    with pytest.raises(GateClosed) as error:
        _validate(world, receipt=receipt)
    assert "config.json" in str(error.value)


def test_the_served_weights_must_be_declared_in_full(world):
    receipt = world.receipt()
    receipt["model"]["files"] = receipt["model"]["files"][:3]
    receipt["protocol_sha256"] = protocol_sha256(receipt)
    with pytest.raises(GateClosed) as error:
        _validate(world, receipt=receipt)
    assert "model files" in str(error.value)


def test_both_source_trees_must_be_pinned(world):
    receipt = world.receipt()
    receipt["source_trees"] = [tree for tree in receipt["source_trees"] if tree["package"] != "vllm_ascend"]
    receipt["protocol_sha256"] = protocol_sha256(receipt)
    with pytest.raises(GateClosed) as error:
        _validate(world, receipt=receipt)
    assert "source_trees must include" in str(error.value)


def test_both_packages_must_be_pinned(world):
    receipt = world.receipt()
    receipt["packages"] = [package for package in receipt["packages"] if package["name"] != "vllm_ascend"]
    receipt["protocol_sha256"] = protocol_sha256(receipt)
    with pytest.raises(GateClosed) as error:
        _validate(world, receipt=receipt)
    assert "packages must include" in str(error.value)


def test_a_receipt_that_is_not_json_is_refused(world):
    gate = world.gate()
    gate.runtime_receipt.write_text("not json", encoding="utf-8")
    with pytest.raises(GateClosed):
        runtime_receipt_identity(gate.transport).validate_runtime_receipt(gate, world.lock)


def test_the_world_itself_is_internally_consistent(world):
    """A guard on the fixture: the receipt describes the files on disk, byte for byte."""

    receipt = world.receipt()
    assert receipt["interpreter"]["sha256"] == sha(world.interpreter.read_bytes())
    assert {entry["path"] for entry in receipt["model"]["files"]} == {
        f"{SUBDIR}/{name}" for name in world.model_files
    }
    assert json.loads(json.dumps(receipt)) == receipt
    assert copy.deepcopy(receipt) == world.receipt()
