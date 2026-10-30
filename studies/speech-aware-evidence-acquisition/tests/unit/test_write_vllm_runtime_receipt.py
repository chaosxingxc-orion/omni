"""The vLLM runtime-receipt PRODUCER (``scripts/write_vllm_runtime_receipt.py``), driven against a fake ``/proc``.

The receipt is not hand-written: it is observed from the running servers it describes (that script's own docstring).
These tests drive it against ``tests/unit/_vllm_world.py``'s fake world -- two fake instances, a container-shaped file
tree and a real model directory -- and check the three things that matter:

1. what it writes is accepted by the gate-time validator and by the per-endpoint live check, with the deployment facts
   (device ids, tensor-parallel size, ``--max-num-seqs``, ``--gpu-memory-utilization``, port) taken from each
   instance and the protocol taken once;
2. it refuses rather than guesses: two instances with different protocols, a server with no device group, a pid that
   is not a vLLM server, an interpreter that disagrees between instances;
3. it never replaces an existing receipt.

No NPU, no server, no network: no process is started and no pid is signalled -- the fake pids are only read.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from _vllm_world import IMAGE_ID, IMAGE_REF, MODEL_LOCK_KEY, SERVED, VllmWorld
from speech_aware_evidence_acquisition.contracts.gate_runtime_vllm import protocol_sha256
from speech_aware_evidence_acquisition.contracts.runtime_identity import runtime_receipt_identity

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/write_vllm_runtime_receipt.py"
PID_A = 4200001
PID_B = 4200002


def _module():
    spec = importlib.util.spec_from_file_location("write_vllm_runtime_receipt", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def world(tmp_path):
    return VllmWorld(tmp_path)


def _argv(world, output: Path, *extra: str) -> list[str]:
    return [
        "--instance", f"tp2-a={PID_A}",
        "--instance", f"tp2-b={PID_B}",
        "--model-lock-key", MODEL_LOCK_KEY,
        "--materialization-root", str(world.materialization_root),
        "--image-ref", IMAGE_REF,
        "--image-id", IMAGE_ID,
        "--site-packages", str(world.site_packages),
        "--source-tree", f"vllm={world.trees['vllm']}",
        "--source-tree", f"vllm_ascend={world.trees['vllm_ascend']}",
        "--package", "vllm",
        "--package", "vllm_ascend",
        "--proc-root", str(world.proc_root),
        "--output", str(output),
        *extra,
    ]


def _spawn_both(world):
    world.spawn("tp2-a", pid=PID_A)
    world.spawn("tp2-b", pid=PID_B)


def test_the_written_receipt_describes_both_instances_and_opens_the_gate(world, tmp_path):
    _spawn_both(world)
    output = tmp_path / "runtime-vllm.json"
    assert _module().main(_argv(world, output)) == 0
    document = json.loads(output.read_text(encoding="utf-8"))

    assert document["schema"] == "saea-runtime-receipt-vllm-v1"
    assert document["model_lock_key"] == MODEL_LOCK_KEY
    assert document["container"] == {"image_ref": IMAGE_REF, "image_id": IMAGE_ID}
    assert document["protocol"]["served_model_name"] == SERVED
    assert document["protocol"]["cwd"] == str(world.cwd)
    assert document["protocol"]["argv0"] == str(world.interpreter)
    assert document["protocol_sha256"] == protocol_sha256(document)
    # The observed protocol is the receipt's protocol, and the geometry/served name never leak into it.
    assert document["protocol"]["server_args"] == world.receipt()["protocol"]["server_args"]
    # Deployment, per instance, exactly as each one runs.
    assert [
        (
            instance["instance_id"],
            instance["port"],
            instance["device_ids"],
            instance["tensor_parallel_size"],
            instance["max_num_seqs"],
            instance["gpu_memory_utilization"],
        )
        for instance in document["deployment"]["instances"]
    ] == [("tp2-a", 8200, [0, 1], 2, 8, 0.9), ("tp2-b", 8201, [2, 3], 2, 8, 0.9)]
    # The weights are re-hashed from disk, not copied from anywhere.
    assert {entry["sha256"] for entry in document["model"]["files"]} == {
        entry["sha256"] for entry in world.receipt()["model"]["files"]
    }
    # And the whole thing is what the gate accepts.
    gate = world.gate(receipt=document)
    runtime_receipt_identity(gate.transport).validate_runtime_receipt(gate, world.lock, full_hash=True)


def test_the_written_receipt_is_byte_identical_to_the_fixtures_own(world, tmp_path):
    """The fixture and the producer agree, so every other test's hand-built world is the real thing."""

    _spawn_both(world)
    output = tmp_path / "runtime-vllm.json"
    assert _module().main(_argv(world, output)) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    # The producer also records the namespace policy and the harness hostname under deployment (unhashed).
    deployment = document["deployment"]
    assert deployment.pop("namespace_policy") == "same-container"
    assert isinstance(deployment.pop("harness_hostname"), str)
    assert document == world.receipt()


def test_distinct_hccl_base_ports_become_per_instance_env_and_leave_one_protocol(world, tmp_path):
    """S12: the producer records the deployment variables per instance and keeps them out of ``protocol.env``."""

    from _vllm_world import PINNED_ENV

    world.spawn("tp2-a", pid=PID_A, environ=dict(PINNED_ENV, HCCL_IF_BASE_PORT="61000", HCCL_SOCKET_IFNAME="eth0"))
    world.spawn("tp2-b", pid=PID_B, environ=dict(PINNED_ENV, HCCL_IF_BASE_PORT="61200", HCCL_SOCKET_IFNAME="eth0"))
    output = tmp_path / "runtime-vllm.json"
    assert _module().main(_argv(world, output)) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert [i["env"] for i in document["deployment"]["instances"]] == [
        {"HCCL_IF_BASE_PORT": "61000", "HCCL_SOCKET_IFNAME": "eth0"},
        {"HCCL_IF_BASE_PORT": "61200", "HCCL_SOCKET_IFNAME": "eth0"},
    ]
    assert document["protocol"]["env"] == world.receipt()["protocol"]["env"]
    assert document["protocol_sha256"] == world.receipt()["protocol_sha256"]
    gate = world.gate(receipt=document)
    runtime_receipt_identity(gate.transport).validate_runtime_receipt(gate, world.lock, full_hash=True)


def test_a_second_instance_only_changes_the_deployment_half(world, tmp_path):
    """The coordinator's requirement: attaching an instance must not move ``protocol_sha256``."""

    _spawn_both(world)
    module = _module()
    one = tmp_path / "one.json"
    assert module.main(_argv(world, one)[:2] + _argv(world, one)[4:]) == 0
    both = tmp_path / "both.json"
    assert module.main(_argv(world, both)) == 0
    single = json.loads(one.read_text(encoding="utf-8"))
    pair = json.loads(both.read_text(encoding="utf-8"))
    assert single["protocol_sha256"] == pair["protocol_sha256"]
    assert [i["instance_id"] for i in single["deployment"]["instances"]] == ["tp2-a"]
    assert [i["instance_id"] for i in pair["deployment"]["instances"]] == ["tp2-a", "tp2-b"]


def test_two_protocols_are_two_receipts(world, tmp_path, capsys):
    world.spawn("tp2-a", pid=PID_A)
    argv = world.argv_for("tp2-b")
    argv[argv.index("--max-model-len") + 1] = "32768"
    world.spawn("tp2-b", pid=PID_B, argv=argv)
    assert _module().main(_argv(world, tmp_path / "out.json")) == 2
    assert "does not share" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


def test_an_instance_without_its_own_device_group_is_refused(world, tmp_path, capsys):
    from _vllm_world import PINNED_ENV

    world.spawn("tp2-a", pid=PID_A, environ=dict(PINNED_ENV))
    (world.proc_root / str(PID_A) / "environ").write_bytes(
        b"\0".join(f"{key}={value}".encode("utf-8") for key, value in PINNED_ENV.items()) + b"\0"
    )
    world.spawn("tp2-b", pid=PID_B)
    assert _module().main(_argv(world, tmp_path / "out.json")) == 2
    assert "ASCEND_RT_VISIBLE_DEVICES" in capsys.readouterr().err


def test_a_pid_that_is_not_a_vllm_server_is_refused(world, tmp_path, capsys):
    world.spawn("tp2-a", pid=PID_A, argv=[str(world.interpreter), "-c", "print(1)"])
    world.spawn("tp2-b", pid=PID_B)
    assert _module().main(_argv(world, tmp_path / "out.json")) == 2
    assert "not a pinnable vLLM server" in capsys.readouterr().err


def test_instances_running_different_interpreters_are_refused(world, tmp_path, capsys):
    other = world.root / "usr/bin/python3.11"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b"ELF-another-python")
    world.spawn("tp2-a", pid=PID_A)
    world.spawn("tp2-b", pid=PID_B, exe=other)
    assert _module().main(_argv(world, tmp_path / "out.json")) == 2
    assert "executable image" in capsys.readouterr().err


def test_a_server_that_does_not_own_its_socket_is_refused(world, tmp_path, capsys):
    """The producer runs the same live check the gate will: an unprovable server never gets a receipt."""

    world.spawn("tp2-a", pid=PID_A, owns_socket=False)
    world.spawn("tp2-b", pid=PID_B)
    assert _module().main(_argv(world, tmp_path / "out.json")) == 2
    assert "listening socket" in capsys.readouterr().err


def test_an_existing_receipt_is_never_replaced(world, tmp_path):
    _spawn_both(world)
    output = tmp_path / "runtime-vllm.json"
    module = _module()
    assert module.main(_argv(world, output)) == 0
    before = output.read_bytes()
    with pytest.raises(module.ProducerError) as error:
        module.main(_argv(world, output))
    assert "already exists" in str(error.value)
    assert output.read_bytes() == before


def test_each_instance_needs_its_own_id(world, tmp_path):
    _spawn_both(world)
    module = _module()
    argv = _argv(world, tmp_path / "out.json")
    argv[argv.index("tp2-b=" + str(PID_B))] = f"tp2-a={PID_B}"
    with pytest.raises(module.ProducerError):
        module.main(argv)
