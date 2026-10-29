"""The vLLM runtime identity's LIVE half: a process proves, per endpoint, that it is the pinned runtime (decision 9.2).

Owner decision 9.2 (2026-10-28, translated from Chinese: "go with the proposal"): ``vllm serve`` runs in the SAME container as the
harness; the pinned "binary" is the Python interpreter plus the vLLM / vllm-ascend source trees plus the image digest;
and the transport must re-verify ON EVERY SEND that a LIVE process with that identity owns the listening socket, as
strictly as the llama path does. The llama path's four checks (receipt agreement, pid liveness, ``/proc/<pid>/exe``
re-hashed, address-aware socket ownership) have no interpreter-shaped answer, so ``core.session.vllm_identity`` adds
the ones that do: namespaces (same container), cmdline, cwd, pinned environment, the instance's own device group, and
the memory map (every pinned library mapped, no second libpython, nothing deleted under a pinned root).

Every check here is driven from a FAKE ``/proc`` (``tests/unit/_vllm_world.py``) built from the receipt's own canonical
command line, so each test breaks exactly one thing and nothing touches an NPU, a server or a real serving process.

Multi-instance (coordinator design input, 2026-10-28): the identity is resolved PER ENDPOINT, so N instances with
their own ports and device groups verify independently, and a set that reuses an endpoint or a pid is refused.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from _vllm_world import PINNED_ENV, VllmWorld
from speech_aware_evidence_acquisition.contracts.gate_runtime_vllm import DEVICE_ENV_KEY, protocol_sha256
from speech_aware_evidence_acquisition.core.session import (
    SESSION_RECEIPT_SCHEMA_V2,
    SessionError,
    runtime_identity,
    runtime_identity_for_receipt,
    write_session_receipt,
)
from speech_aware_evidence_acquisition.core.session import lifecycle as lifecycle_module
from speech_aware_evidence_acquisition.core.session import vllm_identity as vllm_module

PID_A = 4100001
PID_B = 4100002


@pytest.fixture()
def world(tmp_path):
    return VllmWorld(tmp_path)


def _alive(world, session, receipt=None):
    vllm_module.assert_vllm_session_alive(
        session, receipt or world.receipt(), None, proc_root=world.proc_root
    )


# ---------------------------------------------------------------------------
# The happy path, and the seam that reaches it.
# ---------------------------------------------------------------------------


def test_a_pinned_instance_verifies(world):
    world.spawn("tp2-a", pid=PID_A)
    _alive(world, world.session("tp2-a", PID_A))


def test_two_instances_verify_independently(world):
    world.spawn("tp2-a", pid=PID_A)
    world.spawn("tp2-b", pid=PID_B)
    sessions = [world.session("tp2-a", PID_A), world.session("tp2-b", PID_B)]
    vllm_module.assert_vllm_sessions_alive(sessions, world.receipt(), None, proc_root=world.proc_root)


def test_four_instances_with_distinct_hccl_base_ports_share_one_protocol_hash_and_all_verify(world):
    """S12: the base port differs per instance (all share ``--network host``) and is deployment, not protocol."""

    four = world.receipt_with_instances(4)
    ports = [instance["env"]["HCCL_IF_BASE_PORT"] for instance in four["deployment"]["instances"]]
    assert ports == ["61000", "61200", "61400", "61600"]
    assert four["protocol_sha256"] == world.receipt()["protocol_sha256"]
    assert protocol_sha256(four) == four["protocol_sha256"]
    sessions = []
    for index in range(4):
        pid = PID_A + index
        world.spawn(f"tp2-{index}", pid=pid, receipt=four)
        sessions.append(world.session(f"tp2-{index}", pid, receipt=four))
    vllm_module.assert_vllm_sessions_alive(sessions, four, None, proc_root=world.proc_root)


def test_a_live_hccl_base_port_that_differs_from_the_instance_record_is_refused(world):
    four = world.receipt_with_instances(4)
    # Instance 1 is recorded at 61200 but its process carries instance 0's 61000: one protocol hash must not hide it.
    environ = dict(PINNED_ENV, HCCL_IF_BASE_PORT="61000", HCCL_SOCKET_IFNAME="eth0")
    world.spawn("tp2-1", pid=PID_A, receipt=four, environ=environ)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-1", PID_A, receipt=four), four)
    assert "HCCL_IF_BASE_PORT" in str(error.value)


def test_a_missing_or_extra_deployment_variable_is_refused(world):
    four = world.receipt_with_instances(2)
    missing = dict(PINNED_ENV, HCCL_SOCKET_IFNAME="eth0")  # no base port although the record has one
    world.spawn("tp2-0", pid=PID_A, receipt=four, environ=missing)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-0", PID_A, receipt=four), four)
    assert "HCCL_IF_BASE_PORT" in str(error.value)
    extra = dict(PINNED_ENV, HCCL_IF_BASE_PORT="61200", HCCL_SOCKET_IFNAME="eth0", TP_SOCKET_IFNAME="eth0")
    world.spawn("tp2-1", pid=PID_B, receipt=four, environ=extra)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-1", PID_B, receipt=four), four)
    assert "TP_SOCKET_IFNAME" in str(error.value)


def test_a_record_without_env_refuses_a_process_that_carries_a_deployment_variable(world):
    world.spawn("tp2-a", pid=PID_A, environ=dict(PINNED_ENV, HCCL_IF_BASE_PORT="61000"))
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "HCCL_IF_BASE_PORT" in str(error.value)


def test_a_serving_set_may_not_reuse_an_endpoint_or_a_pid(world):
    world.spawn("tp2-a", pid=PID_A)
    world.spawn("tp2-b", pid=PID_B)
    one = world.session("tp2-a", PID_A)
    with pytest.raises(SessionError) as error:
        vllm_module.assert_vllm_sessions_alive([one, one], world.receipt(), None, proc_root=world.proc_root)
    assert "claim the endpoint" in str(error.value)
    other = world.session("tp2-b", PID_A, session_id="vllm-tp2-b")
    with pytest.raises(SessionError) as error:
        vllm_module.assert_vllm_sessions_alive([one, other], world.receipt(), None, proc_root=world.proc_root)
    assert "claim pid" in str(error.value)
    with pytest.raises(SessionError):
        vllm_module.assert_vllm_sessions_alive([], world.receipt(), None, proc_root=world.proc_root)


def test_the_seam_routes_a_vllm_receipt_to_the_vllm_identity_and_everything_else_to_llama(world):
    assert runtime_identity("vllm").transport == "vllm"
    assert runtime_identity("llama-server").transport == "llama-server"
    assert runtime_identity_for_receipt(world.receipt()).transport == "vllm"
    for receipt in ({}, None, {"schema": "saea-runtime-receipt-v3"}, "not a mapping"):
        assert runtime_identity_for_receipt(receipt).transport == "llama-server"
    with pytest.raises(SessionError):
        runtime_identity("sglang")


def test_the_llama_identity_delegates_to_the_unchanged_function(monkeypatch):
    """The llama adapter is a delegation: it calls ``core.session.assert_session_alive``, nothing of its own."""

    from speech_aware_evidence_acquisition.core import session as session_package

    seen: dict = {}
    monkeypatch.setattr(
        session_package, "assert_session_alive", lambda *args: seen.update(args=args), raising=True
    )
    identity = runtime_identity("llama-server")
    identity.assert_session_alive("session", {"runtime": "pinned"}, Path("/runtime"))
    assert seen["args"][0] == "session" and seen["args"][1] == {"runtime": "pinned"}


def test_the_lifecycle_dispatches_a_vllm_receipt_to_the_vllm_adapter(world, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        vllm_module, "attach_vllm_session", lambda path, **kwargs: seen.update(path=path, kwargs=kwargs)
    )
    lifecycle_module.attach_session(
        Path("/nowhere/session.json"), runtime_receipt=world.receipt(), runtime_root=Path("/runtime")
    )
    assert seen["path"] == Path("/nowhere/session.json")
    assert seen["kwargs"]["runtime_receipt"]["schema"] == world.receipt()["schema"]


def test_the_lifecycle_leaves_a_llama_receipt_to_its_own_body(monkeypatch):
    """A llama receipt never reaches an adapter: the historical body runs, and fails exactly where it always did."""

    monkeypatch.setattr(
        vllm_module, "attach_vllm_session", lambda *a, **k: pytest.fail("a llama receipt was routed to vLLM")
    )
    with pytest.raises(SessionError) as error:
        lifecycle_module.attach_session(
            Path("/nowhere/session.json"), runtime_receipt={}, runtime_root=Path("/runtime")
        )
    assert "cannot read session receipt" in str(error.value)


# ---------------------------------------------------------------------------
# Declared agreement.
# ---------------------------------------------------------------------------


def test_a_session_naming_another_binary_is_refused(world):
    world.spawn("tp2-a", pid=PID_A)
    session = world.session("tp2-a", PID_A, binary_sha256="b" * 64)
    with pytest.raises(SessionError) as error:
        _alive(world, session)
    assert "binary identity" in str(error.value)


def test_a_session_naming_other_weights_is_refused(world):
    world.spawn("tp2-a", pid=PID_A)
    session = world.session("tp2-a", PID_A)
    tampered = session.model_files[:-1]
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A, model_files=tampered))
    assert "model file set" in str(error.value)


def test_a_session_for_an_undeployed_endpoint_is_refused(world):
    world.spawn("tp2-a", pid=PID_A)
    session = world.session("tp2-a", PID_A, port=8299)
    with pytest.raises(SessionError) as error:
        _alive(world, session)
    assert "no single deployment instance serves" in str(error.value)


def test_a_two_stage_speech_role_is_not_a_vllm_session(world):
    world.spawn("tp2-a", pid=PID_A)
    session = world.session("tp2-a", PID_A, schema=SESSION_RECEIPT_SCHEMA_V2, role="preprocess")
    with pytest.raises(SessionError):
        _alive(world, session)


def test_a_session_argv_that_is_not_this_instances_is_refused(world):
    """One receipt, N deployments: instance A's session may not carry instance B's command line."""

    world.spawn("tp2-a", pid=PID_A)
    session = world.session("tp2-a", PID_A, argv=tuple(world.argv_for("tp2-b")))
    with pytest.raises(SessionError) as error:
        _alive(world, session)
    assert "port" in str(error.value)


# ---------------------------------------------------------------------------
# Live process identity, one broken thing at a time.
# ---------------------------------------------------------------------------


def test_a_zombie_is_not_a_server(world):
    world.spawn("tp2-a", pid=PID_A, state="Z")
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "awaiting reaping" in str(error.value)


def test_a_pid_that_is_gone_is_refused(world):
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "not a live process" in str(error.value)


@pytest.mark.parametrize("namespace", ["mnt", "net"])
def test_a_process_in_another_container_is_refused(world, namespace):
    namespaces = {"mnt": "mnt:[4026531840]", "net": "net:[4026531840]"}
    namespaces[namespace] = f"{namespace}:[4026539999]"
    world.spawn("tp2-a", pid=PID_A, namespaces=namespaces)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "namespace" in str(error.value)


def test_another_executable_image_is_refused(world):
    other = world.root / "usr/bin/python3.11"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b"ELF-another-python")
    world.spawn("tp2-a", pid=PID_A, exe=other)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "not the pinned interpreter" in str(error.value)


def test_an_interpreter_deleted_after_exec_is_refused(world):
    world.spawn("tp2-a", pid=PID_A)
    (world.proc_root / str(PID_A) / "exe").unlink()
    os.symlink(f"{world.interpreter} (deleted)", world.proc_root / str(PID_A) / "exe")
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "deleted from disk after exec" in str(error.value)


def test_the_running_interpreter_is_rehashed_not_merely_path_matched(world):
    world.spawn("tp2-a", pid=PID_A)
    session = world.session("tp2-a", PID_A)
    world.interpreter.write_bytes(b"ELF-fake-python-stub" * 8 + b"patched")
    with pytest.raises(SessionError) as error:
        _alive(world, session)
    assert "hashes to" in str(error.value)


def test_a_live_command_line_that_drifted_from_the_receipt_is_refused(world):
    argv = world.argv_for("tp2-a")
    argv[argv.index("--max-model-len") + 1] = "32768"
    world.spawn("tp2-a", pid=PID_A, argv=argv)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "is not the one recorded for it" in str(error.value)


def test_a_live_command_line_that_disagrees_with_the_receipt_alone_is_refused(world):
    """Even when the SESSION recorded it: the receipt's protocol, not the session's own argv, is the authority."""

    argv = world.argv_for("tp2-a")
    argv[argv.index("--max-model-len") + 1] = "32768"
    world.spawn("tp2-a", pid=PID_A, argv=argv)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A, argv=tuple(argv)))
    assert "protocol flags" in str(error.value)


def test_a_server_serving_another_model_directory_is_refused(world):
    argv = world.argv_for("tp2-a")
    argv[4] = str(world.root / "cephfs" / "models" / "other")
    world.spawn("tp2-a", pid=PID_A, argv=argv)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A, argv=tuple(argv)))
    assert "pinned model directory" in str(error.value)


def test_a_wrong_working_directory_is_refused(world):
    other = world.root / "elsewhere"
    other.mkdir()
    world.spawn("tp2-a", pid=PID_A, cwd=other)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "pinned working directory" in str(error.value)


def test_a_changed_pinned_environment_variable_is_refused(world):
    environ = dict(PINNED_ENV)
    environ["VLLM_USE_V1"] = "0"
    world.spawn("tp2-a", pid=PID_A, environ=environ)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "VLLM_USE_V1" in str(error.value)


def test_a_missing_pinned_environment_variable_is_refused(world):
    environ = {key: value for key, value in PINNED_ENV.items() if key != "TASK_QUEUE_ENABLE"}
    world.spawn("tp2-a", pid=PID_A, environ=environ)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "TASK_QUEUE_ENABLE" in str(error.value)


def test_an_added_behaviour_relevant_variable_is_refused(world):
    environ = dict(PINNED_ENV)
    environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    world.spawn("tp2-a", pid=PID_A, environ=environ)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "VLLM_ATTENTION_BACKEND" in str(error.value)


def test_unpinned_variables_are_allowed(world):
    """The world always adds HOSTNAME/TERM: a serving process is not required to run in an empty environment."""

    world.spawn("tp2-a", pid=PID_A)
    _alive(world, world.session("tp2-a", PID_A))


def test_an_instance_on_another_device_group_is_refused(world):
    environ = dict(PINNED_ENV)
    environ[DEVICE_ENV_KEY] = "2,3"
    world.spawn("tp2-a", pid=PID_A, environ=environ)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert DEVICE_ENV_KEY in str(error.value)


def test_a_second_interpreter_runtime_in_the_memory_map_is_refused(world):
    other = world.root / "opt/other/libpython3.11.so.1.0"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"ELF-another-libpython")
    world.spawn(
        "tp2-a", pid=PID_A, maps=[str(world.interpreter), str(world.libpython), str(other)]
    )
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "unpinned interpreter runtime" in str(error.value)


def test_an_unmapped_pinned_library_is_refused(world):
    world.spawn("tp2-a", pid=PID_A, maps=[str(world.interpreter)])
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "has not mapped the pinned runtime libraries" in str(error.value)


def test_a_deleted_mapping_under_a_pinned_root_is_refused(world):
    maps = [
        str(world.interpreter),
        str(world.libpython),
        f"{world.trees['vllm'] / 'entrypoints' / 'cli.py'} (deleted)",
    ]
    world.spawn("tp2-a", pid=PID_A, maps=maps)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "deleted mappings under the pinned roots" in str(error.value)


def test_a_deleted_mapping_outside_the_pinned_roots_is_not_an_identity_failure(world):
    maps = [str(world.interpreter), str(world.libpython), "/tmp/scratch-buffer (deleted)"]
    world.spawn("tp2-a", pid=PID_A, maps=maps)
    _alive(world, world.session("tp2-a", PID_A))


def test_a_process_that_owns_no_listening_socket_is_refused(world):
    world.spawn("tp2-a", pid=PID_A, listening=False)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "no LISTEN socket found" in str(error.value)


def test_a_process_that_does_not_own_the_endpoints_socket_is_refused(world):
    """Interpreter and source-tree identity do not prove WHO answers this port (the llama path's fix-2 gap)."""

    world.spawn("tp2-a", pid=PID_A, owns_socket=False)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-a", PID_A))
    assert "does not own the listening socket" in str(error.value)


def test_one_instances_socket_does_not_authorize_another(world):
    world.spawn("tp2-a", pid=PID_A)
    world.spawn("tp2-b", pid=PID_B, listening=False)
    with pytest.raises(SessionError) as error:
        _alive(world, world.session("tp2-b", PID_B))
    assert "8201" in str(error.value)


# ---------------------------------------------------------------------------
# Lifecycle.
# ---------------------------------------------------------------------------


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.stdout = None

    def poll(self):
        return None


def test_start_spawns_the_receipts_own_command_line_and_environment(world, tmp_path):
    world.spawn("tp2-a", pid=PID_A)
    calls: dict = {}

    def popen(argv, **kwargs):
        calls.update(argv=argv, kwargs=kwargs)
        return _FakeProcess(PID_A)

    receipt_path = tmp_path / "session-receipt.json"
    session = vllm_module.start_vllm_session(
        runtime_receipt=world.receipt(),
        host="127.0.0.1",
        port=8200,
        session_id="SAEA-vllm-start",
        extra_argv=["--max-model-len", "24576", "--dtype", "bfloat16", "--reasoning-parser", "qwen3",
                    "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--seed", "0",
                    "--max-num-seqs", "8"],
        popen=popen,
        ready_timeout_seconds=0,
        receipt_path=receipt_path,
        proc_root=world.proc_root,
    )
    assert calls["argv"] == world.argv_for("tp2-a")
    assert calls["kwargs"]["cwd"] == str(world.cwd)
    # The pinned environment plus this instance's device group, and NOTHING inherited from the harness shell.
    assert calls["kwargs"]["env"] == {**PINNED_ENV, DEVICE_ENV_KEY: "0,1"}
    assert session.pid == PID_A and session.port == 8200
    assert session.argv == tuple(world.argv_for("tp2-a"))
    assert receipt_path.exists()
    _alive(world, session)


def test_start_spawns_with_the_instances_recorded_deployment_environment(world):
    four = world.receipt_with_instances(2)
    world.spawn("tp2-1", pid=PID_B, receipt=four)
    calls: dict = {}

    def popen(argv, **kwargs):
        calls.update(argv=argv, kwargs=kwargs)
        return _FakeProcess(PID_B)

    vllm_module.start_vllm_session(
        runtime_receipt=four,
        host="127.0.0.1",
        port=8201,
        session_id="SAEA-vllm-start-env",
        extra_argv=["--max-model-len", "24576", "--dtype", "bfloat16", "--reasoning-parser", "qwen3",
                    "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--seed", "0",
                    "--max-num-seqs", "8"],
        popen=popen,
        ready_timeout_seconds=0,
        proc_root=world.proc_root,
    )
    assert calls["kwargs"]["env"] == {
        **PINNED_ENV,
        DEVICE_ENV_KEY: "2,3",
        "HCCL_IF_BASE_PORT": "61200",
        "HCCL_SOCKET_IFNAME": "eth0",
    }


def test_start_refuses_a_geometry_that_drifted_from_the_receipt(world):
    world.spawn("tp2-a", pid=PID_A)
    with pytest.raises(SessionError) as error:
        vllm_module.start_vllm_session(
            runtime_receipt=world.receipt(),
            host="127.0.0.1",
            port=8200,
            session_id="SAEA-vllm-start",
            extra_argv=["--max-model-len", "32768", "--max-num-seqs", "8"],
            popen=lambda *a, **k: pytest.fail("a drifted geometry reached a spawn"),
            ready_timeout_seconds=0,
            proc_root=world.proc_root,
        )
    assert "protocol flags" in str(error.value)


def test_start_refuses_a_concurrency_that_is_not_this_instances(world):
    world.spawn("tp2-a", pid=PID_A)
    with pytest.raises(SessionError) as error:
        vllm_module.start_vllm_session(
            runtime_receipt=world.receipt(),
            host="127.0.0.1",
            port=8200,
            session_id="SAEA-vllm-start",
            extra_argv=[*world.receipt()["protocol"]["server_args"], "--max-num-seqs", "4"],
            popen=lambda *a, **k: pytest.fail("a drifted deployment reached a spawn"),
            ready_timeout_seconds=0,
            proc_root=world.proc_root,
        )
    assert "--max-num-seqs" in str(error.value)


def test_start_refuses_an_environment_override(world):
    world.spawn("tp2-a", pid=PID_A)
    with pytest.raises(SessionError) as error:
        vllm_module.start_vllm_session(
            runtime_receipt=world.receipt(),
            host="127.0.0.1",
            port=8200,
            session_id="SAEA-vllm-start",
            extra_argv=[*world.receipt()["protocol"]["server_args"], "--max-num-seqs", "8"],
            extra_env={"VLLM_USE_V1": "0"},
            popen=lambda *a, **k: pytest.fail("an environment override reached a spawn"),
            ready_timeout_seconds=0,
            proc_root=world.proc_root,
        )
    assert "serving environment is receipt identity" in str(error.value)


def test_start_refuses_an_interpreter_swapped_between_gate_and_spawn(world):
    world.spawn("tp2-a", pid=PID_A)
    receipt = world.receipt()
    world.interpreter.write_bytes(b"ELF-swapped")
    with pytest.raises(SessionError) as error:
        vllm_module.start_vllm_session(
            runtime_receipt=receipt,
            host="127.0.0.1",
            port=8200,
            session_id="SAEA-vllm-start",
            extra_argv=[*receipt["protocol"]["server_args"], "--max-num-seqs", "8"],
            popen=lambda *a, **k: pytest.fail("a swapped interpreter was executed"),
            ready_timeout_seconds=0,
            proc_root=world.proc_root,
        )
    assert "never executed" in str(error.value)


def test_start_kills_its_own_child_when_the_child_fails_verification(world, monkeypatch):
    receipt = world.receipt()
    # No /proc entry for this pid: the spawned process cannot prove it is the pinned runtime.
    signalled: list = []
    monkeypatch.setattr(vllm_module.os, "kill", lambda pid, signal_number: signalled.append((pid, signal_number)))
    with pytest.raises(SessionError):
        vllm_module.start_vllm_session(
            runtime_receipt=receipt,
            host="127.0.0.1",
            port=8200,
            session_id="SAEA-vllm-start",
            extra_argv=[*receipt["protocol"]["server_args"], "--max-num-seqs", "8"],
            popen=lambda *a, **k: _FakeProcess(PID_A),
            ready_timeout_seconds=0,
            proc_root=world.proc_root,
        )
    assert [pid for pid, _signal in signalled] == [PID_A]


def test_start_refuses_a_speech_feature_role(world):
    with pytest.raises(SessionError) as error:
        vllm_module.start_vllm_session(
            runtime_receipt=world.receipt(),
            host="127.0.0.1",
            port=8200,
            session_id="SAEA-vllm-start",
            role="preprocess",
            popen=lambda *a, **k: pytest.fail("a speech-feature role reached a spawn"),
            proc_root=world.proc_root,
        )
    assert "two-stage speech-feature role" in str(error.value)


def test_attach_binds_to_a_resident_instance_through_its_receipt(world, tmp_path):
    world.spawn("tp2-a", pid=PID_A)
    path = tmp_path / "session-receipt.json"
    write_session_receipt(path, world.session("tp2-a", PID_A))
    attached = vllm_module.attach_vllm_session(
        path, runtime_receipt=world.receipt(), proc_root=world.proc_root
    )
    assert attached.pid == PID_A and attached.port == 8200


def test_attach_refuses_a_forged_receipt(world, tmp_path):
    world.spawn("tp2-a", pid=PID_A)
    path = tmp_path / "session-receipt.json"
    write_session_receipt(path, world.session("tp2-a", PID_B, session_id="vllm-forged"))
    with pytest.raises(SessionError):
        vllm_module.attach_vllm_session(path, runtime_receipt=world.receipt(), proc_root=world.proc_root)


def test_stop_records_an_already_exited_instance_without_signalling_anything(world, tmp_path):
    """The fake pid does not exist on this host, so the one durable outcome is ALREADY_EXITED."""

    gone = 2 ** 22 - 7
    world.spawn("tp2-a", pid=gone)
    receipt_path = tmp_path / "stop-receipt.json"
    document = vllm_module.stop_vllm_session(
        world.session("tp2-a", gone),
        runtime_receipt=world.receipt(),
        receipt_path=receipt_path,
        proc_root=world.proc_root,
    )
    assert document["termination_signal"] == "ALREADY_EXITED"
    assert document["schema"] == "saea-session-stop-receipt-v1"
    assert receipt_path.exists()


def test_stop_refuses_a_session_that_is_not_the_receipts(world):
    world.spawn("tp2-a", pid=PID_A)
    with pytest.raises(SessionError):
        vllm_module.stop_vllm_session(
            world.session("tp2-a", PID_A, binary_sha256="c" * 64),
            runtime_receipt=world.receipt(),
            proc_root=world.proc_root,
        )


# ---------------------------------------------------------------------------
# The transport: the same re-verification, at construction and before every send.
# ---------------------------------------------------------------------------


def _transport(world, monkeypatch, session, receipt=None, **kwargs):
    from speech_aware_evidence_acquisition.core.model.vllm_transport import VllmServerTransport

    monkeypatch.setattr(vllm_module, "DEFAULT_PROC_ROOT", world.proc_root)
    return VllmServerTransport(
        f"http://{session.host}:{session.port}",
        world.root,
        session,
        world.receipt() if receipt is None else receipt,
        world.root / "runtime",
        post=lambda url, body: b"{}",
        served_model_name="qwen3-omni-30b-a3b-thinking",
        **kwargs,
    )


def test_the_transport_verifies_the_vllm_identity_at_construction_and_on_demand(world, monkeypatch):
    world.spawn("tp2-a", pid=PID_A)
    transport = _transport(world, monkeypatch, world.session("tp2-a", PID_A))
    transport.assert_session_alive()
    # The same check again, with the endpoint's socket no longer owned by anyone: refused, not cached.
    (world.proc_root / "net" / "tcp").write_text("  sl  local_address rem_address   st\n", encoding="utf-8")
    with pytest.raises(SessionError) as error:
        transport.assert_session_alive()
    assert "no LISTEN socket found" in str(error.value)


def test_the_transport_refuses_to_exist_without_a_provable_instance(world, monkeypatch):
    world.spawn("tp2-a", pid=PID_A)
    forged = world.session("tp2-a", PID_B, session_id="vllm-forged")
    with pytest.raises(SessionError):
        _transport(world, monkeypatch, forged)


def test_a_vllm_transport_pinning_a_llama_receipt_still_uses_the_llama_check(world, monkeypatch):
    """The override changes WHICH identity answers, never how strictly: a llama receipt keeps the historical call."""

    from speech_aware_evidence_acquisition.core.model import transport as transport_module

    world.spawn("tp2-a", pid=PID_A)
    seen: list = []
    monkeypatch.setattr(transport_module, "assert_session_alive", lambda *args: seen.append(args))
    monkeypatch.setattr(
        vllm_module, "assert_vllm_session_alive", lambda *a, **k: pytest.fail("a llama receipt reached the vLLM check")
    )
    transport = _transport(world, monkeypatch, world.session("tp2-a", PID_A), receipt={})
    transport.assert_session_alive()
    assert len(seen) == 2
