"""The namespace policy of the vLLM runtime identity (plan 2026-10-30, Decision 3; owner item 9.2 reading pending).

``same-container`` (default, today's behaviour) needs the harness and the server in one mount and network namespace.
``shared-pid-host`` (servers in their own containers with ``--pid host``) replaces that by: same host, server pid
visible with the pinned executable and command line, and an endpoint that answers ``/v1/models`` with the recorded
model. The policy is a deployment fact: profile field ``namespace_policy`` -> ``docker run --pid host`` and the
receipt's ``deployment.namespace_policy``; neither moves any hash.

Driven from the fake ``/proc`` of ``tests/unit/_vllm_world.py``; the endpoint probe is replaced by a stub, so nothing
here touches a network, a card or a server.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from _vllm_world import IMAGE_ID, IMAGE_REF, MODEL_LOCK_KEY, SERVED, VllmWorld
from speech_aware_evidence_acquisition.contracts.errors import GateClosed
from speech_aware_evidence_acquisition.contracts.gate_runtime_vllm import (
    DEFAULT_NAMESPACE_POLICY,
    NAMESPACE_POLICIES,
    namespace_policy,
    protocol_sha256,
    validate_vllm_receipt_document,
)
from speech_aware_evidence_acquisition.core import deployment as dep
from speech_aware_evidence_acquisition.core.session import SessionError
from speech_aware_evidence_acquisition.core.session import vllm_identity as vllm_module

PID_A = 4300001
PID_B = 4300002
HOST = "npu-181-harness"
FOREIGN = {"mnt": "mnt:[4026539991]", "net": "net:[4026539992]"}  # another container's namespaces
REPO = Path(__file__).resolve().parents[2]
PROFILE_ID = "qwen3-omni-thinking-bf16-181-4xtp2-v1"
NIC = "enp61s0f2"


@pytest.fixture()
def world(tmp_path):
    w = VllmWorld(tmp_path)
    hostname = w.proc_root / "sys" / "kernel" / "hostname"
    hostname.parent.mkdir(parents=True)
    hostname.write_text(HOST + "\n", encoding="utf-8")
    return w


@pytest.fixture()
def models(monkeypatch):
    """The endpoint probe: ``models.body`` is what GET /v1/models returns, ``models.error`` makes it fail."""

    class Probe:
        body: object = {"object": "list", "data": [{"id": SERVED}]}
        error: Exception | None = None
        urls: list[str] = []

        def __call__(self, url, timeout):
            self.urls.append(url)
            if self.error is not None:
                raise self.error
            return self.body

    probe = Probe()
    probe.urls = []
    monkeypatch.setattr(vllm_module, "http_get_json", probe)
    return probe


def _relaxed(world, hostname=HOST, **deployment):
    receipt = world.receipt()
    receipt["deployment"]["namespace_policy"] = "shared-pid-host"
    if hostname is not None:
        receipt["deployment"]["harness_hostname"] = hostname
    receipt["deployment"].update(deployment)
    return receipt


def _alive(world, receipt, instance="tp2-a", pid=PID_A):
    vllm_module.assert_vllm_session_alive(
        world.session(instance, pid, receipt=receipt), receipt, None, proc_root=world.proc_root
    )


# --- the default is today's behaviour --------------------------------------------------------------------------------


def test_the_default_policy_is_same_container_and_a_receipt_without_one_means_it(world):
    assert DEFAULT_NAMESPACE_POLICY == "same-container" and NAMESPACE_POLICIES == ("same-container", "shared-pid-host")
    assert namespace_policy(world.receipt()) == "same-container"


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("namespace", ["mnt", "net"])
def test_same_container_refuses_another_namespace_with_or_without_the_explicit_field(world, models, namespace, explicit):
    receipt = world.receipt()
    if explicit:
        receipt["deployment"]["namespace_policy"] = "same-container"
    namespaces = {"mnt": "mnt:[4026531840]", "net": "net:[4026531840]"}
    namespaces[namespace] = FOREIGN[namespace]
    world.spawn("tp2-a", pid=PID_A, namespaces=namespaces)
    with pytest.raises(SessionError, match="namespace"):
        _alive(world, receipt)
    assert models.urls == []  # the strict policy never probes the endpoint


# --- shared-pid-host accepts ------------------------------------------------------------------------------------------


def test_shared_pid_host_accepts_servers_in_other_mount_and_network_namespaces(world, models):
    receipt = _relaxed(world)
    world.spawn("tp2-a", pid=PID_A, namespaces=FOREIGN)
    world.spawn("tp2-b", pid=PID_B, namespaces=FOREIGN)
    sessions = [world.session("tp2-a", PID_A, receipt=receipt), world.session("tp2-b", PID_B, receipt=receipt)]
    vllm_module.assert_vllm_sessions_alive(sessions, receipt, None, proc_root=world.proc_root)
    assert models.urls == ["http://127.0.0.1:8200/v1/models", "http://127.0.0.1:8201/v1/models"]


def test_the_relaxed_policy_still_runs_every_other_live_check(world, models):
    """Relaxing the namespace line does not relax the interpreter, environment, device or socket checks."""

    receipt = _relaxed(world)
    world.spawn("tp2-a", pid=PID_A, namespaces=FOREIGN, owns_socket=False)
    with pytest.raises(SessionError, match="does not own the listening socket"):
        _alive(world, receipt)


# --- shared-pid-host refusals: each names the instance and the failed condition ----------------------------------------


def _refused(world, receipt, condition, **spawn):
    world.spawn("tp2-a", pid=PID_A, **{"namespaces": FOREIGN, **spawn})
    with pytest.raises(SessionError) as error:
        _alive(world, receipt)
    message = str(error.value)
    assert "tp2-a" in message and f"condition {condition!r} failed" in message and "shared-pid-host" in message
    return message


def test_a_harness_on_another_host_is_refused(world, models):
    message = _refused(world, _relaxed(world, hostname="some-other-host"), "same-host")
    assert "some-other-host" in message and HOST in message


def test_a_receipt_without_a_recorded_harness_hostname_is_refused_by_the_gate_and_the_live_check(world, models):
    receipt = _relaxed(world, hostname=None)
    with pytest.raises(GateClosed, match="harness_hostname"):
        validate_vllm_receipt_document(receipt)
    world.spawn("tp2-a", pid=PID_A, namespaces=FOREIGN)
    with pytest.raises(SessionError, match="harness_hostname"):  # the session's own structure check
        _alive(world, receipt)
    instance = receipt["deployment"]["instances"][0]  # and the live check itself, should a caller skip both
    with pytest.raises(SessionError) as error:
        vllm_module._verify_namespaces(world.proc_root, PID_A, receipt, instance, "127.0.0.1", 8200)
    assert "tp2-a" in str(error.value) and "condition 'same-host' failed" in str(error.value)


def test_a_server_pid_that_is_not_visible_is_refused(world, models):
    receipt = _relaxed(world)
    instance = receipt["deployment"]["instances"][0]
    with pytest.raises(SessionError) as error:
        vllm_module._verify_namespaces(world.proc_root, PID_A, receipt, instance, "127.0.0.1", 8200)
    assert "tp2-a" in str(error.value) and "condition 'pid-visible' failed" in str(error.value)
    assert "--pid host" in str(error.value)


def test_another_executable_is_refused(world, models):
    other = world.root / "usr/bin/python3.11"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b"ELF-another-python")
    _refused(world, _relaxed(world), "executable-image", exe=other)


def test_another_command_line_is_refused(world, models):
    argv = world.argv_for("tp2-b")  # instance B's command line on instance A's pid
    _refused(world, _relaxed(world), "command-line", argv=argv)


def test_an_unreachable_endpoint_is_refused(world, models):
    models.error = ConnectionRefusedError("connection refused")
    message = _refused(world, _relaxed(world), "endpoint-reachable")
    assert "connection refused" in message


def test_an_endpoint_that_serves_another_model_is_refused(world, models):
    models.body = {"data": [{"id": "some-other-model"}]}
    message = _refused(world, _relaxed(world), "endpoint-reachable")
    assert "some-other-model" in message and SERVED in message
    for junk in (None, [], {"data": "x"}, {"data": [None]}):
        models.body = junk
        with pytest.raises(SessionError, match="endpoint-reachable"):
            _alive(world, _relaxed(world))


# --- the receipt field: optional, validated, outside the protocol hash -------------------------------------------------


def test_the_policy_and_the_hostname_are_deployment_facts_outside_the_protocol_hash(world):
    plain, relaxed = world.receipt(), _relaxed(world)
    assert protocol_sha256(relaxed) == protocol_sha256(plain) == plain["protocol_sha256"]
    validate_vllm_receipt_document(plain)
    validate_vllm_receipt_document(relaxed)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(namespace_policy="same-host"), "namespace_policy"),
        (lambda d: d.update(namespace_policy=None), "namespace_policy"),
        (lambda d: d.update(harness_hostname=""), "harness_hostname"),
        (lambda d: d.update(harness_hostname=7), "harness_hostname"),
        (lambda d: d.update(surprise=1), "optional"),
    ],
)
def test_a_malformed_deployment_section_is_refused(world, mutate, message):
    receipt = world.receipt()
    mutate(receipt["deployment"])
    with pytest.raises(GateClosed, match=message):
        validate_vllm_receipt_document(receipt)


# --- the profile field and the docker argv ----------------------------------------------------------------------------


def _document():
    return json.loads(dep.resolve_profile_path(PROFILE_ID).read_text(encoding="utf-8"))


def test_namespace_policy_is_an_optional_deployment_only_profile_field():
    document = _document()
    assert "namespace_policy" in dep._OPTIONAL_PROFILE_KEYS
    base = dep.validate_deployment_profile(document)
    assert base.namespace_policy is None and base.effective_namespace_policy == "same-container"
    for policy in NAMESPACE_POLICIES:
        profile = dep.validate_deployment_profile(dict(document, namespace_policy=policy))
        assert profile.namespace_policy == policy and profile.effective_namespace_policy == policy
        assert profile.sha256 != base.sha256  # the profile document changed; the protocol did not (below)
        assert dep.deployment_record(profile)["namespace_policy"] == policy
    assert dep.deployment_record(base)["namespace_policy"] == "same-container"
    for bad in ("", "host", "shared-pid", 1, True, ["shared-pid-host"]):
        with pytest.raises(dep.DeploymentError, match="namespace_policy"):
            dep.validate_deployment_profile(dict(document, namespace_policy=bad))


def test_pid_host_is_emitted_only_under_shared_pid_host_and_nothing_else_changes():
    document = _document()
    base = dep.validate_deployment_profile(document)
    relaxed = dep.validate_deployment_profile(dict(document, namespace_policy="shared-pid-host"))
    explicit = dep.validate_deployment_profile(dict(document, namespace_policy="same-container"))
    fragment = dep.model_fragment_values(base)
    for index in range(len(base.instances)):
        plain = dep.docker_run_argv(base, base.instances[index], fragment, nic=NIC)
        assert "--pid" not in plain and "host" == plain[plain.index("--network") + 1]
        pid = dep.docker_run_argv(relaxed, relaxed.instances[index], fragment, nic=NIC)
        assert pid.count("--pid") == 1 and pid[pid.index("--pid") + 1] == "host"
        stripped = list(pid)
        del stripped[stripped.index("--pid"):stripped.index("--pid") + 2]
        # identical but the --pid host pair and the profile hash label (the profile document differs)
        label = f"{dep.LABEL_PROFILE_SHA256}="
        assert [t for t in stripped if not t.startswith(label)] == [t for t in plain if not t.startswith(label)]
        assert dep.docker_run_argv(explicit, explicit.instances[index], fragment, nic=NIC).count("--pid") == 0


def test_the_policy_moves_no_protocol_or_server_argv_hash():
    document = _document()
    base = dep.validate_deployment_profile(document)
    relaxed = dep.validate_deployment_profile(dict(document, namespace_policy="shared-pid-host"))
    fragment = dep.model_fragment_values(base)
    for index in range(len(base.instances)):
        assert dep.instance_server_argv(relaxed, relaxed.instances[index], fragment) == dep.instance_server_argv(
            base, base.instances[index], fragment
        )
        assert dep.instance_container_args(relaxed, relaxed.instances[index], fragment) == dep.instance_container_args(
            base, base.instances[index], fragment
        )
    record, relaxed_record = dep.deployment_record(base), dep.deployment_record(relaxed)
    assert [i["server_argv_sha256"] for i in record["instances"]] == [
        i["server_argv_sha256"] for i in relaxed_record["instances"]
    ]
    assert record["protocol_window_tokens"] == relaxed_record["protocol_window_tokens"]
    assert dep.model_fragment_values(relaxed) == fragment  # the model fragment (the hashed side) is untouched


# --- the receipt writer records both, unhashed ------------------------------------------------------------------------


def _writer():
    path = REPO / "scripts/write_vllm_runtime_receipt.py"
    spec = importlib.util.spec_from_file_location("write_vllm_runtime_receipt_ns", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _writer_argv(world, output, *extra):
    return [
        "--instance", f"tp2-a={PID_A}", "--instance", f"tp2-b={PID_B}",
        "--model-lock-key", MODEL_LOCK_KEY, "--materialization-root", str(world.materialization_root),
        "--image-ref", IMAGE_REF, "--image-id", IMAGE_ID, "--site-packages", str(world.site_packages),
        "--source-tree", f"vllm={world.trees['vllm']}", "--source-tree", f"vllm_ascend={world.trees['vllm_ascend']}",
        "--package", "vllm", "--package", "vllm_ascend",
        "--proc-root", str(world.proc_root), "--output", str(output), *extra,
    ]


def test_the_writer_records_the_policy_and_the_harness_hostname_outside_the_hash(world, models, tmp_path):
    world.spawn("tp2-a", pid=PID_A, namespaces=FOREIGN)
    world.spawn("tp2-b", pid=PID_B, namespaces=FOREIGN)
    output = tmp_path / "runtime-relaxed.json"
    assert _writer().main(_writer_argv(world, output, "--namespace-policy", "shared-pid-host")) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["deployment"]["namespace_policy"] == "shared-pid-host"
    assert document["deployment"]["harness_hostname"] == HOST
    assert document["protocol_sha256"] == world.receipt()["protocol_sha256"] == protocol_sha256(document)
    validate_vllm_receipt_document(document)
    assert models.urls  # the writer's own pre-write verification ran under the relaxed policy


def test_the_writer_default_is_same_container_and_refuses_foreign_namespaces(world, models, tmp_path, capsys):
    world.spawn("tp2-a", pid=PID_A, namespaces=FOREIGN)
    world.spawn("tp2-b", pid=PID_B, namespaces=FOREIGN)
    assert _writer().main(_writer_argv(world, tmp_path / "never.json")) == 2
    assert "namespace" in capsys.readouterr().err and not (tmp_path / "never.json").exists()
