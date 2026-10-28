"""Contract: the Ascend container authority added by NPU port decision 9.1 (2026-10-28).

The WSL2 authority and its guard tests are unchanged. The Ascend container is a second,
record-bound authority: a process inside a container with the record's hostname, on a host the
record names by machine and kernel release. Everything else is refused, and the dispatcher that
the gate and the tool-plan admission call names both authorities when it refuses.
"""

import json
import platform
import socket

import pytest

from speech_aware_evidence_acquisition.contracts import (
    ASCEND_CONTAINER_AUTHORITY_RELPATH,
    ExecutionPlan,
    ExecutionScopeError,
    FrozenCoreGate,
    assert_ascend_container_execution_environment,
    assert_formal_execution_authority,
)
from speech_aware_evidence_acquisition.contracts import plan as plan_module

_HOSTNAME = "saea-ascend-authority"
_KERNEL = "5.15.0-25-generic"


def _plan():
    return ExecutionPlan(
        run_id="SAEA-E-000-smoke",
        execution_profile="baseline-reproduction",
        carrier_lock_key="earnings21-original",
        split_role="dev",
        split_identity_hash="a" * 64,
        planned_model_calls=10,
        planned_gpu_hours=1.0,
        planned_speech_audio_seconds=600,
        protocol_hash="b" * 64,
    )


def _record(tmp_path, **overrides):
    document = {
        "schema": "saea-ascend-container-authority-v1",
        "dated_owner_decision": "decision 9.1, 2026-10-28",
        "owner_words": "authorized",
        "scope": "the NPU port",
        "hosts": [{"name": "host-a", "machine": "aarch64", "kernel_release": _KERNEL}],
        "container_hostname": _HOSTNAME,
        "image": "saea-base:20261027",
    }
    document.update(overrides)
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _facts(**overrides):
    facts = dict(os_name="posix", machine="aarch64", kernel_release=_KERNEL, hostname=_HOSTNAME, in_container=True)
    facts.update(overrides)
    return facts


def test_the_committed_record_is_well_formed():
    document = plan_module._load_ascend_container_authority(plan_module._STUDY_ROOT / ASCEND_CONTAINER_AUTHORITY_RELPATH)
    assert document["container_hostname"] == _HOSTNAME
    assert all(host["machine"] == "aarch64" for host in document["hosts"])


def test_accepts_the_authorized_container(tmp_path):
    assert assert_ascend_container_execution_environment(authority_path=_record(tmp_path), **_facts()) == (
        "ascend-container"
    )


@pytest.mark.parametrize(
    ("facts", "message"),
    [
        (_facts(in_container=False), "inside the authorized container"),
        (_facts(os_name="nt"), "inside the authorized container"),
        (_facts(hostname="3f2c9a1b7d4e"), "container hostname"),
        (_facts(kernel_release="6.6.0-132.0.0.111.oe2403sp3.aarch64"), "does not name host"),
        (_facts(kernel_release="6.6.87.2-microsoft-standard-WSL2"), "does not name host"),
        (_facts(machine="x86_64"), "does not name host"),
    ],
)
def test_refuses_anything_but_the_authorized_container(tmp_path, facts, message):
    with pytest.raises(ExecutionScopeError, match=message):
        assert_ascend_container_execution_environment(authority_path=_record(tmp_path), **facts)


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "saea-ascend-container-authority-v0"},
        {"owner_words": " "},
        {"hosts": []},
        {"hosts": [{"name": "host-a", "machine": "aarch64"}]},
        {"container_hostname": ""},
        {"extra": "key"},
    ],
)
def test_refuses_a_malformed_record(tmp_path, overrides):
    with pytest.raises(ExecutionScopeError, match="authority record"):
        assert_ascend_container_execution_environment(authority_path=_record(tmp_path, **overrides), **_facts())


def test_refuses_a_missing_record(tmp_path):
    with pytest.raises(ExecutionScopeError, match="missing or unreadable"):
        assert_ascend_container_execution_environment(authority_path=tmp_path / "absent.json", **_facts())


def _refuse(message):
    def check():
        raise ExecutionScopeError(message)

    return check


def test_dispatcher_prefers_wsl2_and_never_asks_the_container_check():
    assert assert_formal_execution_authority(wsl=lambda: "wsl2", ascend=_refuse("unreachable")) == "wsl2"


def test_dispatcher_falls_back_to_the_ascend_container():
    assert (
        assert_formal_execution_authority(wsl=_refuse("WSL2"), ascend=lambda: "ascend-container")
        == "ascend-container"
    )


def test_dispatcher_refusal_names_both_authorities():
    with pytest.raises(ExecutionScopeError, match="WSL2 refused; Ascend refused"):
        assert_formal_execution_authority(wsl=_refuse("WSL2 refused"), ascend=_refuse("Ascend refused"))


def _authorized_container(monkeypatch, tmp_path):
    marker = tmp_path / "dockerenv"
    marker.write_text("", encoding="utf-8")
    monkeypatch.setattr(plan_module, "_CONTAINER_MARKER", marker)
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(platform, "release", lambda: _KERNEL)
    monkeypatch.setattr(socket, "gethostname", lambda: _HOSTNAME)


def _gate(tmp_path):
    return FrozenCoreGate(
        e0_receipt=tmp_path / "missing-e0.json",
        runtime_receipt=tmp_path / "missing-runtime.json",
        data_root=tmp_path,
        lock_path=tmp_path / "missing-lock.json",
        runtime_root=tmp_path,
        exposure_ledger=tmp_path / "missing-ledger.md",
        source_root=tmp_path,
    )


def test_gate_admits_the_authorized_container_and_then_reads_its_evidence(monkeypatch, tmp_path):
    """MUTATION PROBE. Revert the gate to the WSL2-only call and this refuses with the authority error."""

    _authorized_container(monkeypatch, tmp_path)
    with pytest.raises(Exception) as refused:
        _gate(tmp_path).assert_model_touch_allowed(_plan())
    assert "authority" not in str(refused.value)


def test_gate_still_refuses_a_container_under_another_hostname(monkeypatch, tmp_path):
    _authorized_container(monkeypatch, tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: "3f2c9a1b7d4e")
    with pytest.raises(ExecutionScopeError, match="WSL2 compute authority.*container hostname"):
        _gate(tmp_path).assert_model_touch_allowed(_plan())


def test_tool_plan_admission_admits_the_authorized_container(monkeypatch, tmp_path):
    """MUTATION PROBE. Revert the admission to the WSL2-only call and this refuses with the authority error."""

    from speech_aware_evidence_acquisition.contracts.tool_plan import assert_registered_tool_plan

    _authorized_container(monkeypatch, tmp_path)
    with pytest.raises(ExecutionScopeError, match="exact ExecutionPlan contract type"):
        assert_registered_tool_plan(
            source_root=tmp_path,
            exposure_ledger=tmp_path / "missing-ledger.md",
            umbrella_lock=tmp_path / "missing-lock.json",
            umbrella_lock_sha256="0" * 64,
            protocol_path=tmp_path / "missing-protocol.json",
            protocol_sha256="0" * 64,
            plan="not a plan",
        )
