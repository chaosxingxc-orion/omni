"""DRAFT ``-v2`` deployment profiles (plan 3.9 decision 4, Q-B): the resident scheduling ceiling moves 64 -> 32 for the
4 x TP2 arm; the 2 x TP4 arm carries a placeholder ceiling (TODO-PLANNER). The ceiling is a DEPLOYMENT fact, so it must
move no hashed byte: ``protocol_sha256``, the fragment hash and the protocol half of the argv stay equal to the ``-v1``
profile's, while the emitted ``docker run`` argv carries the new ``--max-num-seqs``.
"""

from __future__ import annotations

import copy
import json
import shlex
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.contracts.gate_runtime_vllm import classify_server_flags, protocol_sha256
from speech_aware_evidence_acquisition.core import deployment as dep
from speech_aware_evidence_acquisition.core.config import fragment_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
NIC = "enp61s0f2"
PAIRS = {
    "qwen3-omni-thinking-bf16-181-4xtp2-v1": ("qwen3-omni-thinking-bf16-181-4xtp2-v2", 32),
    "qwen3-omni-thinking-bf16-181-2xtp4-v1": ("qwen3-omni-thinking-bf16-181-2xtp4-v2", 64),
}


def _document(profile_id: str) -> dict:
    return json.loads(dep.resolve_profile_path(profile_id).read_text(encoding="utf-8"))


@pytest.mark.parametrize("v1", sorted(PAIRS))
def test_a_v2_profile_validates_is_named_by_its_file_and_differs_only_in_identity_basis_and_ceiling(v1):
    v2, ceiling = PAIRS[v1]
    profile = dep.load_deployment_profile(v2)
    assert profile.profile_id == v2 and profile.max_num_seqs == ceiling
    old, new = _document(v1), _document(v2)
    # admitted_level (Gap C, Decision 2) and namespace_policy (Decision 3, owner 2026-10-30) are OPTIONAL deployment
    # fields added to the 4xTP2 -v2 only; outside every hash
    assert {key for key in set(old) | set(new) if old.get(key) != new.get(key)} <= {
        "profile_id", "basis", "max_num_seqs", "admitted_level", "namespace_policy"}
    assert new["profile_id"] == v2 and new["max_num_seqs"] == ceiling
    # cards, cpusets, ports, env and the memory budget are untouched: the same deployment, a different scheduling ceiling
    for key in ("instances", "gpu_memory_utilization", "max_num_batched_tokens", "tensor_parallel_size", "shm_size"):
        assert old[key] == new[key]
    path = dep.resolve_profile_path(v2)
    assert path.read_text(encoding="utf-8") == json.dumps(new, indent=2, sort_keys=True) + "\n"


def test_the_draft_status_is_stated_in_the_basis():
    assert "DRAFT" in _document("qwen3-omni-thinking-bf16-181-4xtp2-v2")["basis"]
    assert "Q-B" in _document("qwen3-omni-thinking-bf16-181-4xtp2-v2")["basis"]
    assert "TODO-PLANNER" in _document("qwen3-omni-thinking-bf16-181-2xtp4-v2")["basis"]


@pytest.mark.parametrize("v1", sorted(PAIRS))
def test_the_ceiling_moves_no_protocol_hash(v1):
    v2, ceiling = PAIRS[v1]
    first, second = dep.load_deployment_profile(v1), dep.load_deployment_profile(v2)
    assert first.sha256 != second.sha256  # the profile's own hash covers the recorded knob and its identity
    assert first.model_fragment == second.model_fragment
    fragment_before, fragment_after = dep.model_fragment_values(first), dep.model_fragment_values(second)
    assert fragment_hash(fragment_before) == fragment_hash(fragment_after)
    # the protocol half of the emitted argv is identical; only the deployment half carries the ceiling
    argv_before = dep.instance_server_argv(first, first.instances[0], fragment_before)
    argv_after = dep.instance_server_argv(second, second.instances[0], fragment_after)
    _s1, protocol_before, deployment_before = classify_server_flags(argv_before[2:])
    _s2, protocol_after, deployment_after = classify_server_flags(argv_after[2:])
    assert list(protocol_before) == list(protocol_after)
    assert deployment_before["--max-num-seqs"] == str(first.max_num_seqs)
    assert deployment_after["--max-num-seqs"] == str(ceiling)
    # and the receipt's protocol_sha256 is blind to the ceiling in the deployment half
    receipt = {
        "schema": "x",
        "protocol": {"served_model_name": fragment_before["served_model_name"],
                     "server_args": list(fragment_before["server_args"])},
        "deployment": {"instances": [{"max_num_seqs": first.max_num_seqs}]},
    }
    moved = copy.deepcopy(receipt)
    moved["deployment"]["instances"][0]["max_num_seqs"] = ceiling
    assert protocol_sha256(moved) == protocol_sha256(receipt)


def test_plan_emits_max_num_seqs_32_on_every_instance_of_the_4xtp2_v2_profile(capsys):
    assert dep._main(["plan", "qwen3-omni-thinking-bf16-181-4xtp2-v2", "--nic", NIC]) == 0
    out = capsys.readouterr().out
    assert "max-num-seqs 32/instance" in out
    commands = [shlex.split(line) for line in out.splitlines() if line.startswith("docker run")]
    assert len(commands) == 4
    for command in commands:
        assert command.count("--max-num-seqs") == 1
        assert command[command.index("--max-num-seqs") + 1] == "32"
        assert f"HCCL_SOCKET_IFNAME={NIC}" in command
    # the v1 profile is untouched
    assert dep._main(["plan", "qwen3-omni-thinking-bf16-181-4xtp2-v1", "--nic", NIC]) == 0
    v1_out = capsys.readouterr().out
    assert all(shlex.split(line)[shlex.split(line).index("--max-num-seqs") + 1] == "64"
               for line in v1_out.splitlines() if line.startswith("docker run"))


def test_plan_without_a_nic_is_still_refused_for_the_v2_profiles():
    for profile_id, _ceiling in PAIRS.values():
        with pytest.raises(dep.DeploymentError):
            dep._main(["plan", profile_id])


def test_the_deployment_record_names_the_ceiling():
    record = dep.deployment_record(dep.load_deployment_profile("qwen3-omni-thinking-bf16-181-4xtp2-v2"), nic=NIC)
    assert record["hashed"] is False and record["max_num_seqs_per_instance"] == 32
    assert record["protocol_window_tokens"] == 24576


def test_admitted_level_is_a_deployment_field_that_moves_no_protocol_hash():
    v1, v2 = dep.load_deployment_profile("qwen3-omni-thinking-bf16-181-4xtp2-v1"), dep.load_deployment_profile(
        "qwen3-omni-thinking-bf16-181-4xtp2-v2")
    assert _document(v2.profile_id)["admitted_level"] == 32 == v2.max_num_seqs
    assert "admitted_level" not in _document(v1.profile_id)
    fragment_before, fragment_after = dep.model_fragment_values(v1), dep.model_fragment_values(v2)
    assert fragment_hash(fragment_before) == fragment_hash(fragment_after)
    before = classify_server_flags(dep.instance_server_argv(v1, v1.instances[0], fragment_before)[2:])
    after = classify_server_flags(dep.instance_server_argv(v2, v2.instances[0], fragment_after)[2:])
    assert list(before[1]) == list(after[1])
    assert "admitted_level" not in " ".join(dep.instance_server_argv(v2, v2.instances[0], fragment_after))
