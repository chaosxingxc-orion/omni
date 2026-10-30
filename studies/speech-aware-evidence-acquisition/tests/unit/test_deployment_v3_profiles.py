"""``-v3`` deployment profiles and the multimodal-cache-transport flag (S16b, plan 3.11 ruling 3).

``--mm-processor-cache-type`` (vLLM 0.30.0; ``lru``/``shm``) is a cache-transport knob that by source changes no
logits. It is a DEPLOYMENT flag: it comes from the profile, is recorded in ``deployment_record`` and the receipt's
deployment layer, and never enters ``protocol_sha256``. A protocol fragment must not carry it, and any other ``--mm-*``
flag is refused by name until it is classified on purpose.
"""

from __future__ import annotations

import copy
import json
import shlex
from pathlib import Path

import pytest

from _vllm_world import VllmWorld
from speech_aware_evidence_acquisition.contracts import GateClosed
from speech_aware_evidence_acquisition.contracts.gate_runtime_vllm import (
    assert_argv_matches_instance,
    classify_server_flags,
    protocol_sha256,
    split_vllm_server_argv,
)
from speech_aware_evidence_acquisition.core import deployment as dep
from speech_aware_evidence_acquisition.core.config import fragment_hash
from speech_aware_evidence_acquisition.core.session.serving import (
    ServingSessionError,
    declared_server_geometry,
    merge_server_args,
)
from speech_aware_evidence_acquisition.rails.adapters.runtime_view import RuntimeViewError, configured_geometry

NIC = "enp61s0f2"
FLAG = "--mm-processor-cache-type"
ARMS = ("4xtp2", "2xtp4")
V1, V2, V3 = (lambda arm, n=n: f"qwen3-omni-thinking-bf16-181-{arm}-v{n}" for n in (1, 2, 3))
_VLLM_ARGS = ["--max-model-len", "24576", "--reasoning-parser", "qwen3"]
OTHER_MM_FLAGS = ["--mm-processor-cache-gb", "--disable-mm-preprocessor-cache", "--mm-encoder-tp-mode",
                  "--mm-processor-kwargs", "--disable-mm-processor-cache"]


def _document(profile_id: str) -> dict:
    return json.loads(dep.resolve_profile_path(profile_id).read_text(encoding="utf-8"))


def _deployment_half(argv: list[str]):
    """``(served, protocol tokens, deployment flags)`` of an ``instance_server_argv`` (``serve <model>`` stripped)."""

    return classify_server_flags(argv[2:])


# -- the profiles ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
def test_a_v3_profile_differs_from_v2_only_in_identity_basis_and_the_new_field(arm):
    old, new = _document(V2(arm)), _document(V3(arm))
    assert {key for key in set(old) | set(new) if old.get(key) != new.get(key)} == {
        "profile_id", "basis", "mm_processor_cache_type"
    }
    assert new["profile_id"] == V3(arm) and new["mm_processor_cache_type"] == "shm"
    assert "mm_processor_cache_type" not in old and "mm_processor_cache_type" not in _document(V1(arm))
    assert "3.11 ruling 3" in new["basis"]
    assert "/mnt/cephfs/xingchao/omni/work/scratch/badcase/2026-10-29-e0-mm-encoder-preemption-source.md" in new["basis"]
    assert dep.resolve_profile_path(V3(arm)).read_text(encoding="utf-8") == json.dumps(new, indent=2, sort_keys=True) + "\n"
    profile = dep.load_deployment_profile(V3(arm))
    assert profile.mm_processor_cache_type == "shm"
    assert dep.load_deployment_profile(V2(arm)).mm_processor_cache_type is None
    assert dep.load_deployment_profile(V1(arm)).mm_processor_cache_type is None


@pytest.mark.parametrize("value", ["lru", "shm", None])
def test_the_field_is_optional_and_takes_only_lru_or_shm(value):
    document = _document(V2("4xtp2"))
    if value is not None:
        document["mm_processor_cache_type"] = value
    assert dep.validate_deployment_profile(document).mm_processor_cache_type == value


@pytest.mark.parametrize("bad", ["", "LRU", "none", "disk", 1, True, ["shm"]])
def test_any_other_value_is_refused(bad):
    document = _document(V2("4xtp2"))
    document["mm_processor_cache_type"] = bad
    with pytest.raises(dep.DeploymentError, match="mm_processor_cache_type"):
        dep.validate_deployment_profile(document)


# -- hashes unchanged, argv in the deployment half -------------------------------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
def test_the_flag_moves_no_protocol_hash_and_lives_in_the_deployment_half(arm):
    v2, v3 = dep.load_deployment_profile(V2(arm)), dep.load_deployment_profile(V3(arm))
    assert v2.sha256 != v3.sha256  # the profile's own hash covers its identity and the recorded knob
    f2, f3 = dep.model_fragment_values(v2), dep.model_fragment_values(v3)
    assert fragment_hash(f2) == fragment_hash(f3)
    assert "--mm-" not in " ".join(map(str, f3["server_args"])).replace("--limit-mm", "")
    for old, new in zip(v2.instances, v3.instances):
        argv2, argv3 = dep.instance_server_argv(v2, old, f2), dep.instance_server_argv(v3, new, f3)
        assert argv2.count(FLAG) == 0 and argv3.count(FLAG) == 1
        assert argv3[argv3.index(FLAG) + 1] == "shm"
        assert sorted(argv3) == sorted([*argv2, FLAG, "shm"])  # the one added group, nothing else moved
        s2, protocol2, deployment2 = _deployment_half(argv2)
        s3, protocol3, deployment3 = _deployment_half(argv3)
        assert s2 == s3 and list(protocol2) == list(protocol3)  # protocol argv unchanged
        assert deployment3 == {**deployment2, FLAG: "shm"} and FLAG not in deployment2
        assert FLAG not in protocol3
    # protocol_sha256 is blind to it: it hashes everything but the deployment layer
    receipt = {
        "schema": "x",
        "protocol": {"served_model_name": f2["served_model_name"], "server_args": list(f2["server_args"])},
        "deployment": {"instances": [{"max_num_seqs": v2.max_num_seqs}]},
    }
    moved = copy.deepcopy(receipt)
    moved["deployment"]["instances"][0]["mm_processor_cache_type"] = "shm"
    assert protocol_sha256(moved) == protocol_sha256(receipt)


def test_the_emitted_argv_parses_through_the_gate_with_the_flag_in_the_deployment_half():
    profile = dep.load_deployment_profile(V3("4xtp2"))
    fragment = dep.model_fragment_values(profile)
    argv = dep.instance_server_argv(profile, profile.instances[0], fragment)
    parsed = split_vllm_server_argv(["/usr/bin/python3", "-m", "vllm.entrypoints.cli.main", *argv])
    assert parsed.deployment[FLAG] == "shm"
    assert FLAG not in parsed.protocol_args
    assert list(parsed.protocol_args) == list(_deployment_half(dep.instance_server_argv(
        dep.load_deployment_profile(V2("4xtp2")), profile.instances[0], fragment))[1])


def test_the_deployment_record_names_the_flag_and_pins_the_argv():
    for arm in ARMS:
        record = dep.deployment_record(dep.load_deployment_profile(V3(arm)), nic=NIC)
        old = dep.deployment_record(dep.load_deployment_profile(V2(arm)), nic=NIC)
        assert record["hashed"] is False and record["mm_processor_cache_type"] == "shm"
        assert old["mm_processor_cache_type"] is None
        assert record["protocol_window_tokens"] == old["protocol_window_tokens"] == 24576
        assert all(a["server_argv_sha256"] != b["server_argv_sha256"] for a, b in zip(record["instances"], old["instances"]))


def test_plan_emits_the_flag_on_every_v3_instance_and_nothing_on_v1_v2(capsys):
    for arm, count in (("4xtp2", 4), ("2xtp4", 2)):
        assert dep._main(["plan", V3(arm), "--nic", NIC]) == 0
        commands = [shlex.split(line) for line in capsys.readouterr().out.splitlines() if line.startswith("docker run")]
        assert len(commands) == count
        for command in commands:
            assert command.count(FLAG) == 1 and command[command.index(FLAG) + 1] == "shm"
        for older in (V1(arm), V2(arm)):
            assert dep._main(["plan", older, "--nic", NIC]) == 0
            out = capsys.readouterr().out
            assert FLAG not in out
            assert len([line for line in out.splitlines() if line.startswith("docker run")]) == count


# -- the refusals ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", [[FLAG, "shm"], [FLAG, "lru"], [f"{FLAG}=shm"]])
def test_the_flag_inside_a_protocol_fragment_is_refused_everywhere(spelling):
    args = [*_VLLM_ARGS, *spelling]
    with pytest.raises(ServingSessionError, match=FLAG):
        merge_server_args(args, slots=4, context_tokens=4 * 24576, transport="vllm")
    with pytest.raises(ServingSessionError, match=FLAG):
        declared_server_geometry(args, transport="vllm")
    values = {"transport": "vllm", "server_args": args, "serving_slots": 4, "serving_slot_tokens": 24576}
    with pytest.raises(RuntimeViewError, match=FLAG):
        configured_geometry(values, defaults=(8, 8192, 64))
    profile = dep.load_deployment_profile(V3("4xtp2"))
    fragment = dep.model_fragment_values(profile)
    with pytest.raises(dep.DeploymentError, match="deployment flags"):
        dep.instance_server_argv(profile, profile.instances[0], dict(fragment, server_args=[*fragment["server_args"], FLAG, "shm"]))
    # the llama dialect is untouched: the tokens are opaque server flags there
    assert merge_server_args(["-np", "1", "-c", "8", FLAG, "shm"], slots=2, context_tokens=16) == [
        "-np", "2", "-c", "16", FLAG, "shm"
    ]


@pytest.mark.parametrize("flag", OTHER_MM_FLAGS)
def test_any_other_mm_flag_is_refused_by_name(flag):
    for tail in ([flag, "1"], [flag], [f"{flag}=1"]):
        args = [*_VLLM_ARGS, *tail]
        with pytest.raises(ServingSessionError, match=flag):
            merge_server_args(args, slots=4, context_tokens=4 * 24576, transport="vllm")
        with pytest.raises(ServingSessionError, match=flag):
            declared_server_geometry(args, transport="vllm")
        values = {"transport": "vllm", "server_args": args, "serving_slots": 4, "serving_slot_tokens": 24576}
        with pytest.raises(RuntimeViewError, match=flag):
            configured_geometry(values, defaults=(8, 8192, 64))
    # in a receipt's argv / protocol args too (the gate's classifier)
    with pytest.raises(GateClosed, match=flag):
        classify_server_flags([*_VLLM_ARGS, flag, "1"])


def test_the_already_classified_mm_flag_is_still_accepted():
    args = [*_VLLM_ARGS, "--limit-mm-per-prompt", '{"audio": 1, "image": 0, "video": 0}']
    assert merge_server_args(args, slots=4, context_tokens=4 * 24576, transport="vllm")[-2] == "--limit-mm-per-prompt"
    assert declared_server_geometry(args, transport="vllm") == (None, 24576)
    served, protocol, deployment = classify_server_flags(args)
    assert deployment == {} and list(protocol) == args


def test_the_committed_fragment_still_composes_with_the_new_refusals():
    profile = dep.load_deployment_profile(V2("4xtp2"))
    fragment = dep.model_fragment_values(profile)
    dep.instance_server_argv(profile, profile.instances[0], fragment)


@pytest.mark.parametrize("bad", ["disk", "LRU", ""])
def test_the_gate_refuses_a_bad_cache_type_value(bad):
    with pytest.raises(GateClosed, match="lru"):
        classify_server_flags([FLAG, bad])


# -- the receipt gate ------------------------------------------------------------------------------------------------


def test_the_receipt_instance_records_the_flag_and_the_live_argv_must_agree(tmp_path):
    world = VllmWorld(tmp_path)
    receipt = world.receipt_with_instances(2)
    before = receipt["protocol_sha256"]
    instance = receipt["deployment"]["instances"][0]
    instance["mm_processor_cache_type"] = "shm"
    assert protocol_sha256(receipt) == before  # the deployment layer is outside the protocol hash
    argv = world.argv_for(instance["instance_id"], receipt)
    assert argv[argv.index(FLAG) + 1] == "shm"
    # a live argv without the flag, or with the other value, disagrees with the recorded fact
    for wrong in ([t for i, t in enumerate(argv) if t != FLAG and argv[i - 1] != FLAG],
                  [*argv[:argv.index(FLAG) + 1], "lru", *argv[argv.index(FLAG) + 2:]]):
        with pytest.raises(GateClosed, match="mm_processor_cache_type"):
            _assert_matches(world, wrong, receipt, instance)
    # and an instance without the field refuses a live argv that carries the flag
    plain = receipt["deployment"]["instances"][1]
    with pytest.raises(GateClosed, match="mm_processor_cache_type"):
        _assert_matches(world, [*world.argv_for(plain["instance_id"], receipt), FLAG, "shm"], receipt, plain)


def _assert_matches(world, argv, receipt, instance):
    return assert_argv_matches_instance(argv, receipt, instance)
