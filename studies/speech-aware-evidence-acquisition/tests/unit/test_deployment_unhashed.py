"""The deployment layer stays OUT of the protocol hash (owner decision 2026-10-28).

Protocol facts are hashed: the weights, the 24,576-token per-sequence window, the decoding parameters, the reasoning
budget. Deployment facts are recorded in the run manifest and the runtime receipt but never hashed: instance count,
tensor-parallel size, sequences per instance (``--max-num-seqs``), memory utilisation, card assignment. These tests
pin the split on the committed port fragment: the fragment holds no deployment knob; the vLLM argv builder takes
``--max-num-seqs`` from the selected profile; changing the profile moves no protocol hash; changing a protocol field
moves it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.contracts.gate_runtime_vllm import classify_server_flags, protocol_sha256
from speech_aware_evidence_acquisition.core import deployment as dep
from speech_aware_evidence_acquisition.core.config import compose, fragment_hash, protocol_hash, read_fragment

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = "qwen3-omni-thinking-bf16-181-4xtp2-v1"
SMOKE_EXPERIMENT = "slurp-native-stage3-l338-lever-column-fold-b-smoke-4-npu-port-v1"


def _profile(**changes):
    document = json.loads(dep.resolve_profile_path(PROFILE_ID).read_text(encoding="utf-8"))
    document.update(changes)
    return dep.validate_deployment_profile(document)


@pytest.fixture(scope="module")
def fragment():
    return dep.model_fragment_values(_profile())


def _composed_protocol_hash(root: Path) -> str:
    experiment = json.loads((root / "configs" / "experiment" / f"{SMOKE_EXPERIMENT}.json").read_text(encoding="utf-8"))
    cell = experiment["cells"][0]
    composed = compose(root, cell["model_fragment"], cell["dataset_fragment"], cell["baseline_fragment"], SMOKE_EXPERIMENT)
    return protocol_hash(composed.config_hash, composed.values["arm"], experiment["population"]["split_identity_hash"])


def _repo_with_model_fragment(tmp_path: Path, model_fragment: str, document: dict) -> Path:
    """A repo view whose configs are the real ones except ONE model fragment, so a protocol field can be changed."""

    configs = tmp_path / "configs"
    configs.mkdir(parents=True)
    for child in (REPO_ROOT / "configs").iterdir():
        if child.name != "model":
            (configs / child.name).symlink_to(child)
    (configs / "model").mkdir()
    (configs / "model" / f"{model_fragment}.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return tmp_path


# (a) the hashed fragment no longer contains the deployment knobs -------------------------------------------------------


def test_the_hashed_port_fragment_carries_no_deployment_knob(fragment):
    assert "profile_slot_count" not in fragment
    server_args = [str(token) for token in fragment["server_args"]]
    flags = {token.split("=", 1)[0] for token in server_args}
    assert not flags & dep.DEPLOYMENT_ONLY_FLAGS
    assert "--max-num-seqs" not in flags
    served, protocol, deployment = classify_server_flags(server_args)
    assert served is None and deployment == {} and list(protocol) == server_args
    # What stays protocol: the per-sequence window, the decoding and the reasoning budget.
    assert server_args[server_args.index("--max-model-len") + 1] == "24576"
    assert fragment["decoding_params"]["max_tokens"] == 4096 and fragment["reasoning_budget_tokens"] == 3072
    assert "--max-num-seqs 4" not in fragment["geometry_basis"] and "four" not in fragment["geometry_basis"]


# (b) the vLLM argv builder takes --max-num-seqs from the profile ------------------------------------------------------


@pytest.mark.parametrize("max_num_seqs", [8, 24, 64])
def test_the_argv_builder_takes_max_num_seqs_from_the_profile(fragment, max_num_seqs):
    profile = _profile(max_num_seqs=max_num_seqs)
    argv = dep.instance_server_argv(profile, profile.instances[1], fragment)
    assert argv.count("--max-num-seqs") == 1
    assert argv[argv.index("--max-num-seqs") + 1] == str(max_num_seqs)
    _served, protocol, deployment = classify_server_flags(argv[2:])  # argv is ["serve", <model>, *flags]
    assert deployment["--max-num-seqs"] == str(max_num_seqs)
    # The protocol half of the argv is the fragment's own flags in order, whatever the profile says.
    assert list(protocol) == [str(token) for token in fragment["server_args"]]


def test_the_recorded_deployment_names_the_profiles_max_num_seqs(fragment):
    for max_num_seqs in (16, 64):
        record = dep.deployment_record(_profile(max_num_seqs=max_num_seqs))
        assert record["hashed"] is False and record["max_num_seqs_per_instance"] == max_num_seqs
        assert record["protocol_window_tokens"] == 24576


# (c) changing the profile's max-num-seqs does NOT change a protocol hash ----------------------------------------------


def test_changing_the_profile_max_num_seqs_moves_no_protocol_hash(fragment):
    first, second = _profile(max_num_seqs=64), _profile(max_num_seqs=16)
    assert first.sha256 != second.sha256  # the profile's own hash covers the recorded knob ...
    composed_before = _composed_protocol_hash(REPO_ROOT)
    before = dep.deployment_record(first)
    after = dep.deployment_record(second)
    assert [i["server_argv_sha256"] for i in before["instances"]] != [i["server_argv_sha256"] for i in after["instances"]]
    # ... but the hashed fragment, the composed protocol hash and the receipt's protocol hash do not move.
    name = first.model_fragment
    assert fragment_hash(read_fragment(REPO_ROOT, "model", name)) == fragment_hash(fragment)
    assert _composed_protocol_hash(REPO_ROOT) == composed_before
    receipt = {
        "schema": "x",
        "protocol": {"served_model_name": fragment["served_model_name"], "server_args": list(fragment["server_args"])},
        "deployment": {"instances": [{"max_num_seqs": 64}]},
    }
    moved = copy.deepcopy(receipt)
    moved["deployment"]["instances"][0]["max_num_seqs"] = 16
    assert protocol_sha256(moved) == protocol_sha256(receipt)


# (d) changing a protocol field DOES -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["server_args"].__setitem__(d["server_args"].index("--max-model-len") + 1, "32768"),
        lambda d: d["decoding_params"].update(seed=1),
        lambda d: (d.update(reasoning_budget_tokens=2048), d["decoding_params"].update(max_tokens=3072)),
    ],
    ids=["window", "decoding", "reasoning-budget"],
)
def test_changing_a_protocol_field_moves_the_fragment_composed_and_receipt_hashes(tmp_path, fragment, mutate):
    name = _profile().model_fragment
    original = json.loads((REPO_ROOT / "configs" / "model" / f"{name}.json").read_text(encoding="utf-8"))
    changed = copy.deepcopy(original)
    mutate(changed)
    assert fragment_hash(changed) != fragment_hash(original)
    same = _repo_with_model_fragment(tmp_path / "same", name, original)
    other = _repo_with_model_fragment(tmp_path / "other", name, changed)
    assert _composed_protocol_hash(other) != _composed_protocol_hash(same)
    receipt = {"protocol": {"server_args": list(original["server_args"])}, "deployment": {"instances": []}}
    moved = {"protocol": {"server_args": list(changed["server_args"])}, "deployment": {"instances": []}}
    if changed["server_args"] != original["server_args"]:
        assert protocol_sha256(moved) != protocol_sha256(receipt)


def test_mm_processor_cache_type_moves_no_protocol_hash_and_is_recorded(fragment):
    first, second = _profile(), _profile(mm_processor_cache_type="shm")
    assert fragment_hash(dep.model_fragment_values(first)) == fragment_hash(dep.model_fragment_values(second))
    argv_first = dep.instance_server_argv(first, first.instances[0], fragment)
    argv_second = dep.instance_server_argv(second, second.instances[0], fragment)
    _s, protocol_first, deployment_first = classify_server_flags(argv_first[2:])
    _s, protocol_second, deployment_second = classify_server_flags(argv_second[2:])
    assert list(protocol_first) == list(protocol_second)
    assert "--mm-processor-cache-type" not in deployment_first
    assert deployment_second["--mm-processor-cache-type"] == "shm"
    assert dep.deployment_record(first)["mm_processor_cache_type"] is None
    assert dep.deployment_record(second)["mm_processor_cache_type"] == "shm"
    receipt = {
        "schema": "x",
        "protocol": {"served_model_name": fragment["served_model_name"], "server_args": list(fragment["server_args"])},
        "deployment": {"instances": [{"max_num_seqs": 64}]},
    }
    moved = copy.deepcopy(receipt)
    moved["deployment"]["instances"][0]["mm_processor_cache_type"] = "shm"
    assert protocol_sha256(moved) == protocol_sha256(receipt)
