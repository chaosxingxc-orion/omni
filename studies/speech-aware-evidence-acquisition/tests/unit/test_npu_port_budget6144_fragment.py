"""The thinking-budget successor of the NPU-port model fragment (plan 2026-10-30, section 2 M3 and section 8 R1).

``...-budget6144-max7168-port-v1`` is the ``...-budget3072-max4096-port-v1`` fragment with the thinking budget raised
from 3,072 to 6,144 tokens. These checks keep the successor honest and card-free:

* it differs from the current fragment ONLY in the budget-related fields (listed below), everything else byte-equal;
* its fragment hash differs, and the composed config hash and protocol hash of every NPU-port cell move when the cell
  names it (composed through the same ``core.config.compose`` the registration writer uses, in a temporary root, so no
  package file is written);
* the context-window arithmetic: the 24,576-token window holds the measured mean prompt plus the new cap, and the
  registered context-fit gate (slot - 3,072 reserve - cap - 2) is stated numerically so the fit consequence of the
  raised cap (the 55-item supply bound no longer fits) cannot be lost.

The current fragment, the cells and the writer's fragment reference are NOT changed by this file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.core.config import compose, fragment_hash, protocol_hash, read_fragment
from speech_aware_evidence_acquisition.rails.supply_variants import PerceptionBranchError, parse_perception_sampling_variants

REPO_ROOT = Path(__file__).resolve().parents[2]
OLD = "bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1"
NEW = "bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget6144-max7168-port-v1"
EXPERIMENTS = (
    "slurp-native-stage3-l338-lever-column-fold-b-smoke-4-npu-port-v1",
    "slurp-native-stage3-l338-lever-column-fold-b-npu-port-v1",
    "slurp-native-stage3-l338-lever-column-fold-b-comparator-npu-port-v1",
)

#: Every top-level field the successor may change, and why (all derive from the one decision "thinking budget 6,144").
BUDGET_FIELDS = {
    "reasoning_budget_tokens": "the budget itself, 3,072 -> 6,144",
    "decoding_params": "only max_tokens, 4,096 -> 7,168 (= 6,144 + the unchanged 1,024 answer reserve)",
    "timeout_seconds": "420 s x 7,168 / 4,096 = 735 s: the same ~9.8 tok/s floor at the larger cap",
    "geometry_basis": "prose: the 3,072 + 1,024 = 4,096 sentence and the lineage of the successor",
}
SLOT_TOKENS, OPERATIONAL_RESERVE, SPECIAL_RESERVE = 24576, 3072, 2  # rails.context_budget / thinking_system_family
MEAN_PROMPT = 8500
OBSERVED_SOLVER_PROMPT_MAX = {"268b lock": 13465, "five-arm column": 13918}  # l338_cost_inputs.json, sealed Thinking traces


def _fragment(name: str) -> dict:
    return json.loads((REPO_ROOT / "configs" / "model" / f"{name}.json").read_text(encoding="utf-8"))


def test_successor_differs_only_in_the_budget_fields():
    old, new = _fragment(OLD), _fragment(NEW)
    assert list(old) == list(new)  # same keys, same order
    changed = {key for key in old if old[key] != new[key]}
    assert changed == set(BUDGET_FIELDS)
    assert {key for key in old["decoding_params"] if old["decoding_params"][key] != new["decoding_params"][key]} == {"max_tokens"}
    assert (old["reasoning_budget_tokens"], new["reasoning_budget_tokens"]) == (3072, 6144)
    assert old["answer_reserve_tokens"] == new["answer_reserve_tokens"] == 1024
    assert (old["decoding_params"]["max_tokens"], new["decoding_params"]["max_tokens"]) == (4096, 7168)
    assert (old["timeout_seconds"], new["timeout_seconds"]) == (420, 735) and new["timeout_seconds"] == 420 * 7168 // 4096
    # the knob invariant the composer enforces
    assert new["decoding_params"]["max_tokens"] == new["reasoning_budget_tokens"] + new["answer_reserve_tokens"]
    # the serving geometry, the forced-close message and the prompt layout are untouched
    for key in ("server_args", "forced_close_message_sha256", "prompt_order", "enable_thinking", "reasoning_context_policy",
                "reasoning_content_guard", "model_lock_key", "runtime_receipt", "transport", "served_model_name"):
        assert new[key] == old[key], key
    assert new["server_args"][new["server_args"].index("--max-model-len") + 1] == "24576"
    assert re.search(r"6144 \+ 1024 = 7168", new["geometry_basis"]) and "budget3072-max4096-v1" in new["geometry_basis"]


def test_successor_fragment_hash_differs_and_is_pinned():
    old, new = fragment_hash(read_fragment(REPO_ROOT, "model", OLD)), fragment_hash(read_fragment(REPO_ROOT, "model", NEW))
    assert old == "a35b274058d7ac7558d3d0ac32cca8bb3c9321cb13ea89f16b38003fdd18c720"  # the true port fragment hash
    assert new != old
    assert new == "062d169819163123709fe76204a261baa309141fae71be314e69a5eee892965e"


def _stage_root(root: Path, experiment: str, model: str) -> tuple[Path, dict]:
    """A temporary config root holding ONE cell's four fragments with the experiment naming ``model``; nothing in the repo is written."""
    document = json.loads((REPO_ROOT / "configs/experiment" / f"{experiment}.json").read_text(encoding="utf-8"))
    cell = document["cells"][0]
    for kind, name in (("model", model), ("dataset", cell["dataset_fragment"]), ("baseline", cell["baseline_fragment"])):
        (root / "configs" / kind).mkdir(parents=True, exist_ok=True)
        (root / "configs" / kind / f"{name}.json").write_text((REPO_ROOT / "configs" / kind / f"{name}.json").read_text(encoding="utf-8"), encoding="utf-8")
    text = (REPO_ROOT / "configs/experiment" / f"{experiment}.json").read_text(encoding="utf-8")
    text = text.replace(OLD, model).replace(NEW, model)  # every mention of the serving fragment follows the model under test
    (root / "configs" / "experiment").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "experiment" / f"{experiment}.json").write_text(text, encoding="utf-8")
    return root, document


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_composed_and_protocol_hashes_move_with_the_successor(tmp_path, experiment):
    hashes = {}
    for model in (OLD, NEW):
        root, document = _stage_root(tmp_path / model[-40:] / "root", experiment, model)
        composed = compose(root, model, document["cells"][0]["dataset_fragment"], document["cells"][0]["baseline_fragment"], experiment)
        assert composed.values["decoding_params"]["max_tokens"] == (4096 if model == OLD else 7168)
        hashes[model] = (composed.config_hash, protocol_hash(composed.config_hash, composed.values["arm"], document["population"]["split_identity_hash"]))
    assert hashes[OLD][0] != hashes[NEW][0] and hashes[OLD][1] != hashes[NEW][1]
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for pair in hashes.values() for value in pair)


def test_window_holds_the_mean_prompt_plus_the_new_cap_and_so_does_the_registered_gate():
    cap = _fragment(NEW)["decoding_params"]["max_tokens"]
    assert SLOT_TOKENS - (MEAN_PROMPT + cap) == 8908  # physical margin at the mean prompt
    gate_limit = SLOT_TOKENS - OPERATIONAL_RESERVE - cap - SPECIAL_RESERVE
    assert gate_limit == 14334  # the registered per-contact prompt limit at the 7,168 cap (17,406 at 4,096)
    assert gate_limit - MEAN_PROMPT == 5834
    # the sealed Thinking Fold B traces: every observed Solver prompt maximum fits the gate limit
    assert gate_limit - OBSERVED_SOLVER_PROMPT_MAX["268b lock"] == 869 and gate_limit - OBSERVED_SOLVER_PROMPT_MAX["five-arm column"] == 416
    assert SLOT_TOKENS - (OBSERVED_SOLVER_PROMPT_MAX["five-arm column"] + cap) == 3490  # physical margin at the largest observed prompt


def _lever_block() -> dict:
    baseline = json.loads((REPO_ROOT / "configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-smoke-4-baseline-raw-port-v1.json")
                          .read_text(encoding="utf-8"))

    def find(node):
        if isinstance(node, dict):
            if "sampling_variants" in node:
                return node["sampling_variants"]
            for value in node.values():
                found = find(value)
                if found:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = find(value)
                if found:
                    return found
        return None

    return find(baseline)


@pytest.mark.parametrize("cap,items,fits", [(4096, 55, True), (7168, 55, False), (7168, 52, False), (7168, 51, True), (8192, 46, True), (8192, 47, False)])
def test_registered_fit_check_refuses_the_55_item_bound_at_the_raised_cap(cap, items, fits):
    """Characterises the registration blocker of the successor: the lock's 55-item supply bound estimates 15,248 prompt tokens, above the
    14,334-token limit of the 7,168 cap; 51 items (14,308 tokens, 26 to spare) is the largest bound that fits. At an 8,192 cap it is 46."""
    block = json.loads(json.dumps(_lever_block()))
    block["fit_estimate"]["completion_ceiling"], block["max_supplied_items"] = cap, items
    if fits:
        parse_perception_sampling_variants(block, phase_id="perception")
    else:
        with pytest.raises(PerceptionBranchError, match="exceeds the .*-token limit"):
            parse_perception_sampling_variants(block, phase_id="perception")
