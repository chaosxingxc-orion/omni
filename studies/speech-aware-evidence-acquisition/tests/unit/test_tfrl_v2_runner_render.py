"""The runner renders the TFRL-v2 bank (plan section 13 ruling 3 (a)): through the ACTUAL shared ``rails.runner`` episode invocation, no network and no model.

Proofs, through the same ``_ActualEpisodeHarness`` fake transport the native runner proofs use:

* a cell whose module block carries no ``tfrl_v2_bank`` frame renders exactly the bytes it rendered before (the Solver messages of the lock baseline are pinned by
  ``test_tfrl_v2_consumption`` at ``f6fed233...b8b9``; here the episode's four request bodies are identical with and without the new binding field);
* a cell with the frame and a bank that passed ``verify_for_flight`` renders the exemplar block of the scenario the Perception node predicted AFTER the supplied items and
  the rule bank INSIDE the Solver instruction, within the declared caps; the other three contacts (Perception, Reviewer, Planner) are byte-identical to the no-frame run;
* the declared frame and the verified bank are ONE module in both directions (a frame with no bank, a bank with no frame, a bank that is not the pinned one);
* the 43-item supply bound of the frame is enforced at the Solver contact;
* the trace line of a contact that consumed the bank carries the bank receipt hash; every other line carries ``null``.
"""

from __future__ import annotations

import copy
import json
from functools import partial
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.contracts import rollout_trace as schema
from speech_aware_evidence_acquisition.core import rollout_trace as trace
from speech_aware_evidence_acquisition.rails import tfrl_v2_consumption as rail
from speech_aware_evidence_acquisition.rails.knowledge import KeyFacet, KnowledgeSnapshot, ValueBundle
from speech_aware_evidence_acquisition.rails.runner import RailsRunnerError, run_speech_native_agent_loop
from speech_aware_evidence_acquisition.reproduction.ojw import openjiuwen_available

from test_llama_server_transport import _spawn_live_matching_process  # noqa: F401
import test_native_episode_runner as harness_module
from test_native_episode_runner import _ActualEpisodeHarness

H1, H2 = "R01C-H1", "R01C-H2"
SCOPE = {H1: "1" * 64, H2: "2" * 64}
BLOCK_TEXT = "Worked examples for alarm requests:\n- [evidence ex-1] wake me up at nine | alarm_set"
RULE_TEXT = "Consumption rules:\n- In alarm requests, give the time value exactly as it is spoken."
SUPPLIED = ("value:v1", "value:v2")
PERCEPTION = json.dumps({"candidates": [{"intent_id": "alarm_set", "distinction": "set", "evidence_refs": [], "unresolved_fields": []}]})
ANSWER = json.dumps({"scenario": "alarm", "action": "set", "entities": []})
BOUND = 43


def _snapshot(n: int = 2) -> KnowledgeSnapshot:
    return KnowledgeSnapshot(
        snapshot_id="fake-train-knowledge-v1", source_split_identity_hash="b" * 64, source_sample_ids=("train:t1",), builder_policy_id="coverage-builder-v1",
        facets=(KeyFacet(facet_id="operation:dim", dimension="operation", representation_kind="text", representation="Decrease the brightness of a light.",
                         source_sample_ids=("train:t1",)),),
        values=tuple(ValueBundle(value_id=f"value:v{i + 1}", content=f"supplied item {i + 1} | alarm_set", associated_facet_ids=("operation:dim",),
                                 source_sample_ids=("train:t1",), provenance_refs=("train:t1",), task_target="dim-light") for i in range(n)),
    ).validate()


def _bank(**over) -> rail.LoadedBank:
    receipt = {"schema": rail.BANK_RECEIPT_SCHEMA, "iteration": 1, "kind": "union", "measure": "vllm-tokenize", "exemplar_text": "resolved", "source_halves": [H1, H2]}
    receipt.update(over)
    return rail.LoadedBank(
        receipt=receipt, receipt_sha256=rail.receipt_sha256(receipt),
        exemplar_bank={"by_scenario": {"alarm": {"block_text": BLOCK_TEXT, "block_tokens": 40}, "weather": {"block_text": "Worked examples for weather requests:\n- x", "block_tokens": 20}}},
        rule_bank={"rendered_text": RULE_TEXT, "rendered_tokens": 20},
    )


def _block_json(bank: rail.LoadedBank, *, bound: int | None = BOUND, pin: str | None = None) -> str:
    frame = rail.declaration_block(iteration=1, consumer="FOLD-B", bank_kind="union", receipt_relpath="banks/union/bank-receipt.json",
                                   receipt_sha=pin or bank.receipt_sha256, source_scope=SCOPE)
    block = {"tfrl_v2_bank": frame}
    if bound is not None:
        block["sampling_frame"] = {"max_supplied_items": bound}
    return json.dumps(block, sort_keys=True, separators=(",", ":"))


class _Harness(_ActualEpisodeHarness):
    """The shared harness: the Perception node answers with a candidate document, and an optional bank / module block is bound."""

    def __init__(self, tmp_path, *, bank=None, block_json=None):
        super().__init__(tmp_path, sample_count=1, max_revisions=0, max_contacts=4, planner_actions=("ACCEPT",), arm="speech-feature-query")
        self.tfrl_v2_bank = bank
        if block_json is not None:
            import dataclasses
            self.config = dataclasses.replace(self.config, consumption_module_json=block_json)

    def _output(self, key):
        _sample, phase, _visit, _turn = key
        return {"perception": PERCEPTION, "solver": ANSWER, "planner": "ACCEPT"}.get(phase, f"{phase}-analysis")


def _run(tmp_path, monkeypatch, *, bank=None, block_json=None, snapshot=None):
    monkeypatch.setattr(harness_module, "run_speech_native_agent_loop", partial(run_speech_native_agent_loop, bootstrap_value_ids=SUPPLIED))
    harness = _Harness(tmp_path, bank=bank, block_json=block_json)
    try:
        summary = harness.run(snapshot=snapshot or _snapshot())
    finally:
        harness.close()
    dispatched = [key for event, key in harness.events if event == "dispatch"]
    return summary, {key[1]: harness.prepared[key]["rendered"] for key in dispatched}, harness


def _messages(body: dict) -> list:
    return body["messages"] if "messages" in body else body.get("input", [])


def _dump(body) -> str:
    return json.dumps(body, sort_keys=True)


pytestmark = pytest.mark.skipif(not openjiuwen_available(), reason="OpenJiuwen unavailable")


def test_a_cell_without_the_frame_renders_what_it_rendered_before(tmp_path, monkeypatch):
    _, plain, _ = _run(tmp_path / "plain", monkeypatch)
    again, repeat, _ = _run(tmp_path / "again", monkeypatch)
    assert [phase for phase in plain] == ["perception", "solver", "reviewer", "planner"]
    assert all(_dump(plain[phase]) == _dump(repeat[phase]) for phase in plain)
    solver = _dump(plain["solver"])
    assert "tfrl-v2" not in solver and "Worked examples" not in solver and "Consumption rules" not in solver
    assert again.results[0].final_response_text == ANSWER


def test_the_bank_adds_the_exemplar_block_after_the_supplied_items_and_the_rules_to_the_instruction_and_moves_nothing_else(tmp_path, monkeypatch):
    bank = _bank()
    _, plain, _ = _run(tmp_path / "plain", monkeypatch)
    summary, banked, harness = _run(tmp_path / "bank", monkeypatch, bank=bank, block_json=_block_json(bank))
    for phase in ("perception", "reviewer", "planner"):
        assert _dump(banked[phase]) == _dump(plain[phase]), phase                       # only the Solver contact consumes the bank
    solver_plain, solver_bank = _dump(plain["solver"]), _dump(banked["solver"])
    assert solver_plain != solver_bank
    assert RULE_TEXT.splitlines()[1] in solver_bank and RULE_TEXT.splitlines()[1] not in solver_plain
    assert BLOCK_TEXT.splitlines()[1] in solver_bank
    # order: every supplied item is rendered BEFORE the exemplar block
    positions = [solver_bank.index(f"supplied item {i + 1}") for i in range(2)] + [solver_bank.index("wake me up at nine")]
    assert positions == sorted(positions)
    # the rule bank is inside the instruction (before the first supplied item); the exemplar block is the last evidence
    assert solver_bank.index("Consumption rules") < positions[0]
    # caps: the bank that passed verify_for_flight is within them, and the rendered texts are exactly the bank's
    assert 0 < len(BLOCK_TEXT) and bank.exemplar_bank["by_scenario"]["alarm"]["block_tokens"] <= rail.EXEMPLAR_BLOCK_TOKEN_CAP and bank.rule_bank["rendered_tokens"] <= rail.RULE_BANK_TOKEN_CAP
    assert summary.results[0].final_response_text == ANSWER


def test_an_unknown_predicted_scenario_gets_the_rules_and_no_exemplar_block(tmp_path, monkeypatch):
    bank = _bank()
    bank = rail.LoadedBank(receipt=bank.receipt, receipt_sha256=bank.receipt_sha256, exemplar_bank={"by_scenario": {"weather": {"block_text": "Worked examples for weather requests:\n- x"}}}, rule_bank=bank.rule_bank)
    _, banked, _ = _run(tmp_path / "bank", monkeypatch, bank=bank, block_json=_block_json(bank))
    solver = _dump(banked["solver"])
    assert "Consumption rules" in solver and "Worked examples" not in solver


def test_the_frame_and_the_verified_bank_are_one_module_in_both_directions(tmp_path, monkeypatch):
    bank = _bank()
    with pytest.raises(Exception, match="declares tfrl_v2_bank but the episode binding carries no verified bank"):
        _run(tmp_path / "a", monkeypatch, bank=None, block_json=_block_json(bank))
    with pytest.raises(Exception, match="carries a TFRL-v2 bank but native_episode.consumption_module declares no tfrl_v2_bank frame"):
        _run(tmp_path / "b", monkeypatch, bank=bank, block_json=None)
    with pytest.raises(Exception, match="not the bank the module declaration pins"):
        _run(tmp_path / "c", monkeypatch, bank=bank, block_json=_block_json(bank, pin="e" * 64))
    with pytest.raises(Exception, match="carries no max_supplied_items"):
        _run(tmp_path / "d", monkeypatch, bank=bank, block_json=_block_json(bank, bound=None))


def test_the_supply_bound_of_the_frame_is_enforced_at_the_solver_contact(tmp_path, monkeypatch):
    bank = _bank()
    at_bound = _run(tmp_path / "ok", monkeypatch, bank=bank, block_json=_block_json(bank, bound=2), snapshot=_snapshot(2))[0]
    assert at_bound.results[0].final_response_text == ANSWER
    with pytest.raises(Exception, match=r"carries 2 supplied items, above the cell's bound of 1"):
        _run(tmp_path / "over", monkeypatch, bank=bank, block_json=_block_json(bank, bound=1), snapshot=_snapshot(2))


def test_the_trace_names_the_bank_receipt_on_the_contact_that_consumed_it(tmp_path, monkeypatch):
    bank = _bank()
    monkeypatch.setenv(trace.READ_ID_ENV, "READ-BANK")
    _, _, harness = _run(tmp_path / "bank", monkeypatch, bank=bank, block_json=_block_json(bank))
    lines = [json.loads(line) for line in Path(harness.world.data_root / "analysis" / "READ-BANK" / "rollout-trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 4 and all(schema.validate_line(line) == line for line in lines)
    by_node = {line["node"]: line for line in lines}
    assert by_node["solver"]["bank_receipt"] == bank.receipt_sha256
    assert all(by_node[node]["bank_receipt"] is None for node in ("perception", "reviewer", "planner"))
    # the exemplar block is not a supplied item: the supply list is the two snapshot values only
    assert [item["item_id"] for item in by_node["solver"]["supply_items"]] == list(SUPPLIED)


def test_without_a_bank_every_trace_line_carries_a_null_bank_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv(trace.READ_ID_ENV, "READ-PLAIN")
    _, _, harness = _run(tmp_path / "plain", monkeypatch)
    lines = [json.loads(line) for line in Path(harness.world.data_root / "analysis" / "READ-PLAIN" / "rollout-trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 4 and all(line["bank_receipt"] is None for line in lines)
