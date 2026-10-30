"""TFRL-v2 K-sample request plan and per-row credit (``scripts/flights/nx/tfrl_v2_sampler.py``, task T3): model-free, no data root, no card.

Proved here: K decoding seeds are distinct and a pure function of (base seed, iteration, row id, k); the plan is deterministic (same inputs,
same bytes); one Perception contact per row is shared by K Solver contacts that are request-identical except the seed; the cross-fit guard refuses
a bank whose source half is the consuming half; the budget line arithmetic; the reward / positive rule (best intent-correct trajectory beats the
baseline by more than 0, unstable baselines are not counted, missing scores are refused); and that nothing here builds a network client.
"""

from __future__ import annotations

import http.client
import json
import math
import socket
import sys
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NX = REPO / "scripts" / "flights" / "nx"
for _p in (str(REPO / "src"), str(REPO / "scripts" / "analysis"), str(NX)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paired_ci  # noqa: E402
import tfrl_v2_common as cm  # noqa: E402
import tfrl_v2_sampler as sp  # noqa: E402


def half_roster(half: str = cm.HALF_H1, rows: int = 6, audio: float = 2.0) -> dict:
    return {"roster_id": cm.HALF_ROSTER_ID[half], "split_identity_hash": ("a" if half == cm.HALF_H1 else "b") * 64,
            "items": [{"sample_id": f"{half}:row{i:03d}", "audio_seconds": audio + i * 0.5} for i in range(rows)]}


def bank_receipt(source: str = cm.HALF_H2, rows=("H2:rowX",)) -> dict:
    return {"source_halves": [source], "source_split_identities": {source: ("a" if source == cm.HALF_H1 else "b") * 64}, "source_row_ids": list(rows),
            "rule_bank": {"content_sha256": "1" * 64}, "exemplar_bank": {"content_sha256": "2" * 64},
            "measure": cm.MEASURE_VLLM, "exemplar_text": cm.EXEMPLAR_TEXT_RESOLVED}


def plan(half=cm.HALF_H1, iteration=1, **kw):
    receipt = kw.pop("bank_receipt", None)
    if iteration > 1 and receipt is None:
        receipt = bank_receipt(cm.other_half(half))
    return sp.build_plan(half_roster(half), half, iteration, bank_receipt=receipt, **kw)


# ---- seeds -----------------------------------------------------------------------------------------------------------------------------------


def test_k_seeds_are_distinct_per_row_and_deterministic():
    doc = plan(k=4)
    for row in doc["row_plan"]:
        seeds = [c["seed"] for c in row["contacts"] if c["node"] == "solver"]
        assert len(seeds) == 4 and len(set(seeds)) == 4
        assert all(1 <= s < 2**31 for s in seeds)
    assert plan(k=4) == doc
    other = plan(k=4, base_seed=cm.DEFAULT_BASE_SEED + 1)
    seeds = lambda d: [c["seed"] for c in d["row_plan"][0]["contacts"] if c["node"] == "solver"]  # noqa: E731
    assert seeds(other) != seeds(doc)


def test_seed_is_a_pure_function_of_base_row_and_k_and_rows_differ():
    assert cm.derive_seed(5, "r1", 0) == cm.derive_seed(5, "r1", 0)
    assert len({cm.derive_seed(5, f"r{i}", k) for i in range(50) for k in range(4)}) == 200
    assert cm.iteration_seed(5, 1) != cm.iteration_seed(5, 2)


# ---- plan shape ------------------------------------------------------------------------------------------------------------------------------


def test_plan_bytes_are_deterministic():
    assert cm.file_bytes(plan()) == cm.file_bytes(plan())
    assert cm.file_bytes(plan(iteration=2)) == cm.file_bytes(plan(iteration=2))


def test_one_shared_perception_and_k_solver_contacts_per_row():
    doc = plan(k=4)
    assert doc["program"]["program_id"] == "phase-native-module-v1-c7"
    assert doc["program"]["solver_sampler_arms"] == ["s1", "s2", "s3", "s4"]
    assert doc["program"]["solver_arms_in_order"] == ["s1", "s2", "s3", "s4", "aa", "lock"]
    for row in doc["row_plan"]:
        perceptions = [c for c in row["contacts"] if c["node"] == "perception"]
        solvers = [c for c in row["contacts"] if c["node"] == "solver"]
        assert len(perceptions) == 1 and perceptions[0]["shared_by"] == ["s1", "s2", "s3", "s4"]
        assert [c["arm"] for c in solvers] == ["s1", "s2", "s3", "s4"]
        assert len({c["contact_id"] for c in row["contacts"]}) == 5


def test_seed_only_mode_is_request_identical_except_the_seed():
    doc = plan(k=4, variation="seed-only")
    for row in doc["row_plan"]:
        solvers = [c for c in row["contacts"] if c["node"] == "solver"]
        assert len({c["request_identity_sha256"] for c in solvers}) == 1
        assert all(c["variant_id"] is None for c in solvers)
    # the identity differs between rows (it names the row) but never between the K arms of one row
    assert doc["row_plan"][0]["contacts"][1]["request_identity_sha256"] != doc["row_plan"][1]["contacts"][1]["request_identity_sha256"]


def test_rule_variants_mode_varies_requests_and_needs_variants():
    variants = [{"variant_id": f"v{i}", "instruction": "x", "ordering": "as-retrieved"} for i in range(4)]
    doc = plan(k=4, variation="rule-variants", seed_rule_bank={"variants": variants})
    solvers = [c for c in doc["row_plan"][0]["contacts"] if c["node"] == "solver"]
    assert [c["variant_id"] for c in solvers] == ["v0", "v1", "v2", "v3"]
    assert len({c["request_identity_sha256"] for c in solvers}) == 4
    with pytest.raises(sp.SamplerError, match="at least one variant"):
        plan(variation="rule-variants")
    with pytest.raises(sp.SamplerError, match="unknown variation"):
        plan(variation="temperature-ladder")


def test_iteration_rules_and_run_ids():
    with pytest.raises(sp.SamplerError, match="iteration must be"):
        plan(iteration=4)
    with pytest.raises(sp.SamplerError, match="consumes no distilled bank"):
        sp.build_plan(half_roster(), cm.HALF_H1, 1, bank_receipt=bank_receipt())
    with pytest.raises(sp.SamplerError, match="pass its receipt"):
        sp.build_plan(half_roster(), cm.HALF_H1, 2)
    assert sp.run_id_for(cm.HALF_H2, 3) == "SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I3-H2-NPU-PORT-V1"
    with pytest.raises(sp.SamplerError, match="expected"):
        sp.build_plan(half_roster(cm.HALF_H2), cm.HALF_H1, 1)


def test_k_must_be_positive():
    with pytest.raises(sp.SamplerError):
        plan(k=0)


# ---- cross-fit guard -------------------------------------------------------------------------------------------------------------------------


def test_cross_fit_guard_refuses_a_bank_from_the_consuming_half():
    with pytest.raises(sp.SamplerError, match="cross-fit guard"):
        sp.build_plan(half_roster(cm.HALF_H1), cm.HALF_H1, 2, bank_receipt=bank_receipt(cm.HALF_H1))
    with pytest.raises(sp.SamplerError, match="cross-fit guard"):                                    # the union bank is not for a construction half
        sp.build_plan(half_roster(cm.HALF_H2), cm.HALF_H2, 2, bank_receipt={**bank_receipt(cm.HALF_H1), "source_halves": [cm.HALF_H1, cm.HALF_H2]})
    doc = sp.build_plan(half_roster(cm.HALF_H1), cm.HALF_H1, 2, bank_receipt=bank_receipt(cm.HALF_H2))
    assert doc["banks"]["bank_source_halves"] == [cm.HALF_H2] and doc["banks"]["supply_bound_items"] == 43
    assert doc["banks"]["exemplar_block_token_cap"] == 2000 and doc["banks"]["rule_bank_token_cap"] == 400


def test_cross_fit_guard_refuses_row_overlap_and_equal_split_identity():
    overlap = bank_receipt(cm.HALF_H2, rows=("R01C-H1:row001",))
    with pytest.raises(sp.SamplerError, match="source rows of the bank"):
        sp.build_plan(half_roster(cm.HALF_H1), cm.HALF_H1, 2, bank_receipt=overlap)
    same_identity = {**bank_receipt(cm.HALF_H2), "source_split_identities": {cm.HALF_H2: "a" * 64}}      # equals H1's identity in the fixture
    with pytest.raises(sp.SamplerError, match="split identity"):
        sp.build_plan(half_roster(cm.HALF_H1), cm.HALF_H1, 2, bank_receipt=same_identity)
    with pytest.raises(cm.CrossFitError, match="no source_halves"):
        cm.check_cross_fit({}, cm.HALF_H1)


# ---- budget ----------------------------------------------------------------------------------------------------------------------------------


def test_budget_arithmetic_small():
    b = sp.budget([1.5, 2.5], 4)
    assert (b["rows"], b["contacts_per_row"], b["calls"], b["calls_with_fallback_ceiling"]) == (2, 5, 10, 20)
    assert b["audio_seconds"] == 20                                    # (1.5 + 2.5) s x 5 contacts
    assert b["tokens"] == 10 * 4372
    assert b["host_wall_hours"] == round(10 * 4372 / 1422.59 / 3600, 4)
    assert b["card_hours"] == round(10 * 4372 / 1422.59 / 3600 * 8, 4)
    assert b["card_hours_with_margin"] == round(b["card_hours"] * 1.3, 4) or abs(b["card_hours_with_margin"] - b["card_hours"] * 1.3) < 2e-4
    c7 = sp.budget([1.5, 2.5], 4, include_twin_and_lock=True)
    assert (c7["contacts_per_row"], c7["calls"], c7["audio_seconds"]) == (7, 14, 28)


def test_budget_full_column_numbers_at_k4():
    b = sp.budget([2726.8471875 / 625] * 625, 4)
    assert b["calls"] == 3125 and b["tokens"] == 3125 * 4372 == 13_662_500
    assert b["audio_seconds"] == math.ceil(2726.8471875 * 5)
    assert abs(b["card_hours"] - 13_662_500 / 1422.59 / 3600 * 8) < 1e-3


def test_budget_sum_adds_counts_and_rederives_hours():
    a, b = sp.budget([2.0] * 3, 4), sp.budget([2.0] * 2, 4)
    total = sp.sum_budgets([a, b])
    assert total["calls"] == a["calls"] + b["calls"] and total["rows"] == 5 and total["audio_seconds"] == 50
    assert abs(total["card_hours"] - sp.budget([2.0] * 5, 4)["card_hours"]) < 1e-3
    with pytest.raises(sp.SamplerError, match="different shapes"):
        sp.sum_budgets([a, sp.budget([2.0], 8)])
    assert "13 calls" not in sp.budget_line("x", a) and "15 calls" in sp.budget_line("x", a)


def test_plan_carries_both_budget_forms():
    doc = plan(k=4)
    assert doc["budget"]["sampler_contacts"]["contacts_per_row"] == 5 and doc["budget"]["program_c7"]["contacts_per_row"] == 7
    assert doc["budget"]["sampler_contacts"]["rows"] == 6


# ---- credit ----------------------------------------------------------------------------------------------------------------------------------


def table(rows):
    """rows: {row_id: {arm: (f1, intent_ok)}} -> the sampler's score table."""
    return {(r, arm): v for r, arms in rows.items() for arm, v in arms.items()}


def arms(lock, s, intent=(True, True, True, True), aa=None):
    out = {"lock": (lock, True)}
    if aa is not None:
        out["aa"] = (aa, True)
    out.update({f"s{i + 1}": (s[i], intent[i]) for i in range(4)})
    return out


def test_positive_when_best_intent_correct_beats_baseline():
    credit = sp.credit_rows(["a"], 1, 4, table({"a": arms(0.5, [0.5, 0.75, 0.25, 0.0])}))
    row = credit[0]
    assert row["positive"] and row["best_k"] == 1 and row["worst_k"] == 3 and row["reason"] == "positive"
    assert [t["reward"] for t in row["trajectories"]] == [0.0, 0.25, -0.25, -0.5]
    assert row["trajectories"][1]["trajectory_id"] == "tfrl-v2/i1/a/k1"


def test_intent_must_be_correct_for_the_best_trajectory():
    credit = sp.credit_rows(["a"], 1, 4, table({"a": arms(0.5, [0.5, 1.0, 0.25, 0.75], intent=(True, False, True, False))}))
    assert credit[0]["best_k"] == 0 and not credit[0]["positive"] and credit[0]["reason"] == "no-gain-over-baseline"
    none_ok = sp.credit_rows(["a"], 1, 4, table({"a": arms(0.5, [1, 1, 1, 1], intent=(False, False, False, False))}))
    assert none_ok[0]["best_k"] is None and none_ok[0]["reason"] == "no-intent-correct-trajectory" and none_ok[0]["worst_k"] is None


def test_gain_must_be_strictly_positive_and_ties_go_to_the_lowest_k():
    tie = sp.credit_rows(["a"], 1, 4, table({"a": arms(0.5, [0.5, 0.5, 0.5, 0.5])}))
    assert not tie[0]["positive"] and tie[0]["best_k"] == 0
    both = sp.credit_rows(["a"], 1, 4, table({"a": arms(0.5, [0.75, 0.75, 0.25, 0.25])}))
    assert both[0]["positive"] and both[0]["best_k"] == 0 and both[0]["worst_k"] == 3


def test_unstable_baseline_rows_are_not_counted_when_a_twin_is_supplied():
    scores = table({"a": arms(0.5, [1, 1, 1, 1], aa=0.25), "b": arms(0.5, [1, 1, 1, 1], aa=0.5)})
    credit = sp.credit_rows(["a", "b"], 1, 4, scores, twin_arm="aa")
    assert (credit[0]["positive"], credit[0]["reason"], credit[0]["baseline_stable"]) == (False, "baseline-unstable", False)
    assert (credit[1]["positive"], credit[1]["baseline_stable"]) == (True, True)
    assert sp.credit_summary(credit) == {"rows": 2, "positives": 1, "positive_share": 0.5, "reasons": {"baseline-unstable": 1, "positive": 1}}


def test_missing_score_is_refused():
    scores = table({"a": arms(0.5, [1, 1, 1, 1])})
    del scores[("a", "s3")]
    with pytest.raises(sp.SamplerError, match="missing score"):
        sp.credit_rows(["a"], 1, 4, scores)
    with pytest.raises(sp.SamplerError, match="missing score"):
        sp.credit_rows(["a"], 1, 4, table({"a": arms(0.5, [1, 1, 1, 1])}), twin_arm="aa")


def test_scores_from_counts_use_the_repo_per_row_f1_helper():
    counts = {"lock": {"r1": {"slu_f1_word_tp": 1, "slu_f1_char_tp": 3, "slu_f1_word_fp": 1, "slu_f1_char_fp": 1, "slu_f1_word_fn": 0, "slu_f1_char_fn": 0,
                              "scenario_action_exact": 1.0}},
              "s1": {"r1": {"slu_f1_word_tp": 0, "slu_f1_char_tp": 0, "slu_f1_word_fp": 0, "slu_f1_char_fp": 0, "slu_f1_word_fn": 0, "slu_f1_char_fn": 0,
                            "scenario_action_exact": 0.0}}}
    table_ = sp.scores_from_counts(counts)
    assert table_[("r1", "lock")] == (paired_ci.row_f1(counts["lock"]["r1"]), True) and table_[("r1", "lock")][0] == pytest.approx(8 / 10)
    assert table_[("r1", "s1")] == (1.0, False)                         # empty gold and empty prediction scores 1.0 (the repo convention)


def test_scores_from_lines_refuses_two_scorers_without_a_choice():
    base = {"row_id": "r", "arm": "lock", "scenario_action_exact": True, "slu_f1": 0.5}
    lines = [{**base, "scorer_id": "official"}, {**base, "scorer_id": "other", "slu_f1": 0.7}]
    with pytest.raises(sp.SamplerError, match="two scorers"):
        sp.scores_from_lines(lines)
    assert sp.scores_from_lines(lines, scorer_id="other") == {("r", "lock"): (0.7, True)}


def test_read_trace_refuses_a_line_outside_the_schema(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"kind": "contact"}\n', encoding="utf-8")
    with pytest.raises(sp.SamplerError, match="missing fields"):
        sp.read_trace([path])


# ---- CLI and no-model -------------------------------------------------------------------------------------------------------------------------


def test_cli_plan_writes_once_and_prints_the_budget(tmp_path, capsys):
    roster = tmp_path / "h1.json"
    roster.write_text(json.dumps(half_roster()), encoding="utf-8")
    out = tmp_path / "plan.json"
    argv = ["plan", "--half", cm.HALF_H1, "--roster", str(roster), "--iteration", "1", "--out", str(out)]
    assert sp.main(argv) == 0
    first = out.read_bytes()
    printed = capsys.readouterr().out
    assert "sampler contacts" in printed and "program c7" in printed and "calls" in printed
    assert sp.main(argv) == 0 and out.read_bytes() == first             # re-run: unchanged
    assert json.loads(first)["roster"]["file_sha256"] == cm.sha256_file(roster)
    out.write_bytes(b"x")
    assert sp.main(argv) == 2                                           # never overwritten


def test_no_network_client_is_constructed_by_the_plan_or_the_credit(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a network client was constructed")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(http.client.HTTPConnection, "__init__", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    plan(iteration=2)
    sp.credit_rows(["a"], 1, 4, table({"a": arms(0.5, [1, 1, 1, 1])}))
    sp.budget([1.0], 4)


def test_module_imports_no_network_or_model_library_and_is_ascii():
    text = (NX / "tfrl_v2_sampler.py").read_text(encoding="utf-8")
    for needle in ("import socket", "import requests", "import httpx", "import urllib", "import http", "openai", "anthropic", "aiohttp", "subprocess"):
        assert needle not in text, needle
    assert all(ord(ch) < 128 for ch in text) and all(ord(ch) < 128 for ch in Path(__file__).read_text(encoding="utf-8"))
