"""TFRL-v2 distiller (``scripts/flights/nx/tfrl_v2_distiller.py``, task T3): model-free, no data root, no card.

Proved here, on fake trace lines and fake scores: the exemplar bank is keyed by scenario and holds the best trajectory's answer with the supplied
items it used; the per-scenario exemplar block stays within 2,000 tokens (and a smaller cap when asked); a rule survives only with at least 3
distinct supporting rows and the rule text stays within 400 tokens; the ``tfrl`` lines validate against ``saea-rollout-trace-v1``; the outputs are
byte-identical across runs and write-once; the bank receipt names the source half and the cross-fit guard refuses the source half as consumer;
incomplete rows are refused; unsafe labels never reach a bank; and nothing constructs a network client.
"""

from __future__ import annotations

import http.client
import json
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

from speech_aware_evidence_acquisition.contracts import rollout_trace as rt  # noqa: E402
import tfrl_v2_common as cm  # noqa: E402
import tfrl_v2_distiller as dd  # noqa: E402
import tfrl_v2_sampler as sp  # noqa: E402
import tfrl_v2_text as tx  # noqa: E402

HALF = cm.HALF_H1
RUN = sp.run_id_for(HALF, 1)
ATT = sp.attempt_id_for(HALF, 1)
TS = "2026-10-31T00:00:00Z"


def ans(scenario="alarm", action="set", entities=()):
    return {"parsed": True, "scenario": scenario, "action": action, "entities": [{"type": t, "filler": f} for t, f in entities]}


def contact(row, arm, answer, used=(), unused=()):
    items = [{"item_id": i, "rank": n + 1, "score": 0.5, "used": True, "used_basis": "json-form"} for n, i in enumerate(used)]
    items += [{"item_id": i, "rank": len(items) + n + 1, "score": 0.1, "used": False, "used_basis": "json-form"} for n, i in enumerate(unused)]
    line = {"schema": rt.SCHEMA, "kind": "contact", "ts": TS, "run_id": RUN, "attempt_id": ATT, "row_id": row, "arm": arm, "node": "solver",
            "role_id": None, "contact_id": f"{RUN}/{row}/solver/{arm}", "contact_index": 1, "sample_index": 0, "request_id": None, "outcome": "ok",
            "error_type": None, "id_slot": 0, "instance": 0, "supply_variant_id": None, "supply_items": items, "prompt_tokens": 100,
            "completion_tokens": 50, "thinking_tokens": 30, "thinking_tokens_basis": "server-usage-details", "reasoning_chars": 10,
            "thinking_budget": 3072, "max_tokens": 4096, "truncated": False, "finish_reason": "stop", "forced_close": False,
            "answer_only_reask": False, "answer": answer, "response_text_sha256": None, "latency_seconds": 1.0}
    return rt.validate_line(line)


def score(row, arm, f1, ok):
    return rt.validate_line({"schema": rt.SCHEMA, "kind": "score", "ts": TS, "run_id": RUN, "attempt_id": ATT, "row_id": row, "arm": arm,
                             "scorer_id": "official", "scenario_action_exact": ok, "slu_f1": f1})


class Fixture:
    """Rows of one half with per-arm answers and scores; ``add`` registers one row."""

    def __init__(self):
        self.rows: list[str] = []
        self.contacts: list[dict] = []
        self.scores: list[dict] = []

    def add(self, row, lock, trajectories, aa=None):
        """trajectories: four (f1, intent_ok, answer, used_items) tuples for s1..s4."""
        self.rows.append(row)
        aa = lock if aa is None else aa
        self.scores += [score(row, "lock", lock, True), score(row, "aa", aa, True)]
        for i, (f1, ok, answer, used) in enumerate(trajectories):
            arm = f"s{i + 1}"
            self.contacts.append(contact(row, arm, answer, used=used, unused=("unused-a",)))
            self.scores.append(score(row, arm, f1, ok))

    def roster(self, half=HALF):
        return {"roster_id": cm.HALF_ROSTER_ID[half], "split_identity_hash": "c" * 64,
                "items": [{"sample_id": r, "audio_seconds": 2.0} for r in self.rows]}

    def distil(self, **kw):
        return dd.distil(self.roster(), HALF, 1, self.contacts, self.scores, **kw)


def date_row(fx, i, scenario="alarm", filler="monday"):
    """A positive row whose best answer reports a date slot that the worst (intent-flipped) and the middle answers lack."""
    good = ans(scenario, "set", [("date", filler)])
    fx.add(f"r{i:03d}", 0.5, [(1.0, True, good, (f"item-{i}", "item-shared")), (0.5, True, ans(scenario, "set"), ()),
                              (0.5, True, ans(scenario, "set"), ()), (0.0, False, ans(scenario, "query"), ())])


def spurious_row(fx, i):
    """A positive row whose worst answer adds a time slot (a different, rarer difference)."""
    fx.add(f"r{i:03d}", 0.5, [(1.0, True, ans("alarm", "set"), ()), (0.5, True, ans("alarm", "set", [("time", "5 pm")]), ()),
                              (0.75, True, ans("alarm", "set"), ()), (0.75, True, ans("alarm", "set"), ())])


def base_fixture():
    fx = Fixture()
    for i in range(4):
        date_row(fx, i)                                              # 4 rows -> intent_flip + slot_missing(date) survive the vote
    for i in range(4, 6):
        spurious_row(fx, i)                                          # 2 rows -> slot_spurious(time) is refused below the vote
    date_row(fx, 6, scenario="music", filler="jazz")                  # a second scenario
    fx.add("r007", 1.0, [(1.0, True, ans(), ())] * 4)                # lock already right: no gain
    fx.add("r008", 0.5, [(1.0, False, ans("alarm", "query"), ())] * 4)   # best is intent-wrong: no intent-correct trajectory
    return fx


# ---- exemplar bank ---------------------------------------------------------------------------------------------------------------------------


def test_exemplar_bank_is_keyed_by_scenario_and_holds_the_best_answer_with_its_used_items():
    result = base_fixture().distil()
    bank = result["exemplar_bank"]
    assert sorted(bank["by_scenario"]) == ["alarm", "music"]
    entry = bank["by_scenario"]["music"]["entries"][0]
    assert entry["source_row_id"] == "r006" and entry["entities"] == [{"type": "date", "filler": "jazz"}]
    assert entry["evidence_item_ids"] == ["item-6", "item-shared"]            # used items only; "unused-a" is not carried
    assert entry["text"] == "- evidence: item-6, item-shared -> intent: music_set; entities: date=jazz"
    alarm_rows = {e["source_row_id"] for e in bank["by_scenario"]["alarm"]["entries"]}
    # every positive alarm row; r005 repeats r004's exemplar text (no evidence, no entities) and is deduplicated; r007 / r008 are not positives
    assert alarm_rows == {"r000", "r001", "r002", "r003", "r004"}
    assert result["receipt_core"]["credit_summary"]["positives"] == 7


def test_exemplar_block_is_capped_at_2000_tokens_with_the_default_cap():
    fx = Fixture()
    for i in range(20):
        fx.add(f"r{i:03d}", 0.5, [(1.0 - i * 0.001, True, ans("alarm", "set", [("note", "x" * 900 + f" {i}")]), ()), (0.5, True, ans(), ()),
                                  (0.5, True, ans(), ()), (0.25, True, ans(), ())])
    bank = fx.distil()["exemplar_bank"]
    scenario = bank["by_scenario"]["alarm"]
    assert scenario["block_tokens"] <= cm.EXEMPLAR_BLOCK_TOKEN_CAP == 2000
    assert cm.estimate_tokens(scenario["block_text"]) == scenario["block_tokens"]
    assert 1 < len(scenario["entries"]) < 8                                    # the token cap, not the count cap, stopped it
    assert [e["source_row_id"] for e in scenario["entries"]] == [f"r{i:03d}" for i in range(len(scenario["entries"]))]   # best reward first


def test_exemplar_count_cap_and_small_token_cap_and_duplicates():
    fx = Fixture()
    for i in range(12):
        date_row(fx, i, filler=f"day{i}")
    credit = sp.credit_rows(fx.rows, 1, 4, sp.scores_from_lines(fx.scores), twin_arm="aa")
    answers = sp.solver_contacts(fx.contacts, sp.sampler_arms(4))
    bank, fate = dd.build_exemplar_bank(credit, answers, iteration=1, source_half=HALF)
    assert len(bank["by_scenario"]["alarm"]["entries"]) == cm.MAX_EXEMPLARS_PER_SCENARIO == 8
    assert sorted(fate.values()).count("scenario-count-cap") == 4
    tiny, fate2 = dd.build_exemplar_bank(credit, answers, iteration=1, source_half=HALF, token_cap=60, max_per_scenario=100)
    assert tiny["by_scenario"]["alarm"]["block_tokens"] <= 60 and "token-cap" in fate2.values()
    dup = Fixture()
    for i in range(2):
        dup.add(f"r{i:03d}", 0.5, [(1.0, True, ans("alarm", "set", [("date", "monday")]), ("same",)), (0.5, True, ans(), ()), (0.5, True, ans(), ()), (0.0, True, ans(), ())])
    dup_bank, dup_fate = dd.build_exemplar_bank(sp.credit_rows(dup.rows, 1, 4, sp.scores_from_lines(dup.scores)),
                                                sp.solver_contacts(dup.contacts, sp.sampler_arms(4)), iteration=1, source_half=HALF)
    assert [e["source_row_id"] for e in dup_bank["by_scenario"]["alarm"]["entries"]] == ["r000"]      # identical text: only the first row is kept
    assert dup_fate == {"tfrl-v2/i1/r000/k0": "banked", "tfrl-v2/i1/r001/k0": "duplicate-text"}


# ---- rule bank -------------------------------------------------------------------------------------------------------------------------------


def test_rule_survives_only_with_three_distinct_rows():
    result = base_fixture().distil()
    rules = {r["candidate_key"]: r for r in result["rule_bank"]["rules"]}
    assert set(rules) == {"intent_flip|alarm_set|alarm_query", "slot_missing|alarm|date"} or "slot_missing|alarm|date" in rules
    assert "slot_missing|alarm|date" in rules and rules["slot_missing|alarm|date"]["support_count"] == 4
    assert rules["slot_missing|alarm|date"]["text"] == "In alarm requests, report the date entity whenever it is spoken; do not leave it out."
    refused = {r["candidate"]: r for r in result["refused"]}
    assert refused["slot_spurious|alarm|time"]["reason"] == "below-vote" and refused["slot_spurious|alarm|time"]["support_rows"] == 2
    assert "slot_missing|music|date" in refused and refused["slot_missing|music|date"]["support_rows"] == 1
    assert all(r["support_count"] >= 3 for r in result["rule_bank"]["rules"]) and result["rule_bank"]["vote_min"] == 3


def test_a_row_votes_once_and_every_rule_names_its_contrast_pairs():
    result = base_fixture().distil()
    rule = next(r for r in result["rule_bank"]["rules"] if r["candidate_key"] == "slot_missing|alarm|date")
    assert rule["support_rows"] == ["r000", "r001", "r002", "r003"] and len(rule["pair_ids"]) == 4
    pair_ids = {p["pair_id"] for p in result["pairs"]}
    assert set(rule["pair_ids"]) <= pair_ids
    assert all(p["best_f1"] > p["worst_f1"] for p in result["pairs"])


def test_two_supporting_rows_do_not_pass_the_vote_and_three_do():
    for rows, expect in ((2, False), (3, True)):
        fx = Fixture()
        for i in range(rows):
            date_row(fx, i)
        keys = {r["candidate_key"] for r in fx.distil()["rule_bank"]["rules"]}
        assert ("slot_missing|alarm|date" in keys) is expect


def test_rule_bank_text_is_capped_at_400_tokens():
    fx = Fixture()
    for slot in range(40):                                               # 40 slot types, 3 rows each: 40 candidates that pass the vote
        for i in range(3):
            fx.add(f"r{slot:02d}{i}", 0.5, [(1.0, True, ans("alarm", "set", [(f"type{slot:02d}", "v")]), ()), (0.5, True, ans("alarm", "set"), ()),
                                           (0.5, True, ans("alarm", "set"), ()), (0.25, True, ans("alarm", "set"), ())])
    bank = fx.distil()["rule_bank"]
    assert 0 < len(bank["rules"]) < 40
    assert bank["rendered_tokens"] <= cm.RULE_BANK_TOKEN_CAP == 400
    assert cm.estimate_tokens(bank["rendered_text"]) == bank["rendered_tokens"]
    refused = fx.distil()["refused"]
    assert sum(1 for r in refused if r["reason"] == "token-cap") == 40 - len(bank["rules"])


def test_answer_diff_kinds():
    best = ans("alarm", "set", [("date", "monday"), ("time", "5 pm")])
    worst = ans("alarm", "query", [("date", "tuesday"), ("place", "home")])
    kinds = {(d["kind"], d.get("slot_type")) for d in dd.answer_diff(best, worst)}
    assert kinds == {("intent_flip", None), ("slot_value", "date"), ("slot_missing", "time"), ("slot_spurious", "place")}
    assert dd.answer_diff(best, best) == []


# ---- tfrl lines ------------------------------------------------------------------------------------------------------------------------------


def test_tfrl_lines_follow_the_trace_schema():
    result = base_fixture().distil()
    lines = result["tfrl_lines"]
    assert len(lines) == 9 * 4
    for line in lines:
        rt.validate_line(line)
        assert line["kind"] == "tfrl" and line["iteration"] == 1 and line["run_id"] == RUN and line["attempt_id"] == ATT
    by_id = {line["trajectory_id"]: line for line in lines}
    best = by_id["tfrl-v2/i1/r000/k0"]
    assert best["selected"] and best["reason"] == "exemplar-source" and best["reward"] == 0.5
    assert by_id["tfrl-v2/i1/r000/k3"]["reason"] == "contrast-negative" and not by_id["tfrl-v2/i1/r000/k3"]["selected"]
    assert by_id["tfrl-v2/i1/r000/k1"]["reason"] == "positive-row-other-trajectory"
    assert by_id["tfrl-v2/i1/r007/k0"]["reason"] == "non-positive-row:no-gain-over-baseline"
    assert by_id["tfrl-v2/i1/r008/k0"]["reason"] == "non-positive-row:no-intent-correct-trajectory"
    assert sum(1 for line in lines if line["selected"]) == sum(len(s["entries"]) for s in result["exemplar_bank"]["by_scenario"].values())


# ---- determinism, write-once, receipt, cross-fit ------------------------------------------------------------------------------------------------


def test_outputs_are_deterministic_and_write_once(tmp_path):
    receipt_a = dd.write_outputs(base_fixture().distil(), tmp_path / "a")
    receipt_b = dd.write_outputs(base_fixture().distil(), tmp_path / "b")
    assert receipt_a == receipt_b
    names = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert names == ["bank-receipt.json", "contrast-pairs.jsonl", "credit.jsonl", "exemplar-bank.json", "refused-rules.jsonl", "rule-bank.json", "tfrl-trace.jsonl"]
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    dd.write_outputs(base_fixture().distil(), tmp_path / "a")                          # same inputs again: no-op
    (tmp_path / "a" / "rule-bank.json").write_bytes(b"{}\n")
    with pytest.raises(dd.DistillError, match="refusing to overwrite"):
        dd.write_outputs(base_fixture().distil(), tmp_path / "a")


def test_bank_receipt_names_the_source_half_and_the_guard_uses_it(tmp_path):
    fx = base_fixture()
    receipt = dd.write_outputs(fx.distil(), tmp_path)
    assert receipt["source_half"] == HALF and receipt["source_halves"] == [HALF] and receipt["consuming_half"] == cm.HALF_H2
    assert receipt["source_split_identities"] == {HALF: "c" * 64} and receipt["model_calls"] == 0
    assert receipt["exemplar_bank"]["max_block_tokens"] <= 2000 and receipt["rule_bank"]["rendered_tokens"] <= 400
    assert set(receipt["source_row_ids"]) <= set(fx.rows) and "r007" not in receipt["source_row_ids"]
    loaded, exemplars, rules = dd.load_bank(tmp_path / "bank-receipt.json")
    assert loaded == receipt and cm.content_sha256(exemplars) == receipt["exemplar_bank"]["content_sha256"]
    cm.check_cross_fit(receipt, cm.HALF_H2, consuming_row_ids=["other-half-row"], consuming_split_identity="d" * 64)     # the other half may consume it
    with pytest.raises(cm.CrossFitError, match="consuming half"):
        cm.check_cross_fit(receipt, HALF)
    with pytest.raises(cm.CrossFitError, match="split identity"):
        cm.check_cross_fit(receipt, cm.HALF_H2, consuming_split_identity="c" * 64)
    with pytest.raises(cm.CrossFitError, match="source rows"):
        cm.check_cross_fit(receipt, cm.HALF_H2, consuming_row_ids=["r000"])


class _FakeTokenizer:
    """A fake tokenizer client: 4 characters per token plus a 7-token chat wrapper (the contract of the vLLM chat-form /tokenize)."""

    def request_json(self, method, path, payload):
        text = payload["messages"][0]["content"][0]["text"]
        n = -(-len(text) // 4) + 7
        return {"count": n, "tokens": list(range(n))}


def test_sampler_accepts_the_receipt_for_the_other_half_and_refuses_it_for_the_source_half(tmp_path):
    fx = base_fixture()
    ids = {i for c in fx.contacts for item in c["supply_items"] for i in [item["item_id"]]}
    resolver = tx.ItemTextResolver({i: f"text of {i}" for i in ids})
    receipt = dd.write_outputs(fx.distil(resolver=resolver, measure=tx.VllmTokenizeMeasure(_FakeTokenizer())), tmp_path)
    other = {"roster_id": cm.HALF_ROSTER_ID[cm.HALF_H2], "split_identity_hash": "d" * 64, "items": [{"sample_id": "h2-row", "audio_seconds": 1.0}]}
    sp.build_plan(other, cm.HALF_H2, 2, bank_receipt=receipt)
    with pytest.raises(sp.SamplerError, match="cross-fit guard"):
        sp.build_plan(base_fixture().roster(HALF), HALF, 2, bank_receipt=receipt)


def test_load_bank_detects_a_tampered_bank(tmp_path):
    dd.write_outputs(base_fixture().distil(), tmp_path)
    (tmp_path / "exemplar-bank.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(dd.DistillError, match="file hash differs"):
        dd.load_bank(tmp_path / "bank-receipt.json")


# ---- inputs ----------------------------------------------------------------------------------------------------------------------------------


def test_incomplete_rows_are_refused_or_skipped_on_request():
    fx = base_fixture()
    fx.contacts = [c for c in fx.contacts if not (c["row_id"] == "r003" and c["arm"] == "s2")]
    with pytest.raises(dd.DistillError, match="lack K=4"):
        fx.distil()
    result = fx.distil(allow_missing_rows=True)
    assert result["receipt_core"]["counts"]["rows_skipped"] == 1 and result["receipt_core"]["counts"]["rows_used"] == 8
    with pytest.raises(dd.DistillError, match="no row has"):
        dd.distil(fx.roster(), HALF, 1, [], [], allow_missing_rows=True)


def test_rows_of_the_other_half_in_the_trace_are_ignored():
    fx = base_fixture()
    stray = [contact("zzz", f"s{i}", ans()) for i in range(1, 5)] + [score("zzz", a, 0.5, True) for a in ("lock", "aa", "s1", "s2", "s3", "s4")]
    result = dd.distil(fx.roster(), HALF, 1, fx.contacts + stray, fx.scores + stray[4:], with_twin=True)
    assert result["receipt_core"]["counts"]["rows_in_trace_outside_half"] == 1
    assert "zzz" not in json.dumps(result["credit"])


def test_a_later_delivered_contact_supersedes_a_truncated_first_one():
    fx = base_fixture()
    late = contact("r000", "s1", ans("alarm", "set", [("date", "friday")]), used=("late-item",))
    result = dd.distil(fx.roster(), HALF, 1, fx.contacts + [late], fx.scores)
    entry = next(e for e in result["exemplar_bank"]["by_scenario"]["alarm"]["entries"] if e["source_row_id"] == "r000")
    assert entry["entities"] == [{"type": "date", "filler": "friday"}] and entry["evidence_item_ids"] == ["late-item"]


def test_unsafe_labels_and_non_ascii_fillers_never_reach_a_bank():
    fx = Fixture()
    for i in range(3):
        fx.add(f"r{i}", 0.5, [(1.0, True, ans("alarm", "set", [("date", "caf\u00e9")]), ()), (0.5, True, ans("alarm", "set"), ()),
                              (0.5, True, ans("alarm", "set"), ()), (0.0, True, ans("alarm", "set"), ())])
    fx.add("r3", 0.5, [(1.0, True, ans("alarm now", "set"), ()), (0.5, True, ans(), ()), (0.5, True, ans(), ()), (0.0, True, ans(), ())])
    result = fx.distil()
    assert result["exemplar_bank"]["by_scenario"] == {} and result["rule_bank"]["rules"] == []
    assert {line["reason"] for line in result["tfrl_lines"] if line["trajectory_id"].endswith("/k0")} == {"positive-best-not-banked:unsafe-label"}
    assert not dd.labels_safe(ans("a b", "set")) and dd.labels_safe(ans("alarm", "set", [("date", "monday 5")]))


def test_cli_end_to_end_and_exit_codes(tmp_path, capsys):
    fx = base_fixture()
    roster = tmp_path / "h1.json"
    roster.write_text(json.dumps(fx.roster()), encoding="utf-8")
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(rt.dumps_line(line) for line in fx.contacts + fx.scores), encoding="utf-8")
    argv = ["--half", HALF, "--roster", str(roster), "--iteration", "1", "--trace", str(trace), "--out-dir", str(tmp_path / "out")]
    assert dd.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["source_half"] == HALF and summary["rule_bank"]["rules"] == 2
    first = (tmp_path / "out" / "bank-receipt.json").read_bytes()
    assert dd.main(argv) == 0 and (tmp_path / "out" / "bank-receipt.json").read_bytes() == first
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"kind": "tfrl"}\n', encoding="utf-8")
    assert dd.main(argv[:6] + ["--trace", str(bad), "--out-dir", str(tmp_path / "out2")]) == 2        # a line outside the schema is refused


# ---- no model --------------------------------------------------------------------------------------------------------------------------------


def test_no_network_client_is_constructed(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("a network client was constructed")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(http.client.HTTPConnection, "__init__", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    dd.write_outputs(base_fixture().distil(), tmp_path)


def test_module_imports_no_network_or_model_library_and_text_is_ascii():
    for path in (NX / "tfrl_v2_distiller.py", NX / "tfrl_v2_common.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("import socket", "import requests", "import httpx", "import urllib", "import http", "openai", "anthropic", "aiohttp", "subprocess"):
            assert needle not in text, (path.name, needle)
        assert all(ord(ch) < 128 for ch in text), path
    assert "NO MODEL CALL" in (NX / "tfrl_v2_distiller.py").read_text(encoding="utf-8")
    assert all(ord(ch) < 128 for ch in Path(__file__).read_text(encoding="utf-8"))


# ---- the rule-variants seed bank (plan section 11 addendum, ruling 5) --------------------------------------------------------------------


def _rule(rule_id, scenario, support, key):
    return {"rule_id": rule_id, "scenario": scenario, "text": f"text of {rule_id}", "support_count": support, "candidate_key": key}


def test_rule_variants_seed_bank_has_the_four_ruled_variants():
    bank = {"rules": [_rule("r-a", "alarm", 5, "k|alarm|date"), _rule("r-b", "music", 4, "k|music|genre"), _rule("r-c", "alarm", 4, "k|alarm|time"),
                      _rule("r-d", "weather", 3, "k|weather|place"), _rule("r-e", "alarm", 3, "k|alarm|zone")]}
    seed = dd.rule_variants_seed_bank(2, bank)
    names = [v["variant_id"] for v in seed["variants"]]
    assert names == ["k0-no-rules", "k1-full-bank", "k2-row-scenario", "k3-top-half"] and seed["iteration"] == 2
    k0, k1, k2, k3 = seed["variants"]
    assert k0["rule_ids"] == [] and k0["rendered_text"] == ""
    assert k1["rule_ids"] == ["r-a", "r-b", "r-c", "r-d", "r-e"] and k1["rendered_text"].startswith("Consumption rules:\n- text of r-a")
    assert k2["by_scenario"]["alarm"]["rule_ids"] == ["r-a", "r-c", "r-e"] and k2["by_scenario"]["music"]["rule_ids"] == ["r-b"]
    assert sorted(k2["by_scenario"]) == ["alarm", "music", "weather"]
    # top half = ceil(5 / 2) = 3 rules by support desc, ties by candidate key (k|alarm|time < k|music|genre), kept in bank order
    assert k3["rule_ids"] == ["r-a", "r-b", "r-c"]
    # a tie broken by candidate key, not by bank order
    swapped = {"rules": [_rule("r-x", "music", 4, "k|music|genre"), _rule("r-y", "alarm", 4, "k|alarm|time")]}
    assert dd.rule_variants_seed_bank(2, swapped)["variants"][3]["rule_ids"] == ["r-y"]
    # one rule: the top half is that rule, not the empty set
    assert dd.rule_variants_seed_bank(2, {"rules": [_rule("r-a", "alarm", 3, "k")]})["variants"][3]["rule_ids"] == ["r-a"]
    # deterministic, and absent for an empty or missing bank
    assert dd.rule_variants_seed_bank(2, bank) == seed
    assert dd.rule_variants_seed_bank(2, {"rules": []}) is None and dd.rule_variants_seed_bank(2, None) is None
