"""The T READ: ``scripts/analysis/slurp_tfrl_v2_port_read.py`` (per-iteration construction read and the Fold B read of the TFRL-v2 cell).

Model-free, synthetic fixtures (valid ``saea-rollout-trace-v1`` lines, synthetic per-row official count tables). The proofs: (1) the vote over s1..s4; (2) the per-half
construction numbers (positive rate, mean best-of-K gain, skipped rows, rule survivors, exemplar coverage per scenario) are deterministic and independent of line order;
(3) the stop rule: a construction gain of +1.0 pp passes and +0.9 pp stops, an unscored vote is UNDETERMINED, at most 3 iterations, nothing accepted stops; (4) the Fold B words reuse
the port reader's paired machinery with the replicate-widened CI95 and the pooled-and-per-row NOMINATED rule; (5) the iteration verdict needs the gain gate and the twin; (6) the CLI
writes write-once files, refuses a moved recipe, and two runs over equal inputs write equal bytes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "scripts" / "analysis", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import paired_ci as pc  # noqa: E402
import slurp_direct_fold_b_core_read as base  # noqa: E402
import slurp_tfrl_v2_port_read as rd  # noqa: E402
import tfrl_v2_common as cm  # noqa: E402
import tfrl_v2_port_family as tf  # noqa: E402
import tfrl_v2_port_rails as rails  # noqa: E402
import tfrl_v2_sampler as sp  # noqa: E402

from speech_aware_evidence_acquisition.contracts import rollout_trace as rt  # noqa: E402

TS = "2026-10-31T00:00:00Z"
H1, H2 = cm.HALF_H1, cm.HALF_H2


# ---- fixtures --------------------------------------------------------------------------------------------------------------------------------------


def ans(scenario="alarm", action="set", entities=()):
    return {"parsed": True, "scenario": scenario, "action": action, "entities": [{"type": t, "filler": f} for t, f in entities]}


def contact(half, row, arm, answer, iteration=1):
    run, att = sp.run_id_for(half, iteration), sp.attempt_id_for(half, iteration)
    line = {"schema": rt.SCHEMA, "kind": "contact", "ts": TS, "run_id": run, "attempt_id": att, "row_id": row, "arm": arm, "node": "solver", "role_id": None,
            "contact_id": f"{run}/{row}/solver/{arm}", "contact_index": 1, "sample_index": 0, "request_id": None, "outcome": "ok", "error_type": None, "id_slot": 0, "instance": 0,
            "supply_variant_id": None, "bank_receipt": None, "supply_items": [], "prompt_tokens": 100, "completion_tokens": 50, "thinking_tokens": 30, "thinking_tokens_basis": "server-usage-details",
            "reasoning_chars": 10, "thinking_budget": 3072, "max_tokens": 4096, "truncated": False, "finish_reason": "stop", "forced_close": False, "answer_only_reask": False,
            "answer": answer, "response_text_sha256": None, "latency_seconds": 1.0}
    return rt.validate_line(line)


def score(half, row, arm, f1, ok, iteration=1):
    return rt.validate_line({"schema": rt.SCHEMA, "kind": "score", "ts": TS, "run_id": sp.run_id_for(half, iteration), "attempt_id": sp.attempt_id_for(half, iteration), "row_id": row,
                             "arm": arm, "scorer_id": "official", "scenario_action_exact": ok, "slu_f1": f1})


class Half:
    """One half's roster and trace lines; ``add`` registers a row with lock, aa and four (f1, intent_ok, answer) trajectories."""

    def __init__(self, half=H1, scenarios=("alarm", "music")):
        self.half, self.rows, self.contacts, self.scores, self.scen, self.scenarios = half, [], [], [], {}, scenarios

    def add(self, row, lock, trajectories, aa=None, scenario="alarm", skip_arm=None):
        self.rows.append(row)
        self.scen[row] = scenario
        aa = lock if aa is None else aa
        self.scores += [score(self.half, row, "lock", lock, True), score(self.half, row, "aa", aa, True)]
        for i, (f1, ok, answer) in enumerate(trajectories):
            arm = f"s{i + 1}"
            if arm == skip_arm:
                continue
            self.contacts.append(contact(self.half, row, arm, answer))
            self.scores.append(score(self.half, row, arm, f1, ok))

    def roster(self):
        return {"roster_id": cm.HALF_ROSTER_ID[self.half], "split_identity_hash": "c" * 64, "items": [{"sample_id": r, "audio_seconds": 2.0, "stratum": self.scen[r]} for r in self.rows]}

    def credit(self, **kw):
        return rd.half_credit(self.roster(), self.half, 1, self.contacts, self.scores, **kw)


def flat(lock_f1=0.5, f1s=(0.5, 0.5, 0.5, 0.5)):
    return [(f, True, ans()) for f in f1s]


def small_half():
    h = Half()
    h.add("r1", 0.5, [(1.0, True, ans("alarm", "set", [("date", "monday")])), (0.5, True, ans()), (0.5, True, ans()), (0.0, False, ans("alarm", "query"))])        # positive, +0.5
    h.add("r2", 0.5, flat())                                                                                                                                        # no gain
    h.add("r3", 0.5, [(1.0, True, ans()), (0.5, True, ans()), (0.5, True, ans()), (0.5, True, ans())], aa=0.25)                                                      # lock != aa: not counted
    h.add("r4", 0.5, [(0.75, True, ans("music", "play")), (0.5, True, ans()), (0.5, True, ans()), (0.5, True, ans())], scenario="music")                              # positive, +0.25
    h.add("r5", 0.5, flat(), skip_arm="s4")                                                                                                                         # incomplete: skipped
    return h


# ---- (1) the vote ------------------------------------------------------------------------------------------------------------------------------------


def test_vote_is_plurality_with_ties_to_the_lowest_k_and_entities_need_two_of_four():
    answers = [ans("alarm", "set", [("date", "Monday"), ("time", "5 pm")]), ans("alarm", "query", [("date", "monday")]), ans("alarm", "set", [("time", "5  PM")]),
               ans("alarm", "query", [("place", "home")])]
    vote = rd.vote_answer(answers)
    assert (vote["scenario"], vote["action"], vote["intent_votes"], vote["n"]) == ("alarm", "set", 2, 4)                       # 2-2 tie: the intent seen at the lowest k
    assert vote["entities"] == [{"type": "date", "filler": "Monday"}, {"type": "time", "filler": "5 pm"}]                      # support 2 each (normalised), first-seen text, "place" has 1
    assert rd.vote_answer([ans("a", "b"), ans("a", "c"), ans("a", "c"), ans("a", "b")])["action"] == "b"                         # tie again: the lowest k's intent
    assert rd.vote_answer([ans("a", "b"), ans("a", "c"), ans("a", "c")])["action"] == "c"
    with pytest.raises(rd.ReadError):
        rd.vote_answer([])


# ---- (2) the construction numbers ------------------------------------------------------------------------------------------------------------------


def test_half_credit_counts_positives_best_of_k_gain_and_skipped_rows():
    record = small_half().credit(scorer_id="official")
    assert (record["rows_in_half"], record["rows_usable"], record["rows_skipped"], record["skipped_first"]) == (5, 4, 1, ["r5"])
    assert record["positives"] == 2 and record["positive_rate"] == 0.5                      # r1 and r4; r3 is baseline-unstable; r2 has no gain
    assert record["reasons"] == {"baseline-unstable": 1, "no-gain-over-baseline": 1, "positive": 2}
    best = [0.5, 0.0, 0.5, 0.25]                                                            # max reward per usable row: r1 +0.5, r2 0, r3 +0.5 (unstable but its best still counts), r4 +0.25
    assert record["mean_best_of_k_gain_pp"] == pytest.approx(100 * sum(best) / 4)
    assert [v["row_id"] for v in record["votes"]] == ["r1", "r2", "r3", "r4"] and record["votes"][0]["entities"] == []   # a single date mention has support 1 of 4


def test_half_credit_is_deterministic_and_independent_of_line_order():
    h = small_half()
    first = h.credit(scorer_id="official")
    h.contacts.reverse()
    h.scores.reverse()
    second = h.credit(scorer_id="official")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert json.dumps(first, sort_keys=True) == json.dumps(small_half().credit(scorer_id="official"), sort_keys=True)


def test_rule_survivors_and_exemplar_coverage_per_scenario():
    rules = {"rules": [{"candidate_key": "intent_flip|a|b", "support_count": 4}, {"candidate_key": "slot_missing|alarm|date", "support_count": 3}], "rendered_tokens": 90, "token_measure": "vllm-tokenize"}
    assert rd.rule_survivors(rules) == {"rules": 2, "rendered_tokens": 90, "token_measure": "vllm-tokenize", "by_kind": {"intent_flip": 1, "slot_missing": 1}, "support_counts": [4, 3]}
    assert rd.rule_survivors(None) is None and rd.exemplar_coverage(small_half().roster(), None) is None
    bank = {"by_scenario": {"alarm": {"entries": [{"x": 1}, {"x": 2}]}, "weather": {"entries": [{"x": 3}]}}}
    cov = rd.exemplar_coverage(small_half().roster(), bank)
    assert cov["scenarios"] == 2 and cov["scenarios_covered"] == 1 and cov["scenario_share"] == 0.5
    assert cov["by_scenario"] == {"alarm": {"rows": 4, "exemplars": 2, "covered": True}, "music": {"rows": 1, "exemplars": 0, "covered": False}}
    assert cov["row_share"] == 0.8 and cov["scenarios_in_bank_outside_roster"] == ["weather"]


def row(tp, fp, fn, intent=1.0):
    return {"slu_f1_word_tp": tp, "slu_f1_char_tp": tp, "slu_f1_word_fp": fp, "slu_f1_char_fp": fp, "slu_f1_word_fn": fn, "slu_f1_char_fn": fn, pc.INTENT_FIELD: intent}


def tables(vote_tp, lock_tp, fpfn_vote, fpfn_lock, n=10):
    ids = [f"u{i}" for i in range(n)]
    return ({i: row(vote_tp / n, fpfn_vote / n / 2, fpfn_vote / n / 2) for i in ids}, {i: row(lock_tp / n, fpfn_lock / n / 2, fpfn_lock / n / 2) for i in ids}, ids)


def test_construction_gain_is_pooled_slu_f1_of_the_vote_minus_lock_in_pp():
    vote, lock, ids = tables(91.0, 90.0, 18.0, 20.0)            # pooled counts double (word + char): F1(vote) = 2*182/(2*182+36)= .91, F1(lock) = 360/(360+40) = .9
    gain = rd.construction_gain(vote, lock, ids)
    assert gain["gain_pp"] == pytest.approx(1.0, abs=1e-9) and gain["rows"] == 10
    assert rd.construction_gain(None, lock, ids)["gain_pp"] is None and rd.construction_gain(vote, lock, [])["gain_pp"] is None
    with pytest.raises(rd.ReadError, match="no vote or lock count row"):
        rd.construction_gain(vote, lock, ids + ["missing"])


# ---- (3) the stop rule ------------------------------------------------------------------------------------------------------------------------------


def test_the_construction_gate_passes_at_plus_one_pp_and_stops_at_plus_nine_tenths():
    assert rd.construction_gate(1.0)["word"] == rd.WORD_CONTINUE and rd.construction_gate(1.0)["continue"] is True
    assert rd.construction_gate(1.0000004)["word"] == rd.WORD_CONTINUE                  # float error never costs the pass (6 decimals)
    assert rd.construction_gate(0.9)["word"] == rd.WORD_STOP and rd.construction_gate(0.9)["stop"] is True
    assert rd.construction_gate(0.999999)["word"] == rd.WORD_STOP
    assert rd.construction_gate(-3.0)["word"] == rd.WORD_STOP
    undetermined = rd.construction_gate(None)
    assert undetermined["word"] == rd.WORD_UNDETERMINED and undetermined["continue"] is None and undetermined["stop"] is None


def test_the_gate_fires_on_a_computed_gain_from_synthetic_tables():
    for lock_fpfn, vote_fpfn, word in ((20.0, 18.0, rd.WORD_CONTINUE), (20.0, 18.1, rd.WORD_STOP)):
        vote, lock, ids = tables(91.0, 90.0, vote_fpfn, lock_fpfn)
        gain = rd.construction_gain(vote, lock, ids)["gain_pp"]
        assert rd.construction_gate(gain)["word"] == word, gain


def test_the_stop_decision_limits_iterations_and_stops_on_nothing_accepted():
    ok = rd.stop_decision(1, 1.0, True)
    assert ok["stop"] is False and ok["next_iteration_allowed"] is True and ok["reasons"] == []
    fam_stop = rd.stop_decision(1, 0.9, True)
    assert fam_stop["stop"] is True and fam_stop["next_iteration_allowed"] is False and "the family stops at iteration 1" in fam_stop["reasons"][0]
    assert "the family stops" not in rd.stop_decision(2, 0.9, True)["reasons"][0]
    assert rd.stop_decision(3, 5.0, True)["stop"] is True and "last of at most 3" in rd.stop_decision(3, 5.0, True)["reasons"][0]
    assert rd.stop_decision(2, 2.0, False)["reasons"] == ["the iteration accepted nothing"]
    unknown = rd.stop_decision(1, None, True)
    assert unknown["stop"] is False and unknown["undetermined"] is True and unknown["next_iteration_allowed"] is False
    assert rd.stop_decision(1, 1.5, None)["undetermined"] is True


# ---- (4) the Fold B words ------------------------------------------------------------------------------------------------------------------------------


def iv(point, lo, hi):
    return {"point_pp": point, "ci95_lo_pp": lo, "ci95_hi_pp": hi}


def contrast(slu=(1.6, 0.4, 2.8), intent=(0.2, -0.3, 0.7), per_row=(1.2, 0.3, 2.1), aa=0.3):
    return {"delivering": "lock", "ci_definition": "widened", "levels": {}, "versus_delivering": {tf.T_ARM: {"slu_f1": iv(*slu), "intent": iv(*intent), "per_row_slu_f1": iv(*per_row)},
                                                                                                   tf.TWIN: {"slu_f1": iv(aa, -1, 1), "intent": iv(0, -1, 1), "per_row_slu_f1": iv(0, -1, 1)}}}


@pytest.mark.parametrize("kwargs,word,failed", [
    ({}, "NOMINATED", []),
    ({"slu": (1.48, 0.4, 2.8)}, "NEUTRAL", ["slu_point_at_or_above_floor"]),
    ({"slu": (1.6, -0.1, 2.8)}, "NEUTRAL", ["slu_ci_lo_above_zero"]),
    ({"per_row": (-0.5, -1.2, 0.2)}, "NEUTRAL", ["per_row_slu_ci_lo_above_zero"]),          # pooled up, typical row down: emptying is withheld
    ({"intent": (-0.1, -0.5, 0.3)}, "NEUTRAL", ["intent_point_non_negative"]),
    ({"intent": (0.0, -0.7, 0.3)}, "NEUTRAL", ["intent_guard_ci_lo"]),
    ({"slu": (-3.0, -4.0, -1.6)}, "HARM", None),
])
def test_bank_arm_words_apply_the_nominated_rule_over_the_widened_ci(kwargs, word, failed):
    entry = rd.bank_arm_words(contrast(**kwargs), [tf.T_ARM])["words"][tf.T_ARM]
    assert entry["word"] == word
    if failed is not None:
        assert entry["failed"] == failed


def test_bank_arm_words_refuse_a_contrast_without_the_arm_or_the_per_row_delta():
    c = contrast()
    with pytest.raises(rd.ReadError, match="no paired contrast"):
        rd.bank_arm_words(c, ["nope"])
    del c["versus_delivering"][tf.T_ARM]["per_row_slu_f1"]
    with pytest.raises(rd.ReadError, match="no per_row_slu_f1"):
        rd.bank_arm_words(c, [tf.T_ARM])


# ---- (5) the iteration verdict -----------------------------------------------------------------------------------------------------------------------


def verdict(gain, **kwargs):
    c = contrast(**{k: v for k, v in kwargs.items() if k != "twin_word"})
    return rd.iteration_verdict(rd.bank_arm_words(c, [tf.T_ARM]), rd.twin_admissibility(c) if "twin_word" not in kwargs else {"word": kwargs["twin_word"]}, gain)


def test_an_iteration_is_accepted_only_when_every_condition_holds():
    assert verdict(1.0)["word"] == rd.WORD_ACCEPTED
    assert verdict(0.9)["word"] == rd.WORD_NOT_ACCEPTED and verdict(0.9)["conditions"]["construction_gain_gate"] is False
    assert verdict(None)["word"] == rd.WORD_UNDETERMINED
    assert verdict(2.0, aa=1.21)["word"] == rd.WORD_NOT_ACCEPTED and verdict(2.0, aa=1.21)["conditions"]["twin_admissible"] is False
    assert verdict(2.0, aa=1.2)["word"] == rd.WORD_ACCEPTED                                              # exactly 1.2 pp is admissible
    assert verdict(2.0, slu=(1.2, 0.4, 2.0))["word"] == rd.WORD_NOT_ACCEPTED                            # below the SLU floor: not nominated
    assert verdict(None, slu=(1.2, 0.4, 2.0))["word"] == rd.WORD_NOT_ACCEPTED                            # a refusal beats an unknown gain
    c = contrast(per_row=(-0.1, -1.0, 0.6))
    assert rd.iteration_verdict(rd.bank_arm_words(c, [tf.T_ARM]), rd.twin_admissibility(c), 2.0)["conditions"]["per_row_mean_delta_non_negative"] is False


def synthetic_fold_b(n=120):
    ids = [f"u{i:03d}" for i in range(n)]
    lock = {i: row(3, 1, 1) for i in ids}
    better = {i: (row(4, 0, 1) if k % 3 == 0 else row(3, 1, 1)) for k, i in enumerate(ids)}
    scored = {tf.DELIVERING: lock, tf.TWIN: dict(lock), tf.REPLICATE: dict(lock), tf.T_ARM: better}
    return ids, scored, (lambda sid: sid)


def test_foldb_read_reuses_the_paired_machinery_and_is_deterministic():
    ids, scored, cluster = synthetic_fold_b()
    first = rd.foldb_read(scored, ids, cluster, iteration=1, gain_pp=1.2, draws=300)
    second = rd.foldb_read(scored, ids, cluster, iteration=1, gain_pp=1.2, draws=300)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    t = first["contrast"]["versus_delivering"][tf.T_ARM]
    assert t["slu_f1"]["point_pp"] > 0 and t["per_row_slu_f1"]["point_pp"] > 0 and t["slu_f1"]["widening_pp"] == 0.0         # identical copies: no widening
    assert first["contrast"]["noise_model"]["copies"] == ["lock", "aa", "lock-rep2"] and first["twin_admissibility"]["word"] == rd.ADMISSIBLE
    assert first["arm_words"]["words"][tf.T_ARM]["word"] == "NOMINATED" and first["verdict"]["word"] == rd.WORD_ACCEPTED and first["stop"]["next_iteration_allowed"] is True
    stopped = rd.foldb_read(scored, ids, cluster, iteration=1, gain_pp=0.9, draws=300)
    assert stopped["verdict"]["word"] == rd.WORD_NOT_ACCEPTED and stopped["stop"]["stop"] is True and stopped["arm_words"] == first["arm_words"]
    unknown = rd.foldb_read(scored, ids, cluster, iteration=1, gain_pp=None, draws=300)
    assert unknown["verdict"]["word"] == rd.WORD_UNDETERMINED and unknown["stop"]["undetermined"] is True


def test_the_replicate_spread_widens_the_bank_arm_interval():
    ids, scored, cluster = synthetic_fold_b()
    scored[tf.REPLICATE] = {i: (row(4, 0, 1) if k % 10 == 0 else row(3, 1, 1)) for k, i in enumerate(ids)}                  # the replicate drifts from lock
    widened = rd.foldb_read(scored, ids, cluster, iteration=1, gain_pp=1.2, draws=300)["contrast"]["versus_delivering"][tf.T_ARM]["slu_f1"]
    assert widened["widening_pp"] > 0 and widened["ci95_lo_pp"] < widened["bootstrap_ci95_lo_pp"]


def test_foldb_read_refuses_a_missing_column():
    ids, scored, cluster = synthetic_fold_b()
    del scored[tf.REPLICATE]
    with pytest.raises(rd.ReadError, match="missing scored columns"):
        rd.foldb_read(scored, ids, cluster, iteration=1, gain_pp=1.0, draws=50)


# ---- the recipe ----------------------------------------------------------------------------------------------------------------------------------------


def test_the_recipe_loads_and_a_moved_rule_is_refused(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_bytes(rails.recipe_bytes())
    assert rd.load_recipe(path)["read_run_id"] == tf.READ_RUN_ID
    doc = rails.recipe_document()
    doc["stop_rule"]["construction_gain_pp"] = 0.5
    doc["fold_b_read"]["primary"]["bootstrap"]["seed"] = 1
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(rd.ReadError, match="stop_rule is not \\+1.0 pp / 3 iterations") as info:
        rd.load_recipe(path)
    assert "bootstrap seed/draws" in str(info.value)


# ---- (6) the CLI over a synthetic data root ----------------------------------------------------------------------------------------------------------


def scores_file(rows):
    return {rd.SCORER: {"per_sample": [{"sample_id": sid, **values} for sid, values in rows.items()]}}


def build_data_root(root: Path, *, with_columns: bool, vote_better: bool = True) -> None:
    """A synthetic data root: the two construction traces over the REAL half rosters (every 5th row a positive), optional lock / vote count columns, and the sealed Fold B run."""

    for half in cm.HALVES:
        roster = json.loads((REPO / cm.HALF_ROSTER_RELPATH[half]).read_text(encoding="utf-8"))
        h = Half(half)
        for k, item in enumerate(roster["items"]):
            positive = k % 5 == 0
            best = 1.0 if positive else 0.5
            h.add(item["sample_id"], 0.5, [(best, True, ans("alarm", "set", [("date", "monday")])), (0.5, True, ans("alarm", "set", [("date", "monday")])), (0.5, True, ans()), (0.5, True, ans())],
                  scenario=item["stratum"])
        directory = root / "analysis" / tf.trace_read_id(1, half.split("-")[1])
        directory.mkdir(parents=True)
        (directory / "rollout-trace.jsonl").write_text("".join(rt.dumps_line(line) + "\n" for line in h.contacts + h.scores), encoding="utf-8")
        if with_columns:
            lock = {item["sample_id"]: row(3, 1, 1) for item in roster["items"]}
            vote = {item["sample_id"]: (row(4, 0, 1) if vote_better else row(3, 1, 1)) for item in roster["items"]}
            columns = directory / "columns"
            columns.mkdir()
            (columns / rd.COLUMN_SCORE_NAME.format(arm="lock", scorer=rd.SCORER)).write_text(json.dumps(scores_file(lock)), encoding="utf-8")
            (columns / rd.COLUMN_SCORE_NAME.format(arm="vote", scorer=rd.SCORER)).write_text(json.dumps(scores_file(vote)), encoding="utf-8")


def run_cli(argv):
    return rd.main(argv)


@pytest.mark.parametrize("with_columns", [False, True])
def test_the_construction_stage_writes_write_once_files_and_is_deterministic(tmp_path, capsys, with_columns):
    recipe = tmp_path / "recipe.json"
    recipe.write_bytes(rails.recipe_bytes())
    outputs = []
    for name in ("a", "b"):
        root = tmp_path / name
        build_data_root(root, with_columns=with_columns)
        code = run_cli(["--stage", "construction", "--iteration", "1", "--data-root", str(root), "--recipe", str(recipe)])
        assert code == 0
        written = json.loads(capsys.readouterr().out)["written"]
        outputs.append({Path(p).name: Path(p).read_bytes() for p in written})
    assert outputs[0] == outputs[1] and sorted(outputs[0]) == ["construction-i1.json", "vote-answers-i1-H1.jsonl", "vote-answers-i1-H2.jsonl"]
    result = json.loads(outputs[0]["construction-i1.json"])
    assert result["schema"] == rd.CONSTRUCTION_SCHEMA and result["halves"][H1]["rows_usable"] == 313 and result["halves"][H2]["rows_usable"] == 312
    assert result["halves"][H1]["positive_rate"] == pytest.approx(63 / 313, abs=1e-6)             # rows k = 0, 5, ..., 310 -> 63 positives
    assert result["iteration_totals"]["rows_usable"] == 625 and result["union_bank"] is None
    assert len(outputs[0]["vote-answers-i1-H1.jsonl"].decode().splitlines()) == 313
    if with_columns:
        assert result["construction_gain"]["gain_pp"] > 1.0 and result["stop"]["gate"]["word"] == rd.WORD_CONTINUE
    else:
        assert result["construction_gain"]["gain_pp"] is None and result["stop"]["gate"]["word"] == rd.WORD_UNDETERMINED
    root = tmp_path / "a"
    assert run_cli(["--stage", "construction", "--iteration", "1", "--data-root", str(root), "--recipe", str(recipe)]) == 6          # write-once
    assert "exists (write-once)" in capsys.readouterr().err


def test_the_cli_refuses_a_missing_trace_a_moved_recipe_and_a_bad_iteration(tmp_path, capsys):
    recipe = tmp_path / "recipe.json"
    recipe.write_bytes(rails.recipe_bytes())
    assert run_cli(["--stage", "construction", "--data-root", str(tmp_path / "empty"), "--recipe", str(recipe)]) == 2
    assert "missing sealed trace" in capsys.readouterr().err
    doc = rails.recipe_document()
    doc["twin"] = "bb"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    assert run_cli(["--stage", "construction", "--data-root", str(tmp_path / "empty"), "--recipe", str(bad)]) == 2
    assert "recipe" in capsys.readouterr().err
    assert run_cli(["--stage", "construction", "--iteration", "4", "--data-root", str(tmp_path / "empty"), "--recipe", str(recipe)]) == 2
    assert "iteration must be in 1..3" in capsys.readouterr().err


def test_the_read_stage_over_the_real_fold_b_roster_writes_the_verdicts(tmp_path, capsys):
    recipe = tmp_path / "recipe.json"
    recipe.write_bytes(rails.recipe_bytes())
    root = tmp_path / "data"
    build_data_root(root, with_columns=True)
    cell = tf.CELLS["fb"]
    paths = base.run_paths(root, cell["run"], cell["attempt"])
    paths["dir"].mkdir(parents=True)
    paths["trace"].write_text("{}\n", encoding="utf-8")
    paths["outputs"].write_text("{}\n", encoding="utf-8")
    paths["manifest"].write_text(json.dumps({"status": "COMPLETED"}), encoding="utf-8")
    ids, _, _ = base.roster_maps(REPO / tf.ROSTERS["fb"])
    columns = root / rails.ANALYSIS_ROOT / "columns"
    columns.mkdir(parents=True)
    lock = {i: row(3, 1, 1) for i in ids}
    arms = {tf.DELIVERING: lock, tf.TWIN: dict(lock), tf.REPLICATE: dict(lock), tf.T_ARM: {i: (row(4, 0, 1) if k % 3 == 0 else row(3, 1, 1)) for k, i in enumerate(ids)}}
    for arm, rows in arms.items():
        (columns / rd.COLUMN_SCORE_NAME.format(arm=arm, scorer=rd.SCORER)).write_text(json.dumps(scores_file(rows)), encoding="utf-8")
    assert run_cli(["--stage", "construction", "--data-root", str(root), "--recipe", str(recipe)]) == 0
    capsys.readouterr()
    assert run_cli(["--stage", "read", "--data-root", str(root), "--recipe", str(recipe), "--draws", "200"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["verdict"] == rd.WORD_ACCEPTED and summary["within_run"] == "NOMINATED" and summary["twin"] == rd.ADMISSIBLE and summary["next_iteration_allowed"] is True
    verdicts = json.loads((root / rails.ANALYSIS_ROOT / "tfrl-v2-verdicts-i1.json").read_text(encoding="utf-8"))
    assert verdicts["schema"] == rd.VERDICTS_SCHEMA and verdicts["verdict"]["word"] == rd.WORD_ACCEPTED and verdicts["construction"]["gain"]["gain_pp"] > 1.0
    for name in ("recipe", "roster", "manifest", "trace", "outputs", "construction_read", "column_scores:t-v2"):
        assert len(verdicts["inputs"][name]["sha256"]) == 64
    assert run_cli(["--stage", "read", "--data-root", str(root), "--recipe", str(recipe), "--draws", "200"]) == 6
    assert "exists (write-once)" in capsys.readouterr().err
