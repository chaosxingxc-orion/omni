"""Noise model M1: ``scripts/analysis/paired_ci.py`` (the replicate variance, the widened CI95 and per-row F1).

Model-free synthetic fixtures. The proofs: (1) the F1 helpers equal the scorer's pooled F1 and define the per-row F1 (empty rows score 1.0, false fills 0.0); (2) the pairwise
deltas of three request-identical copies and the replicate variance; (3) the widening formula: identical replicates widen by exactly zero, replicates 1.1 pp apart widen by
the stated amount, the widened interval is symmetric around the bootstrap one and keeps the point estimate; (4) the noise-adjusted contrast over real row tables (copies keep
their raw intervals, per-row delta has its own interval); (5) a lever that raises POOLED F1 while the per-row F1 falls is not NOMINATED.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "scripts" / "flights" / "direct", REPO / "scripts" / "flights" / "tfrl", REPO / "scripts" / "flights" / "batch",
               REPO / "scripts" / "analysis", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import l338_port_family as pf  # noqa: E402
import paired_ci as pc  # noqa: E402
import slurp_direct_fold_b_core_read as base  # noqa: E402
import slurp_l338_port_column_read as port  # noqa: E402
import slurp_nxt_thinking_policy_column_read as npc  # noqa: E402

Z = 1.959964


def row(tp, fp, fn, intent=1.0):
    """A per-row score table entry: the word counts equal the char counts, so the pooled counts are twice the given ones."""

    return {"slu_f1_word_tp": tp, "slu_f1_char_tp": tp, "slu_f1_word_fp": fp, "slu_f1_char_fp": fp, "slu_f1_word_fn": fn, "slu_f1_char_fn": fn, pc.INTENT_FIELD: intent}


def iv(point, lo, hi):
    return {"point_pp": point, "ci95_lo_pp": lo, "ci95_hi_pp": hi, "ci95_half_width_pp": (hi - lo) / 2.0, "draws": 100}


# --------------------------------------------------------------------------- (1) F1 helpers


def test_the_constants_equal_the_historical_readers():
    assert pc.SLU_GROUPS == base.SLU_GROUPS and pc.INTENT_FIELD == base.INTENT_FIELD
    assert pc.Z95 == pytest.approx(Z, abs=1e-6)


def test_pooled_f1_equals_the_scorers_and_the_percentile_interval_equals_the_readers():
    rows = {f"s{i}": row(i % 4, i % 3, (i + 1) % 5) for i in range(30)}
    ids = list(rows)
    assert pc.pooled_slu_f1(rows, ids) == pytest.approx(base._pooled(rows, ids))
    assert pc.pooled_slu_f1(rows, ids + ids[:5]) == pytest.approx(base._pooled(rows, ids + ids[:5]))           # repeated ids count again
    replicates = [0.01 * i for i in range(-50, 51)]
    mine, theirs = pc._interval(replicates, 0.004), base.interval(replicates, 0.004)
    assert {k: mine[k] for k in theirs} == pytest.approx(theirs)


def test_per_row_f1_definition_and_the_empty_row_conventions():
    assert pc.row_f1(row(3, 1, 1)) == pytest.approx(2 * 6 / (2 * 6 + 2 + 2))          # word + char counts summed: tp 6, fp 2, fn 2
    assert pc.row_f1(row(0, 0, 0)) == 1.0                                              # no gold and no prediction: a correctly empty row
    assert pc.row_f1(row(0, 2, 0)) == 0.0                                              # a false fill on empty gold
    assert pc.row_f1(row(0, 0, 2)) == 0.0                                              # an emptied row that had gold
    with pytest.raises(pc.PairedCiError, match="finite non-negative"):
        pc.row_f1(row(-1, 0, 0))
    rows = {"a": row(1, 0, 0), "b": row(0, 0, 0), "c": row(0, 1, 1)}
    assert pc.per_row_f1_mean(rows, ["a", "b", "c"]) == pytest.approx((1.0 + 1.0 + 0.0) / 3)
    assert math.isnan(pc.per_row_f1_mean(rows, []))


# --------------------------------------------------------------------------- (2) the pairwise deltas and the replicate variance


def test_three_copies_give_three_pairwise_deltas_in_a_fixed_order():
    levels = {"lock": 72.0, "aa": 73.1, "lock-rep2": 71.5}
    deltas = pc.pairwise_deltas(levels, ["lock", "aa", "lock-rep2"])
    assert [d["pair"] for d in deltas] == [["aa", "lock"], ["lock-rep2", "lock"], ["lock-rep2", "aa"]]
    assert [round(d["delta_pp"], 6) for d in deltas] == [1.1, -0.5, -1.6]
    two = pc.pairwise_deltas({"lock": 72.0, "aa": 73.1}, ["lock", "aa"])
    assert len(two) == 1 and pc.replicate_variance(two)["variance_pp2"] == pytest.approx(1.21) and pc.replicate_variance(two)["n_pairs"] == 1


def test_the_pairwise_delta_inputs_are_refused_by_name():
    levels = {"lock": 72.0, "aa": 73.0, "lock-rep2": 72.5}
    with pytest.raises(pc.PairedCiError, match="starting with 'lock'"):
        pc.pairwise_deltas(levels, ["aa", "lock", "lock-rep2"])
    with pytest.raises(pc.PairedCiError, match="two or more distinct"):
        pc.pairwise_deltas(levels, ["lock"])
    with pytest.raises(pc.PairedCiError, match="two or more distinct"):
        pc.pairwise_deltas(levels, ["lock", "aa", "aa"])
    with pytest.raises(pc.PairedCiError, match="no level"):
        pc.pairwise_deltas({"lock": 72.0, "aa": 73.0}, ["lock", "aa", "lock-rep2"])
    with pytest.raises(pc.PairedCiError, match="no pairwise delta"):
        pc.replicate_variance([])


def test_the_replicate_variance_is_the_mean_squared_pairwise_delta():
    levels = {"lock": 72.0, "aa": 73.1, "lock-rep2": 72.0}
    var = pc.replicate_variance(pc.pairwise_deltas(levels, ["lock", "aa", "lock-rep2"]))
    assert var["n_pairs"] == 3 and var["variance_pp2"] == pytest.approx((1.1 ** 2 + 0.0 + 1.1 ** 2) / 3) and var["rms_pp"] == pytest.approx(math.sqrt(0.8066666666))


# --------------------------------------------------------------------------- (3) the widening


def test_identical_replicates_give_zero_widening_and_the_interval_is_unchanged():
    out = pc.widen_interval(iv(1.0, -0.25, 2.25), 0.0)
    assert out["widening_pp"] == 0.0 and out["ci95_lo_pp"] == -0.25 and out["ci95_hi_pp"] == 2.25 and out["point_pp"] == 1.0
    assert out["bootstrap_ci95_lo_pp"] == -0.25 and out["bootstrap_ci95_hi_pp"] == 2.25 and out["replicate_variance_pp2"] == 0.0


def test_replicates_one_point_one_pp_apart_widen_by_the_stated_amount():
    """Copies lock 72.0 / aa 73.1 / lock-rep2 72.0: deltas +1.1, 0.0, -1.1 -> V_rep = 0.806667 pp^2; a bootstrap CI95 of [-1.0, +1.0] has SE_b = 0.510213; the total SE is
    sqrt(0.510213^2 + 0.806667) = 1.032949 and each side widens by z (SE_t - SE_b) = 1.024544 pp: the reported CI95 is [-2.024544, +2.024544]."""

    var = pc.replicate_variance(pc.pairwise_deltas({"lock": 72.0, "aa": 73.1, "lock-rep2": 72.0}, ["lock", "aa", "lock-rep2"]))["variance_pp2"]
    assert var == pytest.approx(0.806667, abs=1e-6)
    out = pc.widen_interval(iv(0.3, -1.0, 1.0), var)
    assert out["bootstrap_se_pp"] == pytest.approx(0.510213, abs=1e-6) and out["total_se_pp"] == pytest.approx(1.032949, abs=1e-6)
    assert out["widening_pp"] == pytest.approx(1.024544, abs=1e-6)
    assert out["ci95_lo_pp"] == pytest.approx(-2.024544, abs=1e-6) and out["ci95_hi_pp"] == pytest.approx(2.024544, abs=1e-6)
    assert out["point_pp"] == 0.3 and out["ci95_half_width_pp"] == pytest.approx(2.024544, abs=1e-6)
    # symmetric: the lower side moves down by exactly the amount the upper side moves up, and the bootstrap interval is kept
    assert (out["bootstrap_ci95_lo_pp"] - out["ci95_lo_pp"]) == pytest.approx(out["ci95_hi_pp"] - out["bootstrap_ci95_hi_pp"])
    # the formula in the docstring, restated independently: sqrt-sum of variances
    assert out["widening_pp"] == pytest.approx(Z * (math.sqrt((2.0 / (2 * Z)) ** 2 + var) - 2.0 / (2 * Z)), abs=1e-5)


def test_one_replicate_pair_of_1_1_pp_widens_more_than_the_mean_of_three_pairs():
    one = pc.widen_interval(iv(0.0, -1.0, 1.0), 1.1 ** 2)["widening_pp"]
    three = pc.widen_interval(iv(0.0, -1.0, 1.0), (1.1 ** 2 + 0.0 + 1.1 ** 2) / 3)["widening_pp"]
    assert one == pytest.approx(1.376587, abs=1e-6) and three < one


def test_the_widening_is_monotone_in_the_variance_and_refuses_malformed_input():
    widths = [pc.widen_interval(iv(0, -1, 1), v)["widening_pp"] for v in (0.0, 0.1, 0.5, 1.0, 4.0)]
    assert widths == sorted(widths) and widths[0] == 0.0
    with pytest.raises(pc.PairedCiError, match="non-negative"):
        pc.widen_interval(iv(0, -1, 1), -0.1)
    with pytest.raises(pc.PairedCiError, match="non-negative"):
        pc.widen_interval(iv(0, -1, 1), float("nan"))
    with pytest.raises(pc.PairedCiError, match="inverted"):
        pc.widen_interval(iv(0, 1, -1), 0.2)


# --------------------------------------------------------------------------- (4) the noise-adjusted contrast over row tables


def table(n, make):
    return {f"s{i}": make(i) for i in range(n)}


def contrast_of(scored, ids, *, draws=300, seed=20261018):
    return npc.paired_contrasts(scored, ids, lambda sid: sid, seed=seed, draws=draws, delivering="lock")


def adjusted(scored, ids, *, copies=("lock", "aa", "lock-rep2"), draws=300, seed=20261018):
    return pc.noise_adjusted_contrast(scored, ids, lambda sid: sid, contrast_of(scored, ids, draws=draws, seed=seed), copies=copies, seed=seed, draws=draws)


def test_identical_copies_leave_every_interval_unchanged_and_the_copies_keep_theirs():
    n = 60
    ids = [f"s{i}" for i in range(n)]
    lock = table(n, lambda i: row(3 + (i % 3), 1, i % 2))
    arm = table(n, lambda i: row(4 + (i % 3), 1, 0))
    scored = {"lock": lock, "aa": dict(lock), "lock-rep2": dict(lock), "l1-support": arm}
    out = adjusted(scored, ids)
    raw = contrast_of(scored, ids)
    assert out["noise_model"]["metrics"]["slu_f1"]["variance_pp2"] == 0.0 and out["noise_model"]["metrics"]["per_row_slu_f1"]["variance_pp2"] == 0.0
    for arm_name, entry in out["versus_delivering"].items():
        assert entry["slu_f1"]["widening_pp"] == 0.0
        assert entry["slu_f1"]["ci95_lo_pp"] == raw["versus_delivering"][arm_name]["slu_f1"]["ci95_lo_pp"]
        assert entry["intent"]["ci95_hi_pp"] == raw["versus_delivering"][arm_name]["intent"]["ci95_hi_pp"]
    assert out["versus_delivering"]["aa"]["slu_f1"]["point_pp"] == 0.0 and out["versus_delivering"]["aa"]["per_row_slu_f1"]["point_pp"] == 0.0
    assert out["levels"]["lock"]["per_row_slu_f1_pct"] == pytest.approx(100 * pc.per_row_f1_mean(lock, ids)) and out["levels"]["lock"]["slu_f1_pct"] == raw["levels"]["lock"]["slu_f1_pct"]


def test_differing_copies_widen_arms_but_not_the_copies_and_the_record_states_the_pairs():
    n = 80
    ids = [f"s{i}" for i in range(n)]
    lock = table(n, lambda i: row(3 + (i % 3), 1, i % 2))
    aa = table(n, lambda i: row(3 + (i % 3) + (1 if i % 8 == 0 else 0), 1, i % 2))           # a few rows better: the twin offset
    rep = table(n, lambda i: row(3 + (i % 3) - (1 if i % 10 == 0 else 0), 1, i % 2))
    arm = table(n, lambda i: row(5 + (i % 3), 1, 0))
    scored = {"lock": lock, "aa": aa, "lock-rep2": rep, "l1-support": arm}
    out = adjusted(scored, ids)
    raw = contrast_of(scored, ids)
    var = out["noise_model"]["metrics"]["slu_f1"]
    assert var["n_pairs"] == 3 and [p["pair"] for p in var["pairs"]] == [["aa", "lock"], ["lock-rep2", "lock"], ["lock-rep2", "aa"]]
    assert var["variance_pp2"] > 0
    entry = out["versus_delivering"]["l1-support"]["slu_f1"]
    assert entry["widening_pp"] > 0 and entry["ci95_lo_pp"] < raw["versus_delivering"]["l1-support"]["slu_f1"]["ci95_lo_pp"]
    assert entry["ci95_hi_pp"] > raw["versus_delivering"]["l1-support"]["slu_f1"]["ci95_hi_pp"] and entry["point_pp"] == raw["versus_delivering"]["l1-support"]["slu_f1"]["point_pp"]
    # the widening is the same number for every non-copy arm of the run and is exactly the formula
    assert entry["replicate_variance_pp2"] == var["variance_pp2"]
    expected = pc.widen_interval(raw["versus_delivering"]["l1-support"]["slu_f1"], var["variance_pp2"])
    assert entry["ci95_lo_pp"] == pytest.approx(expected["ci95_lo_pp"]) and entry["widening_pp"] == pytest.approx(expected["widening_pp"])
    # the copies ARE the noise: they keep their bootstrap intervals
    twin = out["versus_delivering"]["aa"]["slu_f1"]
    assert twin["widening_pp"] == 0.0 and twin["ci95_lo_pp"] == raw["versus_delivering"]["aa"]["slu_f1"]["ci95_lo_pp"]
    assert out["versus_delivering"]["l1-support"]["per_row_slu_f1"]["replicate_variance_pp2"] == out["noise_model"]["metrics"]["per_row_slu_f1"]["variance_pp2"]
    assert "widened by the replicate variance" in out["ci_definition"] and out["noise_model"]["formula"] == pc.FORMULA


def test_the_noise_model_refuses_a_missing_copy():
    n = 10
    ids = [f"s{i}" for i in range(n)]
    lock = table(n, lambda i: row(3, 1, 1))
    with pytest.raises(pc.PairedCiError, match="no scored column"):
        pc.noise_model({"lock": lock, "aa": lock}, ids, copies=("lock", "aa", "lock-rep2"))


# --------------------------------------------------------------------------- (5) pooled up, per-row down: not NOMINATED


def emptying_fixture(n_big=40, n_small=360):
    """Big rows (100 gold tokens) the arm answers better, small rows (2 gold tokens) the arm EMPTIES: the pooled micro-F1 rises, the typical (per-row) F1 collapses."""

    ids = [f"b{i}" for i in range(n_big)] + [f"m{i}" for i in range(n_small)]
    lock, arm = {}, {}
    for sid in ids:
        if sid.startswith("b"):
            lock[sid], arm[sid] = row(25, 0, 25), row(50, 0, 0)
        else:
            lock[sid], arm[sid] = row(1, 0, 1), row(0, 0, 2)
    return ids, lock, arm


def test_a_lever_that_raises_pooled_f1_but_lowers_per_row_f1_is_not_nominated():
    ids, lock, arm = emptying_fixture()
    scored = {"lock": lock, "aa": dict(lock), "lock-rep2": dict(lock), "l1-support": arm}
    out = adjusted(scored, ids, draws=400)
    entry = out["versus_delivering"]["l1-support"]
    assert entry["slu_f1"]["point_pp"] > 5.0 and entry["slu_f1"]["ci95_lo_pp"] > 1.49                    # pooled: a clear gain, over the floor
    assert entry["per_row_slu_f1"]["point_pp"] < -30.0 and entry["per_row_slu_f1"]["ci95_hi_pp"] < 0       # per-row: a collapse
    words = port.port_arm_words(out)["words"]["l1-support"]
    assert words["hist_word"] == "NOMINATED"                                                             # the pooled-only rule would have nominated it
    assert words["word"] != "NOMINATED" and words["nominated_withheld_by"] == ["per_row_primary_ci_lo_above_zero", "per_row_slu_no_resolved_loss"] and words["word"] == "NEUTRAL"
    assert words["per_row_slu_delta_pp"] == entry["per_row_slu_f1"]["point_pp"]


def test_a_lever_that_improves_pooled_and_per_row_f1_is_nominated():
    n = 400
    ids = [f"s{i}" for i in range(n)]
    lock = table(n, lambda i: row(3, 1, 3))
    arm = table(n, lambda i: row(5, 1, 1))
    out = adjusted({"lock": lock, "aa": dict(lock), "lock-rep2": dict(lock), "l1-support": arm}, ids, draws=200)
    words = port.port_arm_words(out)["words"]["l1-support"]
    assert words["word"] == "NOMINATED" and words["nominated_withheld_by"] == [] and words["conditions"]["per_row_primary_ci_lo_above_zero"] is True


def test_replicate_noise_can_withhold_a_nomination_the_interval_alone_would_make():
    """The same arm and rows; only the copies differ. Identical copies: NOMINATED. Copies that disagree by about a point: the widened CI95 lower bound drops under the floor."""

    n = 400
    ids = [f"s{i}" for i in range(n)]
    lock = table(n, lambda i: row(3, 1, 3))
    arm = table(n, lambda i: row(4, 1, 2))
    quiet = adjusted({"lock": lock, "aa": dict(lock), "lock-rep2": dict(lock), "l1-support": arm}, ids, draws=300)
    noisy_aa = table(n, lambda i: row(4, 1, 3) if i % 4 == 0 else row(3, 1, 3))                 # a twin about a point above lock
    noisy = adjusted({"lock": lock, "aa": noisy_aa, "lock-rep2": dict(lock), "l1-support": arm}, ids, draws=300)
    assert port.port_arm_words(quiet)["words"]["l1-support"]["word"] == "NOMINATED"
    widened = noisy["versus_delivering"]["l1-support"]["slu_f1"]
    assert widened["widening_pp"] > 0.3 and widened["ci95_lo_pp"] < quiet["versus_delivering"]["l1-support"]["slu_f1"]["ci95_lo_pp"] - 0.3
