#!/usr/bin/env python3
"""paired_ci.py -- the noise-adjusted paired contrast of a same-flight column (noise model M1, plan 2026-10-30 task T2).

Model-free and import-only (no CLI). Used by ``slurp_l338_port_column_read.py``; every function is pure over the per-row result tables the official scorer writes
(``{sample_id: {field: number}}``) and over the paired-contrast record of ``slurp_nxt_thinking_policy_column_read.paired_contrasts``.

WHY. The historical CI95 of an arm delta (arm minus lock, paired per row) is an utterance-cluster bootstrap over the roster: it carries the sampling noise of WHICH
utterances were drawn and none of the decoding noise of the run (the lock is re-measured in every run: 64.00 / 64.51 / 64.45; the twin offset swung -0.49 / +0.32 /
-0.37). The port flight therefore carries three REQUEST-IDENTICAL copies of the lock, ``lock`` itself, the twin ``aa`` and the replicate ``lock-rep2``; their three
pairwise deltas per metric measure the decoding noise directly.

THE FORMULA (one place; the recipe ``port_extensions.noise_model`` states the same). For a metric m (pooled SLU-F1, intent accuracy, per-row mean SLU-F1; every value in
percentage points) and the copies C = (lock, aa, lock-rep2):

  pairwise deltas       D_ab = m(a) - m(b) for the pairs (aa, lock), (lock-rep2, lock), (lock-rep2, aa)             (k = 3 pairs; 1 with one replicate)
  replicate variance    V_rep(m) = mean over pairs of D_ab ** 2                                                      (pp ** 2)
                        The copies are one configuration, so the true mean of a delta is 0 and the second moment IS the variance of a delta between two
                        independent decodes (= 2 x the per-copy variance). It is the variance of lock-vs-arm decoding noise when the arm decodes as noisily as lock.
  bootstrap SE          SE_b = (hi - lo) / (2 z)  of the arm's row-paired bootstrap interval [lo, hi] (utterance-cluster, 10,000 draws, seed 20261018), z = 1.959964
  total SE              SE_t = sqrt(SE_b ** 2 + V_rep)                                                               (CONSERVATIVE SUM OF VARIANCES)
  widening              w = z (SE_t - SE_b)   and the reported CI95 is [lo - w, hi + w]   (the point estimate is unchanged)

Identical replicates give V_rep = 0, so w = 0 and the reported CI95 IS the bootstrap interval. The widening is one number per metric per run, the same for every arm of the
run (V_rep does not depend on the arm), and grows with SE_b only through the sum of variances. With one replicate (two copies) V_rep rests on one delta, with all three
copies on three; either is a small-sample estimate and the record says how many pairs it rests on.

PER-ROW F1. Pooled SLU-F1 pools word and char counts over all rows and then takes one F1 (``null_calibration.pooled_f1``): a lever that makes the Solver answer
"emptier" can raise it while the typical row gets worse. The per-row F1 of a row is 2 TP / (2 TP + FP + FN) over the same pooled counts of that row (word and char fields
summed), and 1.0 when a row has no gold and no prediction (a correctly empty row; a false fill on empty gold scores 0). ``per_row_f1_mean`` is the mean over rows. Its
paired delta (arm minus lock) has its own row-paired bootstrap interval, widened by its own V_rep exactly as above.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _path in (str(_REPO / "src"), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from speech_aware_evidence_acquisition.analysis.paired import resampled_statistic  # noqa: E402

Z95 = NormalDist().inv_cdf(0.975)
#: The scorer's per-row count groups: (word, char) names summed per statistic, as ``slurp_direct_fold_b_core_read.SLU_GROUPS`` (re-stated here so the module has no
#: reader import; ``test_paired_ci`` asserts the equality).
SLU_GROUPS = (("slu_f1_word_tp", "slu_f1_char_tp"), ("slu_f1_word_fp", "slu_f1_char_fp"), ("slu_f1_word_fn", "slu_f1_char_fn"))
INTENT_FIELD = "scenario_action_exact"
METRICS = ("slu_f1", "intent", "per_row_slu_f1")
SCHEMA = "saea-paired-ci-noise-model-v1"
FORMULA = ("V_rep = mean over the pairwise deltas of D**2; SE_b = (hi - lo) / (2 z); SE_t = sqrt(SE_b**2 + V_rep); widening w = z (SE_t - SE_b); "
           "reported CI95 = [lo - w, hi + w]; z = 1.959964")


class PairedCiError(ValueError):
    """A refusal: a copy or a count field is missing, or a rule input is malformed."""


# --------------------------------------------------------------------------- per-row F1


def row_counts(row: Mapping[str, float], groups: Sequence[Sequence[str]] = SLU_GROUPS) -> tuple[float, float, float]:
    """(TP, FP, FN) of one row: the word and char fields of each group summed."""

    totals = []
    for names in groups:
        total = 0.0
        for name in names:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise PairedCiError(f"F1 count {name!r} must be a finite non-negative number")
            total += value
        totals.append(total)
    return totals[0], totals[1], totals[2]


def row_f1(row: Mapping[str, float], groups: Sequence[Sequence[str]] = SLU_GROUPS) -> float:
    """The F1 of one row: 2 TP / (2 TP + FP + FN); 1.0 when the row has no gold and no prediction (a correctly empty row)."""

    tp, fp, fn = row_counts(row, groups)
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else 2 * tp / denominator


def per_row_f1_mean(rows: Mapping[str, Mapping[str, float]], ids: Sequence[str], groups: Sequence[Sequence[str]] = SLU_GROUPS) -> float:
    """The mean over ``ids`` (repeats counted) of the per-row F1."""

    return sum(row_f1(rows[i], groups) for i in ids) / len(ids) if ids else float("nan")


def pooled_slu_f1(rows: Mapping[str, Mapping[str, float]], ids: Sequence[str], groups: Sequence[Sequence[str]] = SLU_GROUPS) -> float:
    """The pooled (micro) SLU-F1 over ``ids`` (repeats counted); 0.0 by the scorer's convention when all three totals are zero."""

    tp = fp = fn = 0.0
    for i in ids:
        a, b, c = row_counts(rows[i], groups)
        tp, fp, fn = tp + a, fp + b, fn + c
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def intent_mean(rows: Mapping[str, Mapping[str, float]], ids: Sequence[str], field: str = INTENT_FIELD) -> float:
    return sum(rows[i][field] for i in ids) / len(ids) if ids else float("nan")


# --------------------------------------------------------------------------- the replicate variance and the widening


def pairwise_deltas(levels: Mapping[str, float], copies: Sequence[str], delivering: str = "lock") -> list[dict]:
    """The deltas D_ab = m(a) - m(b) between the request-identical ``copies`` (levels in percentage points), in the order (later copy, earlier copy).

    For the copies (lock, aa, lock-rep2) the pairs read (aa, lock), (lock-rep2, lock), (lock-rep2, aa). Refuses a copy without a level, a repeated copy, fewer than
    two copies, or a delivering arm that is not the first copy."""

    copies = tuple(copies)
    if len(copies) < 2 or len(set(copies)) != len(copies) or copies[0] != delivering:
        raise PairedCiError(f"the copies must be two or more distinct arms starting with {delivering!r}: {list(copies)!r}")
    missing = [c for c in copies if c not in levels]
    if missing:
        raise PairedCiError(f"no level for the request-identical copies {missing}")
    return [{"pair": [b, a], "delta_pp": levels[b] - levels[a]} for a, b in itertools.combinations(copies, 2)]


def replicate_variance(deltas: Sequence[Mapping]) -> dict:
    """V_rep = the mean squared pairwise delta (pp ** 2), with the pairs it rests on."""

    squares = [float(d["delta_pp"]) ** 2 for d in deltas]
    if not squares:
        raise PairedCiError("no pairwise delta: a noise model needs at least one replicate beside the delivering arm")
    variance = sum(squares) / len(squares)
    return {"pairs": [{"pair": list(d["pair"]), "delta_pp": float(d["delta_pp"])} for d in deltas], "n_pairs": len(squares),
            "variance_pp2": variance, "rms_pp": math.sqrt(variance)}


def widen_interval(interval: Mapping, variance_pp2: float, *, z: float = Z95) -> dict:
    """The bootstrap ``interval`` ({point_pp, ci95_lo_pp, ci95_hi_pp, ...}) widened by the replicate variance (see the module docstring).

    The input keys are kept under ``bootstrap_*``; ``ci95_lo_pp`` / ``ci95_hi_pp`` / ``ci95_half_width_pp`` become the widened ones. ``widening_pp`` is the amount added
    to EACH side; it is 0.0 when ``variance_pp2`` is 0."""

    if variance_pp2 < 0 or not math.isfinite(variance_pp2):
        raise PairedCiError("the replicate variance must be a finite non-negative number")
    lo, hi = float(interval["ci95_lo_pp"]), float(interval["ci95_hi_pp"])
    if hi < lo:
        raise PairedCiError("the bootstrap interval is inverted")
    se_b = (hi - lo) / (2.0 * z)
    se_t = math.sqrt(se_b * se_b + variance_pp2)
    widening = z * (se_t - se_b)
    out = dict(interval)
    out.update({
        "bootstrap_ci95_lo_pp": lo, "bootstrap_ci95_hi_pp": hi, "bootstrap_se_pp": se_b,
        "replicate_variance_pp2": float(variance_pp2), "total_se_pp": se_t, "widening_pp": widening,
        "ci95_lo_pp": lo - widening, "ci95_hi_pp": hi + widening, "ci95_half_width_pp": (hi - lo) / 2.0 + widening,
    })
    return out


# --------------------------------------------------------------------------- the noise-adjusted contrast


def _interval(replicates: Sequence[float], point: float, level: float = 0.95) -> dict:
    ordered = sorted(float(v) for v in replicates)
    tail = (1.0 - level) / 2.0 * 100.0

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * q / 100.0
        lo, hi = math.floor(rank), math.ceil(rank)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)

    lo, hi = percentile(tail), percentile(100.0 - tail)
    return {"point_pp": 100.0 * point, "ci95_lo_pp": 100.0 * lo, "ci95_hi_pp": 100.0 * hi, "ci95_half_width_pp": 50.0 * (hi - lo), "draws": len(ordered)}


def per_row_delta_interval(arm_rows: Mapping[str, Mapping[str, float]], ref_rows: Mapping[str, Mapping[str, float]], ids: Sequence[str],
                           cluster_of: Callable[[str], str], *, seed: int, draws: int) -> dict:
    """The row-paired bootstrap interval of the per-row F1 delta (arm minus delivering): the mean over resampled rows of F1_i(arm) - F1_i(lock)."""

    deltas = {i: row_f1(arm_rows[i]) - row_f1(ref_rows[i]) for i in ids}

    def statistic(resampled: Sequence[str]) -> float:
        return sum(deltas[i] for i in resampled) / len(resampled)

    result = resampled_statistic(sample_ids=list(ids), statistic=statistic, seed=seed, n_replicates=draws, ci_level=0.95, cluster_of=cluster_of)
    return _interval(result.replicates, result.point_estimate)


def noise_model(scored: Mapping[str, Mapping[str, Mapping[str, float]]], ids: Sequence[str], *, copies: Sequence[str], delivering: str = "lock") -> dict:
    """The replicate variance of every metric from the request-identical ``copies`` (levels in percentage points over the whole roster)."""

    copies = tuple(copies)
    missing = [c for c in copies if c not in scored]
    if missing:
        raise PairedCiError(f"no scored column for the request-identical copies {missing}")
    ids = list(ids)
    levels = {
        "slu_f1": {c: 100.0 * pooled_slu_f1(scored[c], ids) for c in copies},
        "intent": {c: 100.0 * intent_mean(scored[c], ids) for c in copies},
        "per_row_slu_f1": {c: 100.0 * per_row_f1_mean(scored[c], ids) for c in copies},
    }
    metrics = {name: replicate_variance(pairwise_deltas(levels[name], copies, delivering)) for name in METRICS}
    return {"schema": SCHEMA, "copies": list(copies), "delivering": delivering, "formula": FORMULA, "z": Z95, "levels_pct": levels, "metrics": metrics}


def noise_adjusted_contrast(scored: Mapping[str, Mapping[str, Mapping[str, float]]], ids: Sequence[str], cluster_of: Callable[[str], str], contrast: Mapping, *,
                            copies: Sequence[str], delivering: str = "lock", seed: int, draws: int) -> dict:
    """The paired-contrast record with the decoding term.

    ``contrast`` is the historical record (``paired_contrasts``: ``levels`` and ``versus_delivering[arm][slu_f1 | intent]`` row-paired bootstrap intervals). The result has the
    same shape, so every reader of it keeps working, with:

    * ``versus_delivering[arm][slu_f1 | intent]`` the WIDENED intervals (``bootstrap_*`` keeps the raw ones) for every arm that is not a request-identical copy; the copies
      keep their raw intervals (they ARE the noise, so widening them by it would count it twice) and carry ``widening_pp`` 0.0;
    * ``versus_delivering[arm][per_row_slu_f1]`` the per-row F1 delta with its own bootstrap interval, widened by the per-row replicate variance;
    * ``levels[arm]`` gaining ``per_row_slu_f1_pct`` (the mean over rows of the per-row F1);
    * ``noise_model`` the replicate record (:func:`noise_model`).
    """

    model = noise_model(scored, ids, copies=copies, delivering=delivering)
    copies_set = set(copies)
    out = {key: value for key, value in contrast.items() if key not in ("levels", "versus_delivering")}
    out["levels"] = {arm: dict(level, per_row_slu_f1_pct=100.0 * per_row_f1_mean(scored[arm], ids)) for arm, level in contrast["levels"].items() if arm in scored}
    ref = scored[delivering]
    versus: dict[str, dict] = {}
    for arm, entry in contrast["versus_delivering"].items():
        row = {key: value for key, value in entry.items()}
        if arm in copies_set:
            for metric in ("slu_f1", "intent"):
                row[metric] = dict(entry[metric], widening_pp=0.0, replicate_variance_pp2=0.0, bootstrap_ci95_lo_pp=entry[metric]["ci95_lo_pp"],
                                   bootstrap_ci95_hi_pp=entry[metric]["ci95_hi_pp"])
        else:
            for metric in ("slu_f1", "intent"):
                row[metric] = widen_interval(entry[metric], model["metrics"][metric]["variance_pp2"])
        if arm in scored:
            per_row = per_row_delta_interval(scored[arm], ref, ids, cluster_of, seed=seed, draws=draws)
            row["per_row_slu_f1"] = (dict(per_row, widening_pp=0.0, replicate_variance_pp2=0.0, bootstrap_ci95_lo_pp=per_row["ci95_lo_pp"],
                                          bootstrap_ci95_hi_pp=per_row["ci95_hi_pp"])
                                     if arm in copies_set else widen_interval(per_row, model["metrics"]["per_row_slu_f1"]["variance_pp2"]))
        versus[arm] = row
    out["versus_delivering"] = versus
    out["noise_model"] = model
    out["ci_definition"] = ("CI95 = the row-paired utterance-cluster bootstrap interval widened by the replicate variance of the request-identical copies "
                            f"{list(copies)}: {FORMULA}")
    return out


__all__ = [
    "FORMULA", "INTENT_FIELD", "METRICS", "PairedCiError", "SCHEMA", "SLU_GROUPS", "Z95", "intent_mean", "noise_adjusted_contrast", "noise_model", "pairwise_deltas",
    "per_row_delta_interval", "per_row_f1_mean", "pooled_slu_f1", "replicate_variance", "row_counts", "row_f1", "widen_interval",
]
