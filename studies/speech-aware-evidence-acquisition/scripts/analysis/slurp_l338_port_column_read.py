#!/usr/bin/env python3
"""slurp_l338_port_column_read.py -- the READ of the NPU-port twin of L338 (the lever column on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold B).

Port mode of ``slurp_l338_column_read.py``. That reader binds the historical cell ids and the historical analysis root (``l338_family``); this one takes the cell ids from
``l338_port_family`` (the port run and attempt ids, the port READ run id, the port analysis root) and the reading rules from the port recipe
(``docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/analysis-recipe-v1.json``, ``port_extensions``), never from an edited copy of the historical arm table. Every scoring
function is the historical reader's, imported unchanged (``read_stage``, ``smoke_stage``, ``arm_words``, ``npc.paired_contrasts``: arm minus lock, paired per row,
utterance-cluster bootstrap, 10,000 draws, seed 20261018).

Stages (both write-once):

  smoke  the four-row smoke decode-cost table, ``analysis/<port READ id>/smoke-decode-cost.json`` (schema ``saea-l338-lever-column-smoke-v1``), the input of
         ``scripts/flights/batch/smoke_signoff.py`` and of the port package's ``signoff-conditions-v1.json``. Nothing is scored.
  read   the historical read (``read.json`` and ``README.md``, same format as the historical read) plus ``port-verdicts.json``: the port's three cross-run verdicts.
         Run ``stage3_rollout_score.py --mode perception-branches`` first (``step_score.sh``) so the per-arm column score files exist.

Noise model M1 (plan 2026-10-30, task T2). The port flight carries THREE request-identical copies of the lock (``lock``, the twin ``aa`` and the replicate ``lock-rep2``); the read
adds the decoding term to every arm delta (``scripts/analysis/paired_ci.py``, formula in its docstring and in the recipe ``port_extensions.noise_model``): the CI95 of an arm
minus lock is the row-paired utterance-cluster bootstrap interval widened by the replicate variance (the mean squared pairwise delta among the three copies, summed with the
bootstrap variance). ``read.json`` keeps the interval-only record under ``contrast_bootstrap_only``. Per-row F1 (the mean over rows of each row's F1) is reported beside pooled F1
with its own paired delta and its own widened CI95, and the within-run word NOMINATED additionally needs BOTH the pooled and the per-row lower bound of the primary metric above
zero (``port_arm_words``): a lever that lifts pooled F1 by emptying rows is withheld.

Port verdicts (every number carries ``source`` and ``locator`` naming the input file in ``inputs`` and the place in it; ``inputs`` lists path and sha256 of each)

  PARITY       per arm in ``port_extensions.parity_band.applies_to`` (lock, aa): official pooled SLU-F1 in [71.61, 72.75] AND intent accuracy in [84.91, 85.74], both
               inclusive, both read on the scorer's two-decimal percent; otherwise OUTSIDE-BAND with the metric, the side (below / above) and the distance in pp.
               S0 is PARITY only when lock AND aa are PARITY AND the run is ADMISSIBLE.
  ADMISSIBLE   |SLU-F1(aa) - SLU-F1(lock)| <= 1.20 pp (the point delta of the two official column scorings, from the paired contrast, rounded to 6 decimals before the
               comparison so that a delta of exactly 1.20 is not lost to float error); otherwise NOT-ADMISSIBLE. A run without the twin arm is NOT-ADMISSIBLE.
  IMPROVEMENT  per lever arm (l1-support, l2-union): IMPROVEMENT-ADMISSIBLE only when ALL of (1) the run is ADMISSIBLE, (2) the arm's SLU-F1 (two-decimal) is >= band top
               + cross-run bar (72.75 + 2.5 = 75.25), (3) the paired SLU delta against the port lock has CI95 lower bound > 0 (strict), (4) aa is PARITY (inside the
               band). Otherwise NO-CROSS-RUN-STATEMENT (the recipe: "no cross-run improvement statement is made; the within-run words stand on their own"). The within-run
               word (NOMINATED / HARM / NEUTRAL, ``arm_words``) is reported beside it and is independent of it.

Model-free. Every sealed byte is opened read-only; no utterance text, gold label or model output is printed or persisted.

    python -B scripts/analysis/slurp_l338_port_column_read.py --stage smoke --data-root $SPEECHRL_DATA_DIR
    python -B scripts/analysis/slurp_l338_port_column_read.py --stage read  --data-root $SPEECHRL_DATA_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _path in (str(_REPO / "src"), str(_HERE), str(_REPO / "scripts" / "flights" / "nx")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import l338_family as fam  # noqa: E402
import l338_port_family as pf  # noqa: E402
import paired_ci  # noqa: E402  (noise model M1: the replicate variance, the widened CI95, per-row F1)
import slurp_l338_column_read as hist  # noqa: E402  (the historical reader: every scoring function is used unchanged)

base, npc = hist.base, hist.npc
ReadError = hist.ReadError
VERDICTS_SCHEMA = "saea-l338-port-verdicts-v2"
RECIPE_SCHEMA = "saea-stage3-rollout-analysis-recipe-v1"
WORD_PARITY, WORD_OUTSIDE = "PARITY", "OUTSIDE-BAND"
WORD_ADMISSIBLE, WORD_NOT_ADMISSIBLE = "ADMISSIBLE", "NOT-ADMISSIBLE"
WORD_IMPROVEMENT, WORD_NO_STATEMENT = "IMPROVEMENT-ADMISSIBLE", "NO-CROSS-RUN-STATEMENT"
METRICS = (("slu_f1", "slu_f1_pct", "SLU-F1"), ("intent", "intent_pct", "intent accuracy"))
ADMISSIBILITY_DECIMALS = 6


def two_dp(value: float) -> float:
    """The scorer's two-decimal percent, formatted exactly as the reader's table prints it (``{:.2f}``)."""

    return float(f"{value:.2f}")


def cite(value: Any, source: str, locator: str) -> dict:
    """A number with the input file it came from (``source`` is a key of the verdict file's ``inputs``) and the place in that file (or the computation over it)."""

    return {"value": value, "source": source, "locator": locator}


# --------------------------------------------------------------------------- the rules, read from the port recipe


def load_recipe(path: Path, *, family=pf) -> dict:
    """The port recipe, refused by name unless it binds the port ids and the historical scoring constants."""

    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    problems: list[str] = []
    if recipe.get("schema") != RECIPE_SCHEMA:
        problems.append(f"schema {recipe.get('schema')!r} != {RECIPE_SCHEMA!r}")
    ids = recipe.get("run_ids") or {}
    for key, expected in (("read", family.READ_RUN_ID), ("arm", family.CELLS["arm"]["run"]), ("smoke", family.CELLS["smoke"]["run"])):
        if ids.get(key) != expected:
            problems.append(f"run_ids.{key} {ids.get(key)!r} != {expected!r}")
    if recipe.get("read_run_id") != family.READ_RUN_ID:
        problems.append(f"read_run_id {recipe.get('read_run_id')!r} != {family.READ_RUN_ID!r}")
    if recipe.get("delivering") != fam.DELIVERING or recipe.get("twin") != fam.TWIN:
        problems.append("delivering / twin are not lock / aa")
    if sorted(recipe.get("arms") or []) != sorted(pf.ARM_ORDER):
        problems.append(f"arms {recipe.get('arms')!r} != {sorted(pf.ARM_ORDER)!r}")
    primary = recipe.get("primary") or {}
    if primary.get("draws") != hist.DRAWS or primary.get("seed") != hist.SEED:
        problems.append(f"primary draws/seed {primary.get('draws')!r}/{primary.get('seed')!r} != {hist.DRAWS}/{hist.SEED}")
    if (recipe.get("guard") or {}).get("floors_pp") != {"intent": fam.FLOOR_INTENT_PP, "slu": fam.FLOOR_SLU_PP}:
        problems.append("guard.floors_pp is not the historical floors")
    if not isinstance(recipe.get("port_extensions"), Mapping):
        problems.append("no port_extensions object")
    else:
        model = recipe["port_extensions"].get("noise_model")
        if not isinstance(model, Mapping):
            problems.append("port_extensions.noise_model is absent (the replicate rule and the widened CI definition)")
        else:
            if list(model.get("copies") or []) != list(pf.REPLICATE_SET):
                problems.append(f"noise_model.copies {model.get('copies')!r} != {list(pf.REPLICATE_SET)!r}")
            if model.get("formula") != paired_ci.FORMULA:
                problems.append("noise_model.formula is not paired_ci.FORMULA")
            if model.get("max_replicates") != 2 or model.get("replicate_arms") != list(pf.REPLICATE_ARMS):
                problems.append("noise_model.max_replicates / replicate_arms are not 2 / the port replicate arms")
    if problems:
        raise ReadError(f"the port recipe {path} is refused: " + "; ".join(problems))
    return recipe


def rules_from_recipe(recipe: Mapping) -> dict:
    """The numeric rules of the port verdicts with the locator of each in the recipe (``source`` ``recipe``)."""

    ext = recipe["port_extensions"]
    band, cmp0, bar = ext["parity_band"], ext["registered_comparisons"][0], ext["cross_run_bar"]
    if cmp0.get("arm") != fam.TWIN or cmp0.get("against") != fam.DELIVERING or band.get("applies_to") != [fam.DELIVERING, fam.TWIN]:
        raise ReadError("the recipe's registered comparison / band do not bind aa against lock and the band to lock and aa")
    if bar.get("twin_arm") != fam.TWIN or not bar.get("requires_twin_arm_in_the_same_run"):
        raise ReadError("the recipe's cross-run bar does not require the twin aa in the same run")
    floor = round(band["slu_f1"]["high"] + bar["bar_slu_pp"], 2)
    p = "port_extensions"
    return {
        "band": {"slu_f1": (band["slu_f1"]["low"], band["slu_f1"]["high"]), "intent": (band["intent"]["low"], band["intent"]["high"])},
        "applies_to": list(band["applies_to"]), "twin_threshold_slu_pp": cmp0["threshold_slu_pp"], "bar_slu_pp": bar["bar_slu_pp"], "improvement_floor_slu": floor,
        "cites": {
            "band.slu_f1.low": cite(band["slu_f1"]["low"], "recipe", f"{p}.parity_band.slu_f1.low"),
            "band.slu_f1.high": cite(band["slu_f1"]["high"], "recipe", f"{p}.parity_band.slu_f1.high"),
            "band.intent.low": cite(band["intent"]["low"], "recipe", f"{p}.parity_band.intent.low"),
            "band.intent.high": cite(band["intent"]["high"], "recipe", f"{p}.parity_band.intent.high"),
            "twin_threshold_slu_pp": cite(cmp0["threshold_slu_pp"], "recipe", f"{p}.registered_comparisons[0].threshold_slu_pp"),
            "bar_slu_pp": cite(bar["bar_slu_pp"], "recipe", f"{p}.cross_run_bar.bar_slu_pp"),
            "improvement_floor_slu": cite(floor, "recipe", f"{p}.parity_band.slu_f1.high + {p}.cross_run_bar.bar_slu_pp (rounded to 2 decimals)")}}


# --------------------------------------------------------------------------- the verdicts (pure functions over the paired contrast)


def parity_of_arm(arm: str, levels: Mapping, rules: Mapping) -> dict:
    """PARITY / OUTSIDE-BAND of one arm: both metrics inside their band, inclusive, on the two-decimal reading."""

    out: dict[str, Any] = {}
    outside: list[dict] = []
    for metric, key, label in METRICS:
        raw = levels[arm][key]
        value = two_dp(raw)
        low, high = rules["band"][metric]
        side = "below" if value < low else "above" if value > high else "inside"
        distance = round(low - value, 2) if side == "below" else round(value - high, 2) if side == "above" else 0.0
        out[metric] = {"label": label, "value_pct": cite(raw, "read_json", f"contrast.levels.{arm}.{key}"), "value_pct_2dp": value,
                       "band": [rules["cites"][f"band.{metric}.low"], rules["cites"][f"band.{metric}.high"]], "side": side, "distance_pp": distance}
        if side != "inside":
            outside.append({"metric": metric, "side": side, "distance_pp": distance})
    return {"arm": arm, "word": WORD_PARITY if not outside else WORD_OUTSIDE, "outside": outside, **out}


def admissibility(contrast: Mapping, rules: Mapping) -> dict:
    """ADMISSIBLE when |SLU-F1(aa) - SLU-F1(lock)| <= the registered threshold; a run without the twin supports no cross-run statement."""

    twin = (contrast.get("versus_delivering") or {}).get(fam.TWIN)
    threshold = rules["twin_threshold_slu_pp"]
    if twin is None:
        return {"word": WORD_NOT_ADMISSIBLE, "reason": "cross-flight-no-replicate: the run has no twin arm aa", "threshold_slu_pp": rules["cites"]["twin_threshold_slu_pp"]}
    point = twin["slu_f1"]["point_pp"]
    magnitude = round(abs(point), ADMISSIBILITY_DECIMALS)
    levels = contrast["levels"]
    return {"word": WORD_ADMISSIBLE if magnitude <= threshold else WORD_NOT_ADMISSIBLE, "rule": f"|aa - lock| <= {threshold} pp on official pooled SLU-F1 (point delta)",
            "aa_minus_lock_slu_pp": cite(point, "read_json", "contrast.versus_delivering.aa.slu_f1.point_pp"), "abs_rounded_6dp": magnitude,
            "aa_minus_lock_slu_pp_from_2dp_readings": round(two_dp(levels[fam.TWIN]["slu_f1_pct"]) - two_dp(levels[fam.DELIVERING]["slu_f1_pct"]), 2),
            "aa_minus_lock_intent_pp": cite(twin["intent"]["point_pp"], "read_json", "contrast.versus_delivering.aa.intent.point_pp"),
            "threshold_slu_pp": rules["cites"]["twin_threshold_slu_pp"]}


def port_arm_words(contrast: Mapping, *, floor_slu_pp: float = fam.FLOOR_SLU_PP, floor_intent_pp: float = fam.FLOOR_INTENT_PP) -> dict:
    """The within-run words of the port: the historical ``arm_words`` over the WIDENED CI95 (``contrast`` from ``paired_ci.noise_adjusted_contrast``), with the
    pooled-and-per-row condition added to NOMINATED.

    The primary metric of a lever (L1: SLU-F1, L2: intent accuracy) must have BOTH its pooled and its per-row delta lower bound above zero; for intent accuracy the per-row
    mean IS the pooled one, for SLU-F1 the per-row delta is ``versus_delivering[arm][per_row_slu_f1]``. The guard is checked at both levels too: the L2 SLU guard needs the
    per-row SLU upper bound at or above minus the SLU floor as well. A lever whose historical word would be NOMINATED but fails one of these keeps ``hist_word`` and is
    read NEUTRAL with ``nominated_withheld_by``; HARM is never softened (it only needs the pooled bounds the historical rule uses)."""

    words = hist.arm_words(contrast, floor_slu_pp=floor_slu_pp, floor_intent_pp=floor_intent_pp)
    versus = contrast["versus_delivering"]
    for arm, entry in words["words"].items():
        per_row = versus[arm].get("per_row_slu_f1")
        if per_row is None:
            raise ReadError(f"{arm}: the contrast has no per_row_slu_f1 delta; run it through paired_ci.noise_adjusted_contrast")
        lever = entry["lever"]
        primary_pooled = versus[arm]["slu_f1" if lever == "L1" else "intent"]
        primary_per_row = per_row if lever == "L1" else versus[arm]["intent"]               # intent accuracy is already a mean over rows
        extra = {
            "pooled_primary_ci_lo_above_zero": primary_pooled["ci95_lo_pp"] > 0,
            "per_row_primary_ci_lo_above_zero": primary_per_row["ci95_lo_pp"] > 0,
            "per_row_slu_no_resolved_loss": per_row["ci95_hi_pp"] >= -floor_slu_pp,
        }
        entry["hist_word"] = entry["word"]
        entry["conditions"] = dict(entry["conditions"], **extra)
        entry["per_row_slu_delta_pp"] = per_row["point_pp"]
        entry["per_row_slu_ci95"] = [per_row["ci95_lo_pp"], per_row["ci95_hi_pp"]]
        entry["ci_definition"] = "widened: bootstrap + replicate variance (paired_ci)"
        failed = [name for name, ok in extra.items() if not ok]
        entry["nominated_withheld_by"] = failed if entry["word"] == "NOMINATED" and failed else []
        if entry["nominated_withheld_by"]:
            entry["word"], entry["next_step"] = "NEUTRAL", None
    words["ci_definition"] = contrast.get("ci_definition")
    return words


def improvement_of_arm(arm: str, contrast: Mapping, rules: Mapping, parity: Mapping, admissible: Mapping, within_run_word: str | None) -> dict:
    """IMPROVEMENT-ADMISSIBLE / NO-CROSS-RUN-STATEMENT for one lever arm (see the module docstring for the four conditions)."""

    level = contrast["levels"][arm]["slu_f1_pct"]
    delta = contrast["versus_delivering"][arm]["slu_f1"]
    level2 = two_dp(level)
    floor = rules["improvement_floor_slu"]
    conditions = {
        "run_admissible": admissible["word"] == WORD_ADMISSIBLE,
        "slu_at_or_above_improvement_floor": level2 >= floor,
        "slu_delta_ci95_lower_above_zero": delta["ci95_lo_pp"] > 0,
        "twin_aa_inside_band": parity[fam.TWIN]["word"] == WORD_PARITY,
    }
    per_row = contrast["versus_delivering"][arm].get("per_row_slu_f1")
    if per_row is not None:                                    # noise model M1: the per-row delta must clear zero as well (a record without it is a pre-M1 fixture)
        conditions["per_row_slu_delta_ci95_lower_above_zero"] = per_row["ci95_lo_pp"] > 0
    return {
        "arm": arm, "word": WORD_IMPROVEMENT if all(conditions.values()) else WORD_NO_STATEMENT, "conditions": conditions,
        "slu_f1_pct": cite(level, "read_json", f"contrast.levels.{arm}.slu_f1_pct"), "slu_f1_pct_2dp": level2,
        "improvement_floor_slu": rules["cites"]["improvement_floor_slu"],
        "slu_delta_vs_lock_pp": cite(delta["point_pp"], "read_json", f"contrast.versus_delivering.{arm}.slu_f1.point_pp"),
        "slu_delta_ci95_pp": [cite(delta["ci95_lo_pp"], "read_json", f"contrast.versus_delivering.{arm}.slu_f1.ci95_lo_pp"),
                              cite(delta["ci95_hi_pp"], "read_json", f"contrast.versus_delivering.{arm}.slu_f1.ci95_hi_pp")],
        "intent_pct": cite(contrast["levels"][arm]["intent_pct"], "read_json", f"contrast.levels.{arm}.intent_pct"),
        "intent_delta_vs_lock_pp": cite(contrast["versus_delivering"][arm]["intent"]["point_pp"], "read_json", f"contrast.versus_delivering.{arm}.intent.point_pp"),
        "intent_delta_ci95_pp": [cite(contrast["versus_delivering"][arm]["intent"]["ci95_lo_pp"], "read_json", f"contrast.versus_delivering.{arm}.intent.ci95_lo_pp"),
                                 cite(contrast["versus_delivering"][arm]["intent"]["ci95_hi_pp"], "read_json", f"contrast.versus_delivering.{arm}.intent.ci95_hi_pp")],
        **({} if per_row is None else {
            "per_row_slu_f1_pct": cite(contrast["levels"][arm]["per_row_slu_f1_pct"], "read_json", f"contrast.levels.{arm}.per_row_slu_f1_pct"),
            "per_row_slu_delta_vs_lock_pp": cite(per_row["point_pp"], "read_json", f"contrast.versus_delivering.{arm}.per_row_slu_f1.point_pp"),
            "per_row_slu_delta_ci95_pp": [cite(per_row["ci95_lo_pp"], "read_json", f"contrast.versus_delivering.{arm}.per_row_slu_f1.ci95_lo_pp"),
                                          cite(per_row["ci95_hi_pp"], "read_json", f"contrast.versus_delivering.{arm}.per_row_slu_f1.ci95_hi_pp")]}),
        "within_run_word": cite(within_run_word, "read_json", f"arm_words.words.{arm}.word")}


def port_verdicts(contrast: Mapping, arm_words: Mapping, rules: Mapping, *, inputs: Mapping, interval: Mapping, recipe_status: str | None = None, family=pf) -> dict:
    """The ``port-verdicts.json`` document from a paired-contrast record (``npc.paired_contrasts``), the historical ``arm_words`` and the recipe rules."""

    levels = contrast["levels"]
    missing = [arm for arm in rules["applies_to"] if arm not in levels]
    if missing:
        raise ReadError(f"the parity arms {missing} have no column in the contrast")
    parity = {arm: parity_of_arm(arm, levels, rules) for arm in rules["applies_to"]}
    admissible = admissibility(contrast, rules)
    words = (arm_words or {}).get("words") or {}
    improvement = {arm: improvement_of_arm(arm, contrast, rules, parity, admissible, (words.get(arm) or {}).get("word"))
                   for arm in hist.CANDIDATES if arm in contrast["versus_delivering"]}
    noise = contrast.get("noise_model")
    noise_block = None if noise is None else {
        "copies": list(noise["copies"]), "formula": noise["formula"], "ci_definition": contrast.get("ci_definition"),
        "metrics": {name: {"variance_pp2": cite(m["variance_pp2"], "read_json", f"contrast.noise_model.metrics.{name}.variance_pp2"),
                           "rms_pp": cite(m["rms_pp"], "read_json", f"contrast.noise_model.metrics.{name}.rms_pp"), "n_pairs": m["n_pairs"],
                           "pairs": [{"pair": d["pair"], "delta_pp": cite(d["delta_pp"], "read_json", f"contrast.noise_model.metrics.{name}.pairs[{i}].delta_pp")}
                                     for i, d in enumerate(m["pairs"])]}
                    for name, m in noise["metrics"].items()},
        "note": ("admissibility keeps the registered aa-vs-lock rule; the replicate spread (the other pairwise deltas) is reported beside it and enters no word")}
    reasons = [f"{arm} is {parity[arm]['word']}" for arm in rules["applies_to"] if parity[arm]["word"] != WORD_PARITY]
    if admissible["word"] != WORD_ADMISSIBLE:
        reasons.append(f"the run is {admissible['word']}")
    s0 = WORD_PARITY if not reasons else WORD_OUTSIDE
    return {
        "schema": VERDICTS_SCHEMA, "status": "PORT-VERDICTS (development numbers; claims_benefit false; a reading, never an admission)",
        "read_run_id": family.READ_RUN_ID, "run_id": family.CELLS["arm"]["run"], "recipe_status": recipe_status, "inputs": dict(inputs), "interval": dict(interval),
        "rules": {k: v for k, v in rules["cites"].items()},
        "rule_text": {
            "parity": "arm's SLU-F1 in [band.slu_f1.low, band.slu_f1.high] AND intent in [band.intent.low, band.intent.high], inclusive, two-decimal percent",
            "admissible": "|SLU-F1(aa) - SLU-F1(lock)| <= twin_threshold_slu_pp (point delta; rounded to 6 decimals before the comparison)",
            "improvement": ("IMPROVEMENT-ADMISSIBLE iff run_admissible AND SLU-F1 (2dp) >= improvement_floor_slu AND SLU delta vs lock CI95 lower bound > 0 (widened CI95) AND "
                            "the per-row SLU-F1 delta CI95 lower bound > 0 AND aa PARITY; otherwise NO-CROSS-RUN-STATEMENT"),
            "ci95": "every CI95 is the row-paired bootstrap interval widened by the replicate variance of {lock, aa, lock-rep2} (noise_model.formula)"},
        "parity": parity, "twin_admissibility": admissible, "improvement": improvement, "noise_model": noise_block,
        "summary": {"s0": {"word": s0, "reasons": reasons, "needs": "lock PARITY and aa PARITY and the run ADMISSIBLE"},
                    "s2_may_start": parity[fam.DELIVERING]["word"] == WORD_PARITY,
                    "s2_basis": "recipe port_extensions.parity_band.consequence: OUTSIDE-BAND on the port lock arm -> S2 does not start (a reading; the owner decides)",
                    "improvement": {arm: v["word"] for arm, v in improvement.items()}}}


# --------------------------------------------------------------------------- the noise model (M1): columns, widened contrast, words


def load_port_column_rows(analysis_columns: Path, scorer_id: str = hist.SCORER) -> dict[str, dict]:
    """Per-column per-row official scores of the PORT arm table (the historical four plus the replicate ``lock-rep2``) and the stream columns; refuses a missing file."""

    out: dict[str, dict] = {}
    for arm in [*pf.ARM_ORDER, *hist.stream_column_names()]:
        path = Path(analysis_columns) / hist.COLUMN_SCORE_NAME.format(arm=arm, scorer=scorer_id)
        if not path.exists():
            raise ReadError(f"missing column score file {path}; run `l338_read.sh columns` first")
        out[arm] = base.rows_table(json.loads(path.read_text(encoding="utf-8")), scorer_id)
    return out


def apply_noise_model(result: dict, scored: Mapping, ids: Sequence[str], cluster_of, *, seed: int, draws: int) -> dict:
    """Replace ``result['contrast']`` / ``result['arm_words']`` / ``result['interval']`` by the noise-adjusted ones; the interval-only contrast stays under
    ``contrast_bootstrap_only`` (and its words under ``arm_words_bootstrap_only``) so nothing the historical reader wrote is lost."""

    result["contrast_bootstrap_only"], result["arm_words_bootstrap_only"] = result["contrast"], result["arm_words"]
    contrast = paired_ci.noise_adjusted_contrast(scored, ids, cluster_of, result["contrast"], copies=pf.REPLICATE_SET, delivering=fam.DELIVERING, seed=seed, draws=draws)
    result["contrast"] = contrast
    result["arm_words"] = port_arm_words(contrast)
    result["noise_model"] = contrast["noise_model"]
    result["interval"] = {"kind": "utterance-cluster bootstrap widened by the replicate variance (noise model M1)", "draws": draws, "seed": seed, "level": "CI95",
                          "bootstrap": {"draws": draws, "seed": seed}, "formula": paired_ci.FORMULA, "copies": list(pf.REPLICATE_SET)}
    return result


def noise_markdown(result: Mapping) -> str:
    """The port appendix of the README: the replicate deltas, the widening, the per-row F1 beside the pooled one and the NOMINATED conditions."""

    contrast, model = result["contrast"], result["noise_model"]
    lines = ["## Noise model (M1): decoding term and per-row F1", "", f"CI95 = {contrast['ci_definition']}", "",
             "| metric | replicate pairs (delta pp) | V_rep (pp^2) | rms (pp) |", "| --- | --- | ---: | ---: |"]
    for name, m in model["metrics"].items():
        pairs = "; ".join(f"{'-'.join(p['pair'])} {p['delta_pp']:+.2f}" for p in m["pairs"])
        lines.append(f"| {name} | {pairs} | {m['variance_pp2']:.3f} | {m['rms_pp']:.2f} |")
    lines += ["", "| arm | SLU-F1 | per-row SLU-F1 | SLU d (pp) | CI95 (bootstrap) | CI95 (widened) | per-row d (pp) | per-row CI95 (widened) |", "| --- | ---: | ---: | ---: | --- | --- | ---: | --- |"]
    for arm, entry in contrast["versus_delivering"].items():
        s, r, lv = entry["slu_f1"], entry["per_row_slu_f1"], contrast["levels"][arm]
        lines.append(f"| {arm} | {lv['slu_f1_pct']:.2f} | {lv['per_row_slu_f1_pct']:.2f} | {s['point_pp']:+.2f} | [{s['bootstrap_ci95_lo_pp']:+.2f}, {s['bootstrap_ci95_hi_pp']:+.2f}] | "
                     f"[{s['ci95_lo_pp']:+.2f}, {s['ci95_hi_pp']:+.2f}] | {r['point_pp']:+.2f} | [{r['ci95_lo_pp']:+.2f}, {r['ci95_hi_pp']:+.2f}] |")
    lines += ["", "| arm | word | historical word | withheld by |", "| --- | --- | --- | --- |"]
    for arm, w in result["arm_words"]["words"].items():
        lines.append(f"| {arm} | {w['word']} | {w['hist_word']} | {', '.join(w['nominated_withheld_by']) or ''} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- the sealed-run adapter and the CLI


def _input(path: Path, root: Path | None = None) -> dict:
    p = Path(path)
    return {"path": str(p.relative_to(root)) if root and p.is_relative_to(root) else str(p), "sha256": base.sha256_file(p)}


def _smoke(args: argparse.Namespace, root: Path) -> tuple[dict, Path]:
    cell = pf.CELLS["smoke"]
    paths = base.run_paths(args.data_root, cell["run"], cell["attempt"])
    for name in ("trace", "outputs"):
        if not paths[name].exists():
            raise ReadError(f"missing sealed file {paths[name]}")
    outputs_rows = hist.load_outputs_rows(paths["outputs"])
    contacts = npc.trace_contacts(paths["trace"], fam.THINKING_KNOBS["budget_message"])
    by_arm = npc.attribute(contacts, hist.arm_contact_map(outputs_rows))
    result = hist.smoke_stage(by_arm, missing=hist.missing_columns(outputs_rows, contacts, pf.ARM_ORDER))
    result["throughput"]["arm_mix_registered_rows"] = pf.ARM_MIX_ROWS              # the PORT arm mix (c17: the replicate adds a Perception and a Solver contact)
    result["sealed_hashes"] = {n: base.sha256_file(paths[n]) for n in ("trace", "outputs")}
    result["port"] = {"read_run_id": pf.READ_RUN_ID, "run_id": cell["run"], "attempt_id": cell["attempt"],
                      "note": "slot-normalised GPU hours in `throughput` use the historical 4 slots; the port card-hour bound is PROVISIONAL (l338_port_family.CARD_HOUR_FORMULA)"}
    result["throughput"]["registered_gpu_hours_ceiling"] = None
    return result, root / "smoke-decode-cost.json"


def _read(args: argparse.Namespace, root: Path, *, family=pf) -> tuple[dict, Path, dict, Path]:
    recipe_path = args.recipe if args.recipe.is_absolute() else args.repo_root / args.recipe
    recipe = load_recipe(recipe_path, family=family)
    rules = rules_from_recipe(recipe)
    cell = family.CELLS["arm"]
    paths = base.run_paths(args.data_root, cell["run"], cell["attempt"])
    for name in ("trace", "outputs", "manifest"):
        if not paths[name].exists():
            raise ReadError(f"missing sealed file {paths[name]}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if str(manifest.get("status", "")).upper() != "COMPLETED":
        raise ReadError(f"manifest status is {manifest.get('status')!r}, not COMPLETED")
    arms = load_port_column_rows(root / "columns")
    roster_path = args.repo_root / fam.ROSTER[cell["roster"]]
    ids, cluster_of, scenario_of = base.roster_maps(roster_path)
    for arm, rows in arms.items():
        if set(rows) != set(ids):
            raise ReadError(f"{arm}: the per-sample id set is not the frozen Fold B roster's")
    outputs_rows = hist.load_outputs_rows(paths["outputs"])
    contacts = npc.trace_contacts(paths["trace"], fam.THINKING_KNOBS["budget_message"])
    by_arm = npc.attribute(contacts, hist.arm_contact_map(outputs_rows))
    wanted = {request for row in outputs_rows for request in row["requests"].values()}
    streams = hist.solver_stream_keys(outputs_rows, hist.trace_response_keys(paths["trace"], wanted))
    sampling_path = root / "columns" / "sampling-rows.json"
    sampling_rows = json.loads(sampling_path.read_text(encoding="utf-8"))["rows"] if sampling_path.exists() else None
    gold, n1 = hist._load_json(args.gold_intents), hist._load_json(args.n1_json)
    result = hist.read_stage(arms, ids, cluster_of, scenario_of, by_arm, hist.contact_rows(outputs_rows, contacts), seed=args.seed, draws=args.draws,
                             missing=hist.missing_columns(outputs_rows, contacts, pf.ARM_ORDER), stream_disc=hist.stream_discordance(streams, ids),
                             perception_disc=hist.stream_discordance(hist.perception_stream_keys(outputs_rows), ids), sampling_rows=sampling_rows, gold=gold,
                             outputs_rows=outputs_rows, forms=hist.l1_forms_from_trace(outputs_rows, paths["trace"], n1))
    apply_noise_model(result, arms, ids, cluster_of, seed=args.seed, draws=args.draws)
    result["throughput"]["arm_mix_registered_rows"] = pf.ARM_MIX_ROWS
    result["sealed_hashes"] = {n: base.sha256_file(paths[n]) for n in ("trace", "outputs", "manifest")}
    result["throughput"]["registered_gpu_hours_ceiling"] = None
    result["port"] = {"read_run_id": family.READ_RUN_ID, "run_id": cell["run"], "attempt_id": cell["attempt"], "recipe": str(args.recipe),
                      "note": "slot-normalised GPU hours in `throughput` use the historical 4 slots; the port card-hour bound is PROVISIONAL (l338_port_family.CARD_HOUR_FORMULA)"}
    inputs = {"recipe": _input(recipe_path, args.repo_root), "roster": _input(roster_path, args.repo_root), "manifest": _input(paths["manifest"], args.data_root),
              "trace": _input(paths["trace"], args.data_root), "outputs": _input(paths["outputs"], args.data_root)}
    for arm in pf.ARM_ORDER:
        inputs[f"column_scores:{arm}"] = _input(root / "columns" / hist.COLUMN_SCORE_NAME.format(arm=arm, scorer=hist.SCORER), args.data_root)
    meta = {"recipe": recipe, "rules": rules, "inputs": inputs}
    return result, root / "read.json", meta, root / pf.VERDICTS_NAME


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "read"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path(hist._data_root_env()))
    parser.add_argument("--repo-root", type=Path, default=_REPO)
    parser.add_argument("--analysis-root", type=Path, default=None, help=f"default <data root>/{pf.ANALYSIS_ROOT}")
    parser.add_argument("--recipe", type=Path, default=Path(pf.RECIPE_RELPATH), help="the port analysis recipe (repo-relative or absolute)")
    parser.add_argument("--seed", type=int, default=hist.SEED)
    parser.add_argument("--draws", type=int, default=hist.DRAWS)
    parser.add_argument("--gold-intents", type=Path, default=None, help="optional JSON {sample_id: gold intent id} for the L2 recall tables")
    parser.add_argument("--n1-json", type=Path, default=None, help="optional JSON {sample_id: N1} for the N1-gated L1 forms")
    args = parser.parse_args(argv)
    root = args.analysis_root or (args.data_root / pf.ANALYSIS_ROOT)
    try:
        if args.stage == "smoke":
            outs = [root / "smoke-decode-cost.json"]
        else:
            outs = [root / "read.json", root / pf.VERDICTS_NAME]
        for out in outs:
            if out.exists():
                print(f"REFUSED: {out} exists (write-once)", file=sys.stderr)
                return 6
        root.mkdir(parents=True, exist_ok=True)
        if args.stage == "smoke":
            result, out = _smoke(args, root)
            out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            out.with_name("smoke-decode-cost.md").write_text(hist.render_markdown({"arms_nodes": result["arms"], "throughput": result.get("throughput")}), encoding="utf-8")
            print(json.dumps({"written": str(out)}))
            return 0
        result, out, meta, verdict_out = _read(args, root)
        read_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        inputs = dict(meta["inputs"], read_json={"path": str(out), "sha256": hashlib.sha256(read_text.encode("utf-8")).hexdigest()})
        interval = dict(result["interval"], registered=bool(result["interval"]["draws"] == hist.DRAWS and result["interval"]["seed"] == hist.SEED))
        verdicts = port_verdicts(result["contrast"], result["arm_words"], meta["rules"], inputs=inputs, interval=interval,
                                 recipe_status=(meta["recipe"].get("port_extensions") or {}).get("status"))
        out.write_text(read_text, encoding="utf-8")
        out.with_name("README.md").write_text(hist.render_markdown(result) + "\n" + noise_markdown(result), encoding="utf-8")
        verdict_out.write_text(json.dumps(verdicts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"written": [str(out), str(verdict_out)], "arm_words": {a: w["word"] for a, w in result["arm_words"]["words"].items()},
                          "parity": {a: v["word"] for a, v in verdicts["parity"].items()}, "twin": verdicts["twin_admissibility"]["word"],
                          "improvement": verdicts["summary"]["improvement"]}))
        return 0
    except ReadError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
