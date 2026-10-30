#!/usr/bin/env python3
"""slurp_tfrl_v2_port_read.py -- the READ of the T cell (TFRL-v2 on the BF16 / vLLM-Ascend Thinking core): the per-iteration construction read and the Fold B read.

MODEL-FREE. The reader opens sealed files read-only (rollout traces, the bank receipts the distiller wrote, the per-row official score tables) and never contacts a model,
a server, a tokenizer, an encoder or a scorer. The recipe it applies is ``docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/analysis-recipe-v1.json`` (written by
``scripts/flights/nx/tfrl_v2_port_write_registration.py``; its sha256 is pinned in ``read-row-draft-v1.md`` and in the facts); the numbers and ids come from
``scripts/flights/nx/tfrl_v2_port_rails.py`` so none is spelled twice.

Stage ``construction`` (one iteration, both R01C halves; write-once ``construction-i<t>.json`` and one ``vote-answers-i<t>-<H1|H2>.jsonl`` per half)

  per half     positive rate (positives / usable rows), mean best-of-K gain in pp (100 x the mean over usable rows of the best trajectory's reward against lock, negative values
               included), the reasons rows were not positive, the rows skipped (a row without all K Solver answers and scores is not usable and is never silently dropped);
  banks        rule survivors per half and in the union bank, exemplars per scenario of the union bank and the covered share of scenarios and of rows (a scenario is COVERED when
               the union bank holds at least one exemplar for it); read from the distiller's bank receipts when they exist, null with a note when they do not;
  construction gain  SLU-F1(vote over s1..s4) - SLU-F1(lock) in pp over the usable rows of both halves, pooled. The vote is derived here, model-free, from the trace (scenario_action
               by plurality, ties to the lowest k; an entity kept at support >= 2 of 4) and written to ``vote-answers-i<t>-<half>.jsonl`` for the OFFICIAL scorer to score as
               column ``vote`` outside this reader (``slurp_tfrl_v2_port_vote_score.py`` does, and writes ``construction-gain-i<t>.json``, which the Fold B stage prefers); when the vote column is not scored the gain is null and the stop word is UNDETERMINED, never a pass. The gain is never taken
               from the loop adapter (plan section 11 addendum ruling 6).

Stage ``read`` (one iteration, the Fold B selection flight; write-once ``read-i<t>.json`` and ``tfrl-v2-verdicts-i<t>.json``)

  The bank-consuming arms (``t-v2``) minus the delivering ``lock``, paired per row inside the flight: ``npc.paired_contrasts`` (utterance-cluster bootstrap, 10,000 draws, seed
  20261018) widened by the replicate variance of the request-identical copies {lock, aa, lock-rep2} (``paired_ci.noise_adjusted_contrast``), the NOMINATED rule of the port reader
  (pooled AND per-row lower bounds above zero, no resolved per-row loss, the intent guard), the twin admissibility ``|aa - lock| <= 1.2 pp`` and the iteration verdict:
  ACCEPTED only if NOMINATED, the per-row mean delta >= 0, the twin admissible and the construction gain passes the gate; UNDETERMINED while the construction gain is unknown.

The stop rule (:func:`construction_gate`, :func:`stop_decision`): the loop continues only while the construction gain is >= +1.0 pp (compared after rounding to 6 decimals, so
exactly +1.0 passes and +0.9 stops), at most 3 iterations, and an iteration that accepts nothing stops; a gain under +1.0 pp at iteration 1 stops the family.

    python -B scripts/analysis/slurp_tfrl_v2_port_read.py --stage construction --iteration 1 --data-root $SPEECHRL_DATA_DIR
    python -B scripts/analysis/slurp_tfrl_v2_port_read.py --stage read --iteration 1 --data-root $SPEECHRL_DATA_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _path in (str(_REPO / "src"), str(_HERE), str(_REPO / "scripts" / "flights" / "nx")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import l338_family as fam  # noqa: E402
import paired_ci  # noqa: E402
import slurp_direct_fold_b_core_read as base  # noqa: E402
import slurp_l338_column_read as hist  # noqa: E402
import slurp_nxt_thinking_policy_column_read as npc  # noqa: E402
import tfrl_v2_common as cm  # noqa: E402
import tfrl_v2_distiller as dd  # noqa: E402
import tfrl_v2_port_family as tf  # noqa: E402
import tfrl_v2_port_rails as rails  # noqa: E402
import tfrl_v2_sampler as smp  # noqa: E402

ReadError = hist.ReadError
CONSTRUCTION_SCHEMA = "saea-tfrl-v2-construction-read-v1"
READ_SCHEMA = "saea-tfrl-v2-fold-b-read-v1"
VERDICTS_SCHEMA = "saea-tfrl-v2-verdicts-v1"
VOTE_SCHEMA = "saea-tfrl-v2-vote-answers-v1"
WORD_CONTINUE, WORD_STOP, WORD_UNDETERMINED = "CONTINUE", "STOP", "UNDETERMINED"
WORD_ACCEPTED, WORD_NOT_ACCEPTED = "ACCEPTED", "NOT-ACCEPTED"
ADMISSIBLE, NOT_ADMISSIBLE = "ADMISSIBLE", "NOT-ADMISSIBLE"
SEED, DRAWS = rails.SEED, rails.DRAWS
STOP_GAIN_PP, MAX_ITERATIONS, DECIMALS = rails.STOP_GAIN_PP, rails.MAX_ITERATIONS, rails.GAIN_DECIMALS
VOTE_ARM = rails.VOTE_ARM
VOTE_MIN_SUPPORT = 2
SCORER = base.SCORER
COLUMN_SCORE_NAME = hist.COLUMN_SCORE_NAME


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# the recipe
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def load_recipe(path: Path) -> dict:
    """The T recipe, refused by name unless it binds the T ids, the scoring constants and the stop rule this reader implements."""

    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    problems: list[str] = []
    if recipe.get("schema") != rails.RECIPE_SCHEMA:
        problems.append(f"schema {recipe.get('schema')!r} != {rails.RECIPE_SCHEMA!r}")
    if recipe.get("read_run_id") != tf.READ_RUN_ID:
        problems.append(f"read_run_id {recipe.get('read_run_id')!r} != {tf.READ_RUN_ID!r}")
    if recipe.get("delivering") != tf.DELIVERING or recipe.get("twin") != tf.TWIN or recipe.get("replicates") != [tf.REPLICATE]:
        problems.append("delivering / twin / replicates are not lock / aa / [lock-rep2]")
    if recipe.get("bank_consuming_arms") != [tf.T_ARM]:
        problems.append(f"bank_consuming_arms {recipe.get('bank_consuming_arms')!r} != {[tf.T_ARM]!r}")
    if (recipe.get("arms") or {}).get("fold_b") != list(tf.FOLD_B_ARM_ORDER):
        problems.append("arms.fold_b is not the registered Fold B arm order")
    fold_b = recipe.get("fold_b_read") or {}
    boot = (fold_b.get("primary") or {}).get("bootstrap") or {}
    if (boot.get("seed"), boot.get("draws")) != (SEED, DRAWS):
        problems.append(f"bootstrap seed/draws {boot.get('seed')!r}/{boot.get('draws')!r} != {SEED}/{DRAWS}")
    if fold_b.get("floors_pp") != {"slu": rails.FLOOR_SLU_PP, "intent": rails.FLOOR_INTENT_PP}:
        problems.append("floors_pp are not the node-column floors")
    model = fold_b.get("noise_model") or {}
    if list(model.get("copies") or []) != [tf.DELIVERING, tf.TWIN, tf.REPLICATE] or model.get("formula") != paired_ci.FORMULA:
        problems.append("noise_model.copies / formula are not the port's")
    stop = recipe.get("stop_rule") or {}
    if (stop.get("construction_gain_pp"), stop.get("max_iterations")) != (STOP_GAIN_PP, MAX_ITERATIONS):
        problems.append("stop_rule is not +1.0 pp / 3 iterations")
    if problems:
        raise ReadError(f"the T recipe {path} is refused: " + "; ".join(problems))
    return recipe


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# the stop rule (pure)
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def construction_gate(gain_pp: float | None, *, threshold_pp: float = STOP_GAIN_PP) -> dict:
    """The construction-gain gate: CONTINUE at a gain >= the threshold, STOP under it, UNDETERMINED for an unknown gain. The comparison is made after rounding to 6 decimals, so
    +1.000000 passes (a float error never costs the pass) and +0.9 stops."""

    if gain_pp is None:
        return {"gain_pp": None, "rounded_pp": None, "threshold_pp": threshold_pp, "word": WORD_UNDETERMINED, "continue": None, "stop": None,
                "reason": "the vote column is not scored, so the construction gain is unknown"}
    rounded = round(float(gain_pp), DECIMALS)
    passes = rounded >= threshold_pp
    return {"gain_pp": float(gain_pp), "rounded_pp": rounded, "threshold_pp": threshold_pp, "word": WORD_CONTINUE if passes else WORD_STOP, "continue": passes, "stop": not passes,
            "reason": (f"construction gain {rounded:+.6f} pp is at or above +{threshold_pp} pp" if passes else f"construction gain {rounded:+.6f} pp is under +{threshold_pp} pp")}


def stop_decision(iteration: int, gain_pp: float | None, accepted: bool | None) -> dict:
    """Whether the loop may start iteration ``iteration + 1``. It may only when: the gain gate passes, the iteration accepted something, and ``iteration`` < 3. An unknown gain or an
    unknown acceptance is UNDETERMINED (no next iteration is authorised by a reading that is incomplete)."""

    gate = construction_gate(gain_pp)
    reasons: list[str] = []
    if iteration >= MAX_ITERATIONS:
        reasons.append(f"iteration {iteration} is the last of at most {MAX_ITERATIONS}")
    if gate["stop"]:
        reasons.append(gate["reason"] + ("; the family stops at iteration 1" if iteration == 1 else ""))
    if accepted is False:
        reasons.append("the iteration accepted nothing")
    undetermined = bool(not reasons and (gate["word"] == WORD_UNDETERMINED or accepted is None))
    return {"iteration": iteration, "gate": gate, "accepted": accepted, "stop": bool(reasons), "undetermined": undetermined, "reasons": reasons,
            "next_iteration_allowed": bool(not reasons and not undetermined)}


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# the vote over s1..s4 (pure, model-free)
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def _norm_entity(entity: Mapping[str, str]) -> tuple[str, str]:
    return (" ".join(entity["type"].split()).lower(), " ".join(entity["filler"].split()).lower())


def vote_answer(answers: Sequence[Mapping[str, Any]], *, min_support: int = VOTE_MIN_SUPPORT) -> dict:
    """The vote over the parsed answers of one row's trajectories (in k order): scenario_action by plurality (ties go to the intent first seen at the lowest k), an entity kept when at
    least ``min_support`` trajectories carry it (counted once per trajectory), the filler text of its first occurrence, in first-seen order."""

    if not answers:
        raise ReadError("a vote needs at least one answer")
    intents = [(a["scenario"], a["action"]) for a in answers]
    counts = Counter(intents)
    top = max(counts.values())
    winner = next(intent for intent in intents if counts[intent] == top)
    support: Counter = Counter()
    first: dict[tuple[str, str], Mapping[str, str]] = {}
    for answer in answers:
        for key in {_norm_entity(e) for e in answer["entities"]}:
            support[key] += 1
        for entity in answer["entities"]:
            first.setdefault(_norm_entity(entity), entity)
    kept = [{"type": first[key]["type"], "filler": first[key]["filler"]} for key in first if support[key] >= min_support]
    return {"scenario": winner[0], "action": winner[1], "entities": kept, "intent_votes": top, "n": len(answers)}


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# the construction read (pure over trace lines)
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def half_credit(half_roster: Mapping[str, Any], half: str, iteration: int, contact_lines: Sequence[Mapping], score_lines: Sequence[Mapping], *, k: int = cm.DEFAULT_K,
                scorer_id: str | None = None) -> dict:
    """The per-half construction numbers from one half's trace lines. Rows of other halves in the trace are ignored (and counted)."""

    items = smp._half_rows(half_roster, half)
    row_ids = [str(i["sample_id"]) for i in items]
    scenario_of = {str(i["sample_id"]): str(i["stratum"]) for i in items}
    in_half = set(row_ids)
    arms = smp.sampler_arms(k)
    answers = smp.solver_contacts([c for c in contact_lines if c["row_id"] in in_half], arms)
    scores = smp.scores_from_lines([s for s in score_lines if s["row_id"] in in_half], scorer_id=scorer_id)
    usable, skipped = [], []
    for row_id in row_ids:
        complete = (all((row_id, arm) in answers and (row_id, arm) in scores for arm in arms)
                    and (row_id, smp.BASELINE_ARM) in scores and (row_id, smp.TWIN_ARM) in scores)
        (usable if complete else skipped).append(row_id)
    credit = smp.credit_rows(usable, iteration, k, scores, twin_arm=smp.TWIN_ARM) if usable else []
    best_gain = [max(t["reward"] for t in c["trajectories"]) for c in credit]
    mean_reward = [sum(t["reward"] for t in c["trajectories"]) / len(c["trajectories"]) for c in credit]
    summary = smp.credit_summary(credit)
    votes = []
    for row_id in usable:
        record = vote_answer([answers[(row_id, arm)]["answer"] for arm in arms])
        votes.append({"schema": VOTE_SCHEMA, "row_id": row_id, "iteration": iteration, "half": half, **record})
    return {
        "half": half, "rows_in_half": len(row_ids), "rows_usable": len(usable), "rows_skipped": len(skipped), "skipped_first": skipped[:5],
        "rows_in_trace_outside_half": len({c["row_id"] for c in contact_lines} - in_half), "k": k,
        "positives": summary["positives"], "positive_rate": summary["positive_share"], "reasons": summary["reasons"],
        "mean_best_of_k_gain_pp": round(100.0 * sum(best_gain) / len(best_gain), 9) if best_gain else None,
        "mean_reward_pp": round(100.0 * sum(mean_reward) / len(mean_reward), 9) if mean_reward else None,
        "usable_row_ids": usable, "scenario_of": {r: scenario_of[r] for r in usable}, "votes": votes,
    }


def rule_survivors(rule_bank: Mapping[str, Any] | None) -> dict | None:
    if rule_bank is None:
        return None
    rules = rule_bank["rules"]
    return {"rules": len(rules), "rendered_tokens": rule_bank.get("rendered_tokens"), "token_measure": rule_bank.get("token_measure"),
            "by_kind": dict(sorted(Counter(r["candidate_key"].split("|")[0] for r in rules).items())), "support_counts": [r["support_count"] for r in rules]}


def exemplar_coverage(half_roster: Mapping[str, Any], exemplar_bank: Mapping[str, Any] | None) -> dict | None:
    """Per scenario stratum of the roster: rows, exemplars in the (union) bank, covered; plus the covered share of scenarios and of rows."""

    if exemplar_bank is None:
        return None
    rows: Counter = Counter(str(i["stratum"]) for i in half_roster["items"])
    by_scenario = exemplar_bank.get("by_scenario") or {}
    table = {s: {"rows": rows[s], "exemplars": len((by_scenario.get(s) or {}).get("entries") or []), "covered": bool((by_scenario.get(s) or {}).get("entries"))} for s in sorted(rows)}
    covered = [s for s, v in table.items() if v["covered"]]
    total_rows = sum(rows.values())
    return {"scenarios": len(table), "scenarios_covered": len(covered), "scenario_share": round(len(covered) / len(table), 6) if table else 0.0,
            "row_share": round(sum(table[s]["rows"] for s in covered) / total_rows, 6) if total_rows else 0.0,
            "scenarios_in_bank_outside_roster": sorted(set(by_scenario) - set(rows)), "by_scenario": table}


def construction_gain(vote_rows: Mapping[str, Mapping[str, float]] | None, lock_rows: Mapping[str, Mapping[str, float]] | None, ids: Sequence[str]) -> dict:
    """SLU-F1(vote) - SLU-F1(lock) in pp, pooled over ``ids`` (the usable rows of both halves), from the official per-row count tables; null when either table is absent."""

    if vote_rows is None or lock_rows is None or not ids:
        return {"gain_pp": None, "vote_slu_f1_pct": None, "lock_slu_f1_pct": None, "rows": len(ids),
                "note": "the `vote` column (and the lock column) must be scored by the official scorer on the derived vote copy; until then the gain is unknown"}
    missing = [i for i in ids if i not in vote_rows or i not in lock_rows]
    if missing:
        raise ReadError(f"{len(missing)} usable rows have no vote or lock count row (first: {missing[0]})")
    vote, lock = 100.0 * paired_ci.pooled_slu_f1(vote_rows, ids), 100.0 * paired_ci.pooled_slu_f1(lock_rows, ids)
    return {"gain_pp": vote - lock, "vote_slu_f1_pct": vote, "lock_slu_f1_pct": lock, "rows": len(ids), "note": "pooled over the usable rows of both halves"}


def construction_read(iteration: int, halves: Mapping[str, Mapping], *, banks: Mapping[str, Mapping | None] | None = None, vote_rows: Mapping | None = None,
                      lock_rows: Mapping | None = None) -> dict:
    """The construction read of one iteration. ``halves[half]`` is :func:`half_credit`'s record with the roster under ``"roster"``; ``banks`` holds the optional
    ``{half: {"rule_bank": ..}, "union": {"exemplar_bank": .., "rule_bank": ..}}`` documents."""

    banks = banks or {}
    out_halves, usable_ids = {}, []
    for half in cm.HALVES:
        record = halves[half]
        usable_ids += record["usable_row_ids"]
        bank = banks.get(half) or {}
        out_halves[half] = {key: value for key, value in record.items() if key not in ("usable_row_ids", "scenario_of", "votes", "roster")}
        out_halves[half]["rule_survivors"] = rule_survivors(bank.get("rule_bank"))
    union = banks.get("union") or {}
    for half in cm.HALVES:
        out_halves[half]["exemplar_coverage"] = exemplar_coverage(halves[half]["roster"], union.get("exemplar_bank"))
    gain = construction_gain(vote_rows, lock_rows, usable_ids)
    total_usable = sum(h["rows_usable"] for h in out_halves.values())
    total_pos = sum(h["positives"] for h in out_halves.values())
    return {
        "schema": CONSTRUCTION_SCHEMA, "status": "READING (development numbers; claims_benefit false)", "read_run_id": tf.READ_RUN_ID, "iteration": iteration,
        "halves": out_halves,
        "iteration_totals": {"rows_usable": total_usable, "positives": total_pos, "positive_rate": round(total_pos / total_usable, 6) if total_usable else None},
        "union_bank": None if not union else {"rule_survivors": rule_survivors(union.get("rule_bank")),
                                              "exemplar_scenarios": len((union.get("exemplar_bank") or {}).get("by_scenario") or {})},
        "construction_gain": gain, "stop": stop_decision(iteration, gain["gain_pp"], None),
    }


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# the Fold B read (pure over per-row count tables)
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def twin_admissibility(contrast: Mapping) -> dict:
    twin = (contrast.get("versus_delivering") or {}).get(tf.TWIN)
    if twin is None:
        return {"word": NOT_ADMISSIBLE, "reason": "the run has no twin arm aa", "threshold_pp": rails.TWIN_BAR_PP}
    point = twin["slu_f1"]["point_pp"]
    magnitude = round(abs(point), DECIMALS)
    return {"word": ADMISSIBLE if magnitude <= rails.TWIN_BAR_PP else NOT_ADMISSIBLE, "aa_minus_lock_slu_pp": point, "abs_rounded": magnitude, "threshold_pp": rails.TWIN_BAR_PP}


def bank_arm_words(contrast: Mapping, bank_arms: Sequence[str], *, floor_slu_pp: float = rails.FLOOR_SLU_PP, floor_intent_pp: float = rails.FLOOR_INTENT_PP) -> dict:
    """The within-run word of every bank-consuming arm over the WIDENED CI95 (``contrast`` from ``paired_ci.noise_adjusted_contrast``):

    NOMINATED  pooled SLU point >= the SLU floor AND pooled and per-row SLU lower bounds > 0, no resolved per-row loss (per-row upper bound >= -floor), the intent guard (intent lower
               bound >= -intent floor and intent point >= 0);
    HARM       the pooled SLU or intent upper bound is below minus its floor;
    NEUTRAL    otherwise. The per-row F1 is a repo diagnostic, never an official SLURP number (plan section 10 ruling 4)."""

    versus = contrast["versus_delivering"]
    words = {}
    for arm in bank_arms:
        if arm not in versus:
            raise ReadError(f"{arm}: no paired contrast against {contrast['delivering']}")
        slu, intent, per_row = versus[arm]["slu_f1"], versus[arm]["intent"], versus[arm].get("per_row_slu_f1")
        if per_row is None:
            raise ReadError(f"{arm}: the contrast has no per_row_slu_f1 delta; run it through paired_ci.noise_adjusted_contrast")
        conditions = {
            "slu_point_at_or_above_floor": slu["point_pp"] >= floor_slu_pp,
            "slu_ci_lo_above_zero": slu["ci95_lo_pp"] > 0,
            "per_row_slu_ci_lo_above_zero": per_row["ci95_lo_pp"] > 0,
            "per_row_slu_no_resolved_loss": per_row["ci95_hi_pp"] >= -floor_slu_pp,
            "intent_guard_ci_lo": intent["ci95_lo_pp"] >= -floor_intent_pp,
            "intent_point_non_negative": intent["point_pp"] >= 0,
        }
        harm = slu["ci95_hi_pp"] < -floor_slu_pp or intent["ci95_hi_pp"] < -floor_intent_pp
        word = "NOMINATED" if all(conditions.values()) else "HARM" if harm else "NEUTRAL"
        words[arm] = {"word": word, "conditions": conditions, "failed": [name for name, ok in conditions.items() if not ok],
                      "slu_delta_pp": slu["point_pp"], "slu_ci95": [slu["ci95_lo_pp"], slu["ci95_hi_pp"]],
                      "per_row_slu_delta_pp": per_row["point_pp"], "per_row_slu_ci95": [per_row["ci95_lo_pp"], per_row["ci95_hi_pp"]],
                      "intent_delta_pp": intent["point_pp"], "intent_ci95": [intent["ci95_lo_pp"], intent["ci95_hi_pp"]]}
    return {"floors_pp": {"slu": floor_slu_pp, "intent": floor_intent_pp}, "ci_definition": contrast.get("ci_definition"), "words": words}


def iteration_verdict(words: Mapping, twin: Mapping, gain_pp: float | None, arm: str = tf.T_ARM) -> dict:
    """ACCEPTED only if the arm is NOMINATED, its per-row mean delta is >= 0, the twin is admissible and the construction gain passes the gate; UNDETERMINED while the gain is unknown
    and nothing else already refuses."""

    entry = words["words"][arm]
    gate = construction_gate(gain_pp)
    conditions = {"nominated": entry["word"] == "NOMINATED", "per_row_mean_delta_non_negative": entry["per_row_slu_delta_pp"] >= 0,
                  "twin_admissible": twin["word"] == ADMISSIBLE, "construction_gain_gate": gate["continue"]}
    if any(v is False for v in conditions.values()):
        word = WORD_NOT_ACCEPTED
    elif any(v is None for v in conditions.values()):
        word = WORD_UNDETERMINED
    else:
        word = WORD_ACCEPTED
    return {"arm": arm, "word": word, "conditions": conditions, "within_run_word": entry["word"], "construction_gain": gate}


def foldb_read(scored: Mapping[str, Mapping[str, Mapping[str, float]]], ids: Sequence[str], cluster_of, *, iteration: int, gain_pp: float | None, seed: int = SEED,
               draws: int = DRAWS) -> dict:
    """The Fold B read of one iteration over the per-row official count tables of the flight's arms (``scored[arm][sample_id][field]``)."""

    missing = [a for a in tf.FOLD_B_ARM_ORDER if a not in scored]
    if missing:
        raise ReadError(f"missing scored columns for {missing}")
    bootstrap = npc.paired_contrasts({a: scored[a] for a in tf.FOLD_B_ARM_ORDER}, list(ids), cluster_of, seed=seed, draws=draws, delivering=tf.DELIVERING)
    contrast = paired_ci.noise_adjusted_contrast(scored, list(ids), cluster_of, bootstrap, copies=(tf.DELIVERING, tf.TWIN, tf.REPLICATE), delivering=tf.DELIVERING, seed=seed, draws=draws)
    words = bank_arm_words(contrast, [tf.T_ARM])
    twin = twin_admissibility(contrast)
    verdict = iteration_verdict(words, twin, gain_pp)
    return {"schema": READ_SCHEMA, "status": "READING (development numbers; claims_benefit false)", "read_run_id": tf.READ_RUN_ID, "iteration": iteration, "n": len(ids),
            "contrast": contrast, "contrast_bootstrap_only": bootstrap, "arm_words": words, "twin_admissibility": twin, "verdict": verdict,
            "stop": stop_decision(iteration, gain_pp, None if verdict["word"] == WORD_UNDETERMINED else verdict["word"] == WORD_ACCEPTED),
            "interval": {"kind": "utterance-cluster bootstrap widened by the replicate variance (noise model M1)", "draws": draws, "seed": seed, "level": "CI95",
                         "formula": paired_ci.FORMULA, "copies": [tf.DELIVERING, tf.TWIN, tf.REPLICATE]}}


def verdicts_document(read: Mapping, construction: Mapping | None, inputs: Mapping) -> dict:
    """``tfrl-v2-verdicts-i<t>.json``: the verdict, the stop decision and the numbers behind them, with the input files and their hashes."""

    t_arm = read["contrast"]["versus_delivering"][tf.T_ARM]
    return {
        "schema": VERDICTS_SCHEMA, "status": "TFRL-V2 VERDICTS (development numbers; claims_benefit false; a reading, never an admission)", "read_run_id": tf.READ_RUN_ID,
        "iteration": read["iteration"], "inputs": dict(inputs), "verdict": read["verdict"], "stop": read["stop"], "twin_admissibility": read["twin_admissibility"],
        "arm": {"name": tf.T_ARM, "slu_f1_pct": read["contrast"]["levels"][tf.T_ARM]["slu_f1_pct"], "intent_pct": read["contrast"]["levels"][tf.T_ARM]["intent_pct"],
                "per_row_slu_f1_pct": read["contrast"]["levels"][tf.T_ARM]["per_row_slu_f1_pct"], "slu_delta_vs_lock": t_arm["slu_f1"], "intent_delta_vs_lock": t_arm["intent"],
                "per_row_slu_delta_vs_lock": t_arm["per_row_slu_f1"]},
        "construction": None if construction is None else {"gain": construction["construction_gain"], "iteration_totals": construction["iteration_totals"]},
        "noise_model": read["contrast"]["noise_model"],
    }


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# the sealed-run adapter and the CLI
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def _input(path: Path, root: Path | None = None) -> dict:
    p = Path(path)
    return {"path": str(p.relative_to(root)) if root and p.is_relative_to(root) else str(p), "sha256": base.sha256_file(p)}


def _json_text(doc: Any) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def trace_path(data_root: Path, iteration: int, half: str) -> Path:
    return Path(data_root) / "analysis" / tf.trace_read_id(iteration, half.split("-")[1]) / "rollout-trace.jsonl"


def columns_dir(data_root: Path, iteration: int, half: str) -> Path:
    return Path(data_root) / "analysis" / tf.trace_read_id(iteration, half.split("-")[1]) / "columns"


def _load_scores(directory: Path, arm: str, scorer_id: str) -> dict | None:
    path = Path(directory) / COLUMN_SCORE_NAME.format(arm=arm, scorer=scorer_id)
    return None if not path.is_file() else base.rows_table(json.loads(path.read_text(encoding="utf-8")), scorer_id)


def _load_bank(analysis_root: Path, iteration: int, name: str) -> dict | None:
    path = Path(analysis_root) / "v2" / "banks" / f"i{iteration}" / name / "bank-receipt.json"
    if not path.is_file():
        return None
    receipt, exemplars, rules = dd.load_bank(path)
    return {"receipt": receipt, "exemplar_bank": exemplars, "rule_bank": rules, "path": path}


def _write_once(path: Path, text: str) -> None:
    if Path(path).exists():
        raise FileExistsError(path)
    Path(path).write_text(text, encoding="utf-8")


def run_construction(args: argparse.Namespace, root: Path, recipe: Mapping) -> tuple[dict, dict[str, str]]:
    halves, inputs, contacts_all = {}, {}, {}
    for half in cm.HALVES:
        path = trace_path(args.data_root, args.iteration, half)
        if not path.is_file():
            raise ReadError(f"missing sealed trace {path}")
        roster_path = args.repo_root / cm.HALF_ROSTER_RELPATH[half]
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        contacts, scores, _ = smp.read_trace([path])
        record = half_credit(roster, half, args.iteration, contacts, scores, scorer_id=args.trace_scorer_id)
        record["roster"] = roster
        halves[half] = record
        inputs[f"trace:{half}"] = _input(path, args.data_root)
        inputs[f"roster:{half}"] = _input(roster_path, args.repo_root)
    banks = {}
    for name in (*cm.HALVES, "union"):
        loaded = _load_bank(root, args.iteration, name)
        if loaded is not None:
            banks[name] = loaded
            inputs[f"bank_receipt:{name}"] = _input(loaded["path"], args.data_root)
    vote_rows, lock_rows = {}, {}
    for half in cm.HALVES:
        directory = columns_dir(args.data_root, args.iteration, half)
        vote, lock = _load_scores(directory, VOTE_ARM, args.scorer), _load_scores(directory, tf.DELIVERING, args.scorer)
        if vote is None or lock is None:
            vote_rows = lock_rows = None
            break
        vote_rows.update(vote)
        lock_rows.update(lock)
        inputs[f"column_scores:{half}:{VOTE_ARM}"] = _input(directory / COLUMN_SCORE_NAME.format(arm=VOTE_ARM, scorer=args.scorer), args.data_root)
        inputs[f"column_scores:{half}:{tf.DELIVERING}"] = _input(directory / COLUMN_SCORE_NAME.format(arm=tf.DELIVERING, scorer=args.scorer), args.data_root)
    result = construction_read(args.iteration, halves, banks=banks, vote_rows=vote_rows, lock_rows=lock_rows)
    result["inputs"] = inputs
    vote_texts = {half: "".join(json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for v in halves[half]["votes"]) for half in cm.HALVES}
    return result, vote_texts


def run_read(args: argparse.Namespace, root: Path, recipe: Mapping) -> tuple[dict, dict]:
    cell = tf.CELLS["fb"]
    paths = base.run_paths(args.data_root, cell["run"], cell["attempt"])
    for name in ("trace", "outputs", "manifest"):
        if not paths[name].exists():
            raise ReadError(f"missing sealed file {paths[name]}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if str(manifest.get("status", "")).upper() != "COMPLETED":
        raise ReadError(f"manifest status is {manifest.get('status')!r}, not COMPLETED")
    columns = root / "columns"
    scored = {}
    for arm in tf.FOLD_B_ARM_ORDER:
        rows = _load_scores(columns, arm, args.scorer)
        if rows is None:
            raise ReadError(f"missing column score file {columns / COLUMN_SCORE_NAME.format(arm=arm, scorer=args.scorer)}")
        scored[arm] = rows
    roster_path = args.repo_root / tf.ROSTERS["fb"]
    ids, cluster_of, _ = base.roster_maps(roster_path)
    for arm, rows in scored.items():
        if set(rows) != set(ids):
            raise ReadError(f"{arm}: the per-sample id set is not the frozen Fold B roster's")
    construction_path = root / f"construction-i{args.iteration}.json"
    construction = json.loads(construction_path.read_text(encoding="utf-8")) if construction_path.is_file() else None
    gain = None if construction is None else construction["construction_gain"]["gain_pp"]
    # construction-i<t>.json is write-once and carries a null gain when it was written before the `vote` column was scored; the vote scorer
    # (slurp_tfrl_v2_port_vote_score.py) then writes construction-gain-i<t>.json, which takes precedence (its gain is the same function over the scored columns).
    gain_path = root / f"construction-gain-i{args.iteration}.json"
    scored_gain = json.loads(gain_path.read_text(encoding="utf-8")) if gain_path.is_file() else None
    if scored_gain is not None:
        gain = scored_gain["construction_gain"]["gain_pp"]
        if construction is not None:
            construction = dict(construction, construction_gain=scored_gain["construction_gain"], stop=scored_gain["stop"])
    read = foldb_read(scored, ids, cluster_of, iteration=args.iteration, gain_pp=gain, seed=args.seed, draws=args.draws)
    read["sealed_hashes"] = {n: base.sha256_file(paths[n]) for n in ("trace", "outputs", "manifest")}
    inputs = {"recipe": _input(args.recipe_path, args.repo_root), "roster": _input(roster_path, args.repo_root), "manifest": _input(paths["manifest"], args.data_root),
              "trace": _input(paths["trace"], args.data_root), "outputs": _input(paths["outputs"], args.data_root)}
    for arm in tf.FOLD_B_ARM_ORDER:
        inputs[f"column_scores:{arm}"] = _input(columns / COLUMN_SCORE_NAME.format(arm=arm, scorer=args.scorer), args.data_root)
    if construction is not None:
        inputs["construction_read"] = _input(construction_path, args.data_root)
    if scored_gain is not None:
        inputs["construction_gain"] = _input(gain_path, args.data_root)
    read_text = _json_text(read)
    inputs["read_json"] = {"path": f"read-i{args.iteration}.json", "sha256": hashlib.sha256(read_text.encode("utf-8")).hexdigest()}
    return read, verdicts_document(read, construction, inputs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("construction", "read"), required=True)
    parser.add_argument("--iteration", type=int, default=tf.ITERATION)
    parser.add_argument("--data-root", type=Path, default=Path(hist._data_root_env()))
    parser.add_argument("--repo-root", type=Path, default=_REPO)
    parser.add_argument("--analysis-root", type=Path, default=None, help=f"default <data root>/{rails.ANALYSIS_ROOT}")
    parser.add_argument("--recipe", type=Path, default=Path(rails.RECIPE_RELPATH), help="the T analysis recipe (repo-relative or absolute)")
    parser.add_argument("--scorer", default=SCORER)
    parser.add_argument("--trace-scorer-id", default=None, help="the scorer id of the trace score lines when several scorers wrote them")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--draws", type=int, default=DRAWS)
    args = parser.parse_args(argv)
    args.recipe_path = args.recipe if args.recipe.is_absolute() else args.repo_root / args.recipe
    root = args.analysis_root or (args.data_root / rails.ANALYSIS_ROOT)
    try:
        if not 1 <= args.iteration <= MAX_ITERATIONS:
            raise ReadError(f"iteration must be in 1..{MAX_ITERATIONS}")
        recipe = load_recipe(args.recipe_path)
        outs = ([root / f"construction-i{args.iteration}.json", *[root / f"vote-answers-i{args.iteration}-{h.split('-')[1]}.jsonl" for h in cm.HALVES]] if args.stage == "construction"
                else [root / f"read-i{args.iteration}.json", root / f"tfrl-v2-verdicts-i{args.iteration}.json"])
        for out in outs:
            if out.exists():
                print(f"REFUSED: {out} exists (write-once)", file=sys.stderr)
                return 6
        root.mkdir(parents=True, exist_ok=True)
        if args.stage == "construction":
            result, vote_texts = run_construction(args, root, recipe)
            result["inputs"]["recipe"] = _input(args.recipe_path, args.repo_root)
            _write_once(outs[0], _json_text(result))
            for half, out in zip(cm.HALVES, outs[1:]):
                _write_once(out, vote_texts[half])
            print(json.dumps({"written": [str(o) for o in outs], "gain_pp": result["construction_gain"]["gain_pp"], "gate": result["stop"]["gate"]["word"]}))
            return 0
        read, verdicts = run_read(args, root, recipe)
        _write_once(outs[0], _json_text(read))
        _write_once(outs[1], _json_text(verdicts))
        print(json.dumps({"written": [str(o) for o in outs], "verdict": read["verdict"]["word"], "within_run": read["arm_words"]["words"][tf.T_ARM]["word"],
                          "twin": read["twin_admissibility"]["word"], "next_iteration_allowed": read["stop"]["next_iteration_allowed"]}))
        return 0
    except ReadError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
