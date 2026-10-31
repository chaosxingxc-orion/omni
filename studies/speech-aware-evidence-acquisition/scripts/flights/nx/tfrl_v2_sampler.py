#!/usr/bin/env python3
"""tfrl_v2_sampler.py -- the K-sample REQUEST PLAN and the per-row CREDIT of the TFRL-v2 loop (plan 2026-10-30 section 4, task T3).

MODEL-FREE. This module contacts no model, server, tokenizer, encoder or network and constructs no client: it BUILDS the plan a registered
flight will execute and it READS the sealed outputs (trace ``contact`` / ``score`` lines, official per-row scores) that flight leaves behind. The
flight itself is the registered construction cell (program ``phase-native-module-v1-c7``: one Perception contact, then the Solver arm group
``s1``..``sK`` plus the twin ``aa`` and the delivering ``lock``); the runner consumes that cell, not this file. The run-list written here is the
pre-registration document and the per-contact table the cell's writer (task T4) derives the arm group from.

What it builds, for ONE half and ONE iteration:

* K = 4 Solver trajectories per row. Decoding seeds are ``derive_seed(iteration seed, row id, k)``: distinct per k, a pure function of the inputs.
  Default variation mode ``seed-only``: the K Solver requests of a row are REQUEST-IDENTICAL except the seed (every request field is pinned by
  ``request_identity_sha256``, which excludes only the seed, the arm label and the contact id). Mode ``rule-variants`` additionally gives trajectory
  k the k-th consumption variant of a seed rule bank (plan section 4 "varied consumption instructions and evidence orderings"; plan section 11
  addendum ruling 5: k=0 no rule bank, k=1 the full bank, k=2 the rules of the row's scenario only, k=3 the top half of the bank by support); it is
  opt-in, needs exactly K = 4, and records on every Solver contact a ``rule_variant`` field ``{variant_id, kind, rule_ids, content_sha256}``
  (the field is absent in ``seed-only`` plans, whose bytes are unchanged).
* ONE Perception contact per row, shared by the K Solver contacts.
* The bank pin (a bank counted with an estimate, or whose exemplars do not quote the resolved item text, is refused): from iteration 2 the half consumes the bank distilled from the OTHER half; :func:`tfrl_v2_common.check_cross_fit` refuses a bank
  whose source half is the consuming half (plan section 9 ruling 3). Exemplars REPLACE supplied items (the T supply bound is 43 items plus an
  exemplar block of at most 2,000 tokens); the rule bank is Solver instruction text of at most 400 tokens.
* A dry-run budget line: calls, audio seconds (the L338 convention: the row's audio seconds counted once per generation contact), completion
  tokens at the median 4,372 per contact, and card-hours at the host's measured 1,422.59 output tokens per second on 8 cards.

What it credits (``credit_rows``): per row, the official per-row SLU-F1 of each trajectory (the scorer's ``slu_f1`` per-row value carried by the
trace ``score`` lines, or, from a count table, ``paired_ci.row_f1``; never a new metric) and its intent match (``scenario_action_exact``).
Reward of a trajectory = its row F1 minus the iteration's baseline trajectory (arm ``lock``) row F1. A row is POSITIVE when the best
intent-correct trajectory beats the baseline by more than 0 (ties on F1 go to the lowest k); with a twin supplied, rows where lock and twin differ
are not counted (the baseline is unstable there, plan cell draft section 4).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts" / "analysis"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from speech_aware_evidence_acquisition.contracts import rollout_trace as rt  # noqa: E402
import paired_ci  # noqa: E402  (the repo's per-row F1 helper)
import tfrl_v2_common as cm  # noqa: E402

RUN_LIST_SCHEMA = "saea-tfrl-v2-run-list-v1"
CREDIT_SCHEMA = "saea-tfrl-v2-row-credit-v1"
PROGRAM_ID = "phase-native-module-v1-c7"
SAMPLER_ARM_PREFIX = "s"
BASELINE_ARM = "lock"
TWIN_ARM = "aa"
VARIATIONS = ("seed-only", "rule-variants")
#: ``rule-variants`` gives trajectory k the k-th variant of the seed rule bank, so it needs exactly K = 4 variants (plan section 11 addendum, ruling 5:
#: k=0 no rule bank, k=1 the full bank, k=2 the rules of the row's scenario only, k=3 the top half of the bank by support).
RULE_VARIANT_COUNT = 4
RULE_VARIANT_KINDS = ("none", "full", "scenario", "top-half")
BASELINE_TOLERANCE = 1e-9
GAIN_EPSILON = 1e-12


class SamplerError(ValueError):
    """A refusal of the sampler: bad inputs, a missing score, a bank that fails the cross-fit guard."""


def sampler_arms(k: int) -> list[str]:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise SamplerError("K must be a positive integer")
    return [f"{SAMPLER_ARM_PREFIX}{i + 1}" for i in range(k)]


# ----------------------------------------------------------------------------------------------------------------------------------------
# identities
# ----------------------------------------------------------------------------------------------------------------------------------------


def half_suffix(half: str) -> str:
    cm.other_half(half)                                  # validates the half id
    return half.split("-")[1]                            # "H1" / "H2"


def run_id_for(half: str, iteration: int) -> str:
    return f"SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I{int(iteration)}-{half_suffix(half)}-NPU-PORT-V1"


def attempt_id_for(half: str, iteration: int) -> str:
    return f"attempt-slurp-tfrl-v2-construction-r01c-i{int(iteration)}-{half_suffix(half).lower()}-generation-npu-port-v1"


def trajectory_id(iteration: int, row_id: str, k: int) -> str:
    return f"tfrl-v2/i{int(iteration)}/{row_id}/k{int(k)}"


# ----------------------------------------------------------------------------------------------------------------------------------------
# budget
# ----------------------------------------------------------------------------------------------------------------------------------------


def budget(audio_seconds: Sequence[float], k: int = cm.DEFAULT_K, *, include_twin_and_lock: bool = False,
           median_completion_tokens: int = cm.MEDIAN_COMPLETION_TOKENS, host_tok_s: float = cm.HOST_OUTPUT_TOK_S,
           cards: int = cm.HOST_CARDS, margin: float = cm.MARGIN) -> dict[str, Any]:
    """The dry-run budget of one pass over rows with the given audio seconds.

    Contacts per row = 1 Perception + K Solver (+ the twin ``aa`` and the delivering ``lock`` when ``include_twin_and_lock``: program c7 as a
    whole). ``calls`` is the planned count; ``calls_with_fallback_ceiling`` doubles it (the registered ceiling adds one answer-only fallback per
    budgeted-thinking contact). Audio seconds follow the L338 registration: the row's audio seconds counted once per generation contact, rounded up.
    Tokens = calls x the median completion tokens (applied to every contact, Perception included: an upper-leaning simplification). Card-hours =
    tokens / host tok/s / 3600 x cards (the host figure is the aggregate of all 8 cards, so wall-clock hours on the host x 8 cards).
    """

    sampler_arms(k)
    rows = len(audio_seconds)
    contacts_per_row = 1 + k + (2 if include_twin_and_lock else 0)
    calls = rows * contacts_per_row
    seconds = math.ceil(round(sum(float(s) for s in audio_seconds) * contacts_per_row, 9))
    tokens = calls * int(median_completion_tokens)
    wall_hours = tokens / host_tok_s / 3600.0
    card_hours = wall_hours * cards
    return {
        "rows": rows, "k": k, "contacts_per_row": contacts_per_row, "includes_twin_and_lock": include_twin_and_lock,
        "calls": calls, "calls_with_fallback_ceiling": 2 * calls, "audio_seconds": seconds,
        "median_completion_tokens": int(median_completion_tokens), "tokens": tokens,
        "host_tok_s": host_tok_s, "cards": cards, "host_wall_hours": round(wall_hours, 4), "card_hours": round(card_hours, 4),
        "card_hours_with_margin": round(card_hours * margin, 4), "margin": margin,
    }


def budget_line(label: str, b: Mapping[str, Any]) -> str:
    return (f"{label}: {b['rows']} rows x {b['contacts_per_row']} contacts (K={b['k']}"
            f"{', with aa+lock' if b['includes_twin_and_lock'] else ''}) = {b['calls']} calls "
            f"({b['calls_with_fallback_ceiling']} at the fallback ceiling), {b['audio_seconds']} audio s, "
            f"{b['tokens']} tokens at {b['median_completion_tokens']}/contact, {b['host_wall_hours']} h host wall at {b['host_tok_s']} tok/s, "
            f"{b['card_hours']} card-hours on {b['cards']} cards ({b['card_hours_with_margin']} with the {b['margin']} margin)")


def sum_budgets(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Add the budgets of several halves of the same shape (counts add; the derived hours are re-derived, not summed from rounded parts)."""

    if not parts:
        raise SamplerError("nothing to add")
    first = parts[0]
    for part in parts:
        if (part["k"], part["contacts_per_row"], part["median_completion_tokens"], part["host_tok_s"], part["cards"]) != (
                first["k"], first["contacts_per_row"], first["median_completion_tokens"], first["host_tok_s"], first["cards"]):
            raise SamplerError("budgets of different shapes cannot be added")
    tokens = sum(p["tokens"] for p in parts)
    wall = tokens / first["host_tok_s"] / 3600.0
    return {**first, "rows": sum(p["rows"] for p in parts), "calls": sum(p["calls"] for p in parts),
            "calls_with_fallback_ceiling": sum(p["calls_with_fallback_ceiling"] for p in parts),
            "audio_seconds": sum(p["audio_seconds"] for p in parts), "tokens": tokens, "host_wall_hours": round(wall, 4),
            "card_hours": round(wall * first["cards"], 4), "card_hours_with_margin": round(wall * first["cards"] * first["margin"], 4)}


# ----------------------------------------------------------------------------------------------------------------------------------------
# the request plan
# ----------------------------------------------------------------------------------------------------------------------------------------


def resolve_rule_variant(variant: Mapping[str, Any], scenario: str | None) -> dict[str, Any]:
    """The variant one row's trajectory carries: ``{"variant_id", "kind", "rule_ids", "content_sha256"}``.

    A bank variant (``kind`` in :data:`RULE_VARIANT_KINDS`, built by ``tfrl_v2_distiller.rule_variants_seed_bank``) carries ``rule_ids`` and
    ``rendered_text``; the ``scenario`` kind carries them per scenario in ``by_scenario`` and is resolved with the row's scenario (a row whose scenario
    has no rule gets the empty set). ``content_sha256`` is taken over the id, the kind, the resolved rule ids and the resolved rendered text, so two
    rows with the same variant and the same rules carry the same hash. Any other variant (no ``kind``) is opaque: its hash is over the whole variant.
    """

    kind = variant.get("kind")
    if kind not in RULE_VARIANT_KINDS:
        return {"variant_id": variant["variant_id"], "kind": "opaque", "rule_ids": None, "content_sha256": cm.content_sha256(dict(variant))}
    if kind == "scenario":
        if not scenario:
            raise SamplerError("the 'scenario' rule variant needs the row's scenario (the roster item's stratum); this row has none")
        chosen = (variant.get("by_scenario") or {}).get(scenario) or {"rule_ids": [], "rendered_text": ""}
    else:
        chosen = variant
    rule_ids, rendered = list(chosen.get("rule_ids") or []), chosen.get("rendered_text") or ""
    return {"variant_id": variant["variant_id"], "kind": kind, "rule_ids": rule_ids,
            "content_sha256": cm.content_sha256({"variant_id": variant["variant_id"], "kind": kind, "rule_ids": rule_ids, "rendered_text": rendered})}


def _half_rows(half_roster: Mapping[str, Any], half: str) -> list[Mapping[str, Any]]:
    expected = cm.HALF_ROSTER_ID[half]
    if half_roster.get("roster_id") != expected:
        raise SamplerError(f"the roster is {half_roster.get('roster_id')!r}, expected {expected!r} for half {half}")
    items = half_roster.get("items")
    if not isinstance(items, list) or not items:
        raise SamplerError("the half roster has no items")
    ids = [str(i["sample_id"]) for i in items]
    if len(set(ids)) != len(ids):
        raise SamplerError("the half roster repeats a sample_id")
    return items


def build_plan(
    half_roster: Mapping[str, Any], half: str, iteration: int, *, k: int = cm.DEFAULT_K, base_seed: int = cm.DEFAULT_BASE_SEED,
    temperature: float = cm.DEFAULT_TEMPERATURE, top_p: float = cm.DEFAULT_TOP_P, top_k: int = cm.DEFAULT_TOP_K,
    variation: str = "seed-only", bank_receipt: Mapping[str, Any] | None = None, seed_rule_bank: Mapping[str, Any] | None = None,
    include_twin_and_lock: bool = True, roster_file_sha256: str | None = None,
    execution_run_id: str | None = None, execution_attempt_id: str | None = None,
) -> dict[str, Any]:
    """The run-list of one half and one iteration (a plain document; deterministic: same inputs, same bytes).

    ``bank_receipt`` is the receipt of the bank this half consumes (``None`` exactly at iteration 1). ``seed_rule_bank`` is an optional mapping with
    ``variants`` (``[{"variant_id", "instruction", "ordering"}]``), required by ``variation='rule-variants'``.
    """

    if not isinstance(iteration, int) or isinstance(iteration, bool) or not 1 <= iteration <= 3:
        raise SamplerError("iteration must be 1, 2 or 3")
    if variation not in VARIATIONS:
        raise SamplerError(f"unknown variation {variation!r}; known: {list(VARIATIONS)}")
    arms = sampler_arms(k)
    items = _half_rows(half_roster, half)
    row_ids = [str(i["sample_id"]) for i in items]

    if iteration == 1:
        if bank_receipt is not None:
            raise SamplerError("iteration 1 consumes no distilled bank")
    else:
        if bank_receipt is None:
            raise SamplerError(f"iteration {iteration} consumes the bank distilled from the other half; pass its receipt")
        try:
            cm.check_cross_fit(bank_receipt, half, consuming_row_ids=row_ids, consuming_split_identity=half_roster.get("split_identity_hash"))
        except cm.CrossFitError as error:
            raise SamplerError(str(error)) from error
        try:
            cm.check_bank_measured(bank_receipt)
        except cm.BankNotMeasuredError as error:
            raise SamplerError(str(error)) from error

    variants: list[Mapping[str, Any]] = []
    if variation == "rule-variants":
        if k != RULE_VARIANT_COUNT:
            raise SamplerError(f"variation 'rule-variants' needs K = {RULE_VARIANT_COUNT} (one variant per trajectory); K is {k}")
        variants = list((seed_rule_bank or {}).get("variants") or [])
        if not variants:
            raise SamplerError("variation 'rule-variants' needs a seed rule bank with at least one variant")
        if len(variants) != RULE_VARIANT_COUNT:
            raise SamplerError(f"variation 'rule-variants' needs exactly {RULE_VARIANT_COUNT} variants; the seed rule bank has {len(variants)}")
        if len({v["variant_id"] for v in variants}) != len(variants):
            raise SamplerError("seed rule bank variant ids must be unique")

    it_seed = cm.iteration_seed(base_seed, iteration)
    run_id, attempt_id = execution_run_id or run_id_for(half, iteration), execution_attempt_id or attempt_id_for(half, iteration)
    banks = {
        "rule_bank_sha256": bank_receipt["rule_bank"]["content_sha256"] if bank_receipt else None,
        "exemplar_bank_sha256": bank_receipt["exemplar_bank"]["content_sha256"] if bank_receipt else None,
        "bank_receipt_sha256": cm.content_sha256(bank_receipt) if bank_receipt else None,
        "bank_source_halves": list(bank_receipt["source_halves"]) if bank_receipt else [],
        "bank_measure": bank_receipt["measure"] if bank_receipt else None,
        "exemplar_block_token_cap": cm.EXEMPLAR_BLOCK_TOKEN_CAP, "rule_bank_token_cap": cm.RULE_BANK_TOKEN_CAP,
        "supply_bound_items": 43 if bank_receipt else 51,
        "seed_rule_bank_sha256": cm.content_sha256(dict(seed_rule_bank)) if seed_rule_bank else None,
    }
    decoding = {"temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k)}

    rows_doc: list[dict[str, Any]] = []
    for item in items:
        row_id = str(item["sample_id"])
        contacts: list[dict[str, Any]] = [{
            "contact_id": f"{run_id}/{row_id}/perception", "node": "perception", "arm": None, "shared_by": list(arms),
        }]
        seeds: list[int] = []
        for index, arm in enumerate(arms):
            seed = cm.derive_seed(it_seed, row_id, index)
            seeds.append(seed)
            variant = variants[index] if variants else None
            resolved = resolve_rule_variant(variant, item.get("stratum")) if variant else None
            identity = cm.content_sha256({
                "node": "solver", "row_id": row_id, "run_id": run_id, "decoding": decoding, "banks": {kk: banks[kk] for kk in ("rule_bank_sha256", "exemplar_bank_sha256")},
                "variant": resolved, "perception": f"{run_id}/{row_id}/perception",
            })
            contact = {
                "contact_id": f"{run_id}/{row_id}/solver/{arm}", "node": "solver", "arm": arm, "sample_index": index, "seed": seed,
                "trajectory_id": trajectory_id(iteration, row_id, index), "variant_id": variant["variant_id"] if variant else None,
                "request_identity_sha256": identity,
            }
            if resolved is not None:
                contact["rule_variant"] = resolved
            contacts.append(contact)
        if len(set(seeds)) != len(seeds):
            raise SamplerError(f"row {row_id}: the K seeds are not distinct (change the base seed)")
        rows_doc.append({"row_id": row_id, "audio_seconds": float(item["audio_seconds"]), "contacts": contacts})

    audio = [r["audio_seconds"] for r in rows_doc]
    sampler_budget = budget(audio, k)
    program_budget = budget(audio, k, include_twin_and_lock=True)
    return {
        "schema": RUN_LIST_SCHEMA, "loop_id": cm.LOOP_ID, "run_id": run_id, "attempt_id": attempt_id, "half": half,
        "iteration": iteration, "k": k, "variation": variation,
        "roster": {"roster_id": half_roster["roster_id"], "split_identity_hash": half_roster["split_identity_hash"], "rows": len(rows_doc),
                   "file_sha256": roster_file_sha256},
        "program": {
            "program_id": PROGRAM_ID, "perception_contacts_per_row": 1, "solver_sampler_arms": arms,
            "solver_arms_in_order": arms + ([TWIN_ARM, BASELINE_ARM] if include_twin_and_lock else []),
            "baseline_arm": BASELINE_ARM, "twin_arm": TWIN_ARM if include_twin_and_lock else None,
            "note": "The runner consumes the registered cell (task T4 writer); this run-list is its pre-registration table, not a runner input.",
        },
        "decoding": {**decoding, "seed_rule": "derive_seed(iteration_seed, row_id, k)"},
        "base_seed": int(base_seed), "iteration_seed": it_seed, "banks": banks, "row_plan": rows_doc,
        "budget": {"sampler_contacts": sampler_budget, "program_c7": program_budget},
        "model_calls_made_by_this_file": 0,
    }


def execution_columns(plan):
    """Compile the seed-only K4 request plan into executable Solver declarations.

    Recheck every planned seed; do not convert historical top-p ladders or rule
    variants into a seed-only experiment. Cell registration remains separate.
    """
    if (plan['schema'] != RUN_LIST_SCHEMA or plan['variation'] != 'seed-only'
            or plan['k'] != 4 or plan['banks']['seed_rule_bank_sha256'] is not None):
        raise SamplerError('exact seed-only K4 plan required')
    if plan['iteration_seed'] != cm.iteration_seed(plan['base_seed'], plan['iteration']):
        raise SamplerError('iteration seed differs')
    arms = sampler_arms(4)
    if plan['program']['solver_sampler_arms'] != arms:
        raise SamplerError('trajectory arms differ')
    rows = plan['row_plan']
    if not rows or len({r['row_id'] for r in rows}) != len(rows):
        raise SamplerError('unique real row plan required')
    for row in rows:
        contacts = [c for c in row['contacts'] if c['node'] == 'solver']
        if [c['arm'] for c in contacts] != arms:
            raise SamplerError('ordered trajectory contacts differ')
        if len({c['request_identity_sha256'] for c in contacts}) != 1:
            raise SamplerError('seed-only request identities differ')
        for k, contact in enumerate(contacts):
            if contact['seed'] != cm.derive_seed(plan['iteration_seed'], row['row_id'], k):
                raise SamplerError('planned trajectory seed differs')
    decoding = {k: plan['decoding'][k] for k in ('temperature', 'top_p', 'top_k')}
    return [{'column_id': arm, 'kind': 'decoding', 'node': 'solver',
             'decoding': {**decoding, 'trajectory_seed_base': plan['iteration_seed'], 'trajectory_index': k}}
            for k, arm in enumerate(arms)]


# ----------------------------------------------------------------------------------------------------------------------------------------
# reading a flight's outputs: trace lines, scores
# ----------------------------------------------------------------------------------------------------------------------------------------


def read_trace(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """``(contact_lines, score_lines, tfrl_lines)`` of one or more rollout-trace files, every line validated against ``saea-rollout-trace-v1``."""

    contacts: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    tfrls: list[dict[str, Any]] = []
    for path in paths:
        for number, text in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not text.strip():
                continue
            try:
                line = rt.loads_line(text)
            except rt.RolloutTraceSchemaError as error:
                raise SamplerError(f"{path}:{number}: {error}") from error
            {rt.KIND_CONTACT: contacts, rt.KIND_SCORE: scores, rt.KIND_TFRL: tfrls}[line["kind"]].append(line)
    return contacts, scores, tfrls


def scores_from_lines(score_lines: Iterable[Mapping[str, Any]], *, scorer_id: str | None = None) -> dict[tuple[str, str], tuple[float, bool]]:
    """``{(row_id, arm): (official per-row slu_f1, scenario_action_exact)}`` from trace ``score`` lines.

    Several scorers on the same (row, arm) are ambiguous: pass ``scorer_id`` to choose, or the call refuses.
    """

    table: dict[tuple[str, str], tuple[float, bool, str]] = {}
    for line in score_lines:
        if scorer_id is not None and line["scorer_id"] != scorer_id:
            continue
        key = (line["row_id"], line["arm"])
        if key in table and table[key][2] != line["scorer_id"]:
            raise SamplerError(f"{key}: scores from two scorers ({table[key][2]!r}, {line['scorer_id']!r}); pass scorer_id")
        table[key] = (float(line["slu_f1"]), bool(line["scenario_action_exact"]), line["scorer_id"])
    return {key: (value[0], value[1]) for key, value in table.items()}


def scores_from_counts(arm_rows: Mapping[str, Mapping[str, Mapping[str, float]]]) -> dict[tuple[str, str], tuple[float, bool]]:
    """``{(row_id, arm): (row F1, intent match)}`` from the official scorer's per-row count tables (``{arm: {row_id: {field: number}}}``),
    through ``paired_ci.row_f1`` (2TP / (2TP + FP + FN) over the word and char counts; 1.0 for empty gold and empty prediction)."""

    table: dict[tuple[str, str], tuple[float, bool]] = {}
    for arm, rows in arm_rows.items():
        for row_id, row in rows.items():
            table[(row_id, arm)] = (paired_ci.row_f1(row), float(row[paired_ci.INTENT_FIELD]) >= 1.0)
    return table


def solver_contacts(contact_lines: Iterable[Mapping[str, Any]], arms: Sequence[str]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """``{(row_id, arm): contact line}`` of the DELIVERED Solver contact of each sampler arm: the last ``ok`` Solver contact with a parsed answer
    (an answer-only fallback re-ask, when present, comes later in the file and supersedes the truncated first contact)."""

    wanted = set(arms)
    table: dict[tuple[str, str], Mapping[str, Any]] = {}
    for line in contact_lines:
        if line["node"] != "solver" or line["arm"] not in wanted or line["outcome"] != "ok" or line["answer"] is None or not line["answer"]["parsed"]:
            continue
        table[(line["row_id"], line["arm"])] = line
    return table


# ----------------------------------------------------------------------------------------------------------------------------------------
# credit
# ----------------------------------------------------------------------------------------------------------------------------------------


def credit_rows(
    row_ids: Sequence[str], iteration: int, k: int, scores: Mapping[tuple[str, str], tuple[float, bool]], *,
    baseline_arm: str = BASELINE_ARM, twin_arm: str | None = None,
) -> list[dict[str, Any]]:
    """Per-row credit (``saea-tfrl-v2-row-credit-v1``), in ``row_ids`` order.

    reward(k) = row_f1(trajectory k) - row_f1(baseline). best = the intent-correct trajectory with the highest row F1 (ties: lowest k); worst = the
    trajectory with the lowest row F1 other than best (ties: highest k). positive = best exists, reward(best) > 0, and the baseline is stable
    (no twin supplied, or |lock - twin| <= 1e-9). Refuses when any needed score is missing.
    """

    arms = sampler_arms(k)
    out: list[dict[str, Any]] = []
    for row_id in row_ids:
        needed = [(row_id, baseline_arm)] + [(row_id, arm) for arm in arms] + ([(row_id, twin_arm)] if twin_arm else [])
        missing = [key for key in needed if key not in scores]
        if missing:
            raise SamplerError(f"missing score for {missing[0]} (and {len(missing) - 1} more)" if len(missing) > 1 else f"missing score for {missing[0]}")
        base = scores[(row_id, baseline_arm)][0]
        stable = None if twin_arm is None else abs(scores[(row_id, twin_arm)][0] - base) <= BASELINE_TOLERANCE
        trajectories = []
        for index, arm in enumerate(arms):
            f1, intent_ok = scores[(row_id, arm)]
            trajectories.append({"k_index": index, "arm": arm, "trajectory_id": trajectory_id(iteration, row_id, index), "row_f1": f1,
                                 "intent_ok": intent_ok, "reward": f1 - base})
        correct = [t for t in trajectories if t["intent_ok"]]
        best = max(correct, key=lambda t: (t["row_f1"], -t["k_index"])) if correct else None
        others = [t for t in trajectories if best is None or t["k_index"] != best["k_index"]]
        worst = min(others, key=lambda t: (t["row_f1"], -t["k_index"])) if others and best is not None else None
        if best is None:
            reason = "no-intent-correct-trajectory"
        elif stable is False:
            reason = "baseline-unstable"
        elif best["reward"] > GAIN_EPSILON:
            reason = "positive"
        else:
            reason = "no-gain-over-baseline"
        out.append({
            "schema": CREDIT_SCHEMA, "row_id": row_id, "iteration": iteration, "baseline_arm": baseline_arm, "baseline_f1": base,
            "baseline_stable": stable, "trajectories": trajectories, "best_k": None if best is None else best["k_index"],
            "worst_k": None if worst is None else worst["k_index"], "positive": reason == "positive", "reason": reason,
        })
    return out


def credit_summary(credit: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = len(credit)
    positives = sum(1 for c in credit if c["positive"])
    reasons: dict[str, int] = {}
    for c in credit:
        reasons[c["reason"]] = reasons.get(c["reason"], 0) + 1
    return {"rows": rows, "positives": positives, "positive_share": round(positives / rows, 6) if rows else 0.0,
            "reasons": dict(sorted(reasons.items()))}


# ----------------------------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------------------------


def _load_half(path: Path, half: str) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    document = json.loads(raw.decode("utf-8"))
    _half_rows(document, half)
    return document, cm.sha256_bytes(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="build the run-list of one half and one iteration (writes nothing unless --out)")
    plan.add_argument("--half", choices=cm.HALVES, required=True)
    plan.add_argument("--roster", help="the half roster receipt (default: the committed path of the half)")
    plan.add_argument("--iteration", type=int, required=True)
    plan.add_argument("--k", type=int, default=cm.DEFAULT_K)
    plan.add_argument("--base-seed", type=int, default=cm.DEFAULT_BASE_SEED)
    plan.add_argument("--temperature", type=float, default=cm.DEFAULT_TEMPERATURE)
    plan.add_argument("--variation", choices=VARIATIONS, default="seed-only")
    plan.add_argument("--bank-receipt", help="bank receipt of the bank this half consumes (iteration >= 2)")
    plan.add_argument("--seed-rule-bank", help="JSON with a 'variants' list (variation rule-variants)")
    plan.add_argument("--no-twin-lock", action="store_true", help="plan the sampler arms only (program c6-like); default includes aa and lock")
    plan.add_argument("--out", help="write the run-list here (write-once)")
    bud = sub.add_parser("budget", help="the dry-run budget line for one iteration over both halves")
    bud.add_argument("--k", type=int, default=cm.DEFAULT_K)
    bud.add_argument("--median-tokens", type=int, default=cm.MEDIAN_COMPLETION_TOKENS)
    bud.add_argument("--host-tok-s", type=float, default=cm.HOST_OUTPUT_TOK_S)
    bud.add_argument("--roster-dir", default=str(REPO_ROOT / "docs" / "receipts" / "rosters"))
    args = parser.parse_args(argv)

    try:
        if args.command == "plan":
            roster_path = Path(args.roster) if args.roster else REPO_ROOT / cm.HALF_ROSTER_RELPATH[args.half]
            document, file_sha = _load_half(roster_path, args.half)
            receipt = json.loads(Path(args.bank_receipt).read_text(encoding="utf-8")) if args.bank_receipt else None
            seed_bank = json.loads(Path(args.seed_rule_bank).read_text(encoding="utf-8")) if args.seed_rule_bank else None
            doc = build_plan(document, args.half, args.iteration, k=args.k, base_seed=args.base_seed, temperature=args.temperature,
                             variation=args.variation, bank_receipt=receipt, seed_rule_bank=seed_bank,
                             include_twin_and_lock=not args.no_twin_lock, roster_file_sha256=file_sha)
            data = cm.file_bytes(doc)
            if args.out:
                print(f"{args.out}: {cm.write_once(Path(args.out), data)}")
            print(budget_line(f"{doc['run_id']} sampler contacts", doc["budget"]["sampler_contacts"]))
            print(budget_line(f"{doc['run_id']} program c7", doc["budget"]["program_c7"]))
            return 0
        parts = {False: [], True: []}
        for half in cm.HALVES:
            document, _ = _load_half(Path(args.roster_dir) / Path(cm.HALF_ROSTER_RELPATH[half]).name, half)
            audio = [float(i["audio_seconds"]) for i in document["items"]]
            for twin in (False, True):
                parts[twin].append(budget(audio, args.k, include_twin_and_lock=twin, median_completion_tokens=args.median_tokens, host_tok_s=args.host_tok_s))
        for twin, label in ((False, "sampler contacts"), (True, "program c7")):
            for half, part in zip(cm.HALVES, parts[twin]):
                print(budget_line(f"{half} {label}", part))
            print(budget_line(f"BOTH HALVES {label}", sum_budgets(parts[twin])))
        return 0
    except (SamplerError, cm.CommonError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
