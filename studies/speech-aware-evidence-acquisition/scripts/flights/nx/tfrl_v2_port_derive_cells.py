#!/usr/bin/env python3
"""Derive the five model cells of the T family (TFRL-v2 on the NPU port, iteration 1): config only.

T = the agent system A (the lock harness of the L338 port cell) plus the three TFRL-v2 outputs (exemplar bank, rule bank, vote harness). The cells:

* ``construction-r01c-smoke-4`` / ``construction-r01c-i1-h1`` / ``construction-r01c-i1-h2``: the trajectory-sampling flights on the construction column R01C (four
  rows; the interleaved halves of 313 and 312 rows). Derived from the L338 port Fold B baseline (the lock harness, the port's re-authored stores and regenerated
  retrieval vectors) with the VAR-2 R01C pins (the minus-r01 train-exemplars pool, the R01C speech-query vectors and agreement table, port names, PENDING
  sentinels) and the Solver node's ``decoding`` column group s1..s4 + the twin aa + the delivering lock LAST, compiled by the module compiler (program
  ``phase-native-module-v1-c7``: 1 Perception + 6 Solver contacts per row);
* ``fold-b-smoke-4`` / ``fold-b-i1``: the selection flights on TRAIN Fold B. Derived from the L338 port Fold B cell with ONE Perception-level ``sampling`` column
  ``t-v2`` (a Solver vote ``solver_samples`` n = 3 at 0.6 / 0.95 / 20), the twin aa, the registered replicate lock-rep2 and lock LAST (program
  ``phase-native-module-v1-c10``: 4 Perception + 6 Solver), under T's supply bound (A's bound minus 8) with both TFRL-v2 caps folded into the frame's base prompt
  tokens, and a declarative ``tfrl_v2`` block (bank pins at the zero sentinel).

Every pin that does not exist yet (the R01C port vectors, alias, agreement, supply, the banks, the decision receipt) is the PENDING sentinel (sha256 zero). The
derive REFUSES, with the largest fitting bound, when the T Fold B frame does not fit the active thinking-budget switch (6,144 / 7,168). CONFIG ONLY: no
contact, no ledger row, no receipt, no flip. All cells are HELD.

    python -B scripts/flights/nx/tfrl_v2_port_derive_cells.py [--dry-run] [--repo-root DIR]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import l338_family as fam  # noqa: E402
import l338_port_derive_cells as pdc  # noqa: E402
import l338_port_family as pf  # noqa: E402
import tfrl_v2_port_family as tf  # noqa: E402

from speech_aware_evidence_acquisition.rails import consumption_module as cm  # noqa: E402
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks  # noqa: E402

REPO_ROOT = tf.REPO_ROOT
ZERO = "0" * 64
T_STALE = re.compile(r"\bc15\b|\bc17\b|-c15\b|-c17\b|\b32,400\b|\b94,644\b|\bnp4\b")
HIST_CONSTRUCTION = {"flight": "slurp-native-stage3-tfrl-construction-lock-r01c-v1", "smoke": "slurp-native-stage3-tfrl-construction-lock-r01c-smoke-4-v1"}
HIST_CONSTRUCTION_BASELINE = {"flight": "slurp-native-stage3-tfrl-construction-lock-r01c-baseline-v1", "smoke": "slurp-native-stage3-tfrl-construction-lock-r01c-smoke-4-baseline-v1"}
L338_SOURCE = {"fsmoke": "smoke", "fb": "arm", "csmoke": "arm", "ch1": "arm", "ch2": "arm"}

# ---- the PENDING R01C port artifacts the construction cells pin (none exists yet; paths are the proposed names, shas the zero sentinel) -----------------------------
R01C_ENCODE_RUN = "SAEA-SLURP-STAGE3-CONSTRUCTION-R01C-SPEECH-QUERY-ENCODING-NPU-PORT-V1"
R01C_ALIAS_RUN = "SAEA-SLURP-STAGE3-CONSTRUCTION-R01C-ALIAS-NPU-PORT-V1"
R01C_INPUTS_RUN = "SAEA-SLURP-STAGE3-CONSTRUCTION-R01C-INPUTS-PREPARE-NPU-PORT-V1"
R01C_AGREEMENT_RUN = "SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-R01C-NPU-PORT-V1"
R01C_VECTOR_DIR = "vectors/slurp-stage3-construction-r01c-npu-port-v1"
R01C_PENDING = {
    "segments": {"relpath": f"{R01C_VECTOR_DIR}/query-vectors-speech-segments.jsonl", "ledger_row": R01C_ENCODE_RUN},
    "content": {"relpath": f"{R01C_VECTOR_DIR}/query-vectors-speech-content-v1.jsonl", "ledger_row": R01C_ENCODE_RUN},
    "agreement": {"relpath": f"prepared/{R01C_AGREEMENT_RUN}/agreement.jsonl", "ledger_row": R01C_AGREEMENT_RUN},
    "encode_receipt": {"relpath": f"preparation-runs/{R01C_ENCODE_RUN}/encode-v1/receipt.json"},
    "alias_receipt": {"relpath": f"prepared/{R01C_ALIAS_RUN}/query-dimension-alias.json"},
    "content_provenance": {"relpath": f"{R01C_VECTOR_DIR}/query-vectors-speech-content-v1.provenance.json"},
    "agreement_receipt": {"relpath": f"prepared/{R01C_AGREEMENT_RUN}/receipt.json"},
}
MINUS_R01_RECEIPT = "docs/receipts/knowledge/slurp-kb-size-pool-annotated-exemplars-minus-r01.json"
POOL_TEXT = ("SLURP v2 base program module M1 / cell B1 on the construction roster R01C, NPU port: the top-20 TRAIN construction-pool exemplars recalled by cosine against this row's OWN sealed R01C "
             "speech vector in the content retrieval dimension (the port's vectors, PENDING at the zero sentinel). The store is the size-pool-annotated-exemplars-MINUS-R01 release (34,474 source ids, "
             "0 shared with R01C), pinned DIRECTLY as in the committed VAR-2 R01C cell: this cell does not name consumption_module.construction_roster because that variant accepts only the historical "
             "R01C vector prefix and agreement pin. The lock's own pool contains the R01C rows, which the runner's knowledge-source guard refuses "
             "(docs/receipts/knowledge/slurp-kb-size-pool-annotated-exemplars-minus-r01.json).")
FULL_POOL_RECEIPT = "docs/receipts/knowledge/slurp-kb-size-pool-annotated-exemplars.json"
BANK_DIR = "knowledge/slurp/tfrl-v2"
BANKS = {
    "exemplar_bank": {"store_role": "tfrl-exemplar-bank", "dir": f"{BANK_DIR}/exemplar-bank-i{tf.ITERATION}-npu-port-v1"},
    "rule_bank": {"store_role": "tfrl-rule-bank", "dir": f"{BANK_DIR}/rule-bank-i{tf.ITERATION}-npu-port-v1"},
}
SEED_RULE_BANK_DIR = f"{BANK_DIR}/rule-bank-i0-npu-port-v1"


def rails_pending(key: str) -> list[dict]:
    """What the rails still lack for this cell (the declaration below is data the rails do not read yet; each item names the missing piece and its consequence)."""

    common = [
        {"id": "stage3-algorithm-label", "lacks": "no 'tfrl-v2' record in core.stage3_algorithms",
         "consequence": "the cells wear the closest registered labels (shared-rollout for the sampling cells, tfgrpo-semantic-advantage-v1 for the Fold B cells)"},
    ]
    if tf.kind_of(key) == "construction":
        return common + [
            {"id": "construction-variant-port-names", "lacks": "slurp_module_construction.check_cell accepts the r01c variant only under vectors/slurp-stage3-construction-r01c-v1/ "
                                                                  "and the historical R01C agreement pin",
             "consequence": "the cell pins the minus-r01 train-exemplars twins directly (the committed VAR-2 R01C precedent) and does NOT name construction_roster; the adapter "
                            "needs the port prefix and port agreement pin before the variant can be named"},
            {"id": "solver-decoding-seed-override", "lacks": "the Solver decoding group refuses a seed override and two equal overrides",
             "consequence": "the plan's seed-only iteration-1 trajectories cannot be a Solver decoding group; s1..s4 differ in top_p at temperature 0.7 on the lock seed. "
                            "The seed-true alternative is the Perception-level sampling column (c10), which the sampler's arm-keyed trace and credit tooling cannot label"},
            {"id": "rule-variants", "lacks": "no rail applies a rule-bank text to a Solver arm", "consequence": "the rule-variants mode (iteration 2 and later) cannot run"},
        ]
    return common + [
        {"id": "bank-store-roles", "lacks": "no store roles 'tfrl-exemplar-bank' and 'tfrl-rule-bank' and no scenario-conditioned supply from them",
         "consequence": "the bank pins live in this block (zero sentinel), not in knowledge_stores; nothing reads them, so the bank is not consumed until the rail exists"},
        {"id": "vote-commit-policy", "lacks": "'saea-commit-vote-v1' does not exist (entity union-vote rule unfixed)",
         "consequence": "the closest registered policy saea-commit-plurality-support-v1 (min_support 2, similarity 0.8) is used; greedy_first makes n = 3 one greedy plus two drawn"},
        {"id": "top-rule-per-vote", "lacks": "no rail puts one of the top-3 rules into each of the three vote contacts",
         "consequence": "the three votes differ in seed only; the rule text is declared, not delivered"},
        {"id": "exemplar-block", "lacks": "no rail renders the exemplar block (2,000-token cap) into the Solver prompt",
         "consequence": "the cap is folded into the frame's base prompt tokens so the context fit is already honest"},
    ]


# ---- module blocks ---------------------------------------------------------------------------------------------------------------------------------------


def _compile(block: dict, root: Path):
    return cm.compile_module(block, carrier_module_hooks(repo_root=root), repo_root=root)


def construction_block(base_block: dict, group: str = tf.CONSTRUCTION_GROUP) -> dict:
    """The module block of a construction cell: the lock block with the Solver ``decoding`` column group (default) or the Perception-level ``sampling`` column."""

    block = copy.deepcopy(base_block)
    frame = block.pop("sampling_frame")
    block.pop("construction_roster", None)
    if group == "solver-decoding":
        block["columns"] = [{"column_id": arm, "kind": "decoding", "node": "solver", "decoding": dict(d)} for arm, d in zip(tf.SAMPLER_ARMS, tf.CONSTRUCTION_DECODING)]
        block["decoding_frame"] = {"fit_estimate": dict(frame["fit_estimate"]), "max_supplied_items": frame["max_supplied_items"],
                                   "template_tokens": fam.DELIVERING_TEMPLATE_TOKENS, "twin": True}
    elif group == "perception-sampling":
        block["columns"] = [{"column_id": "s-draw", "kind": "sampling", "node": "perception",
                             "solver_samples": {"n": tf.CONSTRUCTION_SAMPLING_N, "decoding": dict(tf.CONSTRUCTION_SAMPLING_DECODING), "commit": dict(tf.VOTE_COMMIT)}}]
        block["sampling_frame"] = {"fit_estimate": dict(frame["fit_estimate"]), "max_supplied_items": frame["max_supplied_items"], "twin": True}
    else:
        raise ValueError(f"unknown construction group {group!r}")
    return block


def fold_b_block(base_block: dict) -> dict:
    """The module block of a Fold B cell: ONE sampling column ``t-v2`` (Solver vote n = 3); T's bound; both caps folded into the base prompt tokens."""

    block = copy.deepcopy(base_block)
    block["columns"] = [{"column_id": tf.T_ARM, "kind": "sampling", "node": "perception",
                         "solver_samples": {"n": tf.VOTE_N, "decoding": dict(tf.VOTE_DECODING), "commit": dict(tf.VOTE_COMMIT)}}]
    frame = block["sampling_frame"]
    frame["max_supplied_items"] = tf.t_bound()
    frame["fit_estimate"] = dict(frame["fit_estimate"], base_prompt_tokens=tf.fold_b_fit_base_tokens())
    frame["twin"] = True
    frame["replicates"] = list(pf.REPLICATE_ARMS)
    return block


def apply_compiled(base: dict, block: dict, key: str, root: Path, group: str | None = None) -> dict:
    """Compile ``block`` and move ONLY the compiler's outputs onto ``base``: ``policy`` and the node roles (never the lock layer's store pins)."""

    group = group or tf.group_of(key)
    composed = _compile(block, root)
    program = tf.program_id(key, group)
    if (composed.program_id, composed.program_ceiling) != (program, tf.contacts_per_row(key, group)):
        raise ValueError(f"{key}: compiled program {composed.program_id} / {composed.program_ceiling}, expected {program} / {tf.contacts_per_row(key, group)}")
    values = composed.values_on_roster_of(base)
    out = copy.deepcopy(base)
    out["policy"] = values["policy"]
    roles = out["native_episode"]["roles"]
    roles["perception"] = copy.deepcopy(values["native_episode"]["roles"]["perception"])
    roles["solver"] = copy.deepcopy(values["native_episode"]["roles"]["solver"])
    out["native_episode"][cm.CONSUMPTION_MODULE_CONFIG_KEY] = block
    if group == "solver-decoding":
        arms = [a["arm_id"] for a in roles["solver"]["decoding_variants"]["arms"]]
        if tuple(arms) != tf.CONSTRUCTION_ARM_ORDER or "sampling_variants" in roles["perception"]:
            raise ValueError(f"{key}: the compiled decoding arms {arms} are not {list(tf.CONSTRUCTION_ARM_ORDER)} (or the Perception node still samples)")
    elif group == "fold-b":
        arms = [a["arm_id"] for a in roles["perception"]["sampling_variants"]["arms"]]
        if tuple(arms) != tf.FOLD_B_ARM_ORDER:
            raise ValueError(f"{key}: the compiled sampling arms {arms} are not {list(tf.FOLD_B_ARM_ORDER)}")
    return out


# ---- baselines -------------------------------------------------------------------------------------------------------------------------------------------


def _pending(spec: dict, **extra) -> dict:
    return {**spec, "sha256": ZERO, **extra}


def _repo_receipt(root: Path, relpath: str) -> dict:
    return {"relpath": relpath, "root": "repository", "sha256": hashlib.sha256((Path(root) / relpath).read_bytes()).hexdigest()}


def _swap_signal(text: str) -> str:
    return text.replace("TRAIN Fold B", "SLURP-STAGE3-CONSTRUCTION-R01C")


def construction_pins(base: dict, key: str, root: Path) -> dict:
    """The VAR-2 R01C pins on the port Fold B baseline: minus-r01 pool, R01C query sources and agreement (PENDING), the cell's own roster receipt."""

    out = copy.deepcopy(base)
    stores = out["knowledge_stores"]
    pool = stores["train-exemplars"]
    pool["release"] = {"relpath": f"knowledge/slurp/saea-kb-distmodel-v1/size-pool-annotated-exemplars-minus-r01/release.json", "sha256": "83620eb0b02f12424a00cd5572904f4df48a49c329b1b0e1a9321d4b9ec14fd0"}
    pool["snapshot"] = {"relpath": "knowledge/slurp/saea-kb-distmodel-v1/size-pool-annotated-exemplars-minus-r01/snapshot.json",
                        "sha256": "6eaa03aea96a1928dd1ac29ecc71c3955c136498c9f8114266eaa830d1207c63"}
    solver_q = pool["phase_supply"]["solver"]["query_source"]
    pool["phase_supply"]["solver"]["query_source"] = _pending(R01C_PENDING["content"], embedded_signal=_swap_signal(solver_q["embedded_signal"]), kind=solver_q["kind"],
                                                              tool_contact_ledger_row=R01C_PENDING["content"]["ledger_row"])
    pool["phase_supply"]["solver"]["query_source"].pop("ledger_row")
    perception = stores["intent-boundaries"]["phase_supply"]["perception"]
    seg_q, agree = perception["query_source"], perception["agreement_source"]
    perception["query_source"] = _pending(R01C_PENDING["segments"], embedded_signal=_swap_signal(seg_q["embedded_signal"]), kind=seg_q["kind"],
                                          tool_contact_ledger_row=R01C_PENDING["segments"]["ledger_row"])
    perception["query_source"].pop("ledger_row")
    new_agree = _pending(R01C_PENDING["agreement"], embedded_signal=_swap_signal(agree["embedded_signal"]), kind=agree["kind"],
                         tool_contact_ledger_row=R01C_PENDING["agreement"]["ledger_row"])
    new_agree.pop("ledger_row")
    perception["agreement_source"] = new_agree
    for component in out["native_episode"]["consumption_module"]["components"]:
        if component.get("component_id") == "agree-vote-seed":
            component["params"]["agreement_source"] = copy.deepcopy(new_agree)
    out["what_the_train_exemplars_store_is"] = POOL_TEXT
    spec = tf.CELLS[key]
    keep = [r for r in out["preparation_source_receipts"]
            if not any(token in r["relpath"] for token in ("FOLD-B", "filler-fold-b", "slurp-evo-fold-b-v2")) and r["relpath"] != FULL_POOL_RECEIPT]
    keep += [{"relpath": R01C_PENDING[name]["relpath"], "root": "data", "sha256": ZERO} for name in ("encode_receipt", "alias_receipt", "agreement_receipt", "content_provenance")]
    keep += [_repo_receipt(root, MINUS_R01_RECEIPT), _repo_receipt(root, spec["roster"])]
    if key in ("ch1", "ch2"):
        keep += [_repo_receipt(root, tf.R01C_ROSTER), _repo_receipt(root, tf.HALVES_SPLIT_RECEIPT)]
    out["preparation_source_receipts"] = sorted(keep, key=lambda item: (item["root"], item["relpath"]))
    return out


def tfrl_v2_block(key: str) -> dict:
    spec = tf.CELLS[key]
    view = pf.budget_view()
    block = {
        "schema": "saea-tfrl-v2-declaration-v1",
        "status": "DECLARATION: no rail reads this block yet (see rails_pending); it is part of the composed config, so the protocol hash covers it",
        "loop_id": tf.LOOP_ID, "iteration": tf.ITERATION, "role": spec["role"], "plan": tf.PLAN,
        "caps": {"exemplar_block_tokens": tf.EXEMPLAR_BLOCK_TOKEN_CAP, "rule_bank_tokens": tf.RULE_BANK_TOKEN_CAP},
        "rails_pending": rails_pending(key),
    }
    if tf.kind_of(key) == "construction":
        half = spec["half"]
        block.update({
            "kind": "trajectory-sampling", "mode": "seed-only", "k_trajectories": tf.K_TRAJECTORIES, "trajectory_arms": list(tf.SAMPLER_ARMS),
            "trajectory_decoding": {arm: dict(d) for arm, d in zip(tf.SAMPLER_ARMS, tf.CONSTRUCTION_DECODING)},
            "trajectory_note": ("plan section 11 ruling 2 wants trajectories that differ by decoding seed only; the Solver decoding group refuses a seed override, so s1..s4 "
                                "share the lock seed and differ in top_p at temperature 0.7 (rails_pending solver-decoding-seed-override)"),
            "half": half, "other_half": ("R01C-H2" if half == "R01C-H1" else "R01C-H1") if half else None, "trace_read_id": spec["trace_read_id"],
            "rule_bank_i0": {"consumed_by_this_cell": False, "release_dir": SEED_RULE_BANK_DIR,
                             "note": "released for the rule-variants mode of iteration 2 and later; the iteration-1 flight is seed-only and consumes no bank"},
            "banks_consumed": None,
            "credit": "per-row reward = row F1 of the trajectory minus row F1 of lock, counted only where lock and aa agree (model-free, official per-row scores)",
        })
    else:
        block.update({
            "kind": "selection-flight", "a_supply_bound_items": tf.a_bound(), "supply_bound_items": tf.t_bound(),
            "supply_bound_rule": f"A's bound minus {tf.EXEMPLAR_ITEM_EQUIVALENT} (the exemplar block of at most {tf.EXEMPLAR_BLOCK_TOKEN_CAP} tokens is about 8 items' worth at 235 tokens each)",
            "exemplar_policy": "replaces supplied items", "exemplar_policy_note": "plan section 9 ruling 4: a declared T-vs-A protocol difference, not a confound",
            "context_fit": {"solver_prompt_limit_tokens": view["prompt_limit_tokens"], "frame_base_prompt_tokens": tf.fold_b_fit_base_tokens(),
                            "estimated_prompt_tokens_at_bound": tf.fold_b_fit_estimate_tokens(), "largest_fitting_bound": tf.largest_fitting_bound(),
                            "base_is": f"{fam.FIT_ESTIMATE['base_prompt_tokens']} + {fam.DELIVERING_TEMPLATE_TOKENS} template + {tf.EXEMPLAR_BLOCK_TOKEN_CAP} exemplar cap + {tf.RULE_BANK_TOKEN_CAP} rule cap"},
            "vote_harness": {"arm": tf.T_ARM, "n": tf.VOTE_N, "decoding": dict(tf.VOTE_DECODING), "commit": dict(tf.VOTE_COMMIT), "planned_commit_policy": tf.PLANNED_COMMIT_POLICY,
                             "greedy_first": True, "intent_rule": "scenario_action by plurality",
                             "entity_rule": "entities kept at support >= 2 of 3 (the planned union-vote rule is not fixed)",
                             "per_vote_consumption": "the exemplar-bank items of the Perception-predicted scenario plus one of the top-3 ranked rules (declared, not yet delivered)"},
            "banks": {name: {"store_role": spec_["store_role"],
                             "release": {"relpath": f"{spec_['dir']}/release.json", "sha256": ZERO},
                             "snapshot": {"relpath": f"{spec_['dir']}/snapshot.json", "sha256": ZERO},
                             "pin_state": "PENDING (zero sentinel): released after the iteration-1 credit and reflector steps"} for name, spec_ in BANKS.items()},
            "cross_fit": ("R01C is split into interleaved halves H1 (even positions) and H2 (odd positions). From iteration 2 the bank a half consumes is distilled from the OTHER "
                          "half's positives; the iteration-1 construction flights are seed-only and consume no bank"),
            "union": "the Fold B flight consumes the union of both halves' banks; their sources (R01C) are disjoint from Fold A and Fold B",
        })
    return block


def t_baseline(key: str, root: Path = REPO_ROOT) -> dict:
    spec = tf.CELLS[key]
    base = pdc.port_baseline(L338_SOURCE[key], root)
    if tf.kind_of(key) == "construction":
        base = construction_pins(base, key, root)                 # the module block is read AFTER the pins: its agree-vote-seed component carries the R01C agreement pin
        base = apply_compiled(base, construction_block(base["native_episode"]["consumption_module"]), key, root)
    else:
        tf.require_fit()
        base = apply_compiled(base, fold_b_block(base["native_episode"]["consumption_module"]), key, root)
    base["native_contact_fit"] = dict(base["native_contact_fit"], run_id=spec["fit_run"], attempt_id=spec["fit_attempt"], contact_cap=tf.fit_calls(key))
    base.pop("prepared_phase_supply", None)                  # the cell's own supply is prepared under its own run id at the flip
    base["tfrl_v2"] = tfrl_v2_block(key)
    base["what_this_is"] = what_this_is(key)
    return base


# ---- texts -----------------------------------------------------------------------------------------------------------------------------------------------


def arms_text(key: str) -> str:
    if tf.kind_of(key) == "construction":
        return (f"the Solver decoding group {', '.join(tf.SAMPLER_ARMS)} (temperature 0.7, top_p 0.80 / 0.90 / 0.95 / 1.00, top_k 20, the lock seed), the twin {tf.TWIN} and the delivering "
                f"{tf.DELIVERING} arm LAST")
    return (f"the candidate {tf.T_ARM} (one Perception contact and a Solver vote of n = {tf.VOTE_N} at temperature 0.6, top_p 0.95, top_k 20, commit {tf.VOTE_COMMIT['policy']} with "
            f"min_support {tf.VOTE_COMMIT['min_support']}), the twin {tf.TWIN}, the registered replicate {tf.REPLICATE} and the delivering {tf.DELIVERING} arm LAST")


def what_this_is(key: str) -> str:
    spec = tf.CELLS[key]
    p, s = tf.contacts(key)
    where = ("the construction column R01C" + (f" half {spec['half']}" if spec["half"] else "")) if tf.kind_of(key) == "construction" else "TRAIN Fold B"
    return (f"T cell of the three-way comparison (plan {tf.PLAN}), iteration {tf.ITERATION}: {spec['role']}. The L338 port baseline (the lock harness A, the port's re-authored "
            f"stores and regenerated retrieval vectors, raw-audio arm) on {where} with {arms_text(key)}. Program {tf.program_id(key)}: {p} Perception + {s} Solver contacts per row. "
            f"The `tfrl_v2` block declares the TFRL-v2 inputs the rails do not read yet (bank pins at the zero sentinel, caps {tf.EXEMPLAR_BLOCK_TOKEN_CAP} / {tf.RULE_BANK_TOKEN_CAP}, "
            "vote harness, rails_pending). Development numbers only; claims_benefit false.")


def program_sentence(key: str) -> str:
    spec = tf.CELLS[key]
    rows, p, s = spec["rows"], *tf.contacts(key)
    c = p + s
    shape = (f"the Solver decoding group of {len(tf.SAMPLER_ARMS)} trajectories, the twin and lock (1 Perception + {s} Solver contacts per row)" if tf.kind_of(key) == "construction"
             else f"the sampling column {tf.T_ARM} with a Solver vote of n = {tf.VOTE_N} beside the twin, the replicate and lock (2 + 2 (K - 1) + (n - 1) = {c}: {p} Perception + {s} Solver)")
    return (f"The review-off module graph (perception 1, solver 1; Reviewer and Planner never visited) under program {tf.program_id(key)} with {shape}: {rows:,} x {c} = {rows * c:,} is the "
            f"exact program maximum; the plan ceiling adds {c} answer-only fallback contacts per row (one per budgeted-thinking contact): {tf.generation_calls(key):,} = {rows:,} x {2 * c}. "
            f"The live phase fit makes {rows} x {c} x {pf.RAW_FIT_ENDPOINTS_PER_CONTACT} = {tf.fit_calls(key):,} tokenizer endpoint calls (raw-audio arm). The accelerator bound is "
            f"{tf.card_hour_formula(key)}; it is PROVISIONAL, a BOUND never a target, and planned_gpu_hours ({tf.plan_gpu_hours_placeholder(key)}) is the numeric placeholder the plan "
            "field requires until the sweep re-derives it. Fallback contacts and the FIT identity follow the run ids of this cell.")


def purpose_text(key: str) -> str:
    spec = tf.CELLS[key]
    return (f"{spec['run']}: {spec['role']}, NPU port (package {tf.PACKAGE}); BF16 / vLLM-Ascend Thinking core ({pf.MODEL_FRAGMENT}; per-sequence window "
            f"{pf.PER_SEQUENCE_WINDOW_TOKENS:,} tokens; deployment {pf.DEPLOYMENT_REF}) over the original recordings ({pf.PROMPT_ORDER}) with the port's re-authored knowledge and "
            f"regenerated retrieval vectors. Arms: {arms_text(key)}. {' '.join(tf.T_DECLARED_DIFFERENCES[:1])} {pf.DIFFERENCES_TEXT}. Read by the acceptance and kill rules of "
            f"{tf.PACKAGE}/README.md, never as a byte replay of a historical flight.")


def algorithm_label(key: str) -> dict:
    spec = tf.CELLS[key]
    if tf.kind_of(key) == "construction" or spec["smoke"]:
        return {"id": "shared-rollout", "operators_used": ["rollout"] if spec["smoke"] else ["rollout", "scoring"],
                "note": ("TFRL-v2 trajectory sampling: K = 4 Solver trajectories per row on the construction column, scored once by the official scorer; the per-row credit, the "
                         "exemplar distillation and the rule bank are model-free steps outside this cell (tfrl-v2 has no registry record).") if tf.kind_of(key) == "construction" else
                        "Four-row form smoke of the TFRL-v2 selection flight: the request form of the vote arm; scored by no one."}
    return {"id": "tfgrpo-semantic-advantage-v1", "operators_used": ["rollout", "scoring", "context_application"],
            "note": ("TFRL-v2 (a row-level group-relative training-free RL loop) has no registry record; this is the closest registered family. The flight applies the iteration's "
                     "exemplar and rule banks as context and votes at inference; credit and update happened model-free before the flight.")}


# ---- experiments -----------------------------------------------------------------------------------------------------------------------------------------


def _load(kind: str, name: str, root: Path) -> dict:
    return json.loads((Path(root) / "configs" / kind / f"{name}.json").read_text(encoding="utf-8"))


def _roster_ids(key: str) -> list[str]:
    return [item["sample_id"] for item in tf._roster_doc(key)["items"]]


def _requirements(key: str, source_requirements: list[dict]) -> list[dict]:
    spec = tf.CELLS[key]
    decision = {"expected_decision_id": tf.DECISION_ID, "kind": "decision", "reason": f"Decision {tf.DECISION_ID} pinned by scripts/flights/repin_cell.py.",
                "relpath": tf.DECISION_RELPATH, "sha256": ZERO}
    supply = {"kind": "preparation", "relpath": spec["supply_relpath"], "sha256": ZERO,
              "reason": ("This cell's OWN first-phase supply, built model-free under its own run id with core.flight.cli prepare-supply at the flip (item E16); PENDING until the "
                         "main thread prepares it with scripts/flights/repin_cell.py (--allow-output-change must never be passed).")}
    if tf.kind_of(key) == "construction":
        agreement = {"kind": "preparation", "relpath": R01C_PENDING["agreement"]["relpath"], "sha256": ZERO,
                     "reason": ("The R01C exemplar-intent AGREEMENT table on the port (rows-only item, model-free arithmetic against the minus-r01 snapshot, the port pool key vectors "
                                "and the port R01C speech query vectors); PENDING until that row is run.")}
    else:
        agreement = copy.deepcopy(source_requirements[1])
    return [decision, agreement, supply]


def t_experiment(key: str, root: Path = REPO_ROOT) -> dict:
    spec = tf.CELLS[key]
    if tf.kind_of(key) == "fold-b":
        doc = pdc.port_experiment(L338_SOURCE[key], root)
        source_requirements = doc["cells"][0]["readiness"]["requirements"]
    else:
        doc = copy.deepcopy(_load("experiment", HIST_CONSTRUCTION["smoke" if spec["smoke"] else "flight"], root))
        source_requirements = doc["cells"][0]["readiness"]["requirements"]
        doc["service_session"] = tf.SERVICE_SESSION
        doc["population"] = {"count": spec["rows"], "dataset_fragment": spec["dataset"], "roster_receipt": spec["roster"], "split_identity_hash": spec["split_identity_hash"],
                             "decision_population": (f"{spec['rows']} rows of {spec['roster']} (roster {spec['roster_id']}), a subset of the clean construction column "
                                                     "SLURP-STAGE3-CONSTRUCTION-R01C (TRAIN rows outside Fold A and Fold B); discovery role, never a test claim; the rows are "
                                                     "the construction source of the TFRL-v2 banks, which never read Fold A or Fold B")}
        doc["populations"] = [{"dataset_fragment": spec["dataset"], "ids": _roster_ids(key)}]
    doc["experiment_id"] = spec["run"]
    doc["date"] = tf.DATE
    doc["service_session"] = tf.SERVICE_SESSION
    doc["fragment_kind"] = f"saea-slurp-native-stage3-tfrl-v2-{spec['stem']}-npu-port-v1"
    cell = doc["cells"][0]
    cell.update(cell_id=spec["cell"], run_id=spec["run"], attempt_id=spec["attempt"], baseline_fragment=spec["baseline"], model_fragment=pf.MODEL_FRAGMENT,
                dataset_fragment=spec["dataset"])
    cell["readiness"] = {"reason": (f"{spec['role'][0].upper()}{spec['role'][1:]}, T cell of the three-way comparison, NPU port. Registered before contact on {tf.DATE} by script "
                                    f"(scripts/flights/nx/tfrl_v2_port_derive_cells.py) from the L338 port cell ({tf.OWNER_DECISIONS}). HELD: root countersigns the DRAFT rows, the main "
                                    f"thread mints the admission receipt {tf.DECISION_RELPATH}, the deployment profile is filled from the measured speed sweep, every PENDING pin "
                                    "(zero sentinel) is produced by its row and re-pinned, the cell's own supply is prepared at the flip and the cell is flipped. Nothing here "
                                    "authorises a flight, a readiness flip or an admission entry."),
                         "requirements": _requirements(key, source_requirements), "state": "held"}
    calls, fit_cap, audio = tf.generation_calls(key), tf.fit_calls(key), tf.audio_seconds(key)
    gpu = tf.plan_gpu_hours_placeholder(key)
    doc["execution_plan"] = dict(doc["execution_plan"], planned_gpu_hours=gpu, planned_model_calls=calls, planned_speech_audio_seconds=audio)
    doc["family_budget"] = dict(doc["family_budget"], generation_calls_cap=calls, native_task_contract_calls_cap=calls, fit_endpoint_calls_cap=fit_cap,
                                repeated_speech_audio_seconds_cap=audio, gpu_hours_fit_inclusive=gpu)
    doc["generation_budget"] = {"cap": calls, "target": calls, "definition": program_sentence(key)}
    doc["purpose"] = purpose_text(key)
    doc["stage3_algorithm"] = algorithm_label(key)
    if tf.kind_of(key) == "fold-b":
        arms = tf.FOLD_B_ARM_ORDER
        doc["core_read_design"] = {
            "arms": {arm: spec["run"] for arm in arms}, "this_cell": spec["role"],
            "chain": "deployment per PORT_DEPLOYMENT_PROFILE -> construction flights, credit, reflector, bank release -> smoke -> machine sign-off -> the selection flight",
            "reading": (f"official SLU-F1 and intent paired per row: {tf.T_ARM}, the twin {tf.TWIN} and the replicate {tf.REPLICATE} minus the delivering {tf.DELIVERING} arm; "
                        "accepted only by the rule in the package README (within-run bar, widened CI95, intent guard, per-row delta, twin admissibility, construction gain)")}
    else:
        doc.pop("core_read_design", None)
    return doc


def t_campaigns() -> dict:
    construction = {"cells": [{"cell": tf.CELLS[k]["cell"], "experiment": tf.CELLS[k]["cell"]} for k in ("ch1", "ch2")], "schema": "saea-campaign-plan-v1"}
    fold_b = {"cells": [{"cell": tf.CELLS["fb"]["cell"], "experiment": tf.CELLS["fb"]["cell"]}], "schema": "saea-campaign-plan-v1"}
    return {f"slurp-native-stage3-tfrl-v2-construction-r01c-i{tf.ITERATION}-npu-port-v1": construction, f"slurp-native-stage3-tfrl-v2-fold-b-i{tf.ITERATION}-npu-port-v1": fold_b}


# ---- the derive ------------------------------------------------------------------------------------------------------------------------------------------


def _dump(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def verify_against_disk(key: str, base: dict, exp: dict, root: Path) -> dict:
    """The leaves a re-derive would move on the files already on disk (empty for an absent file and for a fixed point, so the first and every later run write the same bytes)."""

    spec = tf.CELLS[key]
    out = {}
    for kind, derived, name in (("baseline", base, spec["baseline"]), ("experiment", exp, spec["cell"])):
        path = Path(root) / "configs" / kind / f"{name}.json"
        out[kind] = {"moved": pdc.moved_leaves(json.loads(path.read_text(encoding="utf-8")), derived) if path.is_file() else []}
    return out


def derive(repo_root: Path = REPO_ROOT, dry_run: bool = False) -> dict:
    root = Path(repo_root)
    tf.select_model_fragment()
    tf.require_fit()
    result = {}
    for key in tf.ALL_KEYS:
        spec = tf.CELLS[key]
        base, exp = t_baseline(key, root), t_experiment(key, root)
        leaks = [f"{kind}:{p}" for kind, doc in (("baseline", base), ("experiment", exp)) for p in pdc.scan_geometry(doc)]
        stale = [f"{kind}:{p}" for kind, doc in (("baseline", base), ("experiment", exp)) for p, v in pdc._leaves(doc) if isinstance(v, str) and T_STALE.search(v)]
        if leaks or stale:
            raise ValueError(f"T {key} texts still state a slot geometry or a fact of another arm table: {leaks + stale}")
        check = verify_against_disk(key, base, exp, root)
        if not dry_run:
            (root / "configs" / "baseline" / f"{spec['baseline']}.json").write_text(_dump(base), encoding="utf-8")
            (root / "configs" / "experiment" / f"{spec['cell']}.json").write_text(_dump(exp), encoding="utf-8")
        result[key] = {"cell": spec["cell"], "baseline": spec["baseline"], "run_id": spec["run"], "fit_run_id": spec["fit_run"], "program": tf.program_id(key),
                       "versus_disk": check}
    for name, doc in t_campaigns().items():
        if not dry_run:
            (root / "configs" / "campaign" / f"{name}.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result.setdefault("campaigns", []).append(name)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--dry-run", action="store_true", help="derive and verify, write nothing")
    parser.add_argument("--model-fragment", choices=pf.MODEL_FRAGMENT_CHOICES, default="default")
    args = parser.parse_args(argv)
    pf.select_model_fragment(args.model_fragment)
    print(json.dumps(derive(Path(args.repo_root), args.dry_run), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
