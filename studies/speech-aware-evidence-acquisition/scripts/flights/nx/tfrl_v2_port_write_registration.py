#!/usr/bin/env python3
"""Write the DRAFT readiness package of the T cell of the three-way comparison (``SAEA-SLURP-TFRL-V2-FOLD-B-NPU-PORT-V1``, iteration 1) from the T fragments.

Config and receipt-pinned bytes only: no adapter contact, no server, no model, tokenizer, encoder, embedder or scorer contact. Flips nothing, mints nothing, countersigns nothing,
and writes NO row into ``docs/exposure-ledger.md`` unless ``append_ledger`` is passed (do not pass it before the owner countersigns). Every file says DRAFT / HELD; every row is
unconsumed.

ONE package ``docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/`` (README, ``registered-facts-v1.json`` with the canonical ``stage3_measurement`` blocks keyed by the T run
ids, ``protocol-registered-v1.json``, ten plan files (generation and FIT of the five model cells), the two smoke sign-off condition files, ``root-admission-draft-v1.json``,
``countersign-row-texts-v1.json`` and ``draft-ledger-rows.md``) and, beside it, the preregistration. The ledger text has two blocks: six preparation rows (R01C encoding, alias, content
derivative, agreement table, interleaved halves, seed rule bank) and the loop block in chain order (construction smoke, the two construction halves, credit and distillation, the
reflector, the bank release, the Fold B smoke and the Fold B flight). The ROWS-ONLY items have no cell, no plan and no registration document here.

The port settings (model fragment, deployment placeholder, declared differences, owner decisions, PROVISIONAL card-hour form, the thinking-budget switch) come from ``l338_port_family``
through ``tfrl_v2_port_family``. The Fold B preparations are the L338 port's (``PREPARATIONS``, re-read from their receipts); the R01C port artifacts, the banks, the supplies and the
decision receipt do not exist yet and are pinned by the zero sentinel. Reached through ``l338_port_write_registration.py --family tfrl-v2``.

    python -B scripts/flights/nx/l338_port_write_registration.py --family tfrl-v2 [--model-fragment budget6144-max7168] --study-commit <sha> --data-root <data root>
        [--registered-at <UTC>] [--root DIR] [--append-ledger] [--no-derive]
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import l338_family as fam  # noqa: E402
import l338_port_family as pf  # noqa: E402
import l338_port_write_registration as lw  # noqa: E402
import tfrl_v2_port_derive_cells as tdc  # noqa: E402
import tfrl_v2_port_family as tf  # noqa: E402
import tfrl_v2_port_rails as rails  # noqa: E402

from speech_aware_evidence_acquisition.core.config import compose, protocol_hash  # noqa: E402

reg = lw.reg
REPO_ROOT = tf.REPO_ROOT
PKG = tf.PACKAGE
STATUS = reg.STATUS
ZERO = tdc.ZERO
MF_MODEL, MF_TAIL = lw.MF_MODEL, lw.MF_TAIL
FOLD_B_CARRIER, FOLD_B_SPLIT = lw.FOLD_B_CARRIER, lw.FOLD_B_SPLIT
INHERITED_FOLD_B = lw.INHERITED
INHERITED_R01C = ("The construction column R01C (625 TRAIN rows outside Fold A and Fold B) was exposed by the historical R01 / R01C construction rows; this cell opens no split. "
                  "Prior exposure is inherited in full.")
PRIMARY_METRIC = lw.hw.PRIMARY_METRIC
PROFILE = "bounded-discovery-probe"
REFLECTOR_CONTACTS = 18
REFLECTOR_RUN = f"SAEA-SLURP-TFRL-V2-REFLECTOR-I{tf.ITERATION}-NPU-PORT-V1"
REFLECTOR_FIT_RUN = f"SAEA-SLURP-TFRL-V2-REFLECTOR-I{tf.ITERATION}-FIT-NPU-PORT-V1"
REFLECTOR_ATTEMPT = f"attempt-slurp-tfrl-v2-reflector-i{tf.ITERATION}-generation-npu-port-v1"
REFLECTOR_FIT_ATTEMPT = f"attempt-slurp-tfrl-v2-reflector-i{tf.ITERATION}-fit-npu-port-v1"
CREDIT_RUN = f"SAEA-SLURP-TFRL-V2-CREDIT-I{tf.ITERATION}-NPU-PORT-V1"
BANK_RELEASE_RUN = f"SAEA-SLURP-TFRL-V2-BANK-RELEASE-I{tf.ITERATION}-NPU-PORT-V1"
RULE_BANK_I0_RUN = "SAEA-SLURP-TFRL-V2-RULE-BANK-I0-RELEASE-NPU-PORT-V1"
HALVES_RUN = "SAEA-SLURP-STAGE3-CONSTRUCTION-R01C-HALVES-NPU-PORT-V1"
CONSTRUCTION_SCOPE_SPLIT = "slurp SLURP-STAGE3-CONSTRUCTION-R01C TRAIN construction"
EXPECTED_EFFECT_PP = 2.0                                   # the low end of the plan's +2 to +4 pp expectation (a registration number, never a claim)
BOOTSTRAP = {"seed": rails.SEED, "draws": rails.DRAWS, "level": "CI95", "note": f"the L338 port reader's values (slurp_l338_column_read.SEED / DRAWS), fixed by the T analysis recipe {rails.RECIPE_RELPATH}"}

DIFF_TEXT = "declared differences (design doc section 8 and T): " + "; ".join(tf.declared_differences())
BAR_SLU_PP, BAR_INTENT_PP, TWIN_BAR_PP, CONSTRUCTION_GAIN_PP = fam.FLOOR_SLU_PP, fam.FLOOR_INTENT_PP, 1.2, 1.0

# ---- acceptance and kill rules (plan section 4 with M1 / M2; draft section 7), owner items and open questions ------------------------------------------------

ACCEPTANCE = {
    "iteration_accepted_only_if": [
        f"SLU-F1(t-v2) - SLU-F1(lock) has point >= {BAR_SLU_PP} pp and a CI95 lower bound > 0 after the CI is WIDENED by the replicate variance of lock / aa / lock-rep2 (noise model M1)",
        f"the intent guard holds: intent CI95 lower bound >= -{BAR_INTENT_PP} pp and intent point >= 0",
        "the per-row mean delta (mean of per-row F1) is >= 0; an arm whose per-row delta is negative cannot be nominated",
        f"twin admissibility: |SLU-F1(aa) - SLU-F1(lock)| <= {TWIN_BAR_PP} pp",
        f"the construction-column gain is >= +{CONSTRUCTION_GAIN_PP} pp: SLU-F1 of the vote over s1..s4 (scenario_action by majority, entity support >= 2 of 4) minus SLU-F1 of lock on the "
        "construction rows of the iteration, computed model-free from the sealed flight scores, never from the loop adapter (plan section 11 addendum ruling 6)",
    ],
    "confirmed_only_if": ("in the Fold A flight of the accepted state the widened CI95 lower bound of t-v2 - lock is > 0 (M2); the optimism term (Fold B delta - Fold A delta) is reported always"),
    "kill_and_stop": [
        "iteration 1 construction gain < +1 pp: stop the family",
        "an accepted iteration fails Fold A: revert to the previous accepted state and report the optimism term",
        "an iteration that accepts nothing: stop",
        "at most 3 iterations",
    ],
    "expected": "+2 to +4 pp SLU over A if consumption (P5) is the bottleneck, else 0 (plan section 4); claims_benefit is false either way",
}
OWNER_ITEMS = [
    {"id": "O-switch-bound", "item": ("Under the thinking-budget switch (6,144 / 7,168) the Solver prompt limit is 14,334 and T's bound of 43 items plus both caps estimates 14,828 tokens: it does NOT fit. "
                                       "The largest fitting T bound is 40 items. The owner / planner chooses the bound (40 items, a smaller cap, or a larger window) before the switch is used for T; "
                                       "the writer refuses until then. The default fragment fits (47 items, 15,768 of 17,406).")},
    {"id": "O-construction-group", "item": ("The construction program is the Solver decoding column group (c7): the plan's seed-only iteration-1 trajectories need a per-arm seed that the group refuses, "
                                             "so s1..s4 differ in top_p at temperature 0.7. The seed-true alternative is the Perception-level sampling column (c10), which the arm-keyed trace and credit "
                                             "tooling cannot label. Confirm the c7 form or fund the labelling change.")},
    {"id": "O-commit-policy", "item": ("saea-commit-vote-v1 does not exist; the vote uses saea-commit-plurality-support-v1 (min_support 2, similarity 0.8). Fix the entity union-vote rule or accept "
                                        "the plurality-support rule before the first Fold B contact.")},
    {"id": "O-fold-a-timing", "item": ("Plan section 10: confirming every accepted iteration on Fold A makes Fold A a second selection fold; the draft recommends one confirmation of the final accepted "
                                        "state. M2 as written confirms each accepted iteration. Decide before iteration 2.")},
    {"id": "O-deployment", "item": "PORT_DEPLOYMENT_PROFILE (instances, tensor-parallel size, sequences per instance) is filled from the measured speed sweep; the card-hour bounds are PROVISIONAL until then."},
]
OPEN_QUESTIONS = [
    "Does the runner's knowledge-source guard cover every module store or only the single knowledge snapshot, and can a store role be swapped per arm (the cross-fit from iteration 2 needs both)?",
    "Every T cell keeps preserve_wire_responses true, as the L338 port cell and the historical R01C column do. Does the M4 trace need the preserved wire responses, or do the sealed per-contact records suffice (the setting only costs disk)?",
    "Which rail renders the exemplar block and the per-vote rule text (rails_pending), and when is the first-phase supply of the Fold B cells re-prepared with it?",
    "The R01C encoder, the alias, the content derivative and the agreement chain have no registration documents or chain scripts yet (rows-only here).",
    "Iterations 2 and 3 (cells, plans, rows) are registered only after the previous read.",
]
FINDINGS = [
    {"id": "construction-group-is-the-solver-decoding-column",
     "finding": ("The construction program is the Solver node's decoding column group s1..s4 + the twin aa + the delivering lock LAST (phase-native-module-v1-c7, 1 Perception + 6 Solver "
                 "contacts per row), compiled by the real module compiler. It is chosen over the Perception-level sampling column because the sampler, the trace writer and the credit "
                 "tooling key scores and delivered contacts by arm label s1..s4, which a sampling column (arms of draws) does not give. The alternative (n = 5, c10) is implemented as an option "
                 "(tfrl_v2_port_derive_cells.construction_block) and tested, not registered."),
     "remedy": "none; the owner confirms or redirects (O-construction-group)."},
    {"id": "seed-only-is-not-expressible-in-c7",
     "finding": ("The plan's iteration-1 trajectories differ by decoding seed only; the Solver decoding group refuses a seed override and two equal overrides, so s1..s4 share the lock seed and "
                 "differ in top_p (0.80 / 0.90 / 0.95 / 1.00) at temperature 0.7, top_k 20."),
     "remedy": "a per-arm seed offset in the Solver decoding group would restore seed-only (rails_pending)."},
    {"id": "vote-commit-policy-is-the-closest-existing",
     "finding": ("saea-commit-vote-v1 does not exist. The Fold B vote uses saea-commit-plurality-support-v1 (plurality over scenario_action, entity clusters kept at support >= 2). The sampling axis "
                 "requires greedy_first, so n = 3 is one greedy contact plus two drawn at lock seed + offset."),
     "remedy": "fix the entity rule and register the policy (O-commit-policy)."},
    {"id": "construction-variant-not-named",
     "finding": ("slurp_module_construction.check_cell accepts the r01c variant only for vectors under vectors/slurp-stage3-construction-r01c-v1/ and the historical R01C agreement pin, which the "
                 "port names do not satisfy. The construction cells therefore pin the minus-r01 train-exemplars twins directly (the committed VAR-2 R01C precedent) and do not name construction_roster."),
     "remedy": "generalise the adapter to the port prefix and agreement pin before the variant is named."},
    {"id": "budget-switch-refuses-the-t-bound",
     "finding": ("T's supply bound is A's bound minus 8 and both caps (2,000 + 400 tokens) are folded into the frame's base prompt tokens. Default: 47 items, 15,768 estimated tokens against a "
                 "limit of 17,406 (fits). Budget switch 6,144 / 7,168: 43 items, 14,828 against 14,334 (does not fit); the largest fitting bound is 40. The writer refuses under the switch."),
     "remedy": "choose the bound (O-switch-bound)."},
    {"id": "t-cells-lock-switched-to-the-port-lock",
     "finding": ("The config loader (rails.native_episode_config) checks EVERY declared consumption_module block with verify_declared_module. Lock v2.2 carries the historical 268-B store pins, "
                 "so a cell that pins the port's re-authored stores differs from what its block compiles to at ten leaves (docs/results/2026-10-30-badcase-verify-declared-module-l338-port.md). "
                 "T never calls apply_to (it moves only the compiler's policy and role leaves, as the L338 port derive does), so the derive cannot hide such a clash from the loader."),
     "remedy": ("Resolved in the T lock switch (plan 3.14 addendum, plan section 13 ruling 3 (c)): the two Fold B cells declare lock v2.2-npu-port (v2.2 + the ten port store-pin leaves); the three "
                "construction cells pin the minus-R01 train-exemplars pool and declare lock v2.2-npu-port-r01c (v2.2-npu-port + exactly four train-exemplars release / snapshot leaves, read from "
                "the minus-R01 build receipt). The derive runs module_self_check before it writes and identities() runs verify_composed_module on every composed T cell, so a lock clash stops "
                "the writer; the loader accepts all five cells.")},
    {"id": "fit-cap-is-the-raw-arm-rule",
     "finding": ("The FIT caps are rows x contacts x 1 counting endpoint (native_contact_required_cap, raw-audio arm), checked at import against the gate's own helper; the packet-arm factor of 4 "
                 "is not used."),
     "remedy": "none."},
    {"id": "credit-counts-only-where-lock-and-aa-agree",
     "finding": ("The twin aa is in the construction group so that per-row credit counts a positive or negative only where lock and aa agree (8.5 % of rows differ within one flight)."),
     "remedy": "none."},
]
PROHIBITIONS = [
    "No test-split, DEV or Fold A contact in any cell here; Fold A is the confirmation cell of a separate package and is never a bank source.",
    "No Fold A or Fold B row is ever a source of the exemplar or rule bank (the construction scope is R01C; forbidden rosters Fold A v2, Fold B v2, DEV-1000).",
    "No Reviewer or Planner visit; no tool schema on any role; max_revisions 0; no reasoning text is delivered, scored or carried to another node.",
    "The delivering lock arm is the L338 port lock, byte for byte; the twin and the replicate are request-identical copies of it.",
    "Every sampled contact carries the lock's thinking fields (budget and cap of the active model fragment); a sampled contact's decoding override changes only the sampling keys.",
    "No deployment value (instances, tensor-parallel size, sequences per instance, devices) is registered here; nothing is flown before PORT_DEPLOYMENT_PROFILE is filled and the bound re-derived.",
    "No benefit claim and no admission: claims_benefit is false; rows stay unconsumed; the loop never advances to iteration 2 or 3 without reading the previous one.",
    "No flight before the coordinator countersigns the DRAFT rows, mints the decision receipt, every PENDING pin is produced and re-pinned and the cells are flipped.",
]


# ---- arm tables and facts re-read from the receipts ---------------------------------------------------------------------------------------------------


def arm_table_rows(kind: str) -> list[dict]:
    if kind == "construction":
        rows = [{"arm": arm, "role": f"trajectory {i + 1} (Solver decoding group)", "axes": {"decoding": dict(d)}}
                for i, (arm, d) in enumerate(zip(tf.SAMPLER_ARMS, tf.CONSTRUCTION_DECODING))]
        rows += [{"arm": tf.TWIN, "role": f"twin (replicate_of {tf.DELIVERING})", "axes": None}, {"arm": tf.DELIVERING, "role": "delivering arm, LAST (greedy, the lock seed)", "axes": None}]
        return rows
    return [{"arm": tf.T_ARM, "role": "candidate: Solver vote n = 3 with the iteration's banks", "axes": {"solver_samples": {"n": tf.VOTE_N, "decoding": dict(tf.VOTE_DECODING), "commit": dict(tf.VOTE_COMMIT)}}},
            {"arm": tf.TWIN, "role": f"twin (replicate_of {tf.DELIVERING})", "axes": None},
            {"arm": tf.REPLICATE, "role": f"replicate (replicate_of {tf.DELIVERING}, arm_kind replicate)", "axes": None},
            {"arm": tf.DELIVERING, "role": "delivering arm, LAST (greedy, the lock seed)", "axes": None}]


def _sha(path: Path) -> str:
    return lw._sha(path)


def audio_pin_check(root: Path, data_root: Path, dataset: str) -> dict:
    """Re-hash every recording of a dataset fragment against its pins on the data root (model-free)."""

    path = Path(root) / "configs" / "dataset" / f"{dataset}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    bad = [item["sample_id"] for item in doc["items"] if not (Path(data_root) / item["media_relpath"]).is_file() or _sha(Path(data_root) / item["media_relpath"]) != item["audio_sha256"]]
    if bad:
        raise SystemExit(f"{len(bad)} recordings of {dataset} do not match their audio pins on the data root (first: {bad[0]})")
    return {"dataset_fragment": dataset, "dataset_fragment_sha256": _sha(path), "checked": len(doc["items"]), "matching": len(doc["items"]), "split_identity_hash": doc["split_identity_hash"],
            "audio_seconds_total": doc["audio_seconds_total"]}


def r01c_facts(root: Path, data_root: Path) -> dict:
    """The R01C roster, its two interleaved halves and the split receipt, re-read and cross-checked (model-free); the 625 audio pins re-hashed on the data root."""

    roster = json.loads((Path(root) / tf.R01C_ROSTER).read_text(encoding="utf-8"))
    ids = [item["sample_id"] for item in roster["items"]]
    split = json.loads((Path(root) / tf.HALVES_SPLIT_RECEIPT).read_text(encoding="utf-8"))
    halves = {}
    for half, key, positions in (("R01C-H1", "ch1", slice(0, None, 2)), ("R01C-H2", "ch2", slice(1, None, 2))):
        spec = tf.CELLS[key]
        half_ids = [item["sample_id"] for item in tf._roster_doc(key)["items"]]
        record = split["halves"][half]
        file_sha = _sha(Path(root) / spec["roster"])
        if half_ids != ids[positions] or record["count"] != len(half_ids) or record["file_sha256"] != file_sha or record["split_identity_hash"] != spec["split_identity_hash"]:
            raise SystemExit(f"the {half} roster, its split receipt and the interleave rule (items[{positions.start}::2] of the R01C roster) disagree")
        halves[half] = {"roster": spec["roster"], "roster_id": spec["roster_id"], "count": len(half_ids), "file_sha256": file_sha, "split_identity_hash": spec["split_identity_hash"],
                        "audio_seconds": round(tf.roster_audio(key), 4), "strata_counts": record["strata_counts"]}
    if sorted(halves["R01C-H1"]["strata_counts"]) != sorted(halves["R01C-H2"]["strata_counts"]) or len(ids) != 625 or len(set(ids)) != 625:
        raise SystemExit("the halves do not cover the same scenarios or R01C is not 625 distinct rows")
    return {"roster": tf.R01C_ROSTER, "roster_id": roster["roster_id"], "roster_file_sha256": _sha(Path(root) / tf.R01C_ROSTER), "split_identity_hash": roster["split_identity_hash"],
            "rows": len(ids), "audio_seconds": round(sum(float(i["audio_seconds"]) for i in roster["items"]), 7), "halves": halves,
            "halves_split_receipt": {"path": tf.HALVES_SPLIT_RECEIPT, "sha256": _sha(Path(root) / tf.HALVES_SPLIT_RECEIPT), "split_mode": split["split_mode"]},
            "audio_pins": audio_pin_check(root, data_root, tf.R01C_DATASET)}


def r01c_audio_cap() -> int:
    return math.ceil(sum(float(i["audio_seconds"]) for i in json.loads((REPO_ROOT / tf.R01C_ROSTER).read_text(encoding="utf-8"))["items"]) - 1e-9)


# ---- identities, plans, cost -----------------------------------------------------------------------------------------------------------------------------


def identities(root: Path) -> dict:
    out = {}
    for key in tf.ALL_KEYS:
        spec = tf.CELLS[key]
        document = json.loads((Path(root) / "configs" / "experiment" / f"{spec['cell']}.json").read_text(encoding="utf-8"))
        cell = document["cells"][0]
        composed = compose(Path(root), cell["model_fragment"], cell["dataset_fragment"], cell["baseline_fragment"], spec["cell"])
        values, split = composed.values, document["population"]["split_identity_hash"]
        tdc.pdc.verify_composed_module(Path(root), values, spec["cell"])          # writer self-check (lock plan 3.14): the loader's own call on the composed T cell
        fit = dict(values.get("native_contact_fit") or {})
        budget = document["family_budget"]
        if (fit.get("run_id"), fit.get("attempt_id"), fit.get("contact_cap")) != (spec["fit_run"], spec["fit_attempt"], tf.fit_calls(key)):
            raise SystemExit(f"the composed {key} cell's native_contact_fit is not the T FIT identity and cap")
        if (budget["generation_calls_cap"], budget["fit_endpoint_calls_cap"], budget["repeated_speech_audio_seconds_cap"]) != (tf.generation_calls(key), tf.fit_calls(key), tf.audio_seconds(key)):
            raise SystemExit(f"the {key} experiment caps are not the T family's arithmetic")
        if values.get("policy") != tf.program_id(key) or (cell["run_id"], cell["attempt_id"]) != (spec["run"], spec["attempt"]) or "tfrl_v2" not in values:
            raise SystemExit(f"the composed {key} cell's program / ids / tfrl_v2 block are not the registered ones")
        if split != spec["split_identity_hash"] or document["population"]["count"] != spec["rows"] or cell["model_fragment"] != pf.MODEL_FRAGMENT:
            raise SystemExit(f"the {key} population or model fragment is not the registered one")
        out[key] = {
            "key": key, "experiment": spec["cell"], "cell": cell["cell_id"], "role": spec["role"], "run_id": cell["run_id"], "attempt_id": cell["attempt_id"],
            "fit_run_id": spec["fit_run"], "fit_attempt_id": spec["fit_attempt"], "baseline_fragment": cell["baseline_fragment"], "dataset_fragment": cell["dataset_fragment"],
            "model_fragment": cell["model_fragment"], "arm": values["arm"], "program": values["policy"], "rows": document["population"]["count"], "split_identity_hash": split,
            "roster_receipt": document["population"].get("roster_receipt"), "composed_config_hash": composed.config_hash,
            "protocol_hash": protocol_hash(composed.config_hash, values["arm"], split), "fit": fit, "family_budget": budget, "execution_plan": document["execution_plan"],
            "execution_profile": document["execution_profile"], "carrier_lock_key": values.get("carrier_lock_key"), "session": document["service_session"],
            "contacts_per_row": tf.contacts_per_row(key), "half": spec["half"], "trace_read_id": spec["trace_read_id"],
        }
    return out


def cost_block() -> dict:
    keys = tf.ALL_KEYS
    flights = ("ch1", "ch2", "fb")
    caps = {k: {"generation_calls": tf.generation_calls(k), "fit_endpoint_calls": tf.fit_calls(k), "speech_audio_seconds": tf.audio_seconds(k)} for k in keys}
    smoke_keys = ("csmoke", "fsmoke")
    three = {"generation_calls": sum(tf.generation_calls(k) for k in smoke_keys) + 3 * sum(tf.generation_calls(k) for k in flights) + 3 * 2 * REFLECTOR_CONTACTS,
             "fit_endpoint_calls": sum(tf.fit_calls(k) for k in smoke_keys) + 3 * sum(tf.fit_calls(k) for k in flights) + 3 * (1 + 2 * REFLECTOR_CONTACTS),
             "speech_audio_seconds": sum(tf.audio_seconds(k) for k in smoke_keys) + 3 * sum(tf.audio_seconds(k) for k in flights)}
    return {
        "status": "PROVISIONAL",
        "formula": {k: tf.card_hour_formula(k) for k in keys},
        "inputs": {"mix_rows": {k: tf.mix_rows(k) for k in keys}, "node_seconds": dict(fam.NODE_SECONDS), "margin": pf.GPU_MARGIN, "round_hours": pf.GPU_ROUND_H,
                   "smoke_wall_clock_hours": pf.SMOKE_WALL_CLOCK_HOURS, "contacts_per_row": {k: tf.contacts(k) for k in keys}},
        "unknowns_from_the_deployment_profile": ["T = tensor_parallel_size", "r = measured lock-only rows per minute of one instance at its sequences_per_instance",
                                                 "instances (wall-clock only; cancels in the card-hours)"],
        "illustration_not_registered": {"lock_rows_per_minute_per_instance": fam.ROWS_PER_MINUTE_FULL,
                                        "card_hours_per_unit_tensor_parallel": {k: tf.plan_gpu_hours_placeholder(k) for k in keys},
                                        "note": "at the historical lock rate the formulas read these numbers x T card-hours; the port's r is measured, not assumed"},
        "plan_field_placeholder": {k: tf.plan_gpu_hours_placeholder(k) for k in keys},
        "plan_field_note": ("execution_plan.planned_gpu_hours and family_budget.gpu_hours_fit_inclusive carry the formula evaluated at the historical rate with T = 1 (smokes 0.5) only because "
                            "the plan field must be a number; they are NOT the NPU bound. The ledger gpu-hours cells carry the formula and fail the exposure gate closed until the sweep "
                            "re-derives the bound and the main thread re-writes plan, rows and receipt together."),
        "caps": caps,
        "caps_note": ("generation = rows x (program contacts + one fallback contact per budgeted-thinking contact), FIT = rows x contacts x 1 counting endpoint (the raw-audio arm; "
                      "native_contact_required_cap), audio = ceil(roster speech-seconds x the generation contacts per row). Construction cells: 7 contacts per row (c7); Fold B cells: 10 (c10)."),
        "family_total_iteration_1": {"generation_calls": tf.cell_total(0), "fit_endpoint_calls": tf.cell_total(1), "speech_audio_seconds": tf.cell_total(2)},
        "rows_only_caps": {"r01c_encoding": {"calls": 625, "speech_audio_seconds": r01c_audio_cap(), "wall_clock_hours_ceiling": 2.0, "npu_hours": 0},
                           "reflector_per_iteration": {"generation_calls": 2 * REFLECTOR_CONTACTS, "fit_endpoint_calls": 1 + 2 * REFLECTOR_CONTACTS, "speech_audio_seconds": 0}},
        "three_iteration_upper_bound": dict(three, note="smokes once, then three iterations of the two construction halves, the Fold B flight and the reflector"),
    }


def serving_block() -> dict:
    return lw.serving_block()


def t_fit_proof() -> dict:
    """The T Fold B context fit at the active budget plus the table for both budgets, recomputed from the family constants."""

    view = pf.budget_view()
    table = tf.fit_table()
    active = table[view["switch"]]
    checks = {"estimate_at_bound_fits": active["fits"], "bound_is_positive": active["t_bound"] >= 1,
              "slot_holds_prompt_plus_cap": fam.OBSERVED_SOLVER_PROMPT_MAX + view["completion_cap"] <= fam.SLOT_TOKENS}
    return {"active_switch": view["switch"], "completion_cap": view["completion_cap"], "solver_prompt_limit_tokens": view["prompt_limit_tokens"], "a_bound": active["a_bound"],
            "t_bound": active["t_bound"], "frame_base_prompt_tokens": tf.fold_b_fit_base_tokens(), "estimate_at_bound_tokens": active["estimate_tokens"],
            "headroom_tokens": active["headroom_tokens"], "largest_fitting_bound": active["largest_fitting_bound"], "checks": checks, "ok": all(checks.values()),
            "per_switch": table,
            "note": ("estimate = 2,000 + 323 template + 2,000 exemplar cap + 400 rule cap + bound x 235 (the registered FIT estimate); the exact per-contact gate re-checks every REAL request. "
                     f"Observed lock Solver prompt maximum {fam.OBSERVED_SOLVER_PROMPT_MAX} tokens (l338_cost_inputs.json).")}


def model_text(key: str) -> str:
    view = pf.budget_view()
    p, s = tf.contacts(key)
    return (f"Frozen Qwen3-Omni-30B-A3B-Thinking BF16 safetensors ({pf.MODEL_LOCK_KEY}) on vLLM-Ascend, BF16 KV, per-sequence window {pf.PER_SEQUENCE_WINDOW_TOKENS:,} tokens, max_tokens "
            f"{view['completion_cap']} (reasoning budget {view['thinking_budget_tokens']} + answer reserve 1024 on both nodes), --reasoning-parser qwen3 with the registered forced-close "
            f"reasoning_end_str (model fragment {pf.MODEL_FRAGMENT}); deployment (instances, tensor-parallel size, sequences per instance, devices) = {pf.DEPLOYMENT_REF}; the server's own audio "
            f"tower over the original recordings under {pf.PROMPT_ORDER}; module program {tf.program_id(key)} ({p} Perception + {s} Solver contacts per row); no tool authority")


def switch_note() -> dict | None:
    return lw.switch_record()


def plan_docs(ids: Mapping[str, Mapping], commit: str, at: str, protocol_sha: str) -> dict:
    docs = {}
    fit_block = t_fit_proof()
    for key in tf.ALL_KEYS:
        i, spec = ids[key], tf.CELLS[key]
        arms = {row["arm"]: {"role": row["role"], "axes": row["axes"]} for row in arm_table_rows(spec["kind"])}
        base = {"cell": i["cell"], "composed_config_hash": i["composed_config_hash"], "expected_session_id": pf.SERVICE_SESSION, "experiment": i["experiment"],
                "family_budget": i["family_budget"], "preparation_base_commit": commit,
                "prepared_phase_supply": {"relpath": spec["supply_relpath"], "sha256": "MINTED_AT_THE_FLIP",
                                          "note": "this cell's own first-phase supply under its own run id (item E16), prepared by scripts/flights/repin_cell.py at the flip"},
                "protocol_document": {"relpath": f"{PKG}/protocol-registered-v1.json", "sha256": protocol_sha}, "registered_at_utc": at, "status": STATUS,
                "thinking_knobs": lw.port_thinking_knobs(), "no_think_node": None, "sampling_arms": json.loads(json.dumps(arms)), "arm_order": list(arms),
                "unused_budget_is_not_attempt_authority": True, "arm_visits": "first", "serving": serving_block(), "accelerator_bound": cost_block(),
                "declared_differences": tf.declared_differences(), "rails_pending": [r["id"] for r in tdc.rails_pending(key)],
                "one_factor": ("T changes the harness against A in the declared ways (exemplars replace supplied items, a rule bank, a Solver vote); it is read against A's lock arm "
                               "inside the same flight, with the replicate variance in every CI95, never as a one-factor contrast.")}
        if spec["trace_read_id"]:
            base["trace_read_id"] = spec["trace_read_id"]
        if switch_note():
            base["model_fragment_switch"] = switch_note()
        gen = dict(i["execution_plan"], attempt_id=i["attempt_id"], carrier_lock_key=i["carrier_lock_key"], protocol_hash=i["protocol_hash"], run_id=i["run_id"],
                   schema="saea-execution-plan-v1", split_identity_hash=i["split_identity_hash"], study="speech-aware-evidence-acquisition")
        route = ("Shared core.flight.cli worker on this cell alone (the smoke step); never scored." if spec["smoke"] else
                 f"Shared core.flight.cli campaign over this cell on the port deployment ({pf.DEPLOYMENT_REF}); one official score per arm column (stage3_rollout_score) after the seal. "
                 "The T chain script is not written yet (open item).")
        docs[tf.plan_file_name(key, "generation")] = {
            **base, "execution_plan": gen, "execution_route": route, "plan_kind": "generation", "schema": "saea-registered-generation-plan-v1",
            "purpose": f"{i['run_id']}: {spec['role']}, NPU port.",
            "scoring": "None: the smoke is never scored." if spec["smoke"] else "Per-arm official column scoring after a successful seal; no partial score."}
        fit = dict(gen, attempt_id=i["fit_attempt_id"], run_id=i["fit_run_id"], planned_model_calls=i["family_budget"]["fit_endpoint_calls_cap"])
        docs[tf.plan_file_name(key, "fit")] = {
            **base, "execution_plan": fit, "plan_kind": "fit", "schema": "saea-registered-fit-plan-v1",
            "execution_route": "The shared core.flight.cli FIT path under this cell's own native_contact_fit ids; request envelopes only, no generation.",
            "context_fit": json.loads(json.dumps(fit_block)), "purpose": f"{i['fit_run_id']}: request envelopes only; reads nothing."}
    return docs


def signoff_docs() -> dict:
    out = {}
    for kind, name in (("construction", "signoff-conditions-construction-v1.json"), ("fold-b", "signoff-conditions-fold-b-v1.json")):
        arms = tf.CONSTRUCTION_ARM_ORDER if kind == "construction" else tf.FOLD_B_ARM_ORDER
        doc = dict(fam.signoff_conditions(), package=tf.PKG_NAME, expected_arms=sorted(arms),
                   table_relpath=f"analysis/{tf.READ_RUN_ID}/smoke-decode-cost-{kind}.json")
        doc["_note"] = (f"T machine sign-off conditions of the {kind} smoke, copied from l338_family.signoff_conditions() with the package, the table path "
                        f"(analysis/{tf.READ_RUN_ID}/smoke-decode-cost-{kind}.json) and the expected arms ({', '.join(sorted(arms))}) replaced. The T smoke reader that writes the table does "
                        "not exist yet (open item); until it does the sign-off is a manual coordinator read of the smoke's per-arm decode cost.")
        out[name] = doc
    return out


# ---- the ledger rows -----------------------------------------------------------------------------------------------------------------------------------


def scope_text(key: str) -> str:
    spec = tf.CELLS[key]
    return (f"DRAFT -- REGISTERED. T cell of the three-way comparison (plan {tf.PLAN}), iteration {tf.ITERATION}: {spec['role'][0].upper()}{spec['role'][1:]}: A's lock harness with "
            f"{tdc.arms_text(key)} (program {tf.program_id(key)}, {tf.contacts_per_row(key)} contacts per row); served by the BF16 / vLLM-Ascend Thinking core over the original "
            f"recordings ({pf.PROMPT_ORDER}) with the port's re-authored knowledge (intent-boundaries-npu-port-v1, slot-contracts-npu-port-v2) and the port's regenerated retrieval vectors "
            f"(CPU float32, forced causal); {DIFF_TEXT}")


def carrier_text(key: str) -> str:
    spec = tf.CELLS[key]
    if spec["kind"] != "construction":
        return f"slurp TRAIN Fold B {spec['rows']}"
    suffix = "-" + spec["half"].split("-")[1] if spec["half"] else ""
    return f"{CONSTRUCTION_SCOPE_SPLIT.replace('R01C TRAIN', 'R01C' + suffix + ' TRAIN')} {spec['rows']}"


def cell_rows(ids: Mapping[str, Mapping], commit: str, at: str, shas: Mapping[str, str], keys=None) -> list[list[str]]:
    rows = []
    for key in keys or tf.ALL_KEYS:
        i, spec = ids[key], tf.CELLS[key]
        for fit in (False, True):
            name = tf.plan_file_name(key, "fit" if fit else "generation")
            calls = i["family_budget"]["fit_endpoint_calls_cap" if fit else "generation_calls_cap"]
            rows.append([at[:10], i["fit_run_id"] if fit else i["run_id"], i["execution_profile"], scope_text(key) + (" -- FIT spend identity" if fit else ""), model_text(key), carrier_text(key),
                         "discovery", i["split_identity_hash"], "no", str(calls), str(i["family_budget"]["repeated_speech_audio_seconds_cap"]), tf.ledger_gpu_cell(key),
                         INHERITED_R01C if spec["kind"] == "construction" else INHERITED_FOLD_B, f"{PKG}/{name}",
                         (f"DRAFT at study commit {commit}; plan SHA256 {shas.get(name, 'pending')}; drafted by the agent on the owner direction of 2026-10-30 and owner decision 2026-10-28 "
                          f"item 4, owner countersignature pending; the decision receipt {tf.DECISION_RELPATH} is not minted; the gpu-hours cell is the PROVISIONAL card-hour formula "
                          f"({tf.card_hour_formula(key)}) and refuses the exposure gate until the measured sweep fills {pf.DEPLOYMENT_REF}; every PENDING pin of the composed config is the "
                          "zero sentinel; nothing authorised.")])
    return rows


def _draft_note(commit: str, what: str) -> str:
    return (f"DRAFT TEXT, NOT YET RUN: model-free (or tool-only) preparation row written at study commit {commit}; {what}; authorises nothing; the main thread amends this row with the "
            "measured facts after the run and the owner countersigns it.")


def prep_rows(facts: Mapping, commit: str, at: str) -> list[list[str]]:
    date = at[:10]
    r01c_split = facts["r01c"]["split_identity_hash"]
    audio = r01c_audio_cap()
    carrier = f"{CONSTRUCTION_SCOPE_SPLIT} 625"
    mf = MF_MODEL
    h1, h2 = facts["r01c"]["halves"]["R01C-H1"], facts["r01c"]["halves"]["R01C-H2"]
    no_new = f"No new speech exposure: every vector read was sealed by {tdc.R01C_ENCODE_RUN}"
    return [
        [date, tdc.R01C_ENCODE_RUN, PROFILE,
         "Port twin of SAEA-SLURP-STAGE3-CONSTRUCTION-R01-SPEECH-QUERY-ENCODING-V1 restricted to the 625 R01C rows: complete-speech retrieval-query encoding for the TFRL-v2 construction roster; no answers or scoring",
         "Frozen local Omni-Embed-Nemotron-3B retrieval tool on the CPU float32 worker pool; no answering core", carrier, "discovery", r01c_split, "no", "625", str(audio), "2.0",
         INHERITED_R01C + " This row adds only the port's pinned retrieval-encoder contacts.", f"{tdc.R01C_PENDING['encode_receipt']['relpath']} (sha256 PENDING)",
         _draft_note(commit, "CPU float32 retrieval encoder with the attention geometry forced causal and measured at worker start (declared differences 8.4 and 8.7); planned_gpu_hours is the attempt's "
                             "wall-clock ceiling and the NPU hours are 0; rows-only item: the encoder registration document and chain script are not written here")],
        [date, tdc.R01C_ALIAS_RUN, "model-free-check",
         ("Port twin of SAEA-SLURP-STAGE3-CONSTRUCTION-R01-ALIAS-V1: carry the port's sealed R01C content-dimension speech-query segment vectors into the intent-boundary retrieval dimension, "
          "values unchanged, segment identities preserved. Zero model, server, encoder, embedder, scorer, tokenizer, network, GPU and paid-API contacts."),
         mf, carrier, "discovery", r01c_split, "no", "0", "0", "0", f"{no_new}. Prior exposure inherited in full.",
         f"{tdc.R01C_PENDING['alias_receipt']['relpath']} (sha256 PENDING); {tdc.R01C_PENDING['segments']['relpath']} (sha256 PENDING)",
         _draft_note(commit, "output values byte-identical to the parent vectors") + " " + MF_TAIL],
        [date, tdc.R01C_INPUTS_RUN, "model-free-check",
         ("The port's model-free CONTENT DERIVATIVE of the R01C speech-query vectors (scripts/build_slurp_query_content_derivative.py), the exemplar channel's query source. "
          "Zero model, server, encoder, embedder, scorer, tokenizer, network, GPU and paid-API contacts."),
         mf, carrier, "discovery", r01c_split, "no", "0", "0", "0", f"{no_new} and its model-free alias. Prior exposure inherited in full.",
         f"{tdc.R01C_PENDING['content_provenance']['relpath']} (sha256 PENDING); {tdc.R01C_PENDING['content']['relpath']} (sha256 PENDING)", _draft_note(commit, "flatten the alias output") + " " + MF_TAIL],
        [date, tdc.R01C_AGREEMENT_RUN, "model-free-check",
         ("Port twin of SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-R01C-V1: the pre-computed, self-excluded exemplar-intent agreement table for R01C, arithmetic over the port minus-r01 pool snapshot, "
          "the port pool key vectors and the port R01C speech query vectors through the repository's own recall/rank/fuse. Zero model contacts."),
         mf, carrier, "discovery", r01c_split, "no", "0", "0", "0", f"{no_new}, its alias and content derivative. No gold label of any R01C row is opened. Prior exposure inherited in full.",
         f"{tdc.R01C_PENDING['agreement']['relpath']} (sha256 PENDING); {tdc.R01C_PENDING['agreement_receipt']['relpath']} (sha256 PENDING)",
         _draft_note(commit, "built against the minus-r01 snapshot, 20 votes per row") + " " + MF_TAIL],
        [date, HALVES_RUN, "model-free-check",
         ("Two frozen, scenario-complete INTERLEAVED halves of R01C (R01C-H1 = even roster positions, R01C-H2 = odd positions; plan section 11 ruling 1) for the TFRL-v2 cross-fit: a pure selection of the "
          "parent receipt's items in parent order, no draw and no seed, each half with its own roster receipt and split identity. Zero model contacts."),
         mf, carrier, "discovery", r01c_split, "no", "0", "0", "0", "No new speech exposure. Prior exposure inherited in full.",
         f"{h1['roster']} (sha256 {h1['file_sha256']}); {h2['roster']} (sha256 {h2['file_sha256']}); {facts['r01c']['halves_split_receipt']['path']} (sha256 {facts['r01c']['halves_split_receipt']['sha256']})",
         (f"RUN BEFORE THIS ROW WAS WRITTEN, MODEL-FREE (DRAFT, root countersignature pending): written by the T3 split tool at study commit {commit}. H1 {h1['count']} rows, roster id {h1['roster_id']}, "
          f"split identity {h1['split_identity_hash']}; H2 {h2['count']} rows, roster id {h2['roster_id']}, split identity {h2['split_identity_hash']}; parent roster file sha256 "
          f"{facts['r01c']['roster_file_sha256']}; split mode {facts['r01c']['halves_split_receipt']['split_mode']}; both halves cover all 18 scenarios. {MF_TAIL}")],
        [date, RULE_BANK_I0_RUN, "model-free-check",
         ("Release of the hand-authored seed rule bank rule-bank-i0 (at least 4 consumption-instruction x evidence-ordering variants) as a knowledge store under the construction scope R01C with forbidden "
          "rosters Fold A v2, Fold B v2 and DEV-1000. The iteration-1 construction flights are seed-only and consume no bank; the bank is held ready for the rule-variants mode of iteration 2 and later. "
          "Zero model contacts."),
         mf, f"{CONSTRUCTION_SCOPE_SPLIT} (scope only)", "discovery", r01c_split, "no", "0", "0", "0", "No speech exposure.",
         f"{tdc.SEED_RULE_BANK_DIR}/release.json (sha256 PENDING)", _draft_note(commit, "authored by the analyst model under the planner's guidance") + " " + MF_TAIL],
    ]


def loop_rows(ids: Mapping[str, Mapping], facts: Mapping, commit: str, at: str, shas: Mapping[str, str]) -> list[list[str]]:
    date = at[:10]
    r01c_split = facts["r01c"]["split_identity_hash"]
    carrier = f"{CONSTRUCTION_SCOPE_SPLIT} 625"
    gpu_smoke = f"PROVISIONAL: {pf.SMOKE_WALL_CLOCK_HOURS} x instances x T NPU card-hours; T and instances from PORT_DEPLOYMENT_PROFILE (pending)"
    construction = cell_rows(ids, commit, at, shas, keys=("csmoke", "ch1", "ch2"))
    fold_b = cell_rows(ids, commit, at, shas, keys=("fsmoke", "fb"))
    credit = [date, CREDIT_RUN, "model-free-check",
              ("Per-row credit of the iteration-1 trajectories against lock (reward = row F1 of the trajectory minus row F1 of lock, from the sealed official per-row scores; counted only where lock and aa "
               "agree), the contrast pairs, and the exemplar distillation (each positive's evidence-to-answer path, keyed by scenario, capped at B_max per scenario). Zero model contacts."),
              MF_MODEL, carrier, "discovery", r01c_split, "no", "0", "0", "0",
              f"No new speech exposure: reads only the outputs of {ids['ch1']['run_id']} and {ids['ch2']['run_id']}.",
              f"prepared/{CREDIT_RUN}/credit.jsonl, contrast-pairs.jsonl, receipt.json (sha256 PENDING)", _draft_note(commit, "official per-row scores only") + " " + MF_TAIL]
    reflector_scope = ("DRAFT -- REGISTERED. The contrast-pair reflector of TFRL-v2 iteration 1: at most one text-only contact per SLURP scenario that has at least one same-row positive / negative contrast "
                       f"pair (at most {REFLECTOR_CONTACTS}), each returning a bounded number of consumption rules that must cite their contrast-pair ids; a rule without a pair is refused by script; "
                       "a deterministic curator ranks the rules. No audio, no Fold A or Fold B material.")
    refl_model = ("Frozen Qwen3-Omni-30B-A3B-Thinking BF16 on vLLM-Ascend, text-only seam, no audio part and no module program; model fragment " + pf.MODEL_FRAGMENT + "; deployment = " + pf.DEPLOYMENT_REF)
    reflector = [
        [date, REFLECTOR_RUN, PROFILE, reflector_scope, refl_model, f"{CONSTRUCTION_SCOPE_SPLIT} (text material only)", "discovery", r01c_split, "no", str(2 * REFLECTOR_CONTACTS), "0", gpu_smoke,
         "No new speech exposure.", f"{PKG}/README.md (rows-only item: no plan file here)",
         (f"DRAFT at study commit {commit}; {REFLECTOR_CONTACTS} contacts plus {REFLECTOR_CONTACTS} answer-only fallback contacts; owner countersignature pending; the gpu-hours cell is the "
          "PROVISIONAL allowance; nothing authorised.")],
        [date, REFLECTOR_FIT_RUN, PROFILE, reflector_scope + " -- FIT spend identity", refl_model, f"{CONSTRUCTION_SCOPE_SPLIT} (text material only)", "discovery", r01c_split, "no",
         str(1 + 2 * REFLECTOR_CONTACTS), "0", gpu_smoke, "No new speech exposure.", f"{PKG}/README.md (rows-only item: no plan file here)",
         (f"DRAFT at study commit {commit}; FIT cap 1 + 2 x {REFLECTOR_CONTACTS} = {1 + 2 * REFLECTOR_CONTACTS} (the prepared-text rule: one base contact plus two per reflector contact); "
          "owner countersignature pending; nothing authorised.")]]
    release = [date, BANK_RELEASE_RUN, "model-free-check",
               ("Release of the iteration-1 exemplar bank (" + tdc.BANKS["exemplar_bank"]["dir"] + ") and rule bank (" + tdc.BANKS["rule_bank"]["dir"] + ") as knowledge stores under the construction scope "
                "R01C, forbidden rosters Fold A v2, Fold B v2 and DEV-1000; both are pinned by the Fold B baselines' tfrl_v2 blocks, whose composed hash therefore moves at the re-pin. Zero model contacts."),
               MF_MODEL, f"{CONSTRUCTION_SCOPE_SPLIT} (scope only)", "discovery", r01c_split, "no", "0", "0", "0", "No speech exposure.",
               f"{tdc.BANKS['exemplar_bank']['dir']}/release.json, {tdc.BANKS['rule_bank']['dir']}/release.json (sha256 PENDING)", _draft_note(commit, "union of the two halves' banks") + " " + MF_TAIL]
    return construction + [credit] + reflector + [release] + fold_b


# ---- measurement blocks, countersign texts, README --------------------------------------------------------------------------------------------------------


def measurement_blocks(ids: Mapping[str, Mapping]) -> dict:
    out = {}
    for key in tf.ALL_KEYS:
        spec, i = tf.CELLS[key], ids[key]
        run = i["run_id"]
        if spec["smoke"]:
            out[run] = {"measurement_design": "form-proof", "expected_effect_pp": 0.0, "claims_benefit": False, "rows": spec["rows"],
                        "primary_metric": "none (structural form, per-arm decode cost and the request form only)",
                        "read_roster": spec["roster_id"], "tuning_roster": "none",
                        "note": (f"Four rows prove FORM only: the {tf.program_id(key)} module program with {tdc.arms_text(key)} loads on the port deployment, no node hits its cap, the live phase fit "
                                 "passes and the receipt seals. Scored by no one; four rows claim nothing under instrument rule 5.")}
        elif spec["kind"] == "construction":
            out[run] = {"measurement_design": "material-source", "expected_effect_pp": 0.0, "claims_benefit": False, "rows": spec["rows"],
                        "primary_metric": "none: per-row scores are read OFFLINE as the credit and distillation material of TFRL-v2",
                        "read_roster": spec["roster_id"], "tuning_roster": "none",
                        "note": ("A measurement-free trajectory-sampling column on the construction roster: it carries no contrast, no floor and no benefit claim; its sealed per-row scores feed the "
                                 "model-free credit, the contrast pairs and the exemplar distillation. The construction gain is read from the sealed scores by the acceptance rule.")}
        else:
            out[run] = {"measurement_design": "same-flight-node-column", "expected_effect_pp": EXPECTED_EFFECT_PP, "claims_benefit": False, "claim": "nominated", "rows": spec["rows"],
                        "primary_metric": PRIMARY_METRIC, "primary_metric_key": "slu",
                        "variant_cost": {"extra_contacts_per_row": tf.contacts_per_row(key) - 2, "extra_audio_seconds": round(tf.roster_audio(key) * (tf.contacts_per_row(key) - 2), 2)},
                        "template_variants": {"node": "perception", "candidate_arms": 1, "twin": True}, "variant_kind": "solver-sampling",
                        "aa_arm_id": tf.TWIN, "replicate_arm_ids": [tf.REPLICATE], "read_roster": "SLURP-EVO-FOLD-B-V2", "tuning_roster": "none",
                        "note": (f"A nomination read, not a benefit claim: expected_effect_pp {EXPECTED_EFFECT_PP} is the low end of the plan's +2 to +4 pp if consumption is the bottleneck (else 0). The null is "
                                 f"the in-flight twin {tf.TWIN} and the replicate {tf.REPLICATE}; the design resolves {BAR_SLU_PP} pp SLU / {BAR_INTENT_PP} pp intent at n = 1,080 (decision 190 node-column "
                                 "floors), with the replicate variance in every CI95.")}
    return out


def countersign_texts(ids: Mapping[str, Mapping]) -> dict:
    head = ("REGISTERED BEFORE CONTACT (root countersigned <UTC timestamp> by the owner on the three-way direction of 2026-10-30: " + tf.OWNER_DECISIONS + "; the T cell of the three-way comparison "
            "on the BF16 / vLLM-Ascend Thinking core; deployment <the filled PORT_DEPLOYMENT_PROFILE: instances, tensor-parallel size, sequences per instance, devices>, card-hour bound "
            "<the re-derived number> written into the gpu-hours cell; ")
    tail = f"; the decision receipt {tf.DECISION_RELPATH} is minted after this countersignature)"
    rows = {}
    for key in tf.ALL_KEYS:
        i = ids[key]
        rows[i["run_id"]] = head + tf.CELLS[key]["role"] + tail
        rows[i["fit_run_id"]] = head + f"live phase-fit identity of the {tf.CELLS[key]['role']}" + tail
    rows[REFLECTOR_RUN] = head + "the contrast-pair reflector of iteration 1 (text only)" + tail
    rows[REFLECTOR_FIT_RUN] = head + "live phase-fit identity of the iteration-1 reflector" + tail
    return {"_note": ("Before countersigning: replace each gpu-hours cell's PROVISIONAL formula with the number re-derived from the measured sweep (and the same number in the experiments' "
                      "planned_gpu_hours / gpu_hours_fit_inclusive and the draft receipt's budget), then replace the leading 'DRAFT -- REGISTERED' scope marker and the 'DRAFT' note prefix "
                      "of each row with the text below and mint the receipt. Rows-only items are countersigned when they are run."),
            "decision_id": tf.DECISION_ID, "decision_receipt": tf.DECISION_RELPATH, "rows": rows}


def cell_table(ids: Mapping[str, Mapping]) -> str:
    lines = ["| cell | run | FIT run | rows | program | generation calls | FIT calls | speech-seconds | composed config hash | protocol hash |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key in tf.ALL_KEYS:
        i = ids[key]
        b = i["family_budget"]
        lines.append(f"| `{i['cell']}` | `{i['run_id']}` | `{i['fit_run_id']}` | {i['rows']} | `{i['program']}` | {b['generation_calls_cap']:,} | {b['fit_endpoint_calls_cap']:,} | "
                     f"{b['repeated_speech_audio_seconds_cap']:,} | `{i['composed_config_hash']}` | `{i['protocol_hash']}` |")
    return "\n".join(lines)


def rails_status_text(state: Mapping, shas: Mapping[str, str]) -> str:
    """The README section that states the rails marker and the three deliverables it rests on."""

    d = state["deliverables"]
    return (f"## Rails status: `{state['marker']}`\n\n"
            f"The marker is `{rails.MARKER_DECLARED}` only when all three deliverables below exist (checked from disk by the writer, `tfrl_v2_port_rails.rails_status`); it is `{rails.MARKER_PENDING}` otherwise. "
            f"{state['meaning']}.\n\n"
            f"- stage-3 label and consumption rail: `{d['stage3_label_and_consumption_rail']['label']}` (`core.stage3_algorithms`) and `{rails.CONSUMPTION_RAIL}` (the `tfrl_v2_bank` frame the module block accepts, the flight-start refusals, the Solver composition, `used_basis` read from the trace): ok {d['stage3_label_and_consumption_rail']['ok']}\n"
            f"- analysis recipe and reader: `{rails.RECIPE_NAME}` (sha256 `{shas[rails.RECIPE_NAME]}`, pinned in `{rails.READ_ROW_NAME}`) and `{rails.READER}`: ok {d['analysis_recipe_and_reader']['ok']}\n"
            f"- bank-release registration: `{rails.BANK_RELEASE_NAME}` (sha256 `{shas[rails.BANK_RELEASE_NAME]}`; per iteration the source halves and split identities, the bank receipt hashes (PENDING), the union hash, the measure, the token counts, the consuming cell ids, and the rule that T0 rows are never bank sources): ok {d['bank_release_registration']['ok']}\n\n")


def readme_text(ids: Mapping[str, Mapping], commit: str, rails_state: Mapping | None = None, rails_shas: Mapping[str, str] | None = None) -> str:
    view = pf.budget_view()
    table = tf.fit_table()
    totals = (tf.cell_total(0), tf.cell_total(1), tf.cell_total(2))
    bullets = lambda items: "\n".join(f"- {x}" for x in items)  # noqa: E731
    return (
        "# T cell: the agent system plus TFRL-v2 on the BF16 / vLLM-Ascend Thinking core, iteration 1 (DRAFT / HELD)\n\n"
        f"The T cell `{tf.CELL_ID}` of the three-way comparison (plan `{tf.PLAN}`; drafts `{tf.DRAFT_PACKAGE}`), registered before contact on {tf.DATE} at study commit `{commit}` by "
        "`scripts/flights/nx/tfrl_v2_port_write_registration.py` from the cells `scripts/flights/nx/tfrl_v2_port_derive_cells.py` derives. Design: "
        f"`{tf.DESIGN_DOC}`. Authority: {tf.OWNER_DECISIONS}. Development numbers only; never a test claim; claims_benefit false. Every row is unconsumed and DRAFT; nothing here authorises a contact.\n\n"
        "## Cells (chain order)\n\n" + cell_table(ids) + "\n\n"
        f"Family total of iteration 1 (five cells): {totals[0]:,} generation calls, {totals[1]:,} FIT calls, {totals[2]:,} speech-seconds. The caps do not depend on the deployment. "
        "Construction cells run 7 contacts per row, Fold B cells 10.\n\n"
        "## The programs\n\n"
        f"- **Construction (c7).** The Solver decoding column group {', '.join(tf.SAMPLER_ARMS)} (temperature 0.7, top_p 0.80 / 0.90 / 0.95 / 1.00, top_k 20, the lock seed), the twin `{tf.TWIN}` and "
        f"`{tf.DELIVERING}` LAST: 1 Perception + 6 Solver contacts per row, compiled by the module compiler. Chosen over the Perception-level sampling column (n = 5, c10) because the sampler's "
        "trace and credit tooling key scores by arm label s1..s4; the alternative is implemented and tested, not registered. The plan's seed-only trajectories need a per-arm seed the Solver "
        "decoding group refuses, so the arms differ in top_p.\n"
        f"- **Fold B (c10).** One Perception-level sampling column `{tf.T_ARM}` (a Solver vote of n = {tf.VOTE_N} at 0.6 / 0.95 / 20, commit `{tf.VOTE_COMMIT['policy']}`, min_support 2), the twin `{tf.TWIN}`, "
        f"the replicate `{tf.REPLICATE}` and `{tf.DELIVERING}` LAST: 4 Perception + 6 Solver contacts per row. `{tf.PLANNED_COMMIT_POLICY}` does not exist; greedy_first makes the vote one greedy plus two drawn.\n"
        f"- **Supply bound.** T's bound is A's bound minus {tf.EXEMPLAR_ITEM_EQUIVALENT} (exemplars replace supplied items) with the 2,000-token exemplar cap and the 400-token rule cap folded into the frame's base "
        f"prompt tokens. Active budget `{view['switch']}`: bound {table[view['switch']]['t_bound']}, estimate {table[view['switch']]['estimate_tokens']:,} of {view['prompt_limit_tokens']:,} tokens.\n\n"
        "## Budget switch outcome\n\n"
        f"- default (3,072 / 4,096): T bound {table['default']['t_bound']}, estimate {table['default']['estimate_tokens']:,} of {table['default']['solver_prompt_limit_tokens']:,}: fits.\n"
        f"- budget6144-max7168: T bound {table['budget6144-max7168']['t_bound']}, estimate {table['budget6144-max7168']['estimate_tokens']:,} of {table['budget6144-max7168']['solver_prompt_limit_tokens']:,}: "
        f"does NOT fit; the largest fitting bound is {table['budget6144-max7168']['largest_fitting_bound']} items. The writer and derive refuse under that switch (owner item O-switch-bound).\n\n"
        "## Acceptance and kill rules (plan section 4 with M1 / M2)\n\n"
        "An iteration is ACCEPTED only if every one of:\n\n" + bullets(ACCEPTANCE["iteration_accepted_only_if"]) + "\n\n"
        f"CONFIRMED only if: {ACCEPTANCE['confirmed_only_if']}.\n\nKill and stop:\n\n" + bullets(ACCEPTANCE["kill_and_stop"]) + f"\n\nExpected: {ACCEPTANCE['expected']}.\n\n"
        + (rails_status_text(rails_state, rails_shas) if rails_state is not None else "")
        + "## What the rails still lack\n\n"
        + bullets(f"`{r['id']}`: {r['lacks']}. {r['consequence'][0].upper()}{r['consequence'][1:]}." for r in tdc.rails_pending("fb") + tdc.rails_pending("ch1")[1:]) + "\n\n"
        "## Serving and accelerator bound (PROVISIONAL)\n\n"
        f"Model fragment `{pf.MODEL_FRAGMENT}`; the deployment is the placeholder `PORT_DEPLOYMENT_PROFILE` ({pf.PORT_DEPLOYMENT_PROFILE['filled_by']}). Bounds:\n\n"
        + bullets(f"{tf.CELLS[k]['stem']}: {tf.card_hour_formula(k)}" for k in tf.ALL_KEYS) + "\n\n"
        "The experiments' numeric planned_gpu_hours are the formula at the historical rate with T = 1 (smokes 0.5), a placeholder the plan field requires; the ledger gpu-hours cells carry the formula, "
        "so the exposure gate refuses these rows until the sweep re-derives the bound.\n\n"
        "## Owner items\n\n" + bullets(f"**{o['id']}**: {o['item']}" for o in OWNER_ITEMS) + "\n\n"
        "## Open questions\n\n" + bullets(OPEN_QUESTIONS) + "\n\n"
        "## Declared differences\n\n" + bullets(tf.declared_differences()) + "\n\n"
        "## Files\n\n"
        "`protocol-registered-v1.json`, ten plan files (generation and FIT of each cell), `registered-facts-v1.json` (stage3_measurement blocks keyed by the T run ids), "
        "`signoff-conditions-construction-v1.json`, `signoff-conditions-fold-b-v1.json`, `root-admission-draft-v1.json` (decision `" + tf.DECISION_ID + "`, relpath `" + tf.DECISION_RELPATH + "`), "
        "`countersign-row-texts-v1.json` and `draft-ledger-rows.md` (six preparation rows and the loop rows in chain order), and the rails files `" + rails.RECIPE_NAME + "`, `" + rails.READ_ROW_NAME + "` and `" + rails.BANK_RELEASE_NAME + "`. "
        "The configs are `configs/{baseline,experiment}/slurp-native-stage3-tfrl-v2-*` and two campaigns.\n\n"
        "## What the main thread does (in order)\n\n"
        "1. Commit this package, the five cells and the campaigns; do not append the ledger rows before the owner countersigns.\n"
        "2. Resolve the owner items; run the speed sweep and fill `PORT_DEPLOYMENT_PROFILE`; re-derive the card-hour bounds; re-run the derive and this writer.\n"
        "3. Run the rows-only items in order (R01C encoding, alias, content derivative, agreement, rule bank i0), re-pin every zero-sentinel pin, then the construction smoke and the two halves.\n"
        "4. Run credit and distillation, the reflector and the bank release; re-pin the Fold B cells' bank pins; run the Fold B smoke and the flight.\n"
        "5. Iterations 2 and 3 are registered only after the previous read.\n"
    )


def write_package(root: Path, commit: str, at: str | None, data_root: Path, *, append_ledger: bool = False, derive: bool = True) -> dict:
    root = Path(root)
    pf_active = pf.active_switch()
    tf.select_model_fragment()
    tf.require_fit()
    facts_path = root / PKG / "registered-facts-v1.json"
    at = at or (json.loads(facts_path.read_text(encoding="utf-8"))["registered_at_utc"] if facts_path.is_file() else datetime.now(timezone.utc).isoformat())
    derived = tdc.derive(root) if derive else None
    prep = lw.read_preparations(data_root)
    r01c = r01c_facts(root, data_root)
    fold_b_pins = audio_pin_check(root, data_root, tf.FOLD_B_DATASET)
    ids = identities(root)
    cost = cost_block()
    protocol = {
        "cells": {k: {kk: v[kk] for kk in ("attempt_id", "cell", "composed_config_hash", "experiment", "fit_attempt_id", "fit_run_id", "protocol_hash", "rows", "run_id", "split_identity_hash",
                                           "program", "contacts_per_row")} for k, v in ids.items()},
        "arms": {"construction": {row["arm"]: {"role": row["role"], "axes": row["axes"]} for row in arm_table_rows("construction")},
                 "fold-b": {row["arm"]: {"role": row["role"], "axes": row["axes"]} for row in arm_table_rows("fold-b")}},
        "blockers": [], "findings": FINDINGS, "port_findings": [{"id": "deployment-is-a-placeholder", "finding": pf.PORT_DEPLOYMENT_PROFILE["not_a_fact_here"],
                                                                  "remedy": "the measured speed sweep fills the profile"}],
        "decision_rule_fixed_here": ACCEPTANCE, "owner_items": OWNER_ITEMS, "open_questions": OPEN_QUESTIONS, "rails_pending": {k: tdc.rails_pending(k) for k in ("ch1", "fb")},
        "failure_policy": "Stop and preserve on the first failure; at most one root-diagnosed recovery re-registration per run id; run-cumulative budgets.",
        "declared_differences": tf.declared_differences(), "serving": serving_block(), "accelerator_bound": cost, "prohibitions": PROHIBITIONS,
        "protocol_id": "slurp-tfrl-v2-fold-b-npu-port-protocol-v1", "registered_at_utc": at, "schema": "saea-registered-protocol-v1", "status": STATUS, "study_commit_at_registration": commit,
        "the_question": ("Does a row-level, training-free loop (trajectories sampled on the construction column, per-row credit, an exemplar bank, a contrast-pair rule bank and a Solver vote) beat the "
                         "lock harness A on TRAIN Fold B, read PAIRED per row inside one flight against the delivering lock arm with the decoding term of the three request-identical copies "
                         "(lock, aa, lock-rep2) in every CI95, and does it hold on Fold A?")}
    if switch_note():
        protocol["model_fragment_switch"] = switch_note()
    protocol_sha = reg.wjson(root, f"{PKG}/protocol-registered-v1.json", protocol)
    plans = plan_docs(ids, commit, at, protocol_sha)
    shas = {name: reg.wjson(root, f"{PKG}/{name}", doc) for name, doc in plans.items()}
    shas["protocol-registered-v1.json"] = protocol_sha
    for name, doc in signoff_docs().items():
        shas[name] = reg.wjson(root, f"{PKG}/{name}", doc)
    rails_shas = rails.write_rails_files(root, reg.wjson, reg.wr)           # the analysis recipe, the READ row draft (recipe sha pinned) and the bank-release registration
    shas.update(rails_shas)
    rails_state = rails.rails_status(root)                                   # reads the three deliverables from disk; flips rails_pending -> rails_declared
    measurement = measurement_blocks(ids)
    facts = {
        "_note": ("Every model-free fact this DRAFT T registration rests on, and the canonical stage3_measurement blocks keyed by the T run ids "
                  "(scripts/checks/stage3_effect_floor_check.py, scripts/checks/flight_registration_check.py)."),
        "accounting": {"loop_id": tf.LOOP_ID, "iteration": tf.ITERATION, "programs": {"construction": tf.CONSTRUCTION_PROGRAM, "fold-b": tf.FOLD_B_PROGRAM},
                       "contacts_per_row": {k: {"perception": tf.contacts(k)[0], "solver": tf.contacts(k)[1]} for k in tf.ALL_KEYS},
                       "arms": {"construction": list(tf.CONSTRUCTION_ARM_ORDER), "fold-b": list(tf.FOLD_B_ARM_ORDER)}, "caps": cost["caps"],
                       "family_total_iteration_1": cost["family_total_iteration_1"], "accelerator_bound": cost,
                       "construction_group": {"chosen": tf.CONSTRUCTION_GROUP, "alternative": "perception-sampling",
                                              "alternative_program": tf.PROGRAM_OF["perception-sampling"], "alternative_contacts": tf.contacts("ch1", "perception-sampling"),
                                              "decoding": [dict(d) for d in tf.CONSTRUCTION_DECODING]},
                       "vote": {"n": tf.VOTE_N, "decoding": dict(tf.VOTE_DECODING), "commit": dict(tf.VOTE_COMMIT), "planned_policy": tf.PLANNED_COMMIT_POLICY}},
        "composed_config_hashes": {k: v["composed_config_hash"] for k, v in ids.items()},
        "protocol_hashes": {k: v["protocol_hash"] for k, v in ids.items()},
        "context_fit": t_fit_proof(), "budget_switch": {"active": pf_active, "per_switch": tf.fit_table()}, "identity": ids, "derive": derived,
        "pins": {"model_fragment": pf.MODEL_FRAGMENT, "serving": serving_block(),
                 "port_pins": {"path": "scripts/flights/nx/l338_port_pins.json", "sha256": lw.file_sha256(root, "scripts/flights/nx/l338_port_pins.json")},
                 "locked_template": {"relpath": fam.LOCKED_TEMPLATE, "sha256": fam.LOCKED_TEMPLATE_SHA},
                 "fold_b_preparations_reused": {k: {"receipt": v["receipt"], "receipt_sha256": v["receipt_sha256"], "artifact": v["artifact"], "artifact_sha256": v["artifact_sha256"]}
                                                for k, v in prep.items() if k != "supply"},
                 "fold_b_supply_not_reused": "the L338 flight's phase supply is a different run's; each T cell prepares its own",
                 "fold_b_audio_pins": fold_b_pins},
        "r01c": r01c,
        "pending_pins": {"r01c_port_artifacts": {k: dict(v, sha256=None, state="PENDING (zero sentinel in the composed config)") for k, v in tdc.R01C_PENDING.items()},
                         "banks": {k: {"release": f"{v['dir']}/release.json", "snapshot": f"{v['dir']}/snapshot.json", "store_role": v["store_role"], "state": "PENDING (zero sentinel)"}
                                   for k, v in tdc.BANKS.items()},
                         "seed_rule_bank_i0": f"{tdc.SEED_RULE_BANK_DIR}/release.json", "decision_receipt": tf.DECISION_RELPATH,
                         "supplies": {k: tf.CELLS[k]["supply_relpath"] for k in tf.ALL_KEYS}},
        "acceptance_and_kill_rules": ACCEPTANCE, "owner_items": OWNER_ITEMS, "open_questions": OPEN_QUESTIONS, "findings": FINDINGS,
        "rails_pending": {"construction": tdc.rails_pending("ch1"), "fold-b": tdc.rails_pending("fb")},
        "rails_status": rails_state, "bank_release": {"path": rails.BANK_RELEASE_RELPATH, "sha256": rails_shas[rails.BANK_RELEASE_NAME]},
        "declared_differences": tf.declared_differences(), "prohibitions": PROHIBITIONS, "registered_at_utc": at,
        "registration_basis": f"{tf.OWNER_DECISIONS}; {tf.DESIGN_DOC}. DRAFT: nothing here authorises a flip or a flight.",
        "read_rule": {"acceptance": ACCEPTANCE, "floors_pp": {"slu": BAR_SLU_PP, "intent": BAR_INTENT_PP}, "twin_bar_pp": TWIN_BAR_PP, "construction_gain_pp": CONSTRUCTION_GAIN_PP,
                      "bootstrap": BOOTSTRAP, "read_run_id": tf.READ_RUN_ID, "trace_read_ids": {k: tf.CELLS[k]["trace_read_id"] for k in ("ch1", "ch2")},
                      "replicates": [tf.TWIN, tf.REPLICATE], "delivering": tf.DELIVERING,
                      "analysis_recipe": {"path": rails.RECIPE_RELPATH, "sha256": rails_shas[rails.RECIPE_NAME], "read_row_draft": {"path": rails.READ_ROW_RELPATH, "sha256": rails_shas[rails.READ_ROW_NAME]},
                                          "reader": rails.READER, "stop_rule": {"construction_gain_pp": rails.STOP_GAIN_PP, "max_iterations": rails.MAX_ITERATIONS}}},
        "roster": {k: {"roster": tf.ROSTERS[k], "roster_id": tf.CELLS[k]["roster_id"], "split_identity_hash": tf.CELLS[k]["split_identity_hash"], "rows": tf.CELLS[k]["rows"]} for k in tf.ALL_KEYS},
        "rows_only_items": {"r01c_encoding": tdc.R01C_ENCODE_RUN, "alias": tdc.R01C_ALIAS_RUN, "content_derivative": tdc.R01C_INPUTS_RUN, "agreement": tdc.R01C_AGREEMENT_RUN, "halves": HALVES_RUN,
                            "rule_bank_i0_release": RULE_BANK_I0_RUN, "credit_and_distillation": CREDIT_RUN, "reflector": [REFLECTOR_RUN, REFLECTOR_FIT_RUN], "bank_release": BANK_RELEASE_RUN},
        "schema": "saea-registered-facts-v1", "stage3_measurement": measurement, "study_commit_at_registration": commit, "thinking_knobs": lw.port_thinking_knobs(), "no_think_node": None,
        "sampling_arms": {"construction": {row["arm"]: row["axes"] for row in arm_table_rows("construction")}, "fold-b": {row["arm"]: row["axes"] for row in arm_table_rows("fold-b")}},
        "locked_decoding": dict(fam.LOCKED_DECODING), "status": "DRAFT-HELD"}
    if switch_note():
        facts["model_fragment_switch"] = switch_note()
    shas["registered-facts-v1.json"] = reg.wjson(root, f"{PKG}/registered-facts-v1.json", facts)
    eligible_runs = tf.eligible_run_ids() + [REFLECTOR_RUN, REFLECTOR_FIT_RUN]
    eligible_attempts = tf.eligible_attempt_ids() + [REFLECTOR_ATTEMPT, REFLECTOR_FIT_ATTEMPT]
    draft = {
        "_draft_note": ("A DRAFT for the MAIN THREAD to mint on the data root (owner decision 2026-10-28 item 5). It authorises nothing while its status is not 'decided'. Before minting, the measured "
                        "sweep fills PORT_DEPLOYMENT_PROFILE and the PROVISIONAL accelerator bound below is replaced by the re-derived number."),
        "antecedents": {"protocol": {"path": f"{PKG}/protocol-registered-v1.json", "sha256": protocol_sha},
                        "registered_facts": {"path": f"{PKG}/registered-facts-v1.json", "sha256": shas["registered-facts-v1.json"]},
                        "plans": {name: {"path": f"{PKG}/{name}", "sha256": shas[name]} for name in plans}, "registration_base_commit": commit},
        "authorization_basis": f"{tf.OWNER_DECISIONS}; {tf.DESIGN_DOC}; the port authority {pf.HISTORICAL_PACKAGE}.",
        "budget": {"cells": {k: v["family_budget"] for k, v in ids.items()}, "paid_api_calls": 0, "plan_budget_is_run_cumulative": True,
                   "accelerator_bound": {"status": "PROVISIONAL", "formula": cost["formula"], "note": "gpu_hours_fit_inclusive in the cells is a numeric placeholder, not the NPU card-hour bound"},
                   "family_total_iteration_1": cost["family_total_iteration_1"], "rows_only_caps": cost["rows_only_caps"]},
        "claim_boundary": "No benefit, system-level, superiority, generalisation, held-out, confirmatory or paper-scale claim.",
        "decision": ("Fly the T cell of the three-way comparison on the BF16 / vLLM-Ascend Thinking core, iteration 1: construction smoke, the two construction halves, the contrast-pair reflector, then "
                     "the Fold B smoke and the Fold B selection flight, on the deployment PORT_DEPLOYMENT_PROFILE once it is filled and every PENDING pin is produced."),
        "decision_id": tf.DECISION_ID, "eligible_attempt_ids": eligible_attempts, "eligible_run_ids": eligible_runs,
        "not_authorized": ["any flight before the cells are flipped and the receipt is minted", "any flight before PORT_DEPLOYMENT_PROFILE is filled", "any Fold A, DEV or test-split contact",
                           "starting or touching any serving session outside the registered deployment", "any ASR, transcript, encoder or paid-API contact beyond the registered rows",
                           "--allow-output-change on the re-pin tool", "iteration 2 or 3 before the previous iteration is read"],
        "owner": "root", "population": {"fold_a_contact_authorized": False, "test_contact_authorized": False, "split_role": "discovery",
                                        "rosters": {k: tf.CELLS[k]["split_identity_hash"] for k in tf.ALL_KEYS}},
        "recorded_at_utc": None, "schema": "saea-campaign-decision-v1",
        "service": {"expected_session_id": pf.SERVICE_SESSION, "model_fragment": pf.MODEL_FRAGMENT, "deployment_profile": "PORT_DEPLOYMENT_PROFILE (pending)", "no_child_start_stop": True},
        "status": "DRAFT-NOT-MINTED"}
    shas["root-admission-draft-v1.json"] = reg.wjson(root, f"{PKG}/root-admission-draft-v1.json", draft)
    prep_rows_ = prep_rows(facts, commit, at)
    loop_rows_ = loop_rows(ids, facts, commit, at, shas)
    prep_text = lw.ledger_block(prep_rows_, (
        "Six DRAFT TEXT preparation rows of the T family, iteration 1 (R01C encoding, alias, content derivative, agreement table, interleaved halves, seed rule bank). The halves row is real (the rosters and "
        "the split receipt exist and were re-read here); the others are not yet run. No row authorises a contact."), tf.PREP_LEDGER_MARK)
    loop_text = lw.ledger_block(loop_rows_, (
        f"{len(loop_rows_)} unconsumed DRAFT rows of the T family, iteration 1, in chain order: the construction smoke and the two construction halves (generation and FIT identity each), credit and "
        "distillation, the reflector (generation and FIT), the bank release, the Fold B smoke and the Fold B flight (generation and FIT identity each); owner direction of 2026-10-30. "
        f"Package {PKG}. No row is countersigned and no row authorises a contact. Each gpu-hours cell of a model flight is the PROVISIONAL NPU card-hour formula, not a number: the exposure gate "
        "refuses these rows until the measured speed sweep fills PORT_DEPLOYMENT_PROFILE and the bound is re-derived."), tf.LEDGER_MARK + " (DRAFT, owner countersignature pending)")
    shas["draft-ledger-rows.md"] = reg.wr(root, f"{PKG}/draft-ledger-rows.md", prep_text + "\n" + loop_text)
    shas["countersign-row-texts-v1.json"] = reg.wjson(root, f"{PKG}/countersign-row-texts-v1.json", countersign_texts(ids))
    shas["README.md"] = reg.wr(root, f"{PKG}/README.md", readme_text(ids, commit, rails_state, rails_shas))
    prereg = (f"# T cell, iteration 1: the agent system plus TFRL-v2, preregistration (DRAFT / HELD)\n\nRegistered {at[:10]} at study commit `{commit}` by script. Development numbers only; claims_benefit false.\n\n"
              + "Runs: " + "; ".join(f"`{ids[k]['run_id']}` / `{ids[k]['fit_run_id']}`" for k in tf.ALL_KEYS) + ".\n\n"
              f"The package is `{PKG}/`; the acceptance and kill rules are in its README and facts.\n\n## The measurement declaration\n\n```json\n"
              + json.dumps({"stage3_measurement": measurement}, indent=2, ensure_ascii=False, sort_keys=True) + "\n```\n")
    shas["__prereg__"] = reg.wr(root, tf.PREREG, prereg)
    result = {"package": PKG, "shas": shas, "derive": derived, "ledger_appended": None}
    if append_ledger:
        ledger = root / reg.LEDGER
        current = io.open(ledger, encoding="utf-8", newline="").read()
        if tf.LEDGER_MARK in current or tf.PREP_LEDGER_MARK in current:
            result["ledger_appended"] = False
        else:
            io.open(ledger, "w", encoding="utf-8", newline="").write(current.rstrip("\n") + "\n\n" + prep_text + "\n" + loop_text)
            result["ledger_appended"] = True
    return result
