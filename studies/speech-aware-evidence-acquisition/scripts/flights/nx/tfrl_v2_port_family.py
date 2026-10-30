"""Shared identities and arithmetic of the T cell of the three-way comparison: ``SAEA-SLURP-TFRL-V2-FOLD-B-NPU-PORT-V1``, iteration 1 (plan
``docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md`` sections 4, 9, 10, 11; drafts ``docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/``).

One module, imported by the T derive, writer and tests, so no T id is spelled twice. Model-free. It reuses the L338 port family (``l338_port_family``: the BF16 /
vLLM-Ascend fragment and its thinking-budget switch, the deployment placeholder, the declared differences, the card-hour constants) and adds only what T changes:

* FIVE model cells with a full config chain, iteration 1 only (iterations 2-3 are registered after the previous read): the R01C construction smoke (4 rows), the two
  construction flights over the interleaved halves R01C-H1 (313 rows) and R01C-H2 (312 rows) (plan section 11 ruling 1), the Fold B smoke (4 rows) and the Fold B
  selection flight (1,080 rows);
* the construction program: one Perception contact and the Solver node's ``decoding`` column group (s1..s4, the twin aa, lock last): ``phase-native-module-v1-c7``, 7
  contacts per row (1 Perception + 6 Solver); the alternative with the Perception-level ``sampling`` column is ``c10`` (:data:`CONSTRUCTION_GROUPS`);
* the Fold B program: one Perception-level ``sampling`` column ``t-v2`` (a Solver vote of n = 3), the twin aa, the registered replicate ``lock-rep2``, lock last:
  K = 4, n = 3, ``phase-native-module-v1-c10`` (4 Perception + 6 Solver);
* the caps, by the port formulas: generation = rows x (contacts + fallback contacts, one per budgeted-thinking contact), FIT = the raw-arm rule of the native contact
  fit gate (rows x contacts x 1 counting endpoint), audio = ceil(roster speech-seconds x (contacts + fallback)).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import l338_family as fam  # noqa: E402
import l338_port_family as pf  # noqa: E402
import tfrl_v2_common as cm  # noqa: E402
import tfrl_v2_sampler as smp  # noqa: E402
from speech_aware_evidence_acquisition.core.preparation_fit import native_contact_required_cap  # noqa: E402

REPO_ROOT = fam.REPO_ROOT

DATE = "2026-10-30"
PKG_NAME = "slurp-tfrl-v2-fold-b-npu-port-v1"
PACKAGE = f"docs/readiness/{DATE}-{PKG_NAME}"
PREREG = f"{PACKAGE}-preregistration.md"
DECISION_ID = "slurp-tfrl-v2-npu-port-admission-v1"
DECISION_RELPATH = "coordination/slurp-tfrl-v2-npu-port-v1/root-admission-v1.json"
DRAFT_PACKAGE = "docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1"
PLAN = "docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md"
DESIGN_DOC = pf.DESIGN_DOC
HISTORICAL_PACKAGE = pf.PACKAGE
CELL_ID = "SAEA-SLURP-TFRL-V2-FOLD-B-NPU-PORT-V1"
LOOP_ID = cm.LOOP_ID
ITERATION = 1
OWNER_DECISIONS = (pf.OWNER_DECISIONS + "; the owner direction of 2026-10-30 (three-way comparison D / A / T on Fold B with Fold A confirmation; the end goal is a better "
                   "training-free RL algorithm; plan sections 8-11 rulings)")
LEDGER_MARK = ("## Cell T NPU port: the agent system plus TFRL-v2 on the BF16 / vLLM-Ascend Thinking core (iteration 1: construction column R01C halves, "
               "Fold B selection flight)")
PREP_LEDGER_MARK = "**NPU PORT PREPARATIONS, REFLECTOR, CREDIT AND BANK ROWS OF THE T CELL, ITERATION 1 (DRAFT).**"

# ---- the READ ids (plan section 11 addendum 2 ruling 4) --------------------------------------------------------------------------------
READ_RUN_ID = "SAEA-SLURP-TFRL-V2-FOLD-B-READ-NPU-PORT-V1"
HALVES = ("H1", "H2")


def trace_read_id(iteration: int, half: str) -> str:
    """``SAEA_ROLLOUT_TRACE_READ_ID`` of a sampling flight: ``<T READ id>-I<t>-<H1|H2>``; the loop adapters read ``analysis/<this id>/rollout-trace.jsonl``."""

    if half not in HALVES:
        raise ValueError(f"half must be one of {HALVES}")
    return f"{READ_RUN_ID}-I{int(iteration)}-{half}"


# ---- the sampling groups ------------------------------------------------------------------------------------------------------------------
K_TRAJECTORIES = cm.DEFAULT_K
#: construction: the Solver-node ``decoding`` column group (c7 shape) or the Perception-level ``sampling`` column with a Solver draw group (c10 shape).
CONSTRUCTION_GROUPS = ("solver-decoding", "perception-sampling")
CONSTRUCTION_GROUP = "solver-decoding"
#: s1..s4: the Solver ``decoding`` group refuses a seed override and two equal overrides (rails.supply_variants.parse_decoding_variants), so the four trajectories share the
#: lock seed and differ in the nucleus width at one temperature (plan section 4: 0.7). top_k stays at the registered 20.
CONSTRUCTION_DECODING = tuple({"temperature": 0.7, "top_p": p, "top_k": 20} for p in (0.80, 0.90, 0.95, 1.00))
#: the seed-true alternative: a draw group of n = 1 greedy + 4 sampled at lock seed + offset 1..4.
CONSTRUCTION_SAMPLING_DECODING = {"temperature": 0.7, "top_p": 0.95, "top_k": 20}
CONSTRUCTION_SAMPLING_N = K_TRAJECTORIES + 1
SAMPLER_ARMS = tuple(smp.sampler_arms(K_TRAJECTORIES))
TWIN = fam.TWIN
DELIVERING = fam.DELIVERING
REPLICATE = pf.REPLICATE_ARMS[0]
T_ARM = "t-v2"
VOTE_N = 3
VOTE_DECODING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
#: The closest EXISTING commit policy to "scenario_action by majority, entities with support >= 2 of 3": a plurality over (scenario, action) and an entity cluster kept
#: at support >= min_support (rails.commit_policies). ``saea-commit-vote-v1`` itself does not exist (rail pending).
VOTE_COMMIT = {"policy": "saea-commit-plurality-support-v1", "min_support": 2, "similarity_threshold": 0.8}
PLANNED_COMMIT_POLICY = "saea-commit-vote-v1"

# ---- contacts per row -------------------------------------------------------------------------------------------------------------------------
#: (Perception, Solver) contacts per row of each program (checked against the module compiler by the derive):
#: construction decoding group: 1 + (K + twin + lock = 6); Fold B: 4 branches' Perception, 4 branches' Solver + (n - 1) draws.
CONTACTS = {
    "solver-decoding": (1, K_TRAJECTORIES + 2),
    "perception-sampling": (3, CONSTRUCTION_SAMPLING_N + 2),
    "fold-b": (4, 4 + (VOTE_N - 1)),
}
PROGRAM_OF = {
    "solver-decoding": "phase-native-module-v1-c7",
    "perception-sampling": "phase-native-module-v1-c10",
    "fold-b": "phase-native-module-v1-c10",
}
CONSTRUCTION_PROGRAM = PROGRAM_OF["solver-decoding"]
FOLD_B_PROGRAM = PROGRAM_OF["fold-b"]
for _g, (_p, _s) in CONTACTS.items():
    if PROGRAM_OF[_g] != f"phase-native-module-v1-c{_p + _s}":
        raise ValueError(f"program of {_g} does not match its contacts")

# ---- the cells --------------------------------------------------------------------------------------------------------------------------------
ROSTERS = {
    "csmoke": "docs/receipts/rosters/slurp-stage3-roll-aa-r01c-smoke-4-v1.json",
    "ch1": cm.HALF_ROSTER_RELPATH[cm.HALF_H1],
    "ch2": cm.HALF_ROSTER_RELPATH[cm.HALF_H2],
    "fsmoke": "docs/receipts/rosters/slurp-stage3-filler-fold-b-c01-smoke-4-v1.json",
    "fb": "docs/receipts/rosters/slurp-evo-fold-b-v2.json",
}
R01C_DATASET = "slurp-speech-native-stage3-construction-r01c"
FOLD_B_DATASET = "slurp-speech-native-evo-fold-b-1080"
R01C_ROSTER = "docs/receipts/rosters/slurp-stage3-construction-r01c.json"
HALVES_SPLIT_RECEIPT = "docs/receipts/rosters/slurp-stage3-construction-r01c-halves-split-receipt.json"
STEMS = {
    "csmoke": "construction-r01c-smoke-4", "ch1": "construction-r01c-i1-h1", "ch2": "construction-r01c-i1-h2",
    "fsmoke": "fold-b-smoke-4", "fb": "fold-b-i1",
}
ALL_KEYS = ("csmoke", "ch1", "ch2", "fsmoke", "fb")
CONSTRUCTION_KEYS = ("csmoke", "ch1", "ch2")
FOLD_B_KEYS = ("fsmoke", "fb")
ROLES = {
    "csmoke": "four-row form smoke of the iteration-1 construction flight (R01C smoke roster)",
    "ch1": "iteration-1 trajectory-sampling flight on the construction half R01C-H1",
    "ch2": "iteration-1 trajectory-sampling flight on the construction half R01C-H2",
    "fsmoke": "four-row form smoke of the Fold B selection flight",
    "fb": "iteration-1 selection flight on TRAIN Fold B",
}
PLAN_STEMS = {"csmoke": "construction-smoke", "ch1": "construction-i1-h1", "ch2": "construction-i1-h2", "fsmoke": "foldb-smoke", "fb": "foldb-i1"}
SERVICE_SESSION = pf.SERVICE_SESSION


def kind_of(key: str) -> str:
    return "construction" if key in CONSTRUCTION_KEYS else "fold-b"


def _roster_doc(key: str) -> dict:
    return json.loads((REPO_ROOT / ROSTERS[key]).read_text(encoding="utf-8"))


def roster_audio(key: str) -> float:
    return sum(float(item["audio_seconds"]) for item in _roster_doc(key)["items"])


def _spec(key: str) -> dict:
    stem = STEMS[key]
    upper = stem.upper()
    doc = _roster_doc(key)
    construction = key in CONSTRUCTION_KEYS
    run = f"SAEA-SLURP-TFRL-V2-{upper}-NPU-PORT-V1"
    fit_run = f"SAEA-SLURP-TFRL-V2-{upper}-FIT-NPU-PORT-V1"
    attempt = f"attempt-slurp-tfrl-v2-{stem}-generation-npu-port-v1"
    fit_attempt = f"attempt-slurp-tfrl-v2-{stem}-fit-npu-port-v1"
    half = {"ch1": cm.HALF_H1, "ch2": cm.HALF_H2}.get(key)
    if half and (run, attempt) != (smp.run_id_for(half, ITERATION), smp.attempt_id_for(half, ITERATION)):
        # the committed sampler's own ids: the run list, the trace READ id and this cell must agree
        raise ValueError(f"the stem rule does not reproduce the sampler's ids for {half}")
    return {
        "key": key, "kind": kind_of(key), "role": ROLES[key], "rows": len(doc["items"]), "roster": ROSTERS[key], "roster_id": doc["roster_id"],
        "split_identity_hash": doc["split_identity_hash"], "dataset": R01C_DATASET if construction else FOLD_B_DATASET,
        "smoke": key.endswith("smoke"), "stem": stem,
        "cell": f"slurp-native-stage3-tfrl-v2-{stem}-npu-port-v1", "baseline": f"slurp-native-stage3-tfrl-v2-{stem}-baseline-raw-port-v1",
        "run": run, "fit_run": fit_run, "attempt": attempt, "fit_attempt": fit_attempt,
        "model": pf.MODEL_FRAGMENT, "session": SERVICE_SESSION,
        "supply_relpath": f"prepared/{run}/phase-supply-v1.json",
        "trace_read_id": trace_read_id(ITERATION, "H1" if key == "ch1" else "H2") if half else None,
        "half": half, "plan_file_stem": PLAN_STEMS[key],
    }


CELLS = {key: _spec(key) for key in ALL_KEYS}
if (CELLS["ch1"]["rows"], CELLS["ch2"]["rows"], CELLS["csmoke"]["rows"], CELLS["fsmoke"]["rows"], CELLS["fb"]["rows"]) != (313, 312, 4, 4, 1080):
    raise ValueError("a T roster's row count moved")


def select_model_fragment() -> None:
    """Make every cell's ``model`` follow the active ``l338_port_family`` thinking-budget switch (call after ``pf.select_model_fragment``)."""

    for spec in CELLS.values():
        spec["model"] = pf.MODEL_FRAGMENT


def plan_file_name(key: str, kind: str) -> str:
    return f"{CELLS[key]['plan_file_stem']}-{kind}-plan-registered-v1.json"


def eligible_run_ids() -> list[str]:
    out: list[str] = []
    for key in ALL_KEYS:
        out += [CELLS[key]["run"], CELLS[key]["fit_run"]]
    return out


def eligible_attempt_ids() -> list[str]:
    out: list[str] = []
    for key in ALL_KEYS:
        out += [CELLS[key]["attempt"], CELLS[key]["fit_attempt"]]
    return out


# ---- the arithmetic -----------------------------------------------------------------------------------------------------------------------------


def group_of(key: str) -> str:
    return CONSTRUCTION_GROUP if kind_of(key) == "construction" else "fold-b"


def contacts(key: str, group: str | None = None) -> tuple[int, int]:
    """(Perception, Solver) contacts per row of the cell's program."""

    return CONTACTS[group or group_of(key)]


def contacts_per_row(key: str, group: str | None = None) -> int:
    return sum(contacts(key, group))


def program_id(key: str, group: str | None = None) -> str:
    return PROGRAM_OF[group or group_of(key)]


def generation_calls(key: str, group: str | None = None) -> int:
    """rows x (program contacts + fallback contacts); the fallback is one answer-only re-ask per budgeted-thinking contact, i.e. as many as there are contacts."""

    c = contacts_per_row(key, group)
    return CELLS[key]["rows"] * (c + c)


def program_calls(key: str, group: str | None = None) -> int:
    return CELLS[key]["rows"] * contacts_per_row(key, group)


def fit_calls(key: str, group: str | None = None) -> int:
    """rows x contacts x one resident counting endpoint (the raw-audio arm), by the native contact fit gate's own helper."""

    return native_contact_required_cap(CELLS[key]["rows"] * contacts_per_row(key, group), feature_packets=False)


def audio_seconds(key: str, group: str | None = None) -> int:
    c = contacts_per_row(key, group)
    return math.ceil(roster_audio(key) * (c + c) - 1e-9)


def mix_rows(key: str, group: str | None = None) -> float:
    """Lock-row equivalents per row: (Perception contacts x Perception seconds + Solver contacts x Solver seconds) / (one Perception + one Solver), to 0.01."""

    p, s = contacts(key, group)
    return fam.round_hundredth((p * fam.NODE_SECONDS["perception"] + s * fam.NODE_SECONDS["solver"]) / (fam.NODE_SECONDS["perception"] + fam.NODE_SECONDS["solver"]))


def card_hours(rows: int, mix: float, *, lock_rows_per_minute_per_instance: float, tensor_parallel_size: int) -> float:
    if lock_rows_per_minute_per_instance <= 0 or tensor_parallel_size < 1:
        raise ValueError("the rate must be positive and the tensor-parallel size at least 1")
    hours = rows * mix * tensor_parallel_size / lock_rows_per_minute_per_instance / 60.0 * pf.GPU_MARGIN
    return round(math.ceil(round(hours / pf.GPU_ROUND_H, 9)) * pf.GPU_ROUND_H, 1)


def plan_gpu_hours_placeholder(key: str) -> float:
    """The numeric ``planned_gpu_hours`` the plan field requires: the PROVISIONAL formula at the historical rate (T = 1); NOT the NPU bound. Smokes keep 0.5."""

    if CELLS[key]["smoke"]:
        return pf.SMOKE_WALL_CLOCK_HOURS
    return card_hours(CELLS[key]["rows"], mix_rows(key), lock_rows_per_minute_per_instance=fam.ROWS_PER_MINUTE_FULL, tensor_parallel_size=1)


def card_hour_formula(key: str) -> str:
    if CELLS[key]["smoke"]:
        return pf.CARD_HOUR_FORMULA["smoke"]
    return (f"PROVISIONAL NPU card-hours = ceil_0.1(1.3 x {CELLS[key]['rows']:,} rows x {mix_rows(key)} lock-row equivalents x T / r / 60), T = tensor-parallel size "
            "(cards per instance), r = the measured lock-only rows per minute of ONE instance at its chosen sequences per instance; the instance count cancels "
            f"(wall-clock = card-hours / (instances x T)); T and r from {pf.DEPLOYMENT_REF}")


def ledger_gpu_cell(key: str) -> str:
    if CELLS[key]["smoke"]:
        return f"PROVISIONAL: {pf.SMOKE_WALL_CLOCK_HOURS} x instances x T NPU card-hours; T and instances from PORT_DEPLOYMENT_PROFILE (pending)"
    return (f"PROVISIONAL: ceil_0.1(1.3 x {CELLS[key]['rows']} x {mix_rows(key)} x T / r / 60) NPU card-hours; T (cards per instance) and r (measured lock rows per minute "
            "per instance) from PORT_DEPLOYMENT_PROFILE (pending)")


# structural checks of the rows x contacts form against the fit gate's own helper, independent of the audio sum
for _key in ALL_KEYS:
    if fit_calls(_key) != CELLS[_key]["rows"] * contacts_per_row(_key) * pf.RAW_FIT_ENDPOINTS_PER_CONTACT:
        raise ValueError(f"the FIT cap of {_key} is not rows x contacts x endpoints")
    if generation_calls(_key) != 2 * program_calls(_key):
        raise ValueError(f"the generation cap of {_key} is not twice the program calls")

# ---- the supply bound and the context fit --------------------------------------------------------------------------------------------------------
EXEMPLAR_BLOCK_TOKEN_CAP = 2000
RULE_BANK_TOKEN_CAP = 400
#: Plan section 9 ruling 4: exemplars REPLACE supplied items; the block of at most 2,000 tokens is about 8 items' worth (235 tokens each), so T's bound is A's minus 8.
EXEMPLAR_ITEM_EQUIVALENT = 8


def a_bound() -> int:
    return pf.budget_view()["max_supplied_items"]


def t_bound() -> int:
    return a_bound() - EXEMPLAR_ITEM_EQUIVALENT


def fold_b_fit_base_tokens() -> int:
    """The sampling frame's ``base_prompt_tokens`` for the Fold B flight: the lock's 2,000 + 323 template tokens plus the two caps (exemplar block 2,000, rule bank 400)."""

    return fam.FIT_ESTIMATE["base_prompt_tokens"] + fam.DELIVERING_TEMPLATE_TOKENS + EXEMPLAR_BLOCK_TOKEN_CAP + RULE_BANK_TOKEN_CAP


def largest_fitting_bound() -> int:
    """The largest T supply bound whose estimate (base incl. both caps + items x 235) stays inside the active Solver prompt limit."""

    return (pf.budget_view()["prompt_limit_tokens"] - fold_b_fit_base_tokens()) // fam.FIT_ESTIMATE["tokens_per_supplied_item"]


def fold_b_fit_estimate_tokens(items: int | None = None) -> int:
    return fold_b_fit_base_tokens() + (t_bound() if items is None else items) * fam.FIT_ESTIMATE["tokens_per_supplied_item"]



#: Declared arm orders (candidates, twin, replicate, delivering arm LAST) of the two programs.
CONSTRUCTION_ARM_ORDER = (*SAMPLER_ARMS, TWIN, DELIVERING)
FOLD_B_ARM_ORDER = (T_ARM, TWIN, REPLICATE, DELIVERING)


def require_fit() -> None:
    """Refuse, clearly, when the T supply bound (A's bound minus 8) plus both caps does not fit the active Solver prompt limit (the 6,144 switch: 43 items do not, 40 do)."""

    view = pf.budget_view()
    estimate = fold_b_fit_estimate_tokens()
    if estimate > view["prompt_limit_tokens"]:
        raise SystemExit(
            f"the T Fold B frame does not fit the {view['switch']!r} budget: bound {t_bound()} items (A's {a_bound()} minus {EXEMPLAR_ITEM_EQUIVALENT}) x {fam.FIT_ESTIMATE['tokens_per_supplied_item']} + base "
            f"{fold_b_fit_base_tokens()} (2,000 + {fam.DELIVERING_TEMPLATE_TOKENS} template + {EXEMPLAR_BLOCK_TOKEN_CAP} exemplar cap + {RULE_BANK_TOKEN_CAP} rule cap) = {estimate} tokens, over the Solver "
            f"prompt limit {view['prompt_limit_tokens']}; the largest fitting T bound is {largest_fitting_bound()} items. Choose the bound (an owner decision) before this switch is used for T.")

def fit_table() -> dict:
    """The T Fold B context fit under BOTH thinking-budget choices, independent of the active switch: the A bound, T's bound, the estimate, the Solver prompt limit, whether it
    fits and the largest fitting T bound (the default fits; the 6,144 / 7,168 switch does not)."""

    base = fold_b_fit_base_tokens()
    item = fam.FIT_ESTIMATE["tokens_per_supplied_item"]
    out = {}
    for name, limit, a in (("default", fam.PROMPT_LIMIT, fam.MAX_SUPPLIED_ITEMS),
                           ("budget6144-max7168", pf.BUDGET6144_EXPECTED["prompt_limit_tokens"], pf.BUDGET6144_EXPECTED["max_supplied_items"])):
        t = a - EXEMPLAR_ITEM_EQUIVALENT
        estimate = base + t * item
        out[name] = {"a_bound": a, "t_bound": t, "base_prompt_tokens": base, "estimate_tokens": estimate, "solver_prompt_limit_tokens": limit, "fits": estimate <= limit,
                     "headroom_tokens": limit - estimate, "largest_fitting_bound": (limit - base) // item}
    return out


def cell_total(index: int, keys=ALL_KEYS) -> int:
    """Family total of one cap over ``keys``: 0 = generation calls, 1 = FIT calls, 2 = speech-seconds."""

    fn = (generation_calls, fit_calls, audio_seconds)[index]
    return sum(fn(k) for k in keys)


T_DECLARED_DIFFERENCES = [
    "T1 (plan section 9 ruling 4): in the T arm exemplars REPLACE supplied items, they do not add to them (supply bound = A's bound minus 8 plus an exemplar block of at most "
    "2,000 tokens; the rule bank is Solver instruction text of at most 400 tokens); A keeps its bound and no bank. A declared T-vs-A protocol difference, not a confound.",
    "T2 (plan section 4 and section 11 ruling 2): iteration-1 trajectories differ in decoding only (no bank exists yet); from iteration 2 the rule-variants mode applies.",
    "T3 (plan section 9 ruling 3 and section 11 ruling 1): the construction column R01C is sampled over two interleaved halves; the bank a half consumes in iteration t+1 is "
    "distilled from the OTHER half's positives; the Fold B flight consumes the union.",
]


def declared_differences() -> list[str]:
    return [*pf.DECLARED_DIFFERENCES, *T_DECLARED_DIFFERENCES]


__all__ = [name for name in globals() if not name.startswith("_")]
