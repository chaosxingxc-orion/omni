"""Shared identities of the NPU-port twin of the 266-E direct-only cell: arm D of the three-way comparison (plan 2026-10-30 sections 5 and 9).

One module, imported by the port derive, writer and tests, so no D port id is spelled twice. Model-free. It reuses the historical 266-E family
(``direct_fb_family``: roster, dataset fragment, program, scorer, bootstrap constants) and the L338 port layer (``l338_port_family``: the BF16 / vLLM-Ascend
model fragment, the serving facts, the deployment placeholder, the declared differences, the PROVISIONAL card-hour form) and adds only what D changes:

* the ids: the D flight is ``SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1`` (owner item O8: the brief's id, which drops ``THINKING`` because there is one core
  on the port) and its cells are a four-row smoke plus three request-identical 1,080-row passes (``lock``, ``aa``, ``ab``: noise model M1, k = 2 replicates);
* the FIT identity: each pass has its OWN live-phase-fit run and attempt (a FIT run is a one-time spend identity), and the cap is the one
  ``core.preparation_fit.build_direct_only_raw_fit_gate`` demands: rows x one contact x one counting endpoint (``direct_only_fit_contact_cap``);
* the caps: generation calls = rows x (1 contact + ``FALLBACK_CONTACTS_PER_ROW_DIRECT`` answer-only fallback), speech seconds = ceil(roster audio x the same
  contacts), the card-hour bound the PROVISIONAL formula with the one-Solver-contact arm mix.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NX_DIR = HERE.parent / "nx"
TFRL_DIR = HERE.parent / "tfrl"
for _path in (str(HERE), str(NX_DIR), str(TFRL_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import direct_fb_family as dfam  # noqa: E402  (the historical 266-E family: rosters, program, scorer; NOT modified)
import l338_port_family as pf  # noqa: E402  (the port layer)

REPO_ROOT = dfam.REPO_ROOT
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from speech_aware_evidence_acquisition.core.preparation_fit import (  # noqa: E402
    DIRECT_ONLY_ENDPOINTS_PER_CONTACT,
    DIRECT_ONLY_PROGRAM_ID,
    RAW_NATIVE_ARM,
    direct_only_fit_contact_cap,
)
from speech_aware_evidence_acquisition.core.reasoning_knobs import FALLBACK_CONTACTS_PER_ROW_DIRECT  # noqa: E402

DATE = "2026-10-30"
SLUG = "direct-only-fold-b-npu-port"
PKG_NAME = f"slurp-{SLUG}-v1"
PACKAGE = f"docs/readiness/{DATE}-{PKG_NAME}"
PREREG = f"{PACKAGE}-preregistration.md"
DECISION_ID = f"slurp-{SLUG}-admission-v1"
DECISION_RELPATH = f"coordination/{PKG_NAME}/root-admission-v1.json"
PLAN = "docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md"
DRAFT = "docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/direct-only-cell-draft-v1.md"
HISTORICAL_PACKAGE = dfam.PACKAGE
HISTORICAL_DECISION_ID = dfam.DECISION_ID
TOOL_MINT = "scripts/flights/nx/l338_port_mint_receipt.py"
LEDGER_MARK = ("## Cell D NPU port: direct-only on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold B (the port twin of the 266-E Thinking arm; "
               "smoke, then three request-identical 1,080-row passes)")
READ_RUN_ID = "SAEA-SLURP-DIRECT-ONLY-FOLD-B-READ-NPU-PORT-V1"

# ---- the program and the arm ------------------------------------------------------------------------------------------------------------
PROGRAM = DIRECT_ONLY_PROGRAM_ID
if (PROGRAM, dfam.POLICY, RAW_NATIVE_ARM) != ("phase-direct-only-v1", "phase-direct-only-v1", "raw-speech-agent-native-tools"):
    raise ValueError("the direct-only program or the raw arm moved")
ARM = RAW_NATIVE_ARM
CONTACTS_PER_ROW = 1                               # phase-direct-only-v1: one phase, max_contacts 1
FALLBACK_CONTACTS_PER_ROW = FALLBACK_CONTACTS_PER_ROW_DIRECT
if FALLBACK_CONTACTS_PER_ROW != 1 or DIRECT_ONLY_ENDPOINTS_PER_CONTACT != 1:
    raise ValueError("the direct fallback or the raw counting endpoint factor moved (the caps below are stated against 1 and 1)")
GENERATION_CONTACTS_PER_ROW = CONTACTS_PER_ROW + FALLBACK_CONTACTS_PER_ROW
MODEL_FRAGMENT = pf.MODEL_FRAGMENT
SERVICE_SESSION = pf.SERVICE_SESSION
DATASET = dfam.DATASET
ROWS, SMOKE_ROWS = dfam.ROWS, dfam.SMOKE_ROWS
ROSTER = dict(dfam.ROSTER)                          # the smoke roster receipt stays inside the 266-E package: a prefix of the parent roster, not a new draw

# ---- the four cells ---------------------------------------------------------------------------------------------------------------------
#: chain order: smoke, lock, aa, ab. ``role`` / ``roster`` / ``stem`` build the ids; ``gpu_hours`` is the numeric plan-field placeholder (the 266-E Thinking figure).
_SPECS = (
    ("smoke", "smoke-4", "four-row decode-cost smoke of arm D", SMOKE_ROWS, "smoke", 0.5, 0),
    ("lock", "", "arm D, pass lock (1 of 3 request-identical passes)", ROWS, "arm", dfam.THINK_ARM_GPU_HOURS, 1),
    ("aa", "aa", "arm D, request-identical twin pass aa (2 of 3)", ROWS, "arm", dfam.THINK_ARM_GPU_HOURS, 1),
    ("ab", "ab", "arm D, second request-identical replicate pass ab (3 of 3)", ROWS, "arm", dfam.THINK_ARM_GPU_HOURS, 1),
)
ALL_KEYS = tuple(spec[0] for spec in _SPECS)
PASS_KEYS = ("lock", "aa", "ab")
REPLICATE_KEYS = ("aa", "ab")
DELIVERING = "lock"


def _spec(key: str, stem: str, role: str, rows: int, roster: str, gpu: float, scoring: int) -> dict:
    mid = f"-{stem}" if stem else ""
    upper = f"-{stem.upper()}" if stem else ""
    return {
        "key": key, "role": role, "rows": rows, "roster": roster, "gpu_hours": gpu, "scoring_jobs": scoring,
        "cell": f"slurp-{SLUG.replace('-npu-port', '')}{mid}-npu-port-v1",
        "baseline": f"slurp-{SLUG.replace('-npu-port', '')}{mid}-baseline-raw-port-v1",
        "run": f"SAEA-SLURP-DIRECT-ONLY-FOLD-B{upper}-NPU-PORT-V1",
        "fit_run": f"SAEA-SLURP-DIRECT-ONLY-FOLD-B{upper}-FIT-NPU-PORT-V1",
        "attempt": f"attempt-slurp-direct-only-fold-b{mid}-generation-npu-port-v1",
        "fit_attempt": f"attempt-slurp-direct-only-fold-b{mid}-fit-npu-port-v1",
        "model": MODEL_FRAGMENT, "session": SERVICE_SESSION,
        "historical_cell": f"slurp-direct-only-fold-b-thinking{'-smoke-4' if key == 'smoke' else ''}-v1",
        "historical_baseline": f"slurp-direct-only-fold-b-thinking{'-smoke-4' if key == 'smoke' else ''}-baseline-v1",
        "historical_run": dfam.CELLS["smoke" if key == "smoke" else "thinking"]["run"],
        "historical_fit_run": dfam.CELLS["smoke" if key == "smoke" else "thinking"]["fit_run"],
    }


CELLS = {spec[0]: _spec(*spec) for spec in _SPECS}
if CELLS["lock"]["run"] != "SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1" or CELLS["lock"]["cell"] != "slurp-direct-only-fold-b-npu-port-v1":
    raise ValueError("the D id rule no longer reproduces the brief's flight id")
if CELLS["smoke"]["fit_run"] != "SAEA-SLURP-DIRECT-ONLY-FOLD-B-SMOKE-4-FIT-NPU-PORT-V1" or CELLS["ab"]["fit_run"] != "SAEA-SLURP-DIRECT-ONLY-FOLD-B-AB-FIT-NPU-PORT-V1":
    raise ValueError("the D FIT id rule moved")
if CELLS["aa"]["attempt"] != "attempt-slurp-direct-only-fold-b-aa-generation-npu-port-v1" or CELLS["lock"]["fit_attempt"] != "attempt-slurp-direct-only-fold-b-fit-npu-port-v1":
    raise ValueError("the D attempt id rule moved")

#: Package-relative plan file names.
def plan_file_name(key: str, plan_kind: str) -> str:
    return f"{key}-{plan_kind}-plan-registered-v1.json"


def eligible_run_ids() -> list[str]:
    """Each cell's generation run id then its FIT run id, in chain order (smoke, lock, aa, ab)."""

    out: list[str] = []
    for key in ALL_KEYS:
        out += [CELLS[key]["run"], CELLS[key]["fit_run"]]
    return out


def eligible_attempt_ids() -> list[str]:
    out: list[str] = []
    for key in ALL_KEYS:
        out += [CELLS[key]["attempt"], CELLS[key]["fit_attempt"]]
    return out


def roster_doc(kind: str) -> dict:
    return json.loads((REPO_ROOT / ROSTER[kind]).read_text(encoding="utf-8"))


def roster_audio(kind: str) -> float:
    return sum(float(item["audio_seconds"]) for item in roster_doc(kind)["items"])


# ---- caps (deployment-independent) ----------------------------------------------------------------------------------------------------------


def generation_calls(key: str) -> int:
    """rows x (one contact + one answer-only fallback contact)."""

    return CELLS[key]["rows"] * GENERATION_CONTACTS_PER_ROW


def program_calls(key: str) -> int:
    return CELLS[key]["rows"] * CONTACTS_PER_ROW


def fit_calls(key: str) -> int:
    """The ``live_phase_fit.contact_cap`` the direct-only raw gate demands: rows x one contact x one counting endpoint (the gate's own helper)."""

    return direct_only_fit_contact_cap(CELLS[key]["rows"])


def audio_seconds(key: str) -> int:
    """Every contact, a fallback re-ask included, re-sends the row's recording: ceil(roster audio seconds x (contacts + fallback contacts per row))."""

    return math.ceil(roster_audio(CELLS[key]["roster"]) * GENERATION_CONTACTS_PER_ROW - 1e-9)


EXPECTED_CAPS = {"smoke": (8, 4, 20), "lock": (2160, 1080, 6310), "aa": (2160, 1080, 6310), "ab": (2160, 1080, 6310)}
EXPECTED_TOTAL = (6488, 3244, 18950)
for _key, _caps in EXPECTED_CAPS.items():
    if (generation_calls(_key), fit_calls(_key), audio_seconds(_key)) != _caps:
        raise ValueError(f"the D caps of {_key} moved: {(generation_calls(_key), fit_calls(_key), audio_seconds(_key))} != {_caps}")
CELL_TOTAL = tuple(sum(EXPECTED_CAPS[k][i] for k in ALL_KEYS) for i in range(3))
if CELL_TOTAL != EXPECTED_TOTAL:
    raise ValueError(f"the D cell total moved: {CELL_TOTAL}")

# ---- the PROVISIONAL accelerator bound ------------------------------------------------------------------------------------------------------
GPU_MARGIN, GPU_ROUND_H = pf.GPU_MARGIN, pf.GPU_ROUND_H
SMOKE_WALL_CLOCK_HOURS = pf.SMOKE_WALL_CLOCK_HOURS
#: One Solver contact in lock-row equivalents: solver / (perception + solver) node seconds, rounded as the L338 family rounds (L3F round_hundredth).
ARM_MIX_ROWS = pf.fam.round_hundredth(pf.fam.NODE_SECONDS["solver"] / (pf.fam.NODE_SECONDS["perception"] + pf.fam.NODE_SECONDS["solver"]))
if ARM_MIX_ROWS != 0.52:
    raise ValueError(f"the one-Solver-contact arm mix moved: {ARM_MIX_ROWS}")


def provisional_card_hours(rows: int, *, lock_rows_per_minute_per_instance: float, tensor_parallel_size: int) -> float:
    """NPU card-hours of one pass = ceil_0.1( 1.3 x rows x 0.52 x T / r / 60 ); T and r from :data:`pf.PORT_DEPLOYMENT_PROFILE` once it is filled. PROVISIONAL."""

    if lock_rows_per_minute_per_instance <= 0 or tensor_parallel_size < 1:
        raise ValueError("the rate must be positive and the tensor-parallel size at least 1")
    hours = rows * ARM_MIX_ROWS * tensor_parallel_size / lock_rows_per_minute_per_instance / 60.0 * GPU_MARGIN
    return round(math.ceil(round(hours / GPU_ROUND_H, 9)) * GPU_ROUND_H, 1)


CARD_HOUR_FORMULA = {
    "arm": (f"PROVISIONAL NPU card-hours of ONE pass = ceil_0.1(1.3 x {ROWS:,} rows x {ARM_MIX_ROWS} lock-row equivalents x T / r / 60), T = tensor-parallel size "
            "(cards per instance), r = the measured lock-only rows per minute of ONE instance at its chosen sequences per instance; the instance count cancels "
            f"(wall-clock = card-hours / (instances x T)); T and r from {pf.DEPLOYMENT_REF}; three passes in the cell"),
    "smoke": f"PROVISIONAL NPU card-hours = {SMOKE_WALL_CLOCK_HOURS} h wall-clock allowance x instances x T (every held card), from {pf.DEPLOYMENT_REF}",
}
CARD_HOUR_FORMULA["lock"] = CARD_HOUR_FORMULA["aa"] = CARD_HOUR_FORMULA["ab"] = CARD_HOUR_FORMULA["arm"]


def ledger_gpu_cell(key: str) -> str:
    """The ledger 'gpu hours' cell: the provisional formula, deliberately NOT a number, so the exposure gate fails closed ('must be a number')."""

    if key == "smoke":
        return f"PROVISIONAL: {SMOKE_WALL_CLOCK_HOURS} x instances x T NPU card-hours; T and instances from PORT_DEPLOYMENT_PROFILE (pending)"
    return (f"PROVISIONAL: ceil_0.1(1.3 x {ROWS} x {ARM_MIX_ROWS} x T / r / 60) NPU card-hours; T (cards per instance) and r (measured lock rows per minute per "
            "instance) from PORT_DEPLOYMENT_PROFILE (pending)")


# ---- texts ---------------------------------------------------------------------------------------------------------------------------------
#: The declared differences that apply to D (design doc section 8): 8.4, 8.6 and 8.7 concern retrieval and knowledge, which D does not use.
APPLICABLE_DIFFERENCES = [d for d in pf.DECLARED_DIFFERENCES if d.split()[0] in {"8.1", "8.2", "8.3", "8.5"}]
if [d.split()[0] for d in APPLICABLE_DIFFERENCES] != ["8.1", "8.2", "8.3", "8.5"]:
    raise ValueError("the port declared differences moved; D carries 8.1, 8.2, 8.3 and 8.5")
D1_DIFFERENCE = (f"D1 reasoning budget {pf.fam.DELIVERING_BUDGET} with forced close and one answer-only fallback instead of the historical "
                 f"{dfam.THINK_MAX_TOKENS:,}-token uncapped completion (36 of 1,080 266-E rows ran to the cap)")
DECLARED_DIFFERENCES = APPLICABLE_DIFFERENCES + [D1_DIFFERENCE]
DIFFERENCES_TEXT = "declared differences (design doc section 8): " + "; ".join(DECLARED_DIFFERENCES)


def model_text() -> str:
    """The ledger model/tool text of a D cell: the per-sequence window and a pointer to the pending deployment profile; the program is the direct-only one."""

    return (f"Frozen Qwen3-Omni-30B-A3B-Thinking BF16 safetensors ({pf.MODEL_LOCK_KEY}) on vLLM-Ascend, BF16 KV, per-sequence window "
            f"{pf.PER_SEQUENCE_WINDOW_TOKENS:,} tokens, max_tokens {pf.fam.COMPLETION_CAP} (reasoning budget {pf.fam.DELIVERING_BUDGET} + answer reserve 1024), "
            f"--reasoning-parser qwen3 with the registered forced-close reasoning_end_str (model fragment {MODEL_FRAGMENT}); deployment "
            f"(instances, tensor-parallel size, sequences per instance, devices) = {pf.DEPLOYMENT_REF}; the server's own audio tower over the original "
            f"recordings under {pf.PROMPT_ORDER}; direct-only program {PROGRAM} (greedy, seed {pf.fam.LOCKED_DECODING['seed']}; no sampling axis); no tool authority")


def signoff_conditions() -> dict:
    """The machine sign-off conditions of the D smoke (owner ruling 314 A form, ``scripts/flights/batch/smoke_signoff.py``): the one node closes within its cap.

    The smoke table is written by a D smoke reader that is NOT part of this package (open item); it must emit ``arms.direct.solver`` with the fields below
    and ``throughput.arm_mix_measured_rows`` / ``arm_mix_registered_rows`` ({ARM_MIX_ROWS} registered)."""

    return {
        "schema": "saea-smoke-signoff-conditions-v1",
        "_note": ("D NPU port: machine sign-off conditions for the four-row smoke (owner ruling 314 A form). The smoke shows that the one Solver contact per row closes within "
                  "its cap (no cap hit, no empty answer, no malformed JSON, no length finish) and that the cost is no more than 1.3x the registered one-Solver-contact "
                  f"mix ({ARM_MIX_ROWS} lock-row equivalents). The smoke table (schema saea-slurp-direct-port-smoke-v1) is written by a D smoke reader that is an open "
                  "item of this package; the 266-E reader scripts/analysis/slurp_direct_fold_b_core_read.py --stage smoke binds the 266-E ids and is not reused."),
        "package": PKG_NAME,
        "ruling": "owner ruling 314 A of 2026-10-14",
        "table_schema": "saea-slurp-direct-port-smoke-v1",
        "table_relpath": f"analysis/{READ_RUN_ID}/smoke-decode-cost.json",
        "expected_arms": ["direct"],
        "min_contacts_per_node": 1,
        "no_think_nodes": {},
        "no_think_max": {"contacts_with_reasoning_chars": 0, "marker_in_content": 0, "reasoning_chars_mean": 0},
        "every_node_max": {"cap_hits": 0, "empty_content": 0, "finish_reason_length": 0, "no_json_object": 0},
        "arm_mix": {"max_ratio": 1.3, "measured_key": "arm_mix_measured_rows", "registered_key": "arm_mix_registered_rows"},
        "extras": [{"id": "unattributed-contacts", "op": "==", "path": "unattributed_contacts", "value": 0}],
    }


__all__ = [name for name in globals() if not name.startswith("_")]
