"""Shared identities of the NPU-port twin of the L338 lever-column Fold B flight (design doc docs/superpowers/specs/2026-10-28-npu-vllm-port-design.md).

One module, imported by the port derive, writer and mint scripts, so no port id is spelled twice. Model-free. It reuses the historical family
(``l338_family``: the arm table, the c15 program, the roster, the call / audio / FIT arithmetic) unchanged and adds only what the port changes:

* the ids: every historical id with its trailing ``-V1`` / ``-v1`` replaced by ``-NPU-PORT-V1`` / ``-npu-port-v1`` (the port flight's committed run id
  ``SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-NPU-PORT-V1`` is this rule applied to the historical one);
* the serving model: the BF16 / vLLM-Ascend fragment ``bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1``. The PROTOCOL facts are
  the per-sequence context window (24,576 tokens), the thinking budget 3,072 / cap 4,096 and the decoding (greedy, seed 20260913); the DEPLOYMENT
  (instances, tensor-parallel size, sequences per instance, devices) is NOT a fact of this registration: it is the one named placeholder
  :data:`PORT_DEPLOYMENT_PROFILE`, filled later by a deployment profile chosen from a measured speed sweep (owner, 2026-10-28);
* the accelerator bound: a PROVISIONAL NPU card-hour formula (:func:`provisional_card_hours`), re-derived from the measured sweep before any flip.
  The experiment's numeric ``planned_gpu_hours`` carries the historical ceiling only because the plan field must be a number; the ledger rows carry the
  formula, not a number, so the exposure gate fails closed on them until the sweep fills the profile.

The caps that do not depend on the deployment carry over from the historical registration unchanged: generation 32,400, FIT 64,800, audio 94,644
seconds for the flight; 120 / 240 / 306 for the smoke.
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

REPO_ROOT = fam.REPO_ROOT

DATE = "2026-10-28"
PKG_NAME = "slurp-l338-lever-column-fold-b-npu-port-v1"
PACKAGE = f"docs/readiness/{DATE}-{PKG_NAME}"
PREREG = f"{PACKAGE}-preregistration.md"
DECISION_ID = "slurp-l338-lever-column-fold-b-npu-port-admission-v1"
DECISION_RELPATH = f"coordination/{PKG_NAME}/root-admission-v1.json"
DESIGN_DOC = "docs/superpowers/specs/2026-10-28-npu-vllm-port-design.md"
HISTORICAL_PACKAGE = fam.PACKAGE
HISTORICAL_DECISION_ID = fam.DECISION_ID
PINS_PATH = HERE / "l338_port_pins.json"
PINS = json.loads(PINS_PATH.read_text(encoding="utf-8"))
OWNER_DECISIONS = ("owner decisions of 2026-10-28 on the NPU port (design doc section 9.1: the saea-base Docker container is the execution authority; "
                   "item 4: the agent drafts the exposure preregistration rows; item 5: the admission receipt is authorised and minted by the main thread; "
                   "item 6: model-free-check rows for the port preparations; the multi-instance deployment input: the deployment is a named placeholder "
                   "filled from a measured speed sweep)")
LEDGER_MARK = ("## Cell L338 NPU port: the lever column on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold B (the port twin of the L338 flight; "
               "smoke then 1,080 rows)")
PREP_LEDGER_MARK = "**NPU PORT MODEL-FREE PREPARATIONS OF THE L338 FOLD B CELL, RUN 2026-10-28.**"
TOOL_MINT = "scripts/flights/nx/l338_port_mint_receipt.py"

# ---- the serving model: protocol facts vs the deployment placeholder ----------------------------------------------------------------
MODEL_FRAGMENT = "bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1"
MODEL_LOCK_KEY = "qwen3-omni-30b-a3b-thinking-bf16"
PER_SEQUENCE_WINDOW_TOKENS = 24576                         # protocol fact: --max-model-len, one sequence's context window
PROMPT_ORDER = "native-raw-speech-query-prefix-v1"
SERVICE_SESSION = "serving-slurp-thinking-module-fold-b-npu-port-pending-deployment-profile-v1"
PORT_DEPLOYMENT_PROFILE = {
    "name": "PORT_DEPLOYMENT_PROFILE",
    "status": "PENDING",
    "fields": {"instances": None, "tensor_parallel_size": None, "sequences_per_instance": None, "devices": None},
    "filled_by": ("a later deployment profile chosen from a measured speed sweep of the BF16 / vLLM-Ascend Thinking server (several vLLM instances, each "
                  "tensor-parallel over a group of Ascend cards; per-instance concurrency chosen from the sweep)"),
    "not_a_fact_here": ("No instance count, tensor-parallel size, sequences-per-instance value or device list is a fact of this registration. The model "
                        "fragment carries no --max-num-seqs and no profile_slot_count (moved to the deployment profile, commit db798a83); the baselines' "
                        "serving_slots 4 is the client-side geometry value of the owner ruling, not a deployment fact."),
}
DEPLOYMENT_REF = "PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep)"
#: Every place in this package that depended on the fragment's pre-sweep ``--max-num-seqs 4``, with where it went. The move was made by commit db798a83
#: (owner decision 2026-10-28: sequences per instance is a recorded deployment fact, not protocol) and carried through its dependents by stream S13.
MAX_NUM_SEQS_DEPENDENTS = [
    f"moved: configs/model/{MODEL_FRAGMENT}.json no longer carries server_args --max-num-seqs; the vLLM argv takes it from the selected deployment profile "
    "(core.deployment.instance_server_argv) and a vllm fragment that declares it is REFUSED (core.session.serving, rails.adapters.runtime_view)",
    f"moved: configs/model/{MODEL_FRAGMENT}.json no longer carries profile_slot_count (core.driver.speech_native requires it, with -np / -c, only on its "
    "llama-profile CLI paths -- prompt-audit, profile, finalize-from-disk -- which a vllm fragment fails closed on)",
    f"moved: configs/model/{MODEL_FRAGMENT}.json geometry_basis text now names the 24576-token per-sequence window and puts the sequence count in the deployment profile",
    "unchanged by owner ruling: configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json serving_slots 4 / serving_slot_tokens 24576 "
    "(rails.adapters.runtime_view.configured_geometry now checks only the --max-model-len window of a vllm fragment against serving_slot_tokens)",
    "unchanged by owner ruling: configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-smoke-4-baseline-raw-port-v1.json serving_slots 4 / "
    "serving_slot_tokens 24576 (same)",
    "moved: the prepared first-phase supply identity (core.preparation.artifact_inputs digests the composed config, model fragment included): moving "
    "the fragment makes every phase supply of these cells stale and the re-pin tool mints the next phase-supply-v<N> (output-equality asserted); "
    "prepared/<port run>/phase-supply-v1.json exists for the flight run and is re-pinned by repin_cell.py at the flip, not by the writer",
    "moved: the composed_config_hash / protocol_hash of every plan in this package (the registration writer was re-run after the move)",
    "open: the historical cost inputs (l338_cost_inputs.json, 4 slots) behind the illustrative 1.95 lock rows per minute; the port bound uses the measured r",
]

# ---- ids -------------------------------------------------------------------------------------------------------------------------------


def _port_upper(value: str) -> str:
    if not value.endswith("-V1"):
        raise ValueError(f"historical id {value!r} does not end in -V1")
    return value[: -len("-V1")] + "-NPU-PORT-V1"


def _port_lower(value: str, suffix: str = "-npu-port-v1") -> str:
    if not value.endswith("-v1"):
        raise ValueError(f"historical id {value!r} does not end in -v1")
    return value[: -len("-v1")] + suffix


def _spec(key: str) -> dict:
    hist = fam.CELLS[key]
    run = _port_upper(hist["run"])
    return {
        "key": key, "role": hist["role"], "rows": hist["rows"], "roster": hist["roster"], "scoring_jobs": hist["scoring_jobs"],
        "historical_cell": hist["cell"], "historical_baseline": hist["baseline"], "historical_run": hist["run"], "historical_fit_run": hist["fit_run"],
        "cell": _port_lower(hist["cell"]),
        "baseline": _port_lower(hist["baseline"], "-raw-port-v1"),
        "run": run,
        "fit_run": _port_upper(hist["fit_run"]),
        "attempt": _port_lower(hist["attempt"]),
        "fit_attempt": _port_lower(hist["fit_attempt"]),
        "model": MODEL_FRAGMENT, "session": SERVICE_SESSION,
        "prefix": hist["prefix"],
        "supply_relpath": f"prepared/{run}/phase-supply-v1.json",
        "plan_gpu_hours_placeholder": hist["gpu_hours"],
    }


CELLS = {key: _spec(key) for key in fam.ALL_KEYS}
ALL_KEYS = fam.ALL_KEYS

# ---- the READ: ids, paths and the sign-off conditions of the port (the historical reader binds the historical ids; the port reader binds these) ------
READ_RUN_ID = _port_upper(fam.READ_RUN_ID)
RECIPE_ID = _port_lower(fam.RECIPE_ID)
ANALYSIS_ROOT = f"analysis/{READ_RUN_ID}"
RECIPE_RELPATH = f"{PACKAGE}/analysis-recipe-v1.json"
SIGNOFF_CONDITIONS_RELPATH = f"{PACKAGE}/signoff-conditions-v1.json"
PORT_READER = "scripts/analysis/slurp_l338_port_column_read.py"
SMOKE_TABLE_RELPATH = f"{ANALYSIS_ROOT}/smoke-decode-cost.json"
VERDICTS_NAME = "port-verdicts.json"
if READ_RUN_ID != "SAEA-SLURP-L338-LEVER-COLUMN-FOLD-B-READ-NPU-PORT-V1":
    raise ValueError("the port id rule no longer reproduces the port READ run id")


def signoff_conditions() -> dict:
    """The port's machine sign-off conditions: the historical conditions for the same arm table (``l338_family.signoff_conditions``) with the package name and the
    smoke decode-cost table path moved to the PORT READ run id. The table is written by the port smoke reader (:data:`PORT_READER` ``--stage smoke``)."""

    doc = dict(fam.signoff_conditions(), package=PKG_NAME, table_relpath=SMOKE_TABLE_RELPATH)
    doc["_note"] = ("L338 NPU port: " + fam.signoff_conditions()["_note"] + " Copied from l338_family.signoff_conditions() with the package and the table path "
                    f"({SMOKE_TABLE_RELPATH}) replaced: the smoke decode-cost table is written by the port reader {PORT_READER} --stage smoke, under the PORT READ run id.")
    return doc

if CELLS["arm"]["run"] != "SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-NPU-PORT-V1" or CELLS["arm"]["cell"] != "slurp-native-stage3-l338-lever-column-fold-b-npu-port-v1":
    raise ValueError("the port id rule no longer reproduces the committed port flight's ids")
if CELLS["arm"]["baseline"] != "slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1":
    raise ValueError("the port id rule no longer reproduces the committed port flight baseline")


def eligible_run_ids() -> list[str]:
    """Each cell's generation run id then its FIT run id, in chain order (smoke, then the flight). The port is on its first registration."""

    out: list[str] = []
    for key in ALL_KEYS:
        out += [CELLS[key]["run"], CELLS[key]["fit_run"]]
    return out


def eligible_attempt_ids() -> list[str]:
    out: list[str] = []
    for key in ALL_KEYS:
        out += [CELLS[key]["attempt"], CELLS[key]["fit_attempt"]]
    return out


# ---- budgets: the deployment-independent caps carry over ------------------------------------------------------------------------------


def generation_calls(key: str) -> int:
    return fam.original_generation_calls(key)


def fit_calls(key: str) -> int:
    return fam.fit_contact_cap(CELLS[key]["rows"])


def audio_seconds(key: str) -> int:
    return fam.audio_cap(CELLS[key]["roster"])


EXPECTED_CAPS = {"arm": (32400, 64800, 94644), "smoke": (120, 240, 306)}
for _key, _caps in EXPECTED_CAPS.items():
    if (generation_calls(_key), fit_calls(_key), audio_seconds(_key)) != _caps:
        raise ValueError(f"the carried-over caps of {_key} moved: {(generation_calls(_key), fit_calls(_key), audio_seconds(_key))} != {_caps}")

# ---- the PROVISIONAL accelerator card-hour bound ---------------------------------------------------------------------------------------
GPU_MARGIN, GPU_ROUND_H = fam.GPU_MARGIN, fam.GPU_ROUND_H
ARM_MIX_ROWS = fam.ARM_MIX_ROWS
SMOKE_WALL_CLOCK_HOURS = fam.SMOKE_GPU_HOURS


def provisional_card_hours(rows: int, *, lock_rows_per_minute_per_instance: float, tensor_parallel_size: int) -> float:
    """NPU card-hours = ceil_0.1( 1.3 x rows x arm mix x T / r / 60 ), with T the tensor-parallel size (cards per instance) and r the MEASURED lock-row
    rate of ONE instance at its chosen sequences per instance (lock-only rows per minute, the unit of the historical 1.95). The instance count cancels:
    wall-clock hours = card-hours / (instances x T). PROVISIONAL: r and T come from :data:`PORT_DEPLOYMENT_PROFILE`."""

    if lock_rows_per_minute_per_instance <= 0 or tensor_parallel_size < 1:
        raise ValueError("the rate must be positive and the tensor-parallel size at least 1")
    hours = rows * ARM_MIX_ROWS * tensor_parallel_size / lock_rows_per_minute_per_instance / 60.0 * GPU_MARGIN
    return round(math.ceil(round(hours / GPU_ROUND_H, 9)) * GPU_ROUND_H, 1)


def provisional_smoke_card_hours(*, instances: int, tensor_parallel_size: int) -> float:
    """The smoke keeps the 0.5 h wall-clock allowance over every held card: 0.5 x instances x T."""

    return round(SMOKE_WALL_CLOCK_HOURS * instances * tensor_parallel_size, 1)


CARD_HOUR_FORMULA = {
    "arm": (f"PROVISIONAL NPU card-hours = ceil_0.1(1.3 x {fam.ROWS:,} rows x {ARM_MIX_ROWS} lock-row equivalents x T / r / 60), T = tensor-parallel size "
            "(cards per instance), r = the measured lock-only rows per minute of ONE instance at its chosen sequences per instance; the instance count "
            f"cancels (wall-clock = card-hours / (instances x T)); T and r from {DEPLOYMENT_REF}"),
    "smoke": (f"PROVISIONAL NPU card-hours = {SMOKE_WALL_CLOCK_HOURS} h wall-clock allowance x instances x T (every held card), from {DEPLOYMENT_REF}"),
}
#: Illustration only, NOT a registered figure: at the historical lock rate (1.95 rows per minute per server) the formula reads 90.3 x T card-hours.
ILLUSTRATION_AT_HISTORICAL_RATE = {"lock_rows_per_minute_per_instance": fam.ROWS_PER_MINUTE_FULL,
                                   "card_hours_per_unit_tensor_parallel": provisional_card_hours(fam.ROWS, lock_rows_per_minute_per_instance=fam.ROWS_PER_MINUTE_FULL,
                                                                                                 tensor_parallel_size=1)}


def ledger_gpu_cell(key: str) -> str:
    """The ledger 'gpu hours' cell: the provisional formula, deliberately NOT a number, so the exposure gate fails closed ('must be a number')."""

    if key == "smoke":
        return f"PROVISIONAL: {SMOKE_WALL_CLOCK_HOURS} x instances x T NPU card-hours; T and instances from PORT_DEPLOYMENT_PROFILE (pending)"
    return (f"PROVISIONAL: ceil_0.1(1.3 x {fam.ROWS} x {ARM_MIX_ROWS} x T / r / 60) NPU card-hours; T (cards per instance) and r (measured lock rows per minute "
            "per instance) from PORT_DEPLOYMENT_PROFILE (pending)")


def model_text() -> str:
    """The ledger model/tool text of a port cell: the per-sequence window and a pointer to the pending deployment profile, never a slot geometry."""

    return (f"Frozen Qwen3-Omni-30B-A3B-Thinking BF16 safetensors ({MODEL_LOCK_KEY}) on vLLM-Ascend, BF16 KV, per-sequence window "
            f"{PER_SEQUENCE_WINDOW_TOKENS:,} tokens, max_tokens {fam.COMPLETION_CAP} (reasoning budget {fam.DELIVERING_BUDGET} + answer reserve 1024 on both nodes), "
            f"--reasoning-parser qwen3 with the registered forced-close reasoning_end_str (model fragment {MODEL_FRAGMENT}); deployment "
            f"(instances, tensor-parallel size, sequences per instance, devices) = {DEPLOYMENT_REF}; the server's own audio tower over the original "
            f"recordings under {PROMPT_ORDER}; module program {fam.PROGRAM}; per-arm sampling axes (perception_samples / solver_samples at temperature 0.6, "
            "top_p 0.95, top_k 20); no tool authority")


DECLARED_DIFFERENCES = [
    "8.1 BF16 weights instead of GGUF Q4_K_M; BF16 KV instead of q8_0",
    "8.2 audio features computed by the server's audio tower from the raw audio instead of llama.cpp mmproj packets",
    "8.3 vLLM sampling and kernels: temperature 0 with a fixed seed is deterministic per server build, not bit-identical to llama.cpp",
    "8.4 causal retrieval encoder (forced and measured); every sealed retrieval vector regenerated",
    "8.5 forced-close whitespace (vLLM reasoning_end_str) until verified",
    "8.6 re-authored semantic knowledge (intent-boundaries-npu-port-v1, slot-contracts-npu-port-v2; decision 9.5)",
    "8.7 CPU float32 retrieval encoder instead of CUDA bfloat16",
]
DIFFERENCES_TEXT = "declared differences (design doc section 8): " + "; ".join(DECLARED_DIFFERENCES)

__all__ = [name for name in globals() if not name.startswith("_")]
