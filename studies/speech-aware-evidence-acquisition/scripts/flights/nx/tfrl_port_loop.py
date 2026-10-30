#!/usr/bin/env python3
"""tfrl_port_loop.py -- the port DRIVER of the training-free RL loop (TFRL-v2; plan 2026-10-30 section 4, task T5).

This is the driver layer only. The historical driver ``scripts/flights/tfrl/tfrl_loop_driver.py`` (decision 263-A) is bound to the
Windows-side flow (a Windows shell batch driver, flips on a Windows-mounted checkout, a CPU-quantised inference session, the
historical cell ids and a datasets lock path on a Windows drive). This module keeps the model-free pieces that are transport-agnostic and re-binds the rest to the port: every path is
relative to the data root (``SPEECHRL_DATA_DIR``), every study command runs through the port's transport (``port_py.sh`` in the
``saea-base`` container, started by a human or by a registered flight, never by this module), and nothing here contacts a model,
a card or a server.

ITERATION (at most 3). Each iteration is five resumable steps; every step writes a write-once receipt under
``<data root>/analysis/<LOOP_ID>/i<t>/`` and a re-run with the same inputs is a no-op:

    sample    Sampler.sample(SamplerRequest) on the construction column (K trajectories per row, temperature T, rule bank, exemplar bank)
              -> trace lines, one per trajectory, each with a per-row reward (row F1 of the trajectory minus row F1 of lock)
    distil    Distiller.distil(...) -> the next rule bank, exemplar bank and vote setting (contrast-pair reflector, exemplar bank)
    read      CandidateReader.read(...) -> the credit vectors of the candidate state on TRAIN Fold B (a registered flight flown by a human;
              the driver only reads its result, and halts with status ``awaiting-read`` while the file is absent)
    accept    accept_candidate(...) -> the ruled selection of ``analysis.sealed_credit.selection`` (CI95 lower bound > 0, point >= the
              within-run floor, above the twin, intent guard) plus the plan's extra clauses (twin admissible within 1.2 pp, no negative
              per-row delta); the replicate variance of M1 widens the interval when supplied
    admit     state admission into the write-once archive of ``analysis.sealed_credit.archive`` (hashes and numbers only)

STOP RULE. At most 3 iterations; stop when an iteration accepts nothing, or when the construction-column gain is under +1 pp
(iteration 1 under +1 pp is the kill criterion of the family).

SAMPLER INTERFACE (filled by task T3). ``SamplerRequest`` -> ``Sequence[Mapping]`` of ``saea-rollout-trace-v1`` lines of kind ``tfrl``:

    inputs   iteration, column_rows (row ids of the construction column), k (trajectories per row, default 4), temperature (default 0.7),
             rule_bank (opaque mapping: {"rules": [{"rule_id", "kind", "text"}]}), exemplar_bank (opaque mapping keyed by scenario),
             lock_row_f1 (row id -> the lock column's row F1, from the construction lock column re-flown on the port), seed
    outputs  exactly ``len(column_rows) * k`` lines, each carrying the keys of ``TFRL_LINE_REQUIRED``; ``reward == row_f1 - lock_row_f1``

TODO-T1: the line shape is the minimal dict of ``TFRL_LINE_REQUIRED`` until task T1 lands ``contracts/rollout_trace.py``; when that
module exists, ``validate_tfrl_line`` delegates to its validator.

No utterance text, gold label or model output is written by this module; the archive is identifiers, hashes and numbers only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts" / "flights" / "tfrl")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import tfrl_family as tf  # noqa: E402  (shared hashing helpers; model-free)
import tfrl_loop_lib as lib  # noqa: E402  (write-once receipts, archive helpers; model-free)

# ---------------------------------------------------------------------------------------------------------------------------
# identities (port family: every historical id with its trailing -V1 / -v1 replaced by -NPU-PORT-V1 / -npu-port-v1)
# ---------------------------------------------------------------------------------------------------------------------------

LOOP_ID = "SAEA-SLURP-TFRL-V2-NPU-PORT-V1"
ANALYSIS_ROOT = f"analysis/{LOOP_ID}"             # the only place under the data root this loop writes
ALGORITHM = "tfrl-v2-trajectory-credit-v1"
MAX_ITERATIONS = 3
DEFAULT_K = 4
DEFAULT_TEMPERATURE = 0.7
STOP_GAIN_PP = 1.0                                  # construction-column gain under this stops the loop (iteration 1: kills the family)
TWIN_ADMISSIBLE_PP = 1.2                            # |twin - lock| bound (the port's ADMISSIBLE rule)
TRANSPORT = "port: saea-base container via work/scripts/port_py.sh against the data root; vLLM-Ascend router for any model contact"

#: The construction column: the plain lock program on TRAIN roster R01C. The historical identity is quoted for the registration delta;
#: the port cell is the same name with the port suffix (the rule of ``l338_port_family``).
CONSTRUCTION = {
    "roster_id": "SLURP-STAGE3-CONSTRUCTION-R01C",
    "roster_relpath": "docs/receipts/rosters/slurp-stage3-construction-r01c.json",
    "dataset_fragment": "slurp-speech-native-stage3-construction-r01c",
    "split_identity_hash_prefix": "f56bc386",
    "rows": 625,
    "program": "phase-native-module-v1-c2",
    "contacts_per_row_lock": 2,
    "historical_cell": "slurp-native-stage3-tfrl-construction-lock-r01c-v1",
    "port_cell": "slurp-native-stage3-tfrl-construction-lock-r01c-npu-port-v1",
    "historical_run_id": "SAEA-SLURP-STAGE3-TFRL-CONSTRUCTION-LOCK-R01C-V1",
    "port_run_id": "SAEA-SLURP-STAGE3-TFRL-CONSTRUCTION-LOCK-R01C-NPU-PORT-V1",
    "port_attempt_id": "attempt-slurp-native-stage3-tfrl-construction-lock-r01c-generation-npu-port-v1",
    "must_reflight_on_port": True,
}

# The minimal `tfrl` line of `saea-rollout-trace-v1` (TODO-T1: replaced by the T1 contract when it lands).
TRACE_SCHEMA = "saea-rollout-trace-v1"
TRACE_KIND = "tfrl"
TFRL_LINE_REQUIRED = (
    "schema", "kind", "iteration", "row_id", "trajectory_id", "k_index", "temperature", "rule_ids", "exemplar_ids",
    "row_f1", "lock_row_f1", "reward", "truncated",
)
TFRL_LINE_OPTIONAL = ("selected", "evidence_order", "supply_item_ids", "thinking_tokens", "answer_sha256", "node", "arm")
_TRACE_MODULE = "speech_aware_evidence_acquisition.contracts.rollout_trace"
_TRACE_VALIDATOR_NAMES = ("validate_tfrl_line", "validate_line", "validate_trace_line", "validate")

# ARTEFACT slots of the archive used by a TFRL-v2 state (the archive's slot names are fixed; the algorithm label lives in the receipts).
SLOT_EXEMPLAR_BANK = "exemplar_index_release"
SLOT_RULE_BANK = "playbook_release"
SLOT_VOTE = "supply_fragment"


class PortLoopError(RuntimeError):
    pass


class SamplerNotImplemented(PortLoopError):
    """The sampler call site is a stub until task T3 fills it."""


# ---------------------------------------------------------------------------------------------------------------------------
# trace line validation
# ---------------------------------------------------------------------------------------------------------------------------


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _local_validate_tfrl_line(line: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if not isinstance(line, Mapping):
        return ["line is not an object"]
    missing = [k for k in TFRL_LINE_REQUIRED if k not in line]
    if missing:
        problems.append(f"missing keys {missing}")
        return problems
    unknown = sorted(set(line) - set(TFRL_LINE_REQUIRED) - set(TFRL_LINE_OPTIONAL))
    if unknown:
        problems.append(f"unknown keys {unknown}")
    if line["schema"] != TRACE_SCHEMA or line["kind"] != TRACE_KIND:
        problems.append(f"schema/kind must be {TRACE_SCHEMA}/{TRACE_KIND}")
    for key in ("iteration", "k_index"):
        if not isinstance(line[key], int) or isinstance(line[key], bool) or line[key] < 0:
            problems.append(f"{key} must be a non-negative integer")
    for key in ("row_id", "trajectory_id"):
        if not isinstance(line[key], str) or not line[key]:
            problems.append(f"{key} must be a non-empty string")
    for key in ("temperature", "row_f1", "lock_row_f1", "reward"):
        if not _finite(line[key]):
            problems.append(f"{key} must be a finite number")
    for key in ("rule_ids", "exemplar_ids"):
        if not isinstance(line[key], list) or not all(isinstance(x, str) for x in line[key]):
            problems.append(f"{key} must be a list of strings")
    if not isinstance(line["truncated"], bool):
        problems.append("truncated must be a boolean")
    if not problems and abs(line["reward"] - (line["row_f1"] - line["lock_row_f1"])) > 1e-9:
        problems.append("reward must equal row_f1 - lock_row_f1")
    return problems


def validate_tfrl_line(line: Mapping[str, Any]) -> list[str]:
    """Problems of one `tfrl` trace line (empty list = valid). Delegates to the T1 contract when ``contracts/rollout_trace.py`` exists."""

    local = _local_validate_tfrl_line(line)       # the port's own reward arithmetic is checked in every case
    try:
        module = importlib.import_module(_TRACE_MODULE)
    except ImportError:
        return local                              # TODO-T1: the contract module is not in the tree yet
    for name in _TRACE_VALIDATOR_NAMES:
        fn = getattr(module, name, None)
        if callable(fn):
            result = fn(line)
            extra = [] if result in (None, True) else (list(result) if isinstance(result, (list, tuple)) else [str(result)])
            return local + extra
    return local


# ---------------------------------------------------------------------------------------------------------------------------
# interfaces
# ---------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplerRequest:
    iteration: int
    column_rows: Sequence[str]
    k: int
    temperature: float
    rule_bank: Mapping[str, Any]
    exemplar_bank: Mapping[str, Any]
    lock_row_f1: Mapping[str, float]
    seed: int = 20261030


class Sampler(Protocol):
    def sample(self, request: SamplerRequest) -> Sequence[Mapping[str, Any]]: ...


class StubSampler:
    """The call site task T3 fills (trajectory sampling through the router transport). Fails closed."""

    def sample(self, request: SamplerRequest) -> Sequence[Mapping[str, Any]]:
        raise SamplerNotImplemented("TODO-T3: the TFRL-v2 trajectory sampler is not implemented; no model call is made here")


@dataclass(frozen=True)
class DistilResult:
    rule_bank: Mapping[str, Any]
    exemplar_bank: Mapping[str, Any]
    vote_n: int = 3
    contrast_pairs: int = 0
    refused_rules: int = 0
    construction_gain_pp: float | None = None


class Distiller(Protocol):
    def distil(self, iteration: int, trajectories: Sequence[Mapping[str, Any]], rule_bank: Mapping[str, Any],
               exemplar_bank: Mapping[str, Any]) -> DistilResult: ...


class StubDistiller:
    def distil(self, iteration, trajectories, rule_bank, exemplar_bank) -> DistilResult:     # noqa: D401
        raise SamplerNotImplemented("TODO-T3: exemplar distillation and the contrast-pair reflector are not implemented")


@dataclass(frozen=True)
class CandidateRead:
    """The Fold B read of the candidate state (credit layout of ``archive.validate_credit``; paired deltas in pp)."""

    run_id: str
    attempt: int
    rows: int
    credit: Mapping[str, Any]
    twin_credit: Mapping[str, Any]
    parent_credit: Mapping[str, Any] | None = None
    replicate_sd_pp: float | None = None            # M1: sd of the lock decoding replicates, pp (None = not supplied)
    per_row_delta_pp: float | None = None           # M1: mean per-row F1 delta, pp (None = not supplied)


class CandidateReader(Protocol):
    def read(self, iteration: int, state_digests: Mapping[str, str]) -> CandidateRead | None: ...


class FileCandidateReader:
    """Reads ``<analysis root>/i<t>/fold-b-read.json`` written after the registered Fold B flight; ``None`` while it is absent."""

    def __init__(self, analysis_dir: Path):
        self.analysis_dir = Path(analysis_dir)

    def read(self, iteration: int, state_digests: Mapping[str, str]) -> CandidateRead | None:
        path = self.analysis_dir / f"i{iteration}" / "fold-b-read.json"
        if not path.is_file():
            return None
        doc = json.loads(path.read_text(encoding="utf-8"))
        return CandidateRead(run_id=doc["run_id"], attempt=int(doc["attempt"]), rows=int(doc["rows"]), credit=doc["credit"],
                             twin_credit=doc["twin_credit"], parent_credit=doc.get("parent_credit"),
                             replicate_sd_pp=doc.get("replicate_sd_pp"), per_row_delta_pp=doc.get("per_row_delta_pp"))


# ---------------------------------------------------------------------------------------------------------------------------
# credit summary (model-free arithmetic over the sampler's lines)
# ---------------------------------------------------------------------------------------------------------------------------


def check_sample(lines: Sequence[Mapping[str, Any]], request: SamplerRequest) -> None:
    """Refuse a sampler result that is not exactly K valid lines per column row with consistent rewards."""

    expected = len(request.column_rows) * request.k
    if len(lines) != expected:
        raise PortLoopError(f"the sampler returned {len(lines)} lines, expected {expected} ({len(request.column_rows)} rows x K={request.k})")
    per_row: dict[str, int] = {}
    seen: set[str] = set()
    for line in lines:
        problems = validate_tfrl_line(line)
        if problems:
            raise PortLoopError(f"invalid tfrl line {line.get('trajectory_id', '?')!r}: {problems}")
        if line["iteration"] != request.iteration:
            raise PortLoopError("a line carries another iteration")
        if line["trajectory_id"] in seen:
            raise PortLoopError(f"duplicate trajectory id {line['trajectory_id']!r}")
        seen.add(line["trajectory_id"])
        per_row[line["row_id"]] = per_row.get(line["row_id"], 0) + 1
        if abs(line["lock_row_f1"] - request.lock_row_f1[line["row_id"]]) > 1e-9:
            raise PortLoopError(f"row {line['row_id']!r}: lock_row_f1 differs from the construction lock column")
    if set(per_row) != set(request.column_rows) or any(n != request.k for n in per_row.values()):
        raise PortLoopError("the sampler did not return exactly K trajectories for every column row")


def credit_summary(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate counts: positives fix a row the lock gets wrong, negatives break a row the lock gets right, contrast rows have both signs."""

    rows: dict[str, list[Mapping[str, Any]]] = {}
    for line in lines:
        rows.setdefault(line["row_id"], []).append(line)
    positives = sum(1 for l in lines if l["lock_row_f1"] < 1.0 and l["reward"] > 0)
    negatives = sum(1 for l in lines if l["lock_row_f1"] >= 1.0 and l["reward"] < 0)
    contrast_rows = sum(1 for ls in rows.values() if any(l["reward"] > 0 for l in ls) and any(l["reward"] < 0 for l in ls))
    n = len(lines)
    return {"trajectories": n, "rows": len(rows), "positives": positives, "negatives": negatives, "contrast_rows": contrast_rows,
            "truncated": sum(1 for l in lines if l["truncated"]),
            "mean_reward_pp": round(100.0 * sum(l["reward"] for l in lines) / n, 6) if n else 0.0}


# ---------------------------------------------------------------------------------------------------------------------------
# acceptance (wraps analysis.sealed_credit.selection.select; transport-agnostic)
# ---------------------------------------------------------------------------------------------------------------------------


def widen_credit(credit: Mapping[str, Any], replicate_sd_pp: float) -> dict[str, Any]:
    """M1: add the decoding-replicate variance to the SLU interval of ``vs_lock`` (quadrature on each half-width).

    TODO-T2: replace by the T2 CI code when it lands; the quadrature rule is this driver's placeholder, not a ruled formula.
    """

    out = json.loads(json.dumps(credit))
    slu = out["vs_lock"]["slu"]
    extra = 1.96 * float(replicate_sd_pp)
    slu["lo"] = slu["point"] - math.hypot(slu["point"] - slu["lo"], extra)
    slu["hi"] = slu["point"] + math.hypot(slu["hi"] - slu["point"], extra)
    return out


def accept_candidate(read: CandidateRead, *, repo_root: Path = REPO_ROOT, rules: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ruled selection plus the plan's extra clauses. Returns ``{"accepted", "status", "failed", "clauses", "rules", "widened"}``."""

    from speech_aware_evidence_acquisition.analysis.sealed_credit import archive as ar  # noqa: PLC0415
    from speech_aware_evidence_acquisition.analysis.sealed_credit import selection as sel  # noqa: PLC0415

    for name, credit in (("credit", read.credit), ("twin_credit", read.twin_credit)):
        try:
            ar.validate_credit(credit)
        except ar.ArchiveError as exc:
            raise PortLoopError(f"{name}: {exc}") from exc
    if read.parent_credit is not None:
        ar.validate_credit(read.parent_credit)
    base_rules = sel.rules_from_floor_table(read.rows, repo_root)
    merged = {**base_rules, **(rules or {})}
    widened = read.replicate_sd_pp is not None
    child = widen_credit(read.credit, read.replicate_sd_pp) if widened else dict(read.credit)
    verdict = sel.select(child, {"role": "lock", "n_rows": read.rows}, read.twin_credit, read.parent_credit, merged)
    clauses = dict(verdict.clauses)
    twin_point = read.twin_credit["vs_lock"]["slu"]["point"]
    clauses["f_twin_admissible"] = {"pass": abs(twin_point) <= TWIN_ADMISSIBLE_PP, "value": twin_point, "threshold": TWIN_ADMISSIBLE_PP,
                                    "op": "abs<="}
    if read.per_row_delta_pp is not None:
        clauses["g_per_row_not_negative"] = {"pass": read.per_row_delta_pp >= 0.0, "value": read.per_row_delta_pp, "threshold": 0.0, "op": ">="}
    failed = [name for name, c in clauses.items() if not c["pass"]]
    accepted = not failed
    return {"accepted": accepted, "status": "accepted" if accepted else "refused", "failed": failed, "clauses": clauses,
            "rules": merged, "widened": widened}


def fold_a_confirms(fold_a_credit: Mapping[str, Any]) -> bool:
    """M2: a lever is confirmed only if the Fold A lower CI95 of the SLU delta is above 0."""

    return bool(fold_a_credit["vs_lock"]["slu"]["lo"] > 0.0)


# ---------------------------------------------------------------------------------------------------------------------------
# state admission (reuses the write-once archive; hashes and numbers only)
# ---------------------------------------------------------------------------------------------------------------------------


def bank_digest(bank: Mapping[str, Any]) -> str:
    return tf.canonical_sha256(dict(bank))


def state_hashes(rule_bank: Mapping[str, Any], exemplar_bank: Mapping[str, Any], vote_n: int) -> dict[str, str]:
    return {SLOT_RULE_BANK: bank_digest(rule_bank), SLOT_EXEMPLAR_BANK: bank_digest(exemplar_bank),
            SLOT_VOTE: tf.canonical_sha256({"vote_n": int(vote_n), "algorithm": ALGORITHM})}


def ensure_root(archive: Any, empty_rule_bank: Mapping[str, Any], empty_exemplar_bank: Mapping[str, Any], vote_n: int) -> str:
    """The iteration-0 root: the port lock with EMPTY banks (zero credit against itself)."""

    from speech_aware_evidence_acquisition.analysis.sealed_credit import archive as ar  # noqa: PLC0415

    root = ar.build_state(artefact_hashes=state_hashes(empty_rule_bank, empty_exemplar_bank, vote_n), parent_id=None, iteration=0,
                          operator_class=None, candidate_id=None, verdict="accepted", arm="typed", credit=lib.zero_credit(),
                          run_id="lock-construction-npu-port", attempt=0)
    lib.put_state_idempotent(archive, root)
    return root["state_id"]


def admit_state(archive: Any, *, iteration: int, parent_id: str, hashes: Mapping[str, str], verdict: str, read: CandidateRead) -> str:
    from speech_aware_evidence_acquisition.analysis.sealed_credit import archive as ar  # noqa: PLC0415

    state = ar.build_state(artefact_hashes=dict(hashes), parent_id=parent_id, iteration=iteration, operator_class="tfrl-v2-distil",
                           candidate_id=hashes[SLOT_RULE_BANK], verdict=verdict, arm="typed", credit=dict(read.credit),
                           run_id=read.run_id, attempt=read.attempt)
    lib.put_state_idempotent(archive, state)
    return state["state_id"]


# ---------------------------------------------------------------------------------------------------------------------------
# the plan and the loop
# ---------------------------------------------------------------------------------------------------------------------------


@dataclass
class LoopConfig:
    data_root: Path
    iterations: int = MAX_ITERATIONS
    k: int = DEFAULT_K
    temperature: float = DEFAULT_TEMPERATURE
    rows: int = CONSTRUCTION["rows"]
    seed: int = 20261030
    repo_root: Path = REPO_ROOT

    def __post_init__(self) -> None:
        if not 1 <= self.iterations <= MAX_ITERATIONS:
            raise PortLoopError(f"iterations must be between 1 and {MAX_ITERATIONS}")
        if self.k < 1:
            raise PortLoopError("K must be at least 1")

    @property
    def analysis_dir(self) -> Path:
        return Path(self.data_root) / ANALYSIS_ROOT


def iteration_plan(iterations: int = MAX_ITERATIONS, k: int = DEFAULT_K, temperature: float = DEFAULT_TEMPERATURE,
                   rows: int = CONSTRUCTION["rows"]) -> list[str]:
    """The text of the dry run: one block per iteration."""

    lines = [
        f"TFRL-v2 port loop {LOOP_ID}  (algorithm {ALGORITHM}, at most {MAX_ITERATIONS} iterations; planned {iterations})",
        f"transport: {TRANSPORT}",
        f"construction column: roster {CONSTRUCTION['roster_id']} ({rows} rows, split hash {CONSTRUCTION['split_identity_hash_prefix']}...), "
        f"dataset fragment {CONSTRUCTION['dataset_fragment']}, program {CONSTRUCTION['program']}",
        f"construction lock column to re-fly on the port: cell {CONSTRUCTION['port_cell']} (run {CONSTRUCTION['port_run_id']}); "
        f"historical cell {CONSTRUCTION['historical_cell']} is NOT reused (the lock changed)",
        f"data root: $SPEECHRL_DATA_DIR, loop outputs under {ANALYSIS_ROOT}/",
    ]
    for t in range(1, iterations + 1):
        lines += [
            f"iteration {t}/{iterations}",
            f"  1 sample   {rows} rows x K={k} at temperature {temperature} = {rows * k} Solver trajectories (sampler: TODO-T3 stub); "
            f"rule bank and exemplar bank from iteration {t - 1}; per-row reward = row F1 - lock row F1",
            f"  2 distil   exemplar bank from positives, rules only from positive/negative contrast pairs, vote n=3 (distiller: TODO-T3 stub)",
            f"  3 read     candidate state on TRAIN Fold B with twin + replicate copies (registered flight, human-launched; result file "
            f"{ANALYSIS_ROOT}/i{t}/fold-b-read.json)",
            f"  4 accept   ruled selection (CI95 lower bound > 0, point >= within-run floor, above twin, intent guard) + twin within "
            f"{TWIN_ADMISSIBLE_PP} pp + no negative per-row delta",
            f"  5 admit    write-once archive state ({SLOT_RULE_BANK}, {SLOT_EXEMPLAR_BANK}, {SLOT_VOTE}); accepted or refused",
            f"  stop rule: stop if nothing is accepted or the construction-column gain is under +{STOP_GAIN_PP} pp"
            + (" (iteration 1: kill the family)" if t == 1 else ""),
        ]
    lines.append(f"total Solver contacts if all {iterations} iterations run: {iterations * rows * k}")
    lines.append("then: Fold A confirmation of each accepted state (lower CI95 > 0), one pass (M2)")
    return lines


def _row_ids_and_lock_f1(path: Path) -> dict[str, float]:
    """The lock column's per-row F1 (``saea-tfrl-row-scores-v1`` layout: ``{"arms": {"lock": {row_id: f1}}}``)."""

    doc = json.loads(path.read_text(encoding="utf-8"))
    lock = doc["arms"]["lock"]
    return {str(k): float(v) for k, v in lock.items()}


@dataclass
class LoopReport:
    status: str = "not-started"
    iterations: list[dict[str, Any]] = field(default_factory=list)
    accepted_states: list[str] = field(default_factory=list)
    halt_reason: str | None = None


def run_loop(cfg: LoopConfig, sampler: Sampler, distiller: Distiller, reader: CandidateReader, *,
             lock_row_f1: Mapping[str, float] | None = None, out: Callable[[str], None] = print) -> LoopReport:
    """Run up to ``cfg.iterations`` iterations; halt at the first missing external input; every step is write-once and resumable."""

    from speech_aware_evidence_acquisition.analysis.sealed_credit.archive import Archive  # noqa: PLC0415

    adir = cfg.analysis_dir
    lock_path = adir / "construction-lock-row-scores.json"
    if lock_row_f1 is None:
        if not lock_path.is_file():
            return LoopReport(status="halted", halt_reason=f"missing {ANALYSIS_ROOT}/construction-lock-row-scores.json "
                                                           f"(the construction lock column must be re-flown on the port and scored first)")
        lock_row_f1 = _row_ids_and_lock_f1(lock_path)
    rows = sorted(lock_row_f1)
    if len(rows) != cfg.rows:
        return LoopReport(status="halted", halt_reason=f"the lock row scores cover {len(rows)} rows, expected {cfg.rows}")

    archive = Archive(adir / "archive")
    rule_bank: Mapping[str, Any] = {"rules": []}
    exemplar_bank: Mapping[str, Any] = {"by_scenario": {}}
    vote_n = 3
    parent_id = ensure_root(archive, rule_bank, exemplar_bank, vote_n)
    report = LoopReport(status="running")

    for t in range(1, cfg.iterations + 1):
        idir = adir / f"i{t}"
        request = SamplerRequest(iteration=t, column_rows=rows, k=cfg.k, temperature=cfg.temperature, rule_bank=rule_bank,
                                 exemplar_bank=exemplar_bank, lock_row_f1=lock_row_f1, seed=cfg.seed)
        inputs = {"iteration": t, "k": cfg.k, "temperature": cfg.temperature, "rows": len(rows), "rule_bank": bank_digest(rule_bank),
                  "exemplar_bank": bank_digest(exemplar_bank), "seed": cfg.seed}

        # 1 sample
        traj_path = idir / "trajectories.jsonl"
        if traj_path.is_file():
            lines = [json.loads(x) for x in traj_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        else:
            try:
                lines = list(sampler.sample(request))
            except SamplerNotImplemented as exc:
                report.status, report.halt_reason = "halted", str(exc)
                return report
            check_sample(lines, request)
            idir.mkdir(parents=True, exist_ok=True)
            traj_path.write_text("".join(json.dumps(l, sort_keys=True) + "\n" for l in lines), encoding="utf-8", newline="\n")
        check_sample(lines, request)
        summary = credit_summary(lines)
        lib.write_receipt(idir, "sample", {"summary": summary}, inputs=inputs)
        out(f"iteration {t}: sampled {summary['trajectories']} trajectories, {summary['positives']} positive, {summary['negatives']} negative, "
            f"{summary['contrast_rows']} contrast rows")

        # 2 distil
        try:
            result = distiller.distil(t, lines, rule_bank, exemplar_bank)
        except SamplerNotImplemented as exc:
            report.status, report.halt_reason = "halted", str(exc)
            return report
        hashes = state_hashes(result.rule_bank, result.exemplar_bank, result.vote_n)
        lib.write_receipt(idir, "distil", {"hashes": hashes, "contrast_pairs": result.contrast_pairs, "refused_rules": result.refused_rules},
                          inputs={**inputs, "trajectories_sha256": tf.sha256_file(traj_path)})
        if hashes == state_hashes(rule_bank, exemplar_bank, vote_n):
            report.iterations.append({"iteration": t, "verdict": "no-change"})
            report.status, report.halt_reason = "stopped", f"iteration {t}: the distilled state equals its parent (nothing accepted)"
            return report

        # 3 read
        read = reader.read(t, hashes)
        if read is None:
            report.status, report.halt_reason = "awaiting-read", f"iteration {t}: no Fold B read of the candidate yet"
            report.iterations.append({"iteration": t, "verdict": "awaiting-read", "hashes": hashes})
            return report

        # 4 accept
        verdict = accept_candidate(read, repo_root=cfg.repo_root)
        lib.write_receipt(idir, "accept", {"status": verdict["status"], "failed": verdict["failed"], "clauses": verdict["clauses"]},
                          inputs={"hashes": hashes, "run_id": read.run_id, "attempt": read.attempt})

        # 5 admit
        state_id = admit_state(archive, iteration=t, parent_id=parent_id, hashes=hashes, verdict=verdict["status"], read=read)
        record = {"iteration": t, "verdict": verdict["status"], "failed": verdict["failed"], "state_id": state_id,
                  "construction_gain_pp": result.construction_gain_pp}
        report.iterations.append(record)
        out(f"iteration {t}: {verdict['status']} {verdict['failed'] or ''} state {state_id[:12]}")
        if not verdict["accepted"]:
            report.status, report.halt_reason = "stopped", f"iteration {t}: nothing accepted"
            return report
        report.accepted_states.append(state_id)
        parent_id, rule_bank, exemplar_bank, vote_n = state_id, result.rule_bank, result.exemplar_bank, result.vote_n
        if result.construction_gain_pp is not None and result.construction_gain_pp < STOP_GAIN_PP:
            report.status = "stopped"
            report.halt_reason = (f"iteration {t}: construction-column gain {result.construction_gain_pp:+.2f} pp under +{STOP_GAIN_PP} pp"
                                  + (" (kill criterion of the family)" if t == 1 else ""))
            return report
    report.status = "complete"
    return report


# ---------------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the iteration plan and exit; touches no file")
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS, help=f"at most {MAX_ITERATIONS}")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--rows", type=int, default=CONSTRUCTION["rows"])
    parser.add_argument("--data-root", default=os.environ.get("SPEECHRL_DATA_DIR") or os.environ.get("SAEA_DATA_ROOT"))
    args = parser.parse_args(argv)
    if not 1 <= args.iterations <= MAX_ITERATIONS:
        parser.error(f"--iterations must be between 1 and {MAX_ITERATIONS}")
    if args.dry_run:
        print("\n".join(iteration_plan(args.iterations, args.k, args.temperature, args.rows)))
        return 0
    if not args.data_root:
        parser.error("--data-root (or SPEECHRL_DATA_DIR) is required without --dry-run")
    cfg = LoopConfig(data_root=Path(args.data_root), iterations=args.iterations, k=args.k, temperature=args.temperature, rows=args.rows)
    report = run_loop(cfg, StubSampler(), StubDistiller(), FileCandidateReader(cfg.analysis_dir))
    print(json.dumps({"status": report.status, "halt_reason": report.halt_reason, "iterations": report.iterations,
                      "accepted_states": report.accepted_states}, indent=1, sort_keys=True))
    return 0 if report.status in ("complete", "stopped") else 3


if __name__ == "__main__":
    raise SystemExit(main())
