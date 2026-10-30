#!/usr/bin/env python3
"""tfrl_v2_loop_adapters.py -- adapters that let the port loop driver (``tfrl_port_loop.py``) drive the TFRL-v2 sampler and distiller (task T3b).

Model-free. The adapters contact no model, tokenizer service, server or card, and construct no client; the token measure and the item-text
resolver are INJECTED. Nothing in ``tfrl_port_loop.py`` changes behaviour: the loop still sees its own ``Sampler`` / ``Distiller`` protocols over
credit-level :class:`Trajectory` objects, and a distiller that supplies no receipts leaves the iteration record as it was.

One :class:`V2Session` holds the shared inputs: the two half rosters, the source of the flights' trace lines, the resolver and the measure, and the
write-once store under ``<session root>/``:

    plans/i<t>-<half>.json           the run list of one half and one iteration (``tfrl_v2_sampler.build_plan``)
    banks/i<t>/<half>/...            the bank files of the bank distilled from <half> at iteration t (``tfrl_v2_distiller.write_outputs``)
    banks/i<t>/union/...             the union bank of both halves (``tfrl_v2_distiller.write_union_outputs``)
    flights/i<t>-<half>/rollout-trace.jsonl   (default location the :class:`TraceFileLines` source reads; a human-flown flight leaves it there)

The registered construction flight is flown by a human; the adapters only plan it and read what it left behind:

* :class:`V2Sampler` (``Sampler`` protocol): builds the plan of EACH half for the iteration (the half consumes the bank distilled from the OTHER
  half at the previous iteration; the cross-fit guard and the measured-bank gate of the plan builder apply), writes the plans, then reads the two
  halves' trace lines. While a half's flight has not left its trace, it raises :class:`FlightNotFlown` (a ``SamplerNotImplemented``, so the loop
  halts exactly as it does for the stub). With the lines present it returns one ``Trajectory`` per (row, k) with the row credit.
* :class:`V2Distiller` (``Distiller`` protocol): distils each half's contact and score lines into ONE bank per source half, merges them into the
  union bank, and returns the union banks as the loop's state, plus the compact bank receipts and the iteration's budget line, which the loop's
  iteration record carries.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts" / "analysis"), str(REPO_ROOT / "scripts" / "flights" / "tfrl"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from speech_aware_evidence_acquisition.contracts import rollout_trace as rt  # noqa: E402
import tfrl_port_loop as pl  # noqa: E402
import tfrl_v2_common as cm  # noqa: E402
import tfrl_v2_distiller as dd  # noqa: E402
import tfrl_v2_sampler as sp  # noqa: E402
import tfrl_v2_text as tx  # noqa: E402

SAMPLED_REASON = "sampled"


class FlightNotFlown(pl.SamplerNotImplemented):
    """The construction flight of a half has not left its trace lines yet: the loop halts and waits for the human-flown flight."""


# ----------------------------------------------------------------------------------------------------------------------------------------
# where the flights' trace lines come from
# ----------------------------------------------------------------------------------------------------------------------------------------


class FlightLines(Protocol):
    def lines(self, half: str, iteration: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
        """``(contact lines, score lines)`` of one half's construction flight at one iteration, or ``None`` while the flight has not run."""

    def sha256s(self, half: str, iteration: int) -> list[str]: ...


class TraceFileLines:
    """Reads ``saea-rollout-trace-v1`` JSONL file(s) at ``<root>/<template>`` (default ``flights/i{iteration}-{half}/rollout-trace.jsonl``)."""

    def __init__(self, root: Path, template: str = "flights/i{iteration}-{half}/rollout-trace.jsonl") -> None:
        self.root = Path(root)
        self.template = template

    def path(self, half: str, iteration: int) -> Path:
        return self.root / self.template.format(iteration=iteration, half=half)

    def lines(self, half: str, iteration: int):
        path = self.path(half, iteration)
        if not path.is_file():
            return None
        contacts, scores, _ = sp.read_trace([path])
        return contacts, scores

    def sha256s(self, half: str, iteration: int) -> list[str]:
        path = self.path(half, iteration)
        return [cm.sha256_file(path)] if path.is_file() else []


def load_half_rosters(repo_root: Path = REPO_ROOT) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """The committed half rosters and the sha256 of their files."""

    docs: dict[str, dict[str, Any]] = {}
    shas: dict[str, str] = {}
    for half in cm.HALVES:
        raw = (Path(repo_root) / cm.HALF_ROSTER_RELPATH[half]).read_bytes()
        docs[half] = json.loads(raw.decode("utf-8"))
        shas[half] = cm.sha256_bytes(raw)
    return docs, shas


# ----------------------------------------------------------------------------------------------------------------------------------------
# the shared session
# ----------------------------------------------------------------------------------------------------------------------------------------


@dataclass
class V2Session:
    root: Path
    rosters: Mapping[str, Mapping[str, Any]]
    flight_lines: FlightLines
    resolver: tx.ItemTextResolver
    measure: tx.TokenMeasure
    roster_sha256: Mapping[str, str | None] = field(default_factory=dict)
    variation: str = "seed-only"
    variants_fn: Callable[[int, Mapping[str, Any]], Mapping[str, Any] | None] | None = None   # (iteration, consumed rule bank) -> seed rule bank
    # ^ ``tfrl_v2_distiller.rule_variants_seed_bank`` for ``variation="rule-variants"`` (k0 none, k1 full, k2 row scenario, k3 top half; K must be 4)
    scorer_id: str | None = None
    with_twin: bool = True
    allow_missing_rows: bool = False

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if set(self.rosters) != set(cm.HALVES):
            raise pl.PortLoopError(f"the session needs the rosters of exactly {list(cm.HALVES)}")
        if self.measure.name != cm.MEASURE_VLLM:
            raise pl.PortLoopError(f"the session measure is {self.measure.name!r}; a flight-bound bank needs {cm.MEASURE_VLLM!r}")

    # -- paths
    def plan_path(self, iteration: int, half: str) -> Path:
        return self.root / "plans" / f"i{iteration}-{half}.json"

    def bank_dir(self, iteration: int, name: str) -> Path:
        return self.root / "banks" / f"i{iteration}" / name

    def column_rows(self) -> list[str]:
        return sorted(str(i["sample_id"]) for half in cm.HALVES for i in self.rosters[half]["items"])

    # -- the bank a half consumes
    def consumed_bank(self, iteration: int, half: str):
        """``(receipt, rule bank)`` of the bank distilled from the OTHER half at the previous iteration; ``(None, None)`` at iteration 1."""

        if iteration == 1:
            return None, None
        source = cm.other_half(half)
        path = self.bank_dir(iteration - 1, source) / "bank-receipt.json"
        if not path.is_file():
            raise pl.PortLoopError(f"iteration {iteration} of {half} consumes the bank of {source} at iteration {iteration - 1}; {path} is missing")
        receipt, _exemplars, rules = dd.load_bank(path)
        return receipt, rules

    # -- plans
    def plan(self, iteration: int, half: str, *, k: int, temperature: float, seed: int) -> dict[str, Any]:
        receipt, rules = self.consumed_bank(iteration, half)
        seed_bank = self.variants_fn(iteration, rules) if (self.variants_fn is not None and rules is not None and rules["rules"]) else None
        variation = self.variation
        if variation == "rule-variants" and not seed_bank:
            variation = "seed-only"
        try:
            doc = sp.build_plan(self.rosters[half], half, iteration, k=k, base_seed=seed, temperature=temperature, variation=variation,
                                bank_receipt=receipt, seed_rule_bank=seed_bank, roster_file_sha256=self.roster_sha256.get(half))
        except (sp.SamplerError, cm.CommonError) as error:
            raise pl.PortLoopError(f"plan of {half} iteration {iteration} refused: {error}") from error
        try:
            cm.write_once(self.plan_path(iteration, half), cm.file_bytes(doc))
        except cm.CommonError as error:
            raise pl.PortLoopError(str(error)) from error
        return doc

    def budget_line(self, iteration: int) -> str:
        """The iteration's budget line (program c7, both halves), re-derived from the plans on disk."""

        parts = []
        for half in cm.HALVES:
            path = self.plan_path(iteration, half)
            if not path.is_file():
                raise pl.PortLoopError(f"no plan of {half} at iteration {iteration}: {path}")
            parts.append(json.loads(path.read_text(encoding="utf-8"))["budget"]["program_c7"])
        return sp.budget_line(f"iteration {iteration} both halves program c7", sp.sum_budgets(parts))

    def flight(self, iteration: int, half: str):
        got = self.flight_lines.lines(half, iteration)
        if got is None:
            raise FlightNotFlown(f"iteration {iteration}: the construction flight of {half} has left no trace lines yet "
                                 f"(run list {self.plan_path(iteration, half)})")
        return got


# ----------------------------------------------------------------------------------------------------------------------------------------
# the tokenizer driver and the session builder (the loop command line's --v2)
# ----------------------------------------------------------------------------------------------------------------------------------------

DEFAULT_TOKENIZE_CONTACT_CAP = 20000
DEFAULT_TOKENIZE_TIMEOUT_SECONDS = 600.0


def tokenize_client(endpoint: str, *, timeout_seconds: float = DEFAULT_TOKENIZE_TIMEOUT_SECONDS, contact_cap: int = DEFAULT_TOKENIZE_CONTACT_CAP,
                    guard: Callable[[], None] | None = None, client_factory: Callable[..., Any] | None = None) -> Any:
    """The registered FIT-gate tokenizer client for ``endpoint`` (``http://host:port``): ``core.preparation_fit.GuardedTokenizer`` over the
    keep-alive ``core.driver.speech_native._LoopbackPromptAuditClient`` -- the very stack ``run_live_fit`` / the port FIT gate build (``client_factory
    or _LoopbackPromptAuditClient``, then ``GuardedTokenizer``). Only the tokenizer routes pass the guard (``POST /tokenize`` here); every contact counts
    against ``contact_cap`` and a cap hit is a refusal. ``guard`` is the per-contact lease check (the flight runtime passes
    ``assert_generation_lease``; this offline driver has no lease and passes none). Constructing the client opens no connection; the first measured
    text does. A test injects ``client_factory`` (called with ``host``, ``port``, ``timeout_seconds``) and never reaches a socket.
    """

    from urllib.parse import urlsplit  # noqa: PLC0415

    from speech_aware_evidence_acquisition.core.preparation_fit import GuardedTokenizer  # noqa: PLC0415

    parts = urlsplit(endpoint if "://" in endpoint else f"http://{endpoint}")
    if parts.scheme != "http" or not parts.hostname or parts.port is None or parts.path not in ("", "/") or parts.query or parts.fragment:
        raise pl.PortLoopError(f"the tokenize endpoint {endpoint!r} must be http://host:port")
    if client_factory is None:
        from speech_aware_evidence_acquisition.core.driver.speech_native import _LoopbackPromptAuditClient  # noqa: PLC0415

        client_factory = _LoopbackPromptAuditClient
    try:
        return GuardedTokenizer(client_factory(host=parts.hostname, port=parts.port, timeout_seconds=timeout_seconds),
                                guard=guard or (lambda: None), contact_cap=contact_cap)
    except Exception as error:  # noqa: BLE001 -- a malformed client configuration or cap
        raise pl.PortLoopError(f"cannot build the tokenize client for {endpoint!r}: {error}") from error


def build_measure(name: str, *, endpoint: str | None = None, model: str | None = None, client_factory: Callable[..., Any] | None = None,
                  contact_cap: int = DEFAULT_TOKENIZE_CONTACT_CAP) -> tx.TokenMeasure:
    """``estimate`` (offline) or ``vllm-tokenize`` over :func:`tokenize_client`; the latter needs ``endpoint``."""

    if name == cm.MEASURE_ESTIMATE:
        return tx.EstimateMeasure()
    if name != cm.MEASURE_VLLM:
        raise pl.PortLoopError(f"unknown measure {name!r}; known: {list(tx.MEASURES)}")
    if not endpoint:
        raise pl.PortLoopError(f"measure {cm.MEASURE_VLLM!r} needs a tokenize endpoint")
    return tx.VllmTokenizeMeasure(tokenize_client(endpoint, client_factory=client_factory, contact_cap=contact_cap), model=model)


def build_session(*, data_root: Path, analysis_root: Path, measure: str, tokenize_endpoint: str | None = None, tokenize_model: str | None = None,
                  variation: str = "seed-only", repo_root: Path = REPO_ROOT, resolver: tx.ItemTextResolver | None = None,
                  flight_lines: FlightLines | None = None, client_factory: Callable[..., Any] | None = None, **session_kw: Any) -> "V2Session":
    """A :class:`V2Session` from the command-line inputs: committed half rosters (and their file hashes), the item-text resolver over the registered
    knowledge stores of ``data_root`` (``ItemTextResolver.from_pins``), the chosen measure, and the write-once store under ``<analysis root>/v2``.
    ``variation='rule-variants'`` sets ``variants_fn`` to ``tfrl_v2_distiller.rule_variants_seed_bank``. The session itself refuses an estimate measure."""

    try:
        docs, shas = load_half_rosters(repo_root)
        the_resolver = resolver or tx.ItemTextResolver.from_pins(Path(data_root))
        the_measure = build_measure(measure, endpoint=tokenize_endpoint, model=tokenize_model, client_factory=client_factory)
    except (OSError, ValueError, KeyError, tx.ItemTextError) as error:
        raise pl.PortLoopError(f"cannot build the V2 session: {error}") from error
    root = Path(analysis_root) / "v2"
    return V2Session(root=root, rosters=docs, flight_lines=flight_lines or TraceFileLines(root), resolver=the_resolver, measure=the_measure,
                     roster_sha256=shas, variation=variation, variants_fn=dd.rule_variants_seed_bank if variation == "rule-variants" else None, **session_kw)


# ----------------------------------------------------------------------------------------------------------------------------------------
# the adapters
# ----------------------------------------------------------------------------------------------------------------------------------------


class V2Sampler:
    """``Sampler`` over the TFRL-v2 plan builder and the flights' trace lines."""

    def __init__(self, session: V2Session) -> None:
        self.session = session

    def sample(self, request: pl.SamplerRequest) -> list[pl.Trajectory]:
        session = self.session
        if sorted(request.column_rows) != session.column_rows():
            raise pl.PortLoopError("the request's column rows are not exactly the rows of the two halves")
        plans = {half: session.plan(request.iteration, half, k=request.k, temperature=request.temperature, seed=request.seed) for half in cm.HALVES}
        missing = [half for half in cm.HALVES if session.flight_lines.lines(half, request.iteration) is None]
        if missing:
            raise FlightNotFlown(f"iteration {request.iteration}: no trace lines yet for {missing}; fly the construction cells of the run lists under "
                                 f"{session.root / 'plans'}")
        out: list[pl.Trajectory] = []
        for half in cm.HALVES:
            contacts, scores = session.flight(request.iteration, half)
            out.extend(self._half(half, plans[half], contacts, scores, request, session.consumed_bank(request.iteration, half)[1]))
        out.sort(key=lambda t: (t.row_id, t.k_index))
        return out

    def _half(self, half: str, plan: Mapping[str, Any], contacts, scores, request: pl.SamplerRequest, consumed_rules) -> list[pl.Trajectory]:
        session = self.session
        row_ids = [str(i["sample_id"]) for i in session.rosters[half]["items"]]
        in_half = set(row_ids)
        arms = sp.sampler_arms(request.k)
        answers = sp.solver_contacts([c for c in contacts if c["row_id"] in in_half], arms)
        try:
            table = sp.scores_from_lines([s for s in scores if s["row_id"] in in_half], scorer_id=session.scorer_id)
            credit = sp.credit_rows(row_ids, request.iteration, request.k, table, twin_arm=sp.TWIN_ARM if session.with_twin else None)
        except sp.SamplerError as error:
            raise pl.PortLoopError(f"{half} iteration {request.iteration}: {error}") from error
        rule_ids = tuple(r["rule_id"] for r in (consumed_rules or {}).get("rules", []))
        variant_rules = {(row["row_id"], c["sample_index"]): tuple(c["rule_variant"]["rule_ids"])
                         for row in plan["row_plan"] for c in row["contacts"]
                         if c["node"] == "solver" and isinstance(c.get("rule_variant"), Mapping) and c["rule_variant"].get("rule_ids") is not None}
        out: list[pl.Trajectory] = []
        for row in credit:
            for traj in row["trajectories"]:
                contact = answers.get((row["row_id"], traj["arm"]))
                if contact is None:
                    raise pl.PortLoopError(f"{half} iteration {request.iteration}: row {row['row_id']} has no delivered Solver answer for {traj['arm']}")
                trace = {"schema": rt.SCHEMA, "kind": rt.KIND_TFRL, "ts": contact["ts"], "run_id": plan["run_id"], "attempt_id": plan["attempt_id"],
                         "row_id": row["row_id"], "arm": traj["arm"], "iteration": request.iteration, "trajectory_id": traj["trajectory_id"],
                         "reward": traj["reward"], "selected": False, "reason": SAMPLED_REASON}
                out.append(pl.Trajectory(trace=trace, k_index=traj["k_index"], temperature=request.temperature, row_f1=traj["row_f1"],
                                         lock_row_f1=row["baseline_f1"],
                                         rule_ids=variant_rules.get((row["row_id"], traj["k_index"]), rule_ids), exemplar_ids=(), truncated=bool(contact["truncated"])))
        return out


class V2Distiller:
    """``Distiller`` over the TFRL-v2 distiller: one bank per source half, then the union."""

    def __init__(self, session: V2Session) -> None:
        self.session = session

    def distil(self, iteration: int, trajectories: Sequence[pl.Trajectory], rule_bank: Mapping[str, Any],
               exemplar_bank: Mapping[str, Any]) -> pl.DistilResult:
        session = self.session
        rows = {t.row_id for t in trajectories}
        if rows != set(session.column_rows()):
            raise pl.PortLoopError("the trajectories do not cover exactly the rows of the two halves")
        k = len(trajectories) // len(rows)
        receipts: dict[str, Mapping[str, Any]] = {}
        loaded = {}
        pairs = refused = 0
        for half in cm.HALVES:
            contacts, scores = session.flight(iteration, half)
            try:
                result = dd.distil(session.rosters[half], half, iteration, contacts, scores, k=k, scorer_id=session.scorer_id, with_twin=session.with_twin,
                                   allow_missing_rows=session.allow_missing_rows, roster_file_sha256=session.roster_sha256.get(half),
                                   trace_sha256s=session.flight_lines.sha256s(half, iteration), resolver=session.resolver, measure=session.measure)
                out_dir = session.bank_dir(iteration, half)
                dd.write_outputs(result, out_dir)
            except (dd.DistillError, sp.SamplerError, cm.CommonError, tx.ItemTextError, tx.MeasureError) as error:
                raise pl.PortLoopError(f"distilling {half} at iteration {iteration} refused: {error}") from error
            loaded[half] = dd.load_bank(out_dir / "bank-receipt.json")
            receipts[half] = self._compact(out_dir)
            pairs += result["receipt_core"]["counts"]["contrast_pairs"]
            refused += result["receipt_core"]["counts"]["rules_refused"]
        try:
            union = dd.union_banks(loaded[cm.HALF_H1], loaded[cm.HALF_H2], measure=session.measure)
            union_dir = session.bank_dir(iteration, "union")
            dd.write_union_outputs(union, union_dir)
        except (dd.DistillError, cm.CommonError, tx.MeasureError) as error:
            raise pl.PortLoopError(f"union of the banks at iteration {iteration} refused: {error}") from error
        receipts["union"] = self._compact(union_dir)
        return pl.DistilResult(rule_bank=union["rule_bank"], exemplar_bank=union["exemplar_bank"], vote_n=cm.VOTE_MIN_ROWS, contrast_pairs=pairs,
                               refused_rules=refused, construction_gain_pp=None, bank_receipts=receipts, budget_line=session.budget_line(iteration))

    def _compact(self, directory: Path) -> dict[str, Any]:
        receipt_path = directory / "bank-receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        return {"dir": str(directory.relative_to(self.session.root)), "receipt_file_sha256": cm.sha256_file(receipt_path), "kind": receipt["kind"],
                "source_halves": receipt["source_halves"], "measure": receipt["measure"], "exemplar_text": receipt["exemplar_text"],
                "exemplar_bank_content_sha256": receipt["exemplar_bank"]["content_sha256"], "exemplar_entries": receipt["exemplar_bank"]["entries"],
                "rule_bank_content_sha256": receipt["rule_bank"]["content_sha256"], "rules": receipt["rule_bank"]["rules"]}
