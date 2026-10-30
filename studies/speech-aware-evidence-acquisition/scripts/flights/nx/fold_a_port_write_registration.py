#!/usr/bin/env python3
"""Write the DRAFT readiness package of the Fold A CONFIRMATION twin of the L338 port flight (TRAIN Fold A, one pass of the c17 arm table, no smoke).

Config and pins only: no adapter contact, no server, no model, tokenizer, encoder, embedder or scorer contact. Flips nothing, mints nothing, countersigns nothing, and writes NO row
into ``docs/exposure-ledger.md`` unless ``--append-ledger`` is passed (do not pass it before the owner countersigns). Every file says DRAFT / HELD.

ONE package ``docs/readiness/2026-10-30-slurp-l338-fold-a-confirmation-npu-port-v1/`` with the same file set as the L338 port package, minus the analysis recipe and the READ row
(the confirmation reading re-points the port recipe at Fold A later): README, ``registered-facts-v1.json``, ``protocol-registered-v1.json``, the two plan files, ``signoff-conditions-v1.json``
(no smoke: the document says so), ``root-admission-draft-v1.json``, ``countersign-row-texts-v1.json``, ``draft-ledger-rows.md`` (the two DRAFT rows of the cell plus the five DRAFT TEXT
rows of the Fold A preparations FA1-FA5, which are NOT yet run) and, beside it, the preregistration.

The Fold A artefacts FA1-FA4 and the cell's first-phase supply do not exist yet: the cell pins them by the zero sentinel and the facts record every one as PENDING; nothing here
re-reads a receipt that is not there. Reached through ``l338_port_write_registration.py --family fold-a-confirmation``.

    python -B scripts/flights/nx/l338_port_write_registration.py --family fold-a-confirmation --study-commit <sha> --data-root <data root> [--registered-at <UTC>]
        [--root DIR] [--append-ledger] [--no-derive] [--model-fragment budget6144-max7168]
"""

from __future__ import annotations

import copy
import io
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fold_a_port_derive_cells as fdc  # noqa: E402
import fold_a_port_family as fa  # noqa: E402
import l338_family as fam  # noqa: E402
import l338_port_family as pf  # noqa: E402
import l338_port_write_registration as w  # noqa: E402

from speech_aware_evidence_acquisition.core.config import compose, protocol_hash  # noqa: E402

reg = w.reg
REPO_ROOT = fa.REPO_ROOT
PKG = fa.PACKAGE
STATUS = reg.STATUS
ZERO = fa.ZERO
MF_MODEL, MF_TAIL = w.MF_MODEL, w.MF_TAIL
#: The Fold A preparations, in build order, with the Fold B port twin each is derived from (names by the Fold B -> Fold A token rule).
PREPARATION_RUNS = {
    "encode": "SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-NPU-PORT-V1",
    "alias": "SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-A-ALIAS-NPU-PORT-V1",
    "content": "SAEA-SLURP-FOLD-A-INPUTS-PREPARE-NPU-PORT-V1",
    "agreement": "SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-A-NPU-PORT-V1",
    "supply": "SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-A-NPU-PORT-PHASE-SUPPLY-V1",
}
PREPARATION_ARTIFACTS = {
    "encode": "preparation-runs/SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-NPU-PORT-V1/encode-v1/receipt.json",
    "alias": "prepared/SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-A-ALIAS-NPU-PORT-V1/query-dimension-alias.json",
    "content": "vectors/slurp-stage3-filler-fold-a-npu-port-v1/query-vectors-speech-content-v1.provenance.json",
    "agreement": "prepared/SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-A-NPU-PORT-V1/receipt.json",
    "supply": f"prepared/{fa.CELLS['arm']['run']}/phase-supply-v1.json",
}


def _namespace() -> SimpleNamespace:
    """The Fold A family as an attribute namespace with the surface the Fold B port writer's helpers read (``pf``): the port family with the Fold A identities laid over it."""

    ns = SimpleNamespace(**{k: v for k, v in vars(pf).items() if not k.startswith("__")})
    for name in ("DATE", "PKG_NAME", "PACKAGE", "PREREG", "DECISION_ID", "DECISION_RELPATH", "HISTORICAL_PACKAGE", "HISTORICAL_DECISION_ID", "LEDGER_MARK", "PREP_LEDGER_MARK",
                 "CELLS", "ALL_KEYS", "READ_RUN_ID", "ANALYSIS_ROOT", "RECIPE_RELPATH", "SIGNOFF_CONDITIONS_RELPATH", "audio_seconds", "eligible_run_ids", "eligible_attempt_ids",
                 "signoff_conditions"):
        setattr(ns, name, getattr(fa, name))
    ns.MODEL_FRAGMENT = pf.MODEL_FRAGMENT                       # the active switch
    return ns


@contextmanager
def bound():
    """Rebind the Fold B port writer's ``pf`` / ``PKG`` to the Fold A family for the duration of a helper call (restored on exit)."""

    saved = (w.pf, w.PKG)
    w.pf, w.PKG = _namespace(), fa.PACKAGE
    try:
        yield w.pf
    finally:
        w.pf, w.PKG = saved


def identities(root: Path) -> dict:
    with bound():
        return w.identities(root)


def measurement_blocks(ids: Mapping[str, Mapping]) -> dict:
    """``w.port_measurement_blocks`` for the one Fold A cell: the helper writes a smoke block as well; a stand-in smoke identity is given and its block dropped."""

    fake = "UNUSED-FOLD-A-HAS-NO-SMOKE"
    blocks = w.port_measurement_blocks(dict(ids, smoke={"run_id": fake}))
    blocks.pop(fake)
    block = blocks[ids["arm"]["run_id"]]
    block["read_roster"] = fa.SPLIT_ID
    block["confirmation_of"] = pf.CELLS["arm"]["run"]
    block["note"] += (f" Fold A CONFIRMATION (plan section 8 R2 / M2): one pass of the same arm table on TRAIN Fold A ({fa.SPLIT_ID}); a lever is accepted only if its Fold A "
                      f"lower CI95 is above zero; the optimism term is the Fold B delta minus the Fold A delta. Fold A is a held-out discovery fold, rollout-exposed by prior cells, "
                      "never a test claim.")
    return blocks


def pending_preparations() -> dict:
    return {key: {"run_id": PREPARATION_RUNS[key], "receipt": PREPARATION_ARTIFACTS[key], "receipt_sha256": fa.sha_for(PREPARATION_ARTIFACTS[key]),
                  "status": "PENDING" if fa.sha_for(PREPARATION_ARTIFACTS[key]) == ZERO else "PINNED"} for key in PREPARATION_RUNS}


# ---- ledger rows --------------------------------------------------------------------------------------------------------------------------


def scope_text(key: str = "arm") -> str:
    spec = fa.CELLS[key]
    return (f"DRAFT -- REGISTERED. Fold A confirmation twin of {spec['historical_run']}. {spec['role'][0].upper()}{spec['role'][1:]}: "
            f"{fa.swap(w.port_scope())}; TRAIN Fold A ({fa.SPLIT_ID}, 1,080 rows, discovery role); served by the BF16 / vLLM-Ascend Thinking core over the original recordings "
            f"({pf.PROMPT_ORDER}) with the port's re-authored knowledge (intent-boundaries-npu-port-v1, slot-contracts-npu-port-v2) and the Fold A twins of the port's regenerated "
            f"retrieval vectors (CPU float32, forced causal), agreement table, content derivative and first-phase supply (preparations FA1-FA5, pending); confirmation reading: a "
            f"lever is accepted only if its Fold A lower CI95 is above zero (plan M2); {pf.DIFFERENCES_TEXT}")


def cell_rows(ids: Mapping[str, Mapping], commit: str, at: str, shas: Mapping[str, str]) -> list[list[str]]:
    rows = []
    i = ids["arm"]
    for fit in (False, True):
        name = fam.plan_file_name("arm", "fit" if fit else "generation")
        calls = i["family_budget"]["fit_endpoint_calls_cap" if fit else "generation_calls_cap"]
        rows.append([at[:10], i["fit_run_id"] if fit else i["run_id"], i["execution_profile"], scope_text() + (" -- FIT spend identity" if fit else ""), pf.model_text(),
                     fa.CARRIER_TEXT, "discovery", i["split_identity_hash"], "no", str(calls), str(i["family_budget"]["repeated_speech_audio_seconds_cap"]),
                     pf.ledger_gpu_cell("arm"), fa.INHERITED, f"{PKG}/{name}",
                     (f"DRAFT at study commit {commit}; plan SHA256 {shas.get(name, 'pending')}; drafted by the agent on owner decision 2026-10-28 item 4, owner countersignature "
                      f"pending; the decision receipt {fa.DECISION_RELPATH} is not minted; the Fold A preparations FA1-FA5 are not run, so the cell's artefact pins are the zero "
                      f"sentinel and the flip refuses it; the gpu-hours cell is the PROVISIONAL card-hour formula ({pf.CARD_HOUR_FORMULA['arm']}) and refuses the exposure gate "
                      f"until the measured sweep fills {pf.DEPLOYMENT_REF}; nothing authorised.")])
    return rows


def preparation_rows(commit: str, at: str) -> list[list[str]]:
    status = (f"DRAFT TEXT (study commit {commit}); NOT YET RUN: the row is written, its artefact sha256 filled and the row countersigned only after the run, which the main thread "
              "does in the saea-base container. ")
    no_new = f"No new speech exposure beyond {PREPARATION_RUNS['encode']}"
    carrier, split = fa.CARRIER_TEXT, fa.SPLIT_IDENTITY_HASH
    common_tail = "authorises nothing."
    return [
        [at[:10], PREPARATION_RUNS["encode"], "bounded-discovery-probe",
         (f"Port twin of SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-V1 (plan fold A of docs/readiness/2026-09-12-slurp-training-query-embedding-registration-v1.json): complete-speech "
          "retrieval-query encoding for TRAIN Fold A; no answers or scoring"),
         "Frozen local Omni-Embed-Nemotron-3B retrieval tool on the CPU float32 worker pool; no answering core", carrier, "discovery", split, "no", str(fam.ROWS),
         str(int(round(fa.ROSTER_AUDIO_SECONDS))), "2.0",
         "The historical Fold A roster, packet and query-encoding exposures carried forward; this row adds only the port's pinned retrieval-encoder contacts",
         "docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/fold-a-confirmation-fragment-draft-v1.md section 3 (the encode protocol fragment "
         "configs/preparation/slurp-tfrl-fold-a-speech-query-encoding-protocol-npu-port-v1.json is a later item of the main thread)",
         (f"{status}CPU float32 retrieval encoder (declared difference 8.7): pinned pool of 4 workers x 16 threads on cores 128-191 of the Ascend host, batch-one forwards; "
          "planned_gpu_hours is the attempt's wall-clock ceiling and the actual GPU/NPU hours are 0. Attention geometry FORCED CAUSAL (declared difference 8.4). One encode-v1 "
          "attempt; no retry, fallback, scorer, answer read, or cross-fold feedback. Owner countersignature pending; nothing authorised.")],
        [at[:10], PREPARATION_RUNS["alias"], "model-free-check",
         (f"Port twin of SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-A-ALIAS-V1: carry the port's sealed TRAIN Fold A content-dimension speech-query segment vectors into the intent-boundary "
          "retrieval dimension, values unchanged, segment identities preserved. Zero model, server, encoder, embedder, scorer, tokenizer, network, GPU and paid-API contacts."),
         MF_MODEL, carrier, "discovery", split, "no", "0", "0", "0", f"{no_new}. Prior exposure inherited in full.",
         f"{PREPARATION_ARTIFACTS['alias']} (sha256 PENDING: filled after the run)",
         f"{status}Output vectors/slurp-intent-boundaries-train-fold-a-1080-npu-port-v1/query-vectors-speech-segments.jsonl. {MF_TAIL}"],
        [at[:10], PREPARATION_RUNS["content"], "model-free-check",
         ("The port's model-free CONTENT DERIVATIVE of the Fold A speech-query vectors (scripts/build_slurp_query_content_derivative.py), the exemplar channel's query source. Zero "
          "model, server, encoder, embedder, scorer, tokenizer, network, GPU and paid-API contacts."),
         MF_MODEL, carrier, "discovery", split, "no", "0", "0", "0", f"{no_new} and its model-free alias. Prior exposure inherited in full.",
         f"{PREPARATION_ARTIFACTS['content']} (sha256 PENDING: filled after the run)",
         f"{status}Output vectors/slurp-stage3-filler-fold-a-npu-port-v1/query-vectors-speech-content-v1.jsonl. {MF_TAIL}"],
        [at[:10], PREPARATION_RUNS["agreement"], "model-free-check",
         (f"Port twin of SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-A-V1: the pre-computed, self-excluded exemplar-intent agreement table for TRAIN Fold A, arithmetic over the pinned "
          "exemplar snapshot, the port pool key vectors and the port Fold A speech query vectors through the repository's own recall/rank/fuse, against the committed never-flown "
          f"agreement-free Fold A comparator cell {fa.COMPARATOR_CELL} (source identity run {fa.COMPARATOR_RUN}). Zero model contacts."),
         MF_MODEL, carrier, "discovery", split, "no", "0", "0", "0", f"{no_new}, its model-free alias and content derivative. No gold label of any Fold A row is opened. Prior exposure inherited in full.",
         f"prepared/{PREPARATION_RUNS['agreement']}/agreement.jsonl and {PREPARATION_ARTIFACTS['agreement']} (sha256 PENDING: filled after the run)",
         f"{status}Builder scripts/build_slurp_intent_exemplar_agreement.py (the historical builder), derived from the comparator pair this package's derive writes (configs/baseline and "
         f"configs/experiment {fa.COMPARATOR_CELL}). {MF_TAIL}"],
        [at[:10], PREPARATION_RUNS["supply"], "model-free-check",
         (f"The Fold A confirmation flight's OWN prepared first-phase Perception supply, built model-free with core.flight.cli prepare-supply under the flight's run id "
          f"{fa.CELLS['arm']['run']} from the Fold A speech-query segment vectors, the port centroid key file and the Fold A agreement table (rank policy "
          "saea-rank-wcos-prior-agree-v1). This row registers the PREPARATION only. Zero model, server, encoder, embedder, scorer, tokenizer, network, GPU and paid-API contacts."),
         MF_MODEL, carrier, "discovery", split, "no", "0", "0", "0", f"{no_new} and the Fold A model-free preparations above. Prior exposure inherited in full.",
         f"{PREPARATION_ARTIFACTS['supply']} (sha256 PENDING: minted by scripts/flights/repin_cell.py at the flip)",
         f"{status}Prepared by the flip's re-pin, output equality asserted; --allow-output-change must never be passed. {MF_TAIL}"],
    ]


def countersign_texts(ids: Mapping[str, Mapping]) -> dict:
    head = ("REGISTERED BEFORE CONTACT (root countersigned <UTC timestamp> by the owner on the port brief: " + fa.OWNER_DECISIONS + "; the Fold A confirmation twin of the L338 lever column "
            "on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold A; deployment <the filled PORT_DEPLOYMENT_PROFILE: instances, tensor-parallel size, sequences per instance, devices>, "
            "card-hour bound <the re-derived number> written into the gpu-hours cell; the Fold A preparations FA1-FA5 run and their sha256 pinned <date>; ")
    tail = f"; the decision receipt {fa.DECISION_RELPATH} is minted after this countersignature)"
    i = ids["arm"]
    return {"_note": ("Before countersigning: run the Fold A preparations FA1-FA5 and record their sha256 in scripts/flights/nx/fold_a_port_pins.json, re-run the derive and the writer, "
                      "replace the gpu-hours cell's PROVISIONAL formula with the number re-derived from the measured sweep, then replace the leading 'DRAFT -- REGISTERED' scope marker "
                      "and the 'DRAFT' note prefix of each row with the text below and mint the receipt."),
            "decision_id": fa.DECISION_ID, "decision_receipt": fa.DECISION_RELPATH,
            "rows": {i["run_id"]: head + fa.CELLS["arm"]["role"] + tail, i["fit_run_id"]: head + f"live phase-fit identity of the {fa.CELLS['arm']['role']}" + tail}}


def readme_text(ids: Mapping[str, Mapping], commit: str) -> str:
    arm = ids["arm"]
    diffs = "\n".join(f"- {d}" for d in pf.DECLARED_DIFFERENCES)
    pend = "\n".join(f"- `{PREPARATION_ARTIFACTS[k]}` ({PREPARATION_RUNS[k]})" for k in PREPARATION_RUNS)
    return (
        "# L338 NPU port, Fold A confirmation: the lever column on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold A (DRAFT / HELD)\n\n"
        f"The Fold A CONFIRMATION twin of the L338 port flight (package `{pf.PACKAGE}`), registered before contact on {fa.DATE} at study commit `{commit}` by "
        "`scripts/flights/nx/l338_port_write_registration.py --family fold-a-confirmation` from the cell `scripts/flights/nx/fold_a_port_derive_cells.py` derives. "
        f"Design: `{fa.DESIGN_DOC}`; plan `docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md` (M2, owner answer R2: Fold A confirmation is required "
        f"for every accepted lever and for the D/A/T comparison). Authority: {fa.OWNER_DECISIONS}. Development numbers only; never a test claim; claims_benefit false.\n\n"
        "## The cell (one pass, no smoke)\n\n"
        "| cell | run | FIT run | rows | generation calls | FIT calls | speech-seconds | accelerator bound |\n|---|---|---|---|---|---|---|---|\n"
        f"| `{arm['cell']}` | `{arm['run_id']}` | `{arm['fit_run_id']}` | {arm['rows']} | {arm['family_budget']['generation_calls_cap']} | "
        f"{arm['family_budget']['fit_endpoint_calls_cap']} | {arm['family_budget']['repeated_speech_audio_seconds_cap']} | PROVISIONAL: {pf.CARD_HOUR_FORMULA['arm']} |\n\n"
        f"It is the Fold B port flight (program `{pf.PROGRAM}`, {pf.CONTACTS_PER_ROW} contacts per row, arms {', '.join(pf.ARM_ORDER)}) moved onto TRAIN Fold A "
        f"(`{fa.SPLIT_ID}`, roster `{fa.ROSTER_PATH}`, dataset fragment `{fa.DATASET_FRAGMENT}`) by one declared transform (every Fold B identity token becomes its Fold A token). "
        f"The calls and FIT caps equal the Fold B flight's (rows are 1,080 in both); the speech-seconds cap is the Fold A roster's: ceil({fa.ROSTER_AUDIO_SECONDS} x {fa.AUDIO_FACTOR}) = "
        f"{fa.audio_seconds():,}. There is no smoke: the programs and harness bytes were smoked and signed off on Fold B (`{pf.PACKAGE}/signoff-conditions-v1.json`).\n\n"
        "## What is pending (the Fold A preparations; nothing is run, nothing is invented)\n\n"
        "The cell pins these Fold A artefacts by the zero sentinel (no flip accepts it). Run each (CPU, model-free, card-free; the encoder is a CPU float32 tool contact), record its "
        "sha256 in `scripts/flights/nx/fold_a_port_pins.json`, re-run the derive and the writer:\n\n"
        f"{pend}\n\n"
        f"The agreement table (FA4) is built against the never-flown, agreement-free Fold A comparator pair `{fa.COMPARATOR_CELL}` (config only, written by the same derive).\n\n"
        "## Confirmation rule\n\n"
        "A lever is accepted only if its Fold A widened CI95 lower bound is above zero (M2); the optimism term (Fold B delta minus Fold A delta) is reported for every accepted state. "
        "The reader and recipe of the port (`scripts/analysis/slurp_l338_port_column_read.py`, `analysis-recipe-v1.json` of the Fold B package) are bound to the Fold B roster; "
        "re-pointing them at Fold A is a separate item. The D and T Fold A passes are separate cells (D: the direct-only writer on the Fold A roster; T: registered after an "
        "accepted iteration exists, because its baseline pins the accepted banks).\n\n"
        "## Serving: protocol facts and the deployment placeholder\n\n"
        f"Model fragment `{pf.MODEL_FRAGMENT}` (transport vLLM, BF16), per-sequence window {pf.PER_SEQUENCE_WINDOW_TOKENS:,} tokens, prompt order `{pf.PROMPT_ORDER}`; the deployment is "
        f"the one named placeholder `PORT_DEPLOYMENT_PROFILE` ({pf.PORT_DEPLOYMENT_PROFILE['filled_by']}), shared with the Fold B cells.\n\n"
        "## Declared differences (design doc section 8)\n\n"
        f"{diffs}\n\n"
        "## Files\n\n"
        "`protocol-registered-v1.json`, the two plan files, `registered-facts-v1.json`, `signoff-conditions-v1.json` (no smoke), `root-admission-draft-v1.json` (decision `"
        + fa.DECISION_ID + "`, relpath `" + fa.DECISION_RELPATH + "`, eligible run ids flight and flight FIT), `countersign-row-texts-v1.json`, `draft-ledger-rows.md` (the two DRAFT cell rows "
        "and the five DRAFT TEXT preparation rows).\n\n"
        "## What the main thread does (in order)\n\n"
        "1. Commit this package, the Fold A cell and comparator configs, and the tests.\n"
        "2. Run FA1-FA4 (card-free); pin their sha256; re-run the derive and this writer; write the preparation rows into the ledger only after each run.\n"
        "3. Fill `PORT_DEPLOYMENT_PROFILE` from the measured sweep; re-derive the card-hour bound; owner countersigns (`countersign-row-texts-v1.json`).\n"
        "4. Mint the admission receipt, re-pin and flip the cell with `scripts/flights/repin_cell.py` (it prepares the cell's own first-phase supply, FA5).\n")


def write_package(root: Path, commit: str, at: str | None, data_root: Path, *, append_ledger: bool = False, derive: bool = True) -> dict:
    root = Path(root)
    facts_path = root / PKG / "registered-facts-v1.json"
    at = at or (json.loads(facts_path.read_text(encoding="utf-8"))["registered_at_utc"] if facts_path.is_file() else datetime.now(timezone.utc).isoformat())
    derived = fdc.derive(root) if derive else None
    ids = identities(root)
    arm = ids["arm"]
    if arm["family_budget"]["repeated_speech_audio_seconds_cap"] != fa.audio_seconds() or arm["rows"] != fam.ROWS or arm["split_identity_hash"] != fa.SPLIT_IDENTITY_HASH:
        raise SystemExit("the derived Fold A experiment does not state the Fold A caps / split identity")
    preparations = pending_preparations()
    switch = w.switch_record()
    protocol = {
        "cells": {k: {kk: v[kk] for kk in ("attempt_id", "cell", "composed_config_hash", "experiment", "fit_attempt_id", "fit_run_id", "protocol_hash", "rows", "run_id",
                                           "split_identity_hash", "historical_run_id")} for k, v in ids.items()},
        "arms": {row["arm"]: {"role": row["role"], "axes": row["axes"]} for row in w.port_arm_table_rows()},
        "blockers": [{"id": "fold-a-preparations-not-run", "finding": ("The Fold A speech-query vectors, alias, content derivative, agreement table and the cell's first-phase supply "
                                                                        "(FA1-FA5) do not exist; the cell pins them by the zero sentinel."),
                      "remedy": "run FA1-FA4, record the sha256 in fold_a_port_pins.json, re-run the derive and the writer"}],
        "findings": w.port_findings(),
        "port_findings": [{"id": "findings-are-fold-b-context", "finding": ("The findings above are the Fold B port package's, carried for context; those that name the Fold B roster describe "
                                                                          "the Fold B flight."), "remedy": "none"},
                          {"id": "deployment-is-a-placeholder", "finding": pf.PORT_DEPLOYMENT_PROFILE["not_a_fact_here"], "remedy": "the measured speed sweep fills the profile"},
                          {"id": "no-smoke", "finding": ("No Fold A smoke is registered: the Fold A cell reuses the programs and harness bytes that are smoked and signed off on Fold B, "
                                                          "and its inputs are verified model-free (owner item O6 of the three-way README)."), "remedy": "none"}],
        "decision_rule_fixed_here": (f"The arm words and thresholds are the Fold B port package's ({pf.PACKAGE}); the confirmation rule is the widened CI95 lower bound above zero "
                                     "(plan M2)."),
        "failure_policy": "Stop and preserve on the first failure; at most one root-diagnosed recovery re-registration per run id; run-cumulative budgets.",
        "declared_differences": list(pf.DECLARED_DIFFERENCES), "serving": w.serving_block(), "accelerator_bound": _cost(),
        "prohibitions": w.hw.PROHIBITIONS + ["No deployment value (instances, tensor-parallel size, sequences per instance, devices) is registered here; the cell is not flown before "
                                             "PORT_DEPLOYMENT_PROFILE is filled and the Fold A preparations FA1-FA5 are run and pinned."],
        "protocol_id": "slurp-l338-fold-a-confirmation-npu-port-protocol-v1", "registered_at_utc": at, "schema": "saea-registered-protocol-v1", "status": STATUS,
        "study_commit_at_registration": commit,
        "the_question": (f"Does the L338 lever column read the same way on held-out TRAIN Fold A as on Fold B? Each candidate arm, the twin {fam.TWIN} and the replicate "
                         f"{', '.join(pf.REPLICATE_ARMS)} are read PAIRED per row against the delivering {fam.DELIVERING} arm on the same 1,080 Fold A rows in one flight, with the decoding "
                         "term of the three request-identical copies in every CI95; a lever is confirmed only if its Fold A lower CI95 is above zero.")}
    if switch:
        protocol["model_fragment_switch"] = switch
    protocol_sha = reg.wjson(root, f"{PKG}/protocol-registered-v1.json", protocol)
    with bound():
        plans = w.plan_docs(ids, commit, at, protocol_sha)
    for doc in plans.values():
        doc["prepared_phase_supply"]["note"] = ("this cell's own first-phase supply under its own run id (item E16), prepared by scripts/flights/repin_cell.py at the flip; none "
                                                "exists yet (the Fold A preparations FA1-FA4 it is built from are pending)")
        doc["confirmation_of"] = {"run_id": pf.CELLS["arm"]["run"], "package": pf.PACKAGE, "rule": "plan M2: widened CI95 lower bound above zero"}
    shas = {name: reg.wjson(root, f"{PKG}/{name}", doc) for name, doc in plans.items()}
    shas["protocol-registered-v1.json"] = protocol_sha
    shas["signoff-conditions-v1.json"] = reg.wjson(root, f"{PKG}/signoff-conditions-v1.json", fa.signoff_conditions())
    facts = {
        "_note": ("Every model-free fact this DRAFT Fold A confirmation registration rests on, and the canonical stage3_measurement block keyed by the Fold A run id. The Fold A "
                  "preparations are PENDING: they are named, not read."),
        "accounting": {"program": pf.PROGRAM, "lock": fam.LOCK, "contacts_per_row": pf.CONTACTS_PER_ROW, "arms": list(pf.ARM_ORDER), "rows": fam.ROWS,
                       "perception_contacts_per_row": pf.PERCEPTION_CONTACTS_PER_ROW, "solver_contacts_per_row": pf.SOLVER_CONTACTS_PER_ROW,
                       "replicate_arms": list(pf.REPLICATE_ARMS), "request_identical_copies": list(pf.REPLICATE_SET), "caps": _cost()["caps"], "accelerator_bound": _cost()},
        "composed_config_hashes": {k: v["composed_config_hash"] for k, v in ids.items()},
        "context_fit": w.port_fit_proof(), "identity": ids, "derive": derived,
        "pins": {"model_fragment": pf.MODEL_FRAGMENT, "serving": w.serving_block(), "fold_a_pins": {"path": "scripts/flights/nx/fold_a_port_pins.json",
                                                                                                 "sha256": w.file_sha256(root, "scripts/flights/nx/fold_a_port_pins.json")
                                                                                                 if (root / "scripts/flights/nx/fold_a_port_pins.json").is_file() else None},
                 "port_pins": {"path": "scripts/flights/nx/l338_port_pins.json", "sha256": w.file_sha256(root, "scripts/flights/nx/l338_port_pins.json")},
                 "roster": {"relpath": fa.ROSTER_PATH, "sha256": fa.sha_for(fa.ROSTER_PATH, root), "split_identity_hash": fa.SPLIT_IDENTITY_HASH,
                            "audio_seconds_total": fa.ROSTER_AUDIO_SECONDS, "items": len(fa.ROSTER["items"])},
                 "dataset_fragment": fa.DATASET_FRAGMENT, "comparator_cell": fa.COMPARATOR_CELL,
                 "signoff_conditions": {"path": f"{PKG}/signoff-conditions-v1.json", "sha256": shas["signoff-conditions-v1.json"]},
                 "preparations": preparations},
        "preparations_status": "PENDING: FA1-FA5 are not run; the cell pins them by the zero sentinel (no flip accepts it)",
        "max_num_seqs_dependents": list(pf.MAX_NUM_SEQS_DEPENDENTS), "declared_differences": list(pf.DECLARED_DIFFERENCES), "prohibitions": protocol["prohibitions"],
        "registered_at_utc": at, "registration_basis": f"{fa.OWNER_DECISIONS}; {fa.DESIGN_DOC}. DRAFT: nothing here authorises a flip or a flight.",
        "read_rule": {"confirmation_rule": "widened CI95 lower bound of every confirmed lever above zero (plan M2); optimism term = Fold B delta minus Fold A delta",
                      "reader": pf.PORT_READER, "reader_status": "bound to the Fold B roster; re-pointing at Fold A is a separate item",
                      "recipe": f"{pf.PACKAGE}/analysis-recipe-v1.json", "read_run_id": fa.READ_RUN_ID, "words": w.port_word_rules(), "word_order": w.hw.WORD_ORDER, "twin": fam.TWIN,
                      "replicates": list(pf.REPLICATE_ARMS), "floors_pp": {"slu": fam.FLOOR_SLU_PP, "intent": fam.FLOOR_INTENT_PP}, "levers": dict(fam.ARM_LEVER),
                      "delivering": fam.DELIVERING},
        "roster": {"arm": fa.ROSTER_PATH, "roster_id": fa.SPLIT_ID, "split_identity_hash": arm["split_identity_hash"]},
        "rows": fam.ROWS, "schema": "saea-registered-facts-v1", "stage3_measurement": measurement_blocks(ids), "study_commit_at_registration": commit,
        "thinking_knobs": w.port_thinking_knobs(), "no_think_node": None, "sampling_arms": {row["arm"]: row["axes"] for row in w.port_arm_table_rows()},
        "locked_decoding": dict(fam.LOCKED_DECODING), "status": "DRAFT-HELD"}
    if switch:
        facts["model_fragment_switch"] = switch
    shas["registered-facts-v1.json"] = reg.wjson(root, f"{PKG}/registered-facts-v1.json", facts)
    draft = {
        "_draft_note": (f"A DRAFT for the MAIN THREAD to mint on the data root (owner decision 2026-10-28 item 5; the mint tool of the Fold B port package, {fa.TOOL_MINT}, is bound to "
                        "that package and needs a Fold A twin). It authorises nothing while its status is not 'decided'. Before minting, the Fold A preparations are pinned, the "
                        "measured sweep fills PORT_DEPLOYMENT_PROFILE and the PROVISIONAL accelerator bound is replaced by the re-derived number."),
        "antecedents": {"protocol": {"path": f"{PKG}/protocol-registered-v1.json", "sha256": protocol_sha},
                        "registered_facts": {"path": f"{PKG}/registered-facts-v1.json", "sha256": shas["registered-facts-v1.json"]},
                        "plans": {name: {"path": f"{PKG}/{name}", "sha256": shas[name]} for name in plans}, "registration_base_commit": commit},
        "authorization_basis": f"{fa.OWNER_DECISIONS}; {fa.DESIGN_DOC}; the Fold B port package {pf.PACKAGE}.",
        "budget": {**arm["family_budget"], "paid_api_calls": 0, "plan_budget_is_run_cumulative": True,
                   "accelerator_bound": {"status": "PROVISIONAL", "formula": dict(pf.CARD_HOUR_FORMULA),
                                         "note": "gpu_hours_fit_inclusive above is the historical numeric placeholder, not the NPU card-hour bound"}},
        "claim_boundary": "No benefit, system-level, superiority, generalisation, held-out, confirmatory or paper-scale claim.",
        "decision": (f"Fly the Fold A confirmation pass of the NPU-port L338 lever column on the BF16 / vLLM-Ascend Thinking core on TRAIN Fold A: the {pf.K}-arm column flight (with the twin "
                     f"and the replicate {', '.join(pf.REPLICATE_ARMS)}), on the deployment PORT_DEPLOYMENT_PROFILE once it is filled and the Fold A preparations are pinned."),
        "decision_id": fa.DECISION_ID, "eligible_attempt_ids": fa.eligible_attempt_ids(), "eligible_run_ids": fa.eligible_run_ids(),
        "not_authorized": ["any flight before the cell is flipped and the receipt is minted", "any flight before PORT_DEPLOYMENT_PROFILE is filled",
                           "any flight before the Fold A preparations FA1-FA5 are run and pinned", "starting or touching any serving session outside the registered deployment",
                           "any ASR, transcript or paid-API contact", "--allow-output-change on the re-pin tool", "any change beyond the registered sampling axes",
                           "DEV or test-split contact"],
        "owner": "root", "population": {"count": fam.ROWS, "fold_a_contact_authorized": True, "split_identity_hash": arm["split_identity_hash"], "split_role": "discovery",
                                        "test_contact_authorized": False},
        "recorded_at_utc": None, "schema": "saea-campaign-decision-v1",
        "service": {"expected_session_id": pf.SERVICE_SESSION, "model_fragment": pf.MODEL_FRAGMENT, "deployment_profile": "PORT_DEPLOYMENT_PROFILE (pending)", "no_child_start_stop": True},
        "status": "DRAFT-NOT-MINTED"}
    shas["root-admission-draft-v1.json"] = reg.wjson(root, f"{PKG}/root-admission-draft-v1.json", draft)
    cell_text_rows = cell_rows(ids, commit, at, shas)
    prep_rows = preparation_rows(commit, at)
    prep_text = w.ledger_block(prep_rows, (
        "Five DRAFT TEXT rows for the Fold A preparations the Fold A confirmation flight pins (FA1 the CPU-encoder tool contact, FA2-FA5 model-free), in build order. NONE has been run: "
        "each row is written into the ledger only after its run, with its artefact sha256 filled. DRAFT, root countersignature pending; no row authorises a contact."), fa.PREP_LEDGER_MARK)
    cell_text = w.ledger_block(cell_text_rows, (
        f"2 unconsumed DRAFT rows (generation and live-phase-fit identity of the Fold A confirmation flight) for the Fold A confirmation twin of the L338 NPU-port lever column "
        f"({fa.OWNER_DECISIONS}). Package {PKG}. No row is countersigned and no row authorises a contact. The gpu-hours cell is the PROVISIONAL NPU card-hour formula, not a number: the "
        "exposure gate refuses these rows until the measured speed sweep fills PORT_DEPLOYMENT_PROFILE and the bound is re-derived. No smoke row: the harness bytes were smoked on Fold B."),
        fa.LEDGER_MARK + " (DRAFT, owner countersignature pending)")
    shas["draft-ledger-rows.md"] = reg.wr(root, f"{PKG}/draft-ledger-rows.md", prep_text + "\n" + cell_text)
    shas["countersign-row-texts-v1.json"] = reg.wjson(root, f"{PKG}/countersign-row-texts-v1.json", countersign_texts(ids))
    shas["README.md"] = reg.wr(root, f"{PKG}/README.md", readme_text(ids, commit))
    prereg = (f"# L338 NPU port, Fold A confirmation: the lever column, preregistration (DRAFT / HELD)\n\nRegistered {at[:10]} at study commit `{commit}` by script. "
              "Development numbers only.\n\n"
              f"Runs: `{arm['run_id']}` / `{arm['fit_run_id']}`.\n\nThe package is `{PKG}/`; the read rule is the Fold B port package's, with the confirmation rule of plan M2.\n\n"
              "## The measurement declaration\n\n```json\n" + json.dumps({"stage3_measurement": measurement_blocks(ids)}, indent=2, ensure_ascii=False, sort_keys=True) + "\n```\n")
    shas["__prereg__"] = reg.wr(root, fa.PREREG, prereg)
    result = {"package": PKG, "shas": shas, "derive": derived, "ledger_appended": None}
    if append_ledger:
        ledger = root / reg.LEDGER
        current = io.open(ledger, encoding="utf-8", newline="").read()
        if fa.LEDGER_MARK in current or fa.PREP_LEDGER_MARK in current:
            result["ledger_appended"] = False
        else:
            io.open(ledger, "w", encoding="utf-8", newline="").write(current.rstrip("\n") + "\n\n" + prep_text + "\n" + cell_text)
            result["ledger_appended"] = True
    return result


def _cost() -> dict:
    with bound():
        return w.cost_block()


if __name__ == "__main__":
    raise SystemExit(w.main(["--family", "fold-a-confirmation", *sys.argv[1:]]))
