#!/usr/bin/env python3
"""Write the DRAFT readiness package of the NPU-port twin of the 266-E direct-only cell (arm D of the three-way comparison, TRAIN Fold B) from the port fragments.

Config and receipt-pinned bytes only: no adapter contact, no server, no model, tokenizer, encoder, embedder or scorer contact. Flips nothing, mints nothing,
countersigns nothing, and writes NO row into ``docs/exposure-ledger.md`` unless ``--append-ledger`` is passed. Every file says DRAFT / HELD.

ONE package ``docs/readiness/2026-10-30-slurp-direct-only-fold-b-npu-port-v1/`` (README, ``registered-facts-v1.json`` with the canonical ``stage3_measurement``
blocks keyed by the port run ids, ``protocol-registered-v1.json``, the eight plan files (generation and FIT of the smoke and of the lock, aa and ab passes),
``signoff-conditions-v1.json``, ``root-admission-draft-v1.json``, ``countersign-row-texts-v1.json`` and ``draft-ledger-rows.md``) and, beside it, the preregistration.
The ledger block is the eight DRAFT rows of the smoke and the three passes (generation and FIT each; owner 2026-10-28 item 4).

The port settings (model fragment, deployment placeholder, declared differences, owner decisions, PROVISIONAL card-hour form) are imported from
``l338_port_family`` through ``direct_port_family``; the 266-E writer ``direct_fb_write_registration.py`` and its package are NOT touched. The deployment
(instances, tensor-parallel size, sequences per instance, devices) is the placeholder ``PORT_DEPLOYMENT_PROFILE``; the ledger's gpu-hours cells carry the
PROVISIONAL formula, NOT a number, so the exposure gate refuses the rows until the sweep fills the profile.

    python -B scripts/flights/direct/direct_port_write_registration.py --study-commit <sha> --data-root <data root> [--registered-at <UTC>]
        [--root DIR] [--append-ledger] [--no-derive]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import direct_port_derive_cells as pdc  # noqa: E402
import direct_port_family as fam  # noqa: E402
import l338_port_family as pf  # noqa: E402
import write_tfrl_iteration_registration as reg  # noqa: E402  (generic helpers: wjson / wr / table_row / LEDGER_COLUMNS / STATUS)

REPO_ROOT = fam.REPO_ROOT
from speech_aware_evidence_acquisition.core.config import compose, fragment_hash, protocol_hash, read_fragment  # noqa: E402

STATUS = reg.STATUS
PKG = fam.PACKAGE
CARRIER = "slurp TRAIN Fold B"
INHERITED = ("TRAIN Fold B discovery rows already exposed by prior SLURP cells (the 266-E Thinking arm SAEA-SLURP-DIRECT-ONLY-FOLD-B-THINKING-V1 and the L338 port "
             "registration included); this cell opens no split. Prior exposure is inherited in full.")
PRIMARY_METRIC = ("official SLURP pooled SLU-F1 (micro; scorer slurp-slu-official-train, the pinned official evaluate.py), natural-mixture weighting, with intent accuracy "
                  "(scenario_action_exact) beside it; the per-row F1 is a repo diagnostic reported beside the pooled F1, never an official SLURP number")
CROSS_RUN_BAR_PP = 2.5
TWIN_BAR_PP = 1.2
ANTECEDENT_RECEIPTS = ("receipts/runtime-vllm-bf16-thinking-npu.json",)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---- identities, caps, proofs ------------------------------------------------------------------------------------------------------------


def identities(root: Path) -> dict:
    out = {}
    for key in fam.ALL_KEYS:
        spec = fam.CELLS[key]
        document = json.loads((Path(root) / "configs" / "experiment" / f"{spec['cell']}.json").read_text(encoding="utf-8"))
        cell = document["cells"][0]
        composed = compose(Path(root), cell["model_fragment"], cell["dataset_fragment"], cell["baseline_fragment"], spec["cell"])
        split = document["population"]["split_identity_hash"]
        values = composed.values
        fit = dict(values["live_phase_fit"])
        if (fit["run_id"], fit["attempt_id"]) != (spec["fit_run"], spec["fit_attempt"]) or (cell["run_id"], cell["attempt_id"]) != (spec["run"], spec["attempt"]):
            raise SystemExit(f"{key}: the composed identities are not the registered ones")
        if fit["contact_cap"] != fam.fit_calls(key):
            raise SystemExit(f"{key}: live_phase_fit contact_cap {fit['contact_cap']} is not rows x 1 x 1 = {fam.fit_calls(key)}")
        if values["arm"] != fam.ARM or values.get("policy") != fam.PROGRAM or values.get("preserve_wire_responses") is not False:
            raise SystemExit(f"{key}: the composed arm / program / wire-response flag is not the registered one")
        if document["family_budget"]["generation_calls_cap"] != fam.generation_calls(key) or document["family_budget"]["fit_endpoint_calls_cap"] != fam.fit_calls(key) \
                or document["family_budget"]["repeated_speech_audio_seconds_cap"] != fam.audio_seconds(key):
            raise SystemExit(f"{key}: the experiment caps are not the family's arithmetic")
        out[key] = {
            "key": key, "experiment": spec["cell"], "cell": cell["cell_id"], "role": spec["role"], "run_id": cell["run_id"], "attempt_id": cell["attempt_id"],
            "fit_run_id": spec["fit_run"], "fit_attempt_id": spec["fit_attempt"], "baseline_fragment": cell["baseline_fragment"],
            "dataset_fragment": cell["dataset_fragment"], "model_fragment": cell["model_fragment"], "arm": values["arm"], "program": values["policy"],
            "rows": document["population"]["count"], "split_identity_hash": split, "roster_receipt": document["population"].get("roster_receipt"),
            "composed_config_hash": composed.config_hash, "protocol_hash": protocol_hash(composed.config_hash, values["arm"], split),
            "fit": fit, "family_budget": document["family_budget"], "execution_plan": document["execution_plan"], "execution_profile": document["execution_profile"],
            "carrier_lock_key": document["execution_plan"].get("carrier_lock_key", "slurp"), "session": document["service_session"],
            "historical_run_id": spec["historical_run"], "historical_fit_run_id": spec["historical_fit_run"],
        }
    return out


def request_identity_proof(root: Path) -> dict:
    """lock, aa and ab are request-identical: every composed key but the live_phase_fit identity and the prose is byte-equal."""

    values = {}
    for key in fam.PASS_KEYS:
        spec = fam.CELLS[key]
        doc = json.loads((Path(root) / "configs" / "experiment" / f"{spec['cell']}.json").read_text(encoding="utf-8"))
        cell = doc["cells"][0]
        values[key] = compose(Path(root), cell["model_fragment"], cell["dataset_fragment"], cell["baseline_fragment"], spec["cell"]).values
    prose = {"live_phase_fit", "what_this_is", "cells", "purpose", "experiment_id", "generation_budget", "core_read_design", "readiness"}
    differing = {}
    for key in fam.REPLICATE_KEYS:
        diff = sorted(name for name in set(values[key]) | set(values["lock"]) if name not in prose and values[key].get(name) != values["lock"].get(name))
        if diff:
            raise SystemExit(f"{key} differs from the lock in composed keys {diff}: not request-identical")
        differing[key] = diff
    return {"compared_to": "lock", "differing_composed_keys_outside_the_fit_identity_and_prose": differing, "none_of_them_enters_a_request": True}


def audio_pin_check(data_root: Path) -> dict:
    """Re-hash every Fold B recording against the dataset fragment's pins on the data root (model-free)."""

    dataset = json.loads((REPO_ROOT / "configs" / "dataset" / f"{fam.DATASET}.json").read_text(encoding="utf-8"))
    bad = []
    for item in dataset["items"]:
        path = Path(data_root) / item["media_relpath"]
        if not path.is_file() or _sha(path) != item["audio_sha256"]:
            bad.append(item["sample_id"])
    if bad:
        raise SystemExit(f"{len(bad)} Fold B recordings do not match their audio pins on the data root (first: {bad[0]})")
    return {"checked": len(dataset["items"]), "matching": len(dataset["items"]),
            "dataset_fragment_sha256": _sha(REPO_ROOT / "configs" / "dataset" / f"{fam.DATASET}.json"),
            "split_identity_hash": dataset["split_identity_hash"], "audio_seconds_total": dataset["audio_seconds_total"]}


def fragment_pins(root: Path) -> dict:
    fragment = read_fragment(Path(root), "model", fam.MODEL_FRAGMENT)
    return {"model_fragment": fam.MODEL_FRAGMENT, "file_sha256": _sha(Path(root) / "configs" / "model" / f"{fam.MODEL_FRAGMENT}.json"),
            "fragment_hash_with_data_root_set": fragment_hash(fragment),
            "note": ("fragment_hash is taken with SAEA_DATA_ROOT / SPEECHRL_DATA_DIR resolved, as flights run (plan section 9 ruling 5); without a data root the "
                     "materialization path does not collapse to the placeholder and the same function returns another value"),
            "reasoning_budget_tokens": fragment["reasoning_budget_tokens"], "answer_reserve_tokens": fragment["answer_reserve_tokens"],
            "max_tokens": fragment["decoding_params"]["max_tokens"], "timeout_seconds": fragment["timeout_seconds"]}


# ---- the read rule and the measurement blocks -----------------------------------------------------------------------------------------------

READ_RULE = {
    "kind": "PREREGISTERED reading of arm D; development numbers only",
    "primary": PRIMARY_METRIC,
    "replicates": {"passes": list(fam.PASS_KEYS), "k": 2, "note": ("lock is the delivering pass; aa and ab are request-identical replicates in the same chain (noise model M1: "
                                                                  "separate runs, not the in-flight twin, because the direct-only program has no arm table). Same-chain replicates "
                                                                  "also absorb session drift, so they measure at least the within-run decoding term.")},
    "twin_admissibility": f"|SLU-F1(aa) - SLU-F1(lock)| <= {TWIN_BAR_PP} pp (the port analysis recipe's twin rule)",
    "a_to_d_gap": f"A-to-D is a cross-run, cross-program comparison read at the cross-run bar of {CROSS_RUN_BAR_PP} pp; the expected gap is about 26 pp, so the bar does not bind",
    "truncation": ("A cap-hit row is sealed as a receipted empty answer (saea-empty-completion-v1 and saea-thinking-answer-fallback-v1) and scored as such; never re-asked "
                   "beyond the one registered answer-only fallback (266-E preregistration section 3; decision 199). The M4 trace logs the truncation flag per contact."),
    "bootstrap": {"seed": 20261011, "draws": 10000, "level": "CI95", "note": "the 266-E values unless the T2 recipe for D fixes others"},
    "analysis_recipe": "not part of this package: the D reader and its recipe are an open item; this block is the rule it must implement",
    "claim_boundary": "No system-level, superiority, generalisation, test or confirmatory claim; a development-fold capability read of the bare core.",
}


def measurement_blocks(ids: Mapping[str, Mapping]) -> dict:
    out = {}
    for key in fam.ALL_KEYS:
        run = ids[key]["run_id"]
        if key == "smoke":
            out[run] = {"claims_benefit": False, "expected_effect_pp": 0.0, "measurement_design": "form-proof", "rows": fam.SMOKE_ROWS,
                        "primary_metric": "none: a four-row decode-cost smoke reads no metric", "read_roster": "SLURP-EVO-FOLD-B-V2", "tuning_roster": "none",
                        "note": ("Four Fold B rows prove FORM only: the raw-audio direct-only request form, the live phase fit of the direct-only raw gate and the seal work, and what "
                                 "one Solver contact costs under the port budget. Scored by no one; four rows claim nothing under instrument rule 5.")}
        else:
            other = "lock" if key != "lock" else "aa"
            out[run] = {"claims_benefit": False, "expected_effect_pp": 0.0, "measurement_design": "cross-flight-with-in-chain-replicate", "rows": fam.ROWS,
                        "primary_metric": PRIMARY_METRIC, "read_roster": "SLURP-EVO-FOLD-B-V2", "tuning_roster": "none", "replicate_run_id": ids[other]["run_id"],
                        "note": ("A capability read of the bare core on the port, no mechanism and no claim: expected_effect_pp is 0.0 and claims_benefit false. lock, aa and ab are "
                                 "three request-identical passes in one chain (noise model M1); the replicate variance enters every CI95 of the A-to-D and T-to-D contrasts.")}
    return out


# ---- blocks shared by the documents ---------------------------------------------------------------------------------------------------


def serving_block() -> dict:
    return {"model_fragment": fam.MODEL_FRAGMENT, "transport": "vllm", "per_sequence_window_tokens": pf.PER_SEQUENCE_WINDOW_TOKENS,
            "deployment_profile": json.loads(json.dumps(pf.PORT_DEPLOYMENT_PROFILE)), "session": fam.SERVICE_SESSION,
            "session_identity": "open owner decision 9.2 (design doc section 9)", "runtime_receipt": "open owner decision 9.3 (design doc section 9)",
            "lock_entries": "open owner decision 9.4 (design doc section 9)"}


def cost_block() -> dict:
    return {
        "status": "PROVISIONAL",
        "formula": {k: fam.CARD_HOUR_FORMULA[k] for k in fam.ALL_KEYS},
        "inputs": {"arm_mix_rows": fam.ARM_MIX_ROWS, "margin": fam.GPU_MARGIN, "round_hours": fam.GPU_ROUND_H, "rows": fam.ROWS,
                   "smoke_wall_clock_hours": fam.SMOKE_WALL_CLOCK_HOURS},
        "unknowns_from_the_deployment_profile": ["T = tensor_parallel_size", "r = measured lock-only rows per minute of one instance at its sequences_per_instance",
                                                 "instances (wall-clock only; cancels in the card-hours)"],
        "arm_mix_note": ("0.52 = one Solver contact in lock-row equivalents, solver / (perception + solver) node seconds of scripts/flights/nx/l338_cost_inputs.json; a "
                         "direct-only contact carries no knowledge, so the prefill side is an over-estimate and the reasoning length is not; the smoke measures the real mix"),
        "plan_field_placeholder": {k: fam.CELLS[k]["gpu_hours"] for k in fam.ALL_KEYS},
        "plan_field_note": ("execution_plan.planned_gpu_hours and family_budget.gpu_hours_fit_inclusive carry the 266-E figures (0.5 / 4.5) only because the plan field must "
                            "be a number; they are NOT the NPU bound. The ledger gpu-hours cells carry the formula and fail the exposure gate closed until the sweep "
                            "re-derives the bound and the main thread re-writes plan, rows and receipt together."),
        "caps": {k: {"generation_calls": fam.generation_calls(k), "fit_endpoint_calls": fam.fit_calls(k), "speech_audio_seconds": fam.audio_seconds(k)} for k in fam.ALL_KEYS},
        "cell_total": {"generation_calls": fam.CELL_TOTAL[0], "fit_endpoint_calls": fam.CELL_TOTAL[1], "speech_audio_seconds": fam.CELL_TOTAL[2]},
        "caps_note": (f"generation = rows x ({fam.CONTACTS_PER_ROW} contact + {fam.FALLBACK_CONTACTS_PER_ROW} answer-only fallback), FIT = rows x {fam.CONTACTS_PER_ROW} x "
                      f"{fam.DIRECT_ONLY_ENDPOINTS_PER_CONTACT} counting endpoint (core.preparation_fit.direct_only_fit_contact_cap, the value build_direct_only_raw_fit_gate demands), "
                      "audio = ceil(roster speech-seconds x the generation contacts per row). The ceiling per Fold B row is 2 contacts per pass with the fallback re-ask; the plan's "
                      "'2 per row' counts the twin only, so the cell makes 3 programme contacts per Fold B row over its three passes."),
    }


def fit_path_block() -> dict:
    return {"gate": "core.preparation_fit.build_direct_only_raw_fit_gate (reached from build_live_phase_fit_gate behind the program id phase-direct-only-v1 on the raw-audio arm)",
            "cap_rule": "live_phase_fit.contact_cap == direct_only_fit_contact_cap(rows) == rows x 1 contact x 1 counting endpoint",
            "envelope_checks": ["sealed direct-only program", "configured cap equals rows x 1", "generation plan covers one contact per row", "complete unique roster in one wave",
                                "every request carries exactly the original audio it names", "audio seconds of the wave (WAV headers) fit the plan's audio ceiling"],
            "audio_ceiling_source": "live_phase_fit_audio_seconds_cap (core.flight.runtime._fit_base_plan) = the generation plan's audio ceiling",
            "status": "landed (commit 56d3c1c3, tests/unit/test_direct_only_fit.py); no longer OPEN"}


def plan_docs(ids: Mapping[str, Mapping], commit: str, at: str, protocol_sha: str) -> dict:
    docs = {}
    for key in fam.ALL_KEYS:
        i, spec = ids[key], fam.CELLS[key]
        base = {"cell": i["cell"], "composed_config_hash": i["composed_config_hash"], "expected_session_id": fam.SERVICE_SESSION, "experiment": i["experiment"],
                "family_budget": i["family_budget"], "preparation_base_commit": commit,
                "protocol_document": {"relpath": f"{PKG}/protocol-registered-v1.json", "sha256": protocol_sha}, "registered_at_utc": at, "status": STATUS,
                "serving": serving_block(), "accelerator_bound": cost_block(),
                "historical_twin": {"run_id": i["historical_run_id"], "fit_run_id": i["historical_fit_run_id"], "package": fam.HISTORICAL_PACKAGE},
                "declared_differences": list(fam.DECLARED_DIFFERENCES), "unused_budget_is_not_attempt_authority": True,
                "one_factor": ("None within the cell: three request-identical passes. Against the 266-E twin the port changes the core precision and serving stack, the speech input path "
                               "and the thinking budget together (the declared differences), so it is read beside the historical arm, never as a one-factor contrast with it.")}
        gen = dict(i["execution_plan"], attempt_id=i["attempt_id"], carrier_lock_key=i["carrier_lock_key"], protocol_hash=i["protocol_hash"], run_id=i["run_id"],
                   schema="saea-execution-plan-v1", split_identity_hash=i["split_identity_hash"], study="speech-aware-evidence-acquisition")
        route = ("Shared core.flight.cli worker on this cell alone (the smoke step); never scored." if key == "smoke" else
                 f"Shared core.flight.cli campaign over {i['cell']} (configs/campaign/{fam.CELLS['lock']['cell']}.json lists lock, aa, ab in chain order) on the port "
                 f"deployment ({pf.DEPLOYMENT_REF}); one official score over the complete manifest after the seal. The port chain script is not written yet (open item).")
        docs[fam.plan_file_name(key, "generation")] = {
            **base, "execution_plan": gen, "execution_route": route, "plan_kind": "generation", "schema": "saea-registered-generation-plan-v1",
            "purpose": f"{i['run_id']}: {spec['role']} of the three-way comparison, NPU port.",
            "scoring": "None: the smoke is never scored." if key == "smoke" else "One first complete official score after a successful seal; no partial score."}
        fit = dict(gen, attempt_id=i["fit_attempt_id"], run_id=i["fit_run_id"], planned_model_calls=i["family_budget"]["fit_endpoint_calls_cap"])
        docs[fam.plan_file_name(key, "fit")] = {
            **base, "execution_plan": fit, "plan_kind": "fit", "schema": "saea-registered-fit-plan-v1", "attached_generation_run_id": i["run_id"],
            "contact_cap": i["fit"]["contact_cap"], "fit_path": fit_path_block(),
            "execution_route": "The shared live phase-fit path (core.preparation_fit.build_direct_only_raw_fit_gate) under this cell's own live_phase_fit identity; request envelopes only, no generation.",
            "purpose": f"{i['fit_run_id']}: the live phase-fit spend identity of {i['run_id']}; request envelopes only; reads nothing.", "scoring": "No fit score."}
    return docs


def scope_text(key: str, ids: Mapping[str, Mapping]) -> str:
    spec = fam.CELLS[key]
    common = (f"served by the BF16 / vLLM-Ascend Thinking core over the original recordings ({pf.PROMPT_ORDER}); arm D of the three-way comparison "
              f"({fam.PLAN} section 5); {fam.DIFFERENCES_TEXT}")
    form = (f"the sealed direct-only form (program {fam.PROGRAM}, one contact per row, task instruction probes/slurp_direct_closed_list.txt, no knowledge supply)")
    if key == "smoke":
        return (f"DRAFT -- REGISTERED. NPU-port twin of SAEA-SLURP-DIRECT-ONLY-FOLD-B-THINKING-SMOKE-4-V1 (cell 266-E). Four-row decode-cost smoke of {form}, TRAIN Fold B 4, "
                f"{common}; no score, no read")
    if key == "lock":
        return (f"DRAFT -- REGISTERED. NPU-port twin of SAEA-SLURP-DIRECT-ONLY-FOLD-B-THINKING-V1 (cell 266-E Thinking arm). {form[0].upper()}{form[1:]} on TRAIN Fold B 1,080, {common}; "
                "pass lock (1 of 3 request-identical passes: lock, aa, ab; noise model M1 replicates); a capability read, no mechanism, no claim")
    return (f"DRAFT -- REGISTERED. Request-identical pass {key} of {ids['lock']['run_id']} (same baseline form, model, dataset and decoding; only the run, attempt and FIT identity differ), "
            f"flown in the same chain right after the previous pass; {'the within-chain A/A null of arm D (2 of 3 passes)' if key == 'aa' else 'the second replicate (3 of 3 passes), giving k = 2 replicates of lock for the replicate variance term of the CI (noise model M1)'}; "
            f"{form}, TRAIN Fold B 1,080, {common}; a capability read, no mechanism, no claim")


def draft_rows(ids: Mapping[str, Mapping], commit: str, at: str, shas: Mapping[str, str]) -> list[list[str]]:
    rows = []
    for key in fam.ALL_KEYS:
        i = ids[key]
        for fit in (False, True):
            name = fam.plan_file_name(key, "fit" if fit else "generation")
            budget = i["family_budget"]
            calls = budget["fit_endpoint_calls_cap" if fit else "generation_calls_cap"]
            cap_text = (f"FIT cap = {i['rows']} rows x 1 contact x 1 resident counting endpoint (core.preparation_fit.build_direct_only_raw_fit_gate, direct_only_fit_contact_cap); "
                        "the speech and gpu cells are the generation row's ceilings, shared and not additive" if fit else
                        f"generation cap {i['rows']} x ({fam.CONTACTS_PER_ROW} contact + {fam.FALLBACK_CONTACTS_PER_ROW} answer-only fallback); speech cap ceil({pdc_audio(key)} x "
                        f"{fam.GENERATION_CONTACTS_PER_ROW})")
            rows.append([at[:10], i["fit_run_id"] if fit else i["run_id"], i["execution_profile"], scope_text(key, ids) + (" -- FIT spend identity" if fit else ""),
                         fam.model_text(), f"{CARRIER} {i['rows']}", "discovery", i["split_identity_hash"], "no", str(calls), str(budget["repeated_speech_audio_seconds_cap"]),
                         fam.ledger_gpu_cell(key), INHERITED, f"{PKG}/{name}",
                         (f"DRAFT at study commit {commit}; plan SHA256 {shas.get(name, 'pending')}; composed config {i['composed_config_hash']}, protocol {i['protocol_hash']}; "
                          f"{cap_text}; drafted by the agent on owner decision 2026-10-28 item 4, owner countersignature pending; the decision receipt {fam.DECISION_RELPATH} is not "
                          f"minted; the gpu-hours cell is the PROVISIONAL card-hour formula ({fam.CARD_HOUR_FORMULA[key]}) and refuses the exposure gate until the measured sweep "
                          f"fills {pf.DEPLOYMENT_REF}; nothing authorised.")])
    return rows


def pdc_audio(key: str) -> str:
    return f"{fam.roster_audio(fam.CELLS[key]['roster']):g}"


def countersign_texts(ids: Mapping[str, Mapping]) -> dict:
    head = ("REGISTERED BEFORE CONTACT (root countersigned <UTC timestamp> by the owner on the port brief: " + pf.OWNER_DECISIONS + "; the NPU-port twin of the 266-E direct-only "
            "Thinking arm on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold B (arm D of the three-way comparison); deployment <the filled PORT_DEPLOYMENT_PROFILE: instances, "
            "tensor-parallel size, sequences per instance, devices>, card-hour bound <the re-derived number> written into the gpu-hours cell; ")
    tail = f"; the decision receipt {fam.DECISION_RELPATH} is minted by the main thread after this countersignature)"
    rows = {}
    for key in fam.ALL_KEYS:
        i = ids[key]
        rows[i["run_id"]] = head + fam.CELLS[key]["role"] + tail
        rows[i["fit_run_id"]] = head + f"live phase-fit identity of the {fam.CELLS[key]['role']}" + tail
    return {"_note": ("Before countersigning: replace each gpu-hours cell's PROVISIONAL formula with the number re-derived from the measured sweep (and the same number in the "
                      "experiment's planned_gpu_hours / gpu_hours_fit_inclusive and the draft receipt's budget), then replace the leading 'DRAFT -- REGISTERED' scope marker and the "
                      "'DRAFT' note prefix of each row with the text below and mint the receipt."),
            "decision_id": fam.DECISION_ID, "decision_receipt": fam.DECISION_RELPATH, "rows": rows, "schema": "saea-countersign-row-texts-v1"}


def ledger_block(rows: list[list[str]], intro: str, mark: str) -> str:
    cols = reg.LEDGER_COLUMNS
    lines = [mark, "", intro, "", "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += [reg.table_row(cells) for cells in rows]
    return "\n".join(lines) + "\n"


def readme_text(ids: Mapping[str, Mapping], commit: str) -> str:
    table = "\n".join(
        f"| `{ids[k]['cell']}` | `{ids[k]['run_id']}` | `{ids[k]['fit_run_id']}` | {ids[k]['rows']} | {ids[k]['family_budget']['generation_calls_cap']} | "
        f"{ids[k]['family_budget']['fit_endpoint_calls_cap']} | {ids[k]['family_budget']['repeated_speech_audio_seconds_cap']} | `{ids[k]['composed_config_hash'][:12]}...` | "
        f"`{ids[k]['protocol_hash'][:12]}...` |" for k in fam.ALL_KEYS)
    diffs = "\n".join(f"- {d}" for d in fam.DECLARED_DIFFERENCES)
    return (
        "# D NPU port: direct-only on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold B (DRAFT / HELD)\n\n"
        f"The NPU-port twin of the 266-E direct-only Thinking arm (historical package `{fam.HISTORICAL_PACKAGE}`), arm D of the three-way comparison "
        f"(`{fam.PLAN}`, sections 5 and 9 ruling 6), registered before contact on {fam.DATE} at study commit `{commit}` by "
        "`scripts/flights/direct/direct_port_write_registration.py` from the cells `scripts/flights/direct/direct_port_derive_cells.py` derives, with the port settings "
        "imported from `scripts/flights/nx/l338_port_family.py`. Text draft: `" + fam.DRAFT + "`. Authority: " + pf.OWNER_DECISIONS + ". Development numbers only; "
        "never a test claim; claims_benefit false.\n\n"
        "## Cells (chain order)\n\n"
        "| cell | run | FIT run | rows | generation calls | FIT calls | speech-seconds | composed config | protocol |\n|---|---|---|---|---|---|---|---|---|\n"
        f"{table}\n\n"
        f"Cell total: {fam.CELL_TOTAL[0]:,} generation calls, {fam.CELL_TOTAL[1]:,} FIT calls, {fam.CELL_TOTAL[2]:,} speech-seconds. One Solver contact per row per pass (program "
        f"`{fam.PROGRAM}`), plus one answer-only fallback contact per row in the plan ceiling; the FIT cap is rows x 1 x 1 counting endpoint, the value "
        "`core.preparation_fit.build_direct_only_raw_fit_gate` demands. The three passes `lock`, `aa`, `ab` are request-identical: their baselines differ only in the "
        "`live_phase_fit` identity (a FIT run is a one-time spend identity) and the prose.\n\n"
        "## The FIT path\n\n"
        f"{fit_path_block()['gate']}; cap rule `{fit_path_block()['cap_rule']}`. It landed in commit 56d3c1c3 (`tests/unit/test_direct_only_fit.py`), so the FIT cap is no longer open.\n\n"
        "## Serving: protocol facts and the deployment placeholder\n\n"
        f"Model fragment `{fam.MODEL_FRAGMENT}` (transport vLLM, BF16). Protocol facts: the per-sequence window of {pf.PER_SEQUENCE_WINDOW_TOKENS:,} tokens, reasoning budget "
        f"{pf.fam.DELIVERING_BUDGET} / cap {pf.fam.COMPLETION_CAP}, greedy decoding seed {pf.fam.LOCKED_DECODING['seed']}, prompt order `{pf.PROMPT_ORDER}`. The DEPLOYMENT is the one named "
        f"placeholder `PORT_DEPLOYMENT_PROFILE` ({pf.PORT_DEPLOYMENT_PROFILE['filled_by']}); no instance count, sequence count or tensor-parallel size is a fact of this registration. "
        "Session identity, the vLLM runtime receipt and the lock entries are open owner decisions 9.2-9.4. If owner item R1 (thinking budget 6,144) is adopted, D is re-pointed to "
        "the budget6144 fragment together with A and T and re-written; the calls, FIT and audio caps do not move with the budget, the card-hour bound does.\n\n"
        "## Accelerator bound (PROVISIONAL)\n\n"
        f"- each pass: {fam.CARD_HOUR_FORMULA['arm']}.\n- smoke: {fam.CARD_HOUR_FORMULA['smoke']}.\n"
        "- The experiments' numeric planned_gpu_hours (4.5 / 0.5) are the 266-E placeholders the plan field requires; the ledger gpu-hours cells carry the formula, so the exposure gate "
        "refuses these rows (\"must be a number\") until the sweep re-derives the bound.\n\n"
        "## Declared differences (design doc section 8, and D1)\n\n"
        f"{diffs}\n\n"
        "8.4, 8.6 and 8.7 concern retrieval and knowledge, which D does not use. D1: the historical D had an 8,192-token completion cap and no thinking budget; the port D runs under the "
        "port budget with forced close and one answer-only fallback, so rows that delivered at 8,192 may now be force-closed (36 of 1,080 266-E rows ran to the cap; delivering rows "
        "had p90 1,641 and p99 3,610 completion tokens). Read the truncation rate from the smoke before the flight.\n\n"
        "## Read rule\n\n"
        f"Primary: {READ_RULE['primary']}. Replicates: lock, aa, ab (k = 2 replicates of lock). {READ_RULE['twin_admissibility']}. {READ_RULE['a_to_d_gap']}. {READ_RULE['truncation']} "
        "The D reader and recipe are an open item (this package carries the rule, not the recipe).\n\n"
        "## Files\n\n"
        "`protocol-registered-v1.json`, the eight plan files (`<smoke|lock|aa|ab>-<generation|fit>-plan-registered-v1.json`), `registered-facts-v1.json` (stage3_measurement blocks "
        "keyed by the port run ids), `signoff-conditions-v1.json` (smoke machine sign-off conditions; the D smoke table writer is an open item), `root-admission-draft-v1.json` "
        f"(decision `{fam.DECISION_ID}`, relpath `{fam.DECISION_RELPATH}`, eligible run ids smoke, smoke FIT, lock, lock FIT, aa, aa FIT, ab, ab FIT), `countersign-row-texts-v1.json`, "
        "`draft-ledger-rows.md` (the eight DRAFT rows).\n\n"
        "## What the main thread does (in order)\n\n"
        "1. Commit this package and the cells. Write D's plans after the R1 decision, so a budget move does not force a second registration.\n"
        "2. Run the speed sweep; fill `PORT_DEPLOYMENT_PROFILE`; re-derive the card-hour bound and write it into planned_gpu_hours, the ledger gpu-hours cells and the draft receipt's budget; "
        "re-run the derive and this writer.\n"
        "3. Owner countersigns the rows (`countersign-row-texts-v1.json`); the main thread mints the decision receipt.\n"
        "4. Re-pin and flip with `scripts/flights/repin_cell.py` from a clean tree; the direct-only program supplies no first phase, so no prepared artifact is created "
        "(`--allow-output-change` is never passed).\n"
        "5. Open items: the D smoke reader and read recipe, the port chain script, the session identity, runtime receipt and lock entries (9.2-9.4).\n")


def write_package(root: Path, commit: str, at: str | None, data_root: Path, *, append_ledger: bool = False, derive: bool = True, verify_audio: bool = True) -> dict:
    root = Path(root)
    facts_path = root / PKG / "registered-facts-v1.json"
    at = at or (json.loads(facts_path.read_text(encoding="utf-8"))["registered_at_utc"] if facts_path.is_file() else datetime.now(timezone.utc).isoformat())
    derived = pdc.derive(root) if derive else None
    ids = identities(root)
    proof = request_identity_proof(root)
    audio = audio_pin_check(Path(data_root)) if verify_audio else None
    lock = ids["lock"]
    roster = fam.roster_doc("arm")
    smoke_roster = fam.roster_doc("smoke")
    measurement = measurement_blocks(ids)
    protocol = {
        "cells": {k: {kk: v[kk] for kk in ("attempt_id", "cell", "composed_config_hash", "experiment", "fit_attempt_id", "fit_run_id", "protocol_hash", "rows", "run_id",
                                           "split_identity_hash", "historical_run_id")} for k, v in ids.items()},
        "blockers": [],
        "findings": [
            {"id": "fit-path-landed", "finding": "The raw-arm direct-only FIT path is core.preparation_fit.build_direct_only_raw_fit_gate (commit 56d3c1c3); the cells carry its cap, rows x 1 x 1.",
             "remedy": "none"},
            {"id": "fit-identity-per-pass", "finding": ("Each pass has its own live_phase_fit run and attempt (a FIT run is a one-time spend identity), so lock, aa and ab have three baseline "
                                                        "fragments that differ only in that identity and the prose; the earlier draft's single shared lock baseline would have reused the lock's FIT run."),
             "remedy": "none; the request-identity proof is recorded in registered-facts"},
            {"id": "budget-d1", "finding": fam.D1_DIFFERENCE, "remedy": "read the truncation rate from the smoke; re-point to the budget6144 fragment with A and T if R1 is adopted"},
            {"id": "deployment-is-a-placeholder", "finding": pf.PORT_DEPLOYMENT_PROFILE["not_a_fact_here"], "remedy": "the measured speed sweep fills the profile"},
            {"id": "twin-count", "finding": ("The plan's '2 contacts per row' counts the twin only; with the second replicate (noise model M1) the cell makes three request-identical passes, "
                                              "3 programme contacts per Fold B row."), "remedy": "none"}],
        "failure_policy": json.loads((root / "configs" / "experiment" / f"{lock['experiment']}.json").read_text(encoding="utf-8"))["failure_policy"],
        "declared_differences": list(fam.DECLARED_DIFFERENCES), "serving": serving_block(), "accelerator_bound": cost_block(), "read_rule": READ_RULE,
        "prohibitions": [
            "No system-level, superiority, generalisation or admission claim from this read; the Thinking model is admitted ONLY for this read.",
            "No reasoning text delivered, persisted (beyond hashes) or scored; preserve_wire_responses stays false on every cell.",
            "No pass before the four-row smoke has sealed and the machine sign-off conditions hold.",
            "No deployment value (instances, tensor-parallel size, sequences per instance, devices) is registered here; the cell is not flown before PORT_DEPLOYMENT_PROFILE "
            "is filled from the measured sweep and the card-hour bound re-derived.",
            "No test-split, Fold A or DEV contact; no flight before root countersigns the DRAFT rows, the receipt is minted and the cells are flipped; --allow-output-change is never passed."],
        "protocol_id": f"{fam.PKG_NAME.replace('-v1', '')}-protocol-v1", "registered_at_utc": at, "schema": "saea-registered-protocol-v1", "status": STATUS,
        "study_commit_at_registration": commit,
        "the_question": ("What does the bare direct model (one Solver call per row, no knowledge, no program) score on TRAIN Fold B 1,080 on the BF16 / vLLM-Ascend Thinking core, "
                         "read in three request-identical passes so the decoding noise is measured, beside the agent system A and the TFRL arm T of the three-way comparison?")}
    protocol_sha = reg.wjson(root, f"{PKG}/protocol-registered-v1.json", protocol)
    plans = plan_docs(ids, commit, at, protocol_sha)
    shas = {name: reg.wjson(root, f"{PKG}/{name}", doc) for name, doc in plans.items()}
    shas["protocol-registered-v1.json"] = protocol_sha
    signoff = fam.signoff_conditions()
    shas["signoff-conditions-v1.json"] = reg.wjson(root, f"{PKG}/signoff-conditions-v1.json", signoff)
    facts = {
        "_note": ("Every model-free fact this DRAFT port registration rests on, and the canonical stage3_measurement blocks keyed by the port run ids "
                  "(scripts/checks/stage3_effect_floor_check.py, scripts/checks/flight_registration_check.py)."),
        "accounting": {"program": fam.PROGRAM, "arm": fam.ARM, "contacts_per_row": fam.CONTACTS_PER_ROW, "fallback_contacts_per_row": fam.FALLBACK_CONTACTS_PER_ROW,
                       "passes": list(fam.PASS_KEYS), "rows": fam.ROWS, "caps": cost_block()["caps"], "cell_total": cost_block()["cell_total"], "accelerator_bound": cost_block()},
        "audio_pins": audio, "composed_config_hashes": {k: v["composed_config_hash"] for k, v in ids.items()},
        "declared_differences": list(fam.DECLARED_DIFFERENCES), "derive": derived, "fit_path": fit_path_block(),
        "identity": ids, "locked_decoding": dict(pf.fam.LOCKED_DECODING),
        "pins": {"model": fragment_pins(root), "serving": serving_block(), "dataset_fragment": fam.DATASET,
                 "knowledge_view": {"relpath": pdc.fam.dfam.KNOWLEDGE_SNAPSHOT_RELPATH, "sha256": pdc.fam.dfam.KNOWLEDGE_SNAPSHOT_SHA256,
                                    "note": "the fold-excluded view the direct-only baseline pins for the runner's roster guard only; the program supplies no evidence"},
                 "signoff_conditions": {"path": f"{PKG}/signoff-conditions-v1.json", "sha256": shas["signoff-conditions-v1.json"]}},
        "prohibitions": protocol["prohibitions"], "read_rule": READ_RULE, "registered_at_utc": at,
        "registration_basis": f"{pf.OWNER_DECISIONS}; {pf.DESIGN_DOC}; {fam.PLAN} section 9. DRAFT: nothing here authorises a flip or a flight.",
        "request_identity_proof": proof,
        "roster": {"arm": roster["roster_id"], "count": len(roster["items"]), "receipt": {"relpath": fam.ROSTER["arm"], "sha256": _sha(REPO_ROOT / fam.ROSTER["arm"])},
                   "smoke_receipt": {"relpath": fam.ROSTER["smoke"], "sha256": _sha(REPO_ROOT / fam.ROSTER["smoke"]),
                                     "ids_are_first_four_of_arm_roster": [i["sample_id"] for i in roster["items"][:4]] == [i["sample_id"] for i in smoke_roster["items"]]},
                   "split_identity_hash": roster["split_identity_hash"], "smoke_split_identity_hash": smoke_roster["split_identity_hash"],
                   "audio_seconds_total": roster["audio_seconds_total"]},
        "rows": fam.ROWS, "schema": "saea-registered-facts-v1", "stage3_measurement": measurement, "study_commit_at_registration": commit, "status": "DRAFT-HELD"}
    shas["registered-facts-v1.json"] = reg.wjson(root, f"{PKG}/registered-facts-v1.json", facts)
    draft = {
        "_draft_note": (f"A DRAFT for the MAIN THREAD to mint on the data root (owner decision 2026-10-28 item 5; the mint tool is the D analogue of {fam.TOOL_MINT}, not written yet). It "
                        "authorises nothing while its status is not 'decided'. Before minting, the measured sweep fills PORT_DEPLOYMENT_PROFILE and the PROVISIONAL accelerator bound "
                        "below is replaced by the re-derived number."),
        "antecedents": {"protocol": {"path": f"{PKG}/protocol-registered-v1.json", "sha256": protocol_sha},
                        "registered_facts": {"path": f"{PKG}/registered-facts-v1.json", "sha256": shas["registered-facts-v1.json"]},
                        "plans": {name: {"path": f"{PKG}/{name}", "sha256": shas[name]} for name in plans}, "registration_base_commit": commit},
        "authorization_basis": f"{pf.OWNER_DECISIONS}; {pf.DESIGN_DOC}; {fam.PLAN} section 9; historical authority owner decision 266-E ({fam.HISTORICAL_PACKAGE}).",
        "budget": {k: ids[k]["family_budget"] for k in fam.ALL_KEYS},
        "accelerator_bound": {"status": "PROVISIONAL", "formula": {k: fam.CARD_HOUR_FORMULA[k] for k in fam.ALL_KEYS},
                              "note": "gpu_hours_fit_inclusive in the budget is the 266-E numeric placeholder, not the NPU card-hour bound"},
        "claim_boundary": "No benefit, system-level, superiority, generalisation, held-out, confirmatory or paper-scale claim.",
        "decision": ("Fly the NPU-port twin of the 266-E direct-only Thinking arm on the BF16 / vLLM-Ascend Thinking core on TRAIN Fold B: four-row smoke, then three request-identical "
                     "1,080-row passes (lock, aa, ab), on the deployment PORT_DEPLOYMENT_PROFILE once it is filled."),
        "decision_id": fam.DECISION_ID, "eligible_attempt_ids": fam.eligible_attempt_ids(), "eligible_run_ids": fam.eligible_run_ids(),
        "not_authorized": ["any flight before the cells are flipped and the receipt is minted", "any flight before PORT_DEPLOYMENT_PROFILE is filled",
                           "starting or touching any serving session outside the registered deployment", "any ASR, transcript, encoder or paid-API contact",
                           "--allow-output-change on the re-pin tool", "persisting reasoning text or raw wire responses", "Fold A, DEV or test-split contact"],
        "owner": "root", "population": {"count": fam.ROWS, "fold_a_contact_authorized": False, "split_identity_hash": lock["split_identity_hash"], "split_role": "discovery",
                                        "test_contact_authorized": False},
        "recorded_at_utc": None, "registered_hashes": {k: {"composed_config": ids[k]["composed_config_hash"], "protocol": ids[k]["protocol_hash"]} for k in fam.ALL_KEYS},
        "schema": "saea-campaign-decision-v1",
        "service": {"expected_session_id": fam.SERVICE_SESSION, "model_fragment": fam.MODEL_FRAGMENT, "deployment_profile": "PORT_DEPLOYMENT_PROFILE (pending)", "no_child_start_stop": True},
        "status": "DRAFT-NOT-MINTED"}
    shas["root-admission-draft-v1.json"] = reg.wjson(root, f"{PKG}/root-admission-draft-v1.json", draft)
    rows = draft_rows(ids, commit, at, shas)
    ledger_text = ledger_block(rows, (
        f"8 unconsumed DRAFT rows (generation and live-phase-fit identity of the smoke and of the lock, aa and ab passes) for arm D of the three-way comparison ({fam.PLAN} "
        f"section 5), drafted under owner decision 2026-10-28 item 4 (the agent drafts the exposure preregistration rows). Package {PKG}. No row is countersigned and no row authorises a "
        "contact. Each gpu-hours cell is the PROVISIONAL NPU card-hour formula, not a number: the exposure gate refuses these rows until the measured speed sweep fills "
        "PORT_DEPLOYMENT_PROFILE. The FIT rows' calls carry the code rule for a raw arm (rows x 1 contact x 1 counting endpoint); the FIT path is "
        "core.preparation_fit.build_direct_only_raw_fit_gate and is no longer open. No serving row: the deployment is not registered yet."), fam.LEDGER_MARK + " (DRAFT, owner countersignature pending)")
    shas["draft-ledger-rows.md"] = reg.wr(root, f"{PKG}/draft-ledger-rows.md", ledger_text)
    shas["countersign-row-texts-v1.json"] = reg.wjson(root, f"{PKG}/countersign-row-texts-v1.json", countersign_texts(ids))
    shas["README.md"] = reg.wr(root, f"{PKG}/README.md", readme_text(ids, commit))
    prereg = (f"# D NPU port: direct-only on the BF16 / vLLM-Ascend Thinking core, preregistration (DRAFT / HELD)\n\nRegistered {at[:10]} at study commit `{commit}` by script. "
              "Development numbers only.\n\n"
              + "Runs: " + "; ".join(f"`{ids[k]['run_id']}` / `{ids[k]['fit_run_id']}`" for k in fam.ALL_KEYS) + ".\n\n"
              f"The package is `{PKG}/`; the read rule is the `read_rule` block of its `registered-facts-v1.json`.\n\n## The measurement declaration\n\n```json\n"
              + json.dumps({"stage3_measurement": measurement}, indent=2, ensure_ascii=False, sort_keys=True) + "\n```\n")
    shas["__prereg__"] = reg.wr(root, fam.PREREG, prereg)
    result = {"package": PKG, "shas": shas, "derive": derived, "ledger_appended": None,
              "hashes": {k: {"composed_config": v["composed_config_hash"], "protocol": v["protocol_hash"]} for k, v in ids.items()}}
    if append_ledger:
        ledger = root / reg.LEDGER
        current = io.open(ledger, encoding="utf-8", newline="").read()
        if fam.LEDGER_MARK in current:
            result["ledger_appended"] = False
        else:
            io.open(ledger, "w", encoding="utf-8", newline="").write(current.rstrip("\n") + "\n\n" + ledger_text)
            result["ledger_appended"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--study-commit", required=True)
    parser.add_argument("--registered-at", default=None, help="default: the stamp already in registered-facts-v1.json, else now")
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--data-root", default=os.environ.get("SAEA_DATA_ROOT") or os.environ.get("SPEECHRL_DATA_DIR"), required=False)
    parser.add_argument("--append-ledger", action="store_true", help="append the DRAFT block at the END of docs/exposure-ledger.md (idempotent)")
    parser.add_argument("--no-derive", action="store_true")
    parser.add_argument("--no-audio-check", action="store_true", help="skip re-hashing the 1,080 recordings (tests)")
    args = parser.parse_args(argv)
    if not args.data_root:
        parser.error("--data-root (or SAEA_DATA_ROOT / SPEECHRL_DATA_DIR) is required to re-check the audio pins")
    result = write_package(Path(args.root), args.study_commit, args.registered_at, Path(args.data_root), append_ledger=args.append_ledger, derive=not args.no_derive,
                           verify_audio=not args.no_audio_check)
    print(json.dumps({k: v for k, v in result.items() if k != "derive"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
