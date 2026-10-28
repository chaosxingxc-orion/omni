#!/usr/bin/env python3
"""Write the DRAFT readiness package of the NPU-port twin of L338 (the lever column, TRAIN Fold B) from the port fragments.

Config and receipt-pinned bytes only: no adapter contact, no server, no model, tokenizer, encoder, embedder or scorer contact. Flips nothing, mints
nothing, countersigns nothing, and writes NO row into ``docs/exposure-ledger.md`` unless ``--append-ledger`` is passed. Every file says DRAFT / HELD.

ONE package ``docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/`` (README, ``registered-facts-v1.json`` with the canonical
``stage3_measurement`` blocks keyed by the port run ids, ``protocol-registered-v1.json``, the four plan files, ``signoff-conditions-v1.json``,
``root-admission-draft-v1.json``, ``countersign-row-texts-v1.json`` and ``draft-ledger-rows.md``) and, beside it, the preregistration. The ledger rows are
two blocks: the four DRAFT preregistration rows of the port smoke and flight (generation and FIT each; owner 2026-10-28 item 4) and the six
model-free-check rows of the port preparations the flight pins (item 6), each fact of which is re-read from its receipt on the data root here.

The deployment (instances, tensor-parallel size, sequences per instance, devices) is the placeholder ``PORT_DEPLOYMENT_PROFILE`` and the accelerator bound
is a PROVISIONAL card-hour formula (``l338_port_family``); the ledger's gpu-hours cells carry that formula, NOT a number, so the exposure gate refuses the
rows ("must be a number") until the sweep fills the profile and the rows are re-written and countersigned.

    python -B scripts/flights/nx/l338_port_write_registration.py --study-commit <sha> --data-root <data root> [--registered-at <UTC>]
        [--root DIR] [--append-ledger] [--no-derive]
"""

from __future__ import annotations

import argparse
import copy
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

import l338_family as fam  # noqa: E402
import l338_port_derive_cells as pdc  # noqa: E402
import l338_port_family as pf  # noqa: E402
import l338_write_registration as hw  # noqa: E402  (the historical writer: arm table, word rules, measurement blocks, scope text)

REPO_ROOT = pf.REPO_ROOT
reg = hw.reg
from speech_aware_evidence_acquisition.core.config import compose, protocol_hash  # noqa: E402

STATUS = reg.STATUS
PKG = pf.PACKAGE
INHERITED = ("TRAIN Fold B discovery rows already exposed by prior SLURP cells (the historical L338 flight and its smoke included); this cell opens no "
             "split. Prior exposure is inherited in full.")
MF_MODEL = "no model or tool contact; local arithmetic over receipt-pinned files"
FOLD_B_CARRIER = "slurp TRAIN Fold B SLURP-EVO-FOLD-B-V2 1080"
FOLD_B_SPLIT = "4407794c4a5d2e8883d93fdf242deb33c0987405d62e05a7d8416f5251a3484a"
MF_TAIL = "It constructs no ExecutionPlan, reserves no contact, releases no store beyond the one named, re-pins no committed config and authorises nothing."

# ---- the port preparations the flight pins (model-free; facts re-read from the receipts) -------------------------------------------------------------
PREPARATIONS = {
    "alias": {"receipt": "prepared/SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-B-ALIAS-NPU-PORT-V1/query-dimension-alias.json",
              "receipt_sha256": "f59dc040b05e2e87637a91d8d677392c50a010701986a8db33bbdfdfdd008adf",
              "artifact": "vectors/slurp-intent-boundaries-train-fold-b-1080-npu-port-v1/query-vectors-speech-segments.jsonl",
              "artifact_sha256": "fe23703a429d4a655b51d0cee921f28eba0e5be8a5511d9fe0f13dc73eb28181"},
    "content": {"receipt": "vectors/slurp-stage3-filler-fold-b-npu-port-v1/query-vectors-speech-content-v1.provenance.json",
                "receipt_sha256": "59c285768247d3f6abb0fea68dfbd5e2f0246074038d7318218f834e232c890c",
                "artifact": "vectors/slurp-stage3-filler-fold-b-npu-port-v1/query-vectors-speech-content-v1.jsonl",
                "artifact_sha256": "e1d3230ed517685f6d0e963bb48a0a5bf205e3c31a7c7aaae22e88b1ac27196f"},
    "minus_r01": {"receipt": "knowledge/slurp/saea-kb-distmodel-v1/size-pool-annotated-exemplars-minus-r01/release.json",
                  "receipt_sha256": "83620eb0b02f12424a00cd5572904f4df48a49c329b1b0e1a9321d4b9ec14fd0",
                  "artifact": "knowledge/slurp/saea-kb-distmodel-v1/size-pool-annotated-exemplars-minus-r01/snapshot.json",
                  "artifact_sha256": None},
    "centroid": {"receipt": "prepared/SAEA-SLURP-INTENT-BOUNDARY-CENTROID-KEYS-NPU-PORT-V1/receipt.json",
                 "receipt_sha256": "200f769444ca5e41381cf28151daa11cef05b86dc8d0eaf26ced8300e342d914",
                 "artifact": "prepared/SAEA-SLURP-INTENT-BOUNDARY-CENTROID-KEYS-NPU-PORT-V1/facets.jsonl",
                 "artifact_sha256": "f05287550e2ee895a6444b2e909db643df6f8d586c648eb7cc32c7b9f2c46d31"},
    "agreement": {"receipt": "prepared/SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-NPU-PORT-V1/receipt.json",
                  "receipt_sha256": "2e032013857b6e36c5c267fe32827d1982949515d6006cacb4ab568897358c92",
                  "artifact": "prepared/SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-NPU-PORT-V1/agreement.jsonl",
                  "artifact_sha256": "db4614a3f8f644c8514cd76bf596446cc9b7043eb222a97973eb8b807f113918"},
    "supply": {"receipt": "prepared/SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-NPU-PORT-V1/phase-supply-v1.json",
               "receipt_sha256": "1141f08c7cd4eb64574e38ecb95cffb3012057ad27ce2c62862c230c776aa630",
               "artifact": None, "artifact_sha256": None},
}
HISTORICAL_MINUS_R01_RELEASE_PIN = "docs/readiness/2026-10-07-slurp-simple-consumption-direction-v1/prepared-review-off-policy-v1.json"
HISTORICAL_MINUS_R01_SNAPSHOT_SHA = "6eaa03aea96a1928dd1ac29ecc71c3955c136498c9f8114266eaa830d1207c63"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_preparations(data_root: Path) -> dict:
    """Re-read every port preparation receipt (and its artifact when pinned) on the data root; refuse on any digest mismatch."""

    out = {}
    for key, item in PREPARATIONS.items():
        receipt = Path(data_root) / item["receipt"]
        if _sha(receipt) != item["receipt_sha256"]:
            raise SystemExit(f"{item['receipt']} moved: sha256 is not {item['receipt_sha256']}")
        if item["artifact_sha256"] and _sha(Path(data_root) / item["artifact"]) != item["artifact_sha256"]:
            raise SystemExit(f"{item['artifact']} moved: sha256 is not {item['artifact_sha256']}")
        doc = json.loads(receipt.read_text(encoding="utf-8"))
        out[key] = {"doc": doc, **item}
    alias = out["alias"]["doc"]["output"]
    if (alias["row_count"], alias["vector_width"], alias["max_absolute_delta"], alias["values_unchanged"], alias["output_sha256"]) != (
            1080, 2048, 0.0, True, PREPARATIONS["alias"]["artifact_sha256"]):
        raise SystemExit("the alias receipt does not state the sealed alias")
    content = out["content"]["doc"]
    if (content["parent_sha256"], content["sha256"], content["row_count"], content["exposure_ledger_row"]) != (
            PREPARATIONS["alias"]["artifact_sha256"], PREPARATIONS["content"]["artifact_sha256"], 1080, "SAEA-SLURP-FOLD-B-INPUTS-PREPARE-NPU-PORT-V1"):
        raise SystemExit("the content-derivative provenance does not chain from the alias")
    if out["minus_r01"]["doc"]["snapshot_sha256"] != HISTORICAL_MINUS_R01_SNAPSHOT_SHA:
        raise SystemExit("the minus-r01 release does not state the historical snapshot identity")
    out["minus_r01"]["snapshot_file_sha256"] = _sha(Path(data_root) / PREPARATIONS["minus_r01"]["artifact"])
    if out["centroid"]["doc"]["artifact"]["sha256"] != PREPARATIONS["centroid"]["artifact_sha256"]:
        raise SystemExit("the centroid receipt does not state the sealed key file")
    agreement = out["agreement"]["doc"]
    if agreement["artifact"]["sha256"] != PREPARATIONS["agreement"]["artifact_sha256"] or agreement["derived_from"]["cell"] != (
            "slurp-native-stage3-l338-lever-column-fold-b-comparator-npu-port-v1"):
        raise SystemExit("the agreement receipt does not state the sealed table from the port comparator")
    supply = out["supply"]["doc"]
    values = supply["output"]["values"]["perception"]
    out["supply"]["rows"] = len(values)
    out["supply"]["items_per_row"] = sorted({len(v) for v in values.values()})
    if supply["kind"] != "phase-supply" or supply["inputs"]["split_identity_hash"] != FOLD_B_SPLIT or len(values) != 1080:
        raise SystemExit("the phase supply does not cover the 1,080 Fold B rows")
    return out


def model_free_rows(prep: Mapping[str, dict], commit: str) -> list[list[str]]:
    status = f"RUN BEFORE THIS ROW WAS WRITTEN, MODEL-FREE (DRAFT, root countersignature pending) 2026-10-28 in the saea-base container; row written at study commit {commit} on branch npu-port-dev."
    alias, content, minus, centroid, agreement, supply = (prep[k] for k in ("alias", "content", "minus_r01", "centroid", "agreement", "supply"))
    a_out, c_doc, m_doc, k_doc, g_doc, s_doc = alias["doc"]["output"], content["doc"], minus["doc"], centroid["doc"], agreement["doc"], supply["doc"]
    k_meas, g_meas = k_doc["measured"], g_doc["measured"]
    no_new = "No new speech exposure: every vector read was sealed by SAEA-SLURP-TFRL-FOLD-B-SPEECH-QUERY-ENCODING-NPU-PORT-V1"
    rows = [
        ["2026-10-28", "SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-B-ALIAS-NPU-PORT-V1", "model-free-check",
         ("Port twin of the historical query-dimension alias SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-B-ALIAS-V1: carry the port's sealed TRAIN Fold B "
          f"content-dimension speech-query segment vectors ({a_out['source_sha256']}) into the intent-boundary retrieval dimension, values unchanged, segment "
          "identities preserved. Zero model, server, encoder, embedder, scorer, tokenizer, network, GPU and paid-API contacts."),
         MF_MODEL, FOLD_B_CARRIER, "discovery", FOLD_B_SPLIT, "no", "0", "0", "0",
         f"{no_new}. Prior exposure inherited in full.",
         f"{alias['receipt']} (sha256 {alias['receipt_sha256']})",
         (f"{status} Output {a_out['output_relpath']} SHA256 {a_out['output_sha256']} (schema {a_out['schema']}); implementation sha256 "
          f"{alias['doc']['inputs']['implementation_sha256']}. Measured: {a_out['row_count']:,} rows, {a_out['segment_count']:,} segments, width "
          f"{a_out['vector_width']}, max_absolute_delta {a_out['max_absolute_delta']}, values_unchanged true, 0 encoder / model / tokenizer contacts. {MF_TAIL}")],
        ["2026-10-28", "SAEA-SLURP-FOLD-B-INPUTS-PREPARE-NPU-PORT-V1", "model-free-check",
         ("The port's model-free CONTENT DERIVATIVE of the Fold B speech-query vectors (the row its provenance names): flatten the one-segment-per-sample "
          "alias output into flat sealed vectors in the content dimension (scripts/build_slurp_query_content_derivative.py), the exemplar channel's query "
          "source. Zero model, server, encoder, embedder, scorer, tokenizer, network, GPU and paid-API contacts."),
         MF_MODEL, FOLD_B_CARRIER, "discovery", FOLD_B_SPLIT, "no", "0", "0", "0",
         f"{no_new} and its model-free alias. Prior exposure inherited in full.",
         f"{content['receipt']} (sha256 {content['receipt_sha256']})",
         (f"{status} Parent {c_doc['parent_relpath']} SHA256 {c_doc['parent_sha256']}; output {PREPARATIONS['content']['artifact']} SHA256 {c_doc['sha256']} "
          f"(schema {c_doc['schema']}, {c_doc['source_dimension']} -> {c_doc['target_dimension']}); {c_doc['row_count']:,} rows, every float byte-identical "
          f"to the parent's single segment vector. {MF_TAIL}")],
        ["2026-10-28", "SAEA-SLURP-STAGE3-MINUS-R01-KNOWLEDGE-REBUILD-NPU-PORT-V1", "model-free-check",
         ("REBUILD on the port data root of the R01-excluded train-exemplars store (knowledge/slurp/saea-kb-distmodel-v1/size-pool-annotated-exemplars-minus-r01; "
          "builder of record scripts/build_slurp_stage3_r01_excluded_exemplar_store.py, a pure selection over sealed TRAIN metadata), the construction pool the "
          "port centroid keys average over. Zero model, server, encoder, embedder, scorer, tokenizer, network, GPU and paid-API contacts."),
         MF_MODEL, "slurp TRAIN construction pool (R01-excluded), metadata only", "discovery", m_doc["source_split_identity_hash"], "no", "0", "0", "0",
         "No speech exposure: the selection reads TRAIN metadata only. Prior exposure inherited in full.",
         f"{minus['receipt']} (sha256 {minus['receipt_sha256']})",
         (f"{status} Release SHA256 {minus['receipt_sha256']} equals the historical release pinned in {HISTORICAL_MINUS_R01_RELEASE_PIN}; snapshot_sha256 "
          f"{m_doc['snapshot_sha256']} equals the historical snapshot identity (snapshot file SHA256 {minus['snapshot_file_sha256']}); source "
          f"{m_doc['source']['relpath']} SHA256 {m_doc['source']['sha256']}; store {m_doc['store_id']}, builder policy {m_doc['builder_policy_id']}. "
          f"The historical build receipt is docs/receipts/knowledge/slurp-kb-size-pool-annotated-exemplars-minus-r01.json. {MF_TAIL}")],
        ["2026-10-28", "SAEA-SLURP-INTENT-BOUNDARY-CENTROID-KEYS-NPU-PORT-V1", "model-free-check",
         ("Port twin of SAEA-SLURP-INTENT-BOUNDARY-CENTROID-KEYS-V1: one unit-normalised mean construction-TRAIN utterance vector per released intent of the "
          "re-authored intent-boundaries-npu-port-v1 store, arithmetic over the port kus3 pool key vectors and the R01-excluded pool snapshot's frozen task_target "
          "labels (pins docs/readiness/2026-10-28-slurp-intent-boundary-centroid-keys-npu-port-v1/pins-v1.json); an intent with no pool member keeps its written "
          "key. Zero model, server, encoder, embedder, scorer, tokenizer, network, GPU and paid-API contacts."),
         MF_MODEL, "slurp TRAIN construction pool (R01-excluded), key side only", "discovery", m_doc["source_split_identity_hash"], "no", "0", "0", "0",
         "No new speech exposure: the builder reads no speech byte; the pool key vectors were sealed by SAEA-SLURP-MF7B-NEMOTRON-EMBEDDING-NPU-PORT-V1 and the "
         "written keys by SAEA-SLURP-INTENT-BOUNDARY-KEY-ENCODING-NPU-PORT-V1. Prior exposure inherited in full.",
         f"{centroid['receipt']} (sha256 {centroid['receipt_sha256']})",
         (f"{status} Artifact {centroid['artifact']} SHA256 {centroid['artifact_sha256']} (schema {k_doc['schema']}, kind {k_doc['kind']}); builder "
          f"scripts/build_slurp_intent_boundary_centroid_keys.py (sha256 {k_doc['builder_sha256']}); generated {k_doc['generated_at']}. Measured: "
          f"{k_meas['intents']} facets, {k_meas['centroid_facets']} centroids over {k_meas['members_total']:,} members, {k_meas['written_fallback_facets']} written "
          f"fallback ({', '.join(k_meas['written_fallback_ids'])}), key-to-key cosine mean {k_meas['key_to_key_cosine_mean']} / max {k_meas['key_to_key_cosine_max']} "
          f"(historical: mean 0.9326 / max 0.993). {MF_TAIL}")],
        ["2026-10-28", "SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-NPU-PORT-V1", "model-free-check",
         ("Port twin of SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-V1: the pre-computed, self-excluded exemplar-intent AGREEMENT table for TRAIN Fold B the "
          "port flight's boundary channel ranks with (w_agree 1.0), arithmetic over the pinned exemplar snapshot, the port pool key vectors and the port Fold B "
          "speech query vectors through the repository's own recall/rank/fuse, against the committed never-flown agreement-free port COMPARATOR cell "
          f"{g_doc['derived_from']['cell']} (source identity run {g_doc['derived_from']['run_id']}, program {g_doc['derived_from']['program_id']}; it declares "
          "no agreement_source, exactly as the historical FILLER BASE did). Zero model, server, encoder, embedder, scorer, network, GPU and paid-API contacts."),
         MF_MODEL, FOLD_B_CARRIER, "discovery", FOLD_B_SPLIT, "no", "0", "0", "0",
         f"{no_new}, its model-free alias and content derivative. No gold label, annotation or reference transcript of any Fold B row is opened. Prior exposure inherited in full.",
         f"{agreement['artifact']} (sha256 {agreement['artifact_sha256']}); {agreement['receipt']} (sha256 {agreement['receipt_sha256']})",
         (f"{status} Builder scripts/build_slurp_intent_exemplar_agreement.py (sha256 {g_doc['builder_sha256']}, the historical builder), 16 workers, vote store "
          f"train-exemplars / vote phase solver / target store intent-boundaries (scratch launcher work/scratch/comparator/build_downstream.sh); generated "
          f"{g_doc['generated_at']}. Measured: {g_meas['roster_rows']:,} rows, every row 20 votes, {g_meas['self_excluded_exemplars']} self-excluded exemplars, "
          f"{g_meas['rows_with_no_vote']} rows with no vote, {g_meas['pool_values_without_a_build_time_target']} pool values without a build-time target, "
          f"{g_meas['distinct_facets_ever_voted_for']} boundary facets ever voted for. {MF_TAIL}")],
        ["2026-10-28", "SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-NPU-PORT-PHASE-SUPPLY-V1", "model-free-check",
         ("The port flight's OWN prepared first-phase Perception supply (item E16), built model-free with core.flight.cli prepare-supply under the flight's run id "
          "SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-NPU-PORT-V1 from the port Fold B speech-query segment vectors, the port centroid key file and the port Fold B "
          "agreement table (rank policy saea-rank-wcos-prior-agree-v1, w_cos 1.0 / w_agree 1.0 / w_prior 0.0). This row registers the PREPARATION only; the "
          "flight's own preregistration row is separate. Zero model, server, encoder, embedder, scorer, tokenizer, network, GPU and paid-API contacts."),
         MF_MODEL, FOLD_B_CARRIER, "discovery", FOLD_B_SPLIT, "no", "0", "0", "0",
         f"{no_new} and the port model-free preparations above. Prior exposure inherited in full.",
         f"{supply['receipt']} (sha256 {supply['receipt_sha256']})",
         (f"{status} Receipt schema {s_doc['schema']}, kind {s_doc['kind']}, inputs config_sha256 {s_doc['inputs']['config_sha256']}, output_sha256 "
          f"{s_doc['output_sha256']}; {supply['rows']:,} Fold B rows, {', '.join(str(n) for n in supply['items_per_row'])} boundary candidates per row. "
          "The port registration moves the composed config (port FIT ids and texts), so this v1 supply is STALE for the registered cell: the main thread "
          "re-prepares it with scripts/flights/repin_cell.py as phase-supply-v2 under the same run id, output equality asserted (--allow-output-change never "
          f"passed), and amends this row. {MF_TAIL}")],
    ]
    return rows


# ---- identities, plans and texts --------------------------------------------------------------------------------------------------------------------


def identities(root: Path) -> dict:
    out = {}
    for key in pf.ALL_KEYS:
        spec = pf.CELLS[key]
        document = json.loads((Path(root) / "configs" / "experiment" / f"{spec['cell']}.json").read_text(encoding="utf-8"))
        cell = document["cells"][0]
        composed = compose(Path(root), cell["model_fragment"], cell["dataset_fragment"], cell["baseline_fragment"], spec["cell"])
        split = document["population"]["split_identity_hash"]
        out[key] = {
            "key": key, "experiment": spec["cell"], "cell": cell["cell_id"], "run_id": cell["run_id"], "attempt_id": cell["attempt_id"],
            "fit_run_id": spec["fit_run"], "fit_attempt_id": spec["fit_attempt"], "baseline_fragment": cell["baseline_fragment"],
            "dataset_fragment": cell["dataset_fragment"], "model_fragment": cell["model_fragment"], "arm": composed.values["arm"],
            "rows": document["population"]["count"], "split_identity_hash": split, "roster_receipt": document["population"].get("roster_receipt"),
            "composed_config_hash": composed.config_hash, "protocol_hash": protocol_hash(composed.config_hash, composed.values["arm"], split),
            "fit": dict(composed.values.get("native_contact_fit") or {}), "family_budget": document["family_budget"],
            "execution_plan": document["execution_plan"], "execution_profile": document["execution_profile"],
            "carrier_lock_key": composed.values["carrier_lock_key"], "program": composed.values["policy"], "session": document["service_session"],
            "historical_run_id": spec["historical_run"], "historical_fit_run_id": spec["historical_fit_run"],
        }
        if out[key]["fit"].get("run_id") != spec["fit_run"] or out[key]["fit"].get("attempt_id") != spec["fit_attempt"]:
            raise SystemExit(f"the composed {key} cell's native_contact_fit is not the port FIT identity")
    return out


def port_fit_proof() -> dict:
    """The historical context-fit proof restated per sequence: the slot count (a deployment value) is dropped; the window is the protocol fact."""

    proof = fam.fit_proof()
    proof.pop("slots", None)
    proof["per_sequence_window_tokens"] = pf.PER_SEQUENCE_WINDOW_TOKENS
    proof["sequences_per_instance"] = pf.DEPLOYMENT_REF
    if proof["slot_tokens"] != pf.PER_SEQUENCE_WINDOW_TOKENS or not proof["ok"]:
        raise SystemExit("the context-fit proof does not hold at the port's per-sequence window")
    proof["note"] += (" The port keeps the per-sequence window (--max-model-len 24576); the token counts are the historical llama.cpp tokenizer estimates and "
                      "observations, re-checked on every REAL request by the exact per-contact gate.")
    return proof


def serving_block() -> dict:
    return {"model_fragment": pf.MODEL_FRAGMENT, "transport": "vllm", "per_sequence_window_tokens": pf.PER_SEQUENCE_WINDOW_TOKENS,
            "deployment_profile": copy.deepcopy(pf.PORT_DEPLOYMENT_PROFILE), "session": pf.SERVICE_SESSION,
            "session_identity": "open owner decision 9.2 (design doc section 9)", "runtime_receipt": "open owner decision 9.3 (design doc section 9)",
            "lock_entries": "open owner decision 9.4 (design doc section 9)"}


def cost_block() -> dict:
    return {
        "status": "PROVISIONAL",
        "formula": dict(pf.CARD_HOUR_FORMULA),
        "inputs": {"arm_mix_rows": pf.ARM_MIX_ROWS, "margin": pf.GPU_MARGIN, "round_hours": pf.GPU_ROUND_H, "rows": fam.ROWS,
                   "smoke_wall_clock_hours": pf.SMOKE_WALL_CLOCK_HOURS},
        "unknowns_from_the_deployment_profile": ["T = tensor_parallel_size", "r = measured lock-only rows per minute of one instance at its sequences_per_instance",
                                                 "instances (wall-clock only; cancels in the card-hours)"],
        "illustration_not_registered": dict(pf.ILLUSTRATION_AT_HISTORICAL_RATE, note=(
            "at the historical llama.cpp lock rate the flight bound would read 90.3 x T card-hours; the port's r is measured, not assumed")),
        "plan_field_placeholder": {k: pf.CELLS[k]["plan_gpu_hours_placeholder"] for k in pf.ALL_KEYS},
        "plan_field_note": ("execution_plan.planned_gpu_hours and family_budget.gpu_hours_fit_inclusive carry the historical figures (90.3 / 0.5) only because the "
                            "plan field must be a number; they are NOT the NPU bound. The ledger gpu-hours cells carry the formula and fail the exposure gate "
                            "closed until the sweep re-derives the bound and the main thread re-writes plan, rows and receipt together."),
        "carried_over_caps": {k: {"generation_calls": pf.generation_calls(k), "fit_endpoint_calls": pf.fit_calls(k), "speech_audio_seconds": pf.audio_seconds(k)}
                              for k in pf.ALL_KEYS},
    }


def plan_docs(ids: Mapping[str, Mapping], commit: str, at: str, protocol_sha: str) -> dict:
    docs = {}
    arms_doc = {row["arm"]: {"role": row["role"], "axes": row["axes"]} for row in hw.arm_table_rows()}
    fit_block = port_fit_proof()
    for key in pf.ALL_KEYS:
        i, spec = ids[key], pf.CELLS[key]
        base = {"cell": i["cell"], "composed_config_hash": i["composed_config_hash"], "expected_session_id": pf.SERVICE_SESSION, "experiment": i["experiment"],
                "family_budget": i["family_budget"], "preparation_base_commit": commit,
                "prepared_phase_supply": {"relpath": spec["supply_relpath"], "sha256": "MINTED_AT_THE_FLIP",
                                          "note": ("this cell's own first-phase supply under its own run id (item E16), prepared or re-prepared by "
                                                   "scripts/flights/repin_cell.py at the flip" + (
                                                       "; the committed phase-supply-v1 (1141f08c...) predates this registration's config change and is "
                                                       "superseded by phase-supply-v2 (output equality asserted)" if key == "arm" else ""))},
                "protocol_document": {"relpath": f"{PKG}/protocol-registered-v1.json", "sha256": protocol_sha}, "registered_at_utc": at, "status": STATUS,
                "thinking_knobs": json.loads(json.dumps(fam.THINKING_KNOBS)), "no_think_node": None,
                "sampling_arms": json.loads(json.dumps(arms_doc)), "arm_order": list(fam.ARM_ORDER), "unused_budget_is_not_attempt_authority": True,
                "arm_visits": "first", "serving": serving_block(), "accelerator_bound": cost_block(),
                "historical_twin": {"run_id": i["historical_run_id"], "fit_run_id": i["historical_fit_run_id"], "package": pf.HISTORICAL_PACKAGE},
                "declared_differences": list(pf.DECLARED_DIFFERENCES),
                "one_factor": ("None within the cell beyond the historical sampling levers; against the historical twin the port changes the serving stack, the speech "
                               "input path, the retrieval vectors and the authored knowledge together (the declared differences), so it is read beside the "
                               "historical flight, never as a one-factor contrast with it.")}
        gen = dict(i["execution_plan"], attempt_id=i["attempt_id"], carrier_lock_key=i["carrier_lock_key"], protocol_hash=i["protocol_hash"],
                   run_id=i["run_id"], schema="saea-execution-plan-v1", split_identity_hash=i["split_identity_hash"], study="speech-aware-evidence-acquisition")
        route = ("Shared core.flight.cli worker on this cell alone (the smoke step); never scored." if key == "smoke" else
                 f"Shared core.flight.cli campaign over this cell on the port deployment ({pf.DEPLOYMENT_REF}); one official score per arm column "
                 "(stage3_rollout_score --mode perception-branches) after the seal. The port chain script is not written yet (open item).")
        docs[fam.plan_file_name(key, "generation")] = {
            **base, "execution_plan": gen, "execution_route": route, "plan_kind": "generation", "schema": "saea-registered-generation-plan-v1",
            "purpose": f"{i['run_id']}: {spec['role']} of L338, NPU port.",
            "scoring": "Per-arm official column scoring after a successful seal; no partial score." if key != "smoke" else "None: the smoke is never scored."}
        fit = dict(gen, attempt_id=i["fit_attempt_id"], run_id=i["fit_run_id"], planned_model_calls=i["family_budget"]["fit_endpoint_calls_cap"])
        docs[fam.plan_file_name(key, "fit")] = {
            **base, "execution_plan": fit, "plan_kind": "fit", "schema": "saea-registered-fit-plan-v1",
            "execution_route": "The shared core.flight.cli FIT path under this cell's own native_contact_fit ids; request envelopes only, no generation.",
            "context_fit": json.loads(json.dumps(fit_block)),
            "purpose": f"{i['fit_run_id']}: request envelopes only; reads nothing."}
    return docs


def scope_text(key: str) -> str:
    spec = pf.CELLS[key]
    return (f"DRAFT -- REGISTERED. NPU-port twin of {spec['historical_run']}. {spec['role'][0].upper()}{spec['role'][1:]}: {hw.SCOPE}; served by the BF16 / "
            f"vLLM-Ascend Thinking core over the original recordings ({pf.PROMPT_ORDER}) with the port's re-authored knowledge (intent-boundaries-npu-port-v1, "
            "slot-contracts-npu-port-v2) and the port's regenerated retrieval vectors (CPU float32, forced causal), the agreement table, centroid keys and "
            f"first-phase supply of the port preparations; {pf.DIFFERENCES_TEXT}")


def draft_rows(ids: Mapping[str, Mapping], commit: str, at: str, shas: Mapping[str, str]) -> list[list[str]]:
    rows = []
    for key in pf.ALL_KEYS:
        i = ids[key]
        for fit in (False, True):
            name = fam.plan_file_name(key, "fit" if fit else "generation")
            calls = i["family_budget"]["fit_endpoint_calls_cap" if fit else "generation_calls_cap"]
            rows.append([at[:10], i["fit_run_id"] if fit else i["run_id"], i["execution_profile"], scope_text(key) + (" -- FIT spend identity" if fit else ""),
                         pf.model_text(), f"slurp TRAIN Fold B {i['rows']}", "discovery", i["split_identity_hash"], "no", str(calls),
                         str(i["family_budget"]["repeated_speech_audio_seconds_cap"]), pf.ledger_gpu_cell(key), INHERITED, f"{PKG}/{name}",
                         (f"DRAFT at study commit {commit}; plan SHA256 {shas.get(name, 'pending')}; drafted by the agent on owner decision 2026-10-28 item 4, owner "
                          f"countersignature pending; the decision receipt {pf.DECISION_RELPATH} is not minted; the gpu-hours cell is the PROVISIONAL card-hour "
                          f"formula ({pf.CARD_HOUR_FORMULA[key]}) and refuses the exposure gate until the measured sweep fills {pf.DEPLOYMENT_REF}; nothing authorised.")])
    return rows


def countersign_texts(ids: Mapping[str, Mapping]) -> dict:
    head = ("REGISTERED BEFORE CONTACT (root countersigned <UTC timestamp> by the owner on the port brief: " + pf.OWNER_DECISIONS + "; the NPU-port twin of the "
            "L338 lever column on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold B; deployment <the filled PORT_DEPLOYMENT_PROFILE: instances, tensor-parallel "
            "size, sequences per instance, devices>, card-hour bound <the re-derived number> written into the gpu-hours cell; ")
    tail = (f"; the decision receipt {pf.DECISION_RELPATH} is minted by {pf.TOOL_MINT} after this countersignature)")
    rows = {}
    for key in pf.ALL_KEYS:
        i = ids[key]
        rows[i["run_id"]] = head + pf.CELLS[key]["role"] + tail
        rows[i["fit_run_id"]] = head + f"live phase-fit identity of the {pf.CELLS[key]['role']}" + tail
    return {"_note": ("Before countersigning: replace each gpu-hours cell's PROVISIONAL formula with the number re-derived from the measured sweep (and the same "
                      "number in the experiment's planned_gpu_hours / gpu_hours_fit_inclusive and the draft receipt's budget), then replace the leading 'DRAFT -- "
                      "REGISTERED' scope marker and the 'DRAFT' note prefix of each row with the text below and mint the receipt."),
            "decision_id": pf.DECISION_ID, "decision_receipt": pf.DECISION_RELPATH, "rows": rows}


def ledger_block(rows: list[list[str]], intro: str, mark: str) -> str:
    cols = reg.LEDGER_COLUMNS
    lines = [mark, "", intro, "", "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += [reg.table_row(cells) for cells in rows]
    return "\n".join(lines) + "\n"


def readme_text(ids: Mapping[str, Mapping], commit: str) -> str:
    arm, smoke = ids["arm"], ids["smoke"]
    deps = "\n".join(f"- {d}" for d in pf.MAX_NUM_SEQS_DEPENDENTS)
    diffs = "\n".join(f"- {d}" for d in pf.DECLARED_DIFFERENCES)
    return (
        "# L338 NPU port: the lever column on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold B (DRAFT / HELD)\n\n"
        f"The NPU-port twin of the L338 lever-column flight (historical package `{pf.HISTORICAL_PACKAGE}`), registered before contact on {pf.DATE} at study "
        f"commit `{commit}` by `scripts/flights/nx/l338_port_write_registration.py` from the cells `scripts/flights/nx/l338_port_derive_cells.py` derives. "
        f"Design: `{pf.DESIGN_DOC}`. Authority: {pf.OWNER_DECISIONS}. Development numbers only; never a test claim; claims_benefit false.\n\n"
        "## Cells (chain order)\n\n"
        "| cell | run | FIT run | rows | generation calls | FIT calls | speech-seconds | accelerator bound |\n|---|---|---|---|---|---|---|---|\n"
        f"| `{smoke['cell']}` | `{smoke['run_id']}` | `{smoke['fit_run_id']}` | {smoke['rows']} | {smoke['family_budget']['generation_calls_cap']} | "
        f"{smoke['family_budget']['fit_endpoint_calls_cap']} | {smoke['family_budget']['repeated_speech_audio_seconds_cap']} | PROVISIONAL: {pf.CARD_HOUR_FORMULA['smoke']} |\n"
        f"| `{arm['cell']}` | `{arm['run_id']}` | `{arm['fit_run_id']}` | {arm['rows']} | {arm['family_budget']['generation_calls_cap']} | "
        f"{arm['family_budget']['fit_endpoint_calls_cap']} | {arm['family_budget']['repeated_speech_audio_seconds_cap']} | PROVISIONAL: {pf.CARD_HOUR_FORMULA['arm']} |\n\n"
        "The calls, FIT and audio caps are the historical registration's (they do not depend on the deployment). The arms, program "
        f"(`{fam.PROGRAM}`, {fam.CONTACTS_PER_ROW} contacts per row), roster, templates, thinking budget {fam.DELIVERING_BUDGET} / cap {fam.COMPLETION_CAP}, decoding "
        f"(greedy, seed {fam.LOCKED_DECODING['seed']}) and read rule are the historical cell's.\n\n"
        "## Serving: protocol facts and the deployment placeholder\n\n"
        f"Model fragment `{pf.MODEL_FRAGMENT}` (transport vLLM, BF16). Protocol facts: the per-sequence context window of {pf.PER_SEQUENCE_WINDOW_TOKENS:,} tokens, "
        f"the reasoning budget and cap, the decoding, the prompt order `{pf.PROMPT_ORDER}`. The DEPLOYMENT -- instances, tensor-parallel size, sequences per instance, "
        f"devices -- is the one named placeholder `PORT_DEPLOYMENT_PROFILE` ({pf.PORT_DEPLOYMENT_PROFILE['filled_by']}). No sequence count, replica count or "
        f"tensor-parallel size is a fact of this registration. Service session name `{pf.SERVICE_SESSION}` is deployment-neutral; session identity, the vLLM "
        "runtime receipt and the lock entries are open owner decisions 9.2-9.4.\n\n"
        "Places that still depend on the model fragment's pre-sweep `--max-num-seqs 4` (left untouched here; the main thread moves them with the profile):\n\n"
        f"{deps}\n\n"
        "## Accelerator bound (PROVISIONAL)\n\n"
        f"- flight: {pf.CARD_HOUR_FORMULA['arm']}.\n- smoke: {pf.CARD_HOUR_FORMULA['smoke']}.\n"
        f"- Illustration only: at the historical lock rate ({fam.ROWS_PER_MINUTE_FULL} rows per minute) the flight bound reads "
        f"{pf.ILLUSTRATION_AT_HISTORICAL_RATE['card_hours_per_unit_tensor_parallel']} x T card-hours.\n"
        "- The experiments' numeric planned_gpu_hours (90.3 / 0.5) are placeholders the plan field requires; the ledger gpu-hours cells carry the formula, so "
        "the exposure gate refuses these rows (\"must be a number\") until the sweep re-derives the bound.\n\n"
        "## Declared differences (design doc section 8)\n\n"
        f"{diffs}\n\n"
        "## Files\n\n"
        "`protocol-registered-v1.json`, the four plan files, `registered-facts-v1.json` (stage3_measurement blocks keyed by the port run ids), "
        "`signoff-conditions-v1.json`, `root-admission-draft-v1.json` (decision `" + pf.DECISION_ID + "`, relpath `" + pf.DECISION_RELPATH + "`, eligible run ids "
        "smoke, smoke FIT, flight, flight FIT), `countersign-row-texts-v1.json`, `draft-ledger-rows.md` (the four DRAFT rows and the six model-free-check rows "
        "of the port preparations).\n\n"
        "## What the main thread does (in order)\n\n"
        "1. Commit this package, the two port cells and the ledger rows.\n"
        "2. Run the speed sweep; fill `PORT_DEPLOYMENT_PROFILE`; move the model fragment's `--max-num-seqs` / `profile_slot_count` / `geometry_basis` and the "
        "baselines' `serving_slots` with it; re-derive the card-hour bound and write it into planned_gpu_hours, the ledger gpu-hours cells and the draft "
        "receipt's budget; re-run the derive and this writer.\n"
        "3. Owner countersigns the rows (`countersign-row-texts-v1.json`); then mint: "
        f"`SAEA_STUDY_ROOT=<study> SAEA_DATA_ROOT=<data root> python -B {pf.TOOL_MINT} --dry` then without `--dry`.\n"
        "4. Re-pin and flip with `scripts/flights/repin_cell.py` from a clean tree (both cells, `--decision coordination/... --decision-id "
        f"{pf.DECISION_ID}`): it prepares the smoke's phase supply, re-prepares the flight's as phase-supply-v2 (output equality asserted; never "
        "`--allow-output-change`) and pins the decision receipt. `SAEA_ALLOW_9P_FLIP=1` is needed because the tree is under `/mnt/` (CephFS, not 9p).\n"
        "5. Open items: the analysis recipe and READ row of the port, the port chain script, the session identity, runtime receipt and lock entries (9.2-9.4).\n"
    )


def file_sha256(root: Path, relpath: str) -> str:
    return hashlib.sha256((Path(root) / relpath).read_bytes()).hexdigest()


def write_package(root: Path, commit: str, at: str | None, data_root: Path, *, append_ledger: bool = False, derive: bool = True) -> dict:
    root = Path(root)
    facts_path = root / PKG / "registered-facts-v1.json"
    at = at or (json.loads(facts_path.read_text(encoding="utf-8"))["registered_at_utc"] if facts_path.is_file() else datetime.now(timezone.utc).isoformat())
    derived = pdc.derive(root) if derive else None
    prep = read_preparations(data_root)
    ids = identities(root)
    arm, smoke = ids["arm"], ids["smoke"]
    protocol = {
        "cells": {k: {kk: v[kk] for kk in ("attempt_id", "cell", "composed_config_hash", "experiment", "fit_attempt_id", "fit_run_id", "protocol_hash",
                                           "rows", "run_id", "split_identity_hash", "historical_run_id")} for k, v in ids.items()},
        "arms": {row["arm"]: {"role": row["role"], "axes": row["axes"]} for row in hw.arm_table_rows()},
        "blockers": [], "findings": hw.FINDINGS, "port_findings": [
            {"id": "deployment-is-a-placeholder", "finding": pf.PORT_DEPLOYMENT_PROFILE["not_a_fact_here"], "remedy": "the measured speed sweep fills the profile"},
            {"id": "historical-findings-are-context", "finding": ("The historical findings are carried for context; those that name the resident np4 llama.cpp "
                                                                  "session, its batching or its cost model describe the historical flight, not the port deployment."),
             "remedy": "none"},
            {"id": "fit-ids-fixed", "finding": ("The committed port flight baseline still carried the historical FIT run and attempt ids; the port derive moves "
                                                "them to the port FIT identity, which changes the composed config and makes phase-supply-v1 stale."),
             "remedy": "repin_cell.py re-prepares phase-supply-v2 with output equality asserted"}],
        "decision_rule_fixed_here": f"The arm words and thresholds are the historical package's ({pf.HISTORICAL_PACKAGE}/README.md, {fam.READER}).",
        "failure_policy": "Stop and preserve on the first failure; at most one root-diagnosed recovery re-registration per run id; run-cumulative budgets.",
        "declared_differences": list(pf.DECLARED_DIFFERENCES), "serving": serving_block(), "accelerator_bound": cost_block(),
        "prohibitions": hw.PROHIBITIONS + ["No deployment value (instances, tensor-parallel size, sequences per instance, devices) is registered here; the port is "
                                           "not flown before PORT_DEPLOYMENT_PROFILE is filled from the measured sweep and the card-hour bound re-derived."],
        "protocol_id": "slurp-l338-lever-column-fold-b-npu-port-protocol-v1", "registered_at_utc": at,
        "schema": "saea-registered-protocol-v1", "status": STATUS, "study_commit_at_registration": commit,
        "the_question": ("Does the L338 lever column read the same way on the BF16 / vLLM-Ascend Thinking core with the port's knowledge and vectors? Each candidate "
                         f"arm and the twin {fam.TWIN} are read PAIRED per row against the delivering {fam.DELIVERING} arm on the same 1,080 Fold B rows in one flight, "
                         "beside (never as a byte replay of) the historical flight.")}
    protocol_sha = reg.wjson(root, f"{PKG}/protocol-registered-v1.json", protocol)
    plans = plan_docs(ids, commit, at, protocol_sha)
    shas = {name: reg.wjson(root, f"{PKG}/{name}", doc) for name, doc in plans.items()}
    shas["protocol-registered-v1.json"] = protocol_sha
    signoff = dict(fam.signoff_conditions(), package=pf.PKG_NAME)
    signoff["_note"] = "L338 NPU port: " + signoff["_note"] + " Copied from l338_family.signoff_conditions(); the port smoke's reader output path is the historical reader's."
    shas["signoff-conditions-v1.json"] = reg.wjson(root, f"{PKG}/signoff-conditions-v1.json", signoff)
    facts = {
        "_note": ("Every model-free fact this DRAFT port registration rests on, and the canonical stage3_measurement blocks keyed by the port run ids "
                  "(scripts/checks/stage3_effect_floor_check.py, scripts/checks/flight_registration_check.py)."),
        "accounting": {"program": fam.PROGRAM, "lock": fam.LOCK, "contacts_per_row": fam.CONTACTS_PER_ROW, "arms": list(fam.ARM_ORDER), "rows": fam.ROWS,
                       "caps": cost_block()["carried_over_caps"], "accelerator_bound": cost_block()},
        "composed_config_hashes": {k: v["composed_config_hash"] for k, v in ids.items()},
        "context_fit": port_fit_proof(), "identity": ids, "derive": derived,
        "pins": {"model_fragment": pf.MODEL_FRAGMENT, "serving": serving_block(), "port_pins": {"path": "scripts/flights/nx/l338_port_pins.json",
                                                                                               "sha256": file_sha256(root, "scripts/flights/nx/l338_port_pins.json")},
                 "locked_template": {"relpath": fam.LOCKED_TEMPLATE, "sha256": fam.LOCKED_TEMPLATE_SHA},
                 "signoff_conditions": {"path": f"{PKG}/signoff-conditions-v1.json", "sha256": shas["signoff-conditions-v1.json"]},
                 "preparations": {k: {"receipt": v["receipt"], "receipt_sha256": v["receipt_sha256"], "artifact": v["artifact"], "artifact_sha256": v["artifact_sha256"]}
                                  for k, v in prep.items()}},
        "max_num_seqs_dependents": list(pf.MAX_NUM_SEQS_DEPENDENTS), "declared_differences": list(pf.DECLARED_DIFFERENCES),
        "prohibitions": protocol["prohibitions"], "registered_at_utc": at,
        "registration_basis": f"{pf.OWNER_DECISIONS}; {pf.DESIGN_DOC}. DRAFT: nothing here authorises a flip or a flight.",
        "read_rule": {"reader": fam.READER, "words": hw.WORD_RULES, "word_order": hw.WORD_ORDER, "twin": fam.TWIN,
                      "floors_pp": {"slu": fam.FLOOR_SLU_PP, "intent": fam.FLOOR_INTENT_PP}, "levers": dict(fam.ARM_LEVER), "delivering": fam.DELIVERING},
        "roster": {"arm": fam.ROSTER["arm"], "smoke": fam.ROSTER["smoke"], "roster_id": fam.READ_ROSTER, "split_identity_hash": arm["split_identity_hash"]},
        "rows": fam.ROWS, "schema": "saea-registered-facts-v1", "stage3_measurement": hw.measurement_blocks(ids),
        "study_commit_at_registration": commit, "thinking_knobs": json.loads(json.dumps(fam.THINKING_KNOBS)), "no_think_node": None,
        "sampling_arms": {row["arm"]: row["axes"] for row in hw.arm_table_rows()}, "locked_decoding": dict(fam.LOCKED_DECODING),
        "status": "DRAFT-HELD"}
    shas["registered-facts-v1.json"] = reg.wjson(root, f"{PKG}/registered-facts-v1.json", facts)
    draft = {
        "_draft_note": (f"A DRAFT for the MAIN THREAD to mint on the data root with {pf.TOOL_MINT} (owner decision 2026-10-28 item 5). It authorises nothing while "
                        "its status is not 'decided'. Before minting, the measured sweep fills PORT_DEPLOYMENT_PROFILE and the PROVISIONAL accelerator bound below "
                        "is replaced by the re-derived number."),
        "antecedents": {"protocol": {"path": f"{PKG}/protocol-registered-v1.json", "sha256": protocol_sha},
                        "registered_facts": {"path": f"{PKG}/registered-facts-v1.json", "sha256": shas["registered-facts-v1.json"]},
                        "plans": {name: {"path": f"{PKG}/{name}", "sha256": shas[name]} for name in plans},
                        "registration_base_commit": commit},
        "authorization_basis": f"{pf.OWNER_DECISIONS}; {pf.DESIGN_DOC}; historical authority {fam.OWNER_RULINGS} ({pf.HISTORICAL_PACKAGE}).",
        "budget": {**arm["family_budget"], "paid_api_calls": 0, "plan_budget_is_run_cumulative": True, "smoke": smoke["family_budget"],
                   "accelerator_bound": {"status": "PROVISIONAL", "formula": dict(pf.CARD_HOUR_FORMULA),
                                         "note": "gpu_hours_fit_inclusive above is the historical numeric placeholder, not the NPU card-hour bound"}},
        "claim_boundary": "No benefit, system-level, superiority, generalisation, held-out, confirmatory or paper-scale claim.",
        "decision": (f"Fly the NPU-port twin of the L338 lever column on the BF16 / vLLM-Ascend Thinking core on TRAIN Fold B: four-row smoke, then the "
                     f"{fam.K}-arm column flight (with the twin), on the deployment PORT_DEPLOYMENT_PROFILE once it is filled."),
        "decision_id": pf.DECISION_ID, "eligible_attempt_ids": pf.eligible_attempt_ids(), "eligible_run_ids": pf.eligible_run_ids(),
        "not_authorized": ["any flight before the cells are flipped and the receipt is minted", "any flight before PORT_DEPLOYMENT_PROFILE is filled",
                           "starting or touching any serving session outside the registered deployment", "any ASR, transcript, encoder or paid-API contact",
                           "--allow-output-change on the re-pin tool", "any change beyond the registered sampling axes", "Fold A, DEV or test-split contact"],
        "owner": "root", "population": {"count": fam.ROWS, "fold_a_contact_authorized": False, "split_identity_hash": arm["split_identity_hash"],
                                        "split_role": "discovery", "test_contact_authorized": False},
        "recorded_at_utc": None, "schema": "saea-campaign-decision-v1",
        "service": {"expected_session_id": pf.SERVICE_SESSION, "model_fragment": pf.MODEL_FRAGMENT, "deployment_profile": "PORT_DEPLOYMENT_PROFILE (pending)",
                    "no_child_start_stop": True},
        "status": "DRAFT-NOT-MINTED"}
    shas["root-admission-draft-v1.json"] = reg.wjson(root, f"{PKG}/root-admission-draft-v1.json", draft)
    cell_rows = draft_rows(ids, commit, at, shas)
    prep_rows = model_free_rows(prep, commit)
    prep_text = ledger_block(prep_rows, (
        "Six model-free-check rows for the port preparations the L338 NPU-port flight pins (owner decision 2026-10-28 item 6), in build order: the query-dimension "
        "alias, the content derivative, the minus-r01 store rebuild, the centroid keys, the agreement table (source identity: the port comparator cell) and the "
        "first-phase supply. Each was run before its row was written; every fact below is re-read from its receipt on the data root by "
        "scripts/flights/nx/l338_port_write_registration.py. DRAFT, root countersignature pending; no row authorises a contact."), pf.PREP_LEDGER_MARK)
    cell_text = ledger_block(cell_rows, (
        f"4 unconsumed DRAFT rows (generation and live-phase-fit identity of the port smoke and the port flight) for the NPU-port twin of L338 ({pf.OWNER_DECISIONS}). "
        f"Package {PKG}. No row is countersigned and no row authorises a contact. Each gpu-hours cell is the PROVISIONAL NPU card-hour formula, not a number: the "
        "exposure gate refuses these rows until the measured speed sweep fills PORT_DEPLOYMENT_PROFILE and the bound is re-derived. No serving row: the deployment "
        "is not registered yet."), pf.LEDGER_MARK + " (DRAFT, owner countersignature pending)")
    shas["draft-ledger-rows.md"] = reg.wr(root, f"{PKG}/draft-ledger-rows.md", prep_text + "\n" + cell_text)
    shas["countersign-row-texts-v1.json"] = reg.wjson(root, f"{PKG}/countersign-row-texts-v1.json", countersign_texts(ids))
    shas["README.md"] = reg.wr(root, f"{PKG}/README.md", readme_text(ids, commit))
    prereg = (f"# L338 NPU port: the lever column, preregistration (DRAFT / HELD)\n\nRegistered {at[:10]} at study commit `{commit}` by script. "
              "Development numbers only.\n\n"
              f"Runs: `{smoke['run_id']}` / `{smoke['fit_run_id']}`; `{arm['run_id']}` / `{arm['fit_run_id']}`.\n\n"
              f"The package is `{PKG}/`; the read rule is the historical package's.\n\n## The measurement declaration\n\n```json\n"
              + json.dumps({"stage3_measurement": hw.measurement_blocks(ids)}, indent=2, ensure_ascii=False, sort_keys=True) + "\n```\n")
    shas["__prereg__"] = reg.wr(root, pf.PREREG, prereg)
    result = {"package": PKG, "shas": shas, "derive": derived, "ledger_appended": None}
    if append_ledger:
        ledger = root / reg.LEDGER
        current = io.open(ledger, encoding="utf-8", newline="").read()
        if pf.LEDGER_MARK in current or pf.PREP_LEDGER_MARK in current:
            result["ledger_appended"] = False
        else:
            io.open(ledger, "w", encoding="utf-8", newline="").write(current.rstrip("\n") + "\n\n" + prep_text + "\n" + cell_text)
            result["ledger_appended"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--study-commit", required=True)
    parser.add_argument("--registered-at", default=None, help="default: the stamp already in registered-facts-v1.json, else now")
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--data-root", default=os.environ.get("SAEA_DATA_ROOT") or os.environ.get("SPEECHRL_DATA_DIR"), required=False)
    parser.add_argument("--append-ledger", action="store_true", help="append both DRAFT blocks at the END of docs/exposure-ledger.md (idempotent)")
    parser.add_argument("--no-derive", action="store_true")
    args = parser.parse_args(argv)
    if not args.data_root:
        parser.error("--data-root (or SAEA_DATA_ROOT / SPEECHRL_DATA_DIR) is required to re-read the preparation receipts")
    result = write_package(Path(args.root), args.study_commit, args.registered_at, Path(args.data_root), append_ledger=args.append_ledger,
                           derive=not args.no_derive)
    print(json.dumps({k: v for k, v in result.items() if k != "derive"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
