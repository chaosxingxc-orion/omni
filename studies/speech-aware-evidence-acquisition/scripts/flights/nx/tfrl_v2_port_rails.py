#!/usr/bin/env python3
"""The rails declarations of the T cell (TFRL-v2): the analysis recipe, the READ row draft, the bank-release registration and the writer check that flips the
package marker from ``rails_pending`` to ``rails_declared`` (plan ``docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md`` section 12
ruling 5).

Model-free and import-only. The T writer (``tfrl_v2_port_write_registration.py``) calls :func:`write_rails_files` and :func:`rails_status`; the T reader
(``scripts/analysis/slurp_tfrl_v2_port_read.py``) imports the names and numbers from here, so no id or number is spelled twice. Nothing here contacts a model, a
server, a tokenizer, an encoder or a scorer.

Three deliverables make the marker flip, and the check below reads each from disk (it trusts no flag):

1. the stage-3 label ``tfrl-v2`` (``core.stage3_algorithms``) and the consumption rail (``rails/tfrl_v2_consumption.py``: the ``tfrl_v2_bank`` declaration the module
   block accepts, the flight-start refusals, the Solver composition and the ``used_basis`` read from the trace);
2. the analysis recipe ``analysis-recipe-v1.json`` (written here, byte-deterministic, its sha256 pinned in ``read-row-draft-v1.md`` and in the facts) and the reader
   ``scripts/analysis/slurp_tfrl_v2_port_read.py``;
3. the bank-release registration ``bank-release-v1.json``: per iteration, the source halves and split identities, the bank receipt hashes (PENDING until a release
   is minted), the union hash, the measure, the token caps and counts, the consuming cell ids, and the rule that T0 (official test split) rows are never bank sources.

The marker says the RAILS are declared; it does not say the five T cells adopt them. The cells keep their closest registered labels and their zero-sentinel bank pins
until the main thread re-mints them at the flip (the frame to write is recorded per consuming cell in the bank-release registration).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts" / "analysis"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import l338_family as fam  # noqa: E402
import paired_ci  # noqa: E402
import slurp_l338_column_read as hist  # noqa: E402
import tfrl_v2_common as cm  # noqa: E402
import tfrl_v2_port_derive_cells as tdc  # noqa: E402
import tfrl_v2_port_family as tf  # noqa: E402
import tfrl_v2_sampler as smp  # noqa: E402
from speech_aware_evidence_acquisition.core import stage3_algorithms as s3  # noqa: E402
from speech_aware_evidence_acquisition.rails import tfrl_v2_consumption as rail  # noqa: E402

ZERO = "0" * 64
RECIPE_NAME = "analysis-recipe-v1.json"
READ_ROW_NAME = "read-row-draft-v1.md"
BANK_RELEASE_NAME = "bank-release-v1.json"
RECIPE_RELPATH = f"{tf.PACKAGE}/{RECIPE_NAME}"
READ_ROW_RELPATH = f"{tf.PACKAGE}/{READ_ROW_NAME}"
BANK_RELEASE_RELPATH = f"{tf.PACKAGE}/{BANK_RELEASE_NAME}"
READER = "scripts/analysis/slurp_tfrl_v2_port_read.py"
CONSUMPTION_RAIL = "src/speech_aware_evidence_acquisition/rails/tfrl_v2_consumption.py"
RECIPE_SCHEMA = "saea-tfrl-v2-analysis-recipe-v1"
RELEASE_SCHEMA = "saea-tfrl-v2-bank-release-registration-v1"
READ_ROW_SCHEMA_NOTE = "DRAFT READ row"
MARKER_PENDING, MARKER_DECLARED = "rails_pending", "rails_declared"
ANALYSIS_ROOT = f"analysis/{tf.READ_RUN_ID}"
VOTE_ARM = "vote"                                        # the derived column of the construction read: the vote over s1..s4 (scored by the official scorer on a derived copy)
MAX_ITERATIONS = 3
STOP_GAIN_PP = 1.0
GAIN_DECIMALS = 6
SEED, DRAWS = hist.SEED, hist.DRAWS
FLOOR_SLU_PP, FLOOR_INTENT_PP, TWIN_BAR_PP = fam.FLOOR_SLU_PP, fam.FLOOR_INTENT_PP, 1.2
#: The rosters that are never bank sources besides the official test split T0: (file stem under docs/receipts/rosters, label).
FORBIDDEN_ROSTERS = (("slurp-evo-fold-a-v2", "Fold A v2"), ("slurp-evo-fold-b-v2", "Fold B v2"), ("slurp-dev-1000-ts2-v1", "DEV-1000"))
T0_TEXT = ("T0, the official SLURP test split, is never a source of the exemplar bank or the rule bank (comparison plan 2026-10-30, section on the evaluation tiers: 'T0 rows are never exemplar "
           "or bank sources'). No T0 roster is registered in this repository; the rule is therefore structural: the only admitted bank sources are the two construction halves R01C-H1 and R01C-H2 "
           "by roster id AND pinned split identity, and the flight-start refusal (rails.tfrl_v2_consumption.verify_bank) names any other source, so a T0 row cannot enter a bank even if a T0 roster is "
           "registered later.")


def _sha_file(root: Path, relpath: str) -> str:
    return hashlib.sha256((Path(root) / relpath).read_bytes()).hexdigest()


def _json_bytes(doc: Any) -> bytes:
    """The bytes ``reg.wjson`` writes (the registration writers' form)."""

    return (json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


# ---- the analysis recipe ----------------------------------------------------------------------------------------------------------------------------


def recipe_document() -> dict:
    """The T analysis recipe. Deterministic: no timestamp, no commit; it moves only when a rule moves."""

    construction_runs = {k: {"run_id": tf.CELLS[k]["run"], "attempt_id": tf.CELLS[k]["attempt"], "roster": tf.CELLS[k]["roster"], "rows": tf.CELLS[k]["rows"]} for k in ("csmoke", "ch1", "ch2")}
    return {
        "schema": RECIPE_SCHEMA, "recipe_id": "slurp-tfrl-v2-fold-b-npu-port-analysis-recipe-v1", "status": "DRAFT (a reading rule; authorises nothing)",
        "claim_boundary": "No benefit, system-level, superiority, generalisation, held-out, confirmatory or paper-scale claim. Development numbers only.",
        "read_run_id": tf.READ_RUN_ID, "loop_id": tf.LOOP_ID, "reader": READER, "plan": tf.PLAN, "package": tf.PACKAGE,
        "analysis_root": ANALYSIS_ROOT,
        "authority": ("docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md sections 2 (M1, M2), 4, 9, 10 (rulings 3 and 4), 11 and 12; the machinery of "
                      "scripts/analysis/slurp_l338_port_column_read.py (paired_ci.py: row-paired utterance-cluster bootstrap, replicate-widened CI95, per-row F1, the NOMINATED rule)"),
        "contacts": {"model": 0, "server": 0, "tokenizer": 0, "encoder": 0, "asr": 0, "scorer": "CPU official scorer jobs over derived copies only (the vote column)"},
        "run_ids": {"construction": construction_runs, "fold_b": {k: {"run_id": tf.CELLS[k]["run"], "attempt_id": tf.CELLS[k]["attempt"], "rows": tf.CELLS[k]["rows"]} for k in ("fsmoke", "fb")},
                    "read": tf.READ_RUN_ID},
        "trace_read_id_rule": f"{tf.READ_RUN_ID}-I<iteration>-<H1|H2>; the trace is <data root>/analysis/<that id>/rollout-trace.jsonl",
        "delivering": tf.DELIVERING, "twin": tf.TWIN, "replicates": [tf.REPLICATE], "bank_consuming_arms": [tf.T_ARM],
        "arms": {"construction": list(tf.CONSTRUCTION_ARM_ORDER), "fold_b": list(tf.FOLD_B_ARM_ORDER)},
        "construction_read": {
            "unit": "one iteration, both halves, from the sealed trace of each half's construction flight (contact and score lines) and the iteration's bank receipts",
            "credit": ("reward(k) = official per-row SLU-F1 of trajectory s_k minus that of lock (tfrl_v2_sampler.credit_rows); a row is POSITIVE when its best intent-correct trajectory beats lock by "
                       "more than 0 and lock and aa agree on the row"),
            "per_half": {
                "positive_rate": "positives / usable rows of the half (a usable row has all K Solver trajectories, lock and aa scored)",
                "mean_best_of_k_gain_pp": "100 x mean over usable rows of max over k of reward(k) (negative values included; the oracle best-of-K, reported beside the positive rate)",
                "rule_survivors": "rules in the half's bank (rule-bank.json of the bank distilled from the half; survivors of the 3-row vote) and, for the iteration, in the union bank",
                "exemplar_coverage_per_scenario": ("for each scenario stratum of the half's roster: rows, exemplars in the union bank, covered = at least one exemplar; plus the covered share of "
                                                   "scenarios and of rows"),
            },
            "construction_gain": {
                "definition": ("SLU-F1(vote) - SLU-F1(lock) over the usable rows of BOTH halves of the iteration, in pp, pooled; vote = scenario_action by plurality over s1..s4 (tie: the lowest k), "
                               "entities kept at support >= 2 of 4 (plan section 12; README acceptance rule). The vote answers are derived model-free from the trace by the reader's `vote` stage; "
                               "the official scorer scores that derived copy as column `vote`; the reader never scores"),
                "never_from": "the loop adapter (plan section 11 addendum ruling 6)",
                "unscored_vote": "when the `vote` column is not scored the gain is null and the stop word is UNDETERMINED, never a pass",
                "vote_arm": VOTE_ARM,
            },
            "outputs": ["construction-i<t>.json", "vote-answers-i<t>.jsonl"],
        },
        "fold_b_read": {
            "unit": "the iteration's Fold B selection flight: the bank-consuming arms minus lock, paired per row inside the flight",
            "columns": "arm-<arm>-<scorer>-scores.json written by stage3_rollout_score --mode perception-branches (official scorer, per-row counts)",
            "primary": {"metric": "SLU-F1", "pooled_and_per_row": True, "bootstrap": {"seed": SEED, "draws": DRAWS, "level": "CI95", "cluster": "utterance"}},
            "noise_model": {"copies": [tf.DELIVERING, tf.TWIN, tf.REPLICATE], "max_replicates": 2, "replicate_arms": [tf.REPLICATE], "formula": paired_ci.FORMULA, "module": "scripts/analysis/paired_ci.py",
                            "words": "the CI95 of every arm delta is the row-paired bootstrap interval widened by the replicate variance of {lock, aa, lock-rep2}; the point estimate is unchanged"},
            "floors_pp": {"slu": FLOOR_SLU_PP, "intent": FLOOR_INTENT_PP},
            "nominated_rule": ("NOMINATED when the pooled SLU-F1 delta has point >= the SLU floor AND widened CI95 lower bound > 0, the per-row SLU-F1 delta has widened CI95 lower bound > 0, no "
                               "resolved per-row loss (per-row CI95 upper bound >= -SLU floor), the intent guard holds (intent CI95 lower bound >= -intent floor and intent point >= 0); HARM when a "
                               "pooled upper bound is below minus its floor; otherwise NEUTRAL (plan section 10 ruling 4; the per-row F1 is a repo diagnostic, never an official SLURP number)"),
            "accepted_only_if": ["the arm is NOMINATED", "the per-row mean delta >= 0", f"twin admissible: |SLU-F1(aa) - SLU-F1(lock)| <= {TWIN_BAR_PP} pp (rounded to {GAIN_DECIMALS} decimals)",
                                 f"the construction gain of the iteration >= +{STOP_GAIN_PP} pp"],
            "confirmed_only_if": "the Fold A flight of the accepted state has a widened CI95 lower bound of t-v2 - lock above 0 (M2); the optimism term is reported always (read by the Fold A reader, not here)",
            "outputs": ["read-i<t>.json", "tfrl-v2-verdicts-i<t>.json"],
        },
        "stop_rule": {
            "construction_gain_pp": STOP_GAIN_PP, "max_iterations": MAX_ITERATIONS, "rounding_decimals": GAIN_DECIMALS,
            "text": (f"the loop continues only while the construction-column gain of the iteration is >= +{STOP_GAIN_PP} pp (compared after rounding to {GAIN_DECIMALS} decimals, so +1.000000 "
                     f"passes and +0.9 stops), at most {MAX_ITERATIONS} iterations, and an iteration that accepts nothing stops; a gain under +{STOP_GAIN_PP} pp at iteration 1 stops the family"),
        },
        "bank_release": {"registration": BANK_RELEASE_RELPATH, "t0_rule": T0_TEXT, "measure_required": rail.MEASURE_VLLM, "measure_refused": [rail.MEASURE_ESTIMATE]},
        "what_is_not_here": ["no model, server or tokenizer contact; the vote column is scored by the official scorer on a derived copy outside the reader",
                             "no admission, no readiness flip, no ledger row", "Fold A confirmation (a separate package and reader)"],
    }


def recipe_bytes() -> bytes:
    return _json_bytes(recipe_document())


def recipe_sha256() -> str:
    return hashlib.sha256(recipe_bytes()).hexdigest()


# ---- the READ row draft ------------------------------------------------------------------------------------------------------------------------------


def read_row_text(recipe_sha: str) -> str:
    return (
        "# T cell: DRAFT READ row (DRAFT, no authority)\n\n"
        f"{READ_ROW_SCHEMA_NOTE} of the T cell of the three-way comparison (`{tf.CELL_ID}`), written by `scripts/flights/nx/tfrl_v2_port_write_registration.py` (rails step) in the column format of the "
        "L338 port package's READ row (`docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/read-row-draft-v1.md`). Nothing here is in `docs/exposure-ledger.md`; nothing here is countersigned, "
        f"consumed or authorised. The recipe it names is `{RECIPE_RELPATH}` (sha256 `{recipe_sha}`); the writer re-run re-pins it, so a moved rule moves this line.\n\n"
        "## The READ row (iteration 1)\n\n"
        "| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        f"| {tf.DATE} | {tf.READ_RUN_ID} | model-free-check | The offline read of the T cell, iteration 1: per-iteration construction read of the two R01C halves (positive rate, mean best-of-K gain, rule survivors, exemplar "
        f"coverage per scenario, the construction gain of the vote over s1..s4 minus lock) and the Fold B read of the bank-consuming arm {tf.T_ARM} minus the delivering lock, paired per row inside the flight, "
        f"CI95 widened by the replicate variance of the three request-identical copies lock / aa / lock-rep2, the NOMINATED rule and the stop rule (+{STOP_GAIN_PP} pp construction gain, at most {MAX_ITERATIONS} "
        f"iterations); reader `{READER}`, recipe sha256 `{recipe_sha}` | CPU official scorer jobs over derived copies only; no model, server, tokenizer or encoder contact | none (reads sealed flights) | "
        f"R01C-H1 {tf.CELLS['ch1']['split_identity_hash'][:12]}..., R01C-H2 {tf.CELLS['ch2']['split_identity_hash'][:12]}..., TRAIN Fold B {tf.CELLS['fb']['split_identity_hash'][:12]}... | discovery | see scope | no | 0 | 0 | 0 | "
        f"inherited in full from the construction and selection flights | `{ANALYSIS_ROOT}/` | DRAFT; claims_benefit false |\n"
    )


# ---- the bank-release registration ----------------------------------------------------------------------------------------------------------------


def _roster(root: Path, relpath: str) -> dict:
    return json.loads((Path(root) / relpath).read_text(encoding="utf-8"))


def _row_ids(doc: Mapping) -> set[str]:
    return {str(item["sample_id"]) for item in doc["items"]}


def forbidden_rosters(root: Path) -> dict:
    """The rosters that are never bank sources, with their identities and the (proved) empty overlap with the construction column."""

    r01c_ids = _row_ids(_roster(root, tf.R01C_ROSTER))
    out = {}
    for stem, label in FORBIDDEN_ROSTERS:
        relpath = f"docs/receipts/rosters/{stem}.json"
        doc = _roster(root, relpath)
        out[label] = {"roster": relpath, "roster_id": doc["roster_id"], "split_identity_hash": doc["split_identity_hash"], "rows": len(doc["items"]),
                      "rows_shared_with_r01c": len(_row_ids(doc) & r01c_ids), "file_sha256": _sha_file(root, relpath)}
    out["T0 official SLURP test split"] = {"roster": None, "roster_id": None, "split_identity_hash": None, "rows": None, "rows_shared_with_r01c": None,
                                           "note": "no roster is registered; excluded structurally (only the two construction halves are admitted sources)"}
    return out


def _consumers(iteration: int) -> dict:
    """The consuming cells of the iteration's banks. Iteration 1 is registered: the union bank is consumed by the Fold B smoke and flight; each half bank by the OTHER half's flight of iteration 2
    (a later registration, named by the sampler's own ids, not yet a cell)."""

    union_cells = [tf.CELLS[k]["cell"] for k in ("fsmoke", "fb")] if iteration == tf.ITERATION else []
    return {
        "union": {"consumer": rail.CONSUMER_FOLD_B, "cell_ids": union_cells, "run_ids": [tf.CELLS[k]["run"] for k in ("fsmoke", "fb")] if iteration == tf.ITERATION else [],
                  "also_consumed_by": f"the Fold A confirmation flight of the accepted state ({rail.CONSUMER_FOLD_A}); a separate package"},
        "half": {half: {"consumer": cm.other_half(half), "planned_run_id": smp.run_id_for(cm.other_half(half), iteration + 1),
                        "planned_attempt_id": smp.attempt_id_for(cm.other_half(half), iteration + 1), "registered": "after the iteration is read; not a cell yet"} for half in cm.HALVES},
    }


def _release(kind: str, iteration: int, sources: Mapping[str, str], consumer: str, receipt_relpath: str, consuming: Mapping) -> dict:
    return {
        "kind": kind, "source_halves": list(sources), "source_split_identities": dict(sources), "consuming_roster": consumer,
        "consuming": dict(consuming), "bank_receipt": {"relpath": receipt_relpath, "sha256": ZERO, "state": "PENDING (zero sentinel): minted when the bank is released"},
        "exemplar_bank_content_sha256": ZERO, "rule_bank_content_sha256": ZERO,
        "measure": {"required": rail.MEASURE_VLLM, "observed": None, "refused": [rail.MEASURE_ESTIMATE], "exemplar_text_required": rail.EXEMPLAR_TEXT_RESOLVED},
        "token_counts": {"exemplar_block_token_cap": tf.EXEMPLAR_BLOCK_TOKEN_CAP, "exemplar_max_block_tokens": None,
                         "rule_bank_token_cap": tf.RULE_BANK_TOKEN_CAP, "rule_bank_rendered_tokens": None},
    }


def bank_release_document(root: Path) -> dict:
    """The bank-release registration (iteration 1 registered; later iterations are registered after the previous read, by the same step)."""

    root = Path(root)
    identities = {half: tf.CELLS["ch1" if half == cm.HALF_H1 else "ch2"]["split_identity_hash"] for half in cm.HALVES}
    sources = {}
    for half, key in ((cm.HALF_H1, "ch1"), (cm.HALF_H2, "ch2")):
        spec = tf.CELLS[key]
        sources[half] = {"roster": spec["roster"], "roster_id": spec["roster_id"], "split_identity_hash": spec["split_identity_hash"], "rows": spec["rows"],
                         "file_sha256": _sha_file(root, spec["roster"])}
    forbidden = forbidden_rosters(root)
    fold_b_ids = _row_ids(_roster(root, tf.ROSTERS["fb"]))
    r01c_ids = _row_ids(_roster(root, tf.R01C_ROSTER))
    halves_ids = [_row_ids(_roster(root, tf.CELLS[k]["roster"])) for k in ("ch1", "ch2")]
    iterations = []
    for iteration in range(1, tf.ITERATION + 1):
        consumers = _consumers(iteration)
        base = f"{ANALYSIS_ROOT}/v2/banks/i{iteration}"
        releases = {f"half-{half}": _release(rail.KIND_HALF, iteration, {half: identities[half]}, cm.other_half(half), f"{base}/{half}/bank-receipt.json", consumers["half"][half])
                    for half in cm.HALVES}
        union = _release(rail.KIND_UNION, iteration, identities, consumers["union"]["consumer"], f"{base}/union/bank-receipt.json", consumers["union"])
        union["union_hash"] = {"sha256": ZERO, "state": "PENDING (zero sentinel): the content hash of the union bank receipt"}
        union["knowledge_release_dirs"] = {name: spec["dir"] for name, spec in tdc.BANKS.items()}
        releases["union"] = union
        frames = {}
        for key in ("fsmoke", "fb"):
            frames[tf.CELLS[key]["cell"]] = rail.declaration_block(iteration=iteration, consumer=rail.CONSUMER_FOLD_B, bank_kind=rail.KIND_UNION,
                                                                  receipt_relpath=union["bank_receipt"]["relpath"], source_scope=identities)
        iterations.append({"iteration": iteration, "status": "REGISTERED, PENDING RELEASE (every receipt hash is the zero sentinel)", "releases": releases,
                           "tfrl_v2_bank_frames": frames,
                           "frame_note": ("the frame the consuming cell's native_episode.consumption_module carries at the flip, with bank_receipt.sha256 replaced by the minted union hash; "
                                          "rails.tfrl_v2_consumption refuses the zero sentinel, a different hash, a source that includes the consuming roster, and an estimate measure"),
                           "construction_flights_of_the_iteration": {"consume_no_bank": iteration == 1,
                                                                    "note": "iteration-1 construction flights are seed-only and consume no bank; from iteration 2 a half consumes the bank distilled from the OTHER half"}})
    return {
        "schema": RELEASE_SCHEMA, "status": "DRAFT-HELD", "package": tf.PACKAGE, "plan": tf.PLAN, "loop_id": tf.LOOP_ID, "max_iterations": MAX_ITERATIONS,
        "claim_boundary": "No benefit claim; a registration of how banks are released and consumed. Authorises nothing.",
        "t0_never_a_bank_source": {"rule": T0_TEXT, "allowed_source_halves": list(cm.HALVES), "allowed_source_identities": identities, "forbidden_rosters": forbidden,
                                   "proofs": {"r01c_rows": len(r01c_ids), "halves_union_equals_r01c": set().union(*halves_ids) == r01c_ids and not (halves_ids[0] & halves_ids[1]),
                                              "fold_b_rows_shared_with_r01c": len(fold_b_ids & r01c_ids)}},
        "measure": {"required": rail.MEASURE_VLLM, "refused": [rail.MEASURE_ESTIMATE], "exemplar_text_required": rail.EXEMPLAR_TEXT_RESOLVED,
                    "note": "the FIT gate's vLLM tokenize measure through an injected registered client; an estimate-measured or ids-only bank is refused by the sampler's plan builder and by the flight-start refusal"},
        "caps": {"exemplar_block_tokens": tf.EXEMPLAR_BLOCK_TOKEN_CAP, "rule_bank_tokens": tf.RULE_BANK_TOKEN_CAP, "exemplars_replace_supplied_items": tf.EXEMPLAR_ITEM_EQUIVALENT},
        "sources": sources, "receipt_hash_definition": "sha256 of the canonical compact JSON of bank-receipt.json (rails.tfrl_v2_consumption.receipt_sha256); it covers both banks' file hashes, the sources, the measure and the counts",
        "refusals": [
            "the bank receipt hash is the zero sentinel (PENDING)", "the bank receipt hash differs from the hash pinned in the module declaration", "the measure is not vllm-tokenize (estimate)",
            "a source half is the consuming roster", "a source is outside the declared construction halves (T0, Fold A, Fold B, DEV-1000 or any other roster)",
            "a source split identity is not the pinned construction identity", "a consuming row is a source row", "a token count is over its cap"],
        "iterations": iterations,
    }


def validate_bank_release(doc: object) -> list[str]:
    """Problems of a bank-release registration document (empty when it is well formed). Structural only; reads nothing."""

    problems: list[str] = []
    if not isinstance(doc, Mapping) or doc.get("schema") != RELEASE_SCHEMA:
        return [f"schema is not {RELEASE_SCHEMA!r}"]
    rule = doc.get("t0_never_a_bank_source")
    if not isinstance(rule, Mapping) or sorted(rule.get("allowed_source_halves") or []) != sorted(cm.HALVES) or "T0" not in str(rule.get("rule", "")):
        problems.append("the T0 rule is missing or does not restrict sources to the two construction halves")
    elif "T0 official SLURP test split" not in (rule.get("forbidden_rosters") or {}):
        problems.append("the forbidden rosters do not name the T0 official SLURP test split")
    elif any(r.get("rows_shared_with_r01c") for r in rule["forbidden_rosters"].values()):
        problems.append("a forbidden roster shares rows with the construction column")
    measure = doc.get("measure") or {}
    if measure.get("required") != rail.MEASURE_VLLM or rail.MEASURE_ESTIMATE not in (measure.get("refused") or []):
        problems.append("the measure must require vllm-tokenize and refuse estimate")
    iterations = doc.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        return problems + ["no iteration is registered"]
    for entry in iterations:
        label = f"iteration {entry.get('iteration')}"
        releases = entry.get("releases") or {}
        if sorted(releases) != sorted([f"half-{h}" for h in cm.HALVES] + ["union"]):
            problems.append(f"{label}: the releases must be the two half banks and the union")
            continue
        for name, release in releases.items():
            receipt = release.get("bank_receipt") or {}
            if not (isinstance(receipt.get("sha256"), str) and len(receipt["sha256"]) == 64):
                problems.append(f"{label} {name}: the bank receipt hash is not a sha256")
            if (release.get("measure") or {}).get("required") != rail.MEASURE_VLLM:
                problems.append(f"{label} {name}: the measure is not vllm-tokenize")
            if release.get("consuming_roster") in release.get("source_halves", []):
                problems.append(f"{label} {name}: a source half is the consuming roster")
            counts = release.get("token_counts") or {}
            if counts.get("exemplar_block_token_cap") != tf.EXEMPLAR_BLOCK_TOKEN_CAP or counts.get("rule_bank_token_cap") != tf.RULE_BANK_TOKEN_CAP:
                problems.append(f"{label} {name}: the token caps are not 2,000 / 400")
        union = releases["union"]
        if sorted(union.get("source_halves") or []) != sorted(cm.HALVES) or not (union.get("union_hash") or {}).get("sha256"):
            problems.append(f"{label}: the union bank must name both halves and carry a union hash")
        if not (union.get("consuming") or {}).get("cell_ids"):
            problems.append(f"{label}: the union bank names no consuming cell")
        for cell, frame in (entry.get("tfrl_v2_bank_frames") or {}).items():
            try:
                rail.parse_bank_declaration(frame)
            except rail.TfrlV2BankError as error:
                problems.append(f"{label}: the frame of {cell} does not parse: {error}")
    return problems


# ---- the writer step and the marker check -----------------------------------------------------------------------------------------------------------


def write_rails_files(root: Path, wjson, wr) -> dict:
    """Write the three package files with the writer's own ``reg.wjson`` / ``reg.wr`` (each returns the file sha256). ``root`` is the registration root."""

    shas = {RECIPE_NAME: wjson(root, RECIPE_RELPATH, recipe_document())}
    shas[READ_ROW_NAME] = wr(root, READ_ROW_RELPATH, read_row_text(shas[RECIPE_NAME]))
    shas[BANK_RELEASE_NAME] = wjson(root, BANK_RELEASE_RELPATH, bank_release_document(root))
    return shas


def rails_status(root: Path) -> dict:
    """Read each deliverable from disk and return the marker. ``rails_declared`` only when all three exist and check out; otherwise ``rails_pending`` with the failing item names."""

    root = Path(root)
    code = REPO_ROOT                                       # code deliverables live in the study tree; the package files live under the registration root
    items: dict[str, dict] = {}

    def ok(name: str, good: bool, **evidence: Any) -> None:
        items[name] = {"ok": bool(good), **evidence}

    try:
        record = s3.resolve_stage3_algorithm(rail.ALGORITHM_ID)
        label_ok = record.id == rail.ALGORITHM_ID
    except s3.Stage3AlgorithmError:
        label_ok = False
    rail_file = code / CONSUMPTION_RAIL
    from speech_aware_evidence_acquisition.rails import consumption_module as cmod  # noqa: PLC0415

    ok("stage3_label_and_consumption_rail", label_ok and rail_file.is_file() and rail.BLOCK_KEY in cmod._BLOCK_OPTIONAL,
       label=rail.ALGORITHM_ID, label_registered=label_ok, rail=CONSUMPTION_RAIL, rail_file=rail_file.is_file(), module_accepts_frame=rail.BLOCK_KEY in cmod._BLOCK_OPTIONAL)

    recipe_path, read_row_path, reader_path = root / RECIPE_RELPATH, root / READ_ROW_RELPATH, code / READER
    recipe_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest() if recipe_path.is_file() else None
    pinned = read_row_path.is_file() and recipe_sha is not None and f"`{recipe_sha}`" in read_row_path.read_text(encoding="utf-8")
    ok("analysis_recipe_and_reader", recipe_sha is not None and recipe_sha == recipe_sha256() and pinned and reader_path.is_file(),
       recipe=RECIPE_RELPATH, recipe_sha256=recipe_sha, read_row=READ_ROW_RELPATH, recipe_sha_pinned_in_read_row=bool(pinned), reader=READER, reader_file=reader_path.is_file())

    release_path = root / BANK_RELEASE_RELPATH
    problems: list[str] = ["file is missing"]
    if release_path.is_file():
        try:
            problems = validate_bank_release(json.loads(release_path.read_text(encoding="utf-8")))
        except ValueError:
            problems = ["file is not JSON"]
    ok("bank_release_registration", not problems, path=BANK_RELEASE_RELPATH, sha256=_sha_file(root, BANK_RELEASE_RELPATH) if release_path.is_file() else None, problems=problems)
    marker = MARKER_DECLARED if all(item["ok"] for item in items.values()) else MARKER_PENDING
    return {"marker": marker, "deliverables": items,
            "meaning": ("the rails are declared (label, consumption declaration with its refusals, analysis recipe and reader, bank-release registration); the five T cells keep their closest "
                        "registered labels and zero-sentinel bank pins until the main thread re-mints them at the flip")}


__all__ = [name for name in globals() if not name.startswith("_")]
