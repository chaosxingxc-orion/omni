"""Register the exact missing A0 population without admission or model contacts.

The selected engineering candidate supplies a provisional service binding. A
separate measured-profile selection must precede admission. This never imports
answers into runtime or alters the historical source runs.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/analysis")]
from register_real_engine_candidate import read, sha, write
from score_a0_full1080 import pin, validate_recipe, verify_sources

NAME = "slurp-a0-remaining353-c2-v1"
RUN = "SAEA-SLURP-A0-REMAINING353-C2-V1"
ATTEMPT = "attempt-a0-remaining353-c2-v1"
AUTHORITY = "docs/decisions/2026-10-31-owner-lock-baseline-and-cost-repair.md"


def derive_documents(experiment, baseline, view, candidate):
    ids = [row["sample_id"] for row in view["items"]]
    if view["count"] != 353 or len(ids) != 353 or len(set(ids)) != 353:
        raise ValueError("exact distinct missing353 required")
    exp, base = copy.deepcopy(experiment), copy.deepcopy(baseline)
    cell, = exp["cells"]
    cell.update(cell_id=NAME, run_id=RUN, attempt_id=ATTEMPT,
                baseline_fragment=NAME, dataset_fragment=NAME,
                model_fragment=candidate["model_fragment"])
    cell["readiness"] = {"state": "held", "reason": "Measured final profile selection and admission pending", "requirements": []}
    exp.update(experiment_id=RUN, service_session=candidate["service_session"],
               split_id=view["split_name"], purpose="Complete exact missing A0 rows; source-bound full1080 scoring follows",
               population={"count": 353, "dataset_fragment": NAME, "split_identity_hash": view["split_identity_hash"]},
               populations=[{"dataset_fragment": NAME, "ids": ids}])
    exp["stage3_algorithm"] = {"id": "none", "operators_used": [], "note": "Frozen two-contact A0 baseline continuation; no RL operator"}
    exp["stage3_measurement"].update(rows=353, read_roster=view["split_name"], claims_benefit=False,
        note="Missing353 generation only; baseline claim requires full1080 source-bound official read")
    exp["execution_plan"].update(planned_model_calls=1412,
        planned_speech_audio_seconds=math.ceil(4 * view["audio_seconds_total"]), planned_gpu_hours=16.0)
    base.pop("prepared_phase_supply", None)
    base["native_contact_fit"].update(run_id=RUN + "-FIT", attempt_id=ATTEMPT + "-fit", contact_cap=706)
    return exp, base


def prepare(repo, data, engineering_candidate):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment, ExecutionPlan
    from speech_aware_evidence_acquisition.core.config import compose, protocol_hash
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from speech_aware_evidence_acquisition.core.flight.campaign import load_campaign_cells
    from audit_real500_a0_inputs import compare_supply
    assert_execution_environment()
    recipe_path = repo / "docs/readiness/2026-10-31-a0-full1080-source-read/analysis-recipe-v1.json"
    recipe = read(recipe_path); validate_recipe(recipe)
    sources = verify_sources(repo, data, recipe)
    view = sources["remaining_view"]
    pkg = repo / "docs/readiness" / NAME
    if pkg.exists():
        raise ValueError("existing registration; inspect before any recovery")
    ledger = repo / "docs/exposure-ledger.md"
    for rid in (RUN, RUN + "-FIT", RUN + "-READ"):
        if "| " + rid + " |" in ledger.read_text() or any((data / k / rid).exists() for k in ("runs", "attempts", "prepared")):
            raise ValueError("existing registration or spend identity")
    candidate = read(repo / "docs/readiness" / engineering_candidate / "candidate.json")
    if candidate["rows"] != 500 or engineering_candidate not in {
        "slurp-real500-tp4-s96-v1", "slurp-real500-tp4-s128-v1", "slurp-real500-tp8-s256-v1", "slurp-real500-tp8-s384-v1"
    }:
        raise ValueError("unexpected provisional engineering profile")
    old_exp = read(repo / "configs/experiment" / (engineering_candidate + ".json"))
    old_cell, = old_exp["cells"]
    old_base = read(repo / "configs/baseline" / (old_cell["baseline_fragment"] + ".json"))
    exp, base = derive_documents(old_exp, old_base, view, candidate)
    for kind, doc in (("experiment", exp), ("baseline", base), ("dataset", view)):
        write(repo / "configs" / kind / (NAME + ".json"), doc)
    load_campaign_cells(repo, NAME)
    lock = repo.parents[1] / "docs/datasets.lock.json"
    def runtime():
        return build_flight_runtime(repo, data, model=candidate["model_fragment"], dataset=NAME,
            baseline=NAME, experiment=NAME, run_id=RUN, attempt_id=ATTEMPT,
            runtime_root=data / "work/scratch/runtime-root", lock_path=lock)
    supply = data / "prepared" / RUN / "phase-supply-v1.json"
    runtime().prepare_phase_supply(supply, max_workers=16)
    full = read(pin(data, recipe["full_parent_supply"]))["output"]
    compare_supply(full, read(supply)["output"], sources["missing"])
    supply_pin = {"relpath": str(supply.relative_to(data)), "sha256": sha(supply)}
    base["prepared_phase_supply"] = supply_pin
    exp["cells"][0]["readiness"]["requirements"] = [{"kind": "preparation", **supply_pin}]
    for kind, doc in (("experiment", exp), ("baseline", base)):
        (repo / "configs" / kind / (NAME + ".json")).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    cfg = compose(repo, candidate["model_fragment"], NAME, NAME, NAME)
    phash = protocol_hash(cfg.config_hash, cfg.values["arm"], view["split_identity_hash"])
    identity = {"cell": NAME, "experiment": NAME, "run_id": RUN, "attempt_id": ATTEMPT,
        "fit_run_id": RUN + "-FIT", "fit_attempt_id": ATTEMPT + "-fit", "rows": 353,
        "composed_config_hash": cfg.config_hash, "protocol_hash": phash, "split_identity_hash": view["split_identity_hash"]}
    spec = {**identity, "workload_kind": "a0-exact-remaining", "model_fragment": candidate["model_fragment"],
        "baseline": NAME, "service_session": candidate["service_session"], "profile": candidate["profile"],
        "runtime_receipt": candidate["runtime_receipt"], "decision_id": NAME + "-owner-admission",
        "decision_relpath": f"coordination/{NAME}/root-admission-v1.json", "authority": AUTHORITY,
        "source_read_recipe": {"relpath": str(recipe_path.relative_to(repo)), "sha256": sha(recipe_path)},
        "source_custody": recipe["custody_summary"], "profile_selection_required": True,
        "profile_selection_state": "pending", "scientific_readiness": "held", "new_model_calls": 0}
    write(pkg / "candidate.json", spec)
    protocol = {"schema": "saea-a0-remaining-protocol-v1", "cells": {"arm": identity},
        "profile": candidate["profile"], "profile_binding": "provisional until measured selection",
        "source_custody": recipe["custody_summary"], "full1080_read_recipe": spec["source_read_recipe"],
        "prepared_phase_supply": supply_pin, "claims_benefit": False,
        "inference": "Exact353 real rows through fresh P->S; no response replay", "normal_generation_calls": 706,
        "fallback_inclusive_cap": 1412, "budget_basis": "16 allocated card-hours is a cap, not an ETA"}
    protocol_path = pkg / "protocol-registered-v1.json"; write(protocol_path, protocol)
    rows = []
    for kind, rid, aid, calls in (("generation", RUN, ATTEMPT, 1412), ("fit", RUN + "-FIT", ATTEMPT + "-fit", 706)):
        budget = {**exp["execution_plan"], "planned_model_calls": calls}
        ExecutionPlan(run_id=rid, attempt_id=aid, carrier_lock_key="slurp", split_identity_hash=view["split_identity_hash"], protocol_hash=phash, **budget).validate(source_root=repo)
        plan = {"schema": f"saea-registered-{kind}-plan-v1", "plan_kind": kind, "cell": NAME,
            "experiment": NAME, "composed_config_hash": cfg.config_hash, "expected_session_id": candidate["service_session"],
            "execution_plan": {**budget, "schema": "saea-execution-plan-v1", "study": "speech-aware-evidence-acquisition",
                "run_id": rid, "attempt_id": aid, "carrier_lock_key": "slurp", "protocol_hash": phash,
                "split_identity_hash": view["split_identity_hash"]}, "prepared_phase_supply": supply_pin,
            "protocol_document": {"relpath": str(protocol_path.relative_to(repo)), "sha256": sha(protocol_path)},
            "status": "REGISTERED BEFORE CONTACT; held for measured profile selection and admission"}
        path = pkg / f"{kind}-plan-registered-v1.json"; write(path, plan)
        rows.append(["2026-10-31", rid, "bounded-discovery-probe", "Exact missing A0 " + kind,
            candidate["model_fragment"], "SLURP TRAIN Fold B remaining353", "discovery", view["split_identity_hash"],
            "no", str(calls), str(budget["planned_speech_audio_seconds"]), "16.0", "All original source costs retained",
            str(path.relative_to(repo)), f"Plan SHA256 {sha(path)}; root countersigned under {AUTHORITY}, coordinator Codex; held; no test tuning"])
    analysis = {"schema": "saea-a0-remaining-read-v1", "read_run_id": RUN + "-READ", "rows": 353,
        "scope": "Official complete continuation score, descriptive only; full baseline uses separately registered1080 read",
        "new_model_calls": 0, "claims_benefit": False, "full1080_read_recipe": spec["source_read_recipe"]}
    path = pkg / "analysis-recipe-v1.json"; write(path, analysis)
    rows.append(["2026-10-31", RUN + "-READ", "model-free-check", "Complete continuation descriptive read", "no model",
        "SLURP TRAIN Fold B remaining353", "discovery", view["split_identity_hash"], "no", "0", "0", "0",
        "Generation/FIT costs retained", str(path.relative_to(repo)), f"Recipe SHA256 {sha(path)}; root countersigned under {AUTHORITY}, coordinator Codex; full1080 claim deferred"])
    with ledger.open("a") as f:
        f.write("\n## A0 exact remaining353\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| " + " | ".join(["---"] * 15) + " |\n")
        f.write("\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n")
    (repo / "docs/readiness" / (NAME + "-preregistration.md")).write_text(
        "# Exact missing A0 continuation\n\nRegister only the353 missing real train rows after pinned727-source verification. "
        "No historical answers enter runtime. Normal706 generation, cap1412 including fallback, separate FIT706. "
        "Measured profile selection, observed resident, admission, repin, clean source and shared preflight are required before dispatch. "
        "Preserve all historical and in-flight costs. Score full1080 once through the separately registered source-bound read.\n")
    checks = runtime().verify_prepared_artifacts()
    if not checks or not all(c["identity_ok"] and c.get("seed_ok", True) for c in checks):
        raise ValueError("prepared continuation identity failed")
    result = {"rows": 353, "checks": checks, "prepared_evidence_equal": True, "new_model_calls": 0,
        "readiness": "held", "profile_selection_state": "pending", "source_read_recipe_sha256": sha(recipe_path)}
    write(pkg / "preparation-check.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "data"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--engineering-candidate", required=True)
    print(json.dumps(prepare(**vars(p.parse_args()))))
