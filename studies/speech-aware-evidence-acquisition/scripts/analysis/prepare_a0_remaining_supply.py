"""Prepare the exact missing A0 rows while topology measurements execute.

Use the shared production FlightRuntime with the frozen two-contact baseline.
No inference, tokenizer, scoring, admission or readiness authorization occurs.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(execution, data, custody, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from speech_aware_evidence_acquisition.core.config import compose
    assert_execution_environment()
    if out.exists() or not out.resolve().is_relative_to((data / "work/codex").resolve()):
        raise ValueError("fresh owned scratch directory required")
    summary = read(custody / "complete.json")
    for name, expected in summary["artifacts"].items():
        if Path(name).name != name or sha(custody / name) != expected:
            raise ValueError("custody artifact differs")
    view = read(custody / "remaining-dataset.json")
    if view["count"] != 353 or len(view["items"]) != 353:
        raise ValueError("expected exact missing353")
    spec = read(execution / "docs/readiness/slurp-real500-tp4-s96-v1/candidate.json")
    old = read(execution / "configs/experiment" / (spec["experiment"] + ".json"))
    old_cell, = old["cells"]
    baseline = read(execution / "configs/baseline" / (old_cell["baseline_fragment"] + ".json"))
    baseline.pop("prepared_phase_supply")
    name = "slurp-a0-remaining353-c2-v1"
    run = "SAEA-SLURP-A0-REMAINING353-C2-V1"
    attempt = "attempt-a0-remaining353-c2-v1"
    baseline["native_contact_fit"].update(run_id=run + "-FIT", attempt_id=attempt + "-fit", contact_cap=706)
    exp = copy.deepcopy(old)
    cell, = exp["cells"]
    cell.update(cell_id=name, run_id=run, attempt_id=attempt, baseline_fragment=name, dataset_fragment=name)
    cell["readiness"] = {"state": "held", "reason": "CPU preparation only; final selected serving profile and admission pending", "requirements": []}
    exp.update(experiment_id=run, split_id=view["split_name"], purpose="Complete missing A0 baseline rows after explicit source custody; preparation only",
               service_session="serving-a0-final-profile-pending", population={"count": 353, "dataset_fragment": name, "split_identity_hash": view["split_identity_hash"]},
               populations=[{"dataset_fragment": name, "ids": [r["sample_id"] for r in view["items"]]}])
    exp["execution_plan"].update(planned_model_calls=1412, planned_speech_audio_seconds=math.ceil(4 * view["audio_seconds_total"]))
    exp["stage3_measurement"].update(rows=353, claims_benefit=False, note="Missing subset preparation; full1080 source-bound official scoring required for baseline claim")
    out.mkdir(parents=True, exist_ok=False)
    study = out / "studies/speech-aware-evidence-acquisition"; study.mkdir(parents=True)
    for directory in ("configs", "docs", "scripts", "src"):
        shutil.copytree(execution / directory, study / directory)
    (out / "docs").mkdir()
    lock = execution.parents[1] / "docs/datasets.lock.json"
    shutil.copyfile(lock, out / "docs/datasets.lock.json")
    for kind, doc in (("baseline", baseline), ("dataset", view), ("experiment", exp)):
        (study / "configs" / kind / (name + ".json")).write_text(json.dumps(doc, indent=2) + "\n")
    runtime = build_flight_runtime(study, data, model=cell["model_fragment"], dataset=name,
        baseline=name, experiment=name, run_id=run, attempt_id=attempt,
        runtime_root=data / "work/scratch/runtime-root", lock_path=lock)
    runtime.prepare_phase_supply(out / "phase-supply.json", max_workers=16)
    full = read(data / "work/codex/lock-supply-equivalence-v3/c2-phase-supply.json")["output"]
    new = read(out / "phase-supply.json")["output"]
    from audit_real500_a0_inputs import compare_supply
    compared = compare_supply(full, new, [r["sample_id"] for r in view["items"]])
    report = {"schema": "saea-a0-remaining-preparation-v1", "rows": len(runtime._samples),
        "split_identity_hash": view["split_identity_hash"], "custody_summary_sha256": sha(custody / "complete.json"),
        "phase_supply_sha256": sha(out / "phase-supply.json"), "evidence_equal_to_full_parent": True,
        "compared_supply_entries": compared, "new_model_calls": 0, "registered_for_inference": False,
        "state": "prepared-held", "normal_generation_calls": 706, "fallback_inclusive_cap": 1412,
        "remaining": ["Freeze measured final serving profile", "Register generation/FIT and source-custody full1080 read", "Admit, repin and preflight before dispatch"]}
    with (out / "complete.json").open("x") as f:
        json.dump(report, f, indent=2); f.write("\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("execution", "data", "custody", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    print(json.dumps(prepare(**vars(p.parse_args()))))
