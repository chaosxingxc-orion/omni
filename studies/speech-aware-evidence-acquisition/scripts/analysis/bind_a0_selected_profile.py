"""Bind held missing353 work to a verified complete-roster engineering decision.

This is offline preparation, not deployment or admission. All compared runs must
be finalized; retain each full result and the explicit throughput/quality reason.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/analysis"), str(ROOT / "scripts/flights")]
from register_a0_remaining import NAME, RUN
from register_real_engine_candidate import read, sha, write
from score_a0_full1080 import pin


def verify_selection(data, decision):
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    required = {"SAEA-SLURP-REAL500-TP4-S96-V1", "SAEA-SLURP-REAL500-TP4-S128-V1", "SAEA-SLURP-REAL500-TP8-S256-V1"}
    if decision.get("schema") != "saea-real500-profile-selection-v1" or decision.get("complete_roster_comparison") is not True or not decision.get("reason"):
        raise ValueError("explicit complete-roster selection required")
    runs, population = set(), None
    selected = None
    for item in decision["candidates"]:
        path = pin(data, item["manifest"])
        bundle = verify_bundle(path)
        rid = bundle.manifest["run_id"]
        if rid in runs or bundle.attempt_state != "FINALIZED" or len(bundle.outputs_rows) != 500:
            raise ValueError("duplicate or incomplete comparison candidate")
        current = {row["sample_id"] for row in bundle.outputs_rows}
        if len(current) != 500 or (population is not None and population != current):
            raise ValueError("comparison populations differ")
        population = current; runs.add(rid)
        timing = item["generation_elapsed_seconds"]
        if not isinstance(timing, (float, int)) or isinstance(timing, bool) or timing <= 0:
            raise ValueError("positive whole-generation timing required")
        started = read(pin(data, item["started"]))
        seal = pin(data, item["generation_manifest"])
        generation = read(seal)
        if generation["run_id"] != rid or generation["status"] != "COMPLETED" or abs(seal.stat().st_mtime - started["time"] - timing) > .01:
            raise ValueError("generation timing disagrees with sealed source boundaries")
        if rid == decision["selected_run_id"]:
            selected = bundle
    if not required <= runs or not runs <= required | {"SAEA-SLURP-REAL500-TP8-S384-V1"} or selected is None:
        raise ValueError("required engineering comparison missing or unknown candidate")
    return selected


def bind(repo, data, source_repo, selection_path):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.core.flight.repin import repin_package, rewrite_dependent_hashes
    from refresh_registered_plan_pins import refresh_package
    from audit_real500_a0_inputs import compare_method, compare_runtime, compare_source
    assert_execution_environment()
    if not selection_path.resolve().is_relative_to(data.resolve()):
        raise ValueError("selection must be retained under data root")
    decision = read(selection_path); selected = verify_selection(data, decision)
    pkg = repo / "docs/readiness" / NAME
    spec = read(pkg / "candidate.json")
    exp_path = repo / "configs/experiment" / (NAME + ".json")
    exp = read(exp_path); cell, = exp["cells"]
    if cell["readiness"]["state"] != "held" or spec.get("profile_selection_state") != "pending":
        raise ValueError("only unbound held continuation may be selected")
    for rid in (RUN, RUN + "-FIT"):
        if (data / "attempts" / rid).exists() or (data / "runs" / rid).exists():
            raise ValueError("existing scientific spend identity")
    if (data / spec["decision_relpath"]).exists():
        raise ValueError("existing admission cannot be rebound")
    fragments = selected.manifest["config"]["fragments"]
    selected_config = compose(source_repo, **fragments)
    if selected_config.config_hash != selected.manifest["config"]["config_hash"]:
        raise ValueError("selected runtime source config differs")
    selected_spec = read(source_repo / "docs/readiness" / fragments["experiment"] / "candidate.json")
    if selected_spec["profile"] != decision["profile"]:
        raise ValueError("decision profile differs from measured profile")
    old_model = read(repo / "configs/model" / (cell["model_fragment"] + ".json"))
    new_model = read(source_repo / "configs/model" / (fragments["model"] + ".json"))
    base = read(repo / "configs/baseline" / (NAME + ".json"))
    compare_method(old_model, new_model, base, read(source_repo / "configs/baseline" / (fragments["baseline"] + ".json")))
    compare_runtime(read(repo / "docs" / old_model["runtime_receipt"]), read(source_repo / "docs" / new_model["runtime_receipt"]))
    compare_source(repo / "src", source_repo / "src")
    before = compose(repo, cell["model_fragment"], NAME, NAME, NAME)
    old_session = spec["service_session"]
    # Copy only the measured model fragment and its receipt, never scientific src.
    for rel in (Path("configs/model") / (fragments["model"] + ".json"), Path("docs") / new_model["runtime_receipt"]):
        target = repo / rel; content = (source_repo / rel).read_bytes()
        if target.exists() and target.read_bytes() != content:
            # Unused candidate fragments may still contain their held receipt;
            # archive bytes before replacing an operational identity.
            archive = data / "work/codex" / (NAME + "-profile-binding-before") / rel
            archive.parent.mkdir(parents=True, exist_ok=True)
            with archive.open("xb") as f: f.write(target.read_bytes())
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
    cell["model_fragment"] = fragments["model"]; exp["service_session"] = selected_spec["service_session"]
    exp_path.write_text(json.dumps(exp, indent=2, sort_keys=True) + "\n")
    spec.update(model_fragment=fragments["model"], service_session=selected_spec["service_session"],
        runtime_receipt=new_model["runtime_receipt"], profile=decision["profile"], profile_selection_state="selected",
        profile_selection={"relpath": str(selection_path.relative_to(data)), "sha256": sha(selection_path)})
    (pkg / "candidate.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    protocol_path = pkg / "protocol-registered-v1.json"; protocol = read(protocol_path)
    protocol.update(profile=decision["profile"], profile_binding="selected from complete real500 comparison", profile_selection=spec["profile_selection"])
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    # Rebind old declared identities before the shared repinner rebuilds evidence.
    from speech_aware_evidence_acquisition.core.config import protocol_hash
    current = compose(repo, cell["model_fragment"], NAME, NAME, NAME)
    split = spec["split_identity_hash"]
    mapping = {before.config_hash: current.config_hash, protocol_hash(before.config_hash,before.values["arm"],split):protocol_hash(current.config_hash,current.values["arm"],split)}
    mapping[old_session] = selected_spec["service_session"]
    mapping = {k:v for k,v in mapping.items() if k != v}
    prereg = repo / "docs/readiness" / (NAME + "-preregistration.md")
    rewrite_dependent_hashes(repo_root=repo, readiness_dir=pkg, preregistration_path=prereg,
        exposure_ledger_path=repo / "docs/exposure-ledger.md", run_ids={RUN,RUN+"-FIT"}, mapping=mapping)
    result = repin_package(repo_root=repo, data_root=data, runtime_root=data / "work/scratch/runtime-root",
        lock_path=repo.parents[1] / "docs/datasets.lock.json", experiments=[(NAME,NAME)], readiness_dir=pkg,
        preregistration_path=prereg, set_ready=False, allow_output_change=False, run_check_fn=None)
    refresh_package(repo,pkg)
    rewrite_dependent_hashes(repo_root=repo, readiness_dir=pkg, preregistration_path=prereg,
        exposure_ledger_path=repo / "docs/exposure-ledger.md", run_ids={RUN,RUN+"-FIT"}, mapping={})
    write(pkg / "profile-binding.json", {"new_model_calls":0,"repin":result,"selection":spec["profile_selection"],
        "readiness":"held","remaining":["matching live resident","admission","clean source","shared preflight","detached dispatch"]})
    return {"new_model_calls":0,"profile":decision["profile"],"ready":False}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("repo","data","source-repo","selection-path"):
        p.add_argument("--"+name,type=Path,required=True)
    print(json.dumps(bind(**vars(p.parse_args()))))
