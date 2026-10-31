"""Export complete real500 evidence without scoring or inference.

Require the original fixed roster, finalized bundle, checkpoint accounting and
pinned start/seal boundary. Public output contains aggregates and artifact hashes.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/analysis")]
from register_real_engine_candidate import read, sha, write


def generation_duration(started, generation, seal_mtime, spec):
    if (started.get("candidate") != spec["cell"] or started.get("profile") != spec["profile"]
            or started.get("rows") != 500 or generation.get("status") != "COMPLETED"
            or generation.get("run_id") != spec["run_id"]):
        raise ValueError("start/seal identity mismatch")
    elapsed = seal_mtime - started["time"]
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("invalid generation duration")
    return elapsed


def reconcile_cost(checkpoint_rows, cost):
    totals = collections.Counter()
    for row in checkpoint_rows:
        totals.update(row.cost)
    mapping = {"calls": "calls_used", "prompt_tokens": "prompt_tokens_total",
               "completion_tokens": "completion_tokens_total", "audio_seconds": "audio_seconds_used"}
    for source, target in mapping.items():
        if not math.isclose(totals[source], cost[target], rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError("checkpoint/final cost mismatch: " + target)
    return dict(totals)


def export(repo, data, candidate, started_path, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    from speech_aware_evidence_acquisition.core.deployment import load_deployment_profile
    from prepare_real_engine_roster import prepare
    assert_execution_environment()
    spec = read(repo / "docs/readiness" / candidate / "candidate.json")
    if spec["cell"] != candidate or spec["rows"] != 500:
        raise ValueError("expected registered real500 candidate")
    directory = data / "runs" / spec["run_id"] / spec["attempt_id"]
    manifest_path = directory / (spec["run_id"] + ".run-manifest.json")
    generation_path = directory / (spec["run_id"] + ".generation-manifest.json")
    bundle = verify_bundle(manifest_path)
    if bundle.attempt_state != "FINALIZED" or bundle.manifest["status"] != "COMPLETED":
        raise ValueError("finalized official result required")
    fragments = bundle.manifest["config"]["fragments"]
    cfg = compose(repo, **fragments)
    if cfg.config_hash != bundle.manifest["config"]["config_hash"]:
        raise ValueError("execution configuration changed")
    _, roster = prepare(repo / "configs/dataset/slurp-speech-native-evo-fold-b-1080.json")
    if read(repo / "configs/dataset" / (fragments["dataset"] + ".json")) != roster:
        raise ValueError("not the fixed real500 runtime dataset")
    expected = {r["sample_id"] for r in roster["items"]}
    if len(bundle.outputs_rows) != 500 or {r["sample_id"] for r in bundle.outputs_rows} != expected:
        raise ValueError("result does not cover exact full roster")
    generation = read(generation_path)
    for key in ("config", "outputs", "cost", "row_checkpoint", "protocol_hash", "study_commit"):
        if generation[key] != bundle.manifest[key]:
            raise ValueError("generation/final manifest mismatch: " + key)
    cp_meta = bundle.manifest["row_checkpoint"]
    checkpoint = read_checkpoint(Path(cp_meta["path"]), expected_sha256=cp_meta["sha256"])
    if len(checkpoint.rows) != 500 or checkpoint.torn_tail_bytes or any(r.keep_defect() for r in checkpoint.rows):
        raise ValueError("complete intact checkpoint required")
    reconcile_cost(checkpoint.rows, bundle.manifest["cost"])
    score_meta = bundle.manifest["scores"]; score_path = Path(score_meta["path"])
    if sha(score_path) != score_meta["sha256"]:
        raise ValueError("official score pin differs")
    agg = read(score_path)["slurp-slu-official-train"]["aggregate"]
    if agg["samples"] != 500 or agg["split_id"] != "train" or not agg["identity_check"]["passed"]:
        raise ValueError("official scoring identity differs")
    started = read(started_path)
    elapsed = generation_duration(started, generation, generation_path.stat().st_mtime, spec)
    if started["study_commit"] != bundle.manifest["study_commit"]:
        raise ValueError("start/seal source commit differs")
    profile = load_deployment_profile(spec["profile"], repo)
    cards = sum(len(i.cards) for i in profile.instances)
    if cards != 8:
        raise ValueError("whole-machine comparison requires eight cards")
    def pin(p):
        return {"relpath": str(p.resolve().relative_to(data.resolve())), "sha256": sha(p)}
    result = {"schema": "saea-real500-verified-result-v1", "candidate": candidate, "run_id": spec["run_id"],
        "rows": 500, "remaining_rows": 0, "generation_sealed": True, "official_finalized": True,
        "cards": cards, "instances": len(profile.instances), "max_num_seqs_per_instance": profile.max_num_seqs,
        "profile": spec["profile"], "generation_start_to_seal_seconds": elapsed,
        "generation_seconds_per_row": elapsed / 500, "allocated_card_hours_until_seal": cards * elapsed / 3600,
        "quality": {k: agg["official"][k] for k in ("slu_f1", "intent_accuracy", "span_f1", "scenario_accuracy")},
        "quality_identity_passed": True, "cost": bundle.manifest["cost"],
        "checkpoint_cost_reconciled": True, "checkpoint_sha256": checkpoint.sha256,
        "split_identity_hash": spec["split_identity_hash"], "scores_sha256": score_meta["sha256"],
        "new_model_calls": 0, "new_scoring_jobs": 0,
        "selection_evidence": {"manifest": pin(manifest_path), "started": pin(started_path),
            "generation_manifest": pin(generation_path), "generation_elapsed_seconds": elapsed},
        "limitations": ["Generation duration includes runner preparation and seal; excludes deployment and official scoring.",
            "Allocated card-hours are not measured utilization; serving counters and FIT require separate reconciliation.",
            "Real500 train comparison does not replace full1080 A0 or full13078 test evaluation."]}
    if not out.resolve().is_relative_to((data / "work/codex").resolve()):
        raise ValueError("owned external output required")
    write(out, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "data", "started-path", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--candidate", required=True)
    print(json.dumps(export(**vars(p.parse_args()))))
