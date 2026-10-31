"""Prepare explicit A0 output custody and the exact remaining real-data roster.

Keep historical outputs first and only the first engineering candidate's missing
IDs. Preserve original canonical records and traces; never rewrite them into a
fictional c2 attempt. This preparation neither scores nor admits inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def partition(parent_ids, historical, candidate):
    parent = set(parent_ids)
    if len(parent) != len(parent_ids):
        raise ValueError("duplicate parent sample")
    if not set(historical) <= parent or not set(candidate) <= parent:
        raise ValueError("source outside parent population")
    selected = {sid: ("historical", historical[sid]) for sid in historical}
    selected.update({sid: ("first-engineering-candidate", candidate[sid])
                     for sid in sorted(set(candidate) - set(historical))})
    return selected, sorted(parent - set(selected))


def prepare(repo, execution, data, recipe_path, compatibility_path, replay_path, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash
    assert_execution_environment()
    if out.exists() or not out.resolve().is_relative_to((data / "work/codex").resolve()):
        raise ValueError("fresh owned preparation directory required")
    recipe = read(recipe_path)
    cp = (data / recipe["checkpoint"]["relpath"]).resolve()
    if not cp.is_relative_to(data.resolve()):
        raise ValueError("checkpoint escapes data root")
    historical = read_checkpoint(cp, expected_sha256=recipe["checkpoint"]["sha256"])
    if len(historical.rows) != 427 or historical.torn_tail_bytes or any(r.keep_defect() for r in historical.rows):
        raise ValueError("historical complete427 proof differs")
    replay = read(replay_path)
    if (replay["checkpoint_sha256"] != historical.sha256 or replay["rows"] != 427
            or replay["contacts"] != 854 or replay["vllm_wire_equal"] is not True
            or replay["delivered_output_equal"] is not True
            or replay["custody_deltas_sha256"] != sha(replay_path.parent / "custody-deltas.json")):
        raise ValueError("historical replay differs")
    compat = read(compatibility_path)
    if compat["replay_proof_sha256"] != sha(replay_path) or compat["rows"] != 500:
        raise ValueError("compatibility proof does not bind replay")
    for key in ("dataset_rows_equal", "model_content_config_equal", "baseline_method_equal", "prepared_evidence_equal"):
        if compat.get(key) is not True:
            raise ValueError("missing input equivalence: " + key)
    spec = read(execution / "docs/readiness/slurp-real500-tp4-s96-v1/candidate.json")
    if spec["run_id"] != "SAEA-SLURP-REAL500-TP4-S96-V1":
        raise ValueError("only the first engineering candidate is eligible")
    exp = execution / "configs/experiment" / (spec["experiment"] + ".json")
    cell, = read(exp)["cells"]
    for kind in ("model", "baseline", "dataset"):
        p = execution / "configs" / kind / (cell[kind + "_fragment"] + ".json")
        if sha(p) != compat["config_pins"]["candidate/" + kind]:
            raise ValueError("candidate config changed after compatibility audit")
    if sha(exp) != compat["config_pins"]["candidate/experiment"]:
        raise ValueError("candidate experiment differs")
    for name, pin in compat["source_changes"].items():
        if sha(execution / "src" / name) != pin["after"]:
            raise ValueError("candidate source delta changed")
    run = data / "runs" / spec["run_id"] / spec["attempt_id"]
    mp = run / (spec["run_id"] + ".run-manifest.json")
    bundle = verify_bundle(mp)
    if bundle.attempt_state != "FINALIZED" or bundle.manifest["status"] != "COMPLETED" or len(bundle.outputs_rows) != 500:
        raise ValueError("engineering candidate is not fully finalized")
    if bundle.manifest["protocol_hash"] != spec["protocol_hash"]:
        raise ValueError("candidate protocol differs")
    source_view = repo / "configs/dataset/slurp-speech-native-evo-fold-b-1080.json"
    parent = read(source_view)
    unsigned = {k: v for k, v in parent.items() if k != "runtime_view_sha256"}
    if hashlib.sha256(canonical(unsigned)).hexdigest() != parent["runtime_view_sha256"]:
        raise ValueError("parent runtime view differs")
    old = {}
    for row in historical.rows:
        output, = row.output_rows
        if output["sample_id"] != row.sample_id or row.sample_id in old:
            raise ValueError("historical output identity differs")
        old[row.sample_id] = output
    new = {r["sample_id"]: r for r in bundle.outputs_rows}
    chosen, remaining = partition([r["sample_id"] for r in parent["items"]], old, new)
    if len(chosen) != 727 or len(remaining) != 353 or len(parent["items"]) != 1080:
        raise ValueError("unexpected coverage partition")
    provenance = []
    historical_rows = historical.by_sample()
    for sid, (source, row) in sorted(chosen.items()):
        provenance.append({"sample_id": sid, "source": source,
            "source_sha256": historical.sha256 if source == "historical" else sha(mp),
            "canonical_output_sha256": hashlib.sha256(canonical(row)).hexdigest(),
            "trace_sha256": historical_rows[sid].trace_sha256 if source == "historical" else bundle.manifest["raw_trace"]["sha256"],
            "request_id": row["request_id"]})
    missing = set(remaining)
    selected_items = [r for r in parent["items"] if r["sample_id"] in missing]
    receipt = {"schema": "saea-a0-source-custody-v1", "historical_rows": 427,
        "engineering_additional_rows": 300, "covered_rows": 727, "remaining_rows": 353,
        "full_rows": 1080, "parent_dataset_sha256": sha(source_view),
        "parent_split_identity_hash": parent["split_identity_hash"],
        "sources": {"historical_checkpoint": {"relpath": str(cp.relative_to(data)), "sha256": historical.sha256},
                    "first_engineering_manifest": {"relpath": str(mp.relative_to(data)), "sha256": sha(mp)}},
        "compatibility_sha256": sha(compatibility_path), "replay_sha256": sha(replay_path),
        "selection": "historical first, then first engineering candidate for missing IDs; no score-based selection",
        "cost_policy": "Retain all historical generation/FIT costs and all engineering candidate costs, including overlaps and diagnostic branches; this is not a spend reset.",
        "new_model_calls": 0, "new_scoring_calls": 0, "scientific_baseline_complete": False,
        "registered_for_inference": False, "remaining": ["Admit and generate missing353 via shared runtime with checkpoint resume",
        "Verify missing353 method/source custody and disjoint union", "One official full1080 score from source-bound outputs; no averaging subset metrics"]}
    view = {**unsigned, "split_name": "SLURP-A0-REMAINING-353-V1", "count": 353,
            "split_identity_hash": split_identity_hash(prefixed("slurp", remaining)),
            "items": selected_items, "audio_seconds_total": round(sum(r["audio_seconds"] for r in selected_items), 9),
            "source_binding_sha256": hashlib.sha256(canonical(receipt)).hexdigest()}
    view["runtime_view_sha256"] = hashlib.sha256(canonical(view)).hexdigest()
    out.mkdir(parents=True, exist_ok=False)
    artifacts = {"custody.json": receipt, "source-map.json": provenance, "remaining-dataset.json": view}
    for name, doc in artifacts.items():
        with (out / name).open("x") as f:
            json.dump(doc, f, indent=2); f.write("\n")
    with (out / "preserved-canonical-outputs.jsonl").open("x") as f:
        for sid in sorted(chosen):
            f.write(json.dumps(chosen[sid][1], ensure_ascii=False, sort_keys=True) + "\n")
    summary = {k: receipt[k] for k in ("covered_rows", "remaining_rows", "full_rows", "new_model_calls", "new_scoring_calls", "scientific_baseline_complete", "registered_for_inference")}
    summary.update(schema="saea-a0-custody-preparation-summary-v1", artifacts={p.name: sha(p) for p in out.iterdir()}, remaining_split_identity_hash=view["split_identity_hash"])
    with (out / "complete.json").open("x") as f:
        json.dump(summary, f, indent=2); f.write("\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "execution", "data", "recipe-path", "compatibility-path", "replay-path", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    print(json.dumps(prepare(**vars(p.parse_args()))))
