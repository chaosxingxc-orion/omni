"""Verify source-bound A0 custody and score its complete1080 union once.

This is an explicit analysis derivative, never a fabricated generation bundle.
The existing bundle verifier checks each generated source; original checkpoint
bytes retain the historical lock output. No inference or response rewriting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/analysis"), str(ROOT / "scripts/checks")]
from prepare_a0_source_custody import canonical, read, sha, partition


def pin(root, value):
    path = (root / value["relpath"]).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path) != value["sha256"]:
        raise ValueError("source pin differs or escapes root")
    return path


def unique_rows(rows):
    result = {}
    for row in rows:
        if row["sample_id"] in result:
            raise ValueError("duplicate output sample")
        result[row["sample_id"]] = row
    return result


def validate_recipe(recipe):
    expected = {
        "schema": "saea-a0-full1080-read-v1",
        "read_run_id": "SAEA-SLURP-A0-FULL1080-SOURCE-READ-V1",
        "continuation_run_id": "SAEA-SLURP-A0-REMAINING353-C2-V1",
        "rows": 1080,
        "new_model_calls": 0,
        "new_official_scoring_jobs": 1,
        "scorer": "slurp-slu-official-train",
        "claims_full_roster_benefit_before_completion": False,
        "selection": "Historical427 first, first real500 candidate missing300, exact remaining353; no outcome selection",
    }
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError("wrong full1080 read scope or scoring recipe")


def complete_union(parent_ids, preserved, missing, new):
    parent, expected = set(parent_ids), set(missing)
    if len(parent) != len(parent_ids) or len(expected) != len(missing):
        raise ValueError("duplicate roster identity")
    if set(preserved) & expected or set(preserved) | expected != parent:
        raise ValueError("preserved/remaining partition differs")
    if set(new) != expected:
        raise ValueError("continuation must cover exactly the missing roster")
    return [({**preserved, **new})[sid] for sid in sorted(parent)]


def verify_sources(repo, data, recipe):
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash
    summary_path = pin(data, recipe["custody_summary"])
    root = summary_path.parent
    summary = read(summary_path)
    for name, digest in summary["artifacts"].items():
        if Path(name).name != name or sha(root / name) != digest:
            raise ValueError("custody artifact changed")
    custody = read(root / "custody.json")
    parent = read(pin(repo, recipe["parent_dataset"]))
    if custody["parent_dataset_sha256"] != recipe["parent_dataset"]["sha256"]:
        raise ValueError("parent population binding differs")
    parent_ids = [x["sample_id"] for x in parent["items"]]
    if len(parent_ids) != 1080 or split_identity_hash(prefixed("slurp", parent_ids)) != custody["parent_split_identity_hash"]:
        raise ValueError("full1080 population differs")
    compat_path = pin(data, recipe["input_compatibility"])
    replay_path = pin(data, recipe["request_replay"])
    if sha(compat_path) != custody["compatibility_sha256"] or sha(replay_path) != custody["replay_sha256"]:
        raise ValueError("compatibility custody differs")
    cp = pin(data, custody["sources"]["historical_checkpoint"])
    historical = read_checkpoint(cp, expected_sha256=custody["sources"]["historical_checkpoint"]["sha256"])
    if len(historical.rows) != 427 or historical.torn_tail_bytes or any(r.keep_defect() for r in historical.rows):
        raise ValueError("historical427 incomplete")
    old = unique_rows([r.output_rows[0] for r in historical.rows])
    mp = pin(data, custody["sources"]["first_engineering_manifest"])
    bundle = verify_bundle(mp)
    if bundle.attempt_state != "FINALIZED" or bundle.manifest["run_id"] != "SAEA-SLURP-REAL500-TP4-S96-V1":
        raise ValueError("first engineering candidate differs or is not finalized")
    selected, missing = partition(parent_ids, old, unique_rows(bundle.outputs_rows))
    if len(selected) != 727 or len(missing) != 353:
        raise ValueError("expected727/353 source partition")
    saved = unique_rows([json.loads(line) for line in (root / "preserved-canonical-outputs.jsonl").read_text().splitlines()])
    source_map = unique_rows(read(root / "source-map.json"))
    if set(saved) != set(selected) or set(source_map) != set(selected):
        raise ValueError("custody inventory differs")
    historical_by_id = historical.by_sample()
    for sid, (source, row) in selected.items():
        if saved[sid] != row:
            raise ValueError("preserved canonical output differs from original source")
        expected = {"sample_id": sid, "source": source,
            "source_sha256": historical.sha256 if source == "historical" else sha(mp),
            "canonical_output_sha256": hashlib.sha256(canonical(row)).hexdigest(),
            "trace_sha256": historical_by_id[sid].trace_sha256 if source == "historical" else bundle.manifest["raw_trace"]["sha256"],
            "request_id": row["request_id"]}
        if source_map[sid] != expected:
            raise ValueError("source map differs from original trace custody")
    view = read(root / "remaining-dataset.json")
    if sorted(x["sample_id"] for x in view["items"]) != missing:
        raise ValueError("remaining roster differs")
    return {"parent_ids": parent_ids, "preserved": saved, "missing": missing,
            "candidate_bundle": bundle, "custody": custody, "remaining_view": view}


def validate_continuation(repo, data, manifest, source):
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.config import compose, effective_request_decoding_params
    from audit_real500_a0_inputs import compare_method, compare_supply, compare_runtime, compare_source
    bundle = verify_bundle(manifest)
    if bundle.manifest["status"] != "COMPLETED" or bundle.attempt_state != "FINALIZED":
        raise ValueError("continuation must be finalized")
    if bundle.manifest["run_id"] != "SAEA-SLURP-A0-REMAINING353-C2-V1":
        raise ValueError("unexpected continuation run")
    current = bundle.manifest["config"]
    fragments = current["fragments"]
    cfg = compose(repo, **fragments)
    if cfg.config_hash != current["config_hash"]:
        raise ValueError("continuation configuration differs from generated bundle")
    old_config = source["candidate_bundle"].manifest["config"]
    # The frozen candidate source tree is explicit in the registration recipe;
    # its method is rechecked independently when constructing this analysis.
    reference = Path(source["reference_repo"])
    previous = compose(reference, **old_config["fragments"])
    if previous.config_hash != old_config["config_hash"]:
        raise ValueError("reference configuration changed")
    def fragment(root, fields, kind):
        return read(root / "configs" / kind / (fields[kind] + ".json"))
    compare_method(fragment(reference, old_config["fragments"], "model"), fragment(repo, fragments, "model"),
                   fragment(reference, old_config["fragments"], "baseline"), fragment(repo, fragments, "baseline"))
    if effective_request_decoding_params(previous.values) != effective_request_decoding_params(cfg.values):
        raise ValueError("effective decoding differs")
    old_model = fragment(reference, old_config["fragments"], "model")
    new_model = fragment(repo, fragments, "model")
    compare_runtime(read(reference / "docs" / old_model["runtime_receipt"]),
                    read(repo / "docs" / new_model["runtime_receipt"]))
    compare_source(reference / "src", repo / "src")
    dataset = fragment(repo, fragments, "dataset")
    if dataset != source["remaining_view"]:
        raise ValueError("continuation dataset differs from exact remaining view")
    # Compare to full parent evidence because missing353 and engineering500 differ.
    full = read(pin(data, source["full_supply_pin"]))["output"]
    compare_supply(full, read(pin(data, cfg.values["prepared_phase_supply"]))["output"], source["missing"])
    return bundle


def run(repo, data, recipe_path, out, continuation_repo=None, continuation_manifest=None, validate_only=False):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.data.lock import load_lock, umbrella_lock_path
    from speech_aware_evidence_acquisition.reproduction.slurp_official_eval import score_slurp_outputs_official_train
    import flight_registration_check as registration
    assert_execution_environment()
    recipe = read(recipe_path)
    validate_recipe(recipe)
    entries = registration.parse_ledger(repo, Path("docs/exposure-ledger.md")).get(recipe["read_run_id"], [])
    if len(entries) != 1 or not registration.is_countersigned(entries[0]) or entries[0]["artifacts"] != str(recipe_path.relative_to(repo)) or sha(recipe_path) not in entries[0].get("notes", ""):
        raise ValueError("full1080 READ not uniquely registered and pinned")
    source = verify_sources(repo, data, recipe)
    source["reference_repo"] = str((data / recipe["reference_repo_relpath"]).resolve())
    if not Path(source["reference_repo"]).is_relative_to(data.resolve()):
        raise ValueError("reference source escapes data root")
    source["full_supply_pin"] = recipe["full_parent_supply"]
    pin(data, source["full_supply_pin"])
    if out.exists() or not out.resolve().is_relative_to((data / "analysis").resolve()):
        raise ValueError("fresh external analysis directory required")
    if validate_only:
        result = {"schema": "saea-a0-full1080-preflight-v1", "verified_preserved_rows": 727,
                  "required_remaining_rows": 353, "new_model_calls": 0, "new_scoring_calls": 0,
                  "full_baseline_complete": False, "recipe_sha256": sha(recipe_path)}
        out.mkdir(parents=True); (out / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    require_clean_study_repo(repo)
    if continuation_repo is None or continuation_manifest is None:
        raise ValueError("full remaining353 source is required before scoring")
    bundle = validate_continuation(continuation_repo, data, continuation_manifest, source)
    # Reuse the standard scoring freeze/carrier/clean-source checks on both
    # verified generated sources before opening any labels or scoring outputs.
    from speech_aware_evidence_acquisition.core.scorers import _assert_scoring_entry_gate
    lock = load_lock(umbrella_lock_path(repo))
    _assert_scoring_entry_gate(source["candidate_bundle"], lock, data, Path(source["reference_repo"]))
    _assert_scoring_entry_gate(bundle, lock, data, continuation_repo)
    rows = complete_union(source["parent_ids"], source["preserved"], source["missing"], unique_rows(bundle.outputs_rows))
    out.mkdir(parents=True)
    (out / "intent.json").write_text(json.dumps({"recipe_sha256": sha(recipe_path), "continuation_manifest_sha256": sha(continuation_manifest), "rows": 1080, "new_model_calls": 0}) + "\n")
    outputs = out / "full1080.outputs.jsonl"
    with outputs.open("x") as f:
        for row in rows:
            f.write(json.dumps({"sample_id": row["sample_id"], "text": row["text"]}, ensure_ascii=False) + "\n")
    scores = score_slurp_outputs_official_train(outputs, lock, data)
    if scores["aggregate"]["samples"] != 1080 or not scores["aggregate"]["identity_check"]["passed"]:
        raise ValueError("full1080 official score identity differs")
    target = out / "scores.json"; target.write_text(json.dumps(scores, indent=2) + "\n")
    result = {"schema": "saea-a0-full1080-source-bound-result-v1", "rows": 1080,
              "recipe_sha256": sha(recipe_path), "outputs_sha256": sha(outputs), "scores_sha256": sha(target),
              "continuation_manifest_sha256": sha(continuation_manifest), "new_model_calls": 0,
              "official_scoring_jobs": 1, "historical_costs_retained": True,
              "source_scope": "427 historical lock +300 first engineering candidate +353 continuation; no subset metric averaging"}
    provenance = {"schema": "saea-a0-full1080-provenance-v1", "source_custody": recipe["custody_summary"],
                  "continuation_manifest": {"relpath": str(continuation_manifest.resolve().relative_to(data.resolve())), "sha256": sha(continuation_manifest)},
                  "selection": recipe["selection"], "historical_checkpoint_costs": "Retained in pinned historical source; incomplete attempt costs remain separate",
                  "first_candidate_full_cost": source["candidate_bundle"].manifest["cost"],
                  "continuation_full_cost": bundle.manifest["cost"],
                  "unaccounted_inflight_cost_is_zero": False}
    provenance_path = out / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    result["provenance_sha256"] = sha(provenance_path)
    (out / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "data", "recipe-path", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("continuation-repo", "continuation-manifest"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--validate-only", action="store_true")
    print(json.dumps(run(**vars(p.parse_args()))))
