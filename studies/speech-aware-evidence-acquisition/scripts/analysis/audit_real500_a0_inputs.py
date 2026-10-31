"""Compare a real500 candidate with the replay-verified full A0 c2 input method.

This read-only audit does not grant checkpoint import or runtime equivalence.
Operational differences are recorded individually; unknown differences fail.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def compare_method(old_model, new_model, old_base, new_base):
    before, after = copy.deepcopy(old_model), copy.deepcopy(new_model)
    changes = {}
    for key in ("runtime_receipt", "timeout_seconds"):
        if before.get(key) != after.get(key):
            changes["model/" + key] = {"before": before[key], "after": after[key]}
        after[key] = before[key]
    if before != after:
        raise ValueError("model content or decoding changed")
    before, after = copy.deepcopy(old_base), copy.deepcopy(new_base)
    for key in ("run_id", "attempt_id", "contact_cap"):
        a, b = before["native_contact_fit"][key], after["native_contact_fit"][key]
        if a != b:
            changes["baseline/native_contact_fit/" + key] = {"before": a, "after": b}
        after["native_contact_fit"][key] = a
    before.pop("prepared_phase_supply", None)
    after.pop("prepared_phase_supply", None)
    if before != after:
        raise ValueError("baseline scientific method changed")
    return changes


def compare_supply(full, selected, ids):
    if set(full) != set(selected):
        raise ValueError("supply output fields changed")
    checked = 0
    for kind in full:
        if set(full[kind]) != set(selected[kind]):
            raise ValueError("supply phase inventory changed")
        for phase, subset in selected[kind].items():
            if set(subset) != set(ids):
                raise ValueError("supply does not cover exactly the selected roster")
            if any(sid not in full[kind][phase] or value != full[kind][phase][sid]
                   for sid, value in subset.items()):
                raise ValueError("prepared evidence differs")
            checked += len(subset)
    if not checked:
        raise ValueError("no prepared evidence compared")
    return checked


def compare_runtime(old, new):
    fields = set(old) | set(new)
    ignored = {"container", "deployment", "protocol_sha256"}
    if any(old.get(k) != new.get(k) for k in fields - ignored):
        raise ValueError("serving content/runtime receipt differs")
    if old["container"]["image_id"] != new["container"]["image_id"]:
        raise ValueError("serving image changed")
    return {"same_image_digest": True, "same_model_files": True,
            "same_interpreter_libraries_packages_sources": True,
            "same_model_protocol": True,
            "deployment_identity_differs": old["deployment"] != new["deployment"]}


def without_timing(source):
    tree = ast.parse(source)
    class RemoveTiming(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module and node.module.endswith("stage_timing") and len(node.names) == 1 and node.names[0].name == "measure" and node.names[0].asname is None:
                return None
            return node
        def visit_FunctionDef(self, node):
            node.decorator_list = [d for d in node.decorator_list
                                   if not (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "measure")]
            return self.generic_visit(node)
    return ast.dump(RemoveTiming().visit(tree), include_attributes=False)


def compare_source(reference, candidate):
    prefix = "speech_aware_evidence_acquisition/"
    before = {str(p.relative_to(reference)): p for p in reference.rglob("*.py")}
    after = {str(p.relative_to(candidate)): p for p in candidate.rglob("*.py")}
    changes = {}
    timing = prefix + "core/stage_timing.py"
    deployment = prefix + "core/deployment.py"
    for name in sorted(before.keys() | after.keys()):
        a = before[name].read_text() if name in before else None
        b = after[name].read_text() if name in after else None
        if a == b:
            continue
        if name == timing and a is None and b is not None:
            classification = "new timing helper; behavior covered by timing tests"
        elif name == deployment and a is not None and b is not None:
            expected = a.replace("    DEFAULT_NAMESPACE_POLICY,\n", "    DEFAULT_NAMESPACE_POLICY,\n    REQUIRED_ENV,\n")
            expected = expected.replace('("PYTHONNOUSERSITE", "1"))', '*sorted(REQUIRED_ENV.items()))')
            if expected != b:
                raise ValueError("unexpected deployment source delta")
            classification = "shared required environment expansion"
        elif a is not None and b is not None and without_timing(a) == without_timing(b):
            classification = "timing imports/decorators only"
        else:
            raise ValueError("scientific source changed: " + name)
        changes[name] = {"before": digest(before[name]) if name in before else None,
                         "after": digest(after[name]), "classification": classification}
    return changes


def resolve_pin(data, pin):
    path = (data / pin["relpath"]).resolve()
    if not path.is_relative_to(data.resolve()) or digest(path) != pin["sha256"]:
        raise ValueError("artifact pin differs or escapes data root")
    return path


def audit(reference, candidate, reference_experiment, candidate_experiment,
          data, supply_proof, replay_proof, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash
    assert_execution_environment()
    if out.exists() or not out.resolve().is_relative_to((data / "work/codex").resolve()):
        raise ValueError("fresh external output required")
    configs = []
    pins = {}
    for label, repo, experiment in (("reference", reference, reference_experiment),
                                     ("candidate", candidate, candidate_experiment)):
        ep = repo / "configs/experiment" / (experiment + ".json")
        cell, = read(ep)["cells"]
        doc = {}
        pins[label + "/experiment"] = digest(ep)
        for kind in ("model", "baseline", "dataset"):
            p = repo / "configs" / kind / (cell[kind + "_fragment"] + ".json")
            doc[kind] = read(p)
            pins[label + "/" + kind] = digest(p)
        configs.append(doc)
    old, new = configs
    changes = compare_method(old["model"], new["model"], old["baseline"], new["baseline"])
    parent = {x["sample_id"]: x for x in old["dataset"]["items"]}
    rows = new["dataset"]["items"]
    ids = [x["sample_id"] for x in rows]
    if len(parent) != 1080 or len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError("expected unique 500-of-1080 population")
    if any(parent.get(x["sample_id"]) != x for x in rows):
        raise ValueError("dataset row differs from full A0 parent")
    for doc in configs:
        identity = split_identity_hash(prefixed("slurp", [x["sample_id"] for x in doc["dataset"]["items"]]))
        if identity != doc["dataset"]["split_identity_hash"]:
            raise ValueError("dataset identity differs")
    proof = read(supply_proof)
    replay = read(replay_proof)
    full_path = supply_proof.parent / "c2-phase-supply.json"
    if (proof["rows"] != 1080 or proof["output_equal"] is not True
            or proof["new_supply_sha256"] != digest(full_path)
            or proof["split_identity_hash"] != old["dataset"]["split_identity_hash"]):
        raise ValueError("full supply equivalence proof differs")
    if (replay["rows"] != 427 or replay["contacts"] != 854
            or replay["vllm_wire_equal"] is not True or replay["delivered_output_equal"] is not True
            or replay["model_sha256"] != pins["reference/model"]
            or replay["split_identity_hash"] != old["dataset"]["split_identity_hash"]
            or replay["custody_deltas_sha256"] != digest(replay_proof.parent / "custody-deltas.json")):
        raise ValueError("historical request replay proof differs")
    selected_path = resolve_pin(data, new["baseline"]["prepared_phase_supply"])
    compared = compare_supply(read(full_path)["output"], read(selected_path)["output"], ids)
    old_runtime_path = reference / "docs" / old["model"]["runtime_receipt"]
    new_runtime_path = candidate / "docs" / new["model"]["runtime_receipt"]
    runtime_evidence = compare_runtime(read(old_runtime_path), read(new_runtime_path))
    source_delta = compare_source(reference / "src", candidate / "src")
    result = {"schema": "saea-real500-a0-input-compatibility-v2", "rows": 500,
              "dataset_rows_equal": True, "model_content_config_equal": True,
              "baseline_method_equal": True, "prepared_evidence_equal": True,
              "compared_supply_entries": compared, "operational_changes": changes,
              "config_pins": pins, "runtime_comparison": runtime_evidence, "source_changes": source_delta,
              "runtime_receipt_pins": {"reference": digest(old_runtime_path), "candidate": digest(new_runtime_path)}, "supply_proof_sha256": digest(supply_proof),
              "replay_proof_sha256": digest(replay_proof),
              "candidate_supply_sha256": digest(selected_path),
              "new_model_calls": 0, "scientific_reuse_authorized": False,
              "remaining": ["Final candidate generation seal and quality scoring",
                            "Explicit review of pinned timing/environment source changes and deployment transition",
                            "Explicit source/projection custody and full1080 scoring integration",
                            "Retain all historical and candidate costs; no outcome selection"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(result, f, indent=2); f.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "data", "supply-proof", "replay-proof", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("reference-experiment", "candidate-experiment"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = audit(**vars(args))
    print(json.dumps({k: result[k] for k in ("rows", "prepared_evidence_equal", "new_model_calls")}))
