"""Configured S1 preparation and restore over the existing native runtime.

CPU preparation never acquires a generation lease, starts a service or posts.
One explicit source/input index is consumed by prepare, restore and READY.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

IMAGE = "sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def read(path):
    return json.loads(Path(path).read_bytes())


def lines(path):
    return [json.loads(x) for x in Path(path).read_bytes().splitlines() if x.strip()]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with tmp.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def resolve(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "index path must be workspace relative")
    return Path(root) / path


def pinned(root, pin):
    path = resolve(root, pin["path"])
    require(file_sha(path) == pin["sha256"], "dependency changed: " + pin["path"])
    return path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def imports(root, task):
    """Explicit packaged source and native dependency locations, no fallback."""
    source = resolve(root, task["source_root"])
    import speech_aware_evidence_acquisition as package
    require(Path(package.__file__).resolve().parent == (source / package.__name__).resolve(),
            "wrong imported source root")
    native = resolve(root, task["native_root"])
    sys.path.insert(0, str(native))
    return load("configured_s1_original_native", native / "run_s_native.py")


def stat_pin(path):
    info = Path(path).stat()
    return dict(size=info.st_size, mtime_ns=info.st_mtime_ns, inode=info.st_ino, device=info.st_dev)


def get_audio(row, root):
    raw = pinned(root, row["audio_source"]).read_bytes()
    obj = json.loads(raw)
    for key in row["audio_source"]["content_path"]:
        obj = obj[key]
    require(obj.get("type") == "input_audio" and sha(canonical(obj)) == row["audio_object_sha256"],
            "inherited audio object pin")
    return obj


def restore(template, audio):
    packet = copy.deepcopy(template)
    content = packet["wire"]["messages"][1]["content"]
    indices = [i for i, part in enumerate(content) if part == {"__reuse_audio__": True}]
    require(len(indices) == 1, "one inherited audio placeholder required")
    content[indices[0]] = audio
    return packet


def context(root, task_path, lane_path, index_path):
    task, lane, index = read(task_path), read(lane_path), read(index_path)
    require(task["stage"] == "S1" and task["schema"] == "configured-s1-task-v1", "S1 task contract")
    require(index["task_sha256"] == file_sha(task_path) and index["lane_sha256"] == file_sha(lane_path),
            "task/lane index binding")
    require(index["lane"] == lane["lane"], "lane index binding")
    for pin in index["dependencies"]:
        pinned(root, pin)
    rows = [row for row in lines(pinned(root, task["manifest"])) if row["lane"] == lane["lane"]]
    require(len(rows) == lane["expected_samples"] and len({r["sample_id"] for r in rows}) == len(rows), "lane roster")
    require({r["audio_source"]["path"] for r in rows} == {p["path"] for p in index["audio"]}, "audio dependency closure")
    require({r["inherited_fit"]["budget"]["path"] for r in rows} <= {p["path"] for p in index["dependencies"]}, "FIT dependency closure")
    expected = {p["path"]: p["sha256"] for p in index["audio"]}
    require(all(expected[r["audio_source"]["path"]] == r["audio_source"]["sha256"] for r in rows), "audio index pins")
    require(task["image"] == IMAGE and task["context_length"] == lane["service"]["context_length"], "image/context binding")
    require(lane["service"]["fixed_image_sha256"] == IMAGE, "inherited service image")
    for asset in task.get("model_assets", []):
        pinned(task["model_root"], asset)
    policy = read(pinned(root, task["runtime_config"]))
    require(policy["max_inflight_requests"] == task["concurrency"] == 64 and policy["s1_max_tokens"] == 256,
            "frozen concurrency/output")
    require(policy["model"] == task["model"], "model binding")
    identity = sha(canonical(dict(task_sha256=file_sha(task_path), lane_sha256=file_sha(lane_path),
                                  index_sha256=file_sha(index_path), image=IMAGE)))
    return task, lane, index, rows, policy, identity


def validate_ready(root, output, task_path, lane_path, index_path, config=None):
    task, lane, index, rows, policy, identity = context(root, task_path, lane_path, index_path)
    output = Path(output)
    ready = read(output / "READY.json")
    require(ready["status"] == "CPU_READY_FOR_REVIEW" and ready["identity"] == identity, "stale READY")
    for pin in ready["artifacts"]:
        require(file_sha(output / pin["path"]) == pin["sha256"], "prepared artifact changed: " + pin["path"])
    for pin in ready["audio_stat_pins"]:
        require(stat_pin(resolve(root, pin["path"])) == pin["stat"], "prepared audio changed: " + pin["path"])
    if config is not None:
        frozen = read(output / "native-UNBOUND.json")
        require({k: v for k, v in config.items() if k != "runtime"} ==
                {k: v for k, v in frozen.items() if k != "runtime"}, "bound config differs from READY")
    return ready


def ensure_ready(root, output, task_path, lane_path, index_path, model_root):
    """Fail closed; an old READY is never a fallback after preparation failure."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ready_path = output / "READY.json"
    try:
        if ready_path.exists():
            try:
                ready = validate_ready(root, output, task_path, lane_path, index_path)
            except (ValueError, OSError, KeyError):
                ready_path.unlink(missing_ok=True)
            else:
                return dict(status="NOOP_ALREADY_READY", identity=ready["identity"], model_calls=0)
        return prepare(root, output, task_path, lane_path, index_path, model_root)
    except BaseException as error:
        ready_path.unlink(missing_ok=True)
        save(output / "PREPARE-FAILED.json", dict(status="FAILED_NO_READY", error=repr(error), model_calls=0))
        raise


def scientific_changes(task, baseline, policy):
    """Explicit policy changes; every unlisted field is frozen by default.

    Legacy tasks retain their instruction-only contract. New tasks declare the
    exact set to avoid silently admitting extra scientific interventions.
    """
    declared = task.get("allowed_changes", ["s1_instruction"])
    require(isinstance(declared, list) and len(set(declared)) == len(declared)
            and all(isinstance(k, str) for k in declared), "invalid allowed_changes")
    changed = sorted(k for k in set(policy) | set(baseline)
                     if (k in policy) != (k in baseline) or policy.get(k) != baseline.get(k))
    require(changed == sorted(declared), "policy differs from declared allowed_changes")
    return changed


def schema_change(previous, current, baseline, policy, changed):
    expected = copy.deepcopy(previous)
    if policy.get("s1_candidate_presentation") == "type_marking_v1":
        require(set(changed) == {"s1_candidate_presentation", "s1_instruction", "s1_none_label"},
                "type marking exact policy changes")
        require(baseline.get("s1_candidate_presentation", "legacy") == "legacy"
                and baseline.get("s1_none_label") == "NONE_OF_FOLD"
                and policy.get("s1_none_label") is None, "type marking None parent")
        labels = expected["json_schema"]["schema"]["properties"]["slot_types"]["items"]["enum"]
        require(labels[-1] == "NONE_OF_FOLD", "type marking parent enum")
        types = labels[:-1]
        expected["json_schema"]["schema"] = {
            "type": "object", "additionalProperties": False, "required": ["slot_type_presence"],
            "properties": {"slot_type_presence": {
                "type": "object", "additionalProperties": False, "required": types,
                "properties": {t: {"type": "boolean"} for t in types}}}}
        require(current == expected, "type marking schema differs outside declared exact transformation")
        return
    if "s1_none_label" in changed:
        old, new = baseline.get("s1_none_label"), policy.get("s1_none_label")
        labels = expected["json_schema"]["schema"]["properties"]["slot_types"]["items"]["enum"]
        if old is not None:
            require(labels[-1] == old, "parent control enum binding")
            labels.pop()
        if new is not None:
            labels.append(new)
    require(current == expected, "schema differs outside declared control enum")


def protocol_wire_change(previous, current, baseline, policy, changed):
    """Verify a configured unified protocol edit against the actual frozen wire."""
    if not set(changed) <= {'s1_instruction', 's1_unified_templates'}:
        return
    if baseline.get('s1_candidate_presentation') != 'unified_candidates_v1':
        return
    from .slot_stages import unified_templates
    expected = copy.deepcopy(previous)
    templates = unified_templates(policy)
    expected['messages'][0]['content'] = policy['s1_instruction'] + templates['system_suffix'].replace(
        '{control_label}', policy['s1_none_label'])
    text = next(part for part in expected['messages'][1]['content'] if part['type'] == 'text')
    body = json.loads(text['text'])
    body['candidate_selection']['instruction'] = templates['user_instruction']
    text['text'] = canonical(body).decode()
    require(current == expected, 'wire differs outside configured S1 protocol text')


def build_rails(root, task, knowledge, policy):
    """Load the explicitly pinned supply once per rails instance, never per Fold."""
    from .slot_stages import StageRails
    if policy.get("s1_consumption") is None:
        require("retrieval_cache" not in task and "provenance_root" not in task,
                "supply inputs require an explicit S1 consumption policy")
        return StageRails(knowledge, policy)
    require("retrieval_cache" in task and "provenance_root" in task,
            "consumption requires pinned cache and explicit provenance root")
    cache = read(pinned(root, task["retrieval_cache"]))
    provenance = resolve(root, task["provenance_root"])
    return StageRails(knowledge, policy, retrieval_cache=cache, provenance_root=provenance)


def prepare(root, output, task_path, lane_path, index_path, model_root):
    started = time.monotonic()
    task, lane, index, rows, policy, identity = context(root, task_path, lane_path, index_path)
    from .slot_stages import Query, StageRails
    from transformers import AutoTokenizer
    knowledge = read(pinned(root, task["knowledge"]))
    rails = build_rails(root, task, knowledge, policy)
    require(len(knowledge["folds"]) == task["folds"] and len(knowledge["cards"]) == task["types"], "knowledge dimensions")
    # Model files are indexed individually; never hash or load model weights.
    require(str(Path(model_root)) == task["model_root"], "explicit model root")
    for pin in task["model_assets"]:
        pinned(model_root, pin)
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True, trust_remote_code=False)
    sys.path.insert(0, str(resolve(root, task["native_root"])))
    helper = load("configured_s1_fit", resolve(root, task["fit_helper"]))
    template_receipt = helper.bind_local_chat_template(tokenizer, model_root)
    baseline = read(pinned(root, task["baseline_runtime_config"]))
    changed = scientific_changes(task, baseline, policy)
    # Supply inputs belong to the candidate task; a pre-supply parent keeps
    # its original path without consuming the candidate's cache/provenance.
    if baseline.get("s1_consumption") is None:
        old_rails = StageRails(knowledge, baseline)
    else:
        old_rails = build_rails(root, task, knowledge, baseline)
    schemas = []
    for fold in knowledge["folds"]:
        q = Query("schema-probe", {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}})
        current_wire = rails.before_model_call("S1", q, fold["fold_id"])["wire"]
        previous_wire = old_rails.before_model_call("S1", q, fold["fold_id"])["wire"]
        current, previous = current_wire['response_format'], previous_wire['response_format']
        schema_change(previous, current, baseline, policy, changed)
        protocol_wire_change(previous_wire, current_wire, baseline, policy, changed)
        schemas.append(current)
    save(output / "GRAMMAR-COMPATIBILITY.json", dict(status="DECLARED_CONTROL_ENUM_CHANGE" if "s1_none_label" in changed else "IDENTICAL_TO_SUCCESSFUL_BASELINE_SCHEMA",
         allowed_changes=changed,
         schemas=schemas, schema_set_sha256=sha(canonical(schemas)), evidence=task["grammar_evidence"],
         general_backend_capability_verified=False, limitation="declared schema diff only; separate actual backend compiler receipt required for new enum"))
    if policy.get("s1_consumption") is not None:
        save(output / "SUPPLY-RECEIPT.json", dict(
            status="PINNED_SUPPLY_LOADED_ONCE", knowledge=task["knowledge"],
            retrieval_cache=task["retrieval_cache"], provenance_root=task["provenance_root"],
            knowledge_id=knowledge["knowledge_id"], cache_id=policy["s1_consumption"]["cache_id"],
            dependency_index_sha256=file_sha(index_path), stage_version=rails.version,
            runtime_packets="prepared_templates", new_encoder_calls=0, new_model_calls=0))
    protocol_edit = (baseline.get('s1_candidate_presentation') == 'unified_candidates_v1'
                     and set(changed) <= {'s1_instruction', 's1_unified_templates'})
    audited_folds = 0
    fit_sources, jobs, maximum, audio_stats = {}, [], 0, []
    for n, row in enumerate(rows):
        audio = get_audio(row, root)
        audio_stats.append(dict(path=row["audio_source"]["path"], stat=stat_pin(resolve(root, row["audio_source"]["path"]))))
        query = Query(row["sample_id"], audio, row["source_component"], row["audio_sha256"])
        budget = row["inherited_fit"]["budget"]
        if budget["path"] not in fit_sources:
            fit_sources[budget["path"]] = {r["sample_id"]: r for r in read(pinned(root, budget))["records"]}
        old = fit_sources[budget["path"]][row["inherited_fit"]["record_sample_id"]]
        require(old["audio_object_sha256"] == row["audio_object_sha256"], "FIT audio binding")
        inherited = old.get("inherited_P_input_upper", old["input_upper"])
        packets, records = [], []
        for fold in knowledge["folds"]:
            packet = rails.before_model_call("S1", query, fold["fold_id"])
            wire = packet["wire"]
            if protocol_edit:
                previous_wire = old_rails.before_model_call("S1", query, fold["fold_id"])["wire"]
                protocol_wire_change(previous_wire, wire, baseline, policy, changed)
                audited_folds += 1
            content = wire["messages"][1]["content"]
            ai = next(i for i, part in enumerate(content) if part["type"] == "input_audio")
            content[ai] = {"__reuse_audio__": True}
            projected = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
                else "\n".join(p["text"] for p in m["content"] if p.get("type") == "text")} for m in wire["messages"]]
            rendered = tokenizer.apply_chat_template(projected, tokenize=False, add_generation_prompt=True)
            options = canonical({k: v for k, v in wire.items() if k != "messages"})
            measured = len(tokenizer.encode(rendered, add_special_tokens=False)) + len(tokenizer.encode(options.decode(), add_special_tokens=False))
            delta = max(len(canonical(wire)), len(rendered.encode()) + len(options), measured) + 256
            total = inherited + delta + wire["max_tokens"]
            require(total < task["context_length"], "S1 FIT overflow")
            maximum = max(maximum, total)
            records.append(dict(branch=packet["branch"], request_id=packet["request_id"], wire_sha256=packet["wire_sha256"],
                inherited_P_input_upper=inherited, nonaudio_bytes=len(canonical(wire)), tokenizer_tokens=measured,
                input_upper=inherited + delta, total_upper=total, audio_object_sha256=row["audio_object_sha256"]))
            packets.append(packet)
        path = output / "templates" / (sha(row["sample_id"].encode()) + ".json")
        save(path, dict(sample_id=row["sample_id"], packets=packets, fit=records))
        jobs.append(dict(**row, template=dict(path=str(path.relative_to(root)), sha256=file_sha(path))))
        if n % 500 == 0:
            print("PREPARED", lane["lane"], n, flush=True)
    if protocol_edit:
        save(output / "PROTOCOL-WIRE-AUDIT.json", dict(status="PASS_ACTUAL_PREPARED_WIRES",
            samples=len(jobs), folds=audited_folds, allowed_changes=changed,
            baseline_runtime=task["baseline_runtime_config"], runtime=task["runtime_config"],
            schema_probes=task["folds"], model_calls=0))
    manifest = output / "manifest.private.jsonl"
    tmp = manifest.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        for job in jobs:
            stream.write(canonical(job) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, manifest)
    cfg = dict(schema="s1-native-stage-v1", run_id=lane["run_id"], attempt_id=task["attempt_id"],
        dataset_partition=task["dataset_partition"], lane=lane["lane"], concurrency=task["concurrency"],
        timeout_seconds=task["timeout_seconds"], expected_rows=len(jobs)*task["folds"], expected_samples=len(jobs),
        full_denominator=task["expected_samples"], manifest=str(manifest.relative_to(root)), manifest_sha256=file_sha(manifest),
        knowledge=task["knowledge"], runtime_policy=policy, stage_version=rails.version, runtime={},
        configured=dict(workspace_root=task["container_workspace"], source_root=task["source_root"],
            task=str(Path(task_path).relative_to(root)), lane=str(Path(lane_path).relative_to(root)),
            index=str(Path(index_path).relative_to(root)), prepared=str(output.relative_to(root)), identity=identity))
    save(output / "native-UNBOUND.json", cfg)
    save(output / "budget.json", dict(status="PASS_FIXED_IMAGE_CPU", image=IMAGE, rows=cfg["expected_rows"],
        samples=len(jobs), manifest_sha256=cfg["manifest_sha256"], identity=identity,
        fixed_image_tokenizer_verified=True, chat_template=template_receipt, max_total_upper=maximum,
        context_length=task["context_length"], audio_encodes=0, model_calls=0,
        preparation_seconds=time.monotonic()-started, knowledge=task["knowledge"]))
    artifacts = [dict(path=str(p.relative_to(output)), sha256=file_sha(p)) for p in sorted(output.rglob("*.json"))
                 if p.name not in ("READY.json", "PREPARE-FAILED.json")]
    artifacts.append(dict(path=manifest.name, sha256=file_sha(manifest)))
    save(output / "READY.json", dict(status="CPU_READY_FOR_REVIEW", identity=identity, image=IMAGE,
        samples=len(jobs), requests=cfg["expected_rows"], service=lane["service"],
        artifacts=artifacts, audio_stat_pins=audio_stats, model_calls=0, npu_dispatched=False))
    (output / "PREPARE-FAILED.json").unlink(missing_ok=True)
    return dict(status="CPU_READY_FOR_REVIEW", lane=lane["lane"], samples=len(jobs), requests=cfg["expected_rows"], max_total_upper=maximum)


def preflight(config, workspace):
    c = read(config)
    binding = c["configured"]
    root = Path(workspace)
    task_path, lane_path, index_path = [resolve(root, binding[k]) for k in ("task", "lane", "index")]
    output = resolve(root, binding["prepared"])
    ready = validate_ready(root, output, task_path, lane_path, index_path, c)
    require(ready["identity"] == binding["identity"], "current READY required")
    task = read(task_path)
    original = imports(root, task)
    from .slot_stages import StageRails, digest
    rails = build_rails(root, task, read(pinned(root, c["knowledge"])), c["runtime_policy"])
    require(rails.version == c["stage_version"], "rails version changed")
    prepared = []
    for row in lines(resolve(root, c["manifest"])):
        template = read(pinned(root, row["template"]))
        audio = get_audio(row, root)
        packets = [restore(p, audio) for p in template["packets"]]
        require([p["branch"] for p in packets] == [f["fold_id"] for f in rails.knowledge["folds"]], "fold inventory")
        require(len(packets) == len(template["fit"]), "FIT inventory")
        for packet, fit in zip(packets, template["fit"]):
            require(packet["sample_id"] == row["sample_id"] and packet["version"] == rails.version and
                digest(packet["wire"]) == packet["wire_sha256"] == fit["wire_sha256"], "restored wire binding")
            require(fit["total_upper"] == fit["input_upper"] + packet["wire"]["max_tokens"] < task["context_length"] and
                    fit["audio_object_sha256"] == row["audio_object_sha256"], "restored FIT binding")
        prepared.append((row, packets))
    require(len(prepared) == c["expected_samples"] and sum(len(p) for r, p in prepared) == c["expected_rows"], "denominator")
    fit = read(output / "budget.json")
    require(fit["status"] == "PASS_FIXED_IMAGE_CPU" and fit["identity"] == ready["identity"] and
            fit["manifest_sha256"] == c["manifest_sha256"], "current FIT")
    if c.get("runtime"):
        lane = read(lane_path)
        for key in ("boot_id", "session_id", "service_pid", "service_id", "shared_guard", "shared_guard_inode"):
            require(c["runtime"][key] == lane["service"][key], "runtime service differs from READY: " + key)
        require(file_sha(c["runtime"]["native_fit_receipt"]["path"]) == file_sha(output / "budget.json"), "captured FIT mismatch")
    return c, rails, prepared, original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "restore", "validate"])
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-root")
    args = parser.parse_args()
    if args.mode == "prepare":
        result = ensure_ready(Path(args.workspace), args.output, args.task, args.lane, args.index, args.model_root)
    elif args.mode == "restore":
        c, rails, prepared, old = preflight(Path(args.output) / "native-UNBOUND.json", args.workspace)
        result = dict(status="RESTORE_PASS", samples=len(prepared), requests=c["expected_rows"], model_calls=0)
    else:
        ready = validate_ready(args.workspace, args.output, args.task, args.lane, args.index)
        result = dict(status="CURRENT_READY_PASS", identity=ready["identity"], model_calls=0)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
