"""Strict version-one configuration loading. No filesystem mutations."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


class ConfigError(ValueError):
    """A configuration or frozen input contract was violated."""


ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
HASH = re.compile(r"[a-fA-F0-9]{64}\Z")
IMAGE = re.compile(r"(?:[^\s@]+@)?sha256:[a-fA-F0-9]{64}\Z")
VARIABLE = re.compile(r"\$\{([^}]+)\}")
ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
MANAGED_ENV = {
    "HOME": "home", "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
    "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data",
    "HF_HOME": "cache/huggingface", "HUGGINGFACE_HUB_CACHE": "cache/huggingface/hub",
    "TRANSFORMERS_CACHE": "cache/transformers", "TORCH_HOME": "cache/torch",
    "NUMBA_CACHE_DIR": "cache/numba", "MPLCONFIGDIR": "cache/matplotlib",
    "PIP_CACHE_DIR": "cache/pip", "CUDA_CACHE_PATH": "cache/cuda",
    "PYTHONPYCACHEPREFIX": "cache/pycache", "LOG_DIR": "logs",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def object_keys(value, required, optional, label):
    if not isinstance(value, dict):
        raise ConfigError(f"{label}: expected object")
    missing, extra = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ConfigError(f"{label}: missing={sorted(missing)}, unknown={sorted(extra)}")
    return value


def string(value, label):
    if not isinstance(value, str) or not value or "\0" in value:
        raise ConfigError(f"{label}: expected nonempty string without NUL")
    return value


def identifier(value, label):
    if not isinstance(value, str) or not ID.fullmatch(value) or value in {".", ".."}:
        raise ConfigError(f"{label}: invalid identifier")
    return value


def number(value, label, integer=False):
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        raise ConfigError(f"{label}: expected positive {'integer' if integer else 'number'}")
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{label}: expected finite positive number")
    return value


def path(value, label):
    value = string(value, label)
    if not Path(value).is_absolute():
        raise ConfigError(f"{label}: absolute path required")
    return str(Path(value).resolve())


def within(value, root, label, strict=False):
    p, r = Path(value).resolve(), Path(root).resolve()
    if not p.is_relative_to(r) or (strict and p == r):
        raise ConfigError(f"{label}: {p} must be inside {r}")
    return str(p)


def expand(value, variables):
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if name not in variables:
                raise ConfigError(f"undeclared variable: {name}")
            return variables[name]
        result = VARIABLE.sub(replace, value)
        if "${" in result:
            raise ConfigError("malformed variable expansion")
        return result
    if isinstance(value, list):
        return [expand(v, variables) for v in value]
    if isinstance(value, dict):
        if any("${" in k for k in value):
            raise ConfigError("variables are not allowed in object keys")
        return {k: expand(v, variables) for k, v in value.items()}
    return value


def artifacts(value, label):
    if not isinstance(value, list):
        raise ConfigError(f"{label}: expected list")
    result = []
    for index, item in enumerate(value):
        object_keys(item, {"path", "sha256"}, {"immutable"}, f"{label}[{index}]")
        if not isinstance(item["sha256"], str) or not HASH.fullmatch(item["sha256"]):
            raise ConfigError(f"{label}[{index}]: invalid sha256")
        if type(item.get("immutable", False)) is not bool:
            raise ConfigError(f"{label}[{index}]: immutable must be boolean")
        result.append({"path": path(item["path"], label), "sha256": item["sha256"].lower(),
                       "immutable": item.get("immutable", False)})
    return result


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ConfigError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_config(filename):
    source = Path(filename).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs,
                         parse_constant=lambda v: (_ for _ in ()).throw(ConfigError(f"invalid number: {v}")))
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot load config: {error}") from error
    object_keys(raw, {"schema_version", "storage", "release", "experiment", "execution", "tasks"}, set(), "config")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ConfigError("schema_version must be integer 1")
    storage = object_keys(raw["storage"], {"shared_root", "run_root"}, set(), "storage")
    variables = {"config_dir": str(source.parent)}
    variables["shared_root"] = path(expand(storage["shared_root"], variables), "storage.shared_root")
    variables["run_root"] = path(expand(storage["run_root"], variables), "storage.run_root")
    within(variables["run_root"], variables["shared_root"], "storage.run_root", strict=True)
    cfg = expand(raw, variables)
    cfg["storage"] = {k: variables[k] for k in ("shared_root", "run_root")}
    release = object_keys(cfg["release"], {"image_digest"}, {"artifacts"}, "release")
    if not isinstance(release["image_digest"], str) or not IMAGE.fullmatch(release["image_digest"]):
        raise ConfigError("release.image_digest: immutable sha256 digest required")
    release["artifacts"] = artifacts(release.get("artifacts", []), "release.artifacts")
    experiment = object_keys(cfg["experiment"], {"id", "model", "thinking", "dataset", "science"}, set(), "experiment")
    identifier(experiment["id"], "experiment.id")
    string(experiment["model"], "experiment.model")
    if not (type(experiment["thinking"]) is bool or experiment["thinking"] in ("on", "off", "ON", "OFF")):
        raise ConfigError("experiment.thinking: expected boolean or on/off")
    if not isinstance(experiment["science"], dict):
        raise ConfigError("experiment.science: expected metadata object")
    dataset = object_keys(experiment["dataset"], {"manifest", "version"}, {"sha256"}, "experiment.dataset")
    dataset["manifest"] = path(dataset["manifest"], "experiment.dataset.manifest")
    string(dataset["version"], "experiment.dataset.version")
    if "sha256" in dataset and (not isinstance(dataset["sha256"], str) or not HASH.fullmatch(dataset["sha256"])):
        raise ConfigError("experiment.dataset.sha256: invalid digest")
    execution = object_keys(cfg["execution"], {"network"},
                            {"controller", "parallelism", "mode", "poll_seconds", "lanes"}, "execution")
    string(execution["network"], "execution.network")
    execution.setdefault("controller", {"kind": "subprocess"})
    object_keys(execution["controller"], {"kind"}, set(), "execution.controller")
    if execution["controller"]["kind"] != "subprocess":
        raise ConfigError("only subprocess controller is supported")
    execution.setdefault("mode", "local")
    if execution["mode"] != "local":
        raise ConfigError("execution.mode must be local; invoke CLI on the target host")
    execution.setdefault("parallelism", {"cpu": 1, "npu": 1})
    parallelism = object_keys(execution["parallelism"], set(), {"cpu", "npu"}, "execution.parallelism")
    for resource in ("cpu", "npu"):
        number(parallelism.setdefault(resource, 1), f"parallelism.{resource}", integer=True)
    number(execution.setdefault("poll_seconds", 1), "execution.poll_seconds")
    lanes = object_keys(execution.setdefault("lanes", {"npu": []}), set(), {"npu"}, "execution.lanes")
    if not isinstance(lanes.setdefault("npu", []), list):
        raise ConfigError("execution.lanes.npu: expected list")
    lane_ids, lane_paths = set(), set()
    for lane in lanes["npu"]:
        object_keys(lane, {"id", "lock_path"}, set(), "lane")
        lid = identifier(lane["id"], "lane.id")
        lane["lock_path"] = within(path(lane["lock_path"], "lane.lock_path"),
                                   Path(variables["run_root"]) / "lanes", "lane.lock_path", strict=True)
        if lid in lane_ids or lane["lock_path"] in lane_paths:
            raise ConfigError("duplicate lane id or lock path")
        lane_ids.add(lid)
        lane_paths.add(lane["lock_path"])
    tasks = cfg["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ConfigError("tasks: expected nonempty list")
    seen = set()
    experiment_dir = Path(variables["run_root"]) / "experiments" / experiment["id"]
    for task in tasks:
        object_keys(task, {"id", "resource", "argv", "cwd"},
                    {"depends_on", "env", "timeout_seconds", "outputs", "artifacts", "lane", "unknown_exit_codes"}, "task")
        tid = identifier(task["id"], "task.id")
        if tid in seen:
            raise ConfigError(f"duplicate task id: {tid}")
        seen.add(tid)
        if task["resource"] not in ("cpu", "npu"):
            raise ConfigError(f"{tid}: resource must be cpu or npu")
        if task["resource"] == "npu" and not lanes["npu"]:
            raise ConfigError(f"{tid}: NPU tasks require shared execution.lanes.npu")
        if "lane" in task and (not isinstance(task["lane"], str) or task["resource"] != "npu" or task["lane"] not in lane_ids):
            raise ConfigError(f"{tid}: invalid lane")
        argv = task["argv"]
        if not isinstance(argv, list) or not argv or any(not isinstance(v, str) or "\0" in v for v in argv) or not argv[0]:
            raise ConfigError(f"{tid}: argv must be a nonempty list of strings without NUL")
        task["cwd"] = path(task["cwd"], f"{tid}.cwd")
        deps = task.setdefault("depends_on", [])
        if not isinstance(deps, list) or any(not isinstance(v, str) for v in deps) or len(set(deps)) != len(deps):
            raise ConfigError(f"{tid}: depends_on must be a list of distinct ids")
        env = task.setdefault("env", {})
        if not isinstance(env, dict) or any(not ENV_KEY.fullmatch(k) or not isinstance(v, str) or "\0" in v for k, v in env.items()):
            raise ConfigError(f"{tid}: env must map valid environment keys to strings")
        for key, value in env.items():
            if key in MANAGED_ENV or key.endswith(("_CACHE_DIR", "_CACHE_HOME", "_TMPDIR", "_LOG_DIR")):
                within(path(value, f"{tid}.env.{key}"), variables["run_root"], f"{tid}.env.{key}", strict=True)
        if "PYTHONDONTWRITEBYTECODE" in env and env["PYTHONDONTWRITEBYTECODE"] != "1":
            raise ConfigError(f"{tid}: PYTHONDONTWRITEBYTECODE must be 1")
        if "timeout_seconds" in task:
            number(task["timeout_seconds"], f"{tid}.timeout_seconds")
        outputs = task.setdefault("outputs", [])
        if not isinstance(outputs, list):
            raise ConfigError(f"{tid}: outputs must be list")
        task["outputs"] = [within(path(p, f"{tid}.output"), experiment_dir / "outputs", f"{tid}.output", strict=True)
                           for p in outputs]
        task["artifacts"] = artifacts(task.get("artifacts", []), f"{tid}.artifacts")
        unknown = task.setdefault("unknown_exit_codes", [75, 255])
        if not isinstance(unknown, list) or any(type(v) is not int or not 1 <= v <= 255 for v in unknown):
            raise ConfigError(f"{tid}.unknown_exit_codes: expected nonzero exit codes")
    by_id = {t["id"]: t for t in tasks}
    visiting, visited, order = set(), set(), []
    def visit(tid):
        if tid in visiting:
            raise ConfigError(f"DAG cycle at {tid}")
        if tid in visited:
            return
        visiting.add(tid)
        for dep in by_id[tid]["depends_on"]:
            if dep not in by_id:
                raise ConfigError(f"{tid}: unknown dependency {dep}")
            visit(dep)
        visiting.remove(tid)
        visited.add(tid)
        order.append(tid)
    for tid in by_id:
        visit(tid)
    return {"config": cfg, "sha256": digest(cfg), "source": str(source),
            "run_dir": str(experiment_dir), "order": order}


def task_environment(document, task):
    """Explicit write locations, with no ambient credential/proxy/cache inheritance."""
    import os
    root = Path(document["run_dir"]) / "tasks" / task["id"] / "work"
    env = {"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONUNBUFFERED": "1"}
    env.update({key: str(root / relative) for key, relative in MANAGED_ENV.items()})
    env.update(task["env"])
    return env
