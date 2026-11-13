"""Detached argv/DAG execution with durable intents and conservative reconciliation.

This is an orchestration layer and migration entry point for existing programs.
It neither replaces FlightRuntime/adapters/scorers nor migrates legacy chains.
Lanes are orchestration quotas ONLY: existing card admission and generation
leases must remain in the business argv. A successful foreground argv must wait
for all service requests to settle. Nonzero NPU exit never releases its lease.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .config import ConfigError, MANAGED_ENV, canonical, digest, load_config, task_environment, within
from .state import Store, identity, now, observe, read_json, stat_key, verified_file


def store_for(doc):
    return Store(doc["config"]["storage"]["run_root"])


def run_path(doc):
    return Path(doc["run_dir"])


def task_path(doc, tid):
    return run_path(doc) / "tasks" / tid


def selected_tasks(doc, phase):
    return [t for t in doc["config"]["tasks"] if phase == "all" or t["resource"] == phase]


def plan(doc, phase="all"):
    tasks = selected_tasks(doc, phase)
    return {"schema_version": 1, "config_sha256": doc["sha256"], "run_dir": doc["run_dir"],
            "phase": phase, "order": [tid for tid in doc["order"] if any(t["id"] == tid for t in tasks)],
            "tasks": [{**t, "effective_env": task_environment(doc, t)} for t in tasks],
            "execution": doc["config"]["execution"],
            "scope": "execution orchestration and explicit legacy migration entry; legacy chains unchanged"}


def assert_frozen(doc):
    root = run_path(doc)
    lock = read_json(root / "CONFIG-LOCK.json")
    if not lock or lock.get("config_sha256") != doc["sha256"]:
        raise ConfigError("run is not prepared or configuration changed; use a new experiment.id")
    frozen = read_json(root / "CONFIG.json")
    if digest(frozen) != doc["sha256"]:
        raise ConfigError("frozen configuration corrupted")
    receipt = read_json(root / "PREPARE.json")
    if not receipt or receipt.get("config_sha256") != doc["sha256"]:
        raise ConfigError("prepare has not completed")
    pinned = lock.get("artifact_sha256", {})
    observed_pins = {entry["path"]: entry["sha256"] for entry in receipt["artifacts"]}
    if not pinned or observed_pins != pinned:
        raise ConfigError("prepare artifacts differ from immutable configuration lock")
    manifest_source = doc["config"]["experiment"]["dataset"]["manifest"]
    if receipt["manifest_sha256"] != pinned.get(manifest_source):
        raise ConfigError("manifest receipt differs from immutable configuration lock")
    # Controls always hash; only large explicitly immutable assets may reuse
    # stat receipts. This does not enumerate directories or rehash dataset trees.
    for entry in receipt["artifacts"]:
        verified_file(entry["path"], entry["sha256"], entry, immutable=entry.get("immutable", False))
    manifest = root / "inputs" / "dataset-manifest"
    if stat_key(manifest) != receipt["manifest_copy_stat"]:
        raise ConfigError("frozen manifest copy changed")
    verified_file(manifest, receipt["manifest_sha256"])
    verify_bindings(doc)
    return receipt


def verify_bindings(doc):
    """Bind launcher/workload fields to science metadata, without model calls."""
    def pointer(value, key):
        try:
            for part in key.split("/")[1:]:
                part = part.replace("~1", "/").replace("~0", "~")
                value = value[int(part)] if isinstance(value, list) else value[part]
            return value
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ConfigError(f"binding pointer missing: {key}") from error
    for binding in doc["config"]["execution"].get("bindings", []):
        verified_file(binding["path"], binding["sha256"])
        downstream = read_json(binding["path"])
        for target, source in binding["matches"].items():
            if canonical(pointer(downstream, target)) != canonical(pointer(doc["config"], source)):
                raise ConfigError(f"downstream configuration mismatch: {binding['path']} {target} != {source}")


def prepare(doc, phase="all"):
    cfg, root, store = doc["config"], run_path(doc), store_for(doc)
    verify_bindings(doc)
    if not Path(cfg["storage"]["shared_root"]).is_dir():
        raise ConfigError("shared_root must already exist; no fallback storage")
    store.mkdir(root)
    guard = store.lock(root / "RUN.lock")
    try:
        previous_lock = read_json(root / "CONFIG-LOCK.json")
        if previous_lock and previous_lock.get("config_sha256") != doc["sha256"]:
            raise ConfigError("configuration changed for this experiment.id; original attempt preserved")
        frozen = read_json(root / "CONFIG.json")
        if frozen is not None and digest(frozen) != doc["sha256"]:
            raise ConfigError("CONFIG.json differs; original attempt preserved")
        cached = read_json(root / "PREPARE.json", {})
        cache_by_path = {v["path"]: v for v in cached.get("artifacts", [])}
        manifest = cfg["experiment"]["dataset"]
        requested = [{"path": manifest["manifest"], "sha256": manifest.get("sha256")}]
        requested += cfg["release"]["artifacts"]
        requested += [{"path": b["path"], "sha256": b["sha256"]} for b in cfg["execution"].get("bindings", [])]
        for task in cfg["tasks"]:
            requested += task["artifacts"]
            if not Path(task["cwd"]).is_dir():
                raise ConfigError(f"task cwd does not exist: {task['id']}")
        checks, known = [], {}
        pinned = previous_lock.get("artifact_sha256", {}) if previous_lock else {}
        for artifact in requested:
            filename, expected = artifact["path"], artifact["sha256"]
            # If no digest was supplied, the first successful preparation pins it.
            if expected is None and filename in cache_by_path:
                expected = cache_by_path[filename]["sha256"]
            if filename in pinned:
                if expected and expected.lower() != pinned[filename]:
                    raise ConfigError(f"input differs from configuration lock: {filename}")
                expected = pinned[filename]
            if filename in known:
                if expected and known[filename]["sha256"] != expected.lower():
                    raise ConfigError(f"conflicting artifact digests: {filename}")
                continue
            entry = verified_file(filename, expected, cache_by_path.get(filename), immutable=artifact.get("immutable", False))
            checks.append(entry)
            known[filename] = entry
        # No existing attempt is overwritten, even if preparation was interrupted.
        if frozen is None:
            store.write(root / "CONFIG.json", cfg, exclusive=True)
        lock = {"config_sha256": doc["sha256"], "created_at": now(), "schema_version": 1,
                "artifact_sha256": {v["path"]: v["sha256"] for v in checks}}
        if previous_lock is None:
            store.write(root / "CONFIG-LOCK.json", lock, exclusive=True)
        snapshot = root / "inputs" / "dataset-manifest"
        manifest_hash = known[manifest["manifest"]]["sha256"]
        if snapshot.exists():
            verified_file(snapshot, manifest_hash,
                          {"stat": cached.get("manifest_copy_stat"), "sha256": manifest_hash})
        else:
            # Streaming copy avoids loading large JSONL manifests into memory.
            store.mkdir(snapshot.parent)
            temporary = snapshot.parent / f".manifest.{uuid.uuid4().hex}.tmp"
            try:
                import hashlib
                checksum = hashlib.sha256()
                fd = os.open(store.checked(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "wb") as target, open(manifest["manifest"], "rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        checksum.update(chunk)
                        target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                if checksum.hexdigest() != manifest_hash:
                    raise ConfigError("manifest changed while copying")
                os.link(temporary, snapshot)
                directory = os.open(snapshot.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                temporary.unlink(missing_ok=True)
        for task in cfg["tasks"]:
            for output in task["outputs"]:
                store.mkdir(Path(output).parent)
            env = task_environment(doc, task)
            for key in MANAGED_ENV:
                store.mkdir(env[key])
        receipt = {"schema_version": 1, "config_sha256": doc["sha256"], "prepared_at": now(),
                   "artifacts": checks, "manifest_sha256": manifest_hash,
                   "manifest_copy_stat": stat_key(snapshot), "experiment": cfg["experiment"],
                   "phase": phase, "state": "PREPARED"}
        store.write(root / "PREPARE.json", receipt)
        return receipt
    finally:
        os.close(guard)


def task_status(doc, task):
    root = task_path(doc, task["id"])
    start = read_json(root / "START.json")
    intent = read_json(root / "DISPATCH.json")
    exit_receipt = read_json(root / "EXIT.json")
    progress = read_json(root / "STATUS.json", {})
    mismatch = bool(intent and intent.get("config_sha256") != doc["sha256"])
    if start:
        mismatch |= not intent or start.get("dispatch_id") != intent.get("dispatch_id")
    if exit_receipt:
        mismatch |= (not intent or exit_receipt.get("dispatch_id") != intent.get("dispatch_id") or
                     exit_receipt.get("config_sha256") != doc["sha256"] or
                     exit_receipt.get("state") not in ("SUCCEEDED", "FAILED", "UNKNOWN"))
        if exit_receipt.get("state") == "SUCCEEDED":
            mismatch |= exit_receipt.get("returncode") != 0 or exit_receipt.get("contact_settled") is not True
    if mismatch:
        state = "UNKNOWN"
        process = {"state": "unknown", "reason": "receipt identity mismatch"}
    elif exit_receipt:
        state = exit_receipt["state"]
    elif intent or start:
        process = observe(start)
        state = "RUNNING" if process["state"] == "alive" else "UNKNOWN"
    else:
        process = {"state": "unknown", "reason": "not dispatched"}
        state = "PENDING"
    if exit_receipt and not mismatch:
        process = observe(start)
    last_progress = max([v.get("at", 0) for v in (intent, start, exit_receipt, progress) if v] or [0]) or None
    return {"id": task["id"], "resource": task["resource"], "state": state,
            "recorded_state": progress.get("state"), "process": process,
            "last_progress": last_progress, "age_seconds": max(0, now() - last_progress) if last_progress else None,
            "start": start, "exit": exit_receipt}


def status(doc, phase="all"):
    root = run_path(doc)
    lock = read_json(root / "CONFIG-LOCK.json")
    if lock and lock.get("config_sha256") != doc["sha256"]:
        raise ConfigError("status config differs from frozen run; use original CONFIG.json")
    summary = read_json(root / "STATUS.json", {})
    tasks = [task_status(doc, t) for t in selected_tasks(doc, phase)]
    states = {t["id"]: task_status(doc, t)["state"] for t in doc["config"]["tasks"]}
    for tid in doc["order"]:
        task = next(t for t in doc["config"]["tasks"] if t["id"] == tid)
        if states[tid] == "PENDING" and any(states[d] in ("FAILED", "BLOCKED") for d in task["depends_on"]):
            states[tid] = "BLOCKED"
    for item in tasks:
        if item["state"] != "PENDING":
            continue
        task = next(t for t in doc["config"]["tasks"] if t["id"] == item["id"])
        if states[item["id"]] == "BLOCKED":
            item["state"] = "BLOCKED"
        elif any(states[d] == "UNKNOWN" for d in task["depends_on"]):
            item["waiting_for"] = "unknown dependency; no redispatch"
    controller = read_json(root / "controllers" / summary.get("controller_id", "none") / "START.json")
    last_progress = max([summary.get("last_progress", 0)] + [t["last_progress"] or 0 for t in tasks]) or None
    return {"config_sha256": doc["sha256"], "run_dir": doc["run_dir"], "phase": phase,
            "prepared": (root / "PREPARE.json").is_file(), "recorded_state": summary.get("state"),
            "controller": {"start": controller, "process": observe(controller)}, "tasks": tasks,
            "last_progress": last_progress, "age_seconds": max(0, now() - last_progress) if last_progress else None}


def reconcile(doc, phase="all"):
    # Observation only: no retries, lease deletion, process signalling or writes.
    result = status(doc, phase)
    result["action"] = "observe_only"
    result["redispatched"] = []
    return result


def _internal_command(cli, root, operation, token, *extra):
    return [sys.executable, "-B", str(Path(cli).resolve()), "run", "--config", str(root / "CONFIG.json"),
            operation, token, *extra]


def _unresolved_controller(root):
    directory = root / "controllers"
    if not directory.exists():
        return None
    for candidate in sorted(directory.iterdir()):
        if (candidate / "DISPATCH.json").exists() and not (candidate / "EXIT.json").exists():
            return str(candidate)
    return None


def launch(doc, cli, phase="all"):
    store, root = store_for(doc), run_path(doc)
    assert_frozen(doc)
    guard = store.lock(root / "RUN.lock")
    try:
        previous = _unresolved_controller(root)
        if previous:
            raise ConfigError(f"prior controller has no EXIT: {previous}; reconcile observes only")
        token = uuid.uuid4().hex
        directory = root / "controllers" / token
        command = _internal_command(cli, root, "--_controller", token, "--phase", phase, "--_guard-fd", str(guard))
        store.write(directory / "DISPATCH.json", {"at": now(), "argv": command, "phase": phase,
                                                 "config_sha256": doc["sha256"]}, exclusive=True)
        environment = _controller_env(doc)
        with store.log(directory / "controller.log") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       cwd=root, env=environment, start_new_session=True, pass_fds=(guard,))
        receipt = {"at": now(), "state": "LAUNCHED", "controller_id": token, "phase": phase,
                   "argv": command, **identity(process.pid)}
        # Controller owns STATUS from here. LAUNCH is independent to avoid overwriting newer state.
        store.write(directory / "LAUNCH.json", receipt, exclusive=True)
        return receipt
    finally:
        os.close(guard)


def _controller_env(doc):
    root = run_path(doc) / "controller-work"
    store = store_for(doc)
    env = {"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    for key, relative in MANAGED_ENV.items():
        env[key] = str(store.mkdir(root / relative))
    return env


def _lease_path(lane):
    return Path(lane["lock_path"] + ".LEASE.json")


def acquire_lane(doc, task):
    """Acquire only after dependencies are ready. Keep unknown persistent leases."""
    store = store_for(doc)
    for lane in doc["config"]["execution"]["lanes"]["npu"]:
        if task.get("lane", lane["id"]) != lane["id"]:
            continue
        try:
            fd = store.lock(lane["lock_path"])
        except BlockingIOError:
            continue
        lease = read_json(_lease_path(lane))
        if lease:
            # A released kernel lock after crash/reboot does not prove task completion.
            exit_path = within(lease["exit_path"], doc["config"]["storage"]["run_root"], "lane lease exit")
            terminal = read_json(exit_path)
            if (not terminal or terminal.get("dispatch_id") != lease.get("dispatch_id") or
                    terminal.get("state") != "SUCCEEDED" or terminal.get("contact_settled") is not True):
                os.close(fd)
                continue
        return lane, fd
    return None


def controller(doc, cli, token, guard, phase):
    store, root = store_for(doc), run_path(doc)
    directory = root / "controllers" / token
    # Inherited lock ownership must be real; a private flag alone is insufficient.
    import fcntl
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if os.fstat(guard).st_ino != (root / "RUN.lock").stat().st_ino:
        raise ConfigError("controller guard does not match RUN.lock")
    intent = read_json(directory / "DISPATCH.json")
    if not intent or intent.get("config_sha256") != doc["sha256"]:
        raise ConfigError("controller has no matching dispatch intent")
    store.write(directory / "START.json", {"at": now(), "argv": intent["argv"], **identity()}, exclusive=True)
    active = {}
    tasks = selected_tasks(doc, phase)
    outcome = "UNKNOWN"
    last_snapshot = None
    last_written = 0
    try:
        assert_frozen(doc)
        while True:
            for tid, record in list(active.items()):
                process, lane_fd = record
                if process.poll() is not None:
                    if lane_fd is not None:
                        os.close(lane_fd)
                    del active[tid]
            states = {t["id"]: task_status(doc, t)["state"] for t in doc["config"]["tasks"]}
            # Propagate failures through arbitrary depth, including other phases.
            for tid in doc["order"]:
                task = next(t for t in doc["config"]["tasks"] if t["id"] == tid)
                if states[tid] == "PENDING" and any(states[d] in ("FAILED", "BLOCKED") for d in task["depends_on"]):
                    states[tid] = "BLOCKED"
            launched = False
            for task in tasks:
                tid, resource = task["id"], task["resource"]
                if states[tid] != "PENDING" or any(states[d] != "SUCCEEDED" for d in task["depends_on"]):
                    continue
                running = sum(1 for candidate in tasks if candidate["resource"] == resource and
                              (candidate["id"] in active or states[candidate["id"]] == "RUNNING"))
                if running >= doc["config"]["execution"]["parallelism"][resource]:
                    continue
                lane, lane_fd = None, None
                if resource == "npu":
                    acquired = acquire_lane(doc, task)
                    if acquired is None:
                        continue
                    lane, lane_fd = acquired
                dispatch_id = uuid.uuid4().hex
                td = task_path(doc, tid)
                argv = _internal_command(cli, root, "--_worker", tid, "--_dispatch-id", dispatch_id)
                fds = () if lane_fd is None else (lane_fd,)
                if lane_fd is not None:
                    argv += ["--_lane-fd", str(lane_fd)]
                intent = {"at": now(), "dispatch_id": dispatch_id, "argv": task["argv"],
                          "worker_argv": argv, "cwd": task["cwd"], "env": task_environment(doc, task),
                          "config_sha256": doc["sha256"], "resource": resource, "lane": lane}
                try:
                    # Journal precedes lease and spawn; any uncertain gap is never retried.
                    store.write(td / "DISPATCH.json", intent, exclusive=True)
                    if lane:
                        store.write(_lease_path(lane), {"dispatch_id": dispatch_id, "task_id": tid,
                                    "run_dir": str(root), "exit_path": str(td / "EXIT.json"), "at": now()})
                    with store.log(td / "worker.log") as log:
                        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                                   cwd=root, env=_controller_env(doc), start_new_session=True, pass_fds=fds)
                    active[tid] = (process, lane_fd)
                    launched = True
                    states[tid] = "RUNNING"
                except BaseException:
                    if lane_fd is not None:
                        os.close(lane_fd)
                    raise
            last_progress = max([intent.get("at", 0)] + [task_status(doc, t)["last_progress"] or 0 for t in tasks])
            snapshot = {"at": now(), "last_progress": last_progress, "state": "RUNNING", "controller_id": token, "phase": phase, "tasks": states}
            signature = (states, tuple(active))
            if signature != last_snapshot or now() - last_written >= 600:
                store.write(root / "STATUS.json", snapshot)
                last_snapshot, last_written = (dict(states), tuple(active)), now()
            if not active and not launched:
                runnable = [t for t in tasks if states[t["id"]] == "PENDING" and
                            all(states[d] == "SUCCEEDED" for d in t["depends_on"])]
                # Wait for busy shared lanes; blocked/unknown dependencies are a quiescent boundary.
                if not runnable:
                    chosen_states = [states[t["id"]] for t in tasks]
                    outcome = ("SUCCEEDED" if all(s == "SUCCEEDED" for s in chosen_states) else
                               "UNKNOWN" if "UNKNOWN" in chosen_states else
                               "FAILED" if any(s in ("FAILED", "BLOCKED") for s in chosen_states) else "WAITING")
                    break
            time.sleep(doc["config"]["execution"]["poll_seconds"])
    except BaseException as error:
        store.write(directory / "ERROR.json", {"at": now(), "error": repr(error)})
        raise
    finally:
        # Active workers inherit lane locks and hold them across controller death.
        for process, lane_fd in active.values():
            if lane_fd is not None:
                os.close(lane_fd)
        terminal = {"at": now(), "last_progress": now(), "state": outcome, "controller_id": token, "phase": phase}
        store.write(directory / "EXIT.json", terminal, exclusive=True)
        store.write(root / "STATUS.json", terminal)
        os.close(guard)
    return 0 if outcome == "SUCCEEDED" else 2


def worker(doc, tid, dispatch_id, lane_fd=None):
    store, td = store_for(doc), task_path(doc, tid)
    task = next((t for t in doc["config"]["tasks"] if t["id"] == tid), None)
    if task is None:
        raise ConfigError("unknown worker task")
    intent = read_json(td / "DISPATCH.json")
    if not intent or intent.get("dispatch_id") != dispatch_id or intent.get("config_sha256") != doc["sha256"]:
        raise ConfigError("worker has no matching dispatch intent")
    if (td / "EXIT.json").exists() or (td / "START.json").exists():
        raise ConfigError("worker already started; no automatic redispatch")
    if task["resource"] == "npu":
        if lane_fd is None or intent.get("lane") is None:
            raise ConfigError("NPU worker requires an inherited shared lane lock")
        if os.fstat(lane_fd).st_ino != Path(intent["lane"]["lock_path"]).stat().st_ino:
            raise ConfigError("worker lane lock identity mismatch")
    store.write(td / "START.json", {"at": now(), "dispatch_id": dispatch_id, "argv": task["argv"],
                                   "cwd": task["cwd"], "env": intent["env"], **identity()}, exclusive=True)
    process = None
    state, code, reason = "UNKNOWN", None, None
    try:
        # Queued tasks may start long after prepare/controller startup. Recheck
        # the frozen inputs in the actual worker before any business invocation.
        assert_frozen(doc)
        # Recheck containment of declared write paths immediately before execution.
        for output in task["outputs"]:
            store.checked(output)
            if os.path.lexists(output):
                raise ConfigError(f"{tid}: output already exists without dispatch; original attempt preserved")
        env = task_environment(doc, task)
        for key in MANAGED_ENV:
            store.mkdir(env[key])
        with store.log(td / "stdout.log") as stdout, store.log(td / "stderr.log") as stderr:
            process = subprocess.Popen(task["argv"], cwd=task["cwd"], env=env, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, start_new_session=True,
                                       pass_fds=(() if lane_fd is None else (lane_fd,)))
            store.write(td / "CHILD.json", {"at": now(), "argv": task["argv"], **identity(process.pid)}, exclusive=True)
            store.write(td / "STATUS.json", {"at": now(), "state": "RUNNING", "dispatch_id": dispatch_id})
            try:
                code = process.wait(timeout=task.get("timeout_seconds"))
            except subprocess.TimeoutExpired:
                # An argv can have external effects already; timeout is not proof of request failure.
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
                reason = "timeout; external request outcome unknown; no retry"
            if reason is None:
                if code in task["unknown_exit_codes"]:
                    state, reason = "UNKNOWN", "configured observation/transport error; no retry"
                elif code == 0:
                    for output in task["outputs"]:
                        store.checked(output)
                    missing = [output for output in task["outputs"] if not Path(output).exists()]
                    state, reason = ("FAILED", f"missing declared outputs: {missing}") if missing else ("SUCCEEDED", None)
                elif code < 0:
                    state, reason = "UNKNOWN", "task terminated by signal; external request outcome unknown"
                else:
                    state, reason = "FAILED", "argv returned nonzero"
                if task["resource"] == "npu" and state != "SUCCEEDED":
                    state = "UNKNOWN"
                    reason = (reason or "NPU task did not succeed") + "; contact-unsettled; lane retained"
    except (OSError, ConfigError) as error:
        # Popen errors before spawn are definite local failures; after spawn remain unknown.
        state = "FAILED" if process is None else "UNKNOWN"
        reason = repr(error)
    finally:
        if task["resource"] == "npu" and state != "SUCCEEDED":
            state = "UNKNOWN"
            if "contact-unsettled" not in (reason or ""):
                reason = (reason or "NPU task interrupted") + "; contact-unsettled; lane retained"
        terminal = {"at": now(), "state": state, "dispatch_id": dispatch_id, "returncode": code,
                    "config_sha256": doc["sha256"],
                    "contact_settled": state == "SUCCEEDED" and code == 0,
                    "reason": reason, "argv": task["argv"]}
        store.write(td / "EXIT.json", terminal, exclusive=True)
        store.write(td / "STATUS.json", terminal)
        if lane_fd is not None:
            os.close(lane_fd)
    return 0 if state == "SUCCEEDED" else 2
