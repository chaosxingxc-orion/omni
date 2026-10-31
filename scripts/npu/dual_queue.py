#!/usr/bin/env python3
"""Run the NPU experiment queue and the CPU analysis queue independently.

The NPU queue remains guarded by ``card_queue_guardian.py`` and owns the card
admission gate. The CPU queue is checked to contain no card-gated jobs, then is
started beside it. A lost SSH session therefore leaves both queue guardians
and their detached jobs alive. This coordinator never retries a job and never
modifies an existing queue state.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

SCHEMA = "saea-dual-queue-v1"
STATE_SCHEMA = "saea-dual-queue-state-v1"
TERMINAL = {"done", "failed", "stopped"}


class DualQueueError(ValueError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.%d" % os.getpid())
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _queue_file(path: object, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(path, str) or not path:
        raise DualQueueError("%s queue path is required" % label)
    queue_path = Path(path).resolve()
    doc = read_json(queue_path)
    if not isinstance(doc, dict) or doc.get("schema") != "saea-card-queue-v1":
        raise DualQueueError("%s queue must use saea-card-queue-v1: %s" % (label, queue_path))
    jobs = doc.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise DualQueueError("%s queue has no jobs" % label)
    return queue_path, doc


def validate_config(config_path: Path) -> dict[str, Any]:
    doc = read_json(config_path)
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise DualQueueError("config must use %s" % SCHEMA)
    npu_path, npu = _queue_file(doc.get("npu_queue"), "npu")
    cpu_path, cpu = _queue_file(doc.get("cpu_queue"), "cpu")
    cpu_dir_hint = Path(doc.get("cpu_queue_dir", cpu_path.with_suffix(""))).resolve()
    npu_dir_hint = Path(doc.get("npu_queue_dir", npu_path.with_suffix(""))).resolve()
    if npu_dir_hint == cpu_dir_hint:
        raise DualQueueError("NPU and CPU queue directories must be distinct")
    # The NPU queue is deliberately a card-consumer queue.  Keeping an
    # ungated job here would let a CPU-only step occupy the experiment lane
    # and make the advertised full-card invariant false.
    for job in npu["jobs"]:
        if job.get("enabled", True) is False:
            continue
        if job.get("requires_cards_free", False) is not True:
            raise DualQueueError(
                "NPU job %r must require the card gate; put CPU analysis in the CPU queue" % job.get("id")
            )
        wait = job.get("wait_for_final")
        if wait:
            try:
                Path(wait).resolve().relative_to(cpu_dir_hint)
            except ValueError:
                pass
            else:
                raise DualQueueError(
                    "NPU job %r waits on a CPU queue artifact; pre-authorise the next NPU identity instead" % job.get("id")
                )
    for job in cpu["jobs"]:
        if not isinstance(job, dict):
            raise DualQueueError("cpu queue contains a non-object job")
        if job.get("enabled", True) is False:
            continue
        if job.get("requires_cards_free", False):
            raise DualQueueError(
                "cpu job %r requests cards; CPU analysis must never gate or occupy the NPU queue" % job.get("id")
            )
        if job.get("cards_policy") not in (None, "all_free", "no_other"):
            raise DualQueueError("cpu job %r has an invalid cards_policy" % job.get("id"))
    npu_dir = npu_dir_hint
    cpu_dir = cpu_dir_hint
    guardian = doc.get("guardian")
    if guardian is None:
        guardian = str(Path(__file__).with_name("card_queue_guardian.py"))
    guardian_path = Path(guardian).resolve()
    if not guardian_path.is_file():
        raise DualQueueError("guardian script does not exist: %s" % guardian_path)
    return {
        "schema": SCHEMA,
        "npu_queue": str(npu_path),
        "cpu_queue": str(cpu_path),
        "npu_queue_dir": str(npu_dir),
        "cpu_queue_dir": str(cpu_dir),
        "guardian": str(guardian_path),
        "poll_s": float(doc.get("poll_s", 15.0)),
    }


def _alive(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def queue_state(path: Path) -> dict[str, Any]:
    state = read_json(path / "state.json")
    return state if isinstance(state, dict) else {"state": "missing"}


def _guardian_cmd(cfg: dict[str, Any], kind: str) -> list[str]:
    return [
        sys.executable,
        cfg["guardian"],
        "--queue",
        cfg[f"{kind}_queue"],
        "--queue-dir",
        cfg[f"{kind}_queue_dir"],
        "--poll-s",
        str(cfg["poll_s"]),
    ]


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def start(config_path: Path, state_dir: Path) -> dict[str, Any]:
    cfg = validate_config(config_path)
    _ensure_dir(str(state_dir))
    lock_path = state_dir / "dual.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise DualQueueError("another dual-queue coordinator holds %s" % lock_path) from error
        state_path = state_dir / "state.json"
        old = read_json(state_path)
        if isinstance(old, dict) and old.get("schema") == STATE_SCHEMA:
            entries = old.get("queues", {})
            if all(_alive(entries.get(kind, {}).get("pid")) for kind in ("npu", "cpu")):
                return old
        queues: dict[str, Any] = {}
        for kind in ("npu", "cpu"):
            qdir = Path(cfg[f"{kind}_queue_dir"])
            _ensure_dir(str(qdir))
            qstate = queue_state(qdir)
            if qstate.get("state") in TERMINAL:
                # A terminal queue is preserved. Starting a new coordinator must
                # never silently use --retry-failed or replay its identities.
                queues[kind] = {"state": "terminal", "queue_state": qstate.get("state"), "pid": None}
                continue
            cmd = _guardian_cmd(cfg, kind)
            log_path = state_dir / (kind + ".guardian.log")
            with log_path.open("ab") as log:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            queues[kind] = {"state": "started", "pid": proc.pid, "cmd": cmd, "log": str(log_path)}
        result = {"schema": STATE_SCHEMA, "config": str(config_path.resolve()), "started": time.time(), "queues": queues}
        write_json(state_path, result)
        return result


def status(config_path: Path, state_dir: Path) -> dict[str, Any]:
    cfg = validate_config(config_path)
    old = read_json(state_dir / "state.json") or {"schema": STATE_SCHEMA, "queues": {}}
    queues = dict(old.get("queues", {}))
    for kind in ("npu", "cpu"):
        qdir = Path(cfg[f"{kind}_queue_dir"])
        entry = dict(queues.get(kind, {}))
        entry["pid_alive"] = _alive(entry.get("pid"))
        entry["queue"] = queue_state(qdir).get("state", "missing")
        queues[kind] = entry
    return {**old, "queues": queues}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "status"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, type=Path)
        p.add_argument("--state-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = start(args.config, args.state_dir) if args.command == "start" else status(args.config, args.state_dir)
    except DualQueueError as error:
        print("refused: %s" % error, file=sys.stderr)
        return 5
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
