#!/usr/bin/env python3
"""RECORDED-run driver for stream S7 (plan 3.4, 3.8 item 9, 3.9 item 8): R0 then the levels of a run list, on host 181.

One detached, checkpointed, resumable run. The run list is DATA (JSON, schema saea-record-run-list-v1); the planner
writes it per branch. This script decides nothing about WHICH levels to run.

R0 (each step checkpointed, in this order; any failure STOPS the chain):
  gate-cards     npu-smi info; every card below the profile's busy_threshold_mib, else STOP (the threshold is never raised)
  disk           CephFS free space of the data root >= r0.min_free_gib
  deploy-up      bash npu_deploy.sh up <profile>            (the only thing that starts servers; never restart/stop/down)
  health         GET /health and /v1/models on every instance
  deploy-status  bash npu_deploy.sh status <profile>        (recorded)
  startup-lines  docker logs <instance> | KV-cache / engine startup lines (KV per card is re-read here)
  receipt        docker run --rm --pid host ... scripts/write_vllm_runtime_receipt.py --instance <name>=<pid> ...
  verify-base    OUT=... bash npu_verify_base.sh <profile> <base-lock> --rehash-weights --receipt <receipt>; 11/11 PASS required
  smoke          npu_smoke_thinking.py on every instance (overflow, template, tool-call, ...); exit 0 required
  manifest       profile hash, ceiling, client seed, decoding seed, server --seed, probe seed, git head
Then, per run and per level: greedy probe BEFORE (sequential send, every instance in parallel), the sweep client with
--requests-per-level = passes x workers x 3 (and --min-tokens when set), greedy probe AFTER, agreement recorded (never
gated). Plan 3.4 thresholds via prelim_chain.evaluate; the plan 3.8 worst-case admissibility rule (zero preemptions on
EVERY instance with a readable counter, sampled KV peak <= 90 %) when the run says so; a ceiling-bound level (waiting
gauge > 0 at the scheduler ceiling) FAILS. A dead endpoint or container is a level FAIL and STOPS the chain; nothing is
ever restarted. A failing level follows the run's on_fail: stop_run (default), continue, or stop_chain. The resident
service is left UP at the end. Rule (b): a level above the list's admitted_level (or the profile ceiling) is refused.

usage (on host 181, detached):
  nohup setsid python3 record_runs.py --run-list <list.json> < /dev/null > <chain-dir>/chain.log 2>&1 &
  resume:  python3 record_runs.py --resume --chain-dir <chain-dir>
  rehearse (CPU only; prints every command): python3 record_runs.py --dry-run --run-list <list.json> [--die r-A2-L32]
Layout: <logs>/record-runs-<ts>/{config.json,run-list.json,chain-status.json,manifest.json,steps/*.json,chain.log};
reports <logs>/sweep-rec-<run id>[-mintok<N>]-<ts>-L<level>.json (+ .row.json, summary, -npu-smi.json, probes).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import prelim_chain as pc  # noqa: E402  (evaluate, Liveness, RealOps, read_cards, ... : one copy of the plan 3.4 rules)

ROOT, LOGS, TREE, IMAGE = pc.ROOT, pc.LOGS, pc.TREE, pc.IMAGE
NPU_DEPLOY = str(SCRIPTS / "npu_deploy.sh")
NPU_VERIFY = str(SCRIPTS / "npu_verify_base.sh")
SMOKE = str(SCRIPTS / "npu_smoke_thinking.py")
PROBE = str(SCRIPTS / "greedy_probe.py")
RUN_LIST_SCHEMA = "saea-record-run-list-v1"
WATCHDOG_HOST = "npu2"  # host 175 ("npu") is not ours: no registry entry is ever written for it
CONTACTS = 3  # contacts per trajectory (the sweep client's default; one request pass = workers x CONTACTS requests)
LOCK_KEY = "qwen3-omni-30b-a3b-thinking-bf16"
EXPECTED_CHECKS = 11
STARTUP_RE = re.compile(r"KV cache size|Available KV cache memory|Initializing a V1 LLM engine|Block size is set")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$")
ON_FAIL = ("stop_run", "continue", "stop_chain")
#: Optional per-run sweep-client overrides (run key "sweep"): {run-list key: synthetic_sweep.py flag}. Absent = the sweep's own
#: defaults (max_tokens 4096, thinking budget 3072, prompt max 20000, timeout 420 s), i.e. the pre-existing behaviour. They exist
#: so a run list can measure a worst case at a RAISED output cap (--min-tokens must not exceed --max-tokens).
SWEEP_FLAGS = {"max_tokens": "--max-tokens", "thinking_budget": "--thinking-budget", "prompt_max": "--prompt-max",
               "timeout": "--timeout"}
PROBE_DECODING_SEED = 777  # greedy_probe.py's fixed decoding seed
EXIT_STOP, EXIT_INTERRUPTED, EXIT_UNEXPECTED, EXIT_GATE_FAILED, EXIT_DIED = 2, 3, 4, 5, 6


class GateFailed(pc.ChainStop):
    """A level of a stop_chain run failed: the planner must re-plan (for example the admitted level drops)."""


# ---------------------------------------------------------------- run list (data) and profile
def profile_path(profile_id: str) -> Path:
    return Path(TREE) / "configs" / "deployment" / f"{profile_id}.json"


def load_profile(profile_id: str) -> dict:
    path = profile_path(profile_id)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise pc.ChainStop(f"deployment profile {path} unreadable: {error}")
    if document.get("profile_id") != profile_id:
        raise pc.ChainStop(f"profile_id {document.get('profile_id')!r} is not the file stem {profile_id!r}")
    document["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return document


def validate_run_list(doc: dict, profile: dict) -> dict:
    """The normalised run list, or ChainStop naming the first violated rule. Pure; no host access."""
    if not isinstance(doc, dict) or doc.get("schema") != RUN_LIST_SCHEMA:
        raise pc.ChainStop(f"run list schema must be {RUN_LIST_SCHEMA!r}")
    ceiling = profile["max_num_seqs"]
    admitted = doc.get("admitted_level")
    if not isinstance(admitted, int) or isinstance(admitted, bool) or not 0 < admitted <= ceiling:
        raise pc.ChainStop(f"admitted_level {admitted!r} must be an integer in 1..{ceiling} (the profile ceiling)")
    if doc.get("profile") != profile["profile_id"]:
        raise pc.ChainStop(f"run list profile {doc.get('profile')!r} != loaded profile {profile['profile_id']!r}")
    runs, seen = doc.get("runs"), set()
    if not isinstance(runs, list) or not runs:
        raise pc.ChainStop("run list needs a non-empty runs list")
    out = []
    for run in runs:
        rid = run.get("id")
        if not isinstance(rid, str) or not RUN_ID.match(rid) or rid in seen:
            raise pc.ChainStop(f"run id {rid!r} is invalid or repeated")
        seen.add(rid)
        levels = run.get("levels")
        if not isinstance(levels, list) or not levels or len(set(levels)) != len(levels) or \
                any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in levels):
            raise pc.ChainStop(f"{rid}: levels must be distinct positive integers")
        over = [v for v in levels if v > admitted]
        if over:
            raise pc.ChainStop(f"{rid}: level(s) {over} exceed the admitted level {admitted} (rule (b): no recorded run "
                               f"above the admitted level)")
        passes = run.get("passes", 1)
        if not isinstance(passes, int) or isinstance(passes, bool) or passes < 1:
            raise pc.ChainStop(f"{rid}: passes must be an integer >= 1")
        min_tokens = run.get("min_tokens")
        if min_tokens is not None and (not isinstance(min_tokens, int) or isinstance(min_tokens, bool) or min_tokens < 1):
            raise pc.ChainStop(f"{rid}: min_tokens must be null or a positive integer")
        on_fail = run.get("on_fail", "stop_run")
        if on_fail not in ON_FAIL:
            raise pc.ChainStop(f"{rid}: on_fail must be one of {ON_FAIL}")
        sweep = run.get("sweep") or {}
        if not isinstance(sweep, dict) or set(sweep) - set(SWEEP_FLAGS) or \
                any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in sweep.values()):
            raise pc.ChainStop(f"{rid}: sweep must be an object over {sorted(SWEEP_FLAGS)} with positive integer values")
        if min_tokens is not None and min_tokens > sweep.get("max_tokens", 4096):
            raise pc.ChainStop(f"{rid}: min_tokens {min_tokens} exceeds the sweep max_tokens "
                               f"{sweep.get('max_tokens', 4096)} (set sweep.max_tokens)")
        max_latency = run.get("max_latency_s", pc.MAX_LATENCY_S)
        if isinstance(max_latency, bool) or not isinstance(max_latency, (int, float)) or max_latency < pc.MAX_LATENCY_S:
            raise pc.ChainStop(f"{rid}: max_latency_s must be a number >= {pc.MAX_LATENCY_S:g} (the plan 3.4 rule can only "
                               f"scale UP with the output cap, never down)")
        if max_latency > sweep.get("timeout", 420):
            raise pc.ChainStop(f"{rid}: max_latency_s {max_latency:g} exceeds the sweep client timeout "
                               f"{sweep.get('timeout', 420)} (set sweep.timeout above it)")
        out.append({"id": rid, "levels": list(levels), "passes": passes, "min_tokens": min_tokens, "sweep": dict(sweep),
                    "max_latency_s": float(max_latency),
                    "admissibility": bool(run.get("admissibility", False)), "on_fail": on_fail,
                    "note": str(run.get("note", ""))})
    r0 = dict(doc.get("r0") or {})
    if not r0.get("base_lock"):
        raise pc.ChainStop("r0.base_lock (the base-lock JSON that npu_verify_base.sh checks against) is required")
    r0.setdefault("min_free_gib", 50)
    r0.setdefault("deploy_wait_seconds", 2400)
    r0.setdefault("smoke_audio", f"{ROOT}/work/scratch/smoke-181/tone.wav")
    return {"schema": RUN_LIST_SCHEMA, "example": bool(doc.get("example", False)), "label": str(doc.get("label", "")),
            "profile": profile["profile_id"], "admitted_level": admitted, "ceiling_bound_is_failure":
            bool(doc.get("ceiling_bound_is_failure", True)), "seed": str(doc.get("seed", pc.SEED)),
            "client_cpuset": str(doc.get("client_cpuset", pc.CLIENT_CPUSET)), "r0": r0, "runs": out}


def prefix_for(logs: str, ts: str, run: dict) -> str:
    mint = f"-mintok{run['min_tokens']}" if run["min_tokens"] else ""
    return f"{logs}/sweep-rec-{run['id']}{mint}-{ts}"


def requests_for(run: dict, level: int, instances: int) -> int:
    """requests-per-level of the sweep client: passes x workers x contacts (one pass keeps every worker busy once)."""
    return run["passes"] * level * instances * CONTACTS


# ---------------------------------------------------------------- exact commands (pure; the dry run prints them)
def docker_user() -> str:
    return f"{os.getuid()}:{os.getgid()}"


def deploy_up_cmd(profile_id: str) -> list[str]:
    return ["bash", NPU_DEPLOY, "up", profile_id]


def deploy_status_cmd(profile_id: str) -> list[str]:
    return ["bash", NPU_DEPLOY, "status", profile_id]


def receipt_cmd(pids: dict[str, int], image_id: str, out: str, user: str) -> list[str]:
    """The runtime receipt producer in a throw-away saea-base container that sees the servers' pids (--pid host), as the
    uid that runs the servers (the producer reads /proc/<pid>/environ). The data root is read-only; only the output
    directory is writable. Zero model contacts; no process is signalled."""
    out_dir = str(Path(out).parent)
    cmd = ["docker", "run", "--rm", "--pid", "host", "--network", "host", "--user", user, "-e", "HOME=/tmp",
           "--cpuset-cpus", pc.CLIENT_CPUSET, "-v", f"{ROOT}:{ROOT}:ro", "-v", f"{out_dir}:{out_dir}", "-w", TREE,
           "-e", f"PYTHONPATH={TREE}/src", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", f"SPEECHRL_DATA_DIR={ROOT}",
           "--entrypoint", "/opt/saea-venv/bin/python", IMAGE, "scripts/write_vllm_runtime_receipt.py"]
    for name, pid in pids.items():
        cmd += ["--instance", f"{name}={pid}"]
    return cmd + ["--model-lock-key", LOCK_KEY, "--materialization-root", ROOT, "--image-ref", IMAGE,
                  "--image-id", image_id, "--output", out]


def verify_cmd(profile_id: str, base_lock: str, receipt: str) -> list[str]:
    return ["bash", NPU_VERIFY, profile_id, base_lock, "--rehash-weights", "--receipt", receipt]


def smoke_cmd(url: str, audio: str, out: str) -> list[str]:
    return ["docker", "run", "--rm", "--network", "host", "--user", docker_user(), "-e", "HOME=/tmp",
            "--cpuset-cpus", pc.CLIENT_CPUSET, "-v", f"{ROOT}:{ROOT}", "-w", TREE, "-e", f"PYTHONPATH={TREE}/src",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "/opt/saea-venv/bin/python", IMAGE,
            SMOKE, "--base-url", url, "--audio", audio, "--out", out]


def probe_cmd(url: str, out: str, bodies: str) -> list[str]:
    return [sys.executable, PROBE, "send", "--url", url, "--out", out, "--sequential", "--bodies-file", bodies]


def compare_cmd(a: str, b: str, out: str) -> list[str]:
    return [sys.executable, PROBE, "compare", a, b, "--out", out]


def sweep_cmd(name: str, urls: list[str], level: int, requests: int, min_tokens: int | None, seed: str, label: str,
              out: str, cpuset: str, sweep: dict | None = None) -> list[str]:
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "host", "--user", docker_user(), "-e", "HOME=/tmp",
           "--cpuset-cpus", cpuset, "-v", f"{ROOT}:{ROOT}", "-w", TREE, "-e", f"PYTHONPATH={TREE}/src",
           "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "/opt/saea-venv/bin/python", IMAGE,
           "scripts/serving/synthetic_sweep.py"]
    for url in urls:
        cmd += ["--base-url", url]
    cmd += ["--model", pc.MODEL, "--levels", str(level), "--requests-per-level", str(requests), "--seed", seed,
            "--liveness-interval", str(pc.LIVENESS_SECONDS), "--label", label, "--out", out]
    if min_tokens is not None:
        cmd += ["--min-tokens", str(min_tokens)]
    for key in SWEEP_FLAGS:
        if sweep and key in sweep:
            cmd += [SWEEP_FLAGS[key], str(sweep[key])]
    return cmd


def seed_from_cmd(cmd: list[str]) -> str | None:
    """The server's --seed value from a container Cmd (None when the flag is absent)."""
    for i, token in enumerate(cmd):
        if token == "--seed" and i + 1 < len(cmd):
            return str(cmd[i + 1])
        if token.startswith("--seed="):
            return token.split("=", 1)[1]
    return None


def startup_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if STARTUP_RE.search(line)]


def verification_verdict(report: dict | None) -> tuple[bool, str]:
    """11/11 PASS required: verdict pass, every one of the 11 checks selected (so --rehash-weights ran) and none failed."""
    if not isinstance(report, dict):
        return False, "verification report unreadable"
    selected, failed = report.get("selected_checks") or [], report.get("failed_checks") or []
    if len(selected) != EXPECTED_CHECKS:
        return False, f"{len(selected)} of {EXPECTED_CHECKS} checks selected (skipped: {report.get('skipped_checks')})"
    if report.get("verdict") != "pass" or failed:
        return False, f"verdict {report.get('verdict')!r}, failed checks {failed}"
    return True, f"{EXPECTED_CHECKS}/{EXPECTED_CHECKS} PASS"


# ---------------------------------------------------------------- real operations (host 181 only)
class RecordOps(pc.RealOps):
    """Reads and runs; starts servers only through npu_deploy.sh up. There is no stop, restart or down in this class:
    the only container it removes is its own sweep client (name saea-rec-*)."""

    def sh(self, argv: list[str], timeout: float, env: dict | None = None) -> tuple[int, str]:
        pc.log("RUN " + " ".join(argv))
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                              env=dict(os.environ, **(env or {})))
        return done.returncode, (done.stdout + done.stderr)

    def hostname(self) -> str:
        return Path("/etc/hostname").read_text().strip()

    def cards_used(self) -> dict[int, int]:
        return {c: v[1] for c, v in pc.read_cards().items()}

    def disk_free_gib(self, path: str) -> float:
        return shutil.disk_usage(path).free / 2**30

    def git_head(self) -> str:
        code, out = self.sh(["git", "-C", TREE, "rev-parse", "HEAD"], 30)
        dirty = self.sh(["git", "-C", TREE, "status", "--porcelain"], 60)[1].strip()
        return out.strip() + ("+dirty" if dirty else "")

    def deploy_up(self, profile_id: str, wait_seconds: int) -> tuple[int, str]:
        return self.sh(deploy_up_cmd(profile_id), wait_seconds + 600, {"DEPLOY_WAIT_SECONDS": str(wait_seconds)})

    def deploy_status(self, profile_id: str) -> str:
        return self.sh(deploy_status_cmd(profile_id), 120)[1]

    def health(self, instances: list[dict]) -> dict[str, bool]:
        out = {}
        for inst in instances:
            base = f"http://127.0.0.1:{inst['port']}"
            h, m = pc.http_get(base + "/health"), pc.http_get(base + "/v1/models")
            out[inst["name"]] = bool(h and h[0] == 200 and m and m[0] == 200 and pc.MODEL in m[1])
        return out

    def dead_instances(self, instances: list[dict]) -> list[str]:
        live = pc.Liveness([(i["name"], f"http://127.0.0.1:{i['port']}") for i in instances], pc.http_get,
                           self.docker_names, pause=pc.LIVENESS_PAUSE_SECONDS)
        return live.dead_instances()

    def startup_lines(self, name: str) -> list[str]:
        return startup_lines(self.sh(["docker", "logs", name], 120)[1])

    def server_seed(self, name: str) -> str | None:
        code, out = self.sh(["docker", "inspect", "-f", "{{json .Config.Cmd}}", name], 30)
        return seed_from_cmd(json.loads(out)) if code == 0 else None

    def write_receipt(self, names: list[str], out: str) -> tuple[int, str]:
        pids = {}
        for name in names:
            code, text = self.sh(["docker", "inspect", "-f", "{{.State.Pid}}", name], 30)
            if code != 0 or not text.strip().isdigit() or int(text) <= 0:
                return 1, f"no running pid for {name}: {text.strip()[:200]}"
            pids[name] = int(text)
        code, image_id = self.sh(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], 60)
        if code != 0 or not image_id.strip().startswith("sha256:"):
            return 1, f"image id unreadable: {image_id.strip()[:200]}"
        user = self.sh(["docker", "inspect", "-f", "{{.Config.User}}", names[0]], 30)[1].strip() or "0:0"
        return self.sh(receipt_cmd(pids, image_id.strip(), out, user), 1800)

    def verify_base(self, profile_id: str, lock: str, receipt: str, out: str) -> tuple[int, str]:
        return self.sh(verify_cmd(profile_id, lock, receipt), 3 * 3600, {"OUT": out})

    def smoke(self, url: str, audio: str, out: str) -> tuple[int, str]:
        return self.sh(smoke_cmd(url, audio, out), 1800)

    def ensure_probe_bodies(self, path: str) -> str:
        import greedy_probe
        bodies = greedy_probe.load_or_build_bodies(path)
        return greedy_probe.bodies_digest(bodies)

    def probe(self, urls: list[str], outs: list[str], bodies: str) -> list[int]:
        with cf.ThreadPoolExecutor(max_workers=len(urls)) as pool:
            return list(pool.map(lambda uo: self.sh(probe_cmd(uo[0], uo[1], bodies), 3600)[0], zip(urls, outs)))

    def compare(self, a: str, b: str, out: str) -> dict | None:
        code, text = self.sh(compare_cmd(a, b, out), 300)
        try:
            doc = json.loads(text[text.index("{"):]) if code == 0 else None
        except ValueError:
            doc = None
        return None if doc is None else {k: doc.get(k) for k in ("requests", "byte_identical", "mismatched_indices",
                                                                 "max_abs_logprob_diff")}

    def measure_level(self, step, prefix, level, urls, min_tokens, label, names=None, *, requests=None,
                      seed=pc.SEED, cpuset=pc.CLIENT_CPUSET, sweep=None) -> dict:
        names = names or [f"instance-{i}" for i in range(len(urls))]
        out = Path(f"{prefix}-L{level}.json")
        client = f"saea-rec-{Path(prefix).name}-L{level}"[:120]
        subprocess.run(["docker", "rm", "-f", client], capture_output=True)  # our own client only, from a dead chain
        command = sweep_cmd(client, urls, level, requests, min_tokens, seed, label, str(out), cpuset, sweep)
        self.set_phase(prefix, level)
        poll = {"waiting": 0.0, "stop": threading.Event(), "kv": [None] * len(urls)}
        dead: dict = {"names": []}

        def poller() -> None:
            while not poll["stop"].is_set():
                for i, url in enumerate(urls):
                    got = pc.http_get(url + "/metrics", timeout=3)
                    if got and got[0] == 200:
                        poll["waiting"] = max(poll["waiting"], pc.prom_sum(got[1], "vllm:num_requests_waiting") or 0.0)
                        kv = pc.prom_sum(got[1], "vllm:kv_cache_usage_perc")
                        if kv is not None:
                            poll["kv"][i] = kv if poll["kv"][i] is None else max(poll["kv"][i], kv)
                poll["stop"].wait(pc.POLL_SECONDS)

        live = pc.Liveness(list(zip(names, urls)), pc.http_get, self.docker_names, pause=pc.LIVENESS_PAUSE_SECONDS)

        def watcher() -> None:  # host-side liveness (container in docker ps AND /v1/models) beside the client's own
            while not poll["stop"].wait(pc.LIVENESS_SECONDS):
                gone = live.dead_instances()
                if gone:
                    dead.update(names=gone, at=pc.utc_now())
                    pc.log(f"{step}: LIVENESS dead instance(s) {gone}; removing the level client {client}")
                    subprocess.run(["docker", "rm", "-f", client], capture_output=True)
                    return

        threads = [threading.Thread(target=poller, daemon=True), threading.Thread(target=watcher, daemon=True)]
        for t in threads:
            t.start()
        t0 = time.time()
        pc.log(f"{step}: client command {' '.join(command)}")
        try:
            with open(f"{prefix}-L{level}.stderr.log", "w") as stderr:
                code = subprocess.run(command, stdout=stderr, stderr=subprocess.STDOUT).returncode
        except BaseException:
            subprocess.run(["docker", "rm", "-f", client], capture_output=True)
            raise
        finally:
            poll["stop"].set()
        t1 = time.time()
        for t in threads:
            t.join(timeout=10)
        if not dead["names"]:
            dead.update(names=live.dead_instances(), at=pc.utc_now())
        report = pc.read_json(out)
        if not dead["names"] and code == 3 and report and report.get("endpoint_died"):
            eps = report["endpoint_died"].get("endpoints") or []
            dead["names"] = [names[e["instance"]] for e in eps if e.get("instance", 99) < len(names)] or [names[0]]
        if dead["names"]:
            raise pc.EndpointDied(dead["names"][0], {
                "dead_instances": dead["names"], "detected_at": dead.get("at"), "client_exit_code": code,
                "client_report_endpoint_died": (report or {}).get("endpoint_died"), "kv_peak_polled": poll["kv"]})
        if code != 0 or not report:
            raise pc.ChainStop(f"{step}: sweep client exited {code}; see {prefix}-L{level}.stderr.log")
        summary = dict(report["levels"][0], instances=len(urls))
        generated = dt.datetime.fromisoformat(report["generated_at"]).timestamp()
        with self.lock:
            snapshot = [s for s in self.samples if "error" not in s]
        window = [s for s in snapshot if t0 <= s["t"] <= t1]
        load = [s for s in snapshot if generated - summary["wall_s"] - 1 <= s["t"] <= generated + 2]
        free = [s["hbm_total_mib"] - used for s in window for used in s["hbm_used_mib"].values()]
        ai = [v for s in load for v in s["aicore_pct"].values()]
        return {"summary": summary, "report": out.name, "report_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "min_hbm_free_mib": min(free) if free else None,
                "aicore_mean": sum(ai) / len(ai) if ai else None, "aicore_max": max(ai) if ai else None,
                "samples": len(load), "poll_waiting_max": poll["waiting"], "poll_kv_peak": poll["kv"], "names": names,
                "instances": len(urls), "max_tokens_sent": (report.get("settings") or {}).get("max_tokens"),
                "min_tokens_sent": (report.get("settings") or {}).get("min_tokens"),
                "client_seed": (report.get("settings") or {}).get("seed"),
                "client_decoding_seed": (report.get("settings") or {}).get("decoding_seed")}


# ---------------------------------------------------------------- the chain
class RecordChain:
    def __init__(self, ops, chain_dir: Path, ts: str, logs: str, run_list: dict, profile: dict, dry: bool = False,
                 sleep=time.sleep) -> None:
        self.ops, self.dir, self.ts, self.logs, self.rl, self.profile, self.dry = ops, chain_dir, ts, logs, run_list, \
            profile, dry
        self.sleep = sleep
        self.steps_dir = chain_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.instances = [{"name": i["name"], "port": i["port"], "cards": i["cards"],
                           "url": f"http://127.0.0.1:{i['port']}"} for i in profile["instances"]]
        self.urls = [i["url"] for i in self.instances]
        self.names = [i["name"] for i in self.instances]
        self.ceiling = profile["max_num_seqs"]
        self.paths = {"receipt": f"{logs}/receipt-{profile['profile_id']}-{ts}.json",
                      "verification": f"{logs}/base-verification-{profile['profile_id']}-rec-{ts}.json",
                      "smoke": f"{logs}/smoke-rec-{ts}", "bodies": str(chain_dir / "probe-bodies.json"),
                      "manifest": str(chain_dir / "manifest.json")}
        self.status = {"chain": chain_dir.name, "mode": "record", "state": "running", "current_step": None,
                       "started": pc.utc_now(), "updated": pc.utc_now(), "pid": os.getpid(), "reason": None,
                       "profile": profile["profile_id"], "admitted_level": run_list["admitted_level"],
                       "example": run_list["example"], "decisions": {}, "steps": {}}
        old = pc.read_json(chain_dir / "chain-status.json")
        if old:
            self.status["decisions"], self.status["steps"] = old.get("decisions", {}), old.get("steps", {})
            self.status["started"] = old.get("started", self.status["started"])
        self.rows: dict[str, list[dict]] = {r["id"]: [] for r in run_list["runs"]}
        self._hb_stop = threading.Event()

    # --- checkpoints and status (same shape as prelim_chain: chain-status.json, steps/<step>.done.json)
    def done(self, step: str) -> dict | None:
        return pc.read_json(self.steps_dir / f"{step}.done.json")

    def mark(self, step: str, decision: dict) -> None:
        pc.write_atomic(self.steps_dir / f"{step}.done.json", json.dumps(
            {"step": step, "finished": pc.utc_now(), "decision": decision}, indent=2, sort_keys=True) + "\n")
        self.status["steps"][step] = {"state": "done", "finished": pc.utc_now(),
                                      "decision": {k: v for k, v in decision.items() if k != "row"}}
        self.write_status()

    def set_current(self, step: str | None) -> None:
        self.status["current_step"], self.status["current_since"] = step, pc.utc_now()
        if step:
            pc.log(f"STEP {step}")
        self.write_status()

    def write_status(self) -> None:
        self.status["updated"] = pc.utc_now()
        pc.write_atomic(self.dir / "chain-status.json", json.dumps(self.status, indent=1, sort_keys=True) + "\n")

    def heartbeat(self) -> None:
        def loop() -> None:
            while not self._hb_stop.wait(60):
                pc.log(f"heartbeat step={self.status['current_step']} since={self.status.get('current_since')}")
                self.write_status()
        threading.Thread(target=loop, daemon=True).start()

    def stop(self, step: str, reason: str) -> None:
        pc.write_atomic(self.steps_dir / f"{step}.failed.json", json.dumps(
            {"step": step, "finished": pc.utc_now(), "reason": reason}, indent=2, sort_keys=True) + "\n")
        self.status["steps"][step] = {"state": "failed", "finished": pc.utc_now(), "reason": reason}
        raise pc.ChainStop(f"{step}: {reason}")

    # --- watchdog registry (npu2 only)
    def jobs(self, kind: str, run: dict | None = None) -> list[dict]:
        host = WATCHDOG_HOST
        if kind == "chain":
            return [{"id": f"record-runs-{self.ts}", "kind": "generic_log", "host": host, "stall_s": 3600,
                     "note": "RECORDED runs (R0 + run list); BAD means the chain stopped (dead endpoint, failed gate or "
                             "failed level; see chain-status.json); never restarted by the chain",
                     "params": {"path": str(self.dir / "chain.log"), "bad_regex": "CHAIN-BAD|Traceback",
                                "done_regex": "CHAIN DONE", "status_file": str(self.dir / "chain-status.json")}}]
        if kind == "servers":
            return [{"id": f"vllm-rec-{n}", "kind": "vllm_server", "host": host,
                     "note": f"resident {self.profile['profile_id']} instance, left UP by record-runs-{self.ts}",
                     "params": {"container": n, "port": i["port"]}} for n, i in zip(self.names, self.instances)]
        return [{"id": f"sweep-rec-{run['id']}-{self.ts}", "kind": "sweep", "host": host, "stall_s": 3600,
                 "note": f"RECORDED run {run['id']} ({run['note']})",
                 "params": {"run_prefix": prefix_for(self.logs, self.ts, run)}}]

    # --- R0
    def step(self, name: str, fn):
        prior = self.done(name)
        if prior:
            pc.log(f"{name}: resumed")
            return prior["decision"]
        self.set_current(name)
        decision = fn()
        self.mark(name, decision)
        return decision

    def r0_cards(self) -> dict:
        threshold = self.profile["busy_threshold_mib"]
        needed = sorted({c for i in self.instances for c in i["cards"]})
        used = self.ops.cards_used()
        missing = [c for c in needed if c not in used]
        busy = {c: used[c] for c in needed if c in used and used[c] >= threshold}
        if missing or busy:
            self.stop("gate-cards", f"cards not below busy_threshold_mib {threshold}: busy {busy}, unreadable {missing}; "
                                    f"STOP (the threshold is never raised, nothing is started)")
        return {"threshold_mib": threshold, "hbm_used_mib": {str(c): used[c] for c in needed}}

    def r0_disk(self) -> dict:
        free = self.ops.disk_free_gib(ROOT)
        need = self.rl["r0"]["min_free_gib"]
        if free < need:
            self.stop("disk", f"CephFS free {free:.1f} GiB < {need} GiB")
        return {"free_gib": round(free, 1), "min_free_gib": need}

    def r0_up(self) -> dict:
        code, out = self.ops.deploy_up(self.profile["profile_id"], self.rl["r0"]["deploy_wait_seconds"])
        pc.write_atomic(self.dir / "deploy-up.log", out)
        if code != 0:
            self.stop("deploy-up", f"npu_deploy.sh up exited {code}; nothing is restarted; see deploy-up.log")
        self.ops.patch_jobs(self.jobs("servers"))
        return {"exit": code, "tail": out.strip().splitlines()[-1:] }

    def r0_health(self) -> dict:
        state = self.ops.health(self.instances)
        if not all(state.values()):
            self.stop("health", f"not healthy: {[n for n, ok in state.items() if not ok]}")
        return {"healthy": state}

    def r0_status(self) -> dict:
        text = self.ops.deploy_status(self.profile["profile_id"])
        pc.write_atomic(self.dir / "deploy-status.txt", text)
        return {"status_lines": text.strip().splitlines()}

    def r0_startup(self) -> dict:
        lines = {n: self.ops.startup_lines(n) for n in self.names}
        return {"startup_lines": lines, "seeds": {n: self.ops.server_seed(n) for n in self.names},
                "note": "KV cache per card is read from these lines (the -v2 ceiling changes the capture sizes)"}

    def r0_receipt(self) -> dict:
        path = Path(self.paths["receipt"])
        if path.exists():  # create-only producer: a left-over file of an unfinished step is moved aside, never deleted
            path.rename(path.with_name(path.name + f".stale-{int(time.time())}"))
        code, out = self.ops.write_receipt(self.names, self.paths["receipt"])
        pc.write_atomic(self.dir / "receipt.log", out)
        if code != 0 or (not self.dry and not path.exists()):
            self.stop("receipt", f"receipt producer exited {code}; see receipt.log")
        return {"receipt": self.paths["receipt"], "sha256": file_sha(self.paths["receipt"])}

    def r0_verify(self) -> dict:
        code, out = self.ops.verify_base(self.profile["profile_id"], self.rl["r0"]["base_lock"], self.paths["receipt"],
                                         self.paths["verification"])
        pc.write_atomic(self.dir / "verify-base.log", out)
        ok, why = verification_verdict(pc.read_json(self.paths["verification"]))
        if code != 0 or not ok:
            self.stop("verify-base", f"exit {code}; {why}; STOP (11/11 PASS required)")
        return {"verdict": why, "report": self.paths["verification"], "sha256": file_sha(self.paths["verification"])}

    def r0_smoke(self) -> dict:
        results = {}
        for index, inst in enumerate(self.instances):
            out = f"{self.paths['smoke']}/i{index}"
            code, text = self.ops.smoke(inst["url"], self.rl["r0"]["smoke_audio"], out)
            pc.write_atomic(self.dir / f"smoke-i{index}.log", text)
            results[inst["name"]] = {"exit": code, "dir": out}
        bad = [n for n, r in results.items() if r["exit"] != 0]
        if bad:
            self.stop("smoke", f"smoke re-run failed on {bad}; see smoke-i*.log and {self.paths['smoke']}")
        return {"results": results}

    def r0_manifest(self) -> dict:
        doc = {"schema": "saea-record-run-manifest-v1", "created": pc.utc_now(), "example": self.rl["example"],
               "host": self.ops.hostname(), "git_head": self.ops.git_head(),
               "profile": self.profile["profile_id"], "profile_sha256": self.profile["_sha256"],
               "max_num_seqs_ceiling": self.ceiling, "admitted_level": self.rl["admitted_level"],
               "client_seed": self.rl["seed"], "client_decoding_seed_default": 20260913,
               "server_seeds": (self.done("startup-lines") or {}).get("decision", {}).get("seeds"),
               "probe": {"decoding_seed": PROBE_DECODING_SEED, "send": "sequential",
                         "bodies_sha256": self.ops.ensure_probe_bodies(self.paths["bodies"])},
               "receipt": self.paths["receipt"], "verification": self.paths["verification"],
               "npu_smi_before": self.ops.cards_used(),
               "greedy_agreement_note": "recorded, never gated (plan 3.9 decision 3)",
               "run_list": self.rl}
        pc.write_atomic(self.paths["manifest"], json.dumps(doc, indent=2, sort_keys=True) + "\n")
        return {"manifest": self.paths["manifest"]}

    def r0(self) -> None:
        for name, fn in (("gate-cards", self.r0_cards), ("disk", self.r0_disk), ("deploy-up", self.r0_up),
                         ("health", self.r0_health), ("deploy-status", self.r0_status),
                         ("startup-lines", self.r0_startup), ("receipt", self.r0_receipt),
                         ("verify-base", self.r0_verify), ("smoke", self.r0_smoke), ("manifest", self.r0_manifest)):
            self.step(name, fn)

    # --- levels
    def check_alive(self, step: str, level: int, run: dict) -> None:
        gone = self.ops.dead_instances(self.instances)
        if gone:
            raise pc.EndpointDied(gone[0], {"dead_instances": gone, "checked_before": step})

    def probe_pair(self, prefix: str, level: int, tag: str) -> list[str]:
        outs = [f"{prefix}-L{level}-probe-{tag}-i{n}.json" for n in range(len(self.urls))]
        codes = self.ops.probe(self.urls, outs, self.paths["bodies"])
        return outs if all(c == 0 for c in codes) else []

    def level_step(self, run: dict, level: int) -> dict:
        step = f"{run['id']}-L{level}"
        prior = self.done(step)
        if prior and "row" in prior["decision"]:
            row = prior["decision"]["row"]
            self.rows[run["id"]].append(row)
            pc.log(f"{step}: resumed, pass={row['passes']}")
            return row
        prefix = prefix_for(self.logs, self.ts, run)
        self.set_current(step)
        try:
            self.check_alive(step, level, run)
            before = self.probe_pair(prefix, level, "before")
            self.check_alive(step, level, run)
            label = (f"RECORDED run {run['id']} level {level} (passes {run['passes']}, min_tokens {run['min_tokens']}, "
                     f"admissibility {run['admissibility']}){' EXAMPLE' if self.rl['example'] else ''}")
            measure = self.ops.measure_level(step, prefix, level, self.urls, run["min_tokens"], label, names=self.names,
                                             requests=requests_for(run, level, len(self.urls)), seed=self.rl["seed"],
                                             cpuset=self.rl["client_cpuset"],
                                             **({"sweep": run["sweep"]} if run["sweep"] else {}))
            after = self.probe_pair(prefix, level, "after")
            self.check_alive(step, level, run)
        except pc.EndpointDied as died:
            self.record_died(run, step, level, died)
            raise
        reasons, row = pc.evaluate(measure, level, self.ceiling, worst_case=run["admissibility"],
                                     max_latency_s=run["max_latency_s"])
        if row["ceiling_bound"] and self.rl["ceiling_bound_is_failure"]:
            reasons.append("ceiling-bound: waiting gauge > 0 at the scheduler ceiling")
        agreement = None
        if before and after:
            agreement = [self.ops.compare(b, a, f"{prefix}-L{level}-probe-compare-i{n}.json")
                         for n, (b, a) in enumerate(zip(before, after))]
        row.update({"step": step, "run": run["id"], "passes_requested": run["passes"], "min_tokens": run["min_tokens"], "sweep_overrides": run["sweep"], "max_latency_s_rule": run["max_latency_s"],
                    "admissibility_rule": run["admissibility"], "admitted_level": self.rl["admitted_level"],
                    "max_num_seqs_ceiling": self.ceiling, "client_seed": self.rl["seed"],
                    "client_decoding_seed": measure.get("client_decoding_seed"),
                    "server_seeds": (self.done("startup-lines") or {}).get("decision", {}).get("seeds"),
                    "max_tokens_sent": measure.get("max_tokens_sent"), "min_tokens_sent": measure.get("min_tokens_sent"),
                    "greedy_probe": {"before": before, "after": after, "sequential_before_vs_after": agreement,
                                     "gated": False, "probe_decoding_seed": PROBE_DECODING_SEED},
                    "passes": not reasons, "failing_reasons": reasons})
        pc.write_atomic(f"{prefix}-L{level}.row.json", json.dumps(row, indent=2, sort_keys=True) + "\n")
        self.rows[run["id"]].append(row)
        self.ops.flush(prefix)
        self.mark(step, {"level": level, "passes": row["passes"], "failing_reasons": reasons,
                         "host_output_tok_s": row["host_output_tok_s"], "p10_e2e_tok_s": row["e2e_tok_s"]["p10"],
                         "latency_max_s": row["latency_s"]["max"], "preemptions_delta": row["preemptions_delta"],
                         "per_instance": row["per_instance"], "ceiling_bound": row["ceiling_bound"],
                         "worst_case_admissible": row["worst_case_admissible"], "row": row})
        pc.log(f"{step}: pass={row['passes']} {reasons} host_output_tok_s={row['host_output_tok_s']:.1f}")
        return row

    def record_died(self, run: dict, step: str, level: int, died: pc.EndpointDied) -> None:
        """A dead endpoint is a level FAIL and a chain STOP. The record is .failed.json, not .done.json."""
        record = {"step": step, "level": level, "finished": pc.utc_now(), "passes": False,
                  "failing_reasons": [f"endpoint_died:{died.name}"], "endpoint_died": died.name, "detail": died.detail}
        pc.write_atomic(self.steps_dir / f"{step}.failed.json", json.dumps(record, indent=2, sort_keys=True) + "\n")
        self.status["steps"][step] = {"state": "failed", "finished": pc.utc_now(), "decision": {
            k: record[k] for k in ("level", "passes", "failing_reasons", "endpoint_died")}}
        try:
            self.ops.flush(prefix_for(self.logs, self.ts, run))
        except Exception:
            pass
        self.write_summary(run, False, level, {"endpoint_died": died.name})
        pc.log(f"{step}: FAIL endpoint_died:{died.name}")

    def write_summary(self, run: dict, done: bool, stopped_at: int | None, extra: dict | None = None) -> None:
        doc = {"label": f"RECORDED run {run['id']} on {self.profile['profile_id']}"
                        f"{' (EXAMPLE run list)' if self.rl['example'] else ''}",
               "synthetic": True, "dataset_rows": 0, "seed": self.rl["seed"],
               "host_output_tok_s_definition": "sum of completion tokens over all instances / level wall time",
               "run": {k: run[k] for k in ("id", "levels", "passes", "min_tokens", "sweep", "max_latency_s", "admissibility", "on_fail")},
               "stopped_at_level": stopped_at, "done": done, "levels": self.rows[run["id"]]}
        doc.update(extra or {})
        pc.write_atomic(f"{prefix_for(self.logs, self.ts, run)}.json", json.dumps(doc, indent=2, sort_keys=True) + "\n")

    def run_one(self, run: dict) -> bool:
        """Every level of one run in list order; returns True when no level failed."""
        self.ops.set_phase(prefix_for(self.logs, self.ts, run))
        self.ops.patch_jobs(self.jobs("sweep", run))
        failed_at = None
        for level in run["levels"]:
            step = f"{run['id']}-L{level}"
            if failed_at is not None:
                if not self.done(step):
                    self.mark(step, {"skipped": f"level {failed_at} failed (on_fail {run['on_fail']})"})
                continue
            row = self.level_step(run, level)
            if not row["passes"]:
                failed_at = level if run["on_fail"] != "continue" else None
                if run["on_fail"] == "stop_chain":
                    self.write_summary(run, True, level)
                    self.status["decisions"][run["id"]] = {"passed_levels": passed(self.rows[run["id"]]),
                                                           "stopped_at_level": level}
                    self.write_status()
                    raise GateFailed(f"{step} failed: {row['failing_reasons']}")
        stopped = failed_at
        self.write_summary(run, True, stopped)
        self.status["decisions"][run["id"]] = {
            "passed_levels": passed(self.rows[run["id"]]), "failed_levels": [r["concurrency_per_instance"] for r in
                                                                             self.rows[run["id"]] if not r["passes"]],
            "stopped_at_level": stopped,
            "ceiling_bound_levels": [r["concurrency_per_instance"] for r in self.rows[run["id"]] if r["ceiling_bound"]]}
        self.write_status()
        return failed_at is None

    def run(self) -> int:
        self.heartbeat()
        try:
            self.ops.patch_jobs(self.jobs("chain"))
            self.r0()
            self.ops.start_sampler()
            for run in self.rl["runs"]:
                self.run_one(run)
            self.ops.stop_sampler()
            gone = self.ops.dead_instances(self.instances)
            if gone:
                raise pc.EndpointDied(gone[0], {"dead_instances": gone, "checked": "end of chain"})
            self.finish_manifest()
            self.status.update(state="done", current_step=None, reason=None)
            self.write_status()
            pc.log("resident service left UP (record_runs never stops or restarts a container)")
            pc.log("CHAIN DONE")
            return 0
        except GateFailed as stop:
            return self.bad("failed", str(stop), EXIT_GATE_FAILED)
        except pc.EndpointDied as died:
            return self.bad("failed", f"endpoint_died:{died.name}", EXIT_DIED)
        except pc.ChainStop as stop:
            return self.bad("failed", str(stop), EXIT_STOP)
        except KeyboardInterrupt:
            self.status.update(state="interrupted", reason="signal; the service is left as it is; rerun with --resume")
            self.write_status()
            pc.log("CHAIN-BAD interrupted by a signal; service left running; rerun with --resume")
            return EXIT_INTERRUPTED
        except Exception as error:
            return self.bad("failed", f"{type(error).__name__}: {error}", EXIT_UNEXPECTED)
        finally:
            self._hb_stop.set()
            try:
                self.ops.stop_sampler()
            except Exception:
                pass

    def bad(self, state: str, reason: str, code: int) -> int:
        self.status.update(state=state, reason=reason)
        self.write_status()
        pc.log(f"CHAIN-BAD stopped at {self.status['current_step']}: {reason}")
        return code

    def finish_manifest(self) -> None:
        doc = pc.read_json(self.paths["manifest"]) or {}
        doc.update(finished=pc.utc_now(), npu_smi_after=self.ops.cards_used(),
                   decisions=self.status["decisions"])
        pc.write_atomic(self.paths["manifest"], json.dumps(doc, indent=2, sort_keys=True) + "\n")


def passed(rows: list[dict]) -> list[int]:
    return [r["concurrency_per_instance"] for r in rows if r["passes"]]


def file_sha(path: str) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------- dry-run operations (CPU only, nothing is executed)
class FakeRecordOps(pc.FakeOps):
    """Scripted operations: every would-be command is built by the SAME builders as the real run and printed."""

    def __init__(self, profile: dict, *, cards_busy: bool = False, verify_fail: bool = False, smoke_fail: bool = False,
                 dead_after: dict | None = None, **kw) -> None:
        super().__init__(**kw)
        self.profile, self.cards_busy, self.verify_fail, self.smoke_fail = profile, cards_busy, verify_fail, smoke_fail
        self.dead_after = dict(dead_after or {})  # step id -> instance name that is dead when the level is checked
        self.commands: list[list[str]] = []
        self.dead_now: list[str] = []

    def cmd(self, argv: list[str]) -> None:
        import shlex
        self.commands.append(argv)
        self.note("RUN " + shlex.join(argv))

    def hostname(self) -> str: return "A800I-03"
    def git_head(self) -> str: return "0" * 40
    def disk_free_gib(self, path) -> float: return 500.0

    def cards_used(self) -> dict:
        self.note("RUN npu-smi info")
        return {c: (9000 if self.cards_busy else 3440) for c in range(8)}

    def deploy_up(self, profile_id, wait_seconds):
        self.cmd(deploy_up_cmd(profile_id))
        return 0, "ready: every instance answers /health and /v1/models (dry run)\n"

    def deploy_status(self, profile_id):
        self.cmd(deploy_status_cmd(profile_id))
        return "\n".join(f"{i['name']} (dry run)" for i in self.profile["instances"])

    def health(self, instances):
        self.note(f"GET /health and /v1/models on {[i['port'] for i in instances]}")
        return {i["name"]: True for i in instances}

    def dead_instances(self, instances):
        return list(self.dead_now)

    def startup_lines(self, name):
        self.cmd(["docker", "logs", name])
        return ["Available KV cache memory: 25.83 GiB (dry run)"]

    def server_seed(self, name): return "0"

    def write_receipt(self, names, out):
        self.cmd(receipt_cmd({n: 1000 + k for k, n in enumerate(names)}, "sha256:" + "0" * 64, out, "0:0"))
        return 0, "dry run\n"

    def verify_base(self, profile_id, lock, receipt, out):
        self.cmd(["env", f"OUT={out}"] + verify_cmd(profile_id, lock, receipt))
        checks = ["image", "versions", "cann", "weights", "endpoints", "protocol", "deployment", "env", "kv",
                  "template", "receipt"]
        failed = ["protocol"] if self.verify_fail else []
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"selected_checks": checks, "failed_checks": failed,
                                         "verdict": "fail" if failed else "pass", "skipped_checks": {}}))
        return (1 if failed else 0), "dry run\n"

    def smoke(self, url, audio, out):
        self.cmd(smoke_cmd(url, audio, out))
        return (1 if self.smoke_fail else 0), "dry run\n"

    def ensure_probe_bodies(self, path):
        self.note(f"greedy probe bodies cached at {path}")
        return "0" * 64

    def probe(self, urls, outs, bodies):
        for url, out in zip(urls, outs):
            self.cmd(probe_cmd(url, out, bodies))
        return [0] * len(urls)

    def compare(self, a, b, out):
        self.cmd(compare_cmd(a, b, out))
        return {"requests": 16, "byte_identical": 16, "mismatched_indices": [], "max_abs_logprob_diff": 0.0}

    def measure_level(self, step, prefix, level, urls, min_tokens, label, names=None, *, requests=None,
                      seed=pc.SEED, cpuset=pc.CLIENT_CPUSET, sweep=None):
        self.cmd(sweep_cmd(f"saea-rec-{Path(prefix).name}-L{level}", urls, level, requests, min_tokens, seed, label,
                           f"{prefix}-L{level}.json", cpuset, sweep))
        if step in self.dead_after:
            self.dead_now = [self.dead_after[step]]
        names = names or [f"instance-{i}" for i in range(len(urls))]
        if step in self.die:
            self.dead_now = [names[0]]
            raise pc.EndpointDied(names[0], {"dead_instances": [names[0]], "scripted": True})
        measure = super().measure_level(step, prefix, level, urls, min_tokens, label, names)
        measure["summary"]["requests"] = requests
        measure["summary"]["ok"] = requests
        measure["client_seed"], measure["client_decoding_seed"] = seed, 20260913
        return measure


def example_hint() -> str:
    return "pass --allow-example to run an EXAMPLE list for real (never do this for the recorded runs)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-list", help="run-list JSON (saea-record-run-list-v1)")
    ap.add_argument("--chain-dir", help="existing chain directory (with --resume)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ts", help="UTC stamp used in every report name (default: now)")
    ap.add_argument("--allow-example", action="store_true", help="run a list marked example=true for real")
    ap.add_argument("--dry-run", action="store_true", help="CPU only: print every command, run nothing")
    ap.add_argument("--out-root", help="dry-run only: output root (default: a temporary directory)")
    ap.add_argument("--fail", default="", help="dry-run only: step ids whose p10 fails, e.g. R-A2-L32")
    ap.add_argument("--die", default="", help="dry-run only: step ids whose level ends with a dead endpoint")
    ap.add_argument("--preempt", default="", help="dry-run only: step ids where instance 0 preempts")
    ap.add_argument("--ceiling-bound", default="", help="dry-run only: levels whose waiting gauge is above 0")
    ap.add_argument("--cards-busy", action="store_true", help="dry-run only: every card reads busy")
    ap.add_argument("--verify-fail", action="store_true", help="dry-run only: verification reports a failed check")
    ap.add_argument("--smoke-fail", action="store_true", help="dry-run only: the smoke exits non-zero")
    args = ap.parse_args()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    split = lambda s: {x for x in s.split(",") if x}
    if args.resume:
        chain_dir = Path(args.chain_dir or sys.exit("--resume needs --chain-dir"))
        config = pc.read_json(chain_dir / "config.json") or sys.exit(f"{chain_dir} holds no chain")
        ts, doc = config["ts"], pc.read_json(chain_dir / "run-list.json")
    else:
        if not args.run_list:
            sys.exit("--run-list is required")
        doc = json.loads(Path(args.run_list).read_text(encoding="utf-8"))
        ts = args.ts or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        chain_dir = None
    try:
        profile = load_profile(doc["profile"])
        run_list = validate_run_list(doc, profile)
    except pc.ChainStop as stop:
        sys.exit(f"refused: {stop}")
    if run_list["example"] and not args.dry_run and not args.allow_example:
        sys.exit(f"refused: the run list is marked EXAMPLE; {example_hint()}")
    if args.dry_run:
        root = Path(args.out_root or tempfile.mkdtemp(prefix="record-runs-dry-"))
        logs = str(root)
        chain_dir = chain_dir or root / f"record-runs-{ts}"
        chain_dir.mkdir(parents=True, exist_ok=True)
        ops = FakeRecordOps(profile, cards_busy=args.cards_busy, verify_fail=args.verify_fail,
                            smoke_fail=args.smoke_fail, fail=split(args.fail), die=split(args.die),
                            preempt=split(args.preempt), ceiling_bound_levels={int(x) for x in split(args.ceiling_bound)})
        pc.log(f"dry-run output under {root}; EXAMPLE={run_list['example']}")
        chain = RecordChain(ops, chain_dir, ts, logs, run_list, profile, dry=True, sleep=lambda _s: None)
        code = chain.run()
        print(json.dumps(pc.read_json(chain_dir / "chain-status.json")["decisions"], indent=1, sort_keys=True))
        return code
    ops = RecordOps(LOGS, ts, args.resume)
    if ops.hostname() != pc.EXPECTED_HOSTNAME:
        sys.exit(f"refusing to run on {ops.hostname()!r}: recorded runs belong to host 181 ({pc.EXPECTED_HOSTNAME})")
    if not args.resume:
        chain_dir = Path(args.chain_dir or f"{LOGS}/record-runs-{ts}")
        if (chain_dir / "config.json").exists():
            sys.exit(f"{chain_dir} already holds a chain; use --resume")
        chain_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(chain_dir, 0o777)
        pc.write_atomic(chain_dir / "run-list.json", json.dumps(doc, indent=2, sort_keys=True) + "\n")
        pc.write_atomic(chain_dir / "config.json", json.dumps(
            {"ts": ts, "mode": "record", "created": pc.utc_now(), "script": str(Path(__file__).resolve())}) + "\n")
    (chain_dir / "steps").mkdir(exist_ok=True)
    os.chmod(chain_dir / "steps", 0o777)
    pc.log(f"chain {chain_dir.name} pid {os.getpid()} resume={args.resume} dir {chain_dir}")
    return RecordChain(ops, chain_dir, ts, LOGS, run_list, profile).run()


if __name__ == "__main__":
    sys.exit(main())
