#!/usr/bin/env python3
"""PRELIMINARY card-run chain P1-P4 on host 181 (plan section 3.7): one detached, checkpointed, resumable run.

  P1  arm A (4 x TP2, max-num-seqs 64): levels per instance 56, then 64 (64 is skipped when 56 fails).
  P2  worst case (--min-tokens 4096), descent 48 -> 40 -> 32 (plan 3.8): on failure repeat one level down until one is
      admissible or 32 fails.
  P3  stop our saea-arm-a-* containers, launch arm B (2 x TP4, MAX_NUM_SEQS=128), levels per instance 16, 32, 48, 64,
      96 with the same stopping rule.
  P4  stop saea-arm-b-*, relaunch arm A with the argv captured from the running containers before P3.

Decisions are taken by this script from the plan 3.4 thresholds (per level: per-sequence e2e p10 >= 10 tok/s, max
latency < 420 s, zero preemptions with a readable counter, >= 2 GiB HBM free per card, zero errors and timeouts).
Worst-case levels (--min-tokens 4096) must also be ADMISSIBLE (plan 3.8): preemption delta exactly 0 on EVERY instance
with a readable counter AND a sampled KV peak <= 90 % on every instance. Per-instance deltas and KV peaks are recorded
in every step decision. During every level a liveness poll (every 30 s: /v1/models per instance plus docker ps for the
container) runs; a dead endpoint or container fails the level with reason endpoint_died:<name> and STOPS the chain
(CHAIN-BAD). The chain never steps down against dead endpoints and never restarts a container on its own.
Preliminary only: synthetic load, no dataset row, nothing here is the recorded baseline. The sweep tool's output_tok_s
is the HOST aggregate (sum over all instances / wall time); every summary here calls it host_output_tok_s.

usage (on the NPU host 181, detached):
  nohup setsid python3 prelim_chain.py [--ts <UTC stamp>] < /dev/null > <chain-dir>/chain.log 2>&1 &
  resume: python3 prelim_chain.py --resume --chain-dir <chain-dir>   (finished steps are skipped)
  rehearsal (CPU only, no docker, no card, no network): python3 prelim_chain.py --dry-run [--fail p1-L64,p2-L40]
  P2b (plan 3.8 decision 3 item 1): python3 prelim_chain.py --p2b [--arm-a-launch <arm-a-launch.json>]
      stops idle saea-arm-a-* survivors, checks the cards, launches a fresh 4-instance arm A from the captured argv via
      npu_serve.sh, runs the worst case at 40 per instance (one pass); if not admissible, repeats on a fresh arm A at 32;
      arm A is left UP. Resume with the same --resume --chain-dir command (the mode is stored in config.json).
  P3 + P3b + P4 (plan 3.8 decision 3 item 3): python3 prelim_chain.py --p3 [--gate-file <chain-status.json>]
      waits (reads the gate file at most every 120 s) until the gate chain is FINAL (state done/failed/interrupted AND its
      pid gone), removes that chain's watchdog entries, stops idle saea-arm-a-*, requires all 8 cards < 4096 MiB, launches
      arm B (2 x TP4, MAX_NUM_SEQS=128) and measures per-instance levels 16, 32, 48, 64, 96 (one pass, seed 20261028, 3.4
      stopping rule); P3b then runs B's worst case (--min-tokens 4096, one pass) at the highest passing level and steps
      down through the measured passing levels until one is admissible (zero preemptions on EVERY instance, KV peak <= 90 %);
      a dead endpoint is a FAIL and STOPS the chain; finally B is stopped and arm A relaunched from the captured argv (P4)
      and waited healthy. Resume with --resume --chain-dir (the mode and the gate file are stored in config.json).
Layout: <logs>/prelim-chain-<ts>/{config.json,chain-status.json,arm-a-launch.json,steps/<step>.done.json,chain.log};
reports <logs>/sweep-arm{A,B}-prelim-P{1,2,2b,3,3b}-<ts>[-mintok4096]-L<level>.json (+ .row.json, summary, -npu-smi.json).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = "/mnt/cephfs/xingchao/omni"
LOGS = f"{ROOT}/work/logs"
SCRIPTS = Path(__file__).resolve().parent
NPU_SERVE = str(SCRIPTS / "npu_serve.sh")
JOBS = f"{ROOT}/work/watchdog/jobs.json"
TREE = f"{ROOT}/work/dev/umbrella/studies/speech-aware-evidence-acquisition"
IMAGE = "saea-base:20261027"
SEED = "20261028"
MODEL = "qwen3-omni-30b-a3b-thinking"
EXPECTED_HOSTNAME = "A800I-03"  # host 181 (alias npu2); the chain refuses to run anywhere else
CLIENT_CPUSET = "224-255"  # NUMA node 7: no instance is pinned there
SAMPLE_SECONDS = 10
POLL_SECONDS = 2.0
MIN_FREE_MIB = 2048
MAX_LATENCY_S = 420.0
MIN_P10 = 10.0
BUSY_MIB = 4096  # npu_serve.sh refuses cards above this; the chain waits for it, it never raises it
B_HEALTH_TIMEOUT_S = 20 * 60
A_HEALTH_TIMEOUT_S = 40 * 60  # four instances compile in parallel (DEPLOY_WAIT_SECONDS 2400 in the deploy layer)
CARDS_FREE_TIMEOUT_S = 300
P1_LEVELS = (56, 64)
P2_LADDER = (48, 40, 32)  # plan 3.8: the allowed worst-case levels, descent order
P2B_LADDER = (40, 32)  # 48 already failed the worst case (P2-L48); P2b starts one level down
P2_MIN_TOKENS = 4096
PRIOR_PASS = {48}  # measured and passed in the earlier preliminary scan (sweep-armA-prelim-20261029T162735)
KV_PEAK_MAX = 0.90  # plan 3.8 rule (a): sampled KV peak of every instance (fraction of the cache) <= 90 %
LIVENESS_SECONDS = 30  # plan 3.8: per-instance liveness poll during every level
LIVENESS_PROBE_TRIES = 3  # /v1/models attempts (5 s apart, 10 s timeout) before an endpoint is called dead
LIVENESS_PAUSE_SECONDS = 5.0
P3_LEVELS = (16, 32, 48, 64, 96)
P3B_MIN_TOKENS = P2_MIN_TOKENS  # B's worst case: every completion forced to 4,096 tokens
GATE_POLL_S = 120  # the gate chain's status file is read at most once every 2 minutes
GATE_FINAL = ("done", "failed", "interrupted")
DEFAULT_GATE = f"{LOGS}/prelim-E-chain-20261029T111900Z/chain-status.json"
E_ENTRY = re.compile(r"^(prelim-E-chain|vllm-prelim-E\d.*|sweep-prelim-E\d.*)$")  # watchdog ids of the E crash trials
OURS = re.compile(r"^saea-arm-[ab]-\d+$")
ARM_B = [
    {"name": "saea-arm-b-0", "cards": "0,1,2,3", "port": 8210, "tp": 4, "cpuset": "128-159,192-223",
     "hccl_base_port": 60400, "max_num_seqs": 128},  # cards 0-3 sit on NUMA nodes 6 and 4
    {"name": "saea-arm-b-1", "cards": "4,5,6,7", "port": 8211, "tp": 4, "cpuset": "0-31,64-95",
     "hccl_base_port": 60500, "max_num_seqs": 128},  # cards 4-7 sit on NUMA nodes 0 and 2
]

ROW1 = re.compile(r"^\|\s*(\d+)\s+\S+\s+\|\s*OK\s*\|")
ROW2 = re.compile(r"^\|\s*\d+\s+\|\s*[0-9A-Fa-f:.]+\s+\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|")


class ChainStop(Exception):
    """A condition that stops the chain with a clear status (not a failed threshold; those are decisions)."""


class EndpointDied(ChainStop):
    """An instance endpoint or container died during a level: the level FAILS (endpoint_died:<name>) and the chain stops."""

    def __init__(self, name: str, detail: dict | None = None) -> None:
        super().__init__(f"endpoint_died:{name}")
        self.name, self.detail = name, detail or {}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"{utc_now()} {message}", flush=True)


def write_atomic(path: Path | str, text: str, mode: int = 0o666) -> None:
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def read_json(path: Path | str):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- pure decision logic (unit tested)
def per_instance_stats(summary: dict, names: list[str], poll_kv: list | None = None) -> list[dict]:
    """Per instance: preemption delta (None when either reading is missing) and KV peak (fraction; the larger of the
    sweep tool's 5 s gauge maximum and the chain's own 2 s poll). Order follows the --base-url order."""
    before, after, gauges = summary.get("metrics_before") or [], summary.get("metrics_after") or [], \
        summary.get("gauges") or []
    out = []
    for i, name in enumerate(names):
        b = before[i] if i < len(before) and isinstance(before[i], dict) else None
        a = after[i] if i < len(after) and isinstance(after[i], dict) else None
        pre = None
        if a and b and a.get("preemptions") is not None and b.get("preemptions") is not None:
            pre = a["preemptions"] - b["preemptions"]
        g = gauges[i] if i < len(gauges) and isinstance(gauges[i], dict) else {}
        seen = [v for v in (((g.get("kv_cache_usage_perc") or {}).get("max")),
                            poll_kv[i] if poll_kv and i < len(poll_kv) else None) if v is not None]
        out.append({"name": name, "preemptions_delta": pre, "kv_peak": max(seen) if seen else None})
    return out


def admissibility(per_instance: list[dict]) -> list[str]:
    """Plan 3.8 rule (a) for a worst-case level: preemption delta exactly 0 on EVERY instance (readable) and the KV
    peak <= KV_PEAK_MAX on every instance. Returns the failing reasons (empty = admissible)."""
    reasons = []
    for entry in per_instance:
        name, pre, kv = entry["name"], entry["preemptions_delta"], entry["kv_peak"]
        if pre is None:
            reasons.append(f"admissibility: preemption counter unreadable on {name}")
        elif pre != 0:
            reasons.append(f"admissibility: {pre:g} preemption(s) on {name}")
        if kv is None:
            reasons.append(f"admissibility: KV peak unreadable on {name}")
        elif kv > KV_PEAK_MAX:
            reasons.append(f"admissibility: KV peak {kv * 100:.1f} % > {KV_PEAK_MAX * 100:.0f} % on {name}")
    return reasons


def evaluate(measure: dict, level: int, max_num_seqs: int, worst_case: bool = False,
             max_latency_s: float | None = None) -> tuple[list[str], dict]:
    """Plan 3.4 thresholds on one measured level; with worst_case also the plan 3.8 admissibility rule. Returns
    (failing reasons, row). A waiting queue at the scheduler ceiling is not a failure; it marks the level ceiling_bound."""
    summary = measure["summary"]
    lat, e2e = summary["latency_s"], summary["e2e_tok_s"]
    limit = MAX_LATENCY_S if max_latency_s is None else float(max_latency_s)  # the 420 s rule scales with the output cap
    reasons: list[str] = []
    if summary["errors"] or summary["timeouts"]:
        reasons.append(f"{summary['errors']} errors, {summary['timeouts']} timeouts")
    if summary["ok"] != summary["requests"]:
        reasons.append(f"ok {summary['ok']} of {summary['requests']} requests")
    if lat["max"] is None or lat["max"] >= limit:
        reasons.append(f"max latency {lat['max']} >= {limit}")
    if e2e["p10"] is None or e2e["p10"] < MIN_P10:
        reasons.append(f"p10 e2e {e2e['p10']} < {MIN_P10}")
    pre = summary["preemptions_delta"]
    if pre is None:
        reasons.append("preemption counter unreadable")
    elif pre > 0:
        reasons.append(f"{pre:g} preemption(s)")
    free = measure["min_hbm_free_mib"]
    if free is None or free < MIN_FREE_MIB:
        reasons.append(f"min HBM free {free} MiB < {MIN_FREE_MIB}")
    count = summary.get("instances") or measure.get("instances") or len(summary.get("gauges") or []) or 1
    per_instance = per_instance_stats(summary, measure.get("names") or [f"instance-{i}" for i in range(count)],
                                      measure.get("poll_kv_peak"))
    if worst_case:
        reasons.extend(admissibility(per_instance))
    gauges = [g for g in (summary.get("gauges") or []) if isinstance(g, dict)]

    def peak(key: str):
        vals = [g[key]["max"] for g in gauges if g.get(key)]
        return max(vals) if vals else None

    waiting_candidates = [v for v in (peak("num_requests_waiting"), measure.get("poll_waiting_max")) if v is not None]
    waiting = max(waiting_candidates) if waiting_candidates else None
    ceiling_bound = bool(waiting and waiting > 0 and level >= max_num_seqs)
    row = {
        "concurrency_per_instance": level, "instances": summary.get("instances", measure.get("instances")),
        "max_num_seqs": max_num_seqs, "requests": summary["requests"], "ok": summary["ok"],
        "errors": summary["errors"], "timeouts": summary["timeouts"], "wall_s": summary["wall_s"],
        "host_output_tok_s": summary["output_tok_s"],  # host aggregate: sum over all instances / wall time
        "completion_tokens": summary["completion_tokens"], "e2e_tok_s": e2e, "decode_tok_s": summary["decode_tok_s"],
        "latency_s": lat, "ttft_s": summary["ttft_s"], "preemptions_delta": pre,
        "prefix_cache_queries_delta": (summary.get("metrics_delta") or {}).get("prefix_cache_queries"),
        "prefix_cache_hits_delta": (summary.get("metrics_delta") or {}).get("prefix_cache_hits"),
        "kv_cache_usage_peak": peak("kv_cache_usage_perc"), "num_requests_running_peak": peak("num_requests_running"),
        "num_requests_waiting_peak": waiting, "ceiling_bound": ceiling_bound,
        "per_instance": per_instance, "worst_case": worst_case,
        "worst_case_admissible": (not admissibility(per_instance)) if worst_case else None,
        "min_hbm_free_mib_per_card": free, "aicore_pct_mean_during_load": measure.get("aicore_mean"),
        "aicore_pct_max_during_load": measure.get("aicore_max"), "samples_in_load_window": measure.get("samples"),
        "report": measure["report"], "report_sha256": measure["report_sha256"],
        "passes": not reasons, "failing_reasons": reasons,
    }
    return reasons, row


def p2_ladder(start: int) -> list[int]:
    """Levels to try for the worst case: the starting level, then one level down at a time (48, 40, 32)."""
    return [level for level in P2_LADDER if level <= start]


def p2_start(passed_p1: list[int]) -> int:
    """The worst case starts at the highest passing P1 level, capped at 48: plan 3.8 allows only 48, 40, 32."""
    return min(max(set(passed_p1) | PRIOR_PASS), P2_LADDER[0])


def gate_final(status: dict | None, pid_alive: bool) -> tuple[bool, str]:
    """The gate chain is FINAL when its state is done/failed/interrupted AND its process has exited."""
    if not isinstance(status, dict):
        return False, "chain-status.json unreadable"
    state = status.get("state")
    if state not in GATE_FINAL:
        return False, f"state {state!r} step {status.get('current_step')!r}"
    if pid_alive:
        return False, f"state {state!r} but pid {status.get('pid')} is still alive"
    return True, f"state {state!r}"


def p3b_ladder(passed: list[int]) -> list[int]:
    """P3b worst-case levels: the highest passing P3 level first, then the lower measured (passing) levels, descending."""
    return sorted({int(level) for level in passed}, reverse=True)


class Liveness:
    """Per-instance liveness (plan 3.8): the container is in `docker ps` AND /v1/models answers 200 (up to
    LIVENESS_PROBE_TRIES attempts). `running` returns the set of running container names, or None when docker itself
    could not be read (then only the endpoint is judged). All I/O is injected, so it is unit tested without a host."""

    def __init__(self, instances: list[tuple[str, str]], get, running, tries: int = LIVENESS_PROBE_TRIES,
                 pause: float = 5.0, sleep=time.sleep) -> None:
        self.instances, self.get, self.running = list(instances), get, running
        self.tries, self.pause, self.sleep = tries, pause, sleep

    def dead_instances(self) -> list[str]:
        up = self.running()
        dead = []
        for name, url in self.instances:
            if up is not None and name not in up:
                dead.append(name)
                continue
            for attempt in range(self.tries):
                got = self.get(url + "/v1/models", timeout=10)
                if got and got[0] == 200:
                    break
                if attempt < self.tries - 1:
                    self.sleep(self.pause)
            else:
                dead.append(name)
        return dead


# ---------------------------------------------------------------- real operations (host 181 only)
def read_cards() -> dict[int, tuple[int, int, int]]:
    text = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=60).stdout
    cards: dict[int, tuple[int, int, int]] = {}
    current = None
    for line in text.splitlines():
        m1 = ROW1.match(line)
        if m1:
            current = int(m1.group(1))
            continue
        m2 = ROW2.match(line)
        if m2 and current is not None:
            cards[current] = (int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
            current = None
    return cards


def http_get(url: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read(4_000_000).decode("utf-8", "replace")
    except Exception:
        return None


def prom_sum(text: str, metric: str) -> float | None:
    total, seen = 0.0, False
    for line in text.splitlines():
        if line.startswith(metric) and line[len(metric):len(metric) + 1] in ("{", " "):
            try:
                total += float(line.rpartition(" ")[2])
                seen = True
            except ValueError:
                pass
    return total if seen else None


class RealOps:
    """Docker, npu-smi, the sweep client and the watchdog registry. Every container name is checked against OURS."""

    def __init__(self, logs: str, ts: str, resume: bool) -> None:
        self.logs, self.ts = logs, ts
        self.samples: list[dict] = []
        self.lock = threading.Lock()
        self.phase: str | None = None  # report prefix the samples belong to
        self.level: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.resume = resume

    # --- sampler (npu-smi every SAMPLE_SECONDS, flushed per phase file about every minute)
    def start_sampler(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        log(f"npu-smi sampler started ({SAMPLE_SECONDS} s)")

    def stop_sampler(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=90)
            log("npu-smi sampler stopped")
        if self.phase:
            self.flush(self.phase)

    def _sample_loop(self) -> None:
        last_flush = time.time()
        while not self._stop.is_set():
            try:
                cards = read_cards()
                sample = {"t": time.time(), "level": self.level, "prefix": self.phase,
                          "aicore_pct": {str(k): v[0] for k, v in cards.items()},
                          "hbm_used_mib": {str(k): v[1] for k, v in cards.items()},
                          "hbm_total_mib": next(iter(cards.values()))[2] if cards else None}
            except Exception as error:  # a failed sample must not kill the chain
                sample = {"t": time.time(), "level": self.level, "prefix": self.phase, "error": repr(error)[:200]}
            with self.lock:
                self.samples.append(sample)
            if time.time() - last_flush > 60 and self.phase:
                self.flush(self.phase)
                last_flush = time.time()
            self._stop.wait(SAMPLE_SECONDS)

    def set_phase(self, prefix: str | None, level: int | None = None) -> None:
        if self.phase and prefix != self.phase:
            self.flush(self.phase)
        if prefix and prefix != self.phase and self.resume:
            old = read_json(f"{prefix}-npu-smi.json")
            if old:
                with self.lock:
                    self.samples = [dict(s, prefix=prefix) for s in old["samples"]] + \
                        [s for s in self.samples if s.get("prefix") != prefix]
        self.phase, self.level = prefix, level

    def flush(self, prefix: str) -> None:
        with self.lock:
            mine = [{k: v for k, v in s.items() if k != "prefix"} for s in self.samples if s.get("prefix") == prefix]
        write_atomic(f"{prefix}-npu-smi.json", json.dumps({"interval_s": SAMPLE_SECONDS, "samples": mine},
                                                         separators=(",", ":")) + "\n")

    # --- one measured level
    def docker_names(self) -> set[str] | None:
        """Names of running containers, or None when `docker ps` itself failed (unknown, not 'all dead')."""
        done = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
        return set(done.stdout.split()) if done.returncode == 0 else None

    def measure_level(self, step: str, prefix: str, level: int, urls: list[str], min_tokens: int | None,
                      label: str, names: list[str] | None = None) -> dict:
        names = names or [f"instance-{i}" for i in range(len(urls))]
        out = Path(f"{prefix}-L{level}.json")
        name = f"saea-sweep-{Path(prefix).name}-L{level}"
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)  # a client left by a dead chain
        command = ["docker", "run", "--rm", "--name", name, "--network", "host", "--user",
                   f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "--cpuset-cpus", CLIENT_CPUSET,
                   "-v", f"{ROOT}:{ROOT}", "-w", TREE, "-e", f"PYTHONPATH={TREE}/src",
                   "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "/opt/saea-venv/bin/python", IMAGE,
                   "scripts/serving/synthetic_sweep.py"]
        for url in urls:
            command += ["--base-url", url]
        command += ["--model", MODEL, "--levels", str(level), "--seed", SEED, "--label", label, "--out", str(out)]
        if min_tokens is not None:
            command += ["--min-tokens", str(min_tokens)]
        self.set_phase(prefix, level)
        poll = {"waiting": 0.0, "running": 0.0, "stop": threading.Event(), "kv": [None] * len(urls),
                "pre_first": [None] * len(urls), "pre_last": [None] * len(urls)}
        dead: dict = {"names": []}

        def poller() -> None:
            while not poll["stop"].is_set():
                for i, url in enumerate(urls):
                    got = http_get(url + "/metrics", timeout=3)
                    if got and got[0] == 200:
                        waiting = prom_sum(got[1], "vllm:num_requests_waiting")
                        running = prom_sum(got[1], "vllm:num_requests_running")
                        kv = prom_sum(got[1], "vllm:kv_cache_usage_perc")
                        pre = prom_sum(got[1], "vllm:num_preemptions_total")
                        if pre is None:
                            pre = prom_sum(got[1], "vllm:num_preemptions")
                        poll["waiting"] = max(poll["waiting"], waiting or 0.0)
                        poll["running"] = max(poll["running"], running or 0.0)
                        if kv is not None:
                            poll["kv"][i] = kv if poll["kv"][i] is None else max(poll["kv"][i], kv)
                        if pre is not None:
                            poll["pre_last"][i] = pre
                            if poll["pre_first"][i] is None:
                                poll["pre_first"][i] = pre
                poll["stop"].wait(POLL_SECONDS)

        live = Liveness(list(zip(names, urls)), http_get, self.docker_names, pause=LIVENESS_PAUSE_SECONDS)

        def watcher() -> None:  # plan 3.8: liveness every 30 s; a dead instance ends the level at once
            while not poll["stop"].wait(LIVENESS_SECONDS):
                gone = live.dead_instances()
                if gone:
                    dead["names"] = gone
                    dead["at"] = utc_now()
                    log(f"{step}: LIVENESS dead instance(s) {gone}; stopping the level client {name}")
                    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
                    return

        thread = threading.Thread(target=poller, daemon=True)
        thread.start()
        watch = threading.Thread(target=watcher, daemon=True)
        watch.start()
        t0 = time.time()
        log(f"{step}: client command {' '.join(command[command.index('scripts/serving/synthetic_sweep.py'):])}")
        try:
            with open(f"{prefix}-L{level}.stderr.log", "w") as stderr:
                code = subprocess.run(command, stdout=stderr, stderr=subprocess.STDOUT).returncode
        except BaseException:  # a signal or error: never leave a client container loading the servers
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            raise
        finally:
            poll["stop"].set()
        t1 = time.time()
        thread.join(timeout=10)
        watch.join(timeout=10)
        if not dead["names"]:  # one last look: an instance may have died after the final poll tick
            dead["names"] = live.dead_instances()
            dead["at"] = utc_now()
        if dead["names"]:
            raise EndpointDied(dead["names"][0], {
                "dead_instances": dead["names"], "detected_at": dead.get("at"), "client_exit_code": code,
                "instances": names, "kv_peak_polled": poll["kv"], "preemptions_first_polled": poll["pre_first"],
                "preemptions_last_polled": poll["pre_last"]})
        if code != 0 or not out.exists():
            raise ChainStop(f"{step}: sweep client exited {code}; see {prefix}-L{level}.stderr.log")
        report = json.loads(out.read_text())
        summary = report["levels"][0]
        generated = dt.datetime.fromisoformat(report["generated_at"]).timestamp()
        with self.lock:
            snapshot = [s for s in self.samples if "error" not in s]
        everything = [s for s in snapshot if t0 <= s["t"] <= t1]
        load = [s for s in snapshot if generated - summary["wall_s"] - 1 <= s["t"] <= generated + 2]
        free = [s["hbm_total_mib"] - used for s in everything for used in s["hbm_used_mib"].values()]
        ai = [v for s in load for v in s["aicore_pct"].values()]
        summary = dict(summary, instances=len(urls))
        return {"summary": summary, "report": out.name, "report_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "min_hbm_free_mib": min(free) if free else None,
                "aicore_mean": sum(ai) / len(ai) if ai else None, "aicore_max": max(ai) if ai else None,
                "samples": len(load), "poll_waiting_max": poll["waiting"], "poll_running_max": poll["running"],
                "poll_kv_peak": poll["kv"], "names": names, "instances": len(urls), "max_tokens_sent": (report.get("settings") or {}).get("max_tokens"),
                "min_tokens_sent": (report.get("settings") or {}).get("min_tokens")}

    # --- containers
    @staticmethod
    def _check_ours(names: list[str]) -> None:
        bad = [n for n in names if not OURS.match(n)]
        if bad:
            raise ChainStop(f"refusing to touch container(s) that are not ours: {bad}")

    def running_ours(self, prefix: str) -> list[str]:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout
        return sorted(n for n in out.split() if OURS.match(n) and n.startswith(prefix))

    def container_running(self, name: str) -> bool:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout
        return name in out.split()

    def running_requests(self, url: str) -> float | None:
        """vllm:num_requests_running of one endpoint, None when unreadable."""
        got = http_get(url + "/metrics", timeout=5)
        return prom_sum(got[1], "vllm:num_requests_running") if got and got[0] == 200 else None

    def stop_containers(self, names: list[str]) -> None:
        self._check_ours(names)
        if not names:
            return
        log(f"docker stop -t 120 {' '.join(names)}")
        subprocess.run(["docker", "stop", "-t", "120", *names], capture_output=True, text=True)

    def wait_cards_free(self, timeout: float = CARDS_FREE_TIMEOUT_S) -> dict:
        deadline = time.time() + timeout
        while True:
            cards = read_cards()
            used = {c: cards[c][1] for c in range(8) if c in cards}
            if len(used) == 8 and all(v < BUSY_MIB for v in used.values()):
                return used
            if time.time() > deadline:
                raise ChainStop(f"cards not below {BUSY_MIB} MiB after {timeout:.0f} s: {used}")
            time.sleep(5)

    def inspect_arm_a(self) -> list[dict]:
        names = self.running_ours("saea-arm-a-")
        if names != [f"saea-arm-a-{i}" for i in range(4)]:
            raise ChainStop(f"expected saea-arm-a-0..3 running, found {names}")
        out = []
        for name in names:
            info = json.loads(subprocess.run(["docker", "inspect", name], capture_output=True, text=True,
                                             check=True).stdout)[0]
            cmd = info["Config"]["Cmd"]
            env = dict(e.split("=", 1) for e in info["Config"]["Env"] if "=" in e)
            devices = [d["PathOnHost"] for d in info["HostConfig"]["Devices"]]
            cards = sorted(int(d.rsplit("davinci", 1)[1]) for d in devices if re.search(r"davinci\d+$", d))
            out.append({
                "name": name, "image": info["Config"]["Image"], "entrypoint": info["Config"]["Entrypoint"],
                "cmd": cmd, "cards": ",".join(map(str, cards)), "port": int(cmd[cmd.index("--port") + 1]),
                "tp": int(cmd[cmd.index("--tensor-parallel-size") + 1]),
                "max_num_seqs": int(cmd[cmd.index("--max-num-seqs") + 1]),
                "cpuset": info["HostConfig"]["CpusetCpus"], "hccl_base_port": int(env["HCCL_IF_BASE_PORT"]),
                "shm_bytes": info["HostConfig"]["ShmSize"], "restart": info["HostConfig"]["RestartPolicy"]["Name"],
                "devices": devices,
                "env_selected": {k: env.get(k) for k in ("HCCL_IF_BASE_PORT", "HCCL_SOCKET_IFNAME",
                                                         "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME")}})
        return out

    def launch(self, spec: dict) -> str:
        self._check_ours([spec["name"]])
        env = dict(os.environ, NAME=spec["name"], CPUSET=spec["cpuset"], HCCL_BASE_PORT=str(spec["hccl_base_port"]),
                   MAX_NUM_SEQS=str(spec["max_num_seqs"]))
        command = ["bash", NPU_SERVE, "thinking", spec["cards"], str(spec["port"]), str(spec["tp"])]
        log(f"launch NAME={spec['name']} CPUSET={spec['cpuset']} HCCL_BASE_PORT={spec['hccl_base_port']} "
            f"MAX_NUM_SEQS={spec['max_num_seqs']} {' '.join(command)}")
        done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
        if done.returncode != 0:
            raise ChainStop(f"npu_serve.sh failed for {spec['name']}: {(done.stderr or done.stdout).strip()[-300:]}")
        match = re.search(r"log (\S+)\s*$", done.stdout.strip())
        return match.group(1) if match else ""

    def argv_of(self, name: str) -> dict:
        info = json.loads(subprocess.run(["docker", "inspect", name], capture_output=True, text=True,
                                         check=True).stdout)[0]
        env = dict(e.split("=", 1) for e in info["Config"]["Env"] if "=" in e)
        return {"cmd": info["Config"]["Cmd"], "cpuset": info["HostConfig"]["CpusetCpus"],
                "hccl_base_port": int(env["HCCL_IF_BASE_PORT"]),
                "devices": [d["PathOnHost"] for d in info["HostConfig"]["Devices"]]}

    def wait_healthy(self, specs: list[dict], timeout: float, on_tick=None) -> dict:
        deadline = time.time() + timeout
        state = {s["name"]: False for s in specs}
        while True:
            for spec in specs:
                if state[spec["name"]]:
                    continue
                if not self.container_running(spec["name"]):
                    raise ChainStop(f"{spec['name']} exited during startup (see its log)")
                base = f"http://127.0.0.1:{spec['port']}"
                health, models = http_get(base + "/health"), http_get(base + "/v1/models")
                if health and health[0] == 200 and models and models[0] == 200 and MODEL in models[1]:
                    state[spec["name"]] = True
                    log(f"{spec['name']} healthy on {spec['port']}")
            if all(state.values()):
                return state
            if time.time() > deadline:
                raise ChainStop(f"endpoints not healthy within {timeout:.0f} s: "
                                f"{[n for n, ok in state.items() if not ok]}")
            if on_tick:
                on_tick()
            time.sleep(10)

    def gate_status(self, path: str):
        return read_json(path)

    def pid_alive(self, pid) -> bool:
        return isinstance(pid, int) and pid > 0 and Path(f"/proc/{pid}").exists()

    def job_ids(self) -> list[str]:
        jobs = read_json(JOBS)
        if isinstance(jobs, dict):
            jobs = jobs.get("jobs", [])
        return [j["id"] for j in jobs] if isinstance(jobs, list) else []

    # --- watchdog registry (atomic replace; the watchdog re-reads jobs.json every cycle)
    def patch_jobs(self, upsert: list[dict], remove: list[str] = (), set_log: dict | None = None) -> None:
        jobs = read_json(JOBS)
        if isinstance(jobs, dict):
            jobs = jobs.get("jobs", [])
        if not isinstance(jobs, list):
            log("CHAIN-WARN jobs.json unreadable; watchdog registry not patched")
            return
        for job in upsert:
            if job not in jobs:
                jobs = [j for j in jobs if j["id"] != job["id"]] + [job]
        jobs = [j for j in jobs if j["id"] not in remove]
        for job_id, path in (set_log or {}).items():
            for j in jobs:
                if j["id"] == job_id:
                    j["params"]["log"] = path
        write_atomic(JOBS, json.dumps(jobs, indent=1) + "\n")
        log(f"watchdog registry patched: upsert={[j['id'] for j in upsert]} remove={list(remove)} "
            f"set_log={list((set_log or {}))}")


# ---------------------------------------------------------------- the chain
class Chain:
    def __init__(self, ops, chain_dir: Path, ts: str, logs: str, mode: str = "chain", gate_path: str | None = None,
                 gate_poll: float = GATE_POLL_S, sleep=time.sleep) -> None:
        self.ops, self.dir, self.ts, self.logs, self.mode = ops, chain_dir, ts, logs, mode
        self.gate_path, self.gate_poll, self.sleep = gate_path, gate_poll, sleep
        self.steps_dir = chain_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.url_name: dict[str, str] = {}
        self.p = {  # report prefixes
            "P1": f"{logs}/sweep-armA-prelim-P1-{ts}", "P2": f"{logs}/sweep-armA-prelim-P2-mintok4096-{ts}",
            "P2b": f"{logs}/sweep-armA-prelim-P2b-mintok4096-{ts}", "P3": f"{logs}/sweep-armB-prelim-P3-{ts}",
            "P3b": f"{logs}/sweep-armB-prelim-P3b-mintok4096-{ts}"}
        self.status = {"chain": chain_dir.name, "mode": mode, "state": "running", "current_step": None,
                       "started": utc_now(), "updated": utc_now(), "pid": os.getpid(), "reason": None,
                       "decisions": {}, "steps": {}, "prefixes": self.p}
        old = read_json(chain_dir / "chain-status.json")
        if old:
            self.status["decisions"], self.status["steps"] = old.get("decisions", {}), old.get("steps", {})
            self.status["started"] = old.get("started", self.status["started"])
        self.rows: dict[str, list[dict]] = {k: [] for k in self.p}
        self._hb_stop = threading.Event()

    # --- checkpoints and status
    def done(self, step: str) -> dict | None:
        return read_json(self.steps_dir / f"{step}.done.json")

    def mark(self, step: str, decision: dict) -> None:
        write_atomic(self.steps_dir / f"{step}.done.json", json.dumps(
            {"step": step, "finished": utc_now(), "decision": decision}, indent=2, sort_keys=True) + "\n")
        self.status["steps"][step] = {"state": "done", "finished": utc_now(),
                                      "decision": {k: v for k, v in decision.items() if k != "row"}}
        self.write_status()

    def set_current(self, step: str | None) -> None:
        self.status["current_step"] = step
        self.status["current_since"] = utc_now()
        if step:
            log(f"STEP {step}")
        self.write_status()

    def write_status(self) -> None:
        self.status["updated"] = utc_now()
        write_atomic(self.dir / "chain-status.json", json.dumps(self.status, indent=1, sort_keys=True) + "\n")

    def heartbeat(self) -> None:
        def loop() -> None:
            while not self._hb_stop.wait(60):
                log(f"heartbeat step={self.status['current_step']} since={self.status.get('current_since')}")
                self.write_status()
        threading.Thread(target=loop, daemon=True).start()

    # --- per-prefix summary (same schema as the arm A driver, so the watchdog `sweep` kind reads it)
    def write_summary(self, phase: str, done: bool, stopped_at: int | None, extra: dict | None = None) -> None:
        arm = "B" if phase in ("P3", "P3b") else "A"
        shape = "2 x TP4, max-num-seqs 128" if arm == "B" else "4 x TP2, max-num-seqs 64"
        doc = {"label": f"PRELIMINARY arm {arm} engine measurement ({shape}), phase {phase}, not the recorded "
                        f"locked baseline", "synthetic": True, "dataset_rows": 0, "seed": SEED,
               "host_output_tok_s_definition": "sum of completion tokens over all instances / level wall time",
               "stopped_at_level": stopped_at, "done": done, "levels": self.rows[phase]}
        doc.update(extra or {})
        write_atomic(f"{self.p[phase]}.json", json.dumps(doc, indent=2, sort_keys=True) + "\n")

    # --- one level step
    def level_step(self, phase: str, step: str, level: int, urls: list[str], max_num_seqs: int,
                   min_tokens: int | None = None) -> dict:
        arm = "B" if phase in ("P3", "P3b") else "A"
        prior = self.done(step)
        if prior and "row" in prior["decision"]:
            row = prior["decision"]["row"]
            self.rows[phase].append(row)
            log(f"{step}: resumed, pass={row['passes']}")
            return row
        self.set_current(step)
        mode = f", forced min_tokens {min_tokens}" if min_tokens else ""
        label = f"PRELIMINARY arm {arm} engine measurement (phase {phase}{mode}), not the recorded locked baseline"
        try:
            measure = self.ops.measure_level(step, self.p[phase], level, urls, min_tokens, label,
                                             names=[self.url_name.get(u, u) for u in urls])
        except EndpointDied as died:
            self.record_died(phase, step, level, died)
            raise
        reasons, row = evaluate(measure, level, max_num_seqs, worst_case=min_tokens is not None)
        row.update({"step": step, "phase": phase, "arm": arm, "min_tokens": min_tokens,
                    "max_tokens_sent": measure.get("max_tokens_sent"),
                    "min_tokens_sent": measure.get("min_tokens_sent")})  # read back from the report's settings
        write_atomic(f"{self.p[phase]}-L{level}.row.json", json.dumps(row, indent=2, sort_keys=True) + "\n")
        self.rows[phase].append(row)
        self.ops.flush(self.p[phase])
        self.mark(step, {"level": level, "passes": row["passes"], "failing_reasons": reasons,
                         "ceiling_bound": row["ceiling_bound"], "host_output_tok_s": row["host_output_tok_s"],
                         "p10_e2e_tok_s": row["e2e_tok_s"]["p10"], "latency_max_s": row["latency_s"]["max"],
                         "preemptions_delta": row["preemptions_delta"], "per_instance": row["per_instance"],
                         "worst_case_admissible": row["worst_case_admissible"],
                         "min_hbm_free_mib_per_card": row["min_hbm_free_mib_per_card"], "row": row})
        log(f"{step}: pass={row['passes']} {reasons} host_output_tok_s={row['host_output_tok_s']:.1f} "
            f"ceiling_bound={row['ceiling_bound']}")
        return row

    def record_died(self, phase: str, step: str, level: int, died: EndpointDied) -> None:
        """A level whose endpoint or container died is a FAIL with reason endpoint_died:<name>; the chain then stops.
        The record is a .failed.json (not .done.json), so a --resume re-measures the level instead of skipping it."""
        record = {"step": step, "level": level, "finished": utc_now(), "passes": False,
                  "failing_reasons": [f"endpoint_died:{died.name}"], "endpoint_died": died.name, "detail": died.detail}
        write_atomic(self.steps_dir / f"{step}.failed.json", json.dumps(record, indent=2, sort_keys=True) + "\n")
        self.status["steps"][step] = {"state": "failed", "finished": utc_now(), "decision": {
            k: record[k] for k in ("level", "passes", "failing_reasons", "endpoint_died")}}
        for old in self.steps_dir.glob("p2b-fresh*.done.json"):  # the arm A of this attempt is dead: relaunch on resume
            if old.name.startswith(f"p2b-fresh{level}-"):
                old.rename(old.with_name(old.name.replace(".done.json", f".died-{self.ts}.json")))
        try:
            self.ops.flush(self.p[phase])
        except Exception:
            pass
        self.write_summary(phase, False, level, {"endpoint_died": died.name})
        log(f"{step}: FAIL endpoint_died:{died.name}")

    def skip(self, step: str, reason: str) -> None:
        if not self.done(step):
            self.mark(step, {"skipped": reason})
            log(f"{step}: skipped ({reason})")

    # --- phases
    def capture_arm_a(self) -> list[dict]:
        path = self.dir / "arm-a-launch.json"
        captured = read_json(path)
        if captured:
            log("arm A argv: reusing the capture from the first run")
            return captured["instances"]
        self.set_current("capture-arm-a")
        specs = self.ops.inspect_arm_a()
        write_atomic(path, json.dumps({"captured": utc_now(), "instances": specs}, indent=2) + "\n")
        self.mark("capture-arm-a", {"instances": [s["name"] for s in specs], "cards": [s["cards"] for s in specs],
                                    "ports": [s["port"] for s in specs]})
        return specs

    def urls(self, specs: list[dict]) -> list[str]:
        urls = [f"http://127.0.0.1:{s['port']}" for s in specs]
        self.url_name.update(zip(urls, (s["name"] for s in specs)))
        return urls

    def phase_p1(self, a_specs: list[dict]) -> list[int]:
        self.ops.set_phase(self.p["P1"])
        urls, mns = self.urls(a_specs), a_specs[0]["max_num_seqs"]
        passed, stopped = [], None
        for level in P1_LEVELS:
            step = f"p1-L{level}"
            if stopped is not None:
                self.skip(step, f"level {stopped} failed")
                continue
            row = self.level_step("P1", step, level, urls, mns)
            if row["passes"]:
                passed.append(level)
            else:
                stopped = level
        self.write_summary("P1", True, stopped)
        decision = {"passed_levels": passed, "stopped_at_level": stopped,
                    "ceiling_bound_levels": [r["concurrency_per_instance"] for r in self.rows["P1"] if r["ceiling_bound"]]}
        self.status["decisions"]["P1"] = decision
        self.write_status()
        return passed

    def phase_p2(self, a_specs: list[dict], passed_p1: list[int]) -> None:
        start = p2_start(passed_p1)
        ladder = p2_ladder(start)
        urls, mns = self.urls(a_specs), a_specs[0]["max_num_seqs"]
        log(f"P2: worst case --min-tokens {P2_MIN_TOKENS}; start {start}; ladder {ladder}")
        self.ops.set_phase(self.p["P2"])
        knee, attempts = None, []
        for level in ladder:
            row = self.level_step("P2", f"p2-L{level}", level, urls, mns, min_tokens=P2_MIN_TOKENS)
            attempts.append({"level": level, "passes": row["passes"], "failing_reasons": row["failing_reasons"]})
            if row["passes"]:
                knee = level
                break
        self.write_summary("P2", True, None if knee else ladder[-1],
                           {"min_tokens": P2_MIN_TOKENS, "worst_case_passing_level": knee})
        self.status["decisions"]["P2"] = {"start_level": start, "ladder": ladder, "worst_case_passing_level": knee,
                                          "attempts": attempts}
        self.write_status()

    # --- P2b: fresh arm A, worst case at 40 (then 32), arm A left up (plan 3.8 decision 3 item 1)
    def p2b_job_entries(self) -> list[dict]:
        prefix = self.p["P2b"]
        return [{"id": "prelim-chain-p2b", "kind": "generic_log", "host": "npu2", "stall_s": 1800,
                 "note": "P2b: fresh arm A + worst case at 40 (then 32); heartbeat every 60 s; BAD means the chain stopped "
                         "(endpoint_died or no admissible level; see chain-status.json)",
                 "params": {"path": str(self.dir / "chain.log"), "bad_regex": "CHAIN-BAD|Traceback",
                            "done_regex": "CHAIN DONE", "status_file": str(self.dir / "chain-status.json")}},
                {"id": "sweep-armA-prelim-P2b-mintok4096", "kind": "sweep", "host": "npu2", "stall_s": 3600,
                 "note": "P2b worst case --min-tokens 4096 on a fresh arm A; a failing level 40 is a decision, "
                         "the next level runs on a relaunched arm A",
                 "params": {"run_prefix": prefix}}]

    def a_job_entries(self, a_specs: list[dict], logs: dict, tag: str = "P2b") -> list[dict]:
        return [{"id": f"vllm-arm-a-{s['name'].rsplit('-', 1)[1]}", "kind": "vllm_server", "host": "npu2",
                 "note": f"arm A relaunched by prelim-chain-{self.ts} ({tag}); left UP afterwards",
                 "params": {"container": s["name"], "port": s["port"], "log": logs.get(s["name"])}} for s in a_specs]

    def wait_idle(self, alive: list[str], a_specs: list[dict]) -> None:
        """Never kill an instance that is serving someone (a finished level drains in seconds)."""
        urls = {s["name"]: f"http://127.0.0.1:{s['port']}" for s in a_specs}
        for name in alive:
            deadline = time.time() + 180
            while True:
                busy = self.ops.running_requests(urls[name]) if name in urls else None
                if not busy:
                    break
                if time.time() > deadline:
                    raise ChainStop(f"{name} still has {busy:g} running request(s) after 180 s; refusing to stop it")
                time.sleep(5)

    def fresh_arm_a(self, tag: str, a_specs: list[dict]) -> None:
        """Stop every saea-arm-a-* (only if idle), require all cards < BUSY_MIB, relaunch the captured argv, wait healthy."""
        if not self.done(f"{tag}-stop"):
            self.set_current(f"{tag}-stop")
            self.ops.set_phase(self.p["P2b"], None)
            alive = self.ops.running_ours("saea-arm-a-")
            self.wait_idle(alive, a_specs)
            # Registry entries of the instances we are about to stop are removed (no false BAD) and re-added healthy.
            self.ops.patch_jobs([], remove=[f"vllm-arm-a-{i}" for i in range(len(a_specs))])
            self.ops.stop_containers(alive)
            deadline = time.time() + 240
            while self.ops.running_ours("saea-arm-a-") and time.time() < deadline:
                time.sleep(3)
            if self.ops.running_ours("saea-arm-a-"):
                raise ChainStop("saea-arm-a-* containers still running 240 s after docker stop")
            used = self.ops.wait_cards_free()
            self.mark(f"{tag}-stop", {"stopped": alive, "hbm_used_mib_after": used})
        if not self.done(f"{tag}-launch"):
            self.set_current(f"{tag}-launch")
            self.ops.wait_cards_free()  # npu_serve.sh refuses above 4096 MiB; the chain never raises that
            logs = {}
            for spec in a_specs:
                if not self.ops.container_running(spec["name"]):
                    logs[spec["name"]] = self.ops.launch(spec)
            self.mark(f"{tag}-launch", {"launched": [s["name"] for s in a_specs], "logs": logs})
        if not self.done(f"{tag}-wait"):
            self.set_current(f"{tag}-wait")
            self.ops.wait_healthy(a_specs, A_HEALTH_TIMEOUT_S, on_tick=self.write_status)
            mismatch = []
            for spec in a_specs:
                now = self.ops.argv_of(spec["name"])
                if not (now["cmd"] == spec["cmd"] and now["cpuset"] == spec["cpuset"]
                        and now["hccl_base_port"] == spec["hccl_base_port"] and now["devices"] == spec["devices"]):
                    mismatch.append(spec["name"])
            if mismatch:  # a different shape would make the measurement meaningless: stop before any load
                raise ChainStop(f"arm A argv differs from the capture for {mismatch}")
            logs = (self.done(f"{tag}-launch") or {}).get("decision", {}).get("logs", {})
            self.ops.patch_jobs(self.a_job_entries([s for s in a_specs if logs.get(s["name"])], logs))
            self.mark(f"{tag}-wait", {"healthy": [s["name"] for s in a_specs], "argv_identical_to_capture": True})

    def phase_p2b(self, a_specs: list[dict]) -> int | None:
        prefix = self.p["P2b"]
        self.ops.patch_jobs(self.p2b_job_entries())
        self.ops.set_phase(prefix)
        self.ops.start_sampler()
        urls, mns = self.urls(a_specs), a_specs[0]["max_num_seqs"]
        ladder = p2_ladder(P2B_LADDER[0])
        log(f"P2b: worst case --min-tokens {P2_MIN_TOKENS}; ladder {ladder}; each level on a fresh arm A")
        knee, attempts = None, []
        for level in ladder:
            step = f"p2b-L{level}"
            prior = self.done(step)
            if not (prior and "row" in prior["decision"]):
                self.fresh_arm_a(f"p2b-fresh{level}", a_specs)
            row = self.level_step("P2b", step, level, urls, mns, min_tokens=P2_MIN_TOKENS)
            attempts.append({"level": level, "passes": row["passes"], "failing_reasons": row["failing_reasons"],
                             "per_instance": row["per_instance"]})
            if row["passes"]:
                knee = level
                break
        self.write_summary("P2b", True, None if knee else ladder[-1],
                           {"min_tokens": P2_MIN_TOKENS, "worst_case_admissible_level": knee})
        self.status["decisions"]["P2b"] = {"ladder": ladder, "worst_case_admissible_level": knee, "attempts": attempts,
                                           "arm_a_left_up": True}
        self.write_status()
        return knee

    def b_job_entries(self, logs: dict) -> list[dict]:
        return [{"id": f"vllm-arm-b-{i}", "kind": "vllm_server", "host": "npu2",
                 "note": "preliminary arm B (P3); removed again when the chain stops it in P4",
                 "params": {"container": s["name"], "port": s["port"], "log": logs.get(s["name"])}}
                for i, s in enumerate(ARM_B)]

    def phase_p3(self, a_specs: list[dict]) -> list[int]:
        phase, prefix = "P3", self.p["P3"]
        self.ops.set_phase(prefix)
        if not self.done("p3-stop-a"):
            self.set_current("p3-stop-a")
            if self.mode == "p3":
                # Only our idle instances are stopped; their registry entries are removed (no false BAD while B holds
                # the cards) and re-added healthy after P4.
                alive = self.ops.running_ours("saea-arm-a-")
                self.wait_idle(alive, a_specs)
                self.ops.patch_jobs([], remove=[f"vllm-arm-a-{i}" for i in range(len(a_specs))])
                self.ops.stop_containers(alive)
            else:
                # The arm A watchdog entries stay registered (they read as failed during B; their note says so).
                self.ops.stop_containers([s["name"] for s in a_specs])
            deadline = time.time() + 180
            while self.ops.running_ours("saea-arm-a-") and time.time() < deadline:
                time.sleep(3)
            if self.ops.running_ours("saea-arm-a-"):
                raise ChainStop("saea-arm-a-* containers still running 180 s after docker stop")
            used = self.ops.wait_cards_free()
            self.mark("p3-stop-a", {"stopped": [s["name"] for s in a_specs], "hbm_used_mib_after": used})
        if not self.done("p3-launch-b"):
            self.set_current("p3-launch-b")
            self.ops.wait_cards_free()  # npu_serve.sh refuses above 4096 MiB; the chain never raises that
            logs = {}
            for spec in ARM_B:
                if not self.ops.container_running(spec["name"]):
                    logs[spec["name"]] = self.ops.launch(spec)
            self.mark("p3-launch-b", {"launched": [s["name"] for s in ARM_B], "logs": logs})
        if not self.done("p3-wait-b"):
            self.set_current("p3-wait-b")
            self.ops.wait_healthy(ARM_B, B_HEALTH_TIMEOUT_S, on_tick=self.write_status)
            # Registered only once healthy: a server still compiling would read as down for several minutes.
            blogs = (self.done("p3-launch-b") or {}).get("decision", {}).get("logs", {})
            self.ops.patch_jobs(self.b_job_entries(blogs))
            self.mark("p3-wait-b", {"healthy": [s["name"] for s in ARM_B], "ports": [s["port"] for s in ARM_B]})
        urls, mns = self.urls(ARM_B), ARM_B[0]["max_num_seqs"]
        passed, stopped = [], None
        for level in P3_LEVELS:
            step = f"p3-L{level}"
            if stopped is not None:
                self.skip(step, f"level {stopped} failed")
                continue
            row = self.level_step(phase, step, level, urls, mns)
            if row["passes"]:
                passed.append(level)
            else:
                stopped = level
        self.write_summary(phase, True, stopped)
        self.status["decisions"]["P3"] = {
            "passed_levels": passed, "stopped_at_level": stopped,
            "host_output_tok_s_by_level": {r["concurrency_per_instance"]: r["host_output_tok_s"] for r in self.rows[phase]},
            "ceiling_bound_levels": [r["concurrency_per_instance"] for r in self.rows[phase] if r["ceiling_bound"]]}
        self.write_status()
        return passed

    # --- P3b: B's worst case at the highest passing P3 level, stepping down until admissible (plan 3.8 rule (a))
    def phase_p3b(self, passed: list[int]) -> int | None:
        phase = "P3b"
        ladder = p3b_ladder(passed)
        self.ops.set_phase(self.p[phase])
        urls, mns = self.urls(ARM_B), ARM_B[0]["max_num_seqs"]
        knee, attempts = None, []
        if not ladder:
            self.skip("p3b-L0", "no P3 level passed; no worst-case level to confirm")
            log("P3b: no passing P3 level; nothing to confirm")
        else:
            log(f"P3b: worst case --min-tokens {P3B_MIN_TOKENS} on arm B; ladder {ladder} (highest passing P3 level first)")
        for level in ladder:
            row = self.level_step(phase, f"p3b-L{level}", level, urls, mns, min_tokens=P3B_MIN_TOKENS)
            attempts.append({"level": level, "passes": row["passes"], "failing_reasons": row["failing_reasons"],
                             "per_instance": row["per_instance"]})
            if row["passes"]:
                knee = level
                break
        self.write_summary(phase, True, None if knee else (ladder[-1] if ladder else None),
                           {"min_tokens": P3B_MIN_TOKENS, "worst_case_admissible_level": knee})
        self.status["decisions"]["P3b"] = {
            "ladder": ladder, "worst_case_admissible_level": knee,
            "admitted_host_concurrency": knee * len(ARM_B) if knee else None, "attempts": attempts}
        self.write_status()
        return knee

    def phase_p4(self, a_specs: list[dict]) -> None:
        self.ops.set_phase(None)
        if not self.done("p4-stop-b"):
            self.set_current("p4-stop-b")
            self.ops.patch_jobs([], remove=[f"vllm-arm-b-{i}" for i in range(len(ARM_B))])
            self.ops.stop_containers([s["name"] for s in ARM_B])
            deadline = time.time() + 180
            while self.ops.running_ours("saea-arm-b-") and time.time() < deadline:
                time.sleep(3)
            if self.ops.running_ours("saea-arm-b-"):
                raise ChainStop("saea-arm-b-* containers still running 180 s after docker stop")
            used = self.ops.wait_cards_free()
            self.mark("p4-stop-b", {"stopped": [s["name"] for s in ARM_B], "hbm_used_mib_after": used})
        if not self.done("p4-launch-a"):
            self.set_current("p4-launch-a")
            self.ops.wait_cards_free()
            logs = {}
            for spec in a_specs:
                if not self.ops.container_running(spec["name"]):
                    logs[spec["name"]] = self.ops.launch(spec)
            self.mark("p4-launch-a", {"launched": [s["name"] for s in a_specs], "logs": logs})
        if not self.done("p4-wait-a"):
            self.set_current("p4-wait-a")
            self.ops.wait_healthy(a_specs, A_HEALTH_TIMEOUT_S, on_tick=self.write_status)
            mismatch = []
            for spec in a_specs:
                now = self.ops.argv_of(spec["name"])
                same = (now["cmd"] == spec["cmd"] and now["cpuset"] == spec["cpuset"]
                        and now["hccl_base_port"] == spec["hccl_base_port"] and now["devices"] == spec["devices"])
                if not same:
                    mismatch.append(spec["name"])
            logs = (self.done("p4-launch-a") or {}).get("decision", {}).get("logs", {})
            if self.mode == "p3":  # the entries were removed before B: add them back, healthy, with the new logs
                self.ops.patch_jobs(self.a_job_entries([s for s in a_specs if logs.get(s["name"])], logs, "P3 chain, P4"))
            else:
                self.ops.patch_jobs([], set_log={f"vllm-arm-a-{s['name'].rsplit('-', 1)[1]}": logs[s["name"]]
                                                 for s in a_specs if logs.get(s["name"])})
            if mismatch:
                log(f"CHAIN-BAD arm A argv differs from the capture for {mismatch}")
            self.mark("p4-wait-a", {"healthy": [s["name"] for s in a_specs], "argv_identical_to_capture": not mismatch,
                                    "mismatch": mismatch})
            self.status["decisions"]["P4"] = {"healthy": [s["name"] for s in a_specs],
                                              "argv_identical_to_capture": not mismatch}
            self.write_status()

    # --- P3 + P3b + P4 chain (plan 3.8 decision 3 items 3 and 4)
    def p3_job_entries(self) -> list[dict]:
        return [{"id": "prelim-chain-p3", "kind": "generic_log", "host": "npu2", "stall_s": 3600,
                 "note": "P3 arm B + P3b worst case + P4 arm A relaunch; waits for the E chain first; heartbeat every 60 s; "
                         "BAD means the chain stopped (endpoint_died or a failed step; see chain-status.json)",
                 "params": {"path": str(self.dir / "chain.log"), "bad_regex": "CHAIN-BAD|Traceback",
                            "done_regex": "CHAIN DONE", "status_file": str(self.dir / "chain-status.json")}},
                {"id": "sweep-armB-prelim-P3", "kind": "sweep", "host": "npu2", "stall_s": 3600,
                 "note": "P3 arm B (2 x TP4, max-num-seqs 128) levels 16, 32, 48, 64, 96 per instance; one pass",
                 "params": {"run_prefix": self.p["P3"]}},
                {"id": "sweep-armB-prelim-P3b-mintok4096", "kind": "sweep", "host": "npu2", "stall_s": 3600,
                 "note": "P3b arm B worst case --min-tokens 4096 at the highest passing P3 level; steps down until admissible",
                 "params": {"run_prefix": self.p["P3b"]}}]

    def gate(self) -> None:
        if self.done("gate"):
            log("gate: already passed (resumed)")
            return
        self.set_current("gate")
        waited = 0
        while True:
            status = self.ops.gate_status(self.gate_path)
            final, why = gate_final(status, self.ops.pid_alive((status or {}).get("pid")))
            self.status["gate"] = {"checked": utc_now(), "final": final, "why": why, "path": self.gate_path}
            self.write_status()
            if final:
                log(f"gate: chain is final ({why})")
                self.mark("gate", {"why": why, "gate_state": (status or {}).get("state"),
                                   "gate_reason": (status or {}).get("reason"), "waited_s": waited})
                return
            if waited % 600 == 0:
                log(f"gate: waiting, {why}")
            self.sleep(self.gate_poll)
            waited += self.gate_poll

    def remove_gate_chain_entries(self) -> None:
        """Once the gate chain is final its watchdog entries are stale: remove them (npu2 registry, one atomic edit)."""
        if self.done("gate-entries"):
            return
        ids = [i for i in self.ops.job_ids() if E_ENTRY.match(i)]
        self.ops.patch_jobs([], remove=ids)
        self.mark("gate-entries", {"removed": ids})

    def run_p3(self) -> int:
        a_specs = self.capture_arm_a()
        self.ops.patch_jobs(self.p3_job_entries())
        self.gate()
        self.remove_gate_chain_entries()
        self.ops.start_sampler()
        passed = self.phase_p3(a_specs)
        knee = self.phase_p3b(passed)
        self.ops.stop_sampler()
        self.phase_p4(a_specs)
        reason = None
        if knee is None:
            reason = "P3b: no admissible worst-case level on arm B (or no P3 level passed); arm A restored"
            log(f"CHAIN-WARN {reason}")
        self.status.update(state="done", current_step=None, reason=reason)
        self.write_status()
        log(f"P3b result: admissible worst-case level {knee} per instance on arm B; arm A restored")
        log("CHAIN DONE")
        return 0

    def run_p2b(self) -> int:
        a_specs = self.capture_arm_a()
        knee = self.phase_p2b(a_specs)
        self.ops.stop_sampler()
        if knee is None:
            self.status.update(state="failed", reason="no admissible worst-case level: 40 and 32 both failed; arm A left up")
            self.write_status()
            log("CHAIN-BAD P2b: no admissible worst-case level (40 and 32 failed); arm A left up")
            return 5
        self.status.update(state="done", current_step=None, reason=None)
        self.write_status()
        log(f"P2b result: admissible worst-case level {knee} per instance; arm A left up")
        log("CHAIN DONE")
        return 0

    def run(self) -> int:
        self.heartbeat()
        try:
            if self.mode == "p2b":
                return self.run_p2b()
            if self.mode == "p3":
                return self.run_p3()
            a_specs = self.capture_arm_a()
            if not self.done("p1-L56"):
                self.set_current("preflight-arm-a")
                self.ops.wait_healthy(a_specs, 120)
            self.ops.start_sampler()
            passed = self.phase_p1(a_specs)
            self.phase_p2(a_specs, passed)
            self.phase_p3(a_specs)
            self.ops.stop_sampler()
            self.phase_p4(a_specs)
            self.status.update(state="done", current_step=None, reason=None)
            self.write_status()
            log("CHAIN DONE")
            return 0
        except ChainStop as stop:
            self.status.update(state="failed", reason=str(stop))
            self.write_status()
            log(f"CHAIN-BAD stopped at {self.status['current_step']}: {stop}")
            return 2
        except KeyboardInterrupt:
            self.status.update(state="interrupted", reason="signal; servers left as they are; rerun with --resume")
            self.write_status()
            log("CHAIN-BAD interrupted by a signal; servers left running; rerun with --resume")
            return 3
        except Exception as error:
            self.status.update(state="failed", reason=f"{type(error).__name__}: {error}")
            self.write_status()
            log(f"CHAIN-BAD unexpected {type(error).__name__}: {error}")
            return 4
        finally:
            self._hb_stop.set()
            try:
                self.ops.stop_sampler()
            except Exception:
                pass


# ---------------------------------------------------------------- dry-run operations (CPU only, no side effects)
class FakeOps:
    """Scripted operations for the rehearsal and the unit tests: nothing touches docker, a card or the registry."""

    def __init__(self, fail: set[str] = frozenset(), unhealthy: bool = False, ceiling_bound_levels: set[int] = frozenset(),
                 free_mib: int = 4100, preempt: set[str] = frozenset(), die: set[str] = frozenset(),
                 kv_peak: dict | None = None) -> None:
        self.fail, self.unhealthy, self.ceiling = set(fail), unhealthy, set(ceiling_bound_levels)
        self.preempt, self.die, self.kv_peak = set(preempt), set(die), dict(kv_peak or {})
        self.free_mib = free_mib
        self.calls: list[str] = []
        self.running = {f"saea-arm-a-{i}" for i in range(4)}
        self.jobs: dict = {}

    def note(self, text: str) -> None:
        self.calls.append(text)
        log(f"[dry-run] {text}")

    def gate_status(self, path): return {"state": "done", "pid": 4242, "reason": None}
    def pid_alive(self, pid) -> bool: return False
    def job_ids(self) -> list[str]:
        return ["prelim-E-chain", "vllm-prelim-E2-shm", "sweep-prelim-E1-base-pass1", "sweep-prelim-E2-shm-pass3",
                "vllm-arm-a-1", "prelim-chain-p2b", "sweep-armA-prelim-P1"]

    def start_sampler(self) -> None: self.note("sampler start")
    def stop_sampler(self) -> None: self.note("sampler stop")
    def set_phase(self, prefix, level=None) -> None: self.note(f"phase {prefix}")
    def flush(self, prefix) -> None: pass

    def measure_level(self, step, prefix, level, urls, min_tokens, label, names=None) -> dict:
        self.note(f"measure {step} level {level} urls {urls} min_tokens {min_tokens}")
        names = names or [f"instance-{i}" for i in range(len(urls))]
        if step in self.die:
            self.running.discard("saea-arm-a-0")
            raise EndpointDied(names[0], {"dead_instances": [names[0]], "scripted": True})
        bad = step in self.fail
        before = [{"preemptions": 3.0, "kv_cache_usage_perc": 0.0} for _ in urls]
        after = [{"preemptions": 3.0 + (2.0 if (step in self.preempt and i == 0) else 0.0),
                  "kv_cache_usage_perc": 0.0} for i in range(len(urls))]
        kv = self.kv_peak.get(step, 0.8)
        kvs = kv if isinstance(kv, list) else [kv] * len(urls)
        summary = {"requests": 3 * level * len(urls), "ok": 3 * level * len(urls), "errors": 0, "timeouts": 0,
                   "wall_s": 600.0, "output_tok_s": 1000.0 + level * 10, "completion_tokens": {"p50": 3000},
                   "e2e_tok_s": {"p10": 9.0 if bad else 12.0}, "decode_tok_s": {"p10": 14.0},
                   "latency_s": {"max": 300.0, "p50": 200.0}, "ttft_s": {"p95": 10.0},
                   "preemptions_delta": sum(a["preemptions"] - b["preemptions"] for a, b in zip(after, before)),
                   "metrics_before": before, "metrics_after": after, "metrics_delta": {},
                   "gauges": [{"num_requests_waiting": {"max": 1.0 if level in self.ceiling else 0.0},
                               "kv_cache_usage_perc": {"max": kvs[i]}} for i in range(len(urls))]}
        return {"summary": summary, "report": f"{Path(prefix).name}-L{level}.json", "report_sha256": "0" * 64,
                "min_hbm_free_mib": self.free_mib, "aicore_mean": 70.0, "aicore_max": 100.0, "samples": 50,
                "poll_waiting_max": 0.0, "names": names, "instances": len(urls), "max_tokens_sent": 4096,
                "min_tokens_sent": min_tokens}

    def running_ours(self, prefix) -> list[str]: return sorted(n for n in self.running if n.startswith(prefix))
    def running_requests(self, url) -> float | None: return 0.0
    def container_running(self, name) -> bool: return name in self.running

    def stop_containers(self, names) -> None:
        self.note(f"docker stop {names}")
        self.running -= set(names)

    def wait_cards_free(self, timeout=0) -> dict:
        self.note("cards 0-7 below 4096 MiB")
        return {c: 100 for c in range(8)}

    def inspect_arm_a(self) -> list[dict]:
        return [{"name": f"saea-arm-a-{i}", "cards": f"{2 * i},{2 * i + 1}", "port": 8200 + i, "tp": 2,
                 "max_num_seqs": 64, "cpuset": ["192-223", "128-159", "0-31", "64-95"][i],
                 "hccl_base_port": 60000 + 100 * i, "cmd": ["serve", "--port", str(8200 + i)],
                 "devices": [f"/dev/davinci{2 * i}", f"/dev/davinci{2 * i + 1}"]} for i in range(4)]

    def launch(self, spec) -> str:
        self.note(f"launch NAME={spec['name']} CPUSET={spec['cpuset']} HCCL_BASE_PORT={spec['hccl_base_port']} "
                  f"MAX_NUM_SEQS={spec['max_num_seqs']} npu_serve.sh thinking {spec['cards']} {spec['port']} {spec['tp']}")
        self.running.add(spec["name"])
        return f"/dry/{spec['name']}.log"

    def argv_of(self, name) -> dict:
        spec = next(s for s in self.inspect_arm_a() if s["name"] == name)
        return {k: spec[k] for k in ("cmd", "cpuset", "hccl_base_port", "devices")}

    def wait_healthy(self, specs, timeout, on_tick=None) -> dict:
        if self.unhealthy and specs[0]["name"].startswith("saea-arm-b"):
            raise ChainStop(f"endpoints not healthy within {timeout:.0f} s: {[s['name'] for s in specs]}")
        self.note(f"healthy {[s['name'] for s in specs]} (timeout {timeout:.0f} s)")
        return {s["name"]: True for s in specs}

    def patch_jobs(self, upsert, remove=(), set_log=None) -> None:
        self.note(f"jobs upsert={[j['id'] for j in upsert]} remove={list(remove)} set_log={list(set_log or {})}")


def preflight() -> int:
    """Read-only checks on host 181: the running arm A argv, endpoints, cards, free CPU node, and the sweep flags."""
    ops = RealOps(LOGS, "preflight", False)
    hostname = Path("/etc/hostname").read_text().strip()
    print(f"host {hostname} (expected {EXPECTED_HOSTNAME})")
    specs = ops.inspect_arm_a()
    print(json.dumps(specs, indent=1))
    for spec in specs:
        base = f"http://127.0.0.1:{spec['port']}"
        print(spec["name"], "health", (http_get(base + "/health") or ("none",))[0], "models",
              (http_get(base + "/v1/models") or ("none",))[0])
    print("cards (aicore %, hbm used MiB, total):", read_cards())
    sweep = subprocess.run(["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                            "--cpuset-cpus", CLIENT_CPUSET, "-v", f"{ROOT}:{ROOT}", "-w", TREE,
                            "-e", f"PYTHONPATH={TREE}/src", "-e", "PYTHONDONTWRITEBYTECODE=1",
                            "--entrypoint", "/opt/saea-venv/bin/python", IMAGE,
                            "scripts/serving/synthetic_sweep.py", "--help"], capture_output=True, text=True)
    print("synthetic_sweep.py --min-tokens flag present:", "--min-tokens" in sweep.stdout, "exit", sweep.returncode)
    print("npu_serve.sh present:", Path(NPU_SERVE).exists(), "jobs.json readable:", read_json(JOBS) is not None)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ts", help="UTC stamp used in every report name (default: now)")
    ap.add_argument("--chain-dir", help="existing chain directory (with --resume)")
    ap.add_argument("--resume", action="store_true", help="skip finished steps of an existing chain directory")
    ap.add_argument("--dry-run", action="store_true", help="rehearse the decision logic with scripted measurements")
    ap.add_argument("--fail", default="", help="dry-run only: comma-separated step ids that fail p10, e.g. p1-L64,p2-L56")
    ap.add_argument("--ceiling-bound", default="", help="dry-run only: levels whose waiting gauge is above 0")
    ap.add_argument("--preempt", default="",
                    help="dry-run only: comma-separated step ids where instance 0 preempts, e.g. p2b-L40")
    ap.add_argument("--die", default="",
                    help="dry-run only: comma-separated step ids whose level ends with a dead endpoint, e.g. p2b-L40")
    ap.add_argument("--unhealthy", action="store_true", help="dry-run only: arm B endpoints never become healthy")
    ap.add_argument("--p2b", action="store_true",
                    help="run P2b (plan 3.8): stop idle saea-arm-a-*, fresh arm A from the capture, worst case at 40 then 32")
    ap.add_argument("--p3", action="store_true",
                    help="run P3 + P3b + P4 (plan 3.8): gate on the E chain, arm B levels, B's worst case, arm A relaunch")
    ap.add_argument("--gate-file", default=DEFAULT_GATE,
                    help="--p3 only: chain-status.json of the chain that holds the cards (read at most every 120 s)")
    ap.add_argument("--kv-peak", default="",
                    help="dry-run only: step=fraction pairs, e.g. p3b-L96=0.93,p3b-L64=0.95")
    ap.add_argument("--arm-a-launch", default=f"{LOGS}/prelim-chain-20261029T094933Z/arm-a-launch.json",
                    help="--p2b and --p3: captured arm A argv to relaunch (copied into the new chain directory)")
    ap.add_argument("--preflight", action="store_true",
                    help="read-only checks on the real host: argv capture, endpoints, cards, sweep flags; changes nothing")
    ap.add_argument("--out-root", help="dry-run only: output root (default: a temporary directory)")
    args = ap.parse_args()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if args.preflight:
        return preflight()
    if args.dry_run:
        root = Path(args.out_root or tempfile.mkdtemp(prefix="prelim-chain-dry-"))
        ts = args.ts or "DRYRUN"
        chain_dir, logs = root / f"prelim-chain-{ts}", str(root)
        chain_dir.mkdir(parents=True, exist_ok=True)
        ops = FakeOps(fail={x for x in args.fail.split(",") if x}, unhealthy=args.unhealthy,
                      ceiling_bound_levels={int(x) for x in args.ceiling_bound.split(",") if x},
                      preempt={x for x in args.preempt.split(",") if x}, die={x for x in args.die.split(",") if x},
                      kv_peak={k: float(v) for k, _, v in (x.partition("=") for x in args.kv_peak.split(",") if x)})
        log(f"dry-run output under {root}")
        if args.p2b or args.p3:
            chain_dir = root / f"prelim-chain-{'p3' if args.p3 else 'p2b'}-{ts}"
            chain_dir.mkdir(parents=True, exist_ok=True)
            write_atomic(chain_dir / "arm-a-launch.json", json.dumps(
                {"captured": utc_now(), "instances": ops.inspect_arm_a()}, indent=2) + "\n")
        mode = "p3" if args.p3 else "p2b" if args.p2b else "chain"
        code = Chain(ops, chain_dir, ts, logs, mode=mode, gate_path="/dry/gate.json", gate_poll=0.0,
                     sleep=lambda _s: None).run()
        print(json.dumps(read_json(chain_dir / "chain-status.json")["decisions"], indent=1, sort_keys=True))
        return code
    hostname = Path("/etc/hostname").read_text().strip()
    if hostname != EXPECTED_HOSTNAME:
        sys.exit(f"refusing to run on {hostname!r}: this chain belongs to host 181 ({EXPECTED_HOSTNAME})")
    if args.resume:
        chain_dir = Path(args.chain_dir or sys.exit("--resume needs --chain-dir"))
        config = read_json(chain_dir / "config.json")
        ts, mode = config["ts"], config.get("mode", "chain")
        gate_file = config.get("gate_file")
    else:
        mode = "p3" if args.p3 else "p2b" if args.p2b else "chain"
        gate_file = args.gate_file if args.p3 else None
        ts = args.ts or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        chain_dir = Path(args.chain_dir or f"{LOGS}/prelim-chain{'-p3' if args.p3 else '-p2b' if args.p2b else ''}-{ts}")
        if (chain_dir / "config.json").exists():
            sys.exit(f"{chain_dir} already holds a chain; use --resume")
        chain_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(chain_dir, 0o777)
        config = {"ts": ts, "mode": mode, "created": utc_now(), "script": str(Path(__file__).resolve())}
        if args.p3:
            config["gate_file"] = gate_file
        if args.p2b or args.p3:
            source = Path(args.arm_a_launch)
            captured = read_json(source)
            if not captured:
                sys.exit(f"--p2b/--p3 need a readable arm A capture: {source}")
            write_atomic(chain_dir / "arm-a-launch.json", json.dumps(captured, indent=2) + "\n")
            config["arm_a_launch_source"] = str(source)
        write_atomic(chain_dir / "config.json", json.dumps(config) + "\n")
    os.chmod(chain_dir / "config.json", 0o666)
    (chain_dir / "steps").mkdir(exist_ok=True)
    os.chmod(chain_dir / "steps", 0o777)
    ops = RealOps(LOGS, ts, args.resume)
    log(f"chain {chain_dir.name} mode {mode} pid {os.getpid()} resume={args.resume} dir {chain_dir}")
    return Chain(ops, chain_dir, ts, LOGS, mode=mode, gate_path=gate_file).run()


if __name__ == "__main__":
    sys.exit(main())
