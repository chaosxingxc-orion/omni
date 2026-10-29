#!/usr/bin/env python3
"""PRELIMINARY E3 chain `prelim-E3` on host 181 (plan section 3.11 ruling 3(b)): is `--mm-processor-cache-type shm`
safe on the crash-regime arm A shape? One detached, checkpointed, resumable run.

  gate        waits (reads the status file at most every 120 s) until the newest `prelim-chain-p3-*/chain-status.json`
              is FINAL (state done/failed/interrupted AND its pid gone). If no such directory appears within 600 s the
              P3 launch was denied: the chain stops having touched nothing. No card work happens before the gate.
  stop        stops our idle saea-arm-a-0..3 and requires cards 0-7 below 4096 MiB.
  base        BASELINE pass: 4 x TP2 launched exactly as the captured arm A (npu_serve_e.sh, EXTRA_ARGS empty), healthy,
              worst case (--min-tokens 4096) at 48 per instance, one pass, seed 20261028, with the 30 s liveness poll;
              per-instance preemptions, KV peak, engine exits and the crash-evidence scan of every instance log.
              A dead instance ends the pass (recorded; no step-down). INFORMATIVE ONLY. Then all four are stopped.
  shm         4 x TP2 relaunched with EXTRA_ARGS="--mm-processor-cache-type shm", healthy. BEFORE load: warm sequential
              A/A probe (greedy_probe.py, the D3 protocol: 16 fixed requests, temperature 0, seed 777, max_tokens 64;
              one warm-up send, then S2 and S3) on instance 0, compared with the D3 baseline warm runs S2 and S3; then
              two worst-case passes at 48 per instance, seeds 20261028 and 20261029. A dead instance ends E3: DROPPED.
  verdict     ADOPTED if both shm passes have zero engine exits AND the warm probe agrees 16/16 with the D3 baseline warm
              runs (shm S2 vs D3 S2 and shm S3 vs D3 S3); otherwise DROPPED with the reasons; INCOMPLETE when the
              evidence is missing (infrastructure failure). The baseline pass never changes the verdict.
  restore     the shm instances are stopped and arm A is relaunched from the capture with the ORIGINAL npu_serve.sh
              (baseline flags), waited healthy, and the vllm-arm-a-* watchdog entries are registered again.

Rails: no dataset row, no exposure entry, no readiness flip, no commit, no push, never a busy card, never --restart, no
study-tree edit. Only saea-arm-a-N and saea-prelim-e3-{base,shm}-N containers are touched. Synthetic load; PRELIMINARY.
The vLLM server flags are never changed except by EXTRA_ARGS through the copy launcher npu_serve_e.sh.

usage (on host 181, detached; no --restart):
  nohup setsid python3 e3_chain.py [--ts <UTC stamp>] < /dev/null > <chain-dir>/chain.log 2>&1 &
  resume: python3 e3_chain.py --resume --chain-dir <chain-dir>        (finished steps are skipped)
  dry-run (CPU only, no docker, no card, no network): python3 e3_chain.py --dry-run [--scenario a,b=N,...]
Layout: <logs>/prelim-E3-chain-<ts>/{config.json,chain-status.json,chain.log,arm-a-launch.json,steps/,*.group.json};
result <logs>/prelim-E3-<ts>.json; sweep reports <logs>/sweep-prelim-E3-{base,shm}-pass<n>-<ts>-L48.json (+ summary,
-npu-smi.json, .stderr.log); probe files <logs>/prelim-E3-shm-probe-{W,S2,S3}-<ts>.json and ...-probe-compare-<ts>.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e_crash_chain as ec  # noqa: E402  (EReal: launch_e, crash_evidence, cards_below, ...)
import greedy_probe as gp  # noqa: E402  (compare_runs, load_results: the D3 protocol and compare logic)
import prelim_chain as pc  # noqa: E402  (RealOps.measure_level with the 30 s liveness poll, helpers)

ROOT, LOGS = pc.ROOT, pc.LOGS
SCRIPTS = Path(__file__).resolve().parent
LEVEL = 48  # per instance: the known crash regime (P2 crashed 2 of 4 instances here)
MIN_TOKENS = ec.MIN_TOKENS  # 4096: every completion forced long
SEEDS = (str(ec.SEED_BASE), str(ec.SEED_BASE + 1))  # shm pass 1 and 2; the baseline pass uses SEEDS[0]
SHM_ARGS = ec.SHM_ARGS
D3_DIR = f"{LOGS}/prelim-D3-greedy-20261029T130527Z"
DEFAULT_CAPTURE = ec.DEFAULT_CAPTURE
GATE_POLL_S = 120
GATE_APPEAR_TIMEOUT_S = 600
GATE_GLOB = "prelim-chain-p3-*"
GATE_FINAL = pc.GATE_FINAL
TAGS = ("base", "shm")
EXTRA = {"base": "", "shm": SHM_ARGS}
E3_OURS = re.compile(r"^saea-prelim-e3-(base|shm)-\d$")
ARM_A_OURS = re.compile(r"^saea-arm-a-\d$")
ARM_B = re.compile(r"^saea-arm-b-\d+$")
PROBE_N = 16
HEALTH_TIMEOUT_S = pc.A_HEALTH_TIMEOUT_S
CONFIRM_SHM = ec.CONFIRM_RE[SHM_ARGS]

utc_now, log, write_atomic, read_json = pc.utc_now, pc.log, pc.write_atomic, pc.read_json


# ---------------------------------------------------------------- pure logic (unit tested)
def cname(tag: str, index: int) -> str:
    return f"saea-prelim-e3-{tag}-{index}"


def newest_gate(logs: str) -> str | None:
    """chain-status.json of the newest `prelim-chain-p3-*` directory (name order = timestamp order), else None."""
    dirs = sorted(p for p in Path(logs).glob(GATE_GLOB) if p.is_dir() and "DRYRUN" not in p.name)
    return str(dirs[-1] / "chain-status.json") if dirs else None


def gate_final(status: dict | None, pid_alive: bool) -> tuple[bool, str]:
    """The P3 chain is FINAL when its state is done/failed/interrupted AND its process has exited."""
    return pc.gate_final(status, pid_alive)


def engine_exits(names: list[str], running: set[str], scans: dict[str, dict]) -> list[dict]:
    """Instances that left the pass dead: container gone, or the log carries the fatal-engine-error or the
    `len(mm_embeddings)=` assertion even though the API process may linger."""
    out = []
    for name in names:
        scan = scans.get(name) or {}
        reasons = []
        if name not in running:
            reasons.append("container_not_running")
        if scan.get("fatal_engine_error"):
            reasons.append("fatal_engine_error_in_log")
        if scan.get("mm_embeddings_assertion"):
            reasons.append("mm_embeddings_assertion_in_log")
        if reasons:
            out.append({"name": name, "reasons": reasons, "execute_mm_encoder": bool(scan.get("execute_mm_encoder")),
                        "died_request_ids": scan.get("died_request_ids", []), "k2_confirmed": scan.get("k2_confirmed", False)})
    return out


def probe_agreement(cmp_s2: dict, cmp_s3: dict, n: int = PROBE_N) -> dict:
    """16/16 with the D3 baseline warm runs: shm S2 vs D3 S2 AND shm S3 vs D3 S3, all n requests compared."""
    ident = [cmp_s2["byte_identical"], cmp_s3["byte_identical"]]
    full = cmp_s2["requests"] == n and cmp_s3["requests"] == n and cmp_s2["count_a"] == n and cmp_s3["count_a"] == n
    return {"identical_s2": ident[0], "identical_s3": ident[1], "n": n, "complete": full,
            "agrees": bool(full and ident[0] == n and ident[1] == n)}


def e3_verdict(shm: dict | None, probe: dict | None, expected_passes: int = 2) -> dict:
    """Plan 3.11 ruling 3(b). `shm` = {"startup_failed": bool, "passes": [...]}; `probe` = probe_agreement output or None."""
    reasons: list[str] = []
    missing: list[str] = []
    if shm is None:
        return {"verdict": "INCOMPLETE", "reasons": ["shm group never ran"]}
    if shm.get("startup_failed"):
        reasons.append(f"shm instances did not become healthy: {shm.get('startup_reason')}")
    passes = shm.get("passes", [])
    for p in passes:
        if p["outcome"] == "exit":
            reasons.append(f"engine exit in shm pass {p['pass']}: {[e['name'] for e in p.get('engine_exits', [])]}")
    if not shm.get("startup_failed"):
        if probe is None:
            if not any(p["outcome"] == "exit" for p in passes):
                missing.append("warm probe missing")
        elif not probe["complete"]:
            missing.append("warm probe incomplete")
        elif not probe["agrees"]:
            reasons.append(f"warm probe disagrees with the D3 baseline: S2 {probe['identical_s2']}/{probe['n']}, "
                           f"S3 {probe['identical_s3']}/{probe['n']}")
        if len(passes) < expected_passes and not any(p["outcome"] == "exit" for p in passes):
            missing.append(f"only {len(passes)} of {expected_passes} shm passes ran")
    if reasons:
        return {"verdict": "DROPPED", "reasons": reasons}
    if missing:
        return {"verdict": "INCOMPLETE", "reasons": missing}
    return {"verdict": "ADOPTED", "reasons": []}


def pass_record(tag: str, index: int, seed: str, names: list[str], measure: dict | None, died: "pc.EndpointDied | None",
                running: set[str], scans: dict[str, dict]) -> dict:
    """One pass: per-instance preemptions, KV peak, liveness and the exit decision, from a finished or an aborted level."""
    row: dict = {"pass": index, "arm": tag, "level": LEVEL, "seed": seed, "min_tokens": MIN_TOKENS}
    if died is not None:
        d = died.detail
        first, last = d.get("preemptions_first_polled"), d.get("preemptions_last_polled")
        kv = d.get("kv_peak_polled")
        inst = []
        for i, name in enumerate(names):
            pre = None
            if first and last and i < len(first) and i < len(last) and first[i] is not None and last[i] is not None:
                pre = last[i] - first[i]
            inst.append({"name": name, "preemptions_delta": pre, "kv_peak": kv[i] if kv and i < len(kv) else None})
        row.update(dead_instances=d.get("dead_instances", []), detected_at=d.get("detected_at"),
                   client_exit_code=d.get("client_exit_code"))
    else:
        s = measure["summary"]
        inst = pc.per_instance_stats(s, names, measure.get("poll_kv_peak"))
        row.update(requests=s["requests"], ok=s["ok"], errors=s["errors"], timeouts=s["timeouts"], wall_s=s["wall_s"],
                   host_output_tok_s=s["output_tok_s"], p10_e2e_tok_s=s["e2e_tok_s"]["p10"],
                   latency_max_s=s["latency_s"]["max"], min_hbm_free_mib=measure["min_hbm_free_mib"],
                   report=measure["report"], report_sha256=measure["report_sha256"])
    exits = engine_exits(names, running, scans)
    dead = {e["name"] for e in exits} | set(row.get("dead_instances", []))
    if died is not None and not exits:  # the level died on the liveness poll but the log scan found nothing: still an exit
        exits = [{"name": n, "reasons": ["endpoint_died"], "execute_mm_encoder": False, "died_request_ids": [],
                  "k2_confirmed": False} for n in row.get("dead_instances", [])]
    for entry in inst:
        entry["alive"] = entry["name"] not in dead
    pres = [e["preemptions_delta"] for e in inst if e["preemptions_delta"] is not None]
    row.update(instances=inst, preemptions_total=sum(pres) if pres else None, engine_exits=exits,
               outcome="exit" if (exits or died is not None) else "completed",
               kv_peak_max=max((e["kv_peak"] for e in inst if e["kv_peak"] is not None), default=None))
    return row


# ---------------------------------------------------------------- real operations (host 181 only)
class E3Real(ec.EReal):
    """ec.EReal (launch_e, crash_evidence, cards_below, log_confirms) + pc.RealOps (4-instance measure_level with the
    30 s liveness poll, arm A launch, registry) with the E3 container-name guard."""

    @staticmethod
    def _check_ours(names: list[str]) -> None:
        bad = [n for n in names if not (E3_OURS.match(n) or ARM_A_OURS.match(n))]
        if bad:
            raise pc.ChainStop(f"refusing to touch container(s) that are not ours: {bad}")

    def find_gate(self) -> str | None:
        return newest_gate(LOGS)

    def scan_logs(self, paths: dict[str, str]) -> dict[str, dict]:
        time.sleep(8)  # the detached `docker logs -f` followers may lag the containers' last lines
        out = {}
        for name, path in paths.items():
            lines = self.log_lines(path) if path else []
            ev = ec.crash_evidence_from_lines(lines)
            ev["server_log"], ev["server_log_lines"] = path, len(lines)
            ev["error_tail"] = [l.strip()[:300] for l in lines[-400:] if re.search(r"ERROR|Error|Traceback", l)][-8:]
            out[name] = ev
        return out

    def measure_level(self, step, prefix, level, urls, min_tokens, label, names=None, seed=None):
        """pc.RealOps.measure_level reads the seed from the module constant pc.SEED; E3 needs one seed per pass, so the
        constant is set for the call and restored (the chain measures from one thread at a time)."""
        saved = pc.SEED
        if seed is not None:
            pc.SEED = str(seed)
        try:
            return pc.RealOps.measure_level(self, step, prefix, level, urls, min_tokens, label, names=names)
        finally:
            pc.SEED = saved

    def probe_send(self, url: str, out: str, bodies_file: str) -> None:
        done = subprocess.run([sys.executable, str(SCRIPTS / "greedy_probe.py"), "send", "--url", url, "--out", out,
                               "--sequential", "--bodies-file", bodies_file], capture_output=True, text=True,
                              timeout=3600, stdin=subprocess.DEVNULL)
        log(f"probe send {Path(out).name}: exit {done.returncode} {(done.stdout or done.stderr).strip()[-200:]}")
        if done.returncode != 0:
            raise pc.ChainStop(f"greedy_probe send failed: {(done.stderr or done.stdout).strip()[-300:]}")


# ---------------------------------------------------------------- the chain
class E3Chain:
    def __init__(self, ops, chain_dir: Path, ts: str, logs: str, capture: list[dict], gate_path: str | None,
                 baseline_dir: str = D3_DIR, gate_poll: float = GATE_POLL_S, sleep=time.sleep) -> None:
        self.ops, self.dir, self.ts, self.logs = ops, chain_dir, ts, logs
        self.capture, self.gate_path, self.baseline_dir = capture, gate_path, baseline_dir
        self.gate_poll, self.sleep = gate_poll, sleep
        self.steps_dir = chain_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.touched = False
        self.status = {"chain": chain_dir.name, "state": "running", "current_step": None, "started": utc_now(),
                       "updated": utc_now(), "pid": os.getpid(), "reason": None, "gate": {}, "decisions": {},
                       "steps": {}, "paths": {"logs": logs, "ts": ts, "baseline_dir": baseline_dir}}
        old = read_json(chain_dir / "chain-status.json")
        if old:
            for key in ("decisions", "steps", "gate"):
                self.status[key] = old.get(key, self.status[key])
            self.status["started"] = old.get("started", self.status["started"])
        self._hb_stop = threading.Event()

    # --- checkpoints and status
    def done(self, step: str):
        return read_json(self.steps_dir / f"{step}.done.json")

    def mark(self, step: str, decision: dict) -> None:
        with self.lock:
            write_atomic(self.steps_dir / f"{step}.done.json", json.dumps(
                {"step": step, "finished": utc_now(), "decision": decision}, indent=2, sort_keys=True, default=str) + "\n")
            self.status["steps"][step] = {"state": "done", "finished": utc_now()}
            self.write_status()

    def set_current(self, step: str | None) -> None:
        with self.lock:
            self.status["current_step"] = step
            self.status["current_since"] = utc_now()
            if step:
                log(f"STEP {step}")
            self.write_status()

    def write_status(self) -> None:
        with self.lock:
            self.status["updated"] = utc_now()
            write_atomic(self.dir / "chain-status.json",
                         json.dumps(self.status, indent=1, sort_keys=True, default=str) + "\n")

    def heartbeat(self) -> None:
        def loop() -> None:
            while not self._hb_stop.wait(60):
                log(f"heartbeat step={self.status['current_step']} since={self.status.get('current_since')}")
                self.write_status()
        threading.Thread(target=loop, daemon=True).start()

    # --- watchdog registry
    def chain_entries(self) -> list[dict]:
        return [{"id": "prelim-E3-chain", "kind": "generic_log", "host": "npu2", "stall_s": 3600,
                 "note": "E3 (plan 3.11 ruling 3b): gate on the P3 chain, baseline pass, then shm probe + two shm passes "
                         "on 4 x TP2 at 48/instance, then arm A restored; heartbeat every 60 s; BAD means the chain stopped",
                 "params": {"path": str(self.dir / "chain.log"), "bad_regex": "CHAIN-BAD|Traceback",
                            "done_regex": "CHAIN DONE", "status_file": str(self.dir / "chain-status.json")}}]

    def prefix_of(self, tag: str, index: int) -> str:
        return f"{self.logs}/sweep-prelim-E3-{tag}-pass{index}-{self.ts}"

    def sweep_entry(self, tag: str, index: int) -> dict:
        return {"id": f"sweep-prelim-E3-{tag}-pass{index}", "kind": "sweep", "host": "npu2", "stall_s": 3600,
                "note": f"E3 {tag} pass {index}: worst case --min-tokens 4096 at {LEVEL}/instance on 4 x TP2; preemptions and "
                        "exits are the subject of the experiment, a BAD event here is expected data, not a defect",
                "params": {"run_prefix": self.prefix_of(tag, index)}}

    def instance_entries(self, tag: str, logs: dict) -> list[dict]:
        return [{"id": f"vllm-prelim-E3-{tag}-{i}", "kind": "vllm_server", "host": "npu2",
                 "note": f"E3 {tag} instance {i} (plan 3.11); a death is the subject of the experiment; removed when the "
                         "group is stopped",
                 "params": {"container": cname(tag, i), "port": self.capture[i]["port"], "log": logs.get(cname(tag, i))}}
                for i in range(len(self.capture))]

    def arm_a_entries(self, logs: dict) -> list[dict]:
        return [{"id": f"vllm-arm-a-{s['name'].rsplit('-', 1)[1]}", "kind": "vllm_server", "host": "npu2",
                 "note": f"arm A relaunched by prelim-E3-chain-{self.ts} after E3; left UP",
                 "params": {"container": s["name"], "port": s["port"], "log": logs.get(s["name"])}} for s in self.capture]

    # --- gate
    def gate(self) -> None:
        if self.done("gate"):
            log("gate: already passed (resumed)")
            return
        self.set_current("gate")
        waited = 0
        path = self.gate_path
        while True:
            if path is None:
                path = self.ops.find_gate()
                if path is None:
                    self.status["gate"] = {"checked": utc_now(), "final": False, "why": "no prelim-chain-p3-* directory yet"}
                    self.write_status()
                    if waited >= GATE_APPEAR_TIMEOUT_S:
                        raise pc.ChainStop(f"no {GATE_GLOB} directory appeared within {GATE_APPEAR_TIMEOUT_S} s: the P3 "
                                           "launch was denied or did not start; nothing was touched")
                    if waited % 600 == 0:
                        log("gate: no P3 chain directory yet")
                    self.sleep(self.gate_poll)
                    waited += self.gate_poll
                    continue
                log(f"gate: resolved the newest P3 chain status file {path}")
            status = self.ops.gate_status(path)
            final, why = gate_final(status, self.ops.pid_alive((status or {}).get("pid")))
            self.status["gate"] = {"checked": utc_now(), "final": final, "why": why, "path": path}
            self.write_status()
            if final:
                log(f"gate: P3 chain is final ({why})")
                self.mark("gate", {"why": why, "p3_state": (status or {}).get("state"),
                                   "p3_reason": (status or {}).get("reason"), "path": path, "waited_s": waited})
                return
            if waited % 600 == 0:
                log(f"gate: waiting, {why}")
            self.sleep(self.gate_poll)
            waited += self.gate_poll

    # --- stop helpers
    def stop_and_wait(self, names: list[str], what: str) -> None:
        alive = [n for n in names if n in self.ops.running_names()]
        if alive:
            self.ops.stop_containers(alive)
        waited = 0
        while any(n in self.ops.running_names() for n in names) and waited < 240:
            self.sleep(3)
            waited += 3
        left = [n for n in names if n in self.ops.running_names()]
        if left:
            raise pc.ChainStop(f"{what}: {left} still running 240 s after docker stop")

    def stop_arm_a(self) -> None:
        if self.done("stop-arm-a"):
            return
        self.set_current("stop-arm-a")
        running = self.ops.running_names()
        b = sorted(n for n in running if ARM_B.match(n))
        if b:
            raise pc.ChainStop(f"arm B containers {b} are still running; E3 does not stop them (P3 chain not restored)")
        names = [s["name"] for s in self.capture]
        alive = [n for n in names if n in running]
        for name in alive:  # never stop an instance that is serving someone
            spec = next(s for s in self.capture if s["name"] == name)
            waited = 0
            while True:
                busy = self.ops.running_requests(f"http://127.0.0.1:{spec['port']}")
                if not busy:
                    break
                if waited >= 180:
                    raise pc.ChainStop(f"{name} still has {busy:g} running request(s) after 180 s; refusing to stop it")
                self.sleep(5)
                waited += 5
        self.touched = True
        self.ops.patch_jobs([], remove=[f"vllm-arm-a-{i}" for i in range(len(names))])
        self.stop_and_wait(alive, "arm A")
        used = self.ops.cards_below(list(range(8)))
        self.mark("stop-arm-a", {"stopped": alive, "hbm_used_mib_after": used})

    # --- one 4-instance group
    def group_names(self, tag: str) -> list[str]:
        return [cname(tag, i) for i in range(len(self.capture))]

    def urls(self) -> list[str]:
        return [f"http://127.0.0.1:{s['port']}" for s in self.capture]

    def launch_group(self, tag: str, extra: str) -> dict:
        """Launch (or adopt) the 4 instances; returns {name: log path}."""
        names = self.group_names(tag)
        prior = self.done(f"{tag}-launch")
        if prior:
            return prior["decision"]["logs"]
        self.set_current(f"{tag}-launch")
        running = self.ops.running_names()
        logs: dict[str, str] = {}
        todo = [(i, n) for i, n in enumerate(names) if n not in running]
        if todo:
            self.ops.cards_below(list(range(8)) if len(todo) == len(names) else
                                 [int(c) for i, _ in todo for c in self.capture[i]["cards"].split(",")])
        for i, name in enumerate(names):
            if name in running:  # died between `docker run` and the checkpoint of a previous run: adopt
                found = sorted(Path(self.logs).glob(f"{name}-*.log"))
                logs[name] = str(found[-1]) if found else ""
            else:
                logs[name] = self.ops.launch_e(name, self.capture[i], extra)
        self.mark(f"{tag}-launch", {"containers": names, "extra_args": extra, "logs": logs})
        return logs

    def wait_group(self, tag: str, extra: str, logs: dict) -> dict:
        """Healthy, argv check, watchdog entries. Returns {"ok": True} or {"ok": False, "reason", "scans"}."""
        prior = self.done(f"{tag}-wait")
        if prior:
            return {"ok": True}
        names = self.group_names(tag)
        self.set_current(f"{tag}-wait")
        specs = [{"name": n, "port": self.capture[i]["port"]} for i, n in enumerate(names)]
        try:
            self.ops.wait_healthy(specs, HEALTH_TIMEOUT_S, on_tick=self.write_status)
        except pc.ChainStop as stop:
            scans = self.ops.scan_logs(logs)
            log(f"{tag}: startup failed: {stop}")
            return {"ok": False, "reason": str(stop), "scans": scans}
        mismatch = []
        for i, name in enumerate(names):
            now = self.ops.argv_of(name)
            spec = self.capture[i]
            want = list(spec["cmd"]) + (extra.split() if extra else [])
            if not (now["cmd"] == want and now["cpuset"] == spec["cpuset"]
                    and now["hccl_base_port"] == spec["hccl_base_port"] and now["devices"] == spec["devices"]):
                mismatch.append(name)
        if mismatch:  # a different shape would make the experiment meaningless: stop before any load
            raise pc.ChainStop(f"{tag}: argv differs from the capture (+extra) for {mismatch}")
        confirms = {}
        if extra:
            confirms = {n: self.ops.log_confirms(logs.get(n, ""), CONFIRM_SHM) for n in names}
        self.ops.patch_jobs(self.instance_entries(tag, logs))
        self.mark(f"{tag}-wait", {"healthy": names, "argv_identical_to_capture_plus_extra": True,
                                  "log_confirms_extra": confirms})
        return {"ok": True}

    def scan_group(self, tag: str, logs: dict) -> tuple[set, dict]:
        scans = self.ops.scan_logs({n: logs.get(n, "") for n in self.group_names(tag)})
        return self.ops.running_names(), scans

    # --- warm sequential A/A probe (D3 protocol) on instance 0, compared with the D3 baseline warm runs
    def probe(self, tag: str) -> dict:
        step = f"{tag}-probe"
        prior = self.done(step)
        if prior:
            return prior["decision"]
        self.set_current(step)
        url = self.urls()[0]
        bodies = f"{self.baseline_dir}/bodies.json"
        paths = {k: f"{self.logs}/prelim-E3-{tag}-probe-{k}-{self.ts}.json" for k in ("W", "S2", "S3")}
        for key in ("W", "S2", "S3"):  # W = warm-up (discarded), then two sequential runs
            self.ops.probe_send(url, paths[key], bodies)
        mine = {k: gp.load_results(paths[k]) for k in ("S2", "S3")}
        theirs = {k: gp.load_results(f"{self.baseline_dir}/{k}.json") for k in ("S2", "S3")}

        def brief(c: dict) -> dict:
            return {k: c[k] for k in ("requests", "count_a", "count_b", "byte_identical", "mismatched_indices",
                                      "max_abs_logprob_diff")}
        cmp_ = {
            f"{tag}_S2_vs_D3_S2": brief(gp.compare_runs(mine["S2"], theirs["S2"])),
            f"{tag}_S3_vs_D3_S3": brief(gp.compare_runs(mine["S3"], theirs["S3"])),
            f"{tag}_S2_vs_D3_S3": brief(gp.compare_runs(mine["S2"], theirs["S3"])),
            f"{tag}_S3_vs_D3_S2": brief(gp.compare_runs(mine["S3"], theirs["S2"])),
            f"{tag}_S2_vs_{tag}_S3": brief(gp.compare_runs(mine["S2"], mine["S3"]))}
        agreement = probe_agreement(cmp_[f"{tag}_S2_vs_D3_S2"], cmp_[f"{tag}_S3_vs_D3_S3"])
        sha = {k: (read_json(paths[k]) or {}).get("bodies_sha256") for k in ("S2", "S3")}
        base_sha = (read_json(f"{self.baseline_dir}/S2.json") or {}).get("bodies_sha256")
        agreement["bodies_match_d3"] = sha["S2"] == base_sha and sha["S3"] == base_sha
        ok_counts = {k: sum(1 for r in mine[k] if r.get("status") == "ok") for k in mine}
        decision = {"agreement": agreement, "comparisons": cmp_, "ok_counts": ok_counts, "files": paths,
                    "instance": self.group_names(tag)[0], "d3_dir": self.baseline_dir}
        if not agreement["bodies_match_d3"]:
            agreement["complete"] = False  # the compare would not be like for like
        path = f"{self.logs}/prelim-E3-{tag}-probe-compare-{self.ts}.json"
        write_atomic(path, json.dumps(decision, indent=1, sort_keys=True) + "\n")
        decision["compare_file"] = path
        self.mark(step, decision)
        log(f"{tag} probe: S2 vs D3 S2 {agreement['identical_s2']}/{PROBE_N}, S3 vs D3 S3 {agreement['identical_s3']}/{PROBE_N}"
            f" agrees={agreement['agrees']}")
        return decision

    # --- one worst-case pass on all four instances
    def run_pass(self, tag: str, index: int, seed: str, logs: dict, dead_on_resume: bool) -> dict:
        step = f"{tag}-pass{index}"
        prior = self.done(step)
        if prior:
            return prior["decision"]
        names = self.group_names(tag)
        prefix = self.prefix_of(tag, index)
        urls = self.urls()
        if dead_on_resume:
            running, scans = self.scan_group(tag, logs)
            died = pc.EndpointDied(next((n for n in names if n not in running), names[0]),
                                   {"dead_instances": [n for n in names if n not in running], "found_on_resume": True})
            row = pass_record(tag, index, seed, names, None, died, running, scans)
            row["scans"] = scans
        else:
            self.ops.patch_jobs([self.sweep_entry(tag, index)])
            self.set_current(step)
            label = (f"PRELIMINARY E3 {tag} worst case pass {index} at {LEVEL}/instance, 4 x TP2, not the recorded locked "
                     "baseline")
            try:
                measure = self.ops.measure_level(step, prefix, LEVEL, urls, MIN_TOKENS, label, names=names, seed=seed)
                died = None
            except pc.EndpointDied as exc:
                measure, died = None, exc
                try:
                    self.ops.flush(prefix)
                except Exception:
                    pass
            running, scans = self.scan_group(tag, logs)
            row = pass_record(tag, index, seed, names, measure, died, running, scans)
            row["scans"] = {n: {k: v for k, v in s.items() if k != "lines"} for n, s in scans.items()}
            if died is not None:
                row["died_detail"] = died.detail
        self.write_pass_summary(prefix, row)
        self.mark(step, row)
        log(f"{step}: {row['outcome']} preemptions={row['preemptions_total']} "
            f"per_instance={[(e['name'][-1], e['preemptions_delta'], e['kv_peak'], e['alive']) for e in row['instances']]} "
            f"exits={[e['name'] for e in row['engine_exits']]}")
        return row

    def write_pass_summary(self, prefix: str, row: dict) -> None:
        """Same schema as the other chains' sweep summaries so the watchdog `sweep` kind reads it."""
        level = {k: v for k, v in row.items() if k not in ("scans", "died_detail")}
        level["concurrency_per_instance"] = LEVEL
        level["min_hbm_free_mib_per_card"] = row.get("min_hbm_free_mib")
        doc = {"label": f"PRELIMINARY E3 {row['arm']} pass {row['pass']} (4 x TP2, worst case at {LEVEL}/instance), "
                        "not the recorded locked baseline", "synthetic": True, "dataset_rows": 0, "seed": row["seed"],
               "done": True, "stopped_at_level": LEVEL if row["outcome"] == "exit" else None, "levels": [level]}
        write_atomic(f"{prefix}.json", json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")

    def run_group(self, tag: str, seeds: tuple, probe: bool) -> dict:
        """Launch + health + (probe) + passes for one group. Returns the group record (also checkpointed)."""
        prior = self.done(f"{tag}-group")
        if prior:
            return prior["decision"]
        extra = EXTRA[tag]
        names = self.group_names(tag)
        rec: dict = {"tag": tag, "extra_args": extra, "startup_failed": False, "passes": [], "probe": None,
                     "containers": names}
        logs = self.launch_group(tag, extra)
        rec["server_logs"] = logs
        started = self.wait_group(tag, extra, logs)
        if not started["ok"]:
            rec.update(startup_failed=True, startup_reason=started["reason"],
                       startup_scans={n: {k: v for k, v in s.items() if k != "lines"} for n, s in started["scans"].items()})
            self.mark(f"{tag}-group", rec)
            return rec
        running = self.ops.running_names()
        dead_on_resume = any(n not in running for n in names)
        if probe and not dead_on_resume:
            rec["probe"] = self.probe(tag)
        for index, seed in enumerate(seeds, 1):
            row = self.run_pass(tag, index, seed, logs, dead_on_resume)
            rec["passes"].append(row)
            if row["outcome"] == "exit":
                break  # a dead instance ends the group (no step-down, no second pass)
            running = self.ops.running_names()
            dead_on_resume = any(n not in running for n in names)
        self.mark(f"{tag}-group", rec)
        write_atomic(self.dir / f"{tag}.group.json", json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
        return rec

    def stop_group(self, tag: str) -> None:
        step = f"{tag}-stop"
        if self.done(step):
            return
        self.set_current(step)
        names = self.group_names(tag)
        self.ops.patch_jobs([], remove=[f"vllm-prelim-E3-{tag}-{i}" for i in range(len(names))])
        self.stop_and_wait(names, f"{tag} group")
        used = self.ops.cards_below(list(range(8)))
        self.mark(step, {"stopped": names, "hbm_used_mib_after": used})

    # --- arm A restore
    def restore_arm_a(self) -> None:
        if self.done("restore-arm-a"):
            return
        self.set_current("restore-arm-a")
        e3 = [n for t in TAGS for n in self.group_names(t)]
        self.ops.patch_jobs([], remove=[f"vllm-prelim-E3-{t}-{i}" for t in TAGS for i in range(len(self.capture))])
        self.stop_and_wait(e3, "E3 groups")
        self.ops.cards_below(list(range(8)))
        logs: dict[str, str] = {}
        running = self.ops.running_names()
        for spec in self.capture:
            if spec["name"] not in running:
                logs[spec["name"]] = self.ops.launch(spec)
        self.ops.wait_healthy(self.capture, HEALTH_TIMEOUT_S, on_tick=self.write_status)
        mismatch = []
        for spec in self.capture:
            now = self.ops.argv_of(spec["name"])
            if not (now["cmd"] == spec["cmd"] and now["cpuset"] == spec["cpuset"]
                    and now["hccl_base_port"] == spec["hccl_base_port"] and now["devices"] == spec["devices"]):
                mismatch.append(spec["name"])
        self.ops.patch_jobs(self.arm_a_entries(logs))
        self.mark("restore-arm-a", {"healthy": [s["name"] for s in self.capture], "logs": logs,
                                    "argv_identical_to_capture": not mismatch, "mismatch": mismatch})
        if mismatch:
            log(f"CHAIN-BAD arm A argv differs from the capture after relaunch: {mismatch}")

    # --- decision
    def decide(self, base: dict | None, shm: dict | None) -> dict:
        probe = ((shm or {}).get("probe") or {}).get("agreement")
        verdict = e3_verdict(shm, probe)
        base_view = None
        if base is not None:
            base_view = {"startup_failed": base.get("startup_failed", False), "startup_reason": base.get("startup_reason"),
                         "passes": [{k: p.get(k) for k in ("pass", "seed", "outcome", "preemptions_total", "kv_peak_max",
                                                           "instances", "engine_exits", "host_output_tok_s", "p10_e2e_tok_s",
                                                           "latency_max_s", "ok", "requests", "errors", "timeouts", "report")}
                                    for p in base.get("passes", [])]}
            exits = [e for p in base.get("passes", []) for e in p.get("engine_exits", [])]
            base_view["engine_exit_count"] = len(exits)
            base_view["note"] = ("informative only; a second crash strengthens E0's hypothesis, no crash does not change "
                                 "the ruling (plan 3.11 ruling 3b)")
        shm_view = None
        if shm is not None:
            shm_view = {"startup_failed": shm.get("startup_failed", False), "startup_reason": shm.get("startup_reason"),
                        "passes": [{k: p.get(k) for k in ("pass", "seed", "outcome", "preemptions_total", "kv_peak_max",
                                                          "instances", "engine_exits", "host_output_tok_s", "p10_e2e_tok_s",
                                                          "latency_max_s", "ok", "requests", "errors", "timeouts", "report")}
                                   for p in shm.get("passes", [])],
                        "probe": probe, "probe_compare_file": ((shm.get("probe") or {}).get("compare_file"))}
        decision = {"label": "prelim-E3", "verdict": verdict["verdict"], "reasons": verdict["reasons"],
                    "rule": "ADOPTED iff both shm passes have zero engine exits AND the warm sequential probe agrees "
                            f"{PROBE_N}/{PROBE_N} with the D3 baseline warm runs S2 and S3 (plan 3.11 ruling 3b)",
                    "level_per_instance": LEVEL, "min_tokens": MIN_TOKENS, "seeds": list(SEEDS),
                    "baseline": base_view, "shm": shm_view}
        path = f"{self.logs}/prelim-E3-{self.ts}.json"
        write_atomic(path, json.dumps(decision, indent=1, sort_keys=True, default=str) + "\n")
        decision["result_file"] = path
        with self.lock:
            self.status["decisions"]["E3"] = decision
        self.write_status()
        log(f"E3 verdict: {verdict['verdict']} {verdict['reasons']}")
        return decision

    def run(self) -> int:
        self.heartbeat()
        failure = None
        try:
            self.ops.patch_jobs(self.chain_entries())
            self.gate()
            self.ops.start_sampler()
            self.stop_arm_a()
            base = self.run_group("base", (SEEDS[0],), probe=False)
            self.stop_group("base")
            shm = self.run_group("shm", SEEDS, probe=True)
            self.ops.stop_sampler()
            decision = self.decide(base, shm)
            self.restore_arm_a()
            if decision["verdict"] == "INCOMPLETE":
                self.status.update(state="failed", current_step=None, reason=f"INCOMPLETE: {decision['reasons']}")
                self.write_status()
                log(f"CHAIN-BAD E3 INCOMPLETE: {decision['reasons']}")
                return 6
            self.status.update(state="done", current_step=None, reason=None)
            self.write_status()
            log("CHAIN DONE")
            return 0
        except pc.ChainStop as stop:
            failure = ("failed", str(stop))
            log(f"CHAIN-BAD stopped at {self.status['current_step']}: {stop}")
            code = 2
        except KeyboardInterrupt:
            self.status.update(state="interrupted", reason="signal; containers left as they are; rerun with --resume")
            self.write_status()
            log("CHAIN-BAD interrupted by a signal; containers left running; rerun with --resume")
            return 3
        except Exception as error:
            failure = ("failed", f"{type(error).__name__}: {error}")
            log(f"CHAIN-BAD unexpected {type(error).__name__}: {error}")
            code = 4
        finally:
            self._hb_stop.set()
            try:
                self.ops.stop_sampler()
            except Exception:
                pass
        # failure path: restore arm A if this chain stopped it (never when the gate or the arm B check refused)
        if self.touched or self.done("stop-arm-a"):
            try:
                self.restore_arm_a()
            except Exception as error:
                failure = ("failed", f"{failure[1]}; arm A restore failed: {type(error).__name__}: {error}")
                log(f"CHAIN-BAD could not restore arm A: {type(error).__name__}: {error}")
        self.status.update(state=failure[0], reason=failure[1])
        self.write_status()
        return code


# ---------------------------------------------------------------- dry-run operations (CPU only, no side effects)
FAKE_CAPTURE = [dict(s) for s in ec.FAKE_CAPTURE]


def synth_results(mismatch: tuple = ()) -> list[dict]:
    return [{"index": i, "status": "ok", "reasoning": "different" if i in mismatch else f"r{i}", "content": "",
             "finish_reason": "length", "completion_tokens": 64, "logprobs": [-0.1, -0.2]} for i in range(PROBE_N)]


class FakeOps:
    """Scripted operations for the rehearsal and the unit tests: nothing touches docker, a card, the network or the
    registry. Scenario keys: gate_waits (reads before the gate opens), gate_state, gate_missing, arm_b_up,
    base_exit (pass die), shm_exit (pass number 1 or 2), shm_start_fail, base_start_fail, probe_mismatch (set of
    indices), busy_arm_a, resume_dead (names missing at resume)."""

    def __init__(self, scenario: dict | None = None, root: Path | None = None) -> None:
        self.sc = dict(scenario or {})
        self.root = root
        self.running = {f"saea-arm-a-{i}" for i in range(4)} | ({"saea-arm-b-0"} if self.sc.get("arm_b_up") else set())
        self.calls: list[str] = []
        self.gate_reads = 0
        self.extra: dict[str, str] = {}
        self.jobs: set[str] = set()
        self.measured: list[tuple] = []
        self.fatal: set[str] = set()

    def note(self, text: str) -> None:
        self.calls.append(text)
        log(f"[dry-run] {text}")

    def start_sampler(self) -> None: self.note("sampler start")
    def stop_sampler(self) -> None: self.note("sampler stop")
    def flush(self, prefix) -> None: pass

    def find_gate(self):
        return None if self.sc.get("gate_missing") else "/dry/p3-status.json"

    def gate_status(self, path):
        self.gate_reads += 1
        if self.gate_reads <= self.sc.get("gate_waits", 2):
            return {"state": "running", "current_step": "p3-L48", "pid": 4242}
        return {"state": self.sc.get("gate_state", "done"), "pid": 4242, "reason": None}

    def pid_alive(self, pid) -> bool:
        return self.gate_reads <= self.sc.get("gate_waits", 2) + self.sc.get("gate_pid_extra", 0)

    def running_names(self) -> set[str]: return set(self.running)
    def running_requests(self, url) -> float: return float(self.sc.get("busy_arm_a", 0))

    def patch_jobs(self, upsert, remove=(), set_log=None) -> None:
        self.note(f"jobs upsert={[j['id'] for j in upsert]} remove={list(remove)}")
        self.jobs |= {j["id"] for j in upsert}
        self.jobs -= set(remove)

    def stop_containers(self, names) -> None:
        self.note(f"docker stop {names}")
        self.running -= set(names)

    def cards_below(self, cards, timeout=0) -> dict:
        self.note(f"cards {cards} below 4096 MiB")
        return {c: 100 for c in cards}

    def launch_e(self, name, spec, extra) -> str:
        self.note(f"launch_e {name} cards {spec['cards']} cpuset {spec['cpuset']} hccl {spec['hccl_base_port']} extra {extra!r}")
        self.running.add(name)
        self.extra[name] = extra
        return f"/dry/{name}.log"

    def launch(self, spec) -> str:
        self.note(f"launch arm A {spec['name']} cards {spec['cards']} (npu_serve.sh, baseline flags)")
        self.running.add(spec["name"])
        return f"/dry/{spec['name']}.log"

    def wait_healthy(self, specs, timeout, on_tick=None) -> dict:
        tag = "shm" if any("-shm-" in s["name"] for s in specs) else "base" if any("-base-" in s["name"] for s in specs) else "A"
        if self.sc.get(f"{tag}_start_fail"):
            for s in specs:
                self.running.discard(s["name"])
            self.fatal |= {s["name"] for s in specs}
            raise pc.ChainStop(f"{specs[0]['name']} exited during startup (see its log)")
        self.note(f"healthy {[s['name'] for s in specs]}")
        return {s["name"]: True for s in specs}

    def argv_of(self, name) -> dict:
        index = int(name.rsplit("-", 1)[1])
        spec = FAKE_CAPTURE[index]
        extra = self.extra.get(name, "")
        return {"cmd": list(spec["cmd"]) + (extra.split() if extra else []), "cpuset": spec["cpuset"],
                "hccl_base_port": spec["hccl_base_port"], "devices": spec["devices"]}

    def log_confirms(self, path, regex) -> bool: return True

    def scan_logs(self, paths) -> dict:
        out = {}
        for name in paths:
            bad = name in self.fatal
            out[name] = {"fatal_engine_error": bad, "mm_embeddings_assertion": bad and self.sc.get("assertion", True),
                         "execute_mm_encoder": bad, "died_request_ids": ["chatcmpl-sweep-c1-w1-t1-k2-ab"] if bad else [],
                         "k2_confirmed": bad, "lines": [], "server_log": paths[name]}
        return out

    def probe_send(self, url, out, bodies_file) -> None:
        mismatch = tuple(self.sc.get("probe_mismatch", ()))
        key = Path(out).name
        results = synth_results(mismatch if ("-S2-" in key or "-S3-" in key) else ())
        self.note(f"probe send {key}")
        write_atomic(out, json.dumps({"url": url, "mode": "sequential", "bodies_sha256": "b0d1", "n": PROBE_N,
                                      "results": results}) + "\n")

    def measure_level(self, step, prefix, level, urls, min_tokens, label, names=None, seed=None) -> dict:
        self.note(f"measure {step} level {level} seed {seed} min_tokens {min_tokens} urls {urls}")
        self.measured.append((step, level, seed, min_tokens))
        tag = "shm" if "-shm-" in names[0] else "base"
        index = int(step.rsplit("pass", 1)[1])
        kill = (tag == "base" and self.sc.get("base_exit") == index) or (tag == "shm" and self.sc.get("shm_exit") == index)
        if kill:
            self.running.discard(names[1])
            self.fatal.add(names[1])
            raise pc.EndpointDied(names[1], {"dead_instances": [names[1]], "preemptions_first_polled": [0.0, 1.0, 2.0, 3.0],
                                             "preemptions_last_polled": [4.0, 9.0, 2.0, 3.0],
                                             "kv_peak_polled": [1.0, 1.0, 0.8, 0.7], "client_exit_code": -9})
        pre = 0.0 if tag == "shm" and not self.sc.get("shm_preempt") else 5.0
        before = [{"preemptions": 1.0} for _ in urls]
        after = [{"preemptions": 1.0 + pre} for _ in urls]
        summary = {"requests": 3 * level * len(urls), "ok": 3 * level * len(urls), "errors": 0, "timeouts": 0, "wall_s": 1000.0,
                   "output_tok_s": 2000.0, "e2e_tok_s": {"p10": 12.0}, "latency_s": {"max": 300.0}, "instances": len(urls),
                   "metrics_before": before, "metrics_after": after,
                   "gauges": [{"kv_cache_usage_perc": {"max": 0.9}} for _ in urls]}
        return {"summary": summary, "report": f"{Path(prefix).name}-L{level}.json", "report_sha256": "0" * 64,
                "min_hbm_free_mib": 4100, "poll_kv_peak": [0.91] * len(urls), "names": names}


def seed_baseline_dir(path: Path) -> str:
    """Synthetic D3 baseline files for the rehearsal and the tests."""
    path.mkdir(parents=True, exist_ok=True)
    for key in ("S2", "S3"):
        write_atomic(path / f"{key}.json", json.dumps({"mode": "sequential", "bodies_sha256": "b0d1", "n": PROBE_N,
                                                       "results": synth_results()}) + "\n")
    write_atomic(path / "bodies.json", "[]\n")
    return str(path)


def parse_scenario(text: str) -> dict:
    sc: dict = {}
    for item in filter(None, text.split(",")):
        key, _, val = item.partition("=")
        if key in ("shm_exit", "base_exit", "gate_waits", "busy_arm_a"):
            sc[key] = int(val)
        elif key == "probe_mismatch":
            sc.setdefault("probe_mismatch", set()).add(int(val))
        elif key == "gate_state":
            sc[key] = val
        else:
            sc[key] = True
    return sc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ts", help="UTC stamp used in every file name (default: now)")
    ap.add_argument("--chain-dir", help="existing chain directory (with --resume)")
    ap.add_argument("--resume", action="store_true", help="skip finished steps of an existing chain directory")
    ap.add_argument("--dry-run", action="store_true", help="rehearse with scripted operations (CPU only)")
    ap.add_argument("--scenario", default="", help="dry-run only: comma list of shm_exit=N, base_exit=N, shm_start_fail, "
                                                    "base_start_fail, probe_mismatch=I, gate_missing, arm_b_up, gate_waits=N")
    ap.add_argument("--gate-file", default="auto",
                    help="chain-status.json of the P3 chain; 'auto' = the newest prelim-chain-p3-*/ (waits up to 600 s for it)")
    ap.add_argument("--arm-a-launch", default=DEFAULT_CAPTURE, help="captured arm A argv (copied into the chain dir)")
    ap.add_argument("--baseline-dir", default=D3_DIR, help="D3 run directory holding bodies.json, S2.json, S3.json")
    ap.add_argument("--out-root", help="dry-run only: output root (default: a temporary directory)")
    args = ap.parse_args()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if args.dry_run:
        root = Path(args.out_root or tempfile.mkdtemp(prefix="prelim-E3-dry-"))
        ts = args.ts or "DRYRUN"
        chain_dir = root / f"prelim-E3-chain-{ts}"
        chain_dir.mkdir(parents=True, exist_ok=True)
        baseline = seed_baseline_dir(root / "D3")
        log(f"dry-run output under {root}")
        chain = E3Chain(FakeOps(parse_scenario(args.scenario), root), chain_dir, ts, str(root), FAKE_CAPTURE,
                        None if args.gate_file == "auto" else args.gate_file, baseline_dir=baseline,
                        sleep=lambda s: None)
        code = chain.run()
        print(json.dumps(read_json(chain_dir / "chain-status.json")["decisions"], indent=1, sort_keys=True))
        return code
    hostname = Path("/etc/hostname").read_text().strip()
    if hostname != pc.EXPECTED_HOSTNAME:
        sys.exit(f"refusing to run on {hostname!r}: this chain belongs to host 181 ({pc.EXPECTED_HOSTNAME})")
    if args.resume:
        chain_dir = Path(args.chain_dir or sys.exit("--resume needs --chain-dir"))
        ts = read_json(chain_dir / "config.json")["ts"]
    else:
        ts = args.ts or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        chain_dir = Path(args.chain_dir or f"{LOGS}/prelim-E3-chain-{ts}")
        if (chain_dir / "config.json").exists():
            sys.exit(f"{chain_dir} already holds a chain; use --resume")
        for name in ("S2.json", "S3.json", "bodies.json"):
            if not (Path(args.baseline_dir) / name).exists():
                sys.exit(f"baseline file missing: {Path(args.baseline_dir) / name}")
        captured = read_json(args.arm_a_launch)
        if not captured:
            sys.exit(f"needs a readable arm A capture: {args.arm_a_launch}")
        chain_dir.mkdir(parents=True, exist_ok=True)
        ec.chmod_quiet(chain_dir, 0o777)
        write_atomic(chain_dir / "arm-a-launch.json", json.dumps(captured, indent=2) + "\n")
        write_atomic(chain_dir / "config.json", json.dumps(
            {"ts": ts, "created": utc_now(), "script": str(Path(__file__).resolve()), "gate_file": args.gate_file,
             "arm_a_launch_source": args.arm_a_launch, "baseline_dir": args.baseline_dir}) + "\n")
    ec.chmod_quiet(chain_dir / "config.json", 0o666)
    (chain_dir / "steps").mkdir(exist_ok=True)
    ec.chmod_quiet(chain_dir / "steps", 0o777)
    config = read_json(chain_dir / "config.json")
    capture = read_json(chain_dir / "arm-a-launch.json")["instances"]
    gate = None if config["gate_file"] == "auto" else config["gate_file"]
    log(f"E3 chain {chain_dir.name} pid {os.getpid()} resume={args.resume} gate {config['gate_file']}")
    return E3Chain(E3Real(args.resume), chain_dir, ts, LOGS, capture, gate, baseline_dir=config["baseline_dir"]).run()


if __name__ == "__main__":
    sys.exit(main())
