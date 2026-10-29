#!/usr/bin/env python3
"""E1 / E2 crash experiments on host 181 (plan section 3.8, decision 2): one detached, checkpointed, resumable run.

Question. Can the preempt-and-resume crash of arm A (`AssertionError ... len(mm_embeddings)=0` in `_execute_mm_encoder`,
bad-case note 2026-10-29-arm-a-exit-under-worst-case.md; source path in the E0 note) be reproduced on ONE TP2 instance,
and does `--mm-processor-cache-type shm` (E0's top knob) prevent it?

  gate   waits (reads the P2b chain-status.json every GATE_POLL_S = 120 s at most) until that chain is FINAL (state done /
         failed / interrupted) and its process is gone. No card work happens before that.
  E1     cards 0,1, port 8200: stops saea-arm-a-0, launches ONE instance with the baseline argv captured from arm A
         instance 0 (same CPUSET, HCCL base port, flags), 16 fixed requests twice (determinism floor), then the worst case
         (--min-tokens 4096, seed 20261028, one pass each) at 48, 56, 56 per instance. Reproduced = the
         `_execute_mm_encoder` assertion recurs (log evidence + died request ids; a `-k2` id confirms the mechanism).
         3 completed passes with >= 10 preemptions each and no exit = "not reproduced in isolation".
  E2     cards 6,7, port 8203, in parallel: stops saea-arm-a-3, launches instance 3's captured argv plus
         EXTRA_ARGS="--mm-processor-cache-type shm" (4 GB default size), the same 16 requests (compared byte for byte with
         E1's), then 3 passes at 56 (64 when the previous pass had < 10 preemptions). Accepted = zero exits in 3 passes
         that each carry >= 10 preemptions. If shm fails (exit or does not start): diagnostic arm
         EXTRA_ARGS="--no-enable-prefix-caching", one pass at 56, labelled DIAGNOSTIC ONLY.
  after  every track stops its E container and relaunches its saea-arm-a-N from the captured argv (original npu_serve.sh),
         waits until healthy and re-registers it with the watchdog, so arm A is whole for P3.

Rails: no dataset row, no exposure entry, no readiness flip, no commit, no push, never a busy card, never --restart; only
containers named saea-prelim-e* and saea-arm-a-0 / saea-arm-a-3 are touched. Preliminary only: synthetic load.
The vLLM server flags are never changed except by EXTRA_ARGS through the COPY launcher npu_serve_e.sh.

usage (on host 181, detached):
  nohup setsid python3 e_crash_chain.py [--ts <UTC stamp>] < /dev/null > <chain-dir>/chain.log 2>&1 &
  resume: python3 e_crash_chain.py --resume --chain-dir <chain-dir>      (finished steps are skipped)
  dry-run (CPU only, no docker, no card, no network): python3 e_crash_chain.py --dry-run [--scenario NAME]
Layout: <logs>/prelim-E-chain-<ts>/{config.json,chain-status.json,chain.log,steps/}; per track
<logs>/prelim-E1-<ts>-*.json and prelim-E2-<ts>-*.json (eq outputs, crash evidence, track result);
sweep reports <logs>/sweep-prelim-E{1,2}-<arm>-pass<n>-<ts>-L<level>.json (+ .stderr.log, -npu-smi.json, summary).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
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

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prelim_chain as pc  # noqa: E402  (helpers reused: Liveness, per_instance_stats, RealOps, http_get, prom_sum, ...)

ROOT, LOGS, TREE, MODEL, IMAGE = pc.ROOT, pc.LOGS, pc.TREE, pc.MODEL, pc.IMAGE
SCRIPTS = Path(__file__).resolve().parent
NPU_SERVE_E = str(SCRIPTS / "npu_serve_e.sh")
FLAGCHECK = str(SCRIPTS / "e_flagcheck.py")
FLAGCHECK_CPUSET = "96-127"  # no instance and no client is pinned there
DEFAULT_CAPTURE = f"{LOGS}/prelim-chain-20261029T094933Z/arm-a-launch.json"
DEFAULT_GATE = f"{LOGS}/prelim-chain-p2b-20261029T105349Z/chain-status.json"
GATE_POLL_S = 120  # the owner asked for at most one read of the gate file every 2 minutes
GATE_FINAL = ("done", "failed", "interrupted")
MIN_TOKENS = 4096
SEED_BASE = 20261028
MIN_PREEMPT = 10  # preemptions per pass that count as "stressed"
MAX_LEVEL = 64  # the --max-num-seqs ceiling of the captured argv
STEP_UP = 8
E1_PLAN = [48, 56, 56]
E2_PLAN = [56, 56, 56]
DIAG_PLAN = [56]
SHM_ARGS = "--mm-processor-cache-type shm"
NOPREFIX_ARGS = "--no-enable-prefix-caching"
E_OURS = re.compile(r"^saea-prelim-e[12](-[a-z]+)?$")
ARM_A_OURS = re.compile(r"^saea-arm-a-[03]$")
EQ_SEED = 20261029
EQ_DECODING_SEED = 777
EQ_MAX_TOKENS = 64
HEALTH_TIMEOUT_S = pc.A_HEALTH_TIMEOUT_S
REQ_ID = re.compile(r"chatcmpl-sweep-c(\d+)-w(\d+)-t(\d+)-k(\d+)-[0-9a-f]+")
TRACKS = {
    # arm = (tag, EXTRA_ARGS, level plan, role, escalate); escalate raises the next level after a pass with < 10 preemptions
    "E1": {"arm_a": "saea-arm-a-0", "capture_index": 0, "arms": [("base", "", E1_PLAN, "candidate", False)]},
    "E2": {"arm_a": "saea-arm-a-3", "capture_index": 3,
           "arms": [("shm", SHM_ARGS, E2_PLAN, "candidate", True),
                    ("noprefix", NOPREFIX_ARGS, DIAG_PLAN, "diagnostic", True)]},
}
CONFIRM_RE = {"--mm-processor-cache-type shm": r"'mm_processor_cache_type': 'shm'",
              "--no-enable-prefix-caching": r"'enable_prefix_caching': False"}  # the server's `non-default args` line
CONTAINER = {"E1-base": "saea-prelim-e1", "E2-shm": "saea-prelim-e2-shm", "E2-noprefix": "saea-prelim-e2-noprefix"}

utc_now, log, write_atomic, read_json = pc.utc_now, pc.log, pc.write_atomic, pc.read_json


# ---------------------------------------------------------------- pure logic (unit tested)
def gate_final(status: dict | None, pid_alive: bool) -> tuple[bool, str]:
    """The P2b chain is FINAL when its state is done/failed/interrupted AND its process has exited."""
    if not isinstance(status, dict):
        return False, "chain-status.json unreadable"
    state = status.get("state")
    if state not in GATE_FINAL:
        return False, f"state {state!r} step {status.get('current_step')!r}"
    if pid_alive:
        return False, f"state {state!r} but pid {status.get('pid')} is still alive"
    return True, f"state {state!r}"


def pass_seed(index: int) -> str:
    """Seed of pass `index` (1-based): 20261028, 20261029, ... so repeated passes at one level never replay a cache."""
    return str(SEED_BASE + index - 1)


def next_level(plan: list[int], index: int, prev: dict | None, escalate: bool = True) -> int:
    """Level of pass `index` (0-based): the plan, and with `escalate` raised by STEP_UP (to MAX_LEVEL) after a completed
    pass that carried fewer than MIN_PREEMPT preemptions, so the next pass stresses the cache more (E2 only)."""
    level = plan[index]
    if escalate and prev and prev.get("outcome") == "completed" and (prev.get("preemptions") is None
                                                        or prev["preemptions"] < MIN_PREEMPT):
        level = min(MAX_LEVEL, max(level, prev["level"] + STEP_UP))
    return level


def _stressed(p: dict) -> bool:
    return p.get("preemptions") is not None and p["preemptions"] >= MIN_PREEMPT


def e1_verdict(passes: list[dict], expected: int = 3) -> dict:
    exits = [p for p in passes if p["outcome"] == "exit"]
    if exits:
        ev = exits[0].get("evidence") or {}
        if ev.get("mm_embeddings_assertion") and ev.get("execute_mm_encoder"):
            return {"verdict": "reproduced", "pass": exits[0]["pass"], "level": exits[0]["level"],
                    "died_request_ids": ev.get("died_request_ids", []), "k2_confirmed": ev.get("k2_confirmed", False)}
        return {"verdict": "exit_other_cause", "pass": exits[0]["pass"], "level": exits[0]["level"],
                "died_request_ids": ev.get("died_request_ids", []), "k2_confirmed": ev.get("k2_confirmed", False)}
    if len(passes) < expected:
        return {"verdict": "incomplete", "passes_done": len(passes)}
    low = [p["pass"] for p in passes if not _stressed(p)]
    if not low:
        return {"verdict": "not reproduced in isolation", "preemptions": [p["preemptions"] for p in passes]}
    return {"verdict": "inconclusive_low_preemptions", "passes_below_threshold": low,
            "preemptions": [p["preemptions"] for p in passes]}


def e2_verdict(passes: list[dict], expected: int = 3) -> dict:
    exits = [p for p in passes if p["outcome"] == "exit"]
    if exits:
        ev = exits[0].get("evidence") or {}
        return {"verdict": "failed_exit", "pass": exits[0]["pass"], "level": exits[0]["level"],
                "mm_embeddings_assertion": bool(ev.get("mm_embeddings_assertion")),
                "died_request_ids": ev.get("died_request_ids", [])}
    if len(passes) < expected:
        return {"verdict": "incomplete", "passes_done": len(passes)}
    low = [p["pass"] for p in passes if not _stressed(p)]
    if not low:
        return {"verdict": "accepted", "preemptions": [p["preemptions"] for p in passes]}
    return {"verdict": "inconclusive_low_preemptions", "passes_below_threshold": low,
            "preemptions": [p["preemptions"] for p in passes]}


def crash_evidence_from_lines(lines: list[str]) -> dict:
    """What the server log says about a death: the assertion lines, the scheduler dump's fatal request ids
    (`scheduled_encoder_inputs` and `resumed_req_ids`), whether any id is a `-k2` contact. Pure function of the lines."""
    hits: list[dict] = []
    ev: dict = {"mm_embeddings_assertion": False, "execute_mm_encoder": False, "fatal_engine_error": False,
                "worker_failed": False, "lines": []}
    first_dump = None
    for number, line in enumerate(lines, 1):
        for key, needle in (("mm_embeddings_assertion", "len(mm_embeddings)="), ("execute_mm_encoder", "_execute_mm_encoder"),
                            ("fatal_engine_error", "EngineCore encountered a fatal error"),
                            ("worker_failed", "Worker failed")):
            if needle in line:
                if not ev[key]:
                    ev["lines"].append({"line": number, "text": line.strip()[:500]})
                ev[key] = True
        if first_dump is None and "Dumping scheduler output" in line:
            first_dump = (number, line)
    encoder_ids: list[str] = []
    resumed_ids: list[str] = []
    if first_dump:
        number, line = first_dump
        m = re.search(r"scheduled_encoder_inputs=\{(.*?)\}", line)
        if m:
            encoder_ids = [x.group(0) for x in REQ_ID.finditer(m.group(1))]
        m = re.search(r"resumed_req_ids=\{(.*?)\}", line)
        if m:
            resumed_ids = [x.group(0) for x in REQ_ID.finditer(m.group(1))]
        ev["dump_line"] = number
    ev["scheduled_encoder_input_ids"] = encoder_ids
    ev["resumed_req_ids"] = resumed_ids
    died = encoder_ids or resumed_ids
    ev["died_request_ids"] = encoder_ids
    ev["k2_confirmed"] = bool(encoder_ids) and all(REQ_ID.search(i).group(4) == "2" for i in encoder_ids)
    ev["contact_index_of_died"] = [int(REQ_ID.search(i).group(4)) for i in died]
    return ev


def compare_eq(a: list[dict], b: list[dict]) -> dict:
    """Byte-for-byte comparison of two lists of recorded completions (same order). Logprob stats are secondary."""
    n = min(len(a), len(b))
    exact, diffs, mismatched = 0, [], []
    for i in range(n):
        ka = (a[i].get("reasoning"), a[i].get("content"), a[i].get("finish_reason"), a[i].get("completion_tokens"))
        kb = (b[i].get("reasoning"), b[i].get("content"), b[i].get("finish_reason"), b[i].get("completion_tokens"))
        if ka == kb and a[i].get("status") == "ok" and b[i].get("status") == "ok":
            exact += 1
        else:
            mismatched.append(i)
        la, lb = a[i].get("logprobs") or [], b[i].get("logprobs") or []
        if la and len(la) == len(lb):
            diffs.append(max(abs(x - y) for x, y in zip(la, lb)))
    return {"requests": n, "count_a": len(a), "count_b": len(b), "byte_identical": exact, "mismatched_indices": mismatched,
            "max_abs_logprob_diff": max(diffs) if diffs else None}


def eq_bodies(ss, model: str) -> list[dict]:
    """16 fixed deterministic requests: 8 trajectories x 2 contacts (contact 1 reuses contact 0's audio and word prefix,
    so the multimodal cache hit path is exercised). temperature 0, fixed seed, short max_tokens."""
    bodies = []
    for i in range(8):
        trajectory = ss.Trajectory(i, f"eq:{EQ_SEED}:{i}", round(2.0 + 0.5 * i, 3), (0, 0))
        audio, words = trajectory.audio_b64(), trajectory.words(400)
        for contact, n_words in enumerate((120 + 10 * i, 180 + 10 * i)):
            bodies.append({"model": model, "messages": ss.contact_messages(audio, words, n_words, contact, 2, 0),
                           "temperature": 0, "seed": EQ_DECODING_SEED, "max_tokens": EQ_MAX_TOKENS,
                           "logprobs": True, "chat_template_kwargs": {"enable_thinking": True}})
    return bodies


def load_sweep_module():
    """The study's synthetic_sweep.py, imported read-only by path (no bytecode written, nothing in the tree changes)."""
    path = f"{TREE}/scripts/serving/synthetic_sweep.py"
    spec = importlib.util.spec_from_file_location("synthetic_sweep_ro", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def post_completion(url: str, body: dict, timeout: float = 900.0) -> dict:
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            doc = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        return {"status": "error", "error": repr(error)[:300]}
    choice = (doc.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = ((choice.get("logprobs") or {}).get("content")) or []
    return {"status": "ok", "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
            "content": message.get("content") or "", "finish_reason": choice.get("finish_reason"),
            "completion_tokens": (doc.get("usage") or {}).get("completion_tokens"),
            "logprobs": [t.get("logprob") for t in content if isinstance(t, dict)]}


def pass_row(measure: dict, level: int, seed: str, index: int, arm: str, name: str) -> dict:
    s = measure["summary"]
    stats = pc.per_instance_stats(s, [name], measure.get("poll_kv_peak"))[0]
    return {"pass": index, "arm": arm, "level": level, "seed": seed, "outcome": "completed",
            "requests": s["requests"], "ok": s["ok"], "errors": s["errors"], "timeouts": s["timeouts"],
            "wall_s": s["wall_s"], "host_output_tok_s": s["output_tok_s"], "p10_e2e_tok_s": s["e2e_tok_s"]["p10"],
            "latency_max_s": s["latency_s"]["max"], "preemptions": stats["preemptions_delta"],
            "kv_peak": stats["kv_peak"], "min_hbm_free_mib": measure["min_hbm_free_mib"],
            "report": measure["report"], "report_sha256": measure["report_sha256"],
            "concurrency_per_instance": level, "output_tok_s": s["output_tok_s"],
            "min_hbm_free_mib_per_card": measure["min_hbm_free_mib"]}


# ---------------------------------------------------------------- real operations (host 181 only)
class EReal(pc.RealOps):
    """pc.RealOps (sampler, docker helpers, launch of arm A, watchdog patching) plus the E-specific operations."""

    jobs_lock = threading.Lock()

    def __init__(self, resume: bool = False) -> None:
        super().__init__(LOGS, "e", resume)

    @staticmethod
    def _check_ours(names: list[str]) -> None:
        bad = [n for n in names if not (E_OURS.match(n) or ARM_A_OURS.match(n))]
        if bad:
            raise pc.ChainStop(f"refusing to touch container(s) that are not ours: {bad}")

    def patch_jobs(self, upsert, remove=(), set_log=None) -> None:
        with self.jobs_lock:
            super().patch_jobs(upsert, remove, set_log)

    def running_names(self) -> set[str]:
        names = self.docker_names()
        return set() if names is None else names

    def gate_status(self, path: str):
        return read_json(path)

    def pid_alive(self, pid) -> bool:
        return isinstance(pid, int) and pid > 0 and Path(f"/proc/{pid}").exists()

    def cards_below(self, cards: list[int], timeout: float = pc.CARDS_FREE_TIMEOUT_S) -> dict:
        deadline = time.time() + timeout
        while True:
            seen = pc.read_cards()
            used = {c: seen[c][1] for c in cards if c in seen}
            if len(used) == len(cards) and all(v < pc.BUSY_MIB for v in used.values()):
                return used
            if time.time() > deadline:
                raise pc.ChainStop(f"cards {cards} not below {pc.BUSY_MIB} MiB after {timeout:.0f} s: {used}")
            time.sleep(5)

    def flag_check(self) -> dict:
        """Device-less container: does vLLM 0.30.0 parse the E2 flags? CPU only, no card, no network."""
        done = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--cpuset-cpus", FLAGCHECK_CPUSET,
             "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
             "-v", "/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro",
             "-v", f"{SCRIPTS}:/s:ro", "--entrypoint", "python3", IMAGE, "/s/e_flagcheck.py"],
            capture_output=True, text=True, timeout=900)
        rows = {}
        for line in done.stdout.splitlines():
            if line.startswith("FLAGCHECK "):
                doc = json.loads(line[len("FLAGCHECK "):])
                rows[" ".join(doc["argv"])] = doc
        shm, bogus = rows.get(SHM_ARGS), rows.get("--mm-processor-cache-type bogus")
        return {"rows": rows, "shm_accepted": bool(shm and shm["accepted"] and shm["mm_processor_cache_type"] == "shm"),
                "bogus_rejected": bool(bogus and not bogus["accepted"]),
                "noprefix_accepted": bool((rows.get(NOPREFIX_ARGS) or {}).get("accepted")),
                "exit_code": done.returncode}

    def launch_e(self, name: str, spec: dict, extra: str) -> str:
        self._check_ours([name])
        env = dict(os.environ, NAME=name, CPUSET=spec["cpuset"], HCCL_BASE_PORT=str(spec["hccl_base_port"]),
                   MAX_NUM_SEQS=str(spec["max_num_seqs"]), EXTRA_ARGS=extra)
        command = ["bash", NPU_SERVE_E, "thinking", spec["cards"], str(spec["port"]), str(spec["tp"])]
        log(f"launch NAME={name} CPUSET={spec['cpuset']} HCCL_BASE_PORT={spec['hccl_base_port']} "
            f"MAX_NUM_SEQS={spec['max_num_seqs']} EXTRA_ARGS={extra!r} {' '.join(command)}")
        done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
        if done.returncode != 0:
            raise pc.ChainStop(f"npu_serve_e.sh failed for {name}: {(done.stderr or done.stdout).strip()[-300:]}")
        match = re.search(r"log (\S+)\s*$", done.stdout.strip())
        return match.group(1) if match else ""

    def log_lines(self, path: str) -> list[str]:
        try:
            return Path(path).read_text(errors="replace").split("\n")
        except OSError:
            return []

    def crash_evidence(self, path: str) -> dict:
        time.sleep(8)  # the detached `docker logs -f` follower may lag the container's last lines
        lines = self.log_lines(path)
        ev = crash_evidence_from_lines(lines)
        ev["server_log"] = path
        ev["server_log_lines"] = len(lines)
        ev["error_tail"] = [l.strip()[:300] for l in lines[-400:] if re.search(r"ERROR|Error|Traceback", l)][-12:]
        return ev

    def log_confirms(self, path: str, regex: str) -> bool:
        pattern = re.compile(regex)
        return any(pattern.search(l) for l in self.log_lines(path)[:600])

    def eq_run(self, url: str, out_path: str) -> list[dict]:
        ss = load_sweep_module()
        results = []
        for i, body in enumerate(eq_bodies(ss, MODEL)):
            r = post_completion(url, body)
            r["index"] = i
            results.append(r)
        write_atomic(out_path, json.dumps({"url": url, "model": MODEL, "max_tokens": EQ_MAX_TOKENS,
                                           "decoding_seed": EQ_DECODING_SEED, "results": results}, indent=1) + "\n")
        return results

    def measure_pass(self, prefix: str, level: int, seed: str, url: str, name: str, label: str,
                     cards: list[int]) -> dict:
        """One worst-case pass on ONE endpoint (synthetic_sweep.py --min-tokens 4096). Raises pc.EndpointDied when the
        container or the endpoint is gone afterwards; the client is left to run to its natural end so that its report
        lists the lost requests (connection refused returns at once)."""
        out = Path(f"{prefix}-L{level}.json")
        client = f"saea-sweep-{Path(prefix).name}-L{level}"
        subprocess.run(["docker", "rm", "-f", client], capture_output=True)
        command = ["docker", "run", "--rm", "--name", client, "--network", "host", "--user",
                   f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "--cpuset-cpus", pc.CLIENT_CPUSET,
                   "-v", f"{ROOT}:{ROOT}", "-w", TREE, "-e", f"PYTHONPATH={TREE}/src",
                   "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "/opt/saea-venv/bin/python", IMAGE,
                   "scripts/serving/synthetic_sweep.py", "--base-url", url, "--model", MODEL, "--levels", str(level),
                   "--seed", seed, "--label", label, "--out", str(out), "--min-tokens", str(MIN_TOKENS)]
        poll = {"stop": threading.Event(), "kv": None, "first": None, "last": None, "last_ok": None}

        def poller() -> None:
            while not poll["stop"].is_set():
                got = pc.http_get(url + "/metrics", timeout=3)
                if got and got[0] == 200:
                    kv = pc.prom_sum(got[1], "vllm:kv_cache_usage_perc")
                    pre = pc.prom_sum(got[1], "vllm:num_preemptions_total")
                    if pre is None:
                        pre = pc.prom_sum(got[1], "vllm:num_preemptions")
                    if kv is not None:
                        poll["kv"] = kv if poll["kv"] is None else max(poll["kv"], kv)
                    if pre is not None:
                        poll["last"] = pre
                        poll["first"] = pre if poll["first"] is None else poll["first"]
                    poll["last_ok"] = utc_now()
                poll["stop"].wait(pc.POLL_SECONDS)

        thread = threading.Thread(target=poller, daemon=True)
        thread.start()
        t0 = time.time()
        log(f"{name}: client {' '.join(command[command.index('scripts/serving/synthetic_sweep.py'):])}")
        try:
            with open(f"{prefix}-L{level}.stderr.log", "w") as stderr:
                code = subprocess.run(command, stdout=stderr, stderr=subprocess.STDOUT).returncode
        except BaseException:
            subprocess.run(["docker", "rm", "-f", client], capture_output=True)
            raise
        finally:
            poll["stop"].set()
        t1 = time.time()
        thread.join(timeout=10)
        dead = pc.Liveness([(name, url)], pc.http_get, self.docker_names, pause=pc.LIVENESS_PAUSE_SECONDS).dead_instances()
        if dead:
            detail = {"dead_instances": dead, "detected_at": utc_now(), "client_exit_code": code,
                      "last_metrics_ok": poll["last_ok"], "kv_peak_polled": poll["kv"],
                      "preemptions_first_polled": poll["first"], "preemptions_last_polled": poll["last"],
                      "pass_seconds": round(t1 - t0, 1)}
            report = read_json(out)
            if report:
                bad = [r for r in report.get("requests", []) if r.get("status") != "ok"]
                detail["report_failed_requests"] = len(bad)
                detail["report_failed_ids"] = [r["request_id"] for r in bad][:300]
            raise pc.EndpointDied(name, detail)
        if code != 0 or not out.exists():
            raise pc.ChainStop(f"sweep client exited {code}; see {prefix}-L{level}.stderr.log")
        report = json.loads(out.read_text())
        summary = dict(report["levels"][0], instances=1)
        with self.lock:
            snapshot = [s for s in self.samples if "error" not in s and t0 <= s["t"] <= t1]
        free = [s["hbm_total_mib"] - used for s in snapshot for card, used in s["hbm_used_mib"].items()
                if int(card) in cards]
        mine = [dict(s, level=level, hbm_used_mib={k: v for k, v in s["hbm_used_mib"].items() if int(k) in cards},
                     aicore_pct={k: v for k, v in s["aicore_pct"].items() if int(k) in cards}) for s in snapshot]
        write_atomic(f"{prefix}-npu-smi.json", json.dumps(
            {"interval_s": pc.SAMPLE_SECONDS, "cards": cards, "samples": mine}, separators=(",", ":")) + "\n")
        return {"summary": summary, "report": out.name, "report_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "min_hbm_free_mib": min(free) if free else None, "poll_kv_peak": [poll["kv"]]}


# ---------------------------------------------------------------- the chain
class EChain:
    def __init__(self, ops, chain_dir: Path, ts: str, logs: str, capture: list[dict], gate_path: str,
                 tracks: tuple[str, ...] = ("E1", "E2"), gate_poll: float = GATE_POLL_S, sleep=time.sleep) -> None:
        self.ops, self.dir, self.ts, self.logs, self.gate_path = ops, chain_dir, ts, logs, gate_path
        self.capture = capture
        self.tracks, self.gate_poll, self.sleep = tracks, gate_poll, sleep
        self.steps_dir = chain_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.status = {"chain": chain_dir.name, "state": "running", "current_step": None, "started": utc_now(),
                       "updated": utc_now(), "pid": os.getpid(), "reason": None, "gate": {}, "decisions": {},
                       "steps": {}, "tracks": {}, "paths": {"logs": logs, "ts": ts}}
        old = read_json(chain_dir / "chain-status.json")
        if old:
            for key in ("decisions", "steps", "tracks", "gate"):
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

    def track_status(self, track: str, **fields) -> None:
        with self.lock:
            self.status["tracks"].setdefault(track, {}).update(fields)
            self.status["tracks"][track]["updated"] = utc_now()
            self.write_status()

    def set_current(self, step: str | None, track: str | None = None) -> None:
        with self.lock:
            if track:
                self.status["tracks"].setdefault(track, {})["current_step"] = step
                self.status["tracks"][track]["since"] = utc_now()
            else:
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
                log(f"heartbeat step={self.status['current_step']} tracks="
                    f"{ {t: v.get('current_step') for t, v in self.status['tracks'].items()} }")
                self.write_status()
        threading.Thread(target=loop, daemon=True).start()

    # --- watchdog registry
    def job_entries(self) -> list[dict]:
        return [{"id": "prelim-E-chain", "kind": "generic_log", "host": "npu2", "stall_s": 3600,
                 "note": "E1/E2 crash experiments (plan 3.8): gate on the P2b chain, then E1 (cards 0,1) and E2 (cards "
                         "6,7) in parallel; heartbeat every 60 s; BAD means the chain stopped (see chain-status.json)",
                 "params": {"path": str(self.dir / "chain.log"), "bad_regex": "CHAIN-BAD|Traceback",
                            "done_regex": "CHAIN DONE", "status_file": str(self.dir / "chain-status.json")}}]

    def sweep_entry(self, track: str, tag: str, index: int, prefix: str) -> dict:
        return {"id": f"sweep-prelim-{track}-{tag}-pass{index}", "kind": "sweep", "host": "npu2", "stall_s": 3600,
                "note": f"{track} {tag} pass {index}, worst case --min-tokens 4096 on ONE endpoint; preemption and "
                        "exit are the subject of the experiment, a BAD event here is expected data, not a defect",
                "params": {"run_prefix": prefix}}

    def a_entry(self, arm_name: str, port: int, log_path: str | None) -> dict:
        return {"id": f"vllm-arm-a-{arm_name.rsplit('-', 1)[1]}", "kind": "vllm_server", "host": "npu2",
                "note": f"arm A instance relaunched by prelim-E-chain-{self.ts} after its E run; left UP",
                "params": {"container": arm_name, "port": port, "log": log_path}}

    # --- gate
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
                log(f"gate: P2b chain is final ({why})")
                self.mark("gate", {"why": why, "p2b_state": (status or {}).get("state"),
                                   "p2b_reason": (status or {}).get("reason"), "waited_s": waited})
                return
            if waited % 600 == 0:
                log(f"gate: waiting, {why}")
            self.sleep(self.gate_poll)
            waited += self.gate_poll

    # --- per-track building blocks
    def prefix_of(self, track: str, tag: str, index: int) -> str:
        return f"{self.logs}/sweep-prelim-{track}-{tag}-pass{index}-{self.ts}"

    def spec_of(self, track: str) -> dict:
        return self.capture[TRACKS[track]["capture_index"]]

    def stop_arm_a(self, track: str) -> None:
        step = f"{track}-stop-arm-a"
        if self.done(step):
            return
        self.set_current(step, track)
        arm = TRACKS[track]["arm_a"]
        spec = self.spec_of(track)
        cards = [int(c) for c in spec["cards"].split(",")]
        if arm in self.ops.running_names():
            url = f"http://127.0.0.1:{spec['port']}"
            deadline = time.time() + 180
            while True:  # never stop an instance that is serving someone
                busy = self.ops.running_requests(url)
                if not busy:
                    break
                if time.time() > deadline:
                    raise pc.ChainStop(f"{arm} still has {busy:g} running request(s) after 180 s; refusing to stop it")
                self.sleep(5)
            self.ops.patch_jobs([], remove=[f"vllm-arm-a-{arm.rsplit('-', 1)[1]}"])
            self.ops.stop_containers([arm])
            deadline = time.time() + 240
            while arm in self.ops.running_names() and time.time() < deadline:
                self.sleep(3)
            if arm in self.ops.running_names():
                raise pc.ChainStop(f"{arm} still running 240 s after docker stop")
        used = self.ops.cards_below(cards)
        self.mark(step, {"stopped": arm, "cards": cards, "hbm_used_mib_after": used})

    def eq_compare(self) -> None:
        """Compare E1's first eq run with each E2 arm's eq run, whichever finishes second triggers it."""
        with self.lock:
            a = (self.done("E1-base-eq") or {}).get("decision", {}).get("results_path")
            if not a:
                return
            base = read_json(a)
            for tag in ("shm", "noprefix"):
                step = f"E2-{tag}-eq"
                prior = self.done(step)
                if not prior or self.done(f"E-eq-compare-{tag}"):
                    continue
                other = read_json(prior["decision"]["results_path"])
                if not base or not other:
                    continue
                cmp_ = compare_eq(base["results"], other["results"])
                self_cmp = (self.done("E1-base-eq") or {}).get("decision", {}).get("self_compare")
                doc = {"e1_vs_e2": cmp_, "e1_self_determinism": self_cmp, "e2_arm": tag}
                path = f"{self.logs}/prelim-E2-eq-compare-{tag}-{self.ts}.json"
                write_atomic(path, json.dumps(doc, indent=1) + "\n")
                self.mark(f"E-eq-compare-{tag}", dict(doc, path=path))
                log(f"eq compare E1 vs E2-{tag}: {cmp_['byte_identical']}/{cmp_['requests']} byte-identical")

    def run_arm(self, track: str, tag: str, extra: str, plan: list[int], role: str, escalate: bool) -> dict:
        """Launch one instance (baseline or a knob), 16 fixed requests, then the passes. Returns the arm record."""
        key = f"{track}-{tag}"
        record_path = self.dir / f"{key}.arm.json"
        prior = read_json(record_path)
        if prior and prior.get("finished"):
            return prior
        spec = self.spec_of(track)
        name = CONTAINER[key]
        url = f"http://127.0.0.1:{spec['port']}"
        label_role = "DIAGNOSTIC ONLY" if role == "diagnostic" else "candidate"
        arm = {"key": key, "container": name, "role": role, "extra_args": extra, "cards": spec["cards"],
               "port": spec["port"], "cpuset": spec["cpuset"], "hccl_base_port": spec["hccl_base_port"],
               "passes": [], "startup": None, "finished": False}
        if prior:
            arm.update({k: prior[k] for k in ("startup", "server_log") if k in prior})
        for index in range(1, len(plan) + 1):  # passes finished in an earlier run of this chain (resume)
            done = self.done(f"{key}-pass{index}")
            if done:
                arm["passes"].append(done["decision"])
                if done["decision"].get("outcome") == "exit":
                    break

        def save() -> None:
            write_atomic(record_path, json.dumps(arm, indent=1, sort_keys=True, default=str) + "\n")

        if arm["passes"] and (arm["passes"][-1].get("outcome") == "exit" or len(arm["passes"]) >= len(plan)):
            return self.finalize_arm(arm, track, plan, role, save)  # nothing left to run: no relaunch on resume

        # --- launch and health
        launch = self.done(f"{key}-launch")
        if not launch and name in self.ops.running_names():  # died between `docker run` and the checkpoint
            found = sorted(Path(self.logs).glob(f"{name}-*.log"))
            self.mark(f"{key}-launch", {"container": name, "log": str(found[-1]) if found else "", "extra_args": extra,
                                        "adopted_on_resume": True})
            launch = self.done(f"{key}-launch")
        if not (launch and name in self.ops.running_names()):
            if launch and name not in self.ops.running_names():
                gone = self.ops.crash_evidence(launch["decision"]["log"])
                if gone["fatal_engine_error"]:  # it died while this chain was not watching: keep the evidence
                    index = len(arm["passes"]) + 1
                    row = {"pass": index, "arm": tag, "level": next_level(plan, index - 1, arm["passes"][-1] if arm["passes"] else None, escalate),
                           "seed": pass_seed(index), "outcome": "exit", "preemptions": None, "evidence": gone,
                           "detail": {"found_on_resume": True}, "concurrency_per_instance": 0}
                    arm["passes"].append(row)
                    arm["server_log"] = launch["decision"]["log"]
                    self.mark(f"{key}-pass{index}", row)
                    save()
                    return self.finalize_arm(arm, track, plan, role, save)
                log(f"{key}: container gone on resume without a fatal error; relaunching cold for the remaining passes")
            self.set_current(f"{key}-launch", track)
            self.ops.cards_below([int(c) for c in spec["cards"].split(",")])
            log_path = self.ops.launch_e(name, spec, extra)
            self.mark(f"{key}-launch", {"container": name, "log": log_path, "extra_args": extra})
            launch = self.done(f"{key}-launch")
        arm["server_log"] = launch["decision"]["log"] if launch else arm.get("server_log", "")
        self.set_current(f"{key}-wait", track)
        try:
            self.ops.wait_healthy([{"name": name, "port": spec["port"]}], HEALTH_TIMEOUT_S, on_tick=self.write_status)
        except pc.ChainStop as stop:
            arm["startup"] = {"ok": False, "reason": str(stop), "evidence": self.ops.crash_evidence(arm["server_log"])}
            arm["verdict"] = {"verdict": "startup_failed"}
            arm["finished"] = True
            save()
            log(f"{key}: startup failed: {stop}")
            return arm
        now = self.ops.argv_of(name)
        expected_cmd = list(spec["cmd"]) + (extra.split() if extra else [])
        same = (now["cmd"] == expected_cmd and now["cpuset"] == spec["cpuset"]
                and now["hccl_base_port"] == spec["hccl_base_port"] and now["devices"] == spec["devices"])
        self.ops.patch_jobs([{"id": f"vllm-prelim-{key}", "kind": "vllm_server", "host": "npu2",
                              "note": f"{key} experiment instance (plan 3.8); preemption BAD events and a death are the "
                                      "subject of the experiment; removed when its arm A instance is restored",
                              "params": {"container": name, "port": spec["port"], "log": arm["server_log"]}}])
        arm["startup"] = {"ok": True, "argv_identical_to_capture_plus_extra": same, "cmd": now["cmd"]}
        if extra:
            arm["startup"]["log_confirms_extra"] = self.ops.log_confirms(arm["server_log"], CONFIRM_RE[extra])
        if not same:  # a different shape would make the experiment meaningless: stop before any load
            raise pc.ChainStop(f"{key}: argv differs from the capture (+extra): {now['cmd']}")
        save()
        if not self.done(f"{key}-wait"):
            self.mark(f"{key}-wait", {"healthy": name, "argv_identical": same})
        # --- 16 fixed requests BEFORE the load
        if not self.done(f"{key}-eq"):
            self.set_current(f"{key}-eq", track)
            if key == "E1-base":
                pa = f"{self.logs}/prelim-E1-eq-a-{self.ts}.json"
                pb = f"{self.logs}/prelim-E1-eq-b-{self.ts}.json"
                ra, rb = self.ops.eq_run(url, pa), self.ops.eq_run(url, pb)
                self.mark(f"{key}-eq", {"results_path": pa, "second_path": pb, "self_compare": compare_eq(ra, rb)})
            else:
                pa = f"{self.logs}/prelim-E2-eq-{tag}-{self.ts}.json"
                self.ops.eq_run(url, pa)
                self.mark(f"{key}-eq", {"results_path": pa})
        self.eq_compare()
        # --- passes
        for index, planned in enumerate(plan, 1):
            step = f"{key}-pass{index}"
            if self.done(step):
                continue  # already in arm["passes"] (rebuilt above)
            prev = arm["passes"][-1] if arm["passes"] else None
            level = next_level(plan, index - 1, prev, escalate)
            seed = pass_seed(index)
            prefix = self.prefix_of(track, tag, index)
            self.ops.patch_jobs([self.sweep_entry(track, tag, index, prefix)])
            self.set_current(step, track)
            label = (f"PRELIMINARY {track} {tag} ({label_role}) worst case pass {index} at {level}/instance, one endpoint, "
                     "not the recorded locked baseline")
            try:
                measure = self.ops.measure_pass(prefix, level, seed, url, name, label,
                                                [int(c) for c in spec["cards"].split(",")])
            except pc.EndpointDied as died:
                evidence = self.ops.crash_evidence(arm["server_log"])
                evidence_path = f"{self.logs}/prelim-{track}-{tag}-pass{index}-crash-evidence-{self.ts}.json"
                write_atomic(evidence_path, json.dumps({"died": died.name, "detail": died.detail, "evidence": evidence},
                                                       indent=1, default=str) + "\n")
                d = died.detail
                row = {"pass": index, "arm": tag, "level": level, "seed": seed, "outcome": "exit",
                       "preemptions": (d["preemptions_last_polled"] - d["preemptions_first_polled"])
                       if d.get("preemptions_last_polled") is not None and d.get("preemptions_first_polled") is not None
                       else None, "kv_peak": d.get("kv_peak_polled"), "detail": d, "evidence": evidence,
                       "evidence_path": evidence_path, "concurrency_per_instance": level}
                self.write_pass_summary(prefix, label, row, done=True)
                arm["passes"].append(row)
                save()
                self.mark(step, row)
                log(f"{step}: EXIT of {died.name}: assertion={evidence['mm_embeddings_assertion']} "
                    f"died_ids={evidence['died_request_ids']} k2={evidence['k2_confirmed']}")
                break
            row = pass_row(measure, level, seed, index, tag, name)
            if row["errors"] or row["timeouts"] or row["ok"] != row["requests"]:
                scan = self.ops.crash_evidence(arm["server_log"])  # the engine may die while the API process lingers
                row["log_scan"] = {k: scan.get(k) for k in ("mm_embeddings_assertion", "fatal_engine_error", "died_request_ids")}
                if scan.get("fatal_engine_error"):
                    row.update(outcome="exit", evidence=scan, preemptions=row["preemptions"])
            self.write_pass_summary(prefix, label, row, done=True)
            arm["passes"].append(row)
            save()
            self.mark(step, row)
            log(f"{step}: {row['outcome']} level {level} preemptions={row['preemptions']} ok={row['ok']}/{row['requests']} "
                f"kv_peak={row['kv_peak']}")
            if row["outcome"] == "exit":
                break
        return self.finalize_arm(arm, track, plan, role, save)

    def finalize_arm(self, arm: dict, track: str, plan: list[int], role: str, save) -> dict:
        arm["passes"].sort(key=lambda p: p["pass"])
        arm["verdict"] = (e1_verdict if track == "E1" else e2_verdict)(arm["passes"], expected=len(plan))
        if role == "diagnostic":
            arm["verdict"] = dict(arm["verdict"], diagnostic_only=True,
                                  verdict="DIAGNOSTIC ONLY: " + arm["verdict"]["verdict"])
        arm["finished"] = True
        save()
        return arm

    def write_pass_summary(self, prefix: str, label: str, row: dict, done: bool) -> None:
        doc = {"label": label, "synthetic": True, "dataset_rows": 0, "seed": row["seed"], "done": done,
               "stopped_at_level": row["level"] if row["outcome"] == "exit" else None, "levels": [
                   {k: v for k, v in row.items() if k not in ("evidence", "detail")}]}
        write_atomic(f"{prefix}.json", json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")

    def stop_e(self, name: str) -> None:
        if name in self.ops.running_names():
            self.ops.stop_containers([name])
            deadline = time.time() + 240
            while name in self.ops.running_names() and time.time() < deadline:
                self.sleep(3)

    def restore_arm_a(self, track: str) -> None:
        step = f"{track}-restore-arm-a"
        if self.done(step):
            return
        arm = TRACKS[track]["arm_a"]
        spec = self.spec_of(track)
        self.set_current(step, track)
        for key, name in CONTAINER.items():
            if key.startswith(track):
                self.stop_e(name)
        self.ops.cards_below([int(c) for c in spec["cards"].split(",")])
        log_path = ""
        if arm not in self.ops.running_names():
            log_path = self.ops.launch(spec)
        else:
            log(f"{arm} already running")
        self.ops.wait_healthy([spec], HEALTH_TIMEOUT_S, on_tick=self.write_status)
        now = self.ops.argv_of(arm)
        same = (now["cmd"] == spec["cmd"] and now["cpuset"] == spec["cpuset"]
                and now["hccl_base_port"] == spec["hccl_base_port"] and now["devices"] == spec["devices"])
        self.ops.patch_jobs([self.a_entry(arm, spec["port"], log_path or None)],
                            remove=[f"vllm-prelim-{k}" for k in CONTAINER if k.startswith(track)])
        self.mark(step, {"arm": arm, "log": log_path, "healthy": True, "argv_identical_to_capture": same})
        if not same:
            log(f"CHAIN-BAD {arm} argv differs from the capture after relaunch")

    def run_track(self, track: str) -> None:
        self.track_status(track, state="running")
        try:
            self.stop_arm_a(track)
            arms, results = TRACKS[track]["arms"], {}
            flag = (self.done("flagcheck") or {}).get("decision", {})
            if track == "E2" and not flag.get("shm_accepted", True):
                log("E2: vLLM 0.30.0 does not accept --mm-processor-cache-type shm; going straight to the diagnostic arm")
                first = {"verdict": {"verdict": "not_accepted_by_vllm"}, "role": "candidate", "extra_args": arms[0][1],
                         "passes": [], "startup": {"ok": False, "flagcheck": flag.get("rows")}, "container": ""}
            else:
                first = self.run_arm(track, *arms[0])
            results[arms[0][0]] = first
            if first["container"]:
                self.stop_e(first["container"])
            if track == "E2" and first["verdict"]["verdict"] in ("startup_failed", "failed_exit", "not_accepted_by_vllm"):
                log("E2: shm arm failed; running the diagnostic arm --no-enable-prefix-caching (DIAGNOSTIC ONLY)")
                results["noprefix"] = self.run_arm(track, *arms[1])
                self.stop_e(results["noprefix"]["container"])
            decision = {"track": track, "arms": {k: {"verdict": v["verdict"], "role": v["role"], "extra_args": v["extra_args"],
                                                       "passes": v["passes"], "startup": v["startup"],
                                                       "server_log": v.get("server_log")} for k, v in results.items()}}
            path = f"{self.logs}/prelim-{track}-{self.ts}.json"
            write_atomic(path, json.dumps(decision, indent=1, sort_keys=True, default=str) + "\n")
            with self.lock:
                self.status["decisions"][track] = {k: v["verdict"] for k, v in results.items()}
                self.status["decisions"][track]["result_file"] = path
            self.track_status(track, state="done", verdict=self.status["decisions"][track])
        except pc.ChainStop as stop:
            self.track_status(track, state="stopped", reason=str(stop))
            log(f"CHAIN-BAD {track} stopped: {stop}")
        except Exception as error:  # a bug must not leave arm A broken: the restore below still runs
            self.track_status(track, state="failed", reason=f"{type(error).__name__}: {error}")
            log(f"CHAIN-BAD {track} unexpected {type(error).__name__}: {error}")
        finally:
            try:
                self.restore_arm_a(track)
            except Exception as error:
                self.track_status(track, restore_error=f"{type(error).__name__}: {error}")
                log(f"CHAIN-BAD {track} could not restore its arm A instance: {type(error).__name__}: {error}")

    def run(self) -> int:
        self.heartbeat()
        try:
            self.ops.patch_jobs(self.job_entries(), remove=["prelim-chain", "sweep-armA-prelim-P2-mintok4096"])
            self.gate()
            if not self.done("flagcheck"):  # after the gate: it is CPU only, but it must not disturb the P2b load
                self.set_current("flagcheck")
                check = self.ops.flag_check()
                self.mark("flagcheck", check)
                log(f"flagcheck: shm_accepted={check['shm_accepted']} bogus_rejected={check['bogus_rejected']} "
                    f"noprefix_accepted={check['noprefix_accepted']}")
            self.ops.start_sampler()
            self.set_current("tracks-running")
            threads = [threading.Thread(target=self.run_track, args=(t,), name=t) for t in self.tracks]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.ops.stop_sampler()
            bad = {t: v for t, v in self.status["tracks"].items() if v.get("state") != "done" or v.get("restore_error")}
            self.set_current(None)
            if bad:
                self.status.update(state="failed", reason=f"tracks not done: {sorted(bad)}")
                self.write_status()
                log(f"CHAIN-BAD tracks not done: {sorted(bad)}")
                return 5
            self.status.update(state="done", current_step=None, reason=None)
            self.write_status()
            log("CHAIN DONE")
            return 0
        except pc.ChainStop as stop:
            self.status.update(state="failed", reason=str(stop))
            self.write_status()
            log(f"CHAIN-BAD stopped at {self.status['current_step']}: {stop}")
            return 2
        except KeyboardInterrupt:
            self.status.update(state="interrupted", reason="signal; containers left as they are; rerun with --resume")
            self.write_status()
            log("CHAIN-BAD interrupted by a signal; containers left running; rerun with --resume")
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
    """Scripted operations for the rehearsal and the unit tests: nothing touches docker, a card, the network or the
    registry. `scenario` keys: e1_exit_pass (int), e1_low (set of passes with < 10 preemptions), e2_start_fail (bool),
    e2_exit_pass (int), e2_low, gate_waits (reads before the gate opens)."""

    def __init__(self, scenario: dict | None = None) -> None:
        self.sc = dict(scenario or {})
        self.lock = threading.Lock()
        self.running = {"saea-arm-a-0", "saea-arm-a-1", "saea-arm-a-2", "saea-arm-a-3"}
        self.calls: list[str] = []
        self.calls_gate: list[tuple] = []
        self.gate_reads = 0
        self.eq_text = self.sc.get("eq_text", {})

    def note(self, text: str) -> None:
        with self.lock:
            self.calls.append(text)
            self.calls_gate.append((self.gate_reads, self.sc.get("gate_waits", 2), text))
        log(f"[dry-run] {text}")

    def start_sampler(self) -> None: self.note("sampler start")
    def stop_sampler(self) -> None: self.note("sampler stop")

    def gate_status(self, path):
        self.gate_reads += 1
        waits = self.sc.get("gate_waits", 2)
        if self.gate_reads <= waits:
            return {"state": "running", "current_step": "p2b-fresh40-wait", "pid": 4242}
        return {"state": self.sc.get("gate_state", "done"), "pid": 4242, "reason": None}

    def pid_alive(self, pid) -> bool: return self.gate_reads <= self.sc.get("gate_waits", 2) + self.sc.get("gate_pid_extra", 0)
    def running_names(self) -> set[str]:
        with self.lock:
            return set(self.running)
    def running_requests(self, url) -> float: return 0.0

    def patch_jobs(self, upsert, remove=(), set_log=None) -> None:
        self.note(f"jobs upsert={[j['id'] for j in upsert]} remove={list(remove)}")

    def stop_containers(self, names) -> None:
        self.note(f"docker stop {names}")
        with self.lock:
            self.running -= set(names)

    def cards_below(self, cards, timeout=0) -> dict:
        self.note(f"cards {cards} below 4096 MiB")
        return {c: 3440 for c in cards}

    def flag_check(self) -> dict:
        return {"shm_accepted": True, "bogus_rejected": True, "noprefix_accepted": True, "rows": {}, "exit_code": 0}

    def launch_e(self, name, spec, extra) -> str:
        self.note(f"launch_e {name} cards {spec['cards']} cpuset {spec['cpuset']} hccl {spec['hccl_base_port']} extra {extra!r}")
        with self.lock:
            self.running.add(name)
        self._extra = getattr(self, "_extra", {})
        self._extra[name] = extra
        return f"/dry/{name}.log"

    def launch(self, spec) -> str:
        self.note(f"launch arm A {spec['name']} cards {spec['cards']} (npu_serve.sh)")
        with self.lock:
            self.running.add(spec["name"])
        return f"/dry/{spec['name']}.log"

    def wait_healthy(self, specs, timeout, on_tick=None) -> dict:
        for s in specs:
            if s["name"] == "saea-prelim-e2-shm" and self.sc.get("e2_start_fail"):
                with self.lock:
                    self.running.discard(s["name"])
                raise pc.ChainStop(f"{s['name']} exited during startup (see its log)")
        self.note(f"healthy {[s['name'] for s in specs]}")
        return {s["name"]: True for s in specs}

    def argv_of(self, name) -> dict:
        cap = FAKE_CAPTURE
        spec = next((s for s in cap if s["name"] == name), None)
        if spec:
            return {k: spec[k] for k in ("cmd", "cpuset", "hccl_base_port", "devices")}
        index = 0 if "e1" in name else 3
        spec = cap[index]
        extra = getattr(self, "_extra", {}).get(name, "")
        return {"cmd": list(spec["cmd"]) + (extra.split() if extra else []), "cpuset": spec["cpuset"],
                "hccl_base_port": spec["hccl_base_port"], "devices": spec["devices"]}

    def crash_evidence(self, path) -> dict:
        ids = ["chatcmpl-sweep-c56-w188-t188-k2-96e10751"]
        return {"mm_embeddings_assertion": "e1" in path, "execute_mm_encoder": "e1" in path, "fatal_engine_error": True,
                "died_request_ids": ids, "k2_confirmed": True, "resumed_req_ids": ids, "server_log": path, "lines": [],
                "scheduled_encoder_input_ids": ids, "error_tail": []}

    def log_confirms(self, path, regex) -> bool: return True

    def eq_run(self, url, out_path) -> list[dict]:
        results = [{"index": i, "status": "ok", "reasoning": f"r{i}", "content": "", "finish_reason": "length",
                    "completion_tokens": 64, "logprobs": [-0.1, -0.2]} for i in range(16)]
        if "shm" in out_path and self.sc.get("eq_mismatch"):
            for i in self.sc["eq_mismatch"]:
                results[i]["reasoning"] = "different"
        write_atomic(out_path, json.dumps({"results": results}) + "\n")
        return results

    def measure_pass(self, prefix, level, seed, url, name, label, cards) -> dict:
        index = int(re.search(r"pass(\d+)", prefix).group(1))
        self.note(f"measure {Path(prefix).name} level {level} seed {seed} url {url}")
        track = "e1" if "E1" in prefix else "e2"
        if self.sc.get(f"{track}_exit_pass") == index and not (track == "e2" and "noprefix" in prefix):
            with self.lock:
                self.running.discard(name)
            raise pc.EndpointDied(name, {"dead_instances": [name], "preemptions_first_polled": 0.0,
                                         "preemptions_last_polled": 7.0, "kv_peak_polled": 1.0})
        low = index in self.sc.get(f"{track}_low", set())
        pre = 4.0 if low else 12.0
        summary = {"requests": 3 * level, "ok": 3 * level, "errors": 0, "timeouts": 0, "wall_s": 900.0,
                   "output_tok_s": 450.0, "e2e_tok_s": {"p10": 9.5}, "latency_s": {"max": 330.0}, "instances": 1,
                   "metrics_before": [{"preemptions": 1.0}], "metrics_after": [{"preemptions": 1.0 + pre}],
                   "gauges": [{"kv_cache_usage_perc": {"max": 1.0}}]}
        return {"summary": summary, "report": f"{Path(prefix).name}-L{level}.json", "report_sha256": "0" * 64,
                "min_hbm_free_mib": 4100, "poll_kv_peak": [1.0]}


FAKE_CAPTURE = [
    {"name": f"saea-arm-a-{i}", "cards": f"{2 * i},{2 * i + 1}", "port": 8200 + i, "tp": 2, "max_num_seqs": 64,
     "cpuset": ["192-223", "128-159", "0-31", "64-95"][i], "hccl_base_port": 60000 + 100 * i,
     "cmd": ["serve", "/m", "--port", str(8200 + i), "--seed", "0"],
     "devices": [f"/dev/davinci{2 * i}", f"/dev/davinci{2 * i + 1}"]} for i in range(4)]


def chmod_quiet(path, mode) -> None:
    try:
        os.chmod(path, mode)
    except OSError:  # a directory made by another uid on the shared mount keeps its (default-ACL) permissions
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ts", help="UTC stamp used in every file name (default: now)")
    ap.add_argument("--chain-dir", help="existing chain directory (with --resume)")
    ap.add_argument("--resume", action="store_true", help="skip finished steps of an existing chain directory")
    ap.add_argument("--dry-run", action="store_true", help="rehearse with scripted operations (CPU only)")
    ap.add_argument("--scenario", default="", help="dry-run only: comma list of e1_exit=N, e2_exit=N, e2_start_fail, "
                                                    "e1_low=N, e2_low=N, eq_mismatch=I")
    ap.add_argument("--gate-file", default=DEFAULT_GATE, help="chain-status.json of the chain that holds the cards")
    ap.add_argument("--arm-a-launch", default=DEFAULT_CAPTURE, help="captured arm A argv (copied into the chain dir)")
    ap.add_argument("--out-root", help="dry-run only: output root (default: a temporary directory)")
    args = ap.parse_args()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if args.dry_run:
        root = Path(args.out_root or tempfile.mkdtemp(prefix="prelim-E-dry-"))
        ts = args.ts or "DRYRUN"
        sc: dict = {}
        for item in filter(None, args.scenario.split(",")):
            key, _, val = item.partition("=")
            if key in ("e1_exit", "e2_exit"):
                sc[f"{key}_pass"] = int(val)
            elif key in ("e1_low", "e2_low"):
                sc.setdefault(key, set()).add(int(val))
            elif key == "e2_start_fail":
                sc["e2_start_fail"] = True
            elif key == "eq_mismatch":
                sc.setdefault("eq_mismatch", []).append(int(val))
        chain_dir = root / f"prelim-E-chain-{ts}"
        chain_dir.mkdir(parents=True, exist_ok=True)
        log(f"dry-run output under {root}")
        chain = EChain(FakeOps(sc), chain_dir, ts, str(root), FAKE_CAPTURE, "/dry/gate.json", gate_poll=0.0,
                       sleep=lambda s: None)
        code = chain.run()
        print(json.dumps(read_json(chain_dir / "chain-status.json")["decisions"], indent=1, sort_keys=True))
        return code
    hostname = Path("/etc/hostname").read_text().strip()
    if hostname != pc.EXPECTED_HOSTNAME:
        sys.exit(f"refusing to run on {hostname!r}: this chain belongs to host 181 ({pc.EXPECTED_HOSTNAME})")
    if args.resume:
        chain_dir = Path(args.chain_dir or sys.exit("--resume needs --chain-dir"))
        config = read_json(chain_dir / "config.json")
        ts = config["ts"]
    else:
        ts = args.ts or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        chain_dir = Path(args.chain_dir or f"{LOGS}/prelim-E-chain-{ts}")
        if (chain_dir / "config.json").exists():
            sys.exit(f"{chain_dir} already holds a chain; use --resume")
        chain_dir.mkdir(parents=True, exist_ok=True)
        chmod_quiet(chain_dir, 0o777)
        captured = read_json(args.arm_a_launch)
        if not captured:
            sys.exit(f"needs a readable arm A capture: {args.arm_a_launch}")
        write_atomic(chain_dir / "arm-a-launch.json", json.dumps(captured, indent=2) + "\n")
        write_atomic(chain_dir / "config.json", json.dumps(
            {"ts": ts, "created": utc_now(), "script": str(Path(__file__).resolve()), "gate_file": args.gate_file,
             "arm_a_launch_source": args.arm_a_launch}) + "\n")
    chmod_quiet(chain_dir / "config.json", 0o666)
    (chain_dir / "steps").mkdir(exist_ok=True)
    chmod_quiet(chain_dir / "steps", 0o777)
    config = read_json(chain_dir / "config.json")
    capture = read_json(chain_dir / "arm-a-launch.json")["instances"]
    log(f"E chain {chain_dir.name} pid {os.getpid()} resume={args.resume} gate {config['gate_file']}")
    return EChain(EReal(args.resume), chain_dir, ts, LOGS, capture, config["gate_file"]).run()


if __name__ == "__main__":
    sys.exit(main())
