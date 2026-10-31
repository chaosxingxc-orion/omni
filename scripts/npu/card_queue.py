#!/usr/bin/env python3
"""Detached card-queue runner: keeps the NPU host busy by launching the next queued job the moment the previous one ends.

Owner rule 2026-10-30: the cards must be kept fully loaded with experiments; no chain may wait for a session to launch
the next one. The queue is DATA (JSON, schema saea-card-queue-v1); this runner decides nothing about WHICH jobs run.

Queue file:
  {"schema": "saea-card-queue-v1", "id": "<queue id>", "jobs": [
     {"id": ..., "kind": ..., "cmd": [argv...], "cwd": ..., "log": ..., "done_marker": "<regex>", "fail_marker": "<regex>",
      "timeout_s": N, "requires_cards_free": true|false, "on_fail": "continue"|"stop",
      optional: "cards_policy": "all_free"|"no_other", "wait_for_final": "<chain-status.json>", "enabled": false,
                "env": {"K": "V"}}]}
  "{queue_dir}" in cmd, cwd, log and env values is replaced by the queue directory.

Behaviour (one job at a time, in file order):
  * each job is its own detached process (setsid, stdin /dev/null, stdout+stderr appended to its log), so the runner dying
    never kills a job; the runner polls the log for the done/fail marker and the job's liveness;
  * verdict: fail marker -> failed (wins); else done marker -> done; else process gone: exit 0 with no done_marker
    configured -> done, anything else -> failed. After the done marker the job gets DONE_GRACE_S to exit, after the fail
    marker FAIL_GRACE_S; a job still alive after that is sent SIGTERM (the chains handle it: servers stay as they are),
    SIGKILL 30 s later. timeout_s expiry is a failure handled the same way;
  * wait_for_final: the job is not launched until that chain-status.json is FINAL: state done/failed/interrupted AND its
    "pid" no longer alive (the rule e3_chain's gate step uses); an unreadable or missing file is not final, so it waits;
  * requires_cards_free waits (never launches blind) until the cards policy holds AND no other queue job runs:
      all_free  (default) every card is free (no process, HBM below the idle threshold: no vllm/VLLMWorker, nothing);
      no_other  no card is held by a process that is not one of ours-and-vllm, i.e. never a card another user holds;
                the resident vLLM service may stay up (use this for a job that brings that service down itself);
    the classification is the watchdog's `npu_cards` parse (npu-smi info + docker ps + /proc/<pid>/cgroup);
  * on_fail "stop": the queue stops, later jobs stay pending (state "stopped"); "continue": the next job runs;
  * the queue file is re-read before every pick, so jobs can be appended or edited while the runner is waiting; jobs
    already started are never changed;
  * state is checkpointed atomically to <queue-dir>/state.json after every change; --resume adopts a job that is still
    running (matched by an environment token, not just a pid) and re-derives the verdict of one that ended meanwhile.
The runner never starts or stops a vLLM server or a container itself; that is the job's own business (npu_deploy.sh, the
chains). It never touches a card another user's process holds.

usage (on the NPU host, detached):
  nohup setsid python3 card_queue.py run --queue <queue.json> --queue-dir <dir> < /dev/null > <dir>/queue.log 2>&1 &
  resume:   python3 card_queue.py run --queue <queue.json> --queue-dir <dir> --resume [--retry-failed]   (same detached form)
  rehearse: python3 card_queue.py run --queue <queue.json> --queue-dir <dir> --dry-run     (prints every command, runs nothing)
  status:   python3 card_queue.py status --queue-dir <dir> [--json]
Python 3.8+ standard library only. Exit codes of `run`: 0 all jobs done, 2 stopped by on_fail=stop, 3 interrupted by a
signal (jobs keep running; --resume), 4 finished with failed jobs (on_fail=continue), 5 bad queue file or usage.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

QUEUE_SCHEMA = "saea-card-queue-v1"
STATE_SCHEMA = "saea-card-queue-state-v1"
ON_FAIL = ("continue", "stop")
POLICIES = ("all_free", "no_other")
DONE_GRACE_S = 300.0
FAIL_GRACE_S = 120.0
KILL_WAIT_S = 30.0
IDLE_HBM_MIB = 8192  # same default as the watchdog npu_cards kind
VLLM_PROC = re.compile(r"vllm", re.I)  # matches vllm, VLLMWorker, VLLM::EngineCore ...
OURS_RE = re.compile(r"^saea-")  # container-name prefix that is ours (watchdog default ours_regex)
TERMINAL = ("done", "failed")
KEEP_BYTES = 8_000_000


def now_s():
    return time.time()


def iso(t=None):
    return datetime.fromtimestamp(now_s() if t is None else t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print("%s %s" % (iso(), msg), flush=True)


def write_atomic(path, text, mode=0o666):
    tmp = "%s.tmp%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.chmod(tmp, mode)
    except OSError:
        pass
    os.replace(tmp, path)
    directory = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- queue file
class QueueError(Exception):
    pass


def _subst(value, qdir):
    if isinstance(value, str):
        return value.replace("{queue_dir}", qdir)
    if isinstance(value, list):
        return [_subst(v, qdir) for v in value]
    if isinstance(value, dict):
        return {k: _subst(v, qdir) for k, v in value.items()}
    return value


def load_queue(path, qdir):
    """Validated queue document with {queue_dir} substituted. Raises QueueError."""
    doc = read_json(path)
    if not isinstance(doc, dict) or doc.get("schema") != QUEUE_SCHEMA:
        raise QueueError("queue file %s unreadable or schema is not %s" % (path, QUEUE_SCHEMA))
    jobs, seen = [], set()
    for raw in doc.get("jobs") or []:
        j = _subst(raw, qdir)
        jid = j.get("id")
        if not isinstance(jid, str) or not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$", jid):
            raise QueueError("bad job id %r" % (jid,))
        if jid in seen:
            raise QueueError("duplicate job id %s" % jid)
        seen.add(jid)
        cmd = j.get("cmd")
        if not (isinstance(cmd, list) and cmd and all(isinstance(x, str) for x in cmd)):
            raise QueueError("%s: cmd must be a non-empty argv list of strings" % jid)
        for key in ("cwd", "log"):
            if not isinstance(j.get(key), str) or not j[key]:
                raise QueueError("%s: %s must be a non-empty path" % (jid, key))
        for key in ("done_marker", "fail_marker"):
            if j.get(key) is not None:
                try:
                    re.compile(j[key])
                except (re.error, TypeError):
                    raise QueueError("%s: %s is not a valid regex" % (jid, key))
        t = j.get("timeout_s")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or t <= 0:
            raise QueueError("%s: timeout_s must be a positive number" % jid)
        if j.get("on_fail", "continue") not in ON_FAIL:
            raise QueueError("%s: on_fail must be one of %s" % (jid, ON_FAIL))
        if j.get("cards_policy", "all_free") not in POLICIES:
            raise QueueError("%s: cards_policy must be one of %s" % (jid, POLICIES))
        if j.get("wait_for_final") is not None and not (isinstance(j["wait_for_final"], str) and j["wait_for_final"]):
            raise QueueError("%s: wait_for_final must be a path" % jid)
        if not isinstance(j.get("requires_cards_free", False), bool):
            raise QueueError("%s: requires_cards_free must be a boolean" % jid)
        if j.get("env") is not None and not (isinstance(j["env"], dict) and
                                             all(isinstance(k, str) and isinstance(v, str) for k, v in j["env"].items())):
            raise QueueError("%s: env must map strings to strings" % jid)
        jobs.append(j)
    if not jobs:
        raise QueueError("queue has no jobs")
    return {"id": str(doc.get("id") or "queue"), "note": doc.get("note", ""), "jobs": jobs}


GATE_FINAL = ("done", "failed", "interrupted")


def final_gate(status, pid_alive_fn):
    """(final, why): the chain-status.json rule of prelim_chain.gate_final / e3_chain's gate step."""
    if not isinstance(status, dict):
        return False, "chain-status.json unreadable"
    state = status.get("state")
    if state not in GATE_FINAL:
        return False, "state %r step %r" % (state, status.get("current_step"))
    pid = status.get("pid")
    if isinstance(pid, int) and pid > 0 and pid_alive_fn(pid):
        return False, "state %r but pid %s is still alive" % (state, pid)
    return True, "state %r" % state


# ---------------------------------------------------------------- card classification (the watchdog npu_cards parse)
CARD1 = re.compile(r"^\|\s*(\d+)\s+(\S+)\s*\|\s*(\S+)\s*\|")
CARD2 = re.compile(r"^\|\s*\d+\s*\|\s*[0-9A-Fa-f:.]+\s*\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)")
PROC = re.compile(r"^\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|\s*(\S+)\s*\|\s*(\d+)\s*\|")


def parse_smi(text):
    """Copy of watchdog.parse_smi: ({npu: {ai, hbm, total}}, {npu: [(pid, name, mem_mb)]})."""
    cards, procs, npu = {}, {}, None
    for line in text.splitlines():
        m = PROC.match(line)
        if m:
            procs.setdefault(int(m.group(1)), []).append((int(m.group(2)), m.group(3), int(m.group(4))))
            continue
        m = CARD2.match(line)
        if m and npu is not None:
            cards[npu] = {"ai": int(m.group(1)), "hbm": int(m.group(2)), "total": int(m.group(3))}
            npu = None
            continue
        m = CARD1.match(line)
        if m:
            npu = int(m.group(1))
    return cards, procs


def classify_cards(smi_text, docker_ps, pid_container, idle_hbm=IDLE_HBM_MIB, ours_re=OURS_RE):
    """{npu: {"cls": free|ours|other, "owners": [container names], "procs": [(pid, name, mem)], "hbm", "ai"}}.

    Same rule as the watchdog: busy = any process listed or HBM above the idle threshold; ours = every owning container
    name matches ours_re; anything else busy (including unattributed HBM and host processes) is `other`."""
    cards, procs = parse_smi(smi_text)
    by_id = {c["id"]: c["name"] for c in (docker_ps or [])}
    out = {}
    for n, c in sorted(cards.items()):
        plist = procs.get(n, [])
        owners = sorted({by_id.get(pid_container(pid), "host") for pid, _, _ in plist})
        busy = bool(plist) or c["hbm"] > idle_hbm
        cls = "free" if not busy else ("ours" if owners and all(ours_re.match(x) for x in owners) else "other")
        # vLLM-Ascend also reports a python3 parent on the same card/container.
        # Only the explicitly named resident containers with a co-located VLLM worker qualify.
        worker_owners = {by_id.get(pid_container(pid), 'host') for pid, name, _ in plist if VLLM_PROC.search(name)}
        resident_python = [pid for pid, name, _ in plist if name in ('python', 'python3')
                           and by_id.get(pid_container(pid), 'host') in worker_owners
                           and re.fullmatch(r'saea-omni-i[0-3]', by_id.get(pid_container(pid), 'host'))]
        out[n] = {"resident_python_pids": resident_python, "cls": cls, "owners": owners, "procs": plist, "hbm": c["hbm"], "ai": c["ai"]}
    return out


def card_blockers(cards, policy):
    """Reasons the cards policy does not hold (empty list = the job may start)."""
    if not cards:
        return ["npu-smi unreadable or not parsed; refusing to launch blind"]
    reasons = []
    for n, c in sorted(cards.items()):
        names = sorted({name for _, name, _ in c["procs"]})
        holders = "%s [%s]" % (",".join(names) or "unattributed HBM %d MiB" % c["hbm"], ",".join(c["owners"]))
        if policy == "all_free":
            if c["cls"] != "free":
                kind = "vllm" if any(VLLM_PROC.search(x) for x in names) else c["cls"]
                reasons.append("card %d held (%s): %s" % (n, kind, holders))
        else:  # no_other: only a foreign (or ours-but-not-vllm) holder blocks
            foreign = c["cls"] == "other"
            ours_non_vllm = c["cls"] == "ours" and any(not VLLM_PROC.search(name) and pid not in c.get("resident_python_pids", []) for pid, name, _ in c["procs"])
            if foreign or ours_non_vllm:
                reasons.append("card %d held by another job (%s): %s" % (n, c["cls"], holders))
    return reasons


class RealCardProbe:
    """Reads the real host: npu-smi info, docker ps, /proc/<pid>/cgroup. Built only by main(); tests inject a fake."""

    def _sh(self, cmd, timeout=30):
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout)
            return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    def __call__(self):
        txt = self._sh(["npu-smi", "info"])
        if not txt:
            return {}
        out = self._sh(["docker", "ps", "--no-trunc", "--format", "{{.ID}}|{{.Names}}|{{.Image}}"])
        ps = []
        for line in (out or "").splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                ps.append({"id": parts[0], "name": parts[1], "image": parts[2]})

        def pid_container(pid):
            try:
                m = re.search(r"([0-9a-f]{64})", open("/proc/%d/cgroup" % pid).read())
                return m.group(1) if m else None
            except OSError:
                return None

        return classify_cards(txt, ps, pid_container)


# ---------------------------------------------------------------- processes
def proc_state(pid):
    """Single-letter /proc state, or None when the process does not exist."""
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return None


def proc_alive(pid, token=None):
    if not isinstance(pid, int) or pid <= 0:
        return False
    st = proc_state(pid)
    if st is None or st == "Z":
        return False
    if token:
        try:
            with open("/proc/%d/environ" % pid, "rb") as f:
                return ("CARD_QUEUE_TOKEN=%s" % token).encode() in f.read().split(b"\0")
        except OSError:
            return True  # unreadable: keep the benefit of the doubt
    return True


def find_token_pid(token):
    """pid of a process carrying CARD_QUEUE_TOKEN=<token> in its environment (job launched but pid not recorded)."""
    needle = ("CARD_QUEUE_TOKEN=%s" % token).encode()
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        try:
            with open("/proc/%s/environ" % name, "rb") as f:
                if needle in f.read().split(b"\0"):
                    return int(name)
        except OSError:
            continue
    return None


def killpg(pid, sig):
    try:
        os.killpg(os.getpgid(pid), sig)
    except (OSError, ProcessLookupError):
        try:
            os.kill(pid, sig)
        except OSError:
            pass


# ---------------------------------------------------------------- the runner
class Runner:
    def __init__(self, queue_path, qdir, probe, poll_s=15.0, resume=False, retry_failed=False, peers=(),
                 done_grace_s=DONE_GRACE_S, fail_grace_s=FAIL_GRACE_S, kill_wait_s=KILL_WAIT_S, sleep=time.sleep,
                 pid_alive=proc_alive):
        self.queue_path, self.qdir, self.probe, self.poll_s = queue_path, qdir, probe, poll_s
        self.resume, self.retry_failed, self.peers = resume, retry_failed, list(peers)
        self.pid_alive = pid_alive
        self.done_grace_s, self.fail_grace_s, self.kill_wait_s, self.sleep = done_grace_s, fail_grace_s, kill_wait_s, sleep
        self.state_path = os.path.join(qdir, "state.json")
        self.queue = None
        self.state = None
        self.children = {}  # job id -> Popen (only for jobs this runner process started)
        self.interrupted = False
        self.scan = {}  # job id -> {"pos", "rest", "done", "fail"}  (marker scan cursor; rebuilt on resume)

    # --- state
    def save(self):
        self.state["updated"] = iso()
        c = {"pending": 0, "waiting": 0, "running": 0, "done": 0, "failed": 0, "skipped": 0}
        for j in self.state["jobs"].values():
            c[j["status"]] = c.get(j["status"], 0) + 1
        self.state["counts"] = c
        self.state["order"] = [j["id"] for j in self.queue["jobs"]]  # file order (json key order is not kept by every reader)
        write_atomic(self.state_path, json.dumps(self.state, indent=1, sort_keys=True) + "\n")

    def load(self):
        self.queue = load_queue(self.queue_path, self.qdir)
        old = read_json(self.state_path)
        if old and not self.resume:
            raise QueueError("%s already holds a queue state; use --resume" % self.qdir)
        if old and old.get("schema") == STATE_SCHEMA:
            self.state = old
            self.state.update(state="running", reason=None, runner_pid=os.getpid(), queue_file=self.queue_path)
            if self.retry_failed:
                for j in self.state["jobs"].values():
                    if j["status"] == "failed":
                        j.update(status="pending", reason=None, exit=None, finished=None, pid=None)
        else:
            self.state = {"schema": STATE_SCHEMA, "queue_id": self.queue["id"], "queue_file": self.queue_path,
                          "state": "running", "reason": None, "runner_pid": os.getpid(), "started": iso(),
                          "current": None, "waiting": None, "jobs": {}}
        self.sync_jobs()
        self.save()

    def sync_jobs(self):
        """Add jobs the queue file gained; a job that already left `pending` is never changed by an edit."""
        for j in self.queue["jobs"]:
            rec = self.state["jobs"].get(j["id"])
            if rec is not None and rec["status"] == "skipped" and j.get("enabled", True) is not False:
                rec.update(status="pending", reason=None)  # a job enabled after the runner started
            if j.get("enabled", True) is False and rec is not None and rec["status"] == "pending":
                rec.update(status="skipped", reason="disabled in the queue file")
            if rec is None:
                self.state["jobs"][j["id"]] = {
                    "status": "skipped" if j.get("enabled", True) is False else "pending", "kind": j.get("kind"),
                    "log": j["log"], "requires_cards_free": bool(j.get("requires_cards_free")),
                    "reason": "disabled in the queue file" if j.get("enabled", True) is False else None,
                    "pid": None, "token": None, "started": None, "finished": None, "exit": None}

    def reload_queue(self):
        try:
            self.queue = load_queue(self.queue_path, self.qdir)
        except QueueError as e:
            log("WARN queue file not reloaded, keeping the previous one: %s" % e)
            return
        # Queue files may be extended while a detached runner is busy.  Persist
        # newly discovered jobs immediately; otherwise a runner can finish its
        # current job with a queue file that has more entries than state.json,
        # making the next pick fail or lose durable status across SSH loss.
        self.sync_jobs()
        self.save()

    def next_job(self):
        # A running job remains ours even if its row was removed from the editable queue.
        # Finish supervising it before admitting another card consumer.
        for record in self.state["jobs"].values():
            if record["status"] == "running" and record.get("job_spec"):
                return record["job_spec"]
        for j in self.queue["jobs"]:
            record = self.state["jobs"][j["id"]]
            st = record["status"]
            if st in ("pending", "waiting", "running"):
                # A resumed process must be judged against the command and markers it actually
                # started with, even when the operator has edited the queue in the meantime.
                return record.get("job_spec", j) if st == "running" else j
        return None

    # --- cards / peers
    def peer_running(self):
        for d in self.peers:
            s = read_json(os.path.join(d, "state.json"))
            if s and proc_alive(s.get("runner_pid")) and s.get("current"):
                return "other queue %s runs job %s" % (s.get("queue_id"), s["current"])
        return None

    def blockers(self, job):
        reasons = []
        if job.get("wait_for_final"):
            final, why = final_gate(read_json(job["wait_for_final"]), self.pid_alive)
            if not final:
                reasons.append("wait_for_final %s: %s" % (job["wait_for_final"], why))
                return reasons  # the chain still owns the cards; do not even read them
        if not job.get("requires_cards_free"):
            return reasons
        peer = self.peer_running()
        if peer:
            reasons.append(peer)
        reasons.extend(card_blockers(self.probe(), job.get("cards_policy", "all_free")))
        return reasons

    # --- log scanning
    def scan_log(self, job, rec, final=False):
        """Advance the marker cursor over the job's log (only bytes after the job's start offset)."""
        cur = self.scan.setdefault(job["id"], {"pos": rec.get("log_offset", 0), "rest": b"", "done": False, "fail": False})
        try:
            with open(job["log"], "rb") as f:
                f.seek(cur["pos"])
                data = f.read(KEEP_BYTES)
        except OSError:
            return cur
        cur["pos"] += len(data)
        buf = cur["rest"] + data
        cut = len(buf) if final else buf.rfind(b"\n") + 1
        cur["rest"] = buf[cut:]
        done_rx = re.compile(job["done_marker"]) if job.get("done_marker") else None
        fail_rx = re.compile(job["fail_marker"]) if job.get("fail_marker") else None
        for line in buf[:cut].decode("utf-8", "replace").splitlines():
            if fail_rx and not cur["fail"] and fail_rx.search(line):
                cur["fail"] = True
                rec["fail_line"] = line[:200]
            if done_rx and not cur["done"] and done_rx.search(line):
                cur["done"] = True
        if len(cur["rest"]) > KEEP_BYTES:
            cur["rest"] = b""
        return cur

    # --- launching
    def launch(self, job, rec):
        os.makedirs(os.path.dirname(job["log"]) or ".", exist_ok=True)
        token = uuid.uuid4().hex
        rec.update(status="running", token=token, started=iso(), started_ts=now_s(), exit=None, reason=None, pid=None,
                   finished=None, job_spec=json.loads(json.dumps(job)),
                   log_offset=os.path.getsize(job["log"]) if os.path.exists(job["log"]) else 0)
        self.state["current"], self.state["waiting"] = job["id"], None
        self.save()  # checkpoint BEFORE the spawn: a resume can find the process by its token
        env = dict(os.environ, CARD_QUEUE_TOKEN=token, PYTHONUNBUFFERED="1")
        env.update(job.get("env") or {})
        with open(job["log"], "ab") as lf:
            lf.write(("=== card_queue job %s start %s ===\n" % (job["id"], iso())).encode())
            lf.flush()
            rec["log_offset"] = lf.tell()
            try:
                proc = subprocess.Popen(job["cmd"], cwd=job["cwd"], stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                        env=env, start_new_session=True, close_fds=True)
            except OSError as e:
                rec.update(status="failed", reason="launch failed: %s" % e, finished=iso())
                self.state["current"] = None
                self.save()
                return None
        self.children[job["id"]] = proc
        rec["pid"] = proc.pid
        self.scan.pop(job["id"], None)
        self.save()
        log("job %s started pid %d: %s" % (job["id"], proc.pid, " ".join(shlex.quote(x) for x in job["cmd"])))
        return proc

    def alive(self, job, rec):
        proc = self.children.get(job["id"])
        if proc is not None:
            return proc.poll() is None
        return proc_alive(rec.get("pid"), rec.get("token"))

    def exit_code(self, job):
        proc = self.children.get(job["id"])
        return proc.returncode if proc is not None else None

    def terminate(self, rec, why):
        pid = rec.get("pid")
        if not pid:
            return
        log("terminating pid %d (%s)" % (pid, why))
        killpg(pid, signal.SIGTERM)
        deadline = now_s() + self.kill_wait_s
        while now_s() < deadline and self.child_running(rec):
            self.sleep(min(1.0, self.poll_s))
        if self.child_running(rec):
            killpg(pid, signal.SIGKILL)
            self.sleep(0.2)
            self.child_running(rec)  # reap

    def child_running(self, rec):
        for p in self.children.values():
            if p.pid == rec.get("pid"):
                return p.poll() is None
        return proc_alive(rec.get("pid"), rec.get("token"))

    # --- one job to its verdict
    def supervise(self, job, rec):
        fail_seen_at = done_seen_at = None
        while True:
            if self.interrupted:
                return None
            alive = self.alive(job, rec)
            cur = self.scan_log(job, rec, final=not alive)
            t = now_s()
            if cur["fail"] and fail_seen_at is None:
                fail_seen_at = t
            if cur["done"] and done_seen_at is None:
                done_seen_at = t
            if not alive:
                rec["exit"] = self.exit_code(job)
                if cur["fail"]:
                    return "failed", "fail marker in log: %s" % rec.get("fail_line", "")
                if cur["done"]:
                    return "done", None
                if job.get("done_marker"):
                    return "failed", "process ended (exit %s) without the done marker" % rec["exit"]
                if rec["exit"] == 0:
                    return "done", None
                return "failed", "process ended with exit %s" % rec["exit"]
            if fail_seen_at is not None and t - fail_seen_at > self.fail_grace_s:
                self.terminate(rec, "fail marker, still alive after %.0f s" % self.fail_grace_s)
                continue
            if done_seen_at is not None and not cur["fail"] and t - done_seen_at > self.done_grace_s:
                self.terminate(rec, "done marker, still alive after %.0f s" % self.done_grace_s)
                continue
            if t - rec["started_ts"] > job["timeout_s"]:
                self.terminate(rec, "timeout_s %s exceeded" % job["timeout_s"])
                rec["exit"] = self.exit_code(job)
                self.scan_log(job, rec, final=True)
                return "failed", "timeout after %s s" % job["timeout_s"]
            self.save()
            self.sleep(self.poll_s)

    def adopt_or_find(self, job, rec):
        """Resume of a job that was `running`: locate its process by the token if the pid was not recorded."""
        if not rec.get("pid") and rec.get("token"):
            rec["pid"] = find_token_pid(rec["token"])
        if rec.get("started_ts") is None:
            rec["started_ts"] = now_s()

    def wait_for_cards(self, job, rec):
        since = None
        last = None
        while not self.interrupted:
            reasons = self.blockers(job)
            if not reasons:
                self.state["waiting"] = None
                return True
            reason = "; ".join(reasons)[:600]
            if since is None:
                since = iso()
            rec["status"] = "waiting"
            self.state["waiting"] = {"job": job["id"], "reason": reason, "since": since,
                                     "policy": job.get("cards_policy", "all_free")}
            self.state["current"] = None
            if reason != last:
                log("job %s waiting: %s" % (job["id"], reason))
                last = reason
            self.save()
            self.sleep(self.poll_s)
        return False

    # --- main loop
    def run(self):
        lock = open(os.path.join(self.qdir, "runner.lock"), "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            print("refused: another runner holds %s/runner.lock" % self.qdir, file=sys.stderr)
            return 5
        try:
            return self._run()
        finally:
            lock.close()

    def _run(self):
        try:
            self.load()
        except QueueError as e:
            print("refused: %s" % e, file=sys.stderr)
            return 5
        log("queue %s runner pid %d resume=%s jobs=%s" % (self.queue["id"], os.getpid(), self.resume,
                                                           [j["id"] for j in self.queue["jobs"]]))
        stopped_by = None
        while True:
            if self.interrupted:
                self.state.update(state="interrupted", reason="signal; running jobs keep running; rerun with --resume")
                self.save()
                return 3
            self.reload_queue()
            job = self.next_job()
            if job is None:
                break
            rec = self.state["jobs"][job["id"]]
            if rec["status"] == "running":  # adopted after a runner restart
                self.adopt_or_find(job, rec)
                if rec.get("pid") is None:
                    # Absence cannot distinguish death before spawn from a short job that ran
                    # and exited before its PID was saved. Never replay possible model calls.
                    cur = self.scan_log(job, rec, final=True)
                    success = cur["done"] and not cur["fail"]
                    rec.update(status="done" if success else "failed", finished=iso(),
                               reason=None if success else "launch outcome unknown; reconcile durable job evidence before retry")
                    self.state["current"] = None
                    self.save()
                    if not success and job.get("on_fail", "continue") == "stop":
                        stopped_by = job["id"]
                        break
                    continue
                self.state["current"] = job["id"]
                log("job %s adopted pid %s" % (job["id"], rec.get("pid")))
            else:
                if (job.get("requires_cards_free") or job.get("wait_for_final")) and not self.wait_for_cards(job, rec):
                    continue
                if self.launch(job, rec) is None and rec["status"] == "failed":
                    log("job %s FAILED: %s" % (job["id"], rec["reason"]))
                    if job.get("on_fail", "continue") == "stop":
                        stopped_by = job["id"]
                        break
                    continue
            verdict = self.supervise(job, rec)
            if verdict is None:
                continue  # interrupted
            status, reason = verdict
            rec.update(status=status, reason=reason, finished=iso())
            self.state["current"] = None
            self.save()
            log("job %s %s%s" % (job["id"], status.upper(), (": " + reason) if reason else ""))
            if status == "failed" and job.get("on_fail", "continue") == "stop":
                stopped_by = job["id"]
                break
        failed = [i for i, r in self.state["jobs"].items() if r["status"] == "failed"]
        if stopped_by:
            self.state.update(state="stopped", reason="job %s failed with on_fail=stop; later jobs stay pending" % stopped_by)
            code = 2
        elif failed:
            self.state.update(state="failed", reason="finished with failed jobs: %s" % failed)
            code = 4
        else:
            self.state.update(state="done", reason=None)
            code = 0
        self.state["current"], self.state["waiting"] = None, None
        self.save()
        log("QUEUE %s (exit %d)" % (self.state["state"].upper(), code))
        return code


# ---------------------------------------------------------------- dry run and status
def dry_run(queue_path, qdir):
    q = load_queue(queue_path, qdir)
    print("# card_queue dry run: queue %s, %d job(s), queue dir %s (nothing is run, no state is written)" %
          (q["id"], len(q["jobs"]), qdir))
    for i, j in enumerate(q["jobs"], 1):
        gate = ("WAIT for cards (policy %s) and no other queue job" % j.get("cards_policy", "all_free")
                if j.get("requires_cards_free") else "no card gate")
        if j.get("wait_for_final"):
            gate = "WAIT for final %s, then %s" % (j["wait_for_final"], gate)
        print("[%d/%d] %s  kind=%s  %s  on_fail=%s  timeout_s=%s%s" % (
            i, len(q["jobs"]), j["id"], j.get("kind"), gate, j.get("on_fail", "continue"), j["timeout_s"],
            "  DISABLED" if j.get("enabled", True) is False else ""))
        print("    cwd : %s" % j["cwd"])
        print("    log : %s" % j["log"])
        print("    done: %s   fail: %s" % (j.get("done_marker"), j.get("fail_marker")))
        env = "".join("%s=%s " % (k, shlex.quote(v)) for k, v in sorted((j.get("env") or {}).items()))
        print("    cmd : %s%s" % (env, " ".join(shlex.quote(x) for x in j["cmd"])))
    return 0


def format_status(s):
    c = s.get("counts", {})
    lines = ["queue %s  state=%s  runner_pid=%s  updated=%s" % (s.get("queue_id"), s.get("state"), s.get("runner_pid"),
                                                                 s.get("updated")),
             "counts: " + " ".join("%s=%s" % (k, v) for k, v in sorted(c.items()) if v)]
    if s.get("reason"):
        lines.append("reason: %s" % s["reason"])
    w = s.get("waiting")
    if w:
        lines.append("WAITING job %s since %s (policy %s): %s" % (w["job"], w["since"], w.get("policy"), w["reason"]))
    jobs = s.get("jobs", {})
    for jid in [i for i in s.get("order", list(jobs)) if i in jobs] + [i for i in jobs if i not in s.get("order", [])]:
        r = jobs[jid]
        extra = r.get("reason") or ""
        lines.append("  %-8s %-34s pid=%s started=%s finished=%s %s" % (r["status"], jid, r.get("pid"), r.get("started"),
                                                                      r.get("finished"), extra))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the queue (launch detached with nohup setsid, see the module docstring)")
    r.add_argument("--queue", required=True, help="queue file (saea-card-queue-v1)")
    r.add_argument("--queue-dir", required=True, help="state directory (state.json, runner.lock, queue.log)")
    r.add_argument("--resume", action="store_true", help="continue from <queue-dir>/state.json (adopts a running job)")
    r.add_argument("--retry-failed", action="store_true", help="with --resume: failed jobs become pending again")
    r.add_argument("--dry-run", action="store_true", help="print every command, run nothing, write nothing")
    r.add_argument("--poll-s", type=float, default=15.0)
    r.add_argument("--peer-queue-dir", action="append", default=[],
                   help="another queue's directory; a cards-free job also waits while that queue runs a job")
    s = sub.add_parser("status", help="print the queue state")
    s.add_argument("--queue-dir", required=True)
    s.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        st = read_json(os.path.join(a.queue_dir, "state.json"))
        if not st:
            print("no state in %s" % a.queue_dir, file=sys.stderr)
            return 1
        print(json.dumps(st, indent=1, sort_keys=True) if a.json else format_status(st))
        return 0
    qdir = os.path.abspath(a.queue_dir)
    if a.dry_run:
        try:
            return dry_run(a.queue, qdir)
        except QueueError as e:
            print("refused: %s" % e, file=sys.stderr)
            return 5
    os.makedirs(qdir, exist_ok=True)
    runner = Runner(a.queue, qdir, RealCardProbe(), poll_s=a.poll_s, resume=a.resume, retry_failed=a.retry_failed,
                    peers=a.peer_queue_dir)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def on_term(*_):
        runner.interrupted = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
