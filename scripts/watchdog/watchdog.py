#!/usr/bin/env python3
"""Log watchdog: evaluates registered jobs on a timer and writes compact summaries.

One instance per host (--host npu2|npu|dev) evaluates only the jobs of that host from jobs.json and writes
  status-<host>.json   one small object per job (state, progress, metrics, last change)
  events-<host>.log    one line per state change or bad case: <UTC ISO> <INFO|WARN|BAD> <job> <message>
  bundles/<job>-<UTC>.json   evidence pointers (paths, line ranges, tail windows) for every BAD event
Models read the summaries (see summary.py); an analyst opens the raw evidence only through a bundle.
Python 3.8+ standard library only. To add a job kind, write a function decorated with @kind('name') (see below).
"""
import argparse, glob, json, os, re, signal, subprocess, sys, time
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
STATES = ("running", "done", "failed", "stalled", "unknown")
STATE_SEV = {"running": "INFO", "done": "INFO", "failed": "BAD", "stalled": "BAD", "unknown": "WARN"}
KINDS = {}


def kind(name):
    def deco(fn):
        KINDS[name] = fn
        return fn
    return deco


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def short(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 3] + "..."


# ---------------------------------------------------------------- environment readers (injected in tests)
class Env:
    def now(self):
        return time.time()

    def sh(self, cmd, timeout=20):
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout)
            return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    def docker_ps(self):
        """List of {id, name, image} for running containers, or None when docker is unavailable."""
        out = self.sh(["docker", "ps", "--no-trunc", "--format", "{{.ID}}|{{.Names}}|{{.Image}}"])
        if out is None:
            return None
        rows = [l.split("|", 2) for l in out.splitlines() if l.count("|") == 2]
        return [{"id": r[0], "name": r[1], "image": r[2]} for r in rows]

    def npu_smi(self):
        return self.sh(["npu-smi", "info"], timeout=30)

    def pid_container(self, pid):
        """Full docker container id of a host pid, or None (not in a container / unreadable)."""
        try:
            m = re.search(r"([0-9a-f]{64})", open("/proc/%d/cgroup" % pid).read())
            return m.group(1) if m else None
        except OSError:
            return None

    def http_get(self, url, timeout=5):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status, r.read(2000000).decode("utf-8", "replace")
        except Exception:
            return None


# ---------------------------------------------------------------- shared helpers
def scan_new(path, slot, from_end=False, cap=8000000):
    """Complete new lines of `path` since slot['off'] as [(lineno, text)]; None if missing. Advances slot."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    first = "off" not in slot
    if first or size < slot["off"]:  # new file, or truncated/rotated
        slot.update(off=0, lines=0)
    with open(path, "rb") as f:
        f.seek(slot["off"])
        data = f.read(cap)
    data = data[:data.rfind(b"\n") + 1]
    lines = data.decode("utf-8", "replace").splitlines()
    out = [(slot["lines"] + i + 1, l) for i, l in enumerate(lines)]
    slot["off"] += len(data)
    slot["lines"] += len(lines)
    if first and from_end:  # baseline: skip history, report only what appears later
        while scan_new(path, slot):
            pass
        return []
    return out


def hits(lines, rx):
    return [(n, l) for n, l in lines if rx.search(l)]


def once(st, key):
    """True the first time `key` is seen for this job (bounded memory)."""
    seen = st.setdefault("seen", [])
    if key in seen:
        return False
    seen.append(key)
    del seen[:-300]
    return True


def ev(path, a=None, b=None, tail=80, why=""):
    """Evidence pointer: lines a..b (with context) or a tail window."""
    if a is not None:
        d = {"path": path, "from_line": max(1, a - 15), "to_line": (b or a) + 15, "hit_lines": [a, b or a]}
    else:
        d = {"path": path, "tail_lines": tail}
    if why:
        d["why"] = why
    return d


def event(sev, msg, *evidence):
    return {"sev": sev, "msg": msg, "evidence": list(evidence)}


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def result(state, progress="", metrics=None, note="", sig=None, events=None, evidence=None):
    return {"state": state, "progress": progress, "metrics": metrics or {}, "note": note, "sig": sig,
            "events": events or [], "evidence": evidence or []}


def file_sig(paths):
    """(signature, newest mtime) of a file set: a changed signature means the job made progress."""
    st = [os.stat(p) for p in paths if os.path.exists(p)]
    top = max([s.st_mtime for s in st] or [0])
    return "%d:%d:%d" % (len(st), sum(s.st_size for s in st), int(top)), top


# ---------------------------------------------------------------- kind: vllm_server
VLLM_ERR = re.compile(r"Traceback|\bERROR\b|ERR9|EngineDeadError|EngineCore.*(failed|died|fatal)|out of memory|\bOOM\b")


def prom(text):
    out = {}
    for l in text.splitlines():
        if l.startswith("vllm:"):
            name, _, val = l.rpartition(" ")
            try:
                k = name.split("{")[0][5:]
                out[k] = out.get(k, 0.0) + float(val)
            except ValueError:
                pass
    return out


@kind("vllm_server")
def vllm_server(job, env, st):
    p = job["params"]
    ps = env.docker_ps()
    if ps is None:
        return result("unknown", note="docker unavailable on this host")
    log = p.get("log")
    if p["container"] not in [c["name"] for c in ps]:
        return result("failed", note="container %s is not running" % p["container"],
                      evidence=[ev(log)] if log else [])
    events, m = [], {}
    r = env.http_get("http://127.0.0.1:%s/v1/models" % p["port"])
    ok = bool(r and r[0] == 200)
    st["down"] = 0 if ok else st.get("down", 0) + 1
    if ok:
        mt = env.http_get("http://127.0.0.1:%s/metrics" % p["port"])
        g = prom(mt[1]) if mt and mt[0] == 200 else {}
        m = {"running": g.get("num_requests_running"), "waiting": g.get("num_requests_waiting"),
             "kv": round(g.get("kv_cache_usage_perc", 0), 3), "preempt": g.get("num_preemptions_total")}
        pre = g.get("num_preemptions_total")
        if pre is not None:
            old = st.get("preempt")
            if old is not None and pre > old:
                events.append(event("BAD", "preemptions increased %g -> %g" % (old, pre), *([ev(log)] if log else [])))
            st["preempt"] = pre
    if log:
        new = scan_new(log, st.setdefault("log", {}), from_end=True) or []
        bad = hits(new, VLLM_ERR)
        if bad:
            events.append(event("BAD", "%d new error line(s), first: %s" % (len(bad), short(bad[0][1], 140)),
                                ev(log, bad[0][0], bad[-1][0]), ev(log)))
    if st["down"] > p.get("down_checks", 3):
        return result("failed", note="endpoint down for %d consecutive checks" % st["down"], metrics=m,
                      events=events, evidence=[ev(log)] if log else [])
    prog = "endpoint ok run=%s wait=%s kv=%s" % (m.get("running"), m.get("waiting"), m.get("kv")) if ok \
        else "endpoint down x%d" % st["down"]
    return result("running", prog, m, events=events)


# ---------------------------------------------------------------- kind: sweep
SWEEP_ERR = re.compile(r"Traceback|\bERROR\b|\bError\b|Exception|FAILED")


def hbm_free_from_smi(smi, level):
    s = [x for x in (smi or {}).get("samples", []) if x.get("level") == level and x.get("hbm_used_mib")]
    if not s:
        return None
    return min(x.get("hbm_total_mib", 65536) - max(x["hbm_used_mib"].values()) for x in s)


@kind("sweep")
def sweep(job, env, st):
    p = job["params"]
    pre = p["run_prefix"]
    thr = {"min_p10": p.get("min_seq_tok_s", 10.0), "max_lat": p.get("max_latency_s", 420.0),
           "min_hbm": p.get("min_hbm_free_mib", 2048)}
    files = glob.glob(pre + "*")
    if not files:
        return result("unknown", note="no files for prefix yet")
    summ, smi = load_json(pre + ".json") or {}, load_json(pre + "-npu-smi.json")
    sum_by_c = {l.get("concurrency_per_instance"): l for l in summ.get("levels", [])}
    rows, events, levels_json = [], [], {}
    for f in sorted(files):
        m = re.search(re.escape(pre) + r"-L(\d+)\.json$", f)
        if m:
            levels_json[int(m.group(1))] = f
    for n, f in sorted(levels_json.items()):
        d = load_json(f)
        bad_read = st.setdefault("unreadable", {})
        if not d or not d.get("levels"):
            bad_read[f] = bad_read.get(f, 0) + 1
            if bad_read[f] >= 3 and once(st, "unreadable:" + f):
                events.append(event("BAD", "level %d report unreadable" % n, ev(f)))
            continue
        lv = d["levels"][0]
        hbm = sum_by_c.get(n, {}).get("min_hbm_free_mib_per_card", hbm_free_from_smi(smi, n))
        row = {"L": n, "tok_s": round(lv.get("output_tok_s") or 0, 1),
               "p10": round((lv.get("e2e_tok_s") or {}).get("p10") or 0, 2),
               "lat_max": round((lv.get("latency_s") or {}).get("max") or 0, 1),
               "pre": lv.get("preemptions_delta"), "err": (lv.get("errors") or 0) + (lv.get("timeouts") or 0),
               "hbm_free": hbm}
        why = []
        if row["p10"] < thr["min_p10"]:
            why.append("p10 %.1f<%g" % (row["p10"], thr["min_p10"]))
        if row["lat_max"] >= thr["max_lat"]:
            why.append("latency max %.0f>=%g" % (row["lat_max"], thr["max_lat"]))
        if row["pre"] is None or row["pre"] > 0:
            why.append("preemptions %s" % ("unreadable" if row["pre"] is None else "%g>0" % row["pre"]))
        if row["err"]:
            why.append("%d errors/timeouts" % row["err"])
        if hbm is not None and hbm < thr["min_hbm"]:
            why.append("min HBM free %d MiB<%d" % (hbm, thr["min_hbm"]))
        row["ok"] = not why
        rows.append(row)
        if why and once(st, "fail:%d" % n):
            events.append(event("BAD", "level %d fails: %s" % (n, "; ".join(why)),
                                ev(f), ev(pre + ".json"), ev(pre + "-npu-smi.json"), ev(pre + ".driver.log")))
        elif not why and once(st, "pass:%d" % n):
            events.append(event("INFO", "level %d passes: %.0f tok/s, p10 %.1f, latency max %.0f s" %
                                (n, row["tok_s"], row["p10"], row["lat_max"])))
    cur = None
    for f in sorted(files):
        m = re.search(re.escape(pre) + r"-L(\d+)\.stderr\.log$", f)
        if not m:
            continue
        n = int(m.group(1))
        new = scan_new(f, st.setdefault("err", {}).setdefault(str(n), {})) or []
        bad = hits(new, SWEEP_ERR)
        if bad and once(st, "stderr:%d" % n):
            events.append(event("BAD", "level %d stderr error: %s" % (n, short(bad[0][1], 140)),
                                ev(f, bad[0][0], bad[-1][0]), ev(f)))
        if n not in levels_json:
            cur = n
    done = bool(summ.get("done"))
    prog = "%d level(s) done %s%s" % (len(rows), [r["L"] for r in rows],
                                      "" if done or cur is None else ", running L%d" % cur)
    note = "stopped at level %s" % summ["stopped_at_level"] if summ.get("stopped_at_level") else ""
    return result("done" if done else "running", prog, {"levels": rows[-12:]}, note=note, sig=file_sig(files),
                  events=events, evidence=[ev(pre + ".driver.log")])


# ---------------------------------------------------------------- kind: pytest
PROG = re.compile(r"^([.FEsxX]+)(?:\s+\[\s*(\d+)%\])?\s*$")
SUMMARY = re.compile(r"^=*\s*(?:\d+ (?:failed|passed|skipped|errors?|warnings?|deselected|xfailed|xpassed)[^\n]*) in [\d.]+s")
COLLECT = re.compile(r"ERROR collecting|errors? during collection|Interrupted: \d+ errors?")


@kind("pytest")
def pytest_kind(job, env, st):
    files = sorted(glob.glob(job["params"]["logs"]))
    if not files:
        return result("unknown", note="no log matches %s" % job["params"]["logs"])
    events, tot, pct, ndone = [], {"passed": 0, "failed": 0, "error": 0}, [], 0
    for f in files:
        s = st.setdefault("f", {}).setdefault(f, {"p": 0, "f": 0, "e": 0, "pct": 0})
        for n, l in scan_new(f, s) or []:
            m = PROG.match(l)
            if m:
                s["p"] += m.group(1).count(".")
                s["f"] += m.group(1).count("F")
                s["e"] += m.group(1).count("E")
                s["pct"] = int(m.group(2)) if m.group(2) else s["pct"]
            elif SUMMARY.match(l):  # the summary line holds the exact totals
                c = {w.rstrip("s"): int(n) for n, w in re.findall(r"(\d+) (passed|failed|errors?)", l)}
                s.update(summary=short(l.strip("= "), 120), pct=100, p=c.get("passed", 0), f=c.get("failed", 0),
                         e=c.get("error", 0))
            elif COLLECT.search(l) and once(st, "collect:%s" % f):
                events.append(event("BAD", "collection error in %s: %s" % (os.path.basename(f), short(l, 120)),
                                    ev(f, n, n), ev(f)))
        tot["passed"] += s["p"]; tot["failed"] += s["f"]; tot["error"] += s["e"]
        pct.append(s["pct"])
        ndone += "summary" in s
        if "summary" in s and (s["f"] or s["e"]) and once(st, "finished-bad:%s" % f):
            events.append(event("WARN", "%s finished with failures: %s" % (os.path.basename(f), s["summary"]), ev(f)))
    done = ndone == len(files)
    prog = "%d%% %d passed %d failed %d error (%d/%d logs finished)" % (
        sum(pct) // len(pct), tot["passed"], tot["failed"], tot["error"], ndone, len(files))
    return result("done" if done else "running", prog, tot, sig=file_sig(files), events=events,
                  evidence=[ev(f) for f in files[:3]])


# ---------------------------------------------------------------- kind: shard_runs (regression shards)
@kind("shard_runs")
def shard_runs(job, env, st):
    """Shard dirs `<tree>/<n>` each hold files.txt (planned), progress.txt (`<file> rc=N t=S`), DONE.
    Optional params base/cand: names of two trees; a file failing in cand but passing in base is a regression."""
    p = job["params"]
    dirs = sorted(glob.glob(p["dirs"]))
    if not dirs:
        return result("unknown", note="no shard dir matches %s" % p["dirs"])
    total = done_dirs = 0
    rc, paths, events, where = {}, [], [], {}
    for d in dirs:
        def lines(name):
            try:
                with open(os.path.join(d, name)) as f:
                    return f.read().splitlines()
            except OSError:
                return []
        prog_path = os.path.join(d, "progress.txt")
        total += len(lines("files.txt"))
        done_dirs += os.path.exists(os.path.join(d, "DONE"))
        paths.append(prog_path)
        for i, l in enumerate(lines("progress.txt")):
            m = re.match(r"^(\S+) rc=(\d+)", l)
            if m:
                key = (os.path.basename(os.path.dirname(d)), m.group(1))
                rc[key] = int(m.group(2))
                where[key] = (d, prog_path, i + 1)
    bad = sorted(k for k, v in rc.items() if v != 0)
    for k in bad:
        if rc[k] == 137 and once(st, "kill:%s:%s" % k):
            events.append(event("WARN", "test file killed by timeout: %s (%s)" % (k[1], k[0]),
                                ev(where[k][1], where[k][2], where[k][2])))
    base, cand = p.get("base"), p.get("cand")
    regress = [k[1] for k in bad if k[0] == cand and rc.get((base, k[1])) == 0] if base and cand else []
    new_regress = [f for f in regress if once(st, "regress:" + f)]
    if new_regress:
        pick = [os.path.join(where[(cand, f)][0], f.replace("/", "_") + ".log") for f in new_regress[:6]]
        events.append(event("BAD", "%d file(s) fail in %s but pass in %s, e.g. %s" %
                            (len(new_regress), cand, base, new_regress[0]), *[ev(x) for x in pick]))
    per_tree = {}
    for k in bad:
        per_tree[k[0]] = per_tree.get(k[0], 0) + 1
    expect = p.get("shards", len(dirs))
    prog = "%d/%d files, %d/%d shards done, nonzero rc %s" % (len(rc), total, done_dirs, expect, per_tree or 0)
    return result("done" if done_dirs >= expect else "running", prog,
                  {"files": len(rc), "planned": total, "nonzero_rc": per_tree, "regressions": len(regress)},
                  sig=file_sig(paths), events=events, evidence=[ev(x) for x in paths[:3]])


# ---------------------------------------------------------------- kind: npu_cards
CARD1 = re.compile(r"^\|\s*(\d+)\s+(\S+)\s*\|\s*(\S+)\s*\|")
CARD2 = re.compile(r"^\|\s*\d+\s*\|\s*[0-9A-Fa-f:.]+\s*\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)")
PROC = re.compile(r"^\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|\s*(\S+)\s*\|\s*(\d+)\s*\|")


def parse_smi(text):
    cards, procs, npu = {}, {}, None
    for l in text.splitlines():
        m = PROC.match(l)
        if m:
            procs.setdefault(int(m.group(1)), []).append((int(m.group(2)), m.group(3), int(m.group(4))))
            continue
        m = CARD2.match(l)
        if m and npu is not None:  # second row of a card block: AICore and HBM
            cards[npu] = {"ai": int(m.group(1)), "hbm": int(m.group(2)), "total": int(m.group(3))}
            npu = None
            continue
        m = CARD1.match(l)
        if m:
            npu = int(m.group(1))
    return cards, procs


@kind("npu_cards")
def npu_cards(job, env, st):
    p = job.get("params", {})
    txt = env.npu_smi()
    if not txt:
        return result("unknown", note="npu-smi unavailable")
    cards, procs = parse_smi(txt)
    if not cards:
        return result("unknown", note="npu-smi output not parsed")
    ps = env.docker_ps() or []
    by_id = {c["id"]: c["name"] for c in ps}
    ours = re.compile(p.get("ours_regex", r"^saea-"))
    idle = p.get("idle_hbm_mib", 8192)
    events, cls, owners = [], {}, {}
    for n, c in sorted(cards.items()):
        names = sorted({by_id.get(env.pid_container(pid), "host") for pid, _, _ in procs.get(n, [])})
        busy = bool(procs.get(n)) or c["hbm"] > idle
        c["cls"] = "free" if not busy else ("ours" if names and all(ours.match(x) for x in names) else "other")
        cls[str(n)], owners[n] = c["cls"], names
        old = st.get("cls", {}).get(str(n))
        if old and old != c["cls"]:
            who = "%s pids %s" % (",".join(owners[n]) or "unattributed", [x[0] for x in procs.get(n, [])][:4])
            if c["cls"] == "other" and old == "free":
                events.append(event("WARN", "card %d now busy by a non-owned job (%s), HBM %d MiB" % (n, who, c["hbm"])))
            else:
                events.append(event("INFO", "card %d %s -> %s (%s)" % (n, old, c["cls"], who)))
    st["cls"] = cls
    grp = lambda k: [int(n) for n, v in sorted(cls.items(), key=lambda x: int(x[0])) if v == k]
    m = {"hbm_mib": [cards[n]["hbm"] for n in sorted(cards)], "ai_pct": [cards[n]["ai"] for n in sorted(cards)],
         "ours": grp("ours"), "other": grp("other"), "free": grp("free")}
    return result("running", "ours=%s other=%s free=%s" % (m["ours"], m["other"], m["free"]), m, events=events)


# ---------------------------------------------------------------- kind: generic_log
@kind("generic_log")
def generic_log(job, env, st):
    p = job["params"]
    f = p["path"]
    new = scan_new(f, st.setdefault("log", {}))
    if new is None:
        return result("unknown", note="log %s missing" % f)
    events = []
    bad = hits(new, re.compile(p["bad_regex"])) if p.get("bad_regex") else []
    if bad:
        events.append(event("BAD", "%d bad line(s), first: %s" % (len(bad), short(bad[0][1], 140)),
                            ev(f, bad[0][0], bad[-1][0]), ev(f)))
    if p.get("done_regex") and hits(new, re.compile(p["done_regex"])):
        st["done"] = True
    last = short(new[-1][1], 120) if new else st.get("last", "")
    st["last"] = last
    return result("done" if st.get("done") else "running", "line %d: %s" % (st["log"]["lines"], last),
                  sig=file_sig([f]), events=events, evidence=[ev(f)])


# ---------------------------------------------------------------- framework
class Watchdog:
    def __init__(self, host, jobs_file, outdir, env=None, interval=60):
        self.host, self.jobs_file, self.dir, self.env, self.interval = host, jobs_file, outdir, env or Env(), interval
        self.state_file = os.path.join(outdir, "state-%s.json" % host)
        self.state = load_json(self.state_file) or {}
        os.makedirs(os.path.join(outdir, "bundles"), exist_ok=True)

    def write_atomic(self, path, text):
        tmp = "%s.tmp%d" % (path, os.getpid())
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)

    def bundle(self, job, ev_, status, now):
        stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = os.path.join(self.dir, "bundles", "%s-%s" % (re.sub(r"[^\w.-]", "_", job["id"]), stamp))
        path, i = base + ".json", 1
        while os.path.exists(path):
            i += 1
            path = "%s-%d.json" % (base, i)
        doc = {"job": job["id"], "kind": job["kind"], "host": self.host, "time": iso(now), "severity": "BAD",
               "message": ev_["msg"], "trigger_values": {"state": status["state"], "progress": status["progress"],
                                                         "metrics": status["metrics"]},
               "evidence": ev_["evidence"]}
        self.write_atomic(path, json.dumps(doc, indent=1))
        return os.path.basename(path)

    def evaluate(self, job, now):
        st = self.state.setdefault(job["id"], {})
        try:
            res = KINDS[job["kind"]](job, self.env, st)
        except Exception as e:  # a broken evaluator must not take the watchdog down
            res = result("unknown", note="evaluator error %s: %s" % (type(e).__name__, e))
        state = res["state"]
        if res["sig"] is not None and res["sig"][0] != st.get("sig"):
            st["sig"], st["changed"] = res["sig"][0], res["sig"][1] or now
        st.setdefault("changed", now)
        quiet = now - st["changed"]
        if state == "running" and res["sig"] is not None and quiet > job.get("stall_s", 1200):
            state, res["note"] = "stalled", "no file progress for %d min" % (quiet // 60)
        events = list(res["events"])
        if state != st.get("state"):
            msg = "%s -> %s%s" % (st.get("state", "new"), state, (": " + res["note"]) if res["note"] else "")
            events.insert(0, event(STATE_SEV[state], msg + ("; " + res["progress"] if res["progress"] else ""),
                                   *res["evidence"]))
            st["state"], st["since"] = state, now
        status = {"kind": job["kind"], "state": state, "progress": short(res["progress"], 160),
                  "metrics": res["metrics"], "note": short(res["note"], 120), "checked": iso(now),
                  "last_change": iso(max(st["changed"], st["since"]))}
        return status, events

    def cycle(self):
        now = self.env.now()
        jobs = load_json(self.jobs_file) or []
        jobs = jobs.get("jobs", []) if isinstance(jobs, dict) else jobs
        mine = [j for j in jobs if j.get("host") == self.host]
        out, lines = {}, []
        for job in mine:
            status, events = self.evaluate(job, now)
            out[job["id"]] = status
            for e in events:
                tail = ""
                if e["sev"] == "BAD":
                    tail = " bundle=" + self.bundle(job, e, status, now)
                lines.append("%s %s %s %s%s" % (iso(now), e["sev"], job["id"], short(e["msg"], 300 - 60 - len(job["id"])), tail))
        doc = {"host": self.host, "updated": iso(now), "interval_s": self.interval, "jobs": out}
        text = json.dumps(doc, separators=(",", ":"))
        if len(text) > 45000:  # keep status.json small: drop metrics first
            for s in out.values():
                s["metrics"] = {}
            text = json.dumps(doc, separators=(",", ":"))
        self.write_atomic(os.path.join(self.dir, "status-%s.json" % self.host), text)
        if lines:
            log = os.path.join(self.dir, "events-%s.log" % self.host)
            if os.path.exists(log) and os.path.getsize(log) > 5000000:
                os.replace(log, log + ".1")
            with open(log, "a") as f:
                f.write("\n".join(l[:300] for l in lines) + "\n")
        for k in set(self.state) - {j["id"] for j in mine}:  # forget removed jobs
            del self.state[k]
        self.write_atomic(self.state_file, json.dumps(self.state))
        return out, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", required=True)
    ap.add_argument("--jobs", default=os.path.join(HERE, "jobs.json"))
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    w = Watchdog(a.host, a.jobs, a.dir, interval=a.interval)
    while not stop:
        t0 = time.time()
        try:
            w.cycle()
        except Exception as e:  # never die on a bad cycle; report on stderr (run/watchdog-<host>.out)
            print("%s cycle error: %s: %s" % (iso(time.time()), type(e).__name__, e), file=sys.stderr, flush=True)
        if a.once:
            break
        while not stop and time.time() - t0 < a.interval:
            time.sleep(1)


if __name__ == "__main__":
    main()
