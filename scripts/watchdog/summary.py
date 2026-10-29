#!/usr/bin/env python3
"""Merge status-<host>.json into status.json and print a one-screen digest (what the coordinator reads).
usage: summary.py [--dir DIR] [--events N] [--quiet]"""
import argparse, glob, json, os, re
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(t):
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--events", type=int, default=12, help="how many recent WARN/BAD events to show")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    hosts, jobs = {}, {}
    for f in sorted(glob.glob(os.path.join(a.dir, "status-*.json"))):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        age = int((now - parse(d["updated"])).total_seconds())  # hosts clocks may differ by ~2 min
        hosts[d["host"]] = {"updated": d["updated"], "age_s": age, "stale": age > 3 * d.get("interval_s", 60) + 150}
        for jid, j in d["jobs"].items():
            jobs[jid] = dict(j, host=d["host"])
    doc = {"generated": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "hosts": hosts, "jobs": jobs}
    tmp = os.path.join(a.dir, "status.json.tmp%d" % os.getpid())
    with open(tmp, "w") as f:
        f.write(json.dumps(doc, separators=(",", ":")))
    os.replace(tmp, os.path.join(a.dir, "status.json"))
    if a.quiet:
        return
    print("WATCHDOG DIGEST %s  hosts: %s" % (doc["generated"], "  ".join(
        "%s %s(%ds)" % (h, "STALE" if v["stale"] else "ok", v["age_s"]) for h, v in sorted(hosts.items()))))
    for jid, j in sorted(jobs.items(), key=lambda x: (x[1]["state"] in ("done", "running"), x[0])):
        age = int((now - parse(j["last_change"])).total_seconds() / 60)
        print("%-8s %-19s %-5s %s%s [chg %dm ago]" % (j["state"].upper(), jid, j["host"], j["progress"][:110],
                                                       (" | " + j["note"]) if j["note"] else "", age))
        for r in (j["metrics"].get("levels") or []):
            print("           L%-3d %6.0f tok/s  p10 %5.1f  lat_max %5.0f s  preempt %s  hbm_free %s  %s" % (
                r["L"], r["tok_s"], r["p10"], r["lat_max"], r["pre"], r["hbm_free"], "ok" if r["ok"] else "FAIL"))
    ev = []
    for f in glob.glob(os.path.join(a.dir, "events-*.log")):
        with open(f, errors="replace") as fh:
            ev += [l.rstrip() for l in fh if re.match(r"\S+ (WARN|BAD) ", l)]
    ev.sort()
    print("RECENT WARN/BAD EVENTS (%d of %d):" % (min(len(ev), a.events), len(ev)))
    for l in ev[-a.events:]:
        print("  " + l[:240])
    if not ev:
        print("  none")
    print("evidence bundles: %d in %s/bundles" % (len(glob.glob(os.path.join(a.dir, "bundles", "*.json"))), a.dir))


if __name__ == "__main__":
    main()
