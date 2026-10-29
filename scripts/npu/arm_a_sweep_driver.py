#!/usr/bin/env python3
"""PRELIMINARY engine measurement (plan arm A, not the recorded locked baseline): run the study's synthetic sweep one
level at a time against the resident instances, sample npu-smi every 30 s, and stop at the first level that fails a
plan 3.4 threshold. Run on the NPU host (host python3, standard library only). Reads no dataset row.

usage: python3 arm_a_sweep_driver.py [--resume] <out-prefix> <levels e.g. 8,16,24,32,48> <base-url> [<base-url> ...]
Writes <out-prefix>-L<level>.json per level (the sweep tool's own report), <out-prefix>-L<level>.row.json (the level's
checkpoint row, written the moment the level finishes), <out-prefix>.json (merged summary) and
<out-prefix>-npu-smi.json (per-card samples). Run it detached (nohup, setsid, no tty). With --resume, levels that
already have a checkpoint row are skipped (their rows and samples are reloaded) and a recorded failing level stops the
run again, so a dead driver is relaunched with the same arguments plus --resume and continues at the next level.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

TREE = "/mnt/cephfs/xingchao/omni/work/dev/umbrella/studies/speech-aware-evidence-acquisition"
ROOT = "/mnt/cephfs/xingchao/omni"
IMAGE = "saea-base:20261027"
SEED = "20261028"
MODEL = "qwen3-omni-30b-a3b-thinking"
SAMPLE_SECONDS = 30
MIN_FREE_MIB = 2048
MAX_LATENCY_S = 420.0
MIN_P10 = 10.0
CLIENT_CPUSET = "224-255"  # NUMA node 7: no instance is pinned there

ROW1 = re.compile(r"^\|\s*(\d+)\s+\S+\s+\|\s*OK\s*\|")
ROW2 = re.compile(r"^\|\s*\d+\s+\|\s*[0-9A-Fa-f:.]+\s+\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|")


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


class Sampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.level: int | None = None
        self.samples: list[dict] = []

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                cards = read_cards()
                self.samples.append({"t": time.time(), "level": self.level,
                                     "aicore_pct": {str(k): v[0] for k, v in cards.items()},
                                     "hbm_used_mib": {str(k): v[1] for k, v in cards.items()},
                                     "hbm_total_mib": next(iter(cards.values()))[2] if cards else None})
            except Exception as error:  # a failed sample must not kill the sweep
                self.samples.append({"t": time.time(), "level": self.level, "error": repr(error)[:200]})
            self.stop.wait(SAMPLE_SECONDS)


def run_level(prefix: str, level: int, urls: list[str]) -> Path:
    out = Path(f"{prefix}-L{level}.json")
    uid, gid = os.getuid(), os.getgid()
    name = f"saea-sweep-{Path(prefix).name}-L{level}"
    # A client container left by a dead driver would still be loading the servers; remove it before (re)running.
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    command = ["docker", "run", "--rm", "--name", name, "--network", "host", "--user", f"{uid}:{gid}", "-e", "HOME=/tmp",
               "--cpuset-cpus", CLIENT_CPUSET, "-v", f"{ROOT}:{ROOT}", "-w", TREE,
               "-e", f"PYTHONPATH={TREE}/src", "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint",
               "/opt/saea-venv/bin/python", IMAGE, "scripts/serving/synthetic_sweep.py"]
    for url in urls:
        command += ["--base-url", url]
    command += ["--model", MODEL, "--levels", str(level), "--seed", SEED,
                "--label", "PRELIMINARY arm A engine measurement, not the recorded locked baseline",
                "--out", str(out)]
    log = open(f"{prefix}-L{level}.stderr.log", "w")
    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 0 or not out.exists():
        raise SystemExit(f"level {level}: sweep exited {code}; see {prefix}-L{level}.stderr.log")
    return out


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def window_samples(samples: list[dict], start: float, end: float) -> list[dict]:
    return [s for s in samples if start <= s["t"] <= end and "error" not in s]


def main() -> int:
    args = sys.argv[1:]
    resume = "--resume" in args
    args = [a for a in args if a != "--resume"]
    prefix, levels, urls = args[0], [int(x) for x in args[1].split(",")], args[2:]
    sampler = Sampler()
    samples_path = Path(f"{prefix}-npu-smi.json")
    if resume and samples_path.exists():
        sampler.samples = json.loads(samples_path.read_text())["samples"]
    rows, reports = [], {}
    stopped_at = None

    def save_samples() -> None:
        write_atomic(samples_path, json.dumps({"interval_s": SAMPLE_SECONDS, "samples": list(sampler.samples)},
                                              separators=(",", ":")) + "\n")

    def save_summary(done: bool) -> None:
        write_atomic(Path(f"{prefix}.json"), json.dumps({
            "label": "PRELIMINARY arm A engine measurement (4 x TP2, max-num-seqs 64), not the recorded locked baseline",
            "synthetic": True, "dataset_rows": 0, "seed": SEED, "base_urls_count": len(urls),
            "stopped_at_level": stopped_at, "done": done, "levels": rows}, indent=2, sort_keys=True) + "\n")

    sampler.start()
    finished = False
    try:
        for level in levels:
            row_path = Path(f"{prefix}-L{level}.row.json")
            if resume and row_path.exists():
                row = json.loads(row_path.read_text())
                rows.append(row)
                print(f"level {level}: resumed from {row_path.name}, pass={row['passes']}", flush=True)
                if not row["passes"]:
                    stopped_at = level
                    break
                continue
            sampler.level = level
            t0 = time.time()
            out = run_level(prefix, level, urls)
            t1 = time.time()
            report = json.loads(out.read_text())
            reports[level] = hashlib.sha256(out.read_bytes()).hexdigest()
            summary = report["levels"][0]
            wall = summary["wall_s"]
            generated = dt.datetime.fromisoformat(report["generated_at"]).timestamp()
            load = window_samples(sampler.samples, generated - wall - 1, generated + 2)
            everything = window_samples(sampler.samples, t0, t1)
            free = [s["hbm_total_mib"] - used for s in everything for used in s["hbm_used_mib"].values()]
            ai = [v for s in load for v in s["aicore_pct"].values()]
            gauges = summary.get("gauges") or []
            kv_peak = max((g["kv_cache_usage_perc"]["max"] for g in gauges
                           if isinstance(g, dict) and g.get("kv_cache_usage_perc")), default=None)
            running_peak = max((g["num_requests_running"]["max"] for g in gauges
                                if isinstance(g, dict) and g.get("num_requests_running")), default=None)
            waiting_peak = max((g["num_requests_waiting"]["max"] for g in gauges
                                if isinstance(g, dict) and g.get("num_requests_waiting")), default=None)
            lat, e2e = summary["latency_s"], summary["e2e_tok_s"]
            reasons = []
            if summary["errors"] or summary["timeouts"]:
                reasons.append(f"{summary['errors']} errors, {summary['timeouts']} timeouts")
            if lat["max"] is None or lat["max"] >= MAX_LATENCY_S:
                reasons.append(f"max latency {lat['max']} >= {MAX_LATENCY_S}")
            if e2e["p10"] is None or e2e["p10"] < MIN_P10:
                reasons.append(f"p10 e2e {e2e['p10']} < {MIN_P10}")
            if summary["preemptions_delta"] is None:
                reasons.append("preemption counter unreadable")
            elif summary["preemptions_delta"] > 0:
                reasons.append(f"{summary['preemptions_delta']:g} preemption(s)")
            min_free = min(free) if free else None
            if min_free is None or min_free < MIN_FREE_MIB:
                reasons.append(f"min HBM free {min_free} MiB < {MIN_FREE_MIB}")
            row = {
                "concurrency_per_instance": level, "instances": len(urls), "requests": summary["requests"],
                "ok": summary["ok"], "errors": summary["errors"], "timeouts": summary["timeouts"],
                "wall_s": wall, "aggregate_output_tok_s": summary["output_tok_s"],
                "completion_tokens": summary["completion_tokens"],
                "e2e_tok_s": e2e, "decode_tok_s": summary["decode_tok_s"], "latency_s": lat, "ttft_s": summary["ttft_s"],
                "preemptions_delta": summary["preemptions_delta"],
                "prefix_cache_queries_delta": summary["metrics_delta"].get("prefix_cache_queries"),
                "prefix_cache_hits_delta": summary["metrics_delta"].get("prefix_cache_hits"),
                "kv_cache_usage_peak": kv_peak, "num_requests_running_peak": running_peak,
                "num_requests_waiting_peak": waiting_peak,
                "min_hbm_free_mib_per_card": min_free,
                "aicore_pct_mean_during_load": sum(ai) / len(ai) if ai else None,
                "aicore_pct_max_during_load": max(ai) if ai else None,
                "samples_in_load_window": len(load), "report": out.name, "report_sha256": reports[level],
                "passes": not reasons, "failing_reasons": reasons,
            }
            rows.append(row)
            write_atomic(row_path, json.dumps(row, indent=2, sort_keys=True) + "\n")  # the level checkpoint
            save_samples()
            print(f"level {level}: pass={not reasons} {reasons}", flush=True)
            if reasons:
                stopped_at = level
            save_summary(False)
            if reasons:
                break
        finished = True
    finally:
        sampler.stop.set()
        sampler.join(timeout=70)
        save_samples()
        save_summary(finished)  # "done" is true only when the loop ended normally (all levels or a failing level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
