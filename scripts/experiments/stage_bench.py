#!/usr/bin/env python3
"""Measure one stage across shard counts and print the scaling curve.

A fixed entry point rather than a hand-assembled set of commands, so the numbers
in a report name their configuration:

    stage_bench.py --stage hash-audit --root TREE --run-root DIR \
        --shards 1,4,16,32 [--threads N] [--jobs N]

Each shard count gets its own run root, plans, runs and merges, and the curve is
written as JSON.  The merged digest is checked to be identical across shard
counts: a faster run that answers a different question is not a speedup, it is a
bug, so the benchmark fails rather than reporting one.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.experiments import stage as stages


def measure(name, root, base, shards, jobs, threads):
    run_root = Path(base) / f"shards-{shards:03d}"
    if (run_root / "stages" / name / "PLAN.json").exists():
        raise stages.StageError(f"{run_root} already holds a plan; use a fresh --run-root")
    wall_started = time.monotonic()
    manifest = stages.plan(name, str(root), shards, str(run_root))
    planned = time.monotonic()
    summary = stages.run_stage(name, str(run_root), jobs, threads)
    ran = time.monotonic()
    merged = stages.merge(name, str(run_root))
    finished = time.monotonic()
    return {"shards": shards, "items": manifest["determinism"]["keys"],
            "plan_seconds": round(planned - wall_started, 3),
            "run_seconds": summary["wall_seconds"],
            "merge_seconds": round(finished - ran, 3),
            "wall_seconds": round(finished - wall_started, 3),
            "cpu_parallelism": summary["cpu_parallelism"],
            "slowest_shard_seconds": summary["slowest_shard_seconds"],
            "peak_rss_bytes": summary["peak_rss_bytes"],
            "jobs": summary["jobs"], "threads": summary["threads_per_shard"],
            "merged_sha256": merged["merged_sha256"],
            "keys_sha256": manifest["determinism"]["keys_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="hash-audit")
    parser.add_argument("--root", required=True, help="input tree")
    parser.add_argument("--run-root", required=True, help="fresh local directory for the runs")
    parser.add_argument("--shards", default="1,4,16,32",
                        help="comma-separated shard counts (default 1,4,16,32)")
    parser.add_argument("--jobs", type=int, default=None,
                        help="parallel shard processes per point (default: one per shard)")
    parser.add_argument("--threads", type=int, default=None, help="worker threads per shard")
    parser.add_argument("--output", help="write the curve here instead of stdout")
    args = parser.parse_args(argv)
    counts = [int(value) for value in args.shards.split(",") if value.strip()]
    try:
        points = [measure(args.stage, Path(args.root).resolve(), args.run_root,
                          count, args.jobs, args.threads) for count in counts]
    except (stages.StageError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error), "state": "REJECTED"}), file=sys.stderr)
        return 2
    digests = {point["merged_sha256"] for point in points}
    baseline = next(point for point in points if point["shards"] == 1)["wall_seconds"] \
        if any(point["shards"] == 1 for point in points) else points[0]["wall_seconds"]
    for point in points:
        point["speedup_vs_single"] = round(baseline / point["wall_seconds"], 2) \
            if point["wall_seconds"] else None
    curve = {"stage": args.stage, "root": str(Path(args.root).resolve()),
             "cpus": stages.available_cpus(), "points": points,
             "identical_output": len(digests) == 1}
    text = json.dumps(curve, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if curve["identical_output"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
