#!/usr/bin/env python3
"""Run one sharded CPU stage: ``plan``, ``run``, ``run-all``, ``merge``, ``verify``.

The stage implementation lives in :mod:`tools.experiments.stage`; this is the
fixed entry point that drives it, so no experiment has to write its own loop.

    stage.py plan    --stage hash-audit --run-root R --root TREE --shards 16
    stage.py run-all --stage hash-audit --run-root R [--jobs N]
    stage.py merge   --stage hash-audit --run-root R
    stage.py verify  --stage hash-audit --run-root R

``run-all`` starts one process per shard.  Each shard has its own dispatch id,
its own directory and its own receipt, so a shard that dies takes only itself
down: ``merge`` then refuses the whole stage rather than emitting a document
that silently lacks a slice.

Every write goes through the run root, which must be on a verified local
filesystem -- the same rule the storage policy applies to all hot state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.experiments import stage as stages
from tools.experiments.local_runtime_storage import mount_for
from tools.experiments.state import read_json


def local_run_root(path):
    """Refuse a run root that is shared, networked or read-only.

    Shard output is hot state: on a shared mount every shard's writes contend for
    the same network round trips, and on the affected hosts it is a policy
    violation outright.  The check happens before the first byte rather than
    after a long run.  Persistence of hot state is a separate gate
    (``local_runtime_storage.require_training_storage``), which training calls.
    """
    resolved = Path(path).resolve()
    record = mount_for(resolved, Path("/proc/self/mountinfo").read_text())
    if record["filesystem"] in stages.SHARED_FILESYSTEMS:
        raise stages.StageError(
            f"run root must be local storage, found {record['filesystem']} at {record['mount']}: "
            f"{resolved}; pass a local --run-root and keep shared mounts read-only")
    if "ro" in record["options"].split(","):
        raise stages.StageError(f"run root is read-only: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def _run_one(args):
    name, run_root, index, threads = args
    return stages._shard_task(args)


def run_all(name, run_root, jobs, threads):
    return stages.run_stage(name, run_root, jobs, threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "run", "run-all", "merge", "verify", "status"))
    parser.add_argument("--stage", required=True, help=f"one of {sorted(stages.STAGES)}")
    parser.add_argument("--run-root", required=True, help="local directory holding stage state")
    parser.add_argument("--root", help="plan: input tree for the stage")
    parser.add_argument("--shards", type=int, default=1, help="plan: number of shards (default 1)")
    parser.add_argument("--shard", type=int, help="run: single shard index")
    parser.add_argument("--jobs", type=int, default=None,
                        help="run-all: parallel shard processes (default: one per shard, capped at CPUs)")
    parser.add_argument("--threads", type=int, default=None, help="per-shard worker threads")
    args = parser.parse_args(argv)
    try:
        run_root = local_run_root(args.run_root)
        if args.command == "plan":
            if not args.root:
                raise stages.StageError("plan needs --root")
            result = stages.plan(args.stage, str(Path(args.root).resolve()), args.shards, run_root)
        elif args.command == "run":
            if args.shard is None:
                raise stages.StageError("run needs --shard")
            result = _run_one((args.stage, run_root, args.shard, args.threads))
        elif args.command == "run-all":
            result = run_all(args.stage, run_root, args.jobs or args.shards, args.threads)
        elif args.command == "merge":
            result = stages.merge(args.stage, run_root)
        elif args.command == "verify":
            result = stages.verify(args.stage, run_root)
        else:
            directory = Path(run_root) / "stages" / args.stage
            result = {name.lower(): read_json(directory / name)
                      for name in ("PLAN.json", "MERGE.json", "VERIFY.json")}
            result["shard_receipts"] = [read_json(path / "RECEIPT.json")
                                        for path in sorted((directory / "shards").glob("*"))]
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        failed = result.get("state") == "UNKNOWN" or result.get("ok") is False
        return 2 if failed else 0
    except (stages.StageError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"error": str(error), "state": "REJECTED"}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())