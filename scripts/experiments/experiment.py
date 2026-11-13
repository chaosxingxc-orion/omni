#!/usr/bin/env python3
"""Unified local deployment CLI for configuration-driven experiment execution."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.experiments.config import ConfigError, load_config
from tools.experiments import runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "plan", "prepare", "run", "status", "reconcile"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("cpu", "npu", "all"), default="all")
    parser.add_argument("--_controller", help=argparse.SUPPRESS)
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    parser.add_argument("--_guard-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_lane-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_dispatch-id", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        doc = load_config(args.config)
        if args._controller or args._worker:
            if args.command != "run" or (args._controller and args._worker):
                raise ConfigError("invalid private invocation")
            if args._controller:
                if args._guard_fd is None:
                    raise ConfigError("controller requires inherited guard")
                return runtime.controller(doc, __file__, args._controller, args._guard_fd, args.phase)
            if not args._dispatch_id:
                raise ConfigError("worker requires dispatch id")
            return runtime.worker(doc, args._worker, args._dispatch_id, args._lane_fd)
        if args.command == "validate":
            result = {"valid": True, "schema_version": 1, "config_sha256": doc["sha256"]}
        elif args.command == "plan":
            result = runtime.plan(doc, args.phase)
        elif args.command == "prepare":
            result = runtime.prepare(doc, args.phase)
        elif args.command == "run":
            result = runtime.launch(doc, __file__, args.phase)
        elif args.command == "status":
            result = runtime.status(doc, args.phase)
        else:
            result = runtime.reconcile(doc, args.phase)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (ConfigError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"error": str(error), "state": "UNKNOWN" if args.command in ("status", "reconcile") else "REJECTED"},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
