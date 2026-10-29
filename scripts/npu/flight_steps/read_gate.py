#!/usr/bin/env python3
"""Refuse unless a recipe's READ run id has a root-countersigned exposure-ledger row (read-only; run inside saea-base).

usage (cwd = study root, PYTHONPATH = study src, as port_py.sh sets): read_gate.py --recipe PATH [--ledger docs/exposure-ledger.md]
Same rule as scripts/flights/batch/read_lib.sh `read_require_countersigned`, loaded the same way (by file path).
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path


def read_run_id(recipe: dict) -> str | None:
    value = recipe.get("read_run_id")
    if isinstance(value, str) and value:
        return value
    run_ids = recipe.get("run_ids")
    if isinstance(run_ids, dict) and isinstance(run_ids.get("read"), str):
        return run_ids["read"]
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--ledger", default="docs/exposure-ledger.md")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    recipe_path = Path(args.recipe)
    if not recipe_path.is_absolute():
        recipe_path = repo / recipe_path
    try:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"REFUSED: cannot read the recipe {recipe_path}: {error}", file=sys.stderr)
        return 1
    run_id = read_run_id(recipe)
    if run_id is None:
        print("REFUSED: the recipe declares no READ run id (read_run_id or run_ids.read)", file=sys.stderr)
        return 1
    spec = importlib.util.spec_from_file_location("frc", repo / "scripts/checks/flight_registration_check.py")
    gate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gate  # the gate's dataclasses resolve their module through sys.modules
    spec.loader.exec_module(gate)
    rows = gate.parse_ledger(repo, Path(args.ledger)).get(run_id, [])
    if not rows:
        print(f"REFUSED: the READ row {run_id} is not in the exposure ledger", file=sys.stderr)
        return 1
    if not any(gate.is_countersigned(row) for row in rows):
        print(f"REFUSED: the READ row {run_id} is not root countersigned", file=sys.stderr)
        return 1
    print(f"READ row {run_id} countersigned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
