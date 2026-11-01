#!/usr/bin/env python3
"""Fail-closed CPU gate for the four-row TFRL-v2 smoke.

Bank readiness alone is insufficient: the smoke package must be flipped from
draft, use a concrete deployment profile, and have no unresolved rail pins.
This checker reads registration documents only and never contacts a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _read(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return value


def audit(package: Path, *, bank_ready: bool) -> dict[str, Any]:
    failures: list[str] = []
    root = package.resolve()
    admission = _read(root / "root-admission-draft-v1.json")
    if admission.get("status") != "DECIDED":
        failures.append(f"root admission status is {admission.get('status')!r}, expected 'DECIDED'")
    service = admission.get("service")
    if not isinstance(service, Mapping) or service.get("deployment_profile") in (None, "", "PORT_DEPLOYMENT_PROFILE (pending)"):
        failures.append("deployment profile is pending")
    if not bank_ready:
        failures.append("bank readiness audit did not pass")
    unresolved = []
    for path in sorted(root.glob("*-smoke-*-plan-registered-v1.json")):
        plan = _read(path)
        unresolved.extend(str(item) for item in plan.get("rails_pending", []) if item)
    if unresolved:
        failures.append("unresolved rails_pending: " + ", ".join(sorted(set(unresolved))))
    signoff = _read(root / "signoff-conditions-fold-b-v1.json")
    if not signoff.get("table_relpath"):
        failures.append("fold-b smoke signoff table is missing")
    return {
        "schema": "saea-tfrl-v2-smoke-admission-audit-v1",
        "package": str(package),
        "ready": not failures,
        "failures": failures,
        "model_calls": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--bank-ready", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = audit(args.package, bank_ready=args.bank_ready)
    except ValueError as exc:
        result = {"schema": "saea-tfrl-v2-smoke-admission-audit-v1", "ready": False, "failures": [str(exc)], "model_calls": 0}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
