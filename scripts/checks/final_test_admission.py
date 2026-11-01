#!/usr/bin/env python3
"""Fail-closed CPU gate for the one-shot three-method full-test read.

This check only inspects committed registration documents.  It never opens
test labels, contacts a model, or starts a queue.  A registration is admitted
only when the frozen 13,078-row roster and all D/A/best-T custody records are
present and mutually named.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

ROSTER_ID = "SLURP-TEST-FULL-13078-V1"
EXPECTED_METHODS = ("direct", "agent", "best_t")


class AdmissionError(ValueError):
    pass


def _read(root: Path, rel: object, label: str) -> tuple[Path, Any]:
    if not isinstance(rel, str) or not rel or "\\" in rel:
        raise AdmissionError(f"{label}: path is not a relative POSIX path")
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts:
        raise AdmissionError(f"{label}: path escapes repository")
    path = root.joinpath(*p.parts)
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"{label}: cannot read {rel}: {exc}") from exc


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(repo_root: Path, registration_path: Path) -> dict[str, Any]:
    failures: list[str] = []
    try:
        reg = json.loads(registration_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"passed": False, "failures": [f"registration: cannot read: {exc}"]}
    if not isinstance(reg, Mapping):
        return {"passed": False, "failures": ["registration: expected object"]}
    if reg.get("schema") != "omni-final-test-registration-v1":
        failures.append("registration: wrong schema")
    if reg.get("test_tuning") is not False:
        failures.append("registration: test_tuning must be false")

    roster_path = reg.get("roster")
    roster = None
    if isinstance(roster_path, str):
        try:
            path, roster = _read(repo_root, roster_path, "roster")
            if reg.get("roster_sha256") != _sha(path):
                failures.append("roster: sha256 does not match registration")
        except AdmissionError as exc:
            failures.append(str(exc))
    else:
        failures.append("registration: roster path is required")
    if isinstance(roster, Mapping):
        if roster.get("roster_id") != ROSTER_ID:
            failures.append("roster: wrong roster_id")
        if roster.get("split_role") != "final-report":
            failures.append("roster: split_role must be final-report")
        if roster.get("status") != "FROZEN":
            failures.append("roster: status must be FROZEN")
        counts = roster.get("counts") if isinstance(roster.get("counts"), Mapping) else {}
        if counts.get("recordings") != 13078:
            failures.append("roster: recordings must equal 13078")
        if roster.get("contacts") not in (0, "0", None):
            failures.append("roster: contacts must be zero before admission")

    methods = reg.get("methods")
    if not isinstance(methods, Mapping) or tuple(methods) != EXPECTED_METHODS:
        failures.append("methods: exactly direct, agent, best_t are required")
        methods = methods if isinstance(methods, Mapping) else {}
    for name in EXPECTED_METHODS:
        method = methods.get(name)
        if not isinstance(method, Mapping):
            failures.append(f"method {name}: registration is missing")
            continue
        for field in ("config", "runtime", "session", "budget", "owner_allowance", "exposure_rows", "queue"):
            rel = method.get(field)
            if not isinstance(rel, str):
                failures.append(f"method {name}: {field} path is required")
                continue
            try:
                _read(repo_root, rel, f"method {name} {field}")
            except AdmissionError as exc:
                failures.append(str(exc))
        if method.get("run_id") in (None, ""):
            failures.append(f"method {name}: detached queue run_id is required")
        if method.get("checkpointed") is not True:
            failures.append(f"method {name}: checkpointed must be true")
        if method.get("test_labels_in_runtime") is not False:
            failures.append(f"method {name}: test_labels_in_runtime must be false")
    return {"passed": not failures, "failures": failures, "methods": list(methods)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = evaluate(args.repo_root.resolve(), args.registration.resolve())
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("PASS" if result["passed"] else "REFUSED")
        for failure in result["failures"]:
            print(f"- {failure}")
    return 0 if result["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
