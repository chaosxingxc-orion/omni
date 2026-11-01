"""CPU-only audit of TFRL-v2 port readiness.

The audit is intentionally conservative: draft plans and zero-sentinel bank
hashes are reported as blockers and never converted into an admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(root: Path) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    def load(name: str) -> dict[str, object] | None:
        path = root / name
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            blockers.append({"id": name, "reason": f"unreadable:{type(exc).__name__}"})
            return None
        if not isinstance(value, dict):
            blockers.append({"id": name, "reason": "not-object"})
            return None
        return value

    admission = load("root-admission-draft-v1.json")
    if admission is not None and admission.get("status") != "DECIDED":
        blockers.append({"id": "root-admission", "reason": str(admission.get("status"))})
    if admission is not None and not admission.get("deployment_profile"):
        blockers.append({"id": "deployment-profile", "reason": "missing"})

    bank = load("bank-release-v1.json")
    if bank is not None:
        text = json.dumps(bank, sort_keys=True)
        if "PENDING" in text or "0" * 64 in text:
            blockers.append({"id": "bank-release", "reason": "pending-or-zero-sentinel"})

    plan_names = sorted(root.glob("*-plan-registered-v1.json"))
    for path in plan_names:
        value = load(path.name)
        if value is not None and "DRAFT-HELD" in str(value.get("status")):
            blockers.append({"id": path.name, "reason": str(value.get("status"))})

    result = {
        "schema": "saea-tfrl-v2-admission-audit-v1",
        "root": str(root),
        "ready": not blockers,
        "model_calls": 0,
        "test_labels_read": False,
        "blockers": blockers,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if result["ready"] else 2)


if __name__ == "__main__":
    main()
