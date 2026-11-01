#!/usr/bin/env python3
"""Validate a provenance bridge from CPU preparation to a TFRL port cell.

The bridge is metadata only. It never renames a run, changes a receipt, or
admits generation. A CPU-isolated preparation may be consumed by a port cell
only when the original identity, protocol hash, receipt hash and pinned
content remain intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _obj(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return value


def _pin(root: Path, pin: Mapping[str, Any], label: str) -> bytes:
    rel = pin.get("relpath")
    expected = pin.get("sha256")
    if not isinstance(rel, str) or Path(rel).is_absolute() or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} is not a relative sha256 pin")
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes its root")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} hash mismatch")
    return raw


def validate(bridge: Mapping[str, Any], *, data_root: Path) -> dict[str, Any]:
    if bridge.get("schema") != "saea-tfrl-cpu-preparation-bridge-v1":
        raise ValueError("unknown bridge schema")
    source = bridge.get("source")
    target = bridge.get("target")
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise ValueError("source and target objects are required")
    source_run = source.get("run_id")
    target_run = target.get("run_id")
    if not isinstance(source_run, str) or not isinstance(target_run, str):
        raise ValueError("source and target run ids are required")
    if "CPU-ISOLATED" not in source_run:
        raise ValueError("source run is not the CPU-isolated preparation identity")
    if "NPU-PORT" not in target_run:
        raise ValueError("target run is not an NPU-port identity")
    if source_run == target_run:
        raise ValueError("source and target run identities must remain distinct")
    raw_receipt = _pin(data_root, source.get("receipt", {}), "source.receipt")
    receipt = json.loads(raw_receipt)
    if not isinstance(receipt, Mapping):
        raise ValueError("source receipt must contain an object")
    if receipt.get("run_id") != source_run:
        raise ValueError("source receipt run_id differs from bridge")
    protocol_hash = source.get("protocol_sha256")
    if receipt.get("protocol_sha256") != protocol_hash:
        raise ValueError("source protocol hash differs from receipt")
    if receipt.get("generation_admitted") is not False:
        raise ValueError("CPU preparation receipt must keep generation_admitted=false")
    for index, pin in enumerate(source.get("content", [])):
        if not isinstance(pin, Mapping):
            raise ValueError(f"source.content[{index}] is not a pin")
        _pin(data_root, pin, f"source.content[{index}]")
    if target.get("source_run_id") != source_run or target.get("source_protocol_sha256") != protocol_hash:
        raise ValueError("target does not bind the original source identity/protocol")
    if bridge.get("generation_admitted") is not False:
        raise ValueError("bridge cannot admit generation")
    return {
        "schema": "saea-tfrl-cpu-preparation-bridge-audit-v1",
        "ready": True,
        "source_run_id": source_run,
        "target_run_id": target_run,
        "receipt_sha256": source["receipt"]["sha256"],
        "generation_admitted": False,
        "content_pin_count": len(source.get("content", [])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bridge", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate(_obj(args.bridge), data_root=args.data_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema": "saea-tfrl-cpu-preparation-bridge-audit-v1", "ready": False, "failure": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
