#!/usr/bin/env python3
"""Fail-closed CPU audit for a registered TFRL-v2 bank release.

This command never creates or repairs a bank.  It makes the pending reasons
explicit before a flight is submitted: zero hashes, missing artifacts,
estimated token counts, missing receipt fields, cap violations and source
overlap are all reported.  Exit status is 0 only when every requested release
is consumable by the v2 rail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ZERO = "0" * 64
SHA_LEN = 64


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == SHA_LEN and value != ZERO and all(c in "0123456789abcdef" for c in value)


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error


def _content_sha(document: Mapping[str, Any]) -> str:
    raw = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def audit(path: Path, *, release: str | None = None) -> dict[str, Any]:
    doc = _read(path)
    if not isinstance(doc, Mapping):
        raise ValueError("bank release must be an object")
    iterations = doc.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        raise ValueError("bank release has no iterations")
    selected: list[tuple[str, Mapping[str, Any]]] = []
    for iteration in iterations:
        if not isinstance(iteration, Mapping) or not isinstance(iteration.get("releases"), Mapping):
            raise ValueError("bank release iteration is malformed")
        for name, value in iteration["releases"].items():
            if isinstance(name, str) and isinstance(value, Mapping) and (release is None or name == release):
                selected.append((name, value))
    if not selected:
        raise ValueError(f"release {release!r} not found" if release else "no releases selected")

    reports: list[dict[str, Any]] = []
    for name, item in selected:
        reasons: list[str] = []
        receipt = item.get("bank_receipt")
        if not isinstance(receipt, Mapping):
            reasons.append("bank_receipt missing")
            receipt = {}
        receipt_hash = receipt.get("sha256")
        if not _sha(receipt_hash):
            reasons.append("receipt hash is zero, malformed, or pending")
        measure = (item.get("measure") or {}).get("observed") if isinstance(item.get("measure"), Mapping) else None
        if measure != "vllm-tokenize":
            reasons.append(f"token measure is {measure!r}, expected 'vllm-tokenize'")
        for key in ("exemplar_bank_content_sha256", "rule_bank_content_sha256"):
            if not _sha(item.get(key)):
                reasons.append(f"{key} is zero or malformed")
        counts = item.get("token_counts")
        if not isinstance(counts, Mapping):
            reasons.append("token_counts missing")
            counts = {}
        if not isinstance(counts.get("exemplar_max_block_tokens"), int):
            reasons.append("exemplar token count is not measured")
        elif counts["exemplar_max_block_tokens"] > int((item.get("token_counts") or {}).get("exemplar_block_token_cap", 2000)):
            reasons.append("exemplar token count exceeds cap")
        if not isinstance(counts.get("rule_bank_rendered_tokens"), int):
            reasons.append("rule token count is not measured")
        elif counts["rule_bank_rendered_tokens"] > int((item.get("token_counts") or {}).get("rule_bank_token_cap", 400)):
            reasons.append("rule token count exceeds cap")
        sources = item.get("source_halves")
        consumer = item.get("consuming_roster")
        if not isinstance(sources, list) or not sources:
            reasons.append("source_halves missing")
        elif consumer in sources:
            reasons.append("consumer roster overlaps a bank source")
        relpath = receipt.get("relpath")
        if isinstance(relpath, str):
            receipt_path = path.parent.parent.parent / relpath if not Path(relpath).is_absolute() else Path(relpath)
            # A registered receipt may live outside the readiness package; do
            # not require it to exist when the release is explicitly pending,
            # but expose the missing path as a concrete blocker.
            if not receipt_path.is_file():
                reasons.append(f"receipt artifact missing: {receipt_path}")
        reports.append({"release": name, "ready": not reasons, "reasons": reasons,
                        "receipt_sha256": receipt_hash, "measure": measure,
                        "source_halves": sources, "consumer": consumer})
    return {"schema": "saea-tfrl-v2-bank-readiness-v1", "release_file": str(path),
            "ready": all(report["ready"] for report in reports), "reports": reports}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("release_file", type=Path)
    parser.add_argument("--release")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit(args.release_file, release=args.release)
    except ValueError as error:
        print(json.dumps({"ready": False, "error": str(error)}, ensure_ascii=False))
        return 2
    raw = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(raw, encoding="utf-8")
    print(raw, end="")
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
