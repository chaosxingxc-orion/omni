#!/usr/bin/env python3
"""Write a saea-runtime-receipt-v2 for a locally built llama.cpp + the pinned GGUF core.

Why this exists: core/session/receipt.py's verify_against_runtime_receipt() admits a serving
session only when the llama-server binary's path AND sha256 match the receipt exactly. The
study's committed receipt (docs/receipts/runtime-q4km.json) records an x86-64 CUDA build, so a
binary built here -- aarch64, CPU backend -- can never match it. Registering a fresh receipt for
the local build is the study's own mechanism, not a workaround: every flight in configs/model/
names its own `runtime_receipt`, and the receipt is what binds identity.

What this does NOT do: change model identity. gguf_files are copied from the pinned receipt and
re-verified against the files on disk, so the thing that determines the measurement -- the
weights -- stays byte-identical, while the binary identity is honestly recorded as different.

usage: make_runtime_receipt.py <build-bin-dir> <out.json> --pinned <runtime-q4km.json>
                              --data-root <dir> [--binary-path <receipt-relative path>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = "saea-runtime-receipt-v2"
STUDY = "speech-aware-evidence-acquisition"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("build_bin", type=Path, help="llama.cpp build/bin directory")
    ap.add_argument("out", type=Path)
    ap.add_argument("--pinned", type=Path, required=True, help="the study's committed receipt")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--binary-path", default="llama.cpp/build/bin/llama-server")
    ap.add_argument("--build-commit", default=None)
    args = ap.parse_args()

    pinned = json.loads(args.pinned.read_text(encoding="utf-8"))
    server = args.build_bin / "llama-server"
    if not server.is_file():
        print(f"no llama-server at {server}", file=sys.stderr)
        return 1

    # Model identity is inherited, never recomputed from a guess: take the pinned entries and
    # prove each one against the bytes actually on disk. A mismatch here means the requant did
    # not reproduce the core, which must stop the run rather than be papered over by a receipt.
    gguf: list[dict[str, object]] = []
    bad = 0
    for entry in pinned.get("gguf_files", []):
        relpath = str(entry["path"])
        local = args.data_root / relpath
        if not local.is_file():
            print(f"MISSING {relpath}", file=sys.stderr)
            bad += 1
            continue
        got = sha256(local)
        if got != entry["sha256"]:
            print(f"MISMATCH {relpath}\n  on disk {got}\n  pinned  {entry['sha256']}",
                  file=sys.stderr)
            bad += 1
            continue
        print(f"ok   {relpath} sha256={got}")
        gguf.append({"bytes": local.stat().st_size, "path": relpath, "sha256": got})
    if bad:
        print(f"{bad} pinned GGUF file(s) are not reproduced on disk; no receipt written",
              file=sys.stderr)
        return 1

    libraries = sorted(
        (
            {
                "bytes": item.stat().st_size,
                "path": f"llama.cpp/build/bin/{item.name}",
                "sha256": sha256(item),
            }
            for item in args.build_bin.iterdir()
            if item.is_file() and item.name != "llama-server"
        ),
        key=lambda item: item["path"],
    )

    version = ""
    try:  # the binary prints "version: N (<short sha>)" on --version
        import subprocess

        version = subprocess.run(
            [str(server), "--version"], capture_output=True, text=True, timeout=120
        ).stderr.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 -- a missing version line must not block the receipt
        pass

    receipt = {
        "gguf_files": gguf,
        "llama_cpp": {
            "binary_bytes": server.stat().st_size,
            "binary_path": args.binary_path,
            "binary_sha256": sha256(server),
            "build_commit": args.build_commit or pinned.get("llama_cpp", {}).get("build_commit"),
            "reported_build": version,
        },
        "model_lock_key": pinned.get("model_lock_key"),
        "runtime_libraries": libraries,
        "schema": SCHEMA,
        "study": STUDY,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  binary  {receipt['llama_cpp']['binary_sha256']}")
    print(f"  commit  {receipt['llama_cpp']['build_commit']}")
    print(f"  gguf    {len(gguf)} file(s), libraries {len(libraries)}")
    print("RUNTIME_RECEIPT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
