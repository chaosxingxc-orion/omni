#!/usr/bin/env python3
"""Apply the npu-port lock amendment (owner decision 9.4, 2026-10-28) to a pristine umbrella lock.

The umbrella ``docs/datasets.lock.json`` is not under version control on the Ascend hosts, so this script is the
reproducible record of the edit. It refuses unless

* the input lock is byte-identical to the pristine lock this amendment was drafted against (``BASE_LOCK_SHA256``),
* ``entries.json`` beside this script is byte-identical to the reviewed entries (``ENTRIES_SHA256``),
* none of the new names or the new profile already exists (add-only), and
* the result is byte-identical to the reviewed amended lock (``AMENDED_LOCK_SHA256``).

Edit: the four entries are appended to ``models``, the amendment record is inserted first in ``amendments`` (the list
is newest first), and the new profile is appended to ``profiles``. Nothing else changes. The lock round-trips
byte-exactly through ``json.dumps(indent=2, ensure_ascii=False)`` plus a trailing newline, which is how it is written.

usage:
  apply_amendment.py --lock <pristine datasets.lock.json> --output <path>   write the amended lock to a new file
  apply_amendment.py --lock <live datasets.lock.json> --in-place            amend in place (atomic replace)
  apply_amendment.py --lock <path> --check-amended                          verify a lock already carries it
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENTRIES = HERE / "entries.json"
BASE_LOCK_SHA256 = "eb39a1caa93605dabc25e8eb692d0d306b8a4aa572157449736cd921d286966d"
ENTRIES_SHA256 = "c3478c92b14ac753e39054aad83437a0c1d30cfdbb1d3115ef411111f006890c"
AMENDED_LOCK_SHA256 = "61f3d111b7bed7dfc191358f2ce557efd824795cc3cdbb28f32761d8b2ee20df"


class AmendmentError(RuntimeError):
    pass


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _serialise(document: dict) -> bytes:
    return (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def load_entries() -> dict:
    raw = ENTRIES.read_bytes()
    if _sha256(raw) != ENTRIES_SHA256:
        raise AmendmentError(f"{ENTRIES} sha256 {_sha256(raw)} differs from the reviewed {ENTRIES_SHA256}")
    return json.loads(raw)


def amend(base_raw: bytes, entries: dict) -> bytes:
    if _sha256(base_raw) != BASE_LOCK_SHA256:
        raise AmendmentError(
            f"input lock sha256 {_sha256(base_raw)} is not the pristine lock {BASE_LOCK_SHA256} this amendment was "
            "drafted against"
        )
    lock = json.loads(base_raw)
    if _serialise(lock) != base_raw:
        raise AmendmentError("input lock does not round-trip through the lock serialisation; refusing to rewrite it")
    existing = {
        item.get("name")
        for value in lock.values()
        if isinstance(value, list)
        for item in value
        if isinstance(item, dict)
    }
    new_names = [entry["name"] for entry in entries["models"]]
    clash = sorted(set(new_names) & existing)
    if clash:
        raise AmendmentError(f"add-only amendment: names already in the lock: {clash}")
    clash_profiles = sorted(set(entries["profiles"]) & set(lock["profiles"]))
    if clash_profiles:
        raise AmendmentError(f"add-only amendment: profiles already in the lock: {clash_profiles}")
    for name, description in entries["profiles"].items():
        lock["profiles"][name] = description
    lock["amendments"].insert(0, entries["amendment"])
    lock["models"].extend(entries["models"])
    return _serialise(lock)


def _atomic_write(path: Path, raw: bytes) -> None:
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            mode = path.stat().st_mode & 0o7777
        else:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--in-place", action="store_true")
    mode.add_argument("--check-amended", action="store_true")
    args = parser.parse_args(argv)

    raw = args.lock.read_bytes()
    if args.check_amended:
        if _sha256(raw) != AMENDED_LOCK_SHA256:
            raise AmendmentError(f"{args.lock} sha256 {_sha256(raw)} is not the amended lock {AMENDED_LOCK_SHA256}")
        print(f"{args.lock} carries the amendment: sha256 {AMENDED_LOCK_SHA256}")
        return 0
    amended = amend(raw, load_entries())
    if _sha256(amended) != AMENDED_LOCK_SHA256:
        raise AmendmentError(
            f"amended lock sha256 {_sha256(amended)} differs from the reviewed {AMENDED_LOCK_SHA256}; nothing written"
        )
    target = args.lock if args.in_place else args.output
    if not args.in_place and target.exists():
        raise AmendmentError(f"{target} exists; refusing to overwrite")
    _atomic_write(target, amended)
    print(f"{target}: {BASE_LOCK_SHA256} -> {AMENDED_LOCK_SHA256} ({len(raw)} -> {len(amended)} bytes)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AmendmentError as error:
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(2)
