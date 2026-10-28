#!/usr/bin/env python3
"""Diff a pinned ModelScope model repo against a local directory and write the
still-missing files as an aria2-style manifest for mt_download.py.

The ModelScope counterpart of the umbrella's scripts/data/hf_complete.py, and
deliberately the same shape: one recursive listing call, sha256 taken from the
listing (ModelScope serves it per file, so nothing has to be guessed), a file
kept only when its own digest reproduces, and a manifest mt_download.py can
resume from. asset_lock.py's fetch_modelscope() shells out to the `modelscope`
CLI, which is neither resumable nor file-selective -- this keeps the lock's
paths and revisions while meeting the resumability requirement.

usage: modelscope_complete.py <model_id> <dest> <manifest> [revision]
Writes an empty manifest when the local copy is already complete.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.modelscope.cn/api/v1/models"
UA = {"User-Agent": "speechrl-fetch/1.0"}
DIR_MODE = "16384"


def listing(model_id: str, revision: str, timeout: float) -> list[dict]:
    url = f"{API}/{model_id}/repo/files?Revision={urllib.parse.quote(revision)}&Recursive=true"
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if payload.get("Code") != 200:
        raise RuntimeError(f"modelscope listing failed for {model_id}: {payload.get('Message')}")
    return [f for f in payload["Data"]["Files"] if f.get("Mode") != DIR_MODE]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url(model_id: str, revision: str, relpath: str) -> str:
    return (
        f"{API}/{model_id}/repo?Revision={urllib.parse.quote(revision)}"
        f"&FilePath={urllib.parse.quote(relpath)}"
    )


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    model_id, dest, manifest = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    revision = sys.argv[4] if len(sys.argv) > 4 else "master"
    timeout = float(os.environ.get("MS_COMPLETE_TIMEOUT", "120"))
    split = os.environ.get("MS_COMPLETE_CONNECTIONS_PER_FILE", "8")
    include = [p for p in os.environ.get("MS_COMPLETE_INCLUDE", "").split(",") if p]

    files = listing(model_id, revision, timeout)
    if include:
        files = [f for f in files if any(fnmatch.fnmatch(f["Path"], p) for p in include)]
    if not files:
        raise RuntimeError(f"no files selected for {model_id} at {revision}")

    lines: list[str] = []
    complete = 0
    for entry in sorted(files, key=lambda f: f["Path"]):
        relpath = entry["Path"]
        want_sha = (entry.get("Sha256") or "").lower()
        want_size = entry.get("Size")
        local = dest / relpath
        if local.is_file():
            size_ok = want_size is None or local.stat().st_size == want_size
            if size_ok and (not want_sha or digest(local) == want_sha):
                complete += 1
                continue
            print(f"  stale, refetching: {relpath}", flush=True)
            local.unlink()
        lines.append(download_url(model_id, revision, relpath))
        lines.append(f"  dir={dest / Path(relpath).parent}")
        lines.append(f"  out={Path(relpath).name}")
        lines.append(f"  split={split}")
        lines.append(f"  max-connection-per-server={split}")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    missing = len(lines) // 5
    print(f"{model_id}@{revision}: {complete} complete, {missing} missing -> {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
