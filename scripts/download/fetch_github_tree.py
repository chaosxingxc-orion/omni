#!/usr/bin/env python3
"""Materialize one GitHub commit's source tree over plain HTTPS, file by file, oid-verified.

Why not `git fetch --depth 1`: on this host that pack transfer dies with "fatal: fetch-pack:
invalid index-pack output" partway through and then restarts from zero, because a pack is one
indivisible stream -- 250 MB with no resume point, through a proxy that gives ~25 kB/s per
connection and cuts long streams. codeload tarballs have the same shape (generated on demand, no
usable Range) and failed the same way.

A tree, by contrast, is thousands of independent small objects. One recursive tree-API call gives
every path, mode, size and blob oid; each file then downloads on its own connection, retries on
its own, resumes for free (an already-correct file is skipped), and -- because a blob's oid is
sha1("blob <len>\\0" + content) -- is verified against git's own identity without needing git.
Many small connections also beat one big one on a per-connection-throttled proxy.

usage: fetch_github_tree.py <slug> <sha> <dest> [--jobs N] [--tree cached.json] [--skip PREFIX]
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import stat
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"
UA = {"User-Agent": "speechrl-tree-fetch/1"}
NET_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    OSError,
    http.client.HTTPException,
    ValueError,
)

_lock = threading.Lock()
_stats = {"fetched": 0, "skipped": 0, "failed": 0}
_bytes = 0


def blob_oid(body: bytes) -> str:
    """git's own object name for this content."""
    return hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()


def get(url: str, timeout: float, headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def load_tree(slug: str, sha: str, timeout: float, cached: Path | None) -> list[dict]:
    if cached and cached.is_file():
        document = json.loads(cached.read_text(encoding="utf-8"))
    else:
        document = json.loads(
            get(f"{API}/repos/{slug}/git/trees/{sha}?recursive=1", timeout,
                {"Accept": "application/vnd.github+json"})
        )
        if cached:
            cached.write_text(json.dumps(document), encoding="utf-8")
    if document.get("truncated"):
        raise SystemExit("tree API response is truncated; this path cannot be trusted")
    return [node for node in document.get("tree", []) if node.get("type") == "blob"]


def already_correct(target: Path, oid: str) -> bool:
    try:
        if target.is_symlink():
            return blob_oid(os.readlink(target).encode()) == oid
        return blob_oid(target.read_bytes()) == oid
    except OSError:
        return False


def write_entry(target: Path, mode: str, body: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "120000":  # a symlink's blob content IS its target path
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(body.decode())
        return
    scratch = target.with_name(target.name + ".incoming")
    scratch.write_bytes(body)
    if mode == "100755":
        scratch.chmod(scratch.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    scratch.replace(target)


def fetch_one(node: dict, slug: str, sha: str, dest: Path, timeout: float, tries: int) -> str:
    global _bytes
    path, oid, mode = node["path"], node["sha"], str(node.get("mode", "100644"))
    target = dest / path
    if already_correct(target, oid):
        return "skipped"
    url = f"{RAW}/{slug}/{sha}/{urllib.request.quote(path)}"
    for attempt in range(tries):
        try:
            body = get(url, timeout)
        except NET_ERRORS:
            time.sleep(min(30.0, 3.0 * (attempt + 1)))
            continue
        if blob_oid(body) != oid:
            # Not retryable by waiting: the bytes are wrong, not late.
            with _lock:
                print(f"  OID MISMATCH {path}", flush=True)
            return "failed"
        try:
            write_entry(target, mode, body)
        except OSError as error:
            with _lock:
                print(f"  write failed {path}: {error}", flush=True)
            return "failed"
        with _lock:
            _bytes += len(body)
        return "fetched"
    with _lock:
        print(f"  unreachable after {tries} tries: {path}", flush=True)
    return "failed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("sha")
    ap.add_argument("dest", type=Path)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max-tries", type=int, default=6)
    ap.add_argument("--tree", type=Path, default=None, help="cache file for the tree listing")
    ap.add_argument("--skip", action="append", default=[], help="skip paths with this prefix")
    ap.add_argument("--passes", type=int, default=4)
    args = ap.parse_args()

    nodes = load_tree(args.slug, args.sha, args.timeout, args.tree)
    if args.skip:
        before = len(nodes)
        nodes = [n for n in nodes if not any(n["path"].startswith(p) for p in args.skip)]
        print(f"skipping {before - len(nodes)} file(s) under {args.skip}", flush=True)
    total = sum(int(n.get("size", 0)) for n in nodes)
    print(f"{args.slug}@{args.sha[:12]} -> {args.dest}", flush=True)
    print(f"{len(nodes)} files, {total / 1e6:.1f} MB, jobs={args.jobs}", flush=True)
    args.dest.mkdir(parents=True, exist_ok=True)

    for round_no in range(1, args.passes + 1):
        for key in _stats:
            _stats[key] = 0
        started = time.time()

        def work(node: dict) -> None:
            try:
                outcome = fetch_one(node, args.slug, args.sha, args.dest,
                                    args.timeout, args.max_tries)
            except BaseException as error:  # noqa: BLE001 -- one file must not end the pass
                outcome = "failed"
                with _lock:
                    print(f"  error {node['path']}: {type(error).__name__}: {error}", flush=True)
            with _lock:
                _stats[outcome] += 1
                done = sum(_stats.values())
                if done % 250 == 0:
                    rate = _bytes / max(1e-9, time.time() - started) / 1e6
                    print(f"  {done}/{len(nodes)} {dict(_stats)} {rate:.2f} MB/s", flush=True)

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(work, nodes))
        print(f"pass {round_no}: {dict(_stats)}", flush=True)
        if not _stats["failed"]:
            print(f"TREE_COMPLETE {len(nodes)} files verified", flush=True)
            return 0
    print(f"{_stats['failed']} file(s) still unverified", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
