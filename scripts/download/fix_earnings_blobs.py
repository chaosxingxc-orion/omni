#!/usr/bin/env python3
"""Repair a partial clone's missing blobs over plain HTTPS.

The git pack protocol to github.com hangs indefinitely on this host (verified:
`git fetch --stdin` with explicit blob oids sits for 5+ minutes with zero bytes
moved), so `git fetch --filter=blob:none --stdin` can never complete the
checkout.  Plain HTTPS to raw.githubusercontent.com and api.github.com both
work, so fetch each missing blob over HTTPS instead and insert it into the
object store under its own identity:

  1. raw.githubusercontent.com/<repo>/<sha>/<path> -- for a file GitHub does
     NOT resolve through Git LFS the response body IS the blob, so
     `git hash-object` must reproduce the exact oid.  Verified, never trusted.
  2. Pointer synthesis -- for a declared-LFS path raw.githubusercontent serves the
     RESOLVED media, so step 1 can never hash to the pointer blob.  But a pointer
     is a pure function of its payload, so download the payload, rebuild the three
     pointer lines from its own sha256 and length, and keep the result only if it
     hashes to the oid git asked for.  The payload is not thrown away either: it is
     filed into .git/lfs/objects, which is exactly where `git lfs pull` would have
     put it, so the checkout smudges real audio instead of re-downloading it.
  3. api.github.com/repos/<repo>/git/blobs/<oid> -- the real blob as base64.  Last,
     not first: it is capped at 60 requests/h unauthenticated AND it hangs outright
     on this host for the media paths (verified: 60 s with no response), apparently
     resolving the LFS payload into the JSON body.

Every insert is gated on the recomputed oid matching the requested oid, so a
wrong or truncated body is dropped rather than written.  Re-runnable: blobs
already in the store are skipped, so an interrupted run resumes for free.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# A truncated chunked body raises http.client.IncompleteRead, which is an HTTPException and a
# ValueError -- NOT an OSError -- so leaving it out of this tuple let one cut stream kill the
# whole run (observed: three silent exits mid-pass before it was caught).
NET_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    OSError,
    RuntimeError,
    http.client.HTTPException,
    ValueError,
)

RAW = "https://raw.githubusercontent.com"
API = "https://api.github.com"
UA = {"User-Agent": "speechrl-blob-repair/1"}

_lock = threading.Lock()
_api_lock = threading.Lock()
_stats = {"raw": 0, "pointer": 0, "api": 0, "failed": 0, "skipped": 0}


def git(repo: str, *args: str, stdin: bytes | None = None) -> bytes:
    out = subprocess.run(
        ["git", "-C", repo, *args],
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {out.stderr.decode()[:200]}")
    return out.stdout


def have(repo: str, oid: str) -> bool:
    """True if the object is present locally (not merely promised)."""
    out = subprocess.run(
        ["git", "-C", repo, "cat-file", "-e", oid],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
    )
    return out.returncode == 0


def missing_blobs(repo: str, rev: str, paths: list[str]) -> list[str]:
    env = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
    out = subprocess.run(
        ["git", "-C", repo, "rev-list", "--objects", "--missing=print", rev, "--", *paths],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
    ).stdout.decode(errors="replace")
    return [l[1:].split()[0] for l in out.splitlines() if l.startswith("?")]


def oid_to_path(repo: str, rev: str) -> dict[str, str]:
    out = git(repo, "ls-tree", "-r", rev).decode(errors="replace")
    table: dict[str, str] = {}
    for line in out.splitlines():
        meta, _, name = line.partition("\t")
        bits = meta.split()
        if len(bits) >= 3:
            table.setdefault(bits[2], name)
    return table


def oid_sizes(slug: str, sha: str, timeout: float) -> dict[str, int]:
    """Authoritative blob size per oid, from one recursive tree-API call.

    The size matters because raw.githubusercontent.com serves *resolved* media
    for a declared-LFS path, so asking it for a 130-byte pointer would stream a
    20 MB mp3 instead.  Knowing the blob's true length lets the raw read stop at
    exactly that many bytes: a pointer costs a pointer, and the oid check below
    still catches the case where those bytes were the head of an mp3.
    """
    try:
        doc = json.loads(get(f"{API}/repos/{slug}/git/trees/{sha}?recursive=1", timeout,
                             {"Accept": "application/vnd.github+json"}))
    except NET_ERRORS as error:
        print(f"  tree api unavailable ({error}); falling back to uncapped reads", flush=True)
        return {}
    sizes: dict[str, int] = {}
    for node in doc.get("tree", []):
        if node.get("type") == "blob" and isinstance(node.get("size"), int):
            sizes.setdefault(str(node.get("sha")), int(node["size"]))
    if doc.get("truncated"):
        print(f"  tree api truncated at {len(sizes)} blobs", flush=True)
    return sizes


def get(url: str, timeout: float, headers: dict[str, str] | None = None,
        cap: int | None = None) -> bytes:
    """GET `url`, reading at most `cap` bytes when a cap is given."""
    head = {**UA, **(headers or {})}
    if cap is not None:
        head["Range"] = f"bytes=0-{max(0, cap - 1)}"
    req = urllib.request.Request(url, headers=head)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read() if cap is None else response.read(cap)


def lfs_pointer(body: bytes) -> bytes:
    """The canonical Git LFS pointer for `body` (spec v1: sorted keys, LF endings)."""
    return (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:" + hashlib.sha256(body).hexdigest().encode() + b"\n"
        b"size " + str(len(body)).encode() + b"\n"
    )


def file_lfs_object(repo: str, body: bytes) -> None:
    """Store `body` in the repo's LFS object cache under its own content address."""
    digest = hashlib.sha256(body).hexdigest()
    target = Path(repo) / ".git" / "lfs" / "objects" / digest[:2] / digest[2:4] / digest
    if target.exists() and target.stat().st_size == len(body):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_suffix(".incoming")
    scratch.write_bytes(body)
    scratch.replace(target)


def insert(repo: str, oid: str, body: bytes) -> bool:
    """Write `body` as a blob only if it hashes to `oid`."""
    got = git(repo, "hash-object", "-t", "blob", "--stdin", stdin=body).decode().strip()
    if got != oid:
        return False
    git(repo, "hash-object", "-t", "blob", "-w", "--stdin", stdin=body)
    return have(repo, oid)


def via_api(repo: str, slug: str, oid: str, timeout: float) -> bool:
    """Fetch the exact blob from the git-blobs API, respecting the rate limit."""
    with _api_lock:  # serialize: 60/h leaves no room for concurrent bursts
        for attempt in range(3):
            try:
                doc = json.loads(get(f"{API}/repos/{slug}/git/blobs/{oid}", timeout,
                                     {"Accept": "application/vnd.github+json"}))
            except urllib.error.HTTPError as error:
                if error.code in (403, 429):
                    reset = error.headers.get("X-RateLimit-Reset")
                    nap = max(5.0, min(900.0, float(reset) - time.time() + 5)) if reset else 60.0
                    print(f"  api rate-limited, sleeping {nap:.0f}s", flush=True)
                    time.sleep(nap)
                    continue
                return False
            except NET_ERRORS:
                time.sleep(5 * (attempt + 1))
                continue
            if doc.get("encoding") != "base64":
                return False
            try:
                body = base64.b64decode(doc.get("content", ""))
            except ValueError:
                return False
            return insert(repo, oid, body)
    return False


def repair(repo: str, slug: str, sha: str, oid: str, path: str | None, timeout: float,
           size: int | None = None) -> str:
    if have(repo, oid):
        return "skipped"
    if path and size is not None:
        try:
            body = get(f"{RAW}/{slug}/{sha}/{urllib.request.quote(path)}", timeout, cap=size)
            if insert(repo, oid, body):
                return "raw"
        except NET_ERRORS:
            pass
    if path:
        # Retried on its own: this body is hashed whole, so a short read is not partial
        # progress, it is nothing -- and a cut stream is the common failure through the proxy.
        for attempt in range(3):
            try:
                payload = get(f"{RAW}/{slug}/{sha}/{urllib.request.quote(path)}", timeout)
            except NET_ERRORS:
                time.sleep(5 * (attempt + 1))
                continue
            try:
                if insert(repo, oid, payload):
                    return "raw"
                if insert(repo, oid, lfs_pointer(payload)):
                    file_lfs_object(repo, payload)
                    return "pointer"
            except (OSError, RuntimeError):
                pass
            break
    return "api" if via_api(repo, slug, oid, timeout) else "failed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--slug", default="revdotcom/speech-datasets")
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--paths", nargs="*", default=["earnings21", "earnings22"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--passes", type=int, default=4)
    args = ap.parse_args()

    sha = git(args.repo, "rev-parse", args.rev).decode().strip()
    table = oid_to_path(args.repo, sha)
    sizes = oid_sizes(args.slug, sha, args.timeout)
    print(f"repo {args.repo}\nrev  {sha}\ntree {len(table)} files, "
          f"{len(sizes)} sizes from the tree api", flush=True)

    for round_no in range(1, args.passes + 1):
        todo = missing_blobs(args.repo, sha, args.paths)
        print(f"{time.strftime('%T')} pass {round_no}: {len(todo)} missing", flush=True)
        if not todo:
            print("BLOBS_COMPLETE", flush=True)
            return 0
        for key in _stats:
            _stats[key] = 0

        def work(oid: str) -> None:
            # Never raises: one unlucky blob must cost one blob, not the pass. Anything that
            # escapes repair() is recorded as a failure and retried by the next pass.
            try:
                outcome = repair(args.repo, args.slug, sha, oid, table.get(oid), args.timeout,
                                 sizes.get(oid))
            except BaseException as error:  # noqa: BLE001
                outcome = "failed"
                with _lock:
                    print(f"  error {oid} {type(error).__name__}: {error}", flush=True)
            with _lock:
                _stats[outcome] += 1
                done = sum(_stats.values())
                if outcome == "failed":
                    print(f"  failed {oid} {table.get(oid, '<unknown path>')}", flush=True)
                elif done % 5 == 0:
                    print(f"  {done}/{len(todo)} {dict(_stats)}", flush=True)

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(work, todo))
        print(f"{time.strftime('%T')} pass {round_no} result: {dict(_stats)}", flush=True)
        if _stats["raw"] + _stats["pointer"] + _stats["api"] == 0:
            print("no progress this pass", file=sys.stderr, flush=True)
            break

    left = missing_blobs(args.repo, sha, args.paths)
    if left:
        print(f"STILL MISSING {len(left)} blobs", file=sys.stderr, flush=True)
        return 1
    print("BLOBS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
