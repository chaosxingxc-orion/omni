#!/usr/bin/env python3
"""Pure-Python stand-in for `aria2c -i <manifest> -j N` (aria2c is not installable here).

Reads the aria2 input-file format emitted by exploring-l4-intelligence/scripts/data/hf_complete.py:
a URL line followed by indented `dir=`, `out=`, `split=` options. Files run in parallel (-j, like
aria2c -j); a file with split=K is fetched as K resumable byte-range segments (like aria2c -s/-x),
split=1 streams with a single connection (the Xet-safe mode). Everything resumes from `.part`
files, so a re-run only fetches what is missing. Honours http(s)_proxy from the environment.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 1 << 20
SYNC_EVERY = 32 << 20
UA = "mt_download/1.0"
_lock = threading.Lock()
_done_bytes = 0


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def parse_manifest(path: Path) -> list[dict]:
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line[0].isspace():
            key, _, value = line.strip().partition("=")
            entries[-1][key] = value
        else:
            entries.append({"url": line.strip()})
    return entries


def remote_size(url: str, timeout: int) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        # Trust Content-Range whatever the status: ModelScope through the corporate proxy (2026-10-27)
        # answers 200 OK + "Content-Range: bytes 0-0/2871" + a 1-byte body, and reading
        # Content-Length there sized ten small files at 1 B each.
        content_range = response.headers.get("Content-Range")
        if content_range:
            return int(content_range.rsplit("/", 1)[1])
        return int(response.headers["Content-Length"])


def stream(response, handle) -> None:
    """Copy the body to `handle`, fsyncing every SYNC_EVERY bytes so that the on-disk size is
    always a valid resume point (and visible to other processes) even after a crash."""
    global _done_bytes
    unsynced = 0
    while chunk := response.read1(CHUNK):
        handle.write(chunk)
        unsynced += len(chunk)
        with _lock:
            _done_bytes += len(chunk)
        if unsynced >= SYNC_EVERY:
            handle.flush()
            os.fsync(handle.fileno())
            unsynced = 0
    handle.flush()
    os.fsync(handle.fileno())


def fetch_range(url: str, part: Path, start: int, end: int, timeout: int, tries: int) -> None:
    """Fill `part` with bytes [start, end] (inclusive), resuming from its current size."""
    want = end - start + 1
    for attempt in range(1, tries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return
        if have > want:
            part.unlink()
            have = 0
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": UA, "Range": f"bytes={start + have}-{end}"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                mode = "ab"
                content_range = response.headers.get("Content-Range")
                if content_range:
                    # A ranged body, whether labelled 206 or (ModelScope edge) 200: it must
                    # start exactly where this part resumes, or appending would corrupt it.
                    first = int(content_range.split()[-1].split("-", 1)[0])
                    if first != start + have:
                        raise OSError(f"Content-Range {content_range!r}, wanted from {start + have}")
                elif response.status == 200:
                    # Some CDN edges (seen on ModelScope through the corporate proxy, 2026-10-27) ignore Range
                    # on small files and answer 200 with the whole body. That is only usable when
                    # the whole body IS the requested range: a part starting at byte 0 whose
                    # Content-Length equals what we want. Anything else is a real error. The
                    # per-file sha256 check after download still covers both paths.
                    length = response.headers.get("Content-Length")
                    if start + have != 0 or length is None or int(length) != want:
                        raise OSError(f"expected 206, got 200 (Content-Length {length}, want {want})")
                    mode = "wb"
                elif response.status != 206:
                    raise OSError(f"expected 206, got {response.status}")
                with part.open(mode) as handle:
                    stream(response, handle)
            if part.stat().st_size == want:
                return
        except Exception as error:  # noqa: BLE001 - transport errors are retried
            log(f"retry {attempt}/{tries} {part.name}: {error}")
            time.sleep(min(60, 3 * attempt))
    raise OSError(f"giving up on {part} after {tries} tries")


def fetch_single(url: str, target: Path, timeout: int, tries: int) -> str:
    """split=1: one streaming GET per attempt, total size taken from that same response."""
    part = target.with_name(target.name + ".part0")
    for attempt in range(1, tries + 1):
        have = part.stat().st_size if part.exists() else 0
        try:
            headers = {"User-Agent": UA}
            if have:
                headers["Range"] = f"bytes={have}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status == 206:
                    size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
                    mode = "ab"
                else:
                    size = int(response.headers["Content-Length"])
                    mode = "wb"
                if target.exists() and target.stat().st_size == size:
                    return f"skip {target}"
                with part.open(mode) as handle:
                    stream(response, handle)
            if part.stat().st_size == size:
                os.replace(part, target)
                return f"OK {target} ({size} B)"
        except urllib.error.HTTPError as error:
            if error.code == 416 and have:  # .part already complete or stale; restart clean
                part.unlink()
            log(f"retry {attempt}/{tries} {target.name}: {error}")
            time.sleep(min(60, 3 * attempt))
        except Exception as error:  # noqa: BLE001 - transport errors are retried
            log(f"retry {attempt}/{tries} {target.name}: {error}")
            time.sleep(min(60, 3 * attempt))
    raise OSError(f"giving up on {target} after {tries} tries")


def fetch_file(entry: dict, root: Path, default_split: int, timeout: int, tries: int) -> str:
    url = entry["url"]
    target_dir = root / entry.get("dir", "")
    target = target_dir / (entry.get("out") or url.rsplit("/", 1)[1])
    target_dir.mkdir(parents=True, exist_ok=True)
    if int(entry.get("split", default_split)) == 1:
        return fetch_single(url, target, timeout, tries)
    for attempt in range(1, tries + 1):
        try:
            size = remote_size(url, timeout)
            break
        except Exception as error:  # noqa: BLE001
            log(f"size retry {attempt}/{tries} {target.name}: {error}")
            time.sleep(min(60, 3 * attempt))
    else:
        raise OSError(f"cannot stat {url}")
    if target.exists() and target.stat().st_size == size:
        return f"skip {target}"
    split = max(1, min(int(entry.get("split", default_split)), max(1, size // (8 * CHUNK))))
    bounds = [(i * size // split, (i + 1) * size // split - 1) for i in range(split)]
    parts = [target.with_name(f"{target.name}.part{i}") for i in range(split)]
    with ThreadPoolExecutor(max_workers=split) as pool:
        futures = [
            pool.submit(fetch_range, url, part, start, end, timeout, tries)
            for part, (start, end) in zip(parts, bounds)
        ]
        for future in futures:
            future.result()
    if split == 1:
        os.replace(parts[0], target)
    else:
        tmp = target.with_name(target.name + ".joining")
        with tmp.open("wb") as out:
            for part in parts:
                with part.open("rb") as handle:
                    while chunk := handle.read(16 * CHUNK):
                        out.write(chunk)
        os.replace(tmp, target)
        for part in parts:
            part.unlink()
    if target.stat().st_size != size:
        raise OSError(f"size mismatch {target}")
    return f"OK {target} ({size} B)"


def reporter(stop: threading.Event, interval: int) -> None:
    last, last_t = 0, time.time()
    while not stop.wait(interval):
        now = time.time()
        with _lock:
            cur = _done_bytes
        log(f"progress {cur / 1e9:.2f} GB this run, {(cur - last) / (now - last_t) / 1e6:.2f} MB/s")
        last, last_t = cur, now


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=Path, required=True)
    parser.add_argument("-d", "--dir", type=Path, default=Path.cwd())
    parser.add_argument("-j", "--jobs", type=int, default=16)
    parser.add_argument("-s", "--split", type=int, default=1, help="default when entry has none")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tries", type=int, default=10)
    parser.add_argument("--summary-interval", type=int, default=60)
    args = parser.parse_args()
    entries = parse_manifest(args.input_file)
    log(f"{len(entries)} file(s), jobs={args.jobs}, default split={args.split}, root={args.dir}")
    stop = threading.Event()
    threading.Thread(target=reporter, args=(stop, args.summary_interval), daemon=True).start()
    failed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(fetch_file, e, args.dir, args.split, args.timeout, args.max_tries): e
            for e in entries
        }
        for future, entry in futures.items():
            try:
                log(future.result())
            except Exception as error:  # noqa: BLE001
                failed += 1
                log(f"FAILED {entry['url']}: {error}")
    stop.set()
    log(f"done: {len(entries) - failed} ok, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
