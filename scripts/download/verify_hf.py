#!/usr/bin/env python3
"""Verify a pinned HF snapshot against the repo's Git LFS sha256 oids.

usage: verify_hf.py <repo_id> <dest> <revision> <model|dataset>
Honours HF_COMPLETE_INCLUDE (the same comma-separated fnmatch list the repo's own
hf_complete.py differ uses), so a file-selective asset is verified over exactly the files it
was meant to fetch -- without it every deliberately-skipped sibling would be reported missing
and a complete asset could never pass (the GGUF repos are >110 GB whole, 19.3 GB selected).
Files with an LFS oid are sha256-hashed; others are size-checked. Mismatched files are deleted
so the next fetch round re-downloads them. Verified (path, size, mtime) triples are cached in
<dest>/.hfd/verified.tsv so an interrupted or repeated verification resumes instead of re-hashing.
Exit 0 when everything matches, 1 otherwise.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    repo, dest_text, revision, repo_type = sys.argv[1:5]
    dest = Path(dest_text)
    api = HfApi(endpoint=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    info = (api.dataset_info if repo_type == "dataset" else api.model_info)(
        repo, revision=revision, files_metadata=True
    )
    include = [p for p in os.environ.get("HF_COMPLETE_INCLUDE", "").split(",") if p]
    siblings = [
        sibling
        for sibling in info.siblings
        if not include
        or any(fnmatch.fnmatch(sibling.rfilename, pattern) for pattern in include)
    ]
    if include and not siblings:
        print(f"HF_COMPLETE_INCLUDE matched no file in {repo}", file=sys.stderr, flush=True)
        return 1
    cache_path = dest / ".hfd" / "verified.tsv"
    cache: set[str] = set()
    if cache_path.exists():
        cache = set(cache_path.read_text(encoding="utf-8").splitlines())

    def check(sibling) -> tuple[str, bool, str]:
        path = dest / sibling.rfilename
        try:
            stat = path.stat()
        except OSError:
            return sibling.rfilename, False, "missing"
        key = f"{sibling.rfilename}\t{stat.st_size}\t{int(stat.st_mtime)}"
        if key in cache:
            return key, True, "cached"
        if sibling.size is not None and stat.st_size != sibling.size:
            return key, False, f"size {stat.st_size} != {sibling.size}"
        lfs = getattr(sibling, "lfs", None)
        expected = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        if expected and sha256(path) != expected:
            return key, False, "sha256 mismatch"
        return key, True, "sha256" if expected else "size"

    bad = 0
    with ThreadPoolExecutor(max_workers=8) as pool, cache_path.open("a", encoding="utf-8") as out:
        for key, ok, how in pool.map(check, siblings):
            if ok:
                if how != "cached":
                    out.write(key + "\n")
                    out.flush()
                continue
            bad += 1
            name = key.split("\t", 1)[0]
            print(f"BAD {name}: {how}", flush=True)
            if how != "missing":
                (dest / name).unlink(missing_ok=True)
    print(f"VERIFY repo={repo} revision={info.sha} files={len(siblings)}"
          f"{'/' + str(len(info.siblings)) + ' selected' if include else ''} bad={bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
