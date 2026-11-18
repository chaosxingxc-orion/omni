"""Deterministic sharded stages: plan, run one shard, merge, verify.

A stage is CPU work whose answer does not depend on how it is split.  The driver
freezes the work list once (``plan``), hashes one deterministic slice per shard
(``run_shard``), concatenates the slices back into stable-key order (``merge``)
and re-derives the result from the inputs rather than from the receipts
(``verify``).

Two properties matter more than speed, and the tests pin both:

* ``shards`` never changes the bytes of the merged output.  One shard and
  thirty-two shards produce the same file, so an old single-process result stays
  comparable with a new parallel one.
* A single unsettled shard makes the whole stage unsettled.  There is no partial
  merge: a reader never sees a document that silently lacks a slice.

Shards are independent: each has its own dispatch id, receipt and output
directory, and every write goes through :class:`~tools.experiments.state.Store`,
which refuses any path outside the run root.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import resource
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from .config import ConfigError, canonical
from .state import Store, read_json, now


SCHEMA = "omni-stage-v1"
CHUNK = 1024 * 1024

# Filesystems where stage output is never acceptable.  The set names shared and
# network filesystems rather than an allowlist of local ones: a stage result is
# derived, reproducible data, so the invariant worth enforcing is "shards write
# only to local storage" -- which is what the affected hosts must never violate
# by writing Ceph -- not "the operator's disk is one of three types".  A
# container's own rootfs is overlay, and refusing it would break the fixed image.
SHARED_FILESYSTEMS = frozenset({
    "ceph", "nfs", "nfs4", "cifs", "smb3", "smbfs", "glusterfs", "lustre",
    "gpfs", "beegfs", "9p", "fuse.ceph", "fuse.glusterfs", "fuse.sshfs", "afs",
})


class StageError(ConfigError):
    """A plan, shard or merged result violated the stage contract."""


# -- resource quota ----------------------------------------------------------------------------

def available_cpus():
    """The CPU count this process may actually use, honouring cgroup limits.

    ``os.cpu_count()`` and ``nproc`` report the host's cores; inside a container
    that is routinely wrong in the expensive direction.  The affinity mask is
    authoritative, and a cgroup quota can be stricter still.
    """
    count = len(os.sched_getaffinity(0))
    for path, scale in (("/sys/fs/cgroup/cpu.max", 0), ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", 1)):
        try:
            if scale == 0:
                quota, period = Path(path).read_text().split()[:2]
                if quota != "max":
                    count = min(count, max(1, -(-int(quota) // int(period))))
            else:
                quota = int(Path(path).read_text())
                period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
                if quota > 0:
                    count = min(count, max(1, -(-quota // period)))
        except (OSError, ValueError, IndexError):
            continue
    return max(1, count)


def peak_rss_bytes():
    """Peak resident set of this process, in bytes.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS; the platform check
    keeps a receipt from silently reporting a 1024x-wrong figure.  A shard's
    memory has to stay flat in the size of the stage, so this is recorded next
    to its timing rather than measured separately.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if os.uname().sysname == "Darwin" else peak * 1024


# -- the stage contract ------------------------------------------------------------------------

class Stage:
    """Four methods, no state.  A subclass names itself and nothing else."""

    name = ""

    def keys(self, params):
        """Yield stable string keys for every unit of work.  Order is irrelevant."""
        raise NotImplementedError

    def record(self, key, params):
        """Return one JSON-serialisable record.  Must contain ``key``."""
        raise NotImplementedError

    def verify_record(self, key, record, params):
        """Return True when ``record`` still matches the input.  Never trusts the record."""
        raise NotImplementedError


class HashAudit(Stage):
    """Every file beneath one root, with content digest and identity metadata.

    This is the ``data.audit`` component: the answer is a manifest, the work is
    I/O plus sha256, and the natural split is by file.  Symbolic links are
    recorded by target and never followed, so an audit cannot escape its tree.
    """

    name = "hash-audit"

    def keys(self, root):
        root = Path(root)
        if not root.is_dir():
            raise StageError(f"hash-audit root is not a directory: {root}")
        found = []

        def refused(error):
            # An unreadable directory cannot be enumerated, and ``os.walk`` would
            # skip it without a word.  It becomes a key of its own: the manifest
            # then names the gap instead of claiming the tree was covered, and one
            # root-owned subdirectory does not stop a multi-million-file audit.
            found.append(str(Path(error.filename).relative_to(root)))

        for directory, subdirectories, files in os.walk(root, followlinks=False, onerror=refused):
            subdirectories.sort()
            for name in files:
                key = str(Path(directory, name).relative_to(root))
                if "\n" in key or "\0" in key:
                    raise StageError(f"path is not representable in the key file: {key!r}")
                found.append(key)
        # Sorting here is the point: directory enumeration order differs between
        # filesystems and between hosts, and the plan digest must not.
        return sorted(found)

    def record(self, key, root):
        path = Path(root) / key
        info = path.lstat()
        if path.is_symlink():
            return {"key": key, "kind": "symlink", "target": os.readlink(path)}
        if os.path.isdir(path):
            # Only an unreadable directory is a key, so this either confirms the
            # gap the plan found or reports that it has since been opened up --
            # which is drift, and verification is what notices it.
            try:
                with os.scandir(path) as entries:
                    next(entries, None)
            except OSError as error:
                return {"key": key, "kind": "unreadable", "error": error.strerror}
            return {"key": key, "kind": "directory"}
        if not os.path.isfile(path):
            return {"key": key, "kind": "special"}
        checksum = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(CHUNK), b""):
                checksum.update(chunk)
        return {"key": key, "kind": "file", "sha256": checksum.hexdigest(),
                "size": info.st_size, "mtime_ns": info.st_mtime_ns,
                "mode": info.st_mode & 0o777}

    def verify_record(self, key, record, root):
        return self.record(key, root) == record


STAGES = {stage.name: stage for stage in (HashAudit(),)}


def stage_for(name):
    if name not in STAGES:
        raise StageError(f"unknown stage: {name}; known: {sorted(STAGES)}")
    return STAGES[name]


# -- plan --------------------------------------------------------------------------------------

def _stream_write(store, filename, chunks):
    """Write an arbitrarily large artifact atomically and return its sha256."""
    filename = store.checked(filename)
    store.mkdir(filename.parent)
    temporary = filename.parent / f".{filename.name}.{uuid.uuid4().hex}.tmp"
    checksum = hashlib.sha256()
    try:
        fd = os.open(store.checked(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            try:
                for chunk in chunks:
                    checksum.update(chunk)
                    stream.write(chunk)
            finally:
                # A source generator holds a thread pool; close it deterministically
                # instead of waiting for the collector, or a failed stage keeps
                # worker threads alive after the error has already been raised.
                close = getattr(chunks, "close", None)
                if close is not None:
                    close()
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, filename)
        directory = os.open(filename.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return checksum.hexdigest()


def stage_dir(run_root, name):
    return Store(run_root).checked(Path(run_root) / "stages" / name)


def key_file(run_root, name, index):
    return stage_dir(run_root, name) / "shards" / f"{index:03d}" / "KEYS.txt"


def read_keys(filename):
    with open(filename, "r", encoding="utf-8", errors="surrogateescape") as stream:
        for line in stream:
            yield line[:-1] if line.endswith("\n") else line


def plan(name, params, shards, run_root):
    """Freeze the work list once and hand every shard its own key file.

    ``shards`` is recorded in the plan, so a later ``run`` cannot silently use a
    different split than the one that was audited.
    """
    stage = stage_for(name)
    if type(shards) is not int or shards < 1:
        raise StageError("shards must be a positive integer")
    store, directory = Store(run_root), stage_dir(run_root, name)
    if (directory / "PLAN.json").exists():
        raise StageError(f"stage {name} is already planned at {directory}; a plan is immutable")
    keys = stage.keys(params)
    # Sorted-key modulo: each shard takes every ``shards``-th key, which is
    # balanced for a uniform tree and, unlike contiguous blocks, does not put one
    # directory's cost on one shard.  Keys are appended to each shard file in
    # ascending order, so a shard's own output is already sorted and the merge
    # only has to interleave the shard streams.
    digests, counts = hashlib.sha256(), [0] * shards
    handles = []
    try:
        for index in range(shards):
            path = key_file(run_root, name, index)
            store.mkdir(path.parent)
            fd = os.open(store.checked(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            handles.append(os.fdopen(fd, "wb", buffering=CHUNK))
        for position, key in enumerate(keys):
            data = (key + "\n").encode("utf-8", "surrogateescape")
            digests.update(data)
            handles[position % shards].write(data)
            counts[position % shards] += 1
        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        for handle in handles:
            handle.close()
    manifest = {
        "schema": SCHEMA,
        "stage": name,
        "params": params,
        "parallelism": {"shards": shards},
        "determinism": {"order": "stable-key", "assignment": "sorted-index-modulo-shards",
                        "keys_sha256": digests.hexdigest(), "keys": len(keys)},
        "shards": [{"index": index, "items": counts[index],
                    "keys_sha256": _file_digest(key_file(run_root, name, index))}
                   for index in range(shards)],
        "planned_at": now(),
    }
    store.write(directory / "PLAN.json", manifest, exclusive=True)
    return manifest


def _file_digest(filename):
    checksum = hashlib.sha256()
    with open(filename, "rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


# -- run one shard -----------------------------------------------------------------------------

def _ordered_map(function, items, workers, window):
    """Map with bounded in-flight work, yielding results in input order.

    ``ThreadPoolExecutor.map`` submits every item at once, which for a million
    small files means a million futures in memory before the first byte is
    written.  This keeps at most ``window`` results pending.
    """
    if workers <= 1:
        for item in items:
            yield function(item)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = []
        iterator = iter(items)
        for item in iterator:
            pending.append(pool.submit(function, item))
            if len(pending) >= window:
                break
        while pending:
            yield pending.pop(0).result()
            item = next(iterator, None)
            if item is not None:
                pending.append(pool.submit(function, item))


def run_shard(name, run_root, index, threads=None):
    """Hash one shard's key file.  A shard that does not finish stays unsettled."""
    stage = stage_for(name)
    store = Store(run_root)
    manifest = read_json(stage_dir(run_root, name) / "PLAN.json")
    if manifest is None:
        raise StageError(f"stage {name} has no plan; run plan first")
    if not 0 <= index < manifest["parallelism"]["shards"]:
        raise StageError(f"shard {index} is outside the planned {manifest['parallelism']['shards']}")
    expected = manifest["shards"][index]
    directory = stage_dir(run_root, name) / "shards" / f"{index:03d}"
    if (directory / "RECEIPT.json").exists():
        raise StageError(f"shard {index} already ran; a shard is not silently re-run")
    source = directory / "KEYS.txt"
    if _file_digest(source) != expected["keys_sha256"]:
        raise StageError(f"shard {index}: key file differs from the plan")
    dispatch = uuid.uuid4().hex
    store.write(directory / "DISPATCH.json",
                {"at": now(), "dispatch_id": dispatch, "shard": index, "threads": threads or 1},
                exclusive=True)
    started, written = time.monotonic(), 0
    threads = threads or available_cpus()

    def record(key):
        item = stage.record(key, manifest["params"])
        if item.get("key") != key:
            raise StageError(f"record key mismatch for {key!r}")
        return canonical(item) + b"\n"

    try:
        checksum = _stream_write(store, directory / "records.jsonl",
                                 _ordered_map(record, read_keys(source), threads,
                                              max(8, threads * 32)))
        written = expected["items"]
        receipt = {"schema": SCHEMA, "stage": name, "shard": index,
                   "shards": manifest["parallelism"]["shards"], "dispatch_id": dispatch,
                   "state": "SUCCEEDED", "items": written, "threads": threads,
                   "records_sha256": checksum, "seconds": round(time.monotonic() - started, 3),
                   "peak_rss_bytes": peak_rss_bytes(), "finished_at": now()}
    except BaseException as error:
        # The shard's outcome is unknown, not failed: partial output on disk is
        # not evidence about the files it never reached.  No retry is automatic.
        store.write(directory / "RECEIPT.json",
                    {"schema": SCHEMA, "stage": name, "shard": index, "dispatch_id": dispatch,
                     "state": "UNKNOWN", "reason": repr(error),
                     "seconds": round(time.monotonic() - started, 3), "finished_at": now()})
        raise
    store.write(directory / "RECEIPT.json", receipt, exclusive=True)
    return receipt


# -- run every shard of a plan -----------------------------------------------------------------

def _shard_task(args):
    """Module-level so a process pool can pickle it.  Returns a small summary."""
    name, run_root, index, threads = args
    started = time.monotonic()
    receipt = run_shard(name, run_root, index, threads)
    return {"shard": index, "state": receipt["state"], "items": receipt["items"],
            "seconds": round(time.monotonic() - started, 3), "threads": receipt["threads"],
            "peak_rss_bytes": receipt["peak_rss_bytes"]}


def run_stage(name, run_root, jobs=None, threads=None):
    """Run every planned shard, one process each, and summarise the whole stage.

    Shards are separate processes rather than threads because the work is CPU
    bound: sha256 releases the GIL, but path resolution and JSON do not, and a
    process pool also means one shard's crash cannot take the others' state with
    it.  ``jobs`` defaults to one process per shard; ``threads`` defaults to an
    even split of the CPUs this process may use, so ``jobs * threads`` stays
    within the quota instead of each shard assuming it owns the machine.
    """
    manifest = read_json(Path(run_root) / "stages" / name / "PLAN.json")
    if manifest is None:
        raise StageError(f"stage {name} has no plan; run plan first")
    indexes = [shard["index"] for shard in manifest["shards"]]
    jobs = max(1, min(jobs or len(indexes), len(indexes)))
    threads = threads or max(1, available_cpus() // jobs)
    work = [(name, run_root, index, threads) for index in indexes]
    started = time.monotonic()
    if jobs <= 1:
        results = [_shard_task(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_shard_task, work))
    elapsed = round(time.monotonic() - started, 3)
    return {"stage": name, "run_root": run_root, "shards": len(results), "jobs": jobs,
            "threads_per_shard": threads, "items": sum(r["items"] for r in results),
            "state": "SUCCEEDED" if all(r["state"] == "SUCCEEDED" for r in results) else "UNKNOWN",
            "wall_seconds": elapsed,
            # Per-shard service time over wall time: >1 means the shards really
            # overlapped, and the ceiling is the slowest shard, not the sum.
            "cpu_parallelism": round(sum(r["seconds"] for r in results) / elapsed, 2) if elapsed else 0,
            "slowest_shard_seconds": max((r["seconds"] for r in results), default=0),
            # Sum, not max: each shard is its own process, so the peak of the
            # stage is what the whole set of processes held at once.
            "peak_rss_bytes": sum(r["peak_rss_bytes"] for r in results),
            "receipts": results}


# -- merge -------------------------------------------------------------------------------------

def _iter_records(filename):
    with open(filename, "rb") as stream:
        for line in stream:
            if line.strip():                yield json.loads(line)


def merge(name, run_root):
    """Concatenate settled shards back into one stable-key document.

    The k-way merge restores global key order without loading the stage into
    memory, so the merged bytes -- and therefore the digest -- depend only on the
    shard contents, never on the order the shards happened to finish.
    """
    store, directory = Store(run_root), stage_dir(run_root, name)
    manifest = read_json(directory / "PLAN.json")
    if manifest is None:
        raise StageError(f"stage {name} has no plan; run plan first")
    sources, receipts = [], []
    for shard in manifest["shards"]:
        index = shard["index"]
        shard_dir = directory / "shards" / f"{index:03d}"
        receipt = read_json(shard_dir / "RECEIPT.json")
        if not receipt or receipt.get("state") != "SUCCEEDED":
            state = (receipt or {}).get("state", "MISSING")
            raise StageError(f"shard {index} is {state}; no partial merge")
        if receipt.get("items") != shard["items"]:
            raise StageError(f"shard {index} has {receipt.get('items')} records, plan says {shard['items']}")
        if _file_digest(shard_dir / "records.jsonl") != receipt["records_sha256"]:
            raise StageError(f"shard {index}: records changed after the receipt was written")
        sources.append(shard_dir / "records.jsonl")
        receipts.append({"index": index, "dispatch_id": receipt["dispatch_id"],
                         "records_sha256": receipt["records_sha256"]})

    merged = _stream_write(store, directory / "merged.jsonl",
                           _merged_records(sources))
    result = {"schema": SCHEMA, "stage": name, "merged_sha256": merged,
              "ordering": "stable-key", "shards": receipts,
              "parallelism": manifest["parallelism"],
              "planned_keys_sha256": manifest["determinism"]["keys_sha256"],
              "merged_at": now()}
    store.write(directory / "MERGE.json", result)
    return result


def _merged_records(sources):
    streams = [_iter_records(source) for source in sources]
    try:
        previous = None
        for record in heapq.merge(*streams, key=lambda item: item["key"]):
            if previous is not None and record["key"] <= previous:
                raise StageError(f"merged keys are not strictly increasing at {record['key']!r}")
            previous = record["key"]
            yield canonical(record) + b"\n"
    finally:
        for stream in streams:
            stream.close()


# -- verify ------------------------------------------------------------------------------------

def verify(name, run_root):
    """Re-derive the stage from its inputs and compare.  No receipt is trusted.

    A shard receipt says what a process believed when it exited.  Verification
    asks the filesystem again: every key re-enumerated, every file re-hashed, and
    the plan digest re-derived from scratch.  Anything that moved since the run
    is a mismatch, which is exactly what an audit is for.
    """
    stage = stage_for(name)
    store, directory = Store(run_root), stage_dir(run_root, name)
    manifest = read_json(directory / "PLAN.json")
    result = read_json(directory / "MERGE.json")
    if manifest is None or result is None:
        raise StageError(f"stage {name} has no plan/merge to verify")
    if _file_digest(directory / "merged.jsonl") != result["merged_sha256"]:
        raise StageError("merged output changed after the merge receipt")
    keys = stage.keys(manifest["params"])
    digests = hashlib.sha256()
    for key in keys:
        digests.update((key + "\n").encode("utf-8", "surrogateescape"))
    drift = None
    if digests.hexdigest() != manifest["determinism"]["keys_sha256"]:
        drift = "the input tree changed since the plan was frozen"
    mismatches, checked = [], 0
    previous = None
    for record in _iter_records(directory / "merged.jsonl"):
        checked += 1
        if previous is not None and record["key"] <= previous:
            mismatches.append({"key": record["key"], "reason": "ordering"})
        previous = record["key"]
        if drift is None and not stage.verify_record(record["key"], record, manifest["params"]):
            mismatches.append({"key": record["key"], "reason": "record differs from the input"})
        if len(mismatches) > 20:
            break
    report = {"schema": SCHEMA, "stage": name, "ok": drift is None and not mismatches and
              checked == manifest["determinism"]["keys"], "checked": checked,
              "planned": manifest["determinism"]["keys"], "keys_sha256": digests.hexdigest(),
              "drift": drift, "mismatches": mismatches, "verified_at": now()}
    store.write(directory / "VERIFY.json", report)
    return report
