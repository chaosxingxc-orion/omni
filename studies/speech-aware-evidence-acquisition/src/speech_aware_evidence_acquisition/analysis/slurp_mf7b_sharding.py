"""Process-level sharding of an MF-7b text pass across pinned CPU worker processes.

The historical MF-7b key pass (``scripts/slurp_mf7b_nemotron_embedding_pass.py``) encodes the 11,514 TRAIN
utterance texts in consecutive batches of ``--text-batch`` (16 on 2026-09-06) inside one process. This
module splits that SAME batch sequence across N worker processes without changing any batch.

* The items are cut into the identical consecutive chunks the single-process loop forms. Each worker
  receives a contiguous run of whole chunks and forwards each chunk exactly as the single-process loop
  would, with the same members, order, padding and prompt. Batch composition is therefore independent of
  the worker count.
* Each worker pins itself to its own core slice and sets its thread count BEFORE importing torch. It
  loads the stack once and forwards only its own chunks.
* The merge restores chunk order and refuses a missing or duplicated chunk. Given identical vectors,
  the merged ``(ids, vectors)`` (and therefore the JSONL bytes ``write_vectors_jsonl`` emits) equals the
  single-process result.

The encoder entry points are named by an importable module path (``backend``), so a spawned worker
imports them itself. Nothing callable is pickled. Tests point ``backend`` at a fake module.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The real encoder entry points (``load_stack``, ``render``, ``encode``).
DEFAULT_BACKEND = "speech_aware_evidence_acquisition.core.speech_embedding"

#: Identity of the partition rule recorded in a sharded set's receipt block.
PARTITION_RULE = "saea-mf7b-contiguous-whole-batch-shards-v1"


class ShardingError(RuntimeError):
    """A malformed shard plan, a failed worker or an incomplete merge."""


def chunk_items(items: Sequence[tuple[str, str]], batch: int) -> list[list[tuple[str, str]]]:
    """The consecutive chunks the single-process loop forms: ``items[o:o+batch]``."""

    if type(batch) is not int or batch <= 0:
        raise ShardingError("text batch must be a positive int")
    return [list(items[offset : offset + batch]) for offset in range(0, len(items), batch)]


def partition_chunks(count: int, workers: int) -> list[range]:
    """Contiguous, balanced chunk-index ranges (sizes differ by at most one; empty shards refused)."""

    if type(workers) is not int or workers <= 0:
        raise ShardingError("workers must be a positive int")
    if workers > count:
        raise ShardingError(f"{workers} workers for {count} chunks would leave a worker idle")
    base, extra = divmod(count, workers)
    ranges, start = [], 0
    for worker in range(workers):
        stop = start + base + (1 if worker < extra else 0)
        ranges.append(range(start, stop))
        start = stop
    return ranges


def parse_cores(text: str) -> tuple[int, int]:
    """``"FIRST-LAST"`` (inclusive) -> ``(first, last)``."""

    try:
        first, last = (int(part) for part in str(text).split("-"))
    except ValueError as error:
        raise ShardingError(f"cores must be FIRST-LAST, got {text!r}") from error
    if first < 0 or last < first:
        raise ShardingError(f"cores must be FIRST-LAST with FIRST <= LAST, got {text!r}")
    return first, last


def core_slices(first: int, last: int, workers: int) -> list[tuple[int, int]]:
    """Disjoint inclusive core slices covering ``first..last``, one per worker."""

    total = last - first + 1
    if workers > total:
        raise ShardingError(f"{workers} workers need at least {workers} cores, got {total}")
    return [(first + part.start, first + part.stop - 1) for part in partition_chunks(total, workers)]


@dataclass(frozen=True)
class ShardSpec:
    worker: int
    cores: tuple[int, int]
    model_dir: str
    runtime: str
    device: str
    prompt: str
    chunks: tuple[tuple[int, tuple[tuple[str, str], ...]], ...]
    backend: str = DEFAULT_BACKEND
    #: ``None`` loads the stack as shipped; ``"causal"`` forces the geometry and verifies it on the loaded weights.
    attention_geometry: str | None = None


def encode_shard(spec: ShardSpec) -> tuple[list[tuple[int, list[str], np.ndarray]], str | None]:
    """Forward one shard's chunks in order; ``([(chunk_index, ids, float64 matrix)], first rendered)``.

    Each chunk is forwarded exactly as the single-process loop forwards it.
    """

    backend = importlib.import_module(spec.backend)
    processor, model, template = backend.load_stack(
        Path(spec.model_dir), runtime=spec.runtime, attention_geometry=spec.attention_geometry)
    if spec.attention_geometry is not None:
        measured = backend.measure_attention_geometry(processor, model, device=spec.device)
        if measured != spec.attention_geometry:
            raise ShardingError(f"worker {spec.worker} measured attention geometry {measured!r}, "
                                f"not the forced {spec.attention_geometry!r}")
    results, first_rendered = [], None
    for index, chunk in spec.chunks:
        rendered = [
            backend.render(processor, template, [{"type": "text", "text": spec.prompt + text}])
            for _identifier, text in chunk
        ]
        if first_rendered is None:
            first_rendered = rendered[0]
        vectors = backend.encode(processor, model, rendered, None, device=spec.device)
        if len(vectors) != len(chunk):
            raise ShardingError(f"chunk {index}: encoder returned {len(vectors)} vectors for {len(chunk)} texts")
        # float64 carries every float32-derived Python float exactly, so the parent's
        # float() round trip reproduces the single-process tuples bit for bit.
        results.append((index, [identifier for identifier, _text in chunk], np.asarray(vectors, dtype=np.float64)))
    return results, first_rendered


def pin_worker(cores: tuple[int, int], device: str) -> int:
    """Pin this process to ``cores`` and size its thread pools; call BEFORE torch is imported."""

    first, last = cores
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, range(first, last + 1))
    threads = last - first + 1
    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    if device == "cpu":
        # The Ascend image's torch auto-loads torch_npu, which needs the driver; a CPU worker has none.
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        sys.modules.setdefault("torch_npu", None)
    return threads


def _worker_main(spec: ShardSpec, queue) -> None:
    try:
        threads = pin_worker(spec.cores, spec.device)
        if spec.backend == DEFAULT_BACKEND:
            import torch

            torch.set_num_threads(threads)
        results, first_rendered = encode_shard(spec)
        queue.put((spec.worker, "ok", results, first_rendered))
    except BaseException as error:  # reported to the parent, which aborts the whole pass
        queue.put((spec.worker, "error", f"{type(error).__name__}: {error}", None))


def spawn_launcher(specs: Sequence[ShardSpec]) -> list[tuple[int, str, object, str | None]]:
    """Run one spawned process per shard; returns one ``(worker, status, payload, rendered)`` each."""

    import queue as queue_module

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = {
        spec.worker: context.Process(target=_worker_main, args=(spec, queue), daemon=True) for spec in specs
    }
    for process in processes.values():
        process.start()
    outcomes: list[tuple[int, str, object, str | None]] = []
    reported: set[int] = set()
    try:
        while len(reported) < len(specs):
            try:
                outcome = queue.get(timeout=5)
            except queue_module.Empty:
                # A worker killed outright (OOM, signal) never reports; do not wait for it forever.
                dead = [w for w, p in processes.items() if w not in reported and p.exitcode not in (None, 0)]
                if dead:
                    outcomes.extend((w, "error", f"exited with code {processes[w].exitcode}", None) for w in dead)
                    break
                continue
            outcomes.append(outcome)
            reported.add(outcome[0])
            if outcome[1] != "ok":
                break
    finally:
        failed = any(status != "ok" for _w, status, _p, _r in outcomes) or len(reported) < len(specs)
        for process in processes.values():
            if failed and process.is_alive():
                process.terminate()
            process.join()
    return outcomes


def merge_shards(
    outcomes: Sequence[tuple[int, str, object, str | None]], chunk_count: int
) -> tuple[list[str], list[tuple[float, ...]]]:
    """Chunk-ordered ``(ids, vectors)``; refuses a failed worker, a missing chunk or a duplicate."""

    failures = [f"worker {worker}: {payload}" for worker, status, payload, _ in outcomes if status != "ok"]
    if failures:
        raise ShardingError("sharded text pass failed: " + "; ".join(failures))
    by_index: dict[int, tuple[list[str], np.ndarray]] = {}
    for _worker, _status, results, _rendered in outcomes:
        for index, ids, matrix in results:
            if index in by_index:
                raise ShardingError(f"chunk {index} was encoded twice")
            by_index[index] = (ids, matrix)
    if sorted(by_index) != list(range(chunk_count)):
        raise ShardingError(f"merged {len(by_index)} of {chunk_count} chunks")
    ids: list[str] = []
    vectors: list[tuple[float, ...]] = []
    for index in range(chunk_count):
        chunk_ids, matrix = by_index[index]
        ids.extend(chunk_ids)
        vectors.extend(tuple(float(v) for v in row) for row in matrix)
    return ids, vectors


def sharded_text_pass(
    items: Sequence[tuple[str, str]],
    prompt: str,
    *,
    model_dir: Path,
    runtime: str,
    device: str,
    batch: int,
    workers: int,
    cores: tuple[int, int],
    attention_geometry: str | None = None,
    backend: str = DEFAULT_BACKEND,
    launcher: Callable[[Sequence[ShardSpec]], list] = spawn_launcher,
) -> tuple[list[str], list[tuple[float, ...]], dict[str, object], str | None]:
    """Encode ``items`` across ``workers`` pinned processes; ``(ids, vectors, sharding block, rendered[0])``.

    ``forwards`` in the block equals the single-process loop's forward count (one per chunk).
    """

    chunks = chunk_items(items, batch)
    slices = core_slices(*cores, workers)
    specs = [
        ShardSpec(
            worker=worker,
            cores=slices[worker],
            model_dir=str(model_dir),
            runtime=runtime,
            device=device,
            prompt=prompt,
            chunks=tuple((index, tuple(chunks[index])) for index in part),
            backend=backend,
            attention_geometry=attention_geometry,
        )
        for worker, part in enumerate(partition_chunks(len(chunks), workers))
    ]
    outcomes = launcher(specs)
    ids, vectors = merge_shards(outcomes, len(chunks))
    rendered = next((r for worker, _s, _p, r in sorted(outcomes, key=lambda o: o[0]) if worker == 0), None)
    block = {
        "rule": PARTITION_RULE,
        "workers": workers,
        "cores": f"{cores[0]}-{cores[1]}",
        "core_slices": [f"{first}-{last}" for first, last in slices],
        "chunks": len(chunks),
        "forwards": len(chunks),
    }
    return ids, vectors, block, rendered


__all__ = [
    "DEFAULT_BACKEND",
    "PARTITION_RULE",
    "ShardSpec",
    "ShardingError",
    "chunk_items",
    "core_slices",
    "encode_shard",
    "merge_shards",
    "parse_cores",
    "partition_chunks",
    "pin_worker",
    "sharded_text_pass",
    "spawn_launcher",
]
