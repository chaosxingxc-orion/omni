"""The port's CPU encoder runtime: a pool of pinned worker processes behind the preparation backends.

A CPU encoder inventory (``saea-cpu-encoder-inventory-v1``) pins what the historical in-process inventory
pins (model files by size and SHA-256), plus the runtime (``cpu-float32``), the worker interpreter, the
worker count, the threads per worker and the first core of the worker slices. Package versions are
verified inside each worker's own interpreter, because the harness interpreter need not carry torch.

Concurrency never changes a vector. Every request is its own batch-one forward on one worker with a
fixed thread count, so the pool size only changes wall-clock time. The preparation lifecycle charges a
whole window of contacts (``contact(sources, batch=True)``) before ``decode_batch`` runs it across
the workers.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cpu_encoder_worker import pack_wave, read_frame, write_frame
from .speech_embedding import ENCODER_RUNTIMES

CPU_INVENTORY_SCHEMA = "saea-cpu-encoder-inventory-v1"
_CPU_KEYS = {
    "schema", "model_dir_relpath", "files", "packages", "runtime", "python_executable", "workers",
    "threads_per_worker", "first_core",
}
_SRC_ROOT = Path(__file__).resolve().parents[2]


class CpuEncoderPoolError(RuntimeError):
    """Refuse a CPU encoder inventory, worker or result that does not match its pins."""


def check_cpu_inventory(inventory):
    """Shape of a CPU encoder inventory (file pins are verified by the preparation runtime)."""

    if not isinstance(inventory, dict) or set(inventory) != _CPU_KEYS:
        raise CpuEncoderPoolError("CPU encoder inventory has the wrong keys")
    if inventory["schema"] != CPU_INVENTORY_SCHEMA:
        raise CpuEncoderPoolError("CPU encoder inventory schema is invalid")
    if inventory["runtime"] not in ENCODER_RUNTIMES or not inventory["runtime"].startswith("cpu-"):
        raise CpuEncoderPoolError("CPU encoder inventory must name a CPU runtime")
    for key in ("workers", "threads_per_worker"):
        if type(inventory[key]) is not int or inventory[key] < 1:
            raise CpuEncoderPoolError(f"CPU encoder inventory {key} must be a positive integer")
    if type(inventory["first_core"]) is not int or inventory["first_core"] < 0:
        raise CpuEncoderPoolError("CPU encoder inventory first_core must be a non-negative integer")
    if not isinstance(inventory["python_executable"], str) or not os.path.isabs(inventory["python_executable"]):
        raise CpuEncoderPoolError("CPU encoder inventory python_executable must be an absolute path")
    if not isinstance(inventory["packages"], dict) or not {"torch", "transformers"} <= set(inventory["packages"]):
        raise CpuEncoderPoolError("CPU encoder inventory must pin torch and transformers")
    return inventory


def worker_slices(inventory):
    """Disjoint ``first-last`` core slices, one per worker."""

    width, first = inventory["threads_per_worker"], inventory["first_core"]
    return [f"{first + index * width}-{first + (index + 1) * width - 1}" for index in range(inventory["workers"])]


class CpuEncoderPool:
    """Start the pinned workers once; run each request batch-one on one worker; keep caller order."""

    def __init__(self, inventory, model_dir, *, spawn=subprocess.Popen):
        self.inventory = check_cpu_inventory(inventory)
        slices = worker_slices(inventory)
        allowed = os.sched_getaffinity(0)
        first, width = inventory["first_core"], inventory["threads_per_worker"]
        if not set(range(first, first + width * len(slices))) <= allowed:
            raise CpuEncoderPoolError("CPU encoder worker slices exceed this process's cores")
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [str(_SRC_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]),
            TORCH_DEVICE_BACKEND_AUTOLOAD="0", HF_HUB_OFFLINE="1")
        command = [inventory["python_executable"], "-m", "speech_aware_evidence_acquisition.core.cpu_encoder_worker",
                   str(model_dir), inventory["runtime"]]
        self.workers, self.locks = [], []
        try:
            for spec in slices:
                self.workers.append(spawn(
                    [*command, spec, str(inventory["threads_per_worker"]), ",".join(sorted(inventory["packages"]))],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment))
                self.locks.append(threading.Lock())
            for worker in self.workers:  # all workers load concurrently; collect their reports in order
                report = read_frame(worker.stdout)
                if not report or not report.get("ready"):
                    raise CpuEncoderPoolError("CPU encoder worker did not start")
                if report["packages"] != inventory["packages"]:
                    raise CpuEncoderPoolError("CPU encoder worker package versions differ from the inventory")
                if report["threads"] != inventory["threads_per_worker"]:
                    raise CpuEncoderPoolError("CPU encoder worker thread count differs from the inventory")
        except BaseException:
            self.close()
            raise
        self.executor = ThreadPoolExecutor(max_workers=len(self.workers))

    @property
    def size(self):
        return len(self.workers)

    def _run(self, index, request):
        worker = self.workers[index % len(self.workers)]
        with self.locks[index % len(self.workers)]:
            write_frame(worker.stdin, request)
            reply = read_frame(worker.stdout)
        if reply is None:
            raise CpuEncoderPoolError("CPU encoder worker exited")
        if "error" in reply:
            raise CpuEncoderPoolError(f"CPU encoder worker failed: {reply['error']}")
        vector = reply["vector"]
        if not isinstance(vector, (list, tuple)) or not all(type(value) is float for value in vector):
            raise CpuEncoderPoolError("CPU encoder worker returned a non-float vector")
        return vector

    def map(self, requests):
        futures = [self.executor.submit(self._run, index, request) for index, request in enumerate(requests)]
        return [future.result() for future in futures]

    def close(self):
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for worker in self.workers:
            for stream in (worker.stdin, worker.stdout):
                if stream is not None:
                    stream.close()
        for worker in self.workers:
            try:
                worker.wait(timeout=60)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        self.workers = []


class _CpuBackend:
    """Lazy pool start inside the first charged contact, as the in-process backends load lazily."""

    def __init__(self, inventory, protocol, data_root, event, *, model_dir, pool_cls=CpuEncoderPool):
        self.inventory, self.protocol, self.event = check_cpu_inventory(inventory), protocol, event
        self.model_dir, self.pool_cls, self.pool = model_dir, pool_cls, None

    @property
    def concurrency(self):
        return self.inventory["workers"]

    def _pool(self):
        if self.pool is None:
            self.event("model_load_started", workers=self.inventory["workers"],
                       threads_per_worker=self.inventory["threads_per_worker"], runtime=self.inventory["runtime"])
            self.pool = self.pool_cls(self.inventory, self.model_dir)
            self.event("model_load_finished")
        return self.pool

    def _encode(self, requests):
        pool = self._pool()
        self.event("encoder_forward_started", batch_size=len(requests))
        vectors = pool.map(requests)
        self.event("encoder_forward_finished", batch_size=len(requests))
        return vectors

    def __call__(self, source):
        return self.decode_batch([source])[0]

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None


class CpuSpeechEncoder(_CpuBackend):
    """Speech queries: parent-side bounded audio prefetch, worker-side batch-one forwards."""

    def __init__(self, inventory, protocol, data_root, event, *, model_dir, prefetch, pool_cls=CpuEncoderPool):
        super().__init__(inventory, protocol, data_root, event, model_dir=model_dir, pool_cls=pool_cls)
        self.prefetch = prefetch

    def prepare_sources(self, sources):
        self.prefetch.prepare_sources(sources)

    def decode_batch(self, sources):
        self.event("data_wait_started")
        waves = [self.prefetch.take(source) for source in sources]
        self.event("data_wait_finished")
        bounds = {
            "max_audio_samples": self.protocol["max_audio_samples"],
            "require_complete_audio": True,
            "max_sequence_tokens": self.protocol["max_sequence_tokens"],
        }
        return self._encode([
            {"content": [{"type": "text", "text": self.protocol["query_prompt"]}, {"type": "audio"}],
             "audio": [pack_wave(wave)], "bounds": bounds}
            for wave in waves
        ])

    def close(self):
        try:
            self.prefetch.close()
        finally:
            super().close()


class CpuPassageEncoder(_CpuBackend):
    """Text facet passages, one batch-one forward each."""

    def decode_batch(self, texts):
        bounds = {"max_sequence_tokens": self.protocol["max_sequence_tokens"]}
        return self._encode([
            {"content": [{"type": "text", "text": self.protocol["passage_prompt"] + text}], "bounds": bounds}
            for text in texts
        ])


__all__ = [
    "CPU_INVENTORY_SCHEMA",
    "CpuEncoderPool",
    "CpuEncoderPoolError",
    "CpuPassageEncoder",
    "CpuSpeechEncoder",
    "check_cpu_inventory",
    "worker_slices",
]
