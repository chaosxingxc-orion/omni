"""Pure CPU state machine for prepare -> ready -> generation scheduling.

This module is intentionally independent of the live runner.  It defines the
durable transitions that a future runner integration must preserve, including
P-to-S dependencies, FIFO ready admission, slot/prefix affinity, cancellation
draining, and restart-safe checkpoints.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


class QueueStateError(ValueError):
    pass


STATES = frozenset({"prepare", "ready", "generation", "draining", "observed", "failed", "cancelled"})


@dataclass
class Job:
    job_id: str
    sample_id: str
    stage: int
    prefix_family: str
    instance_id: str
    depends_on: str | None = None
    state: str = "prepare"
    result_digest: str | None = None

    def as_json(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "sample_id": self.sample_id,
            "stage": self.stage,
            "prefix_family": self.prefix_family,
            "instance_id": self.instance_id,
            "depends_on": self.depends_on,
            "state": self.state,
            "result_digest": self.result_digest,
        }


class QueueStateMachine:
    """A restartable, deterministic scheduler state model.

    ``complete_prepare`` is the only transition into ``ready``.  Generation
    admission consumes ready entries once, while an in-flight cancellation
    becomes ``draining`` and cannot be retried or silently replayed.
    """

    schema = "saea-three-queue-state-v1"

    def __init__(self, *, slots: int, members: Iterable[str] = ()):
        if type(slots) is not int or slots < 1:
            raise QueueStateError("slots must be a positive integer")
        self.slots = slots
        self.jobs: dict[str, Job] = {}
        self.prepare_queue: list[str] = []
        self.ready_queue: list[str] = []
        self.generation_queue: list[str] = []
        self.inflight: set[str] = set()
        for member in members:
            if not isinstance(member, str) or not member:
                raise QueueStateError("member identifiers must be nonempty strings")

    def submit(self, job_id: str, *, sample_id: str, stage: int,
               prefix_family: str, instance_id: str,
               depends_on: str | None = None) -> None:
        if not all(isinstance(x, str) and x for x in (job_id, sample_id, prefix_family, instance_id)):
            raise QueueStateError("job identity fields must be nonempty strings")
        if type(stage) is not int or stage < 0:
            raise QueueStateError("stage must be a nonnegative integer")
        if depends_on is not None:
            if not isinstance(depends_on, str) or not depends_on:
                raise QueueStateError("depends_on must be a nonempty job id")
            if depends_on not in self.jobs:
                raise QueueStateError("dependency must be submitted first")
            predecessor = self.jobs[depends_on]
            if predecessor.stage >= stage:
                raise QueueStateError("dependency must belong to an earlier stage")
        if job_id in self.jobs:
            raise QueueStateError("duplicate job submission")
        self.jobs[job_id] = Job(
            job_id, sample_id, stage, prefix_family, instance_id, depends_on=depends_on
        )
        self.prepare_queue.append(job_id)

    def complete_prepare(self, job_id: str, *, ok: bool = True) -> None:
        job = self._job(job_id)
        if job.state != "prepare":
            raise QueueStateError("only prepared jobs can complete preparation")
        if job.depends_on is not None:
            predecessor = self._job(job.depends_on)
            if predecessor.state != "observed":
                raise QueueStateError("job dependency has not reached observed state")
        self._remove(self.prepare_queue, job_id)
        if ok:
            job.state = "ready"
            self.ready_queue.append(job_id)
        else:
            job.state = "failed"

    def admit_generation(self, *, stage: int, available_slots: int | None = None,
                         instance_id: str | None = None,
                         prefix_family: str | None = None) -> tuple[str, ...]:
        if type(stage) is not int or stage < 0:
            raise QueueStateError("stage must be a nonnegative integer")
        capacity = self.slots if available_slots is None else available_slots
        if type(capacity) is not int or not 0 <= capacity <= self.slots:
            raise QueueStateError("available_slots exceeds declared capacity")
        # Caller capacity is an additional bound, not permission to forget
        # requests admitted by an earlier call (including draining requests).
        capacity = min(capacity, self.slots - len(self.inflight))
        selected: list[str] = []
        for job_id in tuple(self.ready_queue):
            if len(selected) >= capacity:
                break
            job = self.jobs[job_id]
            if job.state != "ready" or job.stage != stage:
                continue
            if instance_id is not None and job.instance_id != instance_id:
                continue
            if prefix_family is not None and job.prefix_family != prefix_family:
                continue
            selected.append(job_id)
        for job_id in selected:
            self._remove(self.ready_queue, job_id)
            self.jobs[job_id].state = "generation"
            self.generation_queue.append(job_id)
            self.inflight.add(job_id)
        return tuple(selected)

    def settle(self, job_id: str, *, observed: bool, result_digest: str | None = None) -> None:
        job = self._job(job_id)
        if job.state not in {"generation", "draining"} or job_id not in self.inflight:
            raise QueueStateError("only an admitted generation may settle")
        if observed:
            if not isinstance(result_digest, str) or not result_digest:
                raise QueueStateError("observed generation requires a result digest")
        self._remove(self.generation_queue, job_id)
        self.inflight.remove(job_id)
        if observed:
            job.state = "observed"
            job.result_digest = result_digest
        else:
            job.state = "failed"

    def cancel(self, job_id: str) -> None:
        job = self._job(job_id)
        if job.state == "prepare":
            self._remove(self.prepare_queue, job_id)
            job.state = "cancelled"
        elif job.state == "ready":
            self._remove(self.ready_queue, job_id)
            job.state = "cancelled"
        elif job.state == "generation":
            job.state = "draining"
        else:
            raise QueueStateError("job is not cancellable")

    def checkpoint(self) -> str:
        payload = {
            "schema": self.schema,
            "slots": self.slots,
            "jobs": {key: value.as_json() for key, value in sorted(self.jobs.items())},
            "prepare_queue": list(self.prepare_queue),
            "ready_queue": list(self.ready_queue),
            "generation_queue": list(self.generation_queue),
            "inflight": sorted(self.inflight),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def checkpoint_to(self, path: str | Path) -> Path:
        """Durably replace a queue checkpoint file.

        The live adapter may be interrupted between callbacks, so publication
        is write-temp, fsync-file, atomic-replace, fsync-directory.  Readers
        therefore see either the previous complete checkpoint or the new
        complete checkpoint; they never observe a partially-written JSON
        document.  The operation intentionally replaces an existing path,
        because a queue advances the same run checkpoint over time.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(self.checkpoint().encode("utf-8"))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            # os.replace consumes the temporary name on success.  On failure,
            # remove only our own temporary file and preserve the old state.
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return target

    @classmethod
    def restore_file(cls, path: str | Path) -> "QueueStateMachine":
        """Restore one complete checkpoint file, failing closed on read errors."""
        checkpoint = Path(path)
        try:
            payload = checkpoint.read_bytes()
        except OSError as exc:
            raise QueueStateError(f"checkpoint file is unreadable: {checkpoint}") from exc
        return cls.restore(payload)

    @classmethod
    def restore(cls, payload: str | bytes | Mapping[str, object]) -> "QueueStateMachine":
        value = json.loads(payload) if isinstance(payload, (str, bytes)) else dict(payload)
        if value.get("schema") != cls.schema:
            raise QueueStateError("checkpoint schema mismatch")
        machine = cls(slots=value.get("slots"))
        jobs = value.get("jobs")
        if not isinstance(jobs, Mapping):
            raise QueueStateError("checkpoint jobs must be an object")
        for job_id, row in jobs.items():
            if not isinstance(row, Mapping) or row.get("job_id") != job_id:
                raise QueueStateError("checkpoint job identity mismatch")
            job = Job(
                job_id,
                row["sample_id"],
                row["stage"],
                row["prefix_family"],
                row["instance_id"],
                row.get("depends_on"),
                row["state"],
                row.get("result_digest"),
            )
            if job.state not in STATES:
                raise QueueStateError("checkpoint contains unknown job state")
            machine.jobs[job_id] = job
        for name in ("prepare_queue", "ready_queue", "generation_queue"):
            queue = value.get(name)
            if not isinstance(queue, list) or len(queue) != len(set(queue)):
                raise QueueStateError("checkpoint queue is invalid")
            setattr(machine, name, list(queue))
        inflight = value.get("inflight")
        if not isinstance(inflight, list) or len(inflight) != len(set(inflight)):
            raise QueueStateError("checkpoint inflight set is invalid")
        machine.inflight = set(inflight)
        machine._validate_checkpoint()
        return machine

    def _validate_checkpoint(self) -> None:
        if len(self.inflight) > self.slots:
            raise QueueStateError("checkpoint exceeds generation capacity")
        queued = set(self.prepare_queue) | set(self.ready_queue) | set(self.generation_queue)
        if not queued <= self.jobs.keys() or not self.inflight <= self.jobs.keys():
            raise QueueStateError("checkpoint references unknown job")
        for job in self.jobs.values():
            if job.depends_on is not None:
                if job.depends_on not in self.jobs:
                    raise QueueStateError("checkpoint references unknown dependency")
                if self.jobs[job.depends_on].stage >= job.stage:
                    raise QueueStateError("checkpoint dependency stage mismatch")
        for job_id in self.prepare_queue:
            if self.jobs[job_id].state != "prepare": raise QueueStateError("prepare queue state mismatch")
        for job_id in self.ready_queue:
            if self.jobs[job_id].state != "ready": raise QueueStateError("ready queue state mismatch")
        if set(self.generation_queue) != self.inflight:
            raise QueueStateError("generation queue and inflight set differ")
        for job_id in self.generation_queue:
            if self.jobs[job_id].state not in {"generation", "draining"}:
                raise QueueStateError("generation queue state mismatch")

    def _job(self, job_id: str) -> Job:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise QueueStateError("unknown job") from exc

    @staticmethod
    def _remove(queue: list[str], job_id: str) -> None:
        try:
            queue.remove(job_id)
        except ValueError as exc:
            raise QueueStateError("job is missing from its queue") from exc


__all__ = ["Job", "QueueStateError", "QueueStateMachine"]
