"""Native pipeline journal using the public durable-file adapter.

The path-based adapter persists checkpoints; the owner retains the lease, native journal and recovery.
Generation entries represent occupied coordinator slots, including tools;
they must not be interpreted as a model-call cost counter.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .queue_adapter import QueueCallbackAdapter
from .queue_state import QueueStateError, QueueStateMachine


def _digest(value: object) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError):
        encoded = repr(value).encode()
    return hashlib.sha256(encoded).hexdigest()


class QueuePipelineBridge:
    """Track one fresh coordinator without changing its metered callbacks.

    Inspect/reconcile existing state through QueueCallbackAdapter. Automatic
    live-coordinator restoration is unsupported; a nonempty machine is refused
    to prevent reusing logical member identities and replaying requests.
    """

    def __init__(self, adapter: QueueCallbackAdapter):
        if not isinstance(adapter, QueueCallbackAdapter):
            raise TypeError("adapter must be a QueueCallbackAdapter")
        if adapter.machine.jobs:
            raise QueueStateError("pipeline bridge requires an empty machine; reconcile restored jobs first")
        self.adapter = adapter
        self._active: dict[int, str] = {}
        self._last_job: dict[int, str] = {}
        self._position: dict[int, tuple[tuple[int, int], str]] = {}
        self._sequence: dict[int, int] = {}
        self._bound = False

    @classmethod
    def create(cls, checkpoint, *, slots: int):
        # Preserve the public path-based adapter API and its atomic publisher.
        return cls(QueueCallbackAdapter(QueueStateMachine(slots=slots), checkpoint))

    def bind(self, *, slots: int) -> None:
        if self._bound or self.adapter.machine.jobs:
            raise QueueStateError("bridge cannot be reused by another coordinator")
        if self.adapter.machine.slots != slots:
            raise QueueStateError("queue capacity differs from coordinator slots")
        self._bound = True

    def _submit(self, member: int, kind: str) -> None:
        key, family = self._position[member]
        sequence = self._sequence.get(member, 0)
        job_id = f"member-{member}:round-{key[0]}:node-{key[1]}:operation-{sequence}:{kind}"
        self.adapter.submit(job_id, sample_id=str(member), stage=sequence,
                            prefix_family=family, instance_id=f"pipeline-member-{member}",
                            depends_on=self._last_job.get(member))
        self._active[member] = job_id
        self._sequence[member] = sequence + 1

    def prepare_submitted(self, member: int, key: tuple[int, int], family: str) -> None:
        if member in self._active:
            raise QueueStateError("previous prepared operation has not settled")
        previous = self._position.get(member)
        if previous is not None and key <= previous[0]:
            raise QueueStateError("pipeline stage must advance")
        self._position[member] = (key, family)
        self._submit(member, "prepared")

    def prepare_completed(self, member: int, *, ok: bool) -> None:
        job_id = self._active[member]
        if self.adapter.machine.jobs[job_id].state == "cancelled":
            self._active.pop(member)
            return
        self.adapter.prepare_callback(job_id, ok=ok)
        if not ok:
            self._active.pop(member)

    def contact_submitted(self, member: int, kind: str) -> None:
        if kind not in {"model", "tool"}:
            raise QueueStateError("unknown contact kind")
        if member not in self._active:
            # Continuations reuse role preparation, with one job per contact.
            self._submit(member, kind)
            self.adapter.prepare_callback(self._active[member], ok=True)
        job = self.adapter.machine.jobs[self._active[member]]
        if job.state != "ready":
            raise QueueStateError("contact is not ready")

    def generation_admitted(self, member: int) -> None:
        job_id = self._active[member]
        job = self.adapter.machine.jobs[job_id]
        selected = self.adapter.admit_generation(stage=job.stage, available_slots=1,
                                                  instance_id=job.instance_id,
                                                  prefix_family=job.prefix_family)
        if selected != (job_id,):
            raise QueueStateError("queue failed to admit the dispatched contact")

    def generation_completed(self, member: int, *, observed: bool, result: Any = None) -> None:
        job_id = self._active[member]
        self.adapter.generation_callback(job_id, observed=observed,
                                         result_digest=_digest(result) if observed else None)
        self._active.pop(member)
        self._last_job[member] = job_id

    def abort(self) -> None:
        for job_id in tuple(self._active.values()):
            if self.adapter.machine.jobs[job_id].state in {"prepare", "ready", "generation"}:
                self.adapter.cancel(job_id)

    def finish(self, member: int) -> None:
        job_id = self._active.get(member)
        if job_id is None:
            return
        state = self.adapter.machine.jobs[job_id].state
        if state in {"generation", "draining"}:
            raise QueueStateError("member cannot finish with an unsettled contact")
        if state in {"prepare", "ready"}:
            self.adapter.cancel(job_id)
        self._active.pop(member)


__all__ = ["QueuePipelineBridge"]
