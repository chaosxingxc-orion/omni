"""Optional bridge from :class:`NodePipelineCoordinator` to durable queues.

The bridge is opt-in.  The coordinator keeps its historical barrier path when
``queue_bridge`` is ``None``; an explicitly supplied bridge records the same
prepare/contact lifecycle in ``QueueCallbackAdapter`` and therefore gives a
detached runner a durable, fail-closed checkpoint without replaying requests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
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
    """Bind coordinator members to one durable three-queue adapter."""

    def __init__(self, adapter: QueueCallbackAdapter):
        if not isinstance(adapter, QueueCallbackAdapter):
            raise QueueStateError("adapter must be a QueueCallbackAdapter")
        self.adapter = adapter
        self._active: dict[int, tuple[str, int, str, str]] = {}
        self._last_job: dict[int, str] = {}

    @classmethod
    def create(cls, checkpoint: str | Path, *, slots: int) -> "QueuePipelineBridge":
        return cls(QueueCallbackAdapter(QueueStateMachine(slots=slots), checkpoint))

    def prepare_submitted(self, member: int, key: tuple[int, int], family: str) -> None:
        if member in self._active:
            raise QueueStateError(f"member {member} submitted a second prepare before settle")
        round_index, node_index = key
        # QueueStateMachine validates dependencies by a single monotonically
        # increasing stage integer.  Encode the coordinator's two-dimensional
        # (round, node) key so the first node of a later round still follows
        # the final node of the previous round.
        stage = round_index * 1_000_000 + node_index
        job_id = f"member-{member}:round-{round_index}:stage-{stage}"
        # Every stage after the first observed operation depends on that
        # member's immediately preceding operation.  This covers both P->S
        # within a round and the first stage of a resumed round; omitting the
        # edge would let a stale callback unlock a future request.
        depends_on = self._last_job.get(member)
        instance_id = f"pipeline-member-{member}"
        self.adapter.submit(
            job_id,
            sample_id=str(member),
            stage=stage,
            prefix_family=family,
            instance_id=instance_id,
            depends_on=depends_on,
        )
        self._active[member] = (job_id, stage, family, instance_id)

    def prepare_completed(self, member: int, *, ok: bool) -> None:
        job_id, _stage, _family, _instance = self._active[member]
        self.adapter.prepare_callback(job_id, ok=ok)

    def generation_admitted(self, member: int) -> None:
        job_id, stage, family, instance_id = self._active[member]
        admitted = self.adapter.admit_generation(
            stage=stage, available_slots=1, instance_id=instance_id, prefix_family=family
        )
        if admitted != (job_id,):
            raise QueueStateError(f"queue admitted {admitted!r} instead of {job_id!r}")

    def generation_completed(self, member: int, *, observed: bool, result: Any = None) -> None:
        job_id, _stage, _family, _instance = self._active.pop(member)
        self.adapter.generation_callback(
            job_id, observed=observed, result_digest=_digest(result) if observed else None
        )
        self._last_job[member] = job_id


__all__ = ["QueuePipelineBridge"]
