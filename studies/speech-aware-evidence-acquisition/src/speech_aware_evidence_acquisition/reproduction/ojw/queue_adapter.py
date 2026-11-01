"""Fail-closed callback seam for the three-queue state machine.

The live runner is deliberately not imported here.  A runner adapter owns one
``QueueStateMachine`` and publishes a durable checkpoint after every accepted
callback.  Unknown, duplicate, or out-of-order callbacks raise before a new
checkpoint is written; callers must reconcile the original request instead of
retrying it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .queue_state import QueueStateError, QueueStateMachine


class QueueCallbackAdapter:
    """Small integration boundary for a future detached runner."""

    def __init__(self, machine: QueueStateMachine, checkpoint: str | Path):
        if not isinstance(machine, QueueStateMachine):
            raise QueueStateError("machine must be a QueueStateMachine")
        self.machine = machine
        self.checkpoint = Path(checkpoint)
        self.machine.checkpoint_to(self.checkpoint)

    @classmethod
    def restore(cls, checkpoint: str | Path) -> "QueueCallbackAdapter":
        """Restore only from an existing complete checkpoint."""
        path = Path(checkpoint)
        return cls(QueueStateMachine.restore_file(path), path)

    def submit(self, job_id: str, *, sample_id: str, stage: int,
               prefix_family: str, instance_id: str,
               depends_on: str | None = None) -> None:
        self.machine.submit(
            job_id,
            sample_id=sample_id,
            stage=stage,
            prefix_family=prefix_family,
            instance_id=instance_id,
            depends_on=depends_on,
        )
        self._persist()

    def prepare_callback(self, job_id: str, *, ok: bool = True) -> None:
        """Accept one preparation callback and publish the resulting state."""
        self.machine.complete_prepare(job_id, ok=ok)
        self._persist()

    def admit_generation(self, *, stage: int, available_slots: int | None = None,
                         instance_id: str | None = None,
                         prefix_family: str | None = None) -> tuple[str, ...]:
        admitted = self.machine.admit_generation(
            stage=stage,
            available_slots=available_slots,
            instance_id=instance_id,
            prefix_family=prefix_family,
        )
        if admitted:
            self._persist()
        return admitted

    def generation_callback(self, job_id: str, *, observed: bool,
                            result_digest: str | None = None) -> None:
        """Settle the original request; duplicate callbacks are rejected."""
        self.machine.settle(
            job_id,
            observed=observed,
            result_digest=result_digest,
        )
        self._persist()

    def cancel(self, job_id: str) -> None:
        self.machine.cancel(job_id)
        self._persist()

    def _persist(self) -> None:
        self.machine.checkpoint_to(self.checkpoint)


__all__ = ["QueueCallbackAdapter"]
