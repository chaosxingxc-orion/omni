import pytest

from speech_aware_evidence_acquisition.reproduction.ojw.queue_state import (
    QueueStateError, QueueStateMachine,
)


def test_prepare_ready_generation_preserves_stage_and_affinity():
    q = QueueStateMachine(slots=2)
    q.submit("p0", sample_id="s0", stage=0, prefix_family="p", instance_id="i0")
    q.submit("s0", sample_id="s0", stage=1, prefix_family="s", instance_id="i0")
    q.complete_prepare("p0")
    assert q.admit_generation(stage=1) == ()
    assert q.admit_generation(stage=0, instance_id="i1") == ()
    assert q.admit_generation(stage=0, instance_id="i0", prefix_family="p") == ("p0",)
    q.settle("p0", observed=True, result_digest="r0")
    q.complete_prepare("s0")
    assert q.admit_generation(stage=1, instance_id="i0", prefix_family="s") == ("s0",)


def test_cancel_inflight_drains_without_replay():
    q = QueueStateMachine(slots=1)
    q.submit("p", sample_id="s", stage=0, prefix_family="p", instance_id="i")
    q.complete_prepare("p")
    assert q.admit_generation(stage=0) == ("p",)
    q.cancel("p")
    assert q.admit_generation(stage=0) == ()
    q.settle("p", observed=False)
    assert q.jobs["p"].state == "failed"


def test_checkpoint_restore_rejects_unknown_or_mismatched_queue_state():
    q = QueueStateMachine(slots=1)
    q.submit("p", sample_id="s", stage=0, prefix_family="p", instance_id="i")
    q.complete_prepare("p")
    restored = QueueStateMachine.restore(q.checkpoint())
    assert restored.ready_queue == ["p"]
    bad = q.checkpoint().replace('"state":"ready"', '"state":"observed"')
    with pytest.raises(QueueStateError, match="ready queue state"):
        QueueStateMachine.restore(bad)


def test_fake_callback_runner_preserves_fifo_dependencies_and_draining():
    """Exercise the callback seam without touching the live runner."""
    q = QueueStateMachine(slots=2)
    q.submit("p0", sample_id="s0", stage=0, prefix_family="shared", instance_id="i0")
    q.submit("p1", sample_id="s1", stage=0, prefix_family="shared", instance_id="i1")
    q.submit("s0", sample_id="s0", stage=1, prefix_family="shared", instance_id="i0")
    callbacks: list[tuple[str, str]] = []

    def on_prepare(job_id: str) -> None:
        q.complete_prepare(job_id)
        callbacks.append(("prepare", job_id))

    def on_terminal(job_id: str, *, observed: bool) -> None:
        q.settle(job_id, observed=observed, result_digest=(f"digest:{job_id}" if observed else None))
        callbacks.append(("terminal", job_id))

    on_prepare("p0")
    on_prepare("p1")
    assert q.admit_generation(stage=0) == ("p0", "p1")
    q.cancel("p0")
    assert q.admit_generation(stage=0) == ()
    on_terminal("p0", observed=False)
    on_terminal("p1", observed=True)
    assert q.admit_generation(stage=1) == ()
    on_prepare("s0")
    assert q.admit_generation(stage=1, instance_id="i0", prefix_family="shared") == ("s0",)
    on_terminal("s0", observed=True)
    assert callbacks == [
        ("prepare", "p0"), ("prepare", "p1"),
        ("terminal", "p0"), ("terminal", "p1"),
        ("prepare", "s0"), ("terminal", "s0"),
    ]


def test_fake_callback_checkpoint_restore_keeps_inflight_callback_set():
    q = QueueStateMachine(slots=1)
    q.submit("p", sample_id="s", stage=0, prefix_family="p", instance_id="i")
    q.complete_prepare("p")
    assert q.admit_generation(stage=0) == ("p",)
    q.cancel("p")
    restored = QueueStateMachine.restore(q.checkpoint())
    assert restored.generation_queue == ["p"]
    assert restored.inflight == {"p"}
    restored.settle("p", observed=False)
    with pytest.raises(QueueStateError, match="only an admitted generation"):
        restored.settle("p", observed=True, result_digest="duplicate")
