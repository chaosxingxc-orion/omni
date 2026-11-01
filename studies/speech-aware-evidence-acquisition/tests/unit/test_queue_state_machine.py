import pytest

from speech_aware_evidence_acquisition.reproduction.ojw.queue_state import (
    QueueStateError, QueueStateMachine,
)


def test_prepare_ready_generation_preserves_stage_and_affinity():
    q = QueueStateMachine(slots=2)
    q.submit("p0", sample_id="s0", stage=0, prefix_family="p", instance_id="i0")
    q.submit("s0", sample_id="s0", stage=1, prefix_family="s", instance_id="i0", depends_on="p0")
    q.complete_prepare("p0")
    assert q.admit_generation(stage=1) == ()
    assert q.admit_generation(stage=0, instance_id="i1") == ()
    assert q.admit_generation(stage=0, instance_id="i0", prefix_family="p") == ("p0",)
    with pytest.raises(QueueStateError, match="dependency"):
        q.complete_prepare("s0")
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
    """Exercise the callback seam without touching the live runner.

    The fake callbacks model the only contract a future adapter needs: prepare
    completion is explicit, generation admission is one-shot, and cancellation
    waits for the original request's terminal callback instead of replaying it.
    """
    q = QueueStateMachine(slots=2)
    q.submit("p0", sample_id="s0", stage=0, prefix_family="shared", instance_id="i0")
    q.submit("p1", sample_id="s1", stage=0, prefix_family="shared", instance_id="i1")
    q.submit("s0", sample_id="s0", stage=1, prefix_family="shared", instance_id="i0", depends_on="p0")

    callbacks: list[tuple[str, str]] = []

    def on_prepare(job_id: str, ok: bool = True) -> None:
        q.complete_prepare(job_id, ok=ok)
        callbacks.append(("prepare", job_id))

    def on_generation_terminal(job_id: str, *, observed: bool) -> None:
        q.settle(job_id, observed=observed, result_digest=(f"digest:{job_id}" if observed else None))
        callbacks.append(("terminal", job_id))

    on_prepare("p0")
    on_prepare("p1")
    assert q.admit_generation(stage=0) == ("p0", "p1")
    q.cancel("p1")
    # A cancelled in-flight request is draining; it cannot be admitted again.
    assert q.admit_generation(stage=0) == ()

    # The dependent S request becomes eligible only after its own prepare
    # callback; a stale P result cannot satisfy this dependency implicitly.
    assert q.admit_generation(stage=1) == ()
    with pytest.raises(QueueStateError, match="dependency"):
        on_prepare("s0")
    on_generation_terminal("p0", observed=True)
    on_generation_terminal("p1", observed=False)
    # A real adapter calls this only after the predecessor's observed result.
    on_prepare("s0")
    assert q.admit_generation(stage=1, instance_id="i0", prefix_family="shared") == ("s0",)
    on_generation_terminal("s0", observed=True)

    assert callbacks == [
        ("prepare", "p0"),
        ("prepare", "p1"),
        ("terminal", "p0"),
        ("terminal", "p1"),
        ("prepare", "s0"),
        ("terminal", "s0"),
    ]
    assert {job_id: job.state for job_id, job in q.jobs.items()} == {
        "p0": "observed", "p1": "failed", "s0": "observed"
    }


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
    assert restored.jobs["p"].state == "failed"
    with pytest.raises(QueueStateError, match="only an admitted generation"):
        restored.settle("p", observed=True, result_digest="duplicate")


def test_checkpoint_file_replacement_is_complete_and_restorable(tmp_path):
    q = QueueStateMachine(slots=1)
    q.submit("p", sample_id="s", stage=0, prefix_family="shared", instance_id="i")
    path = tmp_path / "run" / "queue.json"
    assert q.checkpoint_to(path) == path
    assert QueueStateMachine.restore_file(path).prepare_queue == ["p"]

    q.complete_prepare("p")
    q.admit_generation(stage=0)
    q.settle("p", observed=True, result_digest="digest:p")
    q.checkpoint_to(path)
    restored = QueueStateMachine.restore_file(path)
    assert restored.jobs["p"].state == "observed"
    assert restored.jobs["p"].result_digest == "digest:p"
    assert not list(path.parent.glob(".queue.json.*.tmp"))


def test_checkpoint_file_keeps_previous_state_when_serialization_fails(tmp_path, monkeypatch):
    q = QueueStateMachine(slots=1)
    q.submit("p", sample_id="s", stage=0, prefix_family="shared", instance_id="i")
    path = tmp_path / "queue.json"
    q.checkpoint_to(path)
    original = path.read_bytes()
    monkeypatch.setattr(q, "checkpoint", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        q.checkpoint_to(path)
    assert path.read_bytes() == original
