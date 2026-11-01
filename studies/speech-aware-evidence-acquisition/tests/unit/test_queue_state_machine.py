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
