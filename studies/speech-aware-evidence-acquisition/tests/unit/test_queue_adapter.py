import pytest

from speech_aware_evidence_acquisition.reproduction.ojw.queue_adapter import (
    QueueCallbackAdapter,
)
from speech_aware_evidence_acquisition.reproduction.ojw.queue_state import (
    QueueStateError, QueueStateMachine,
)


def test_callbacks_publish_checkpoint_and_restore(tmp_path):
    path = tmp_path / "queue.json"
    adapter = QueueCallbackAdapter(QueueStateMachine(slots=1), path)
    adapter.submit("p", sample_id="s", stage=0, prefix_family="shared", instance_id="i")
    adapter.prepare_callback("p")
    assert adapter.admit_generation(stage=0) == ("p",)
    adapter.generation_callback("p", observed=True, result_digest="digest:p")

    restored = QueueCallbackAdapter.restore(path)
    assert restored.machine.jobs["p"].state == "observed"
    assert restored.machine.jobs["p"].result_digest == "digest:p"


def test_duplicate_or_out_of_order_callbacks_fail_without_new_state(tmp_path):
    path = tmp_path / "queue.json"
    adapter = QueueCallbackAdapter(QueueStateMachine(slots=1), path)
    adapter.submit("p", sample_id="s", stage=0, prefix_family="shared", instance_id="i")
    before = path.read_bytes()
    with pytest.raises(QueueStateError, match="only an admitted generation"):
        adapter.generation_callback("p", observed=False)
    assert path.read_bytes() == before

    adapter.prepare_callback("p")
    adapter.admit_generation(stage=0)
    adapter.generation_callback("p", observed=False)
    before = path.read_bytes()
    with pytest.raises(QueueStateError, match="only an admitted generation"):
        adapter.generation_callback("p", observed=True, result_digest="duplicate")
    assert path.read_bytes() == before


def test_dependency_callback_order_remains_fail_closed(tmp_path):
    path = tmp_path / "queue.json"
    adapter = QueueCallbackAdapter(QueueStateMachine(slots=2), path)
    adapter.submit("p", sample_id="s", stage=0, prefix_family="shared", instance_id="i")
    adapter.submit("s", sample_id="s", stage=1, prefix_family="shared", instance_id="i", depends_on="p")
    with pytest.raises(QueueStateError, match="dependency"):
        adapter.prepare_callback("s")
    assert QueueStateMachine.restore_file(path).jobs["s"].state == "prepare"
