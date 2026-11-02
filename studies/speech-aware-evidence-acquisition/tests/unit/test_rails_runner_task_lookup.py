"""Shared contact lookup keeps controller identity without scanning the roster."""
import json

import pytest

from speech_aware_evidence_acquisition.core.tracing import TraceBuffer
from speech_aware_evidence_acquisition.rails import runner
from speech_aware_evidence_acquisition.rails.core_client import PreparedCoreRequest
from speech_aware_evidence_acquisition.reproduction.ojw import openjiuwen_available
from test_rails_runner import (
    _FakeAdapter, _FakeSample, _ImmediateStopController, _RecordingAgentClient,
    _knowledge_snapshot, _plan, _SHA_A, _SHA_C,
)
from speech_aware_evidence_acquisition.rails import AudioRef


pytestmark = pytest.mark.skipif(
    not openjiuwen_available(), reason="OpenJiuwen unavailable",
)


class _NamedController(_ImmediateStopController):
    def __init__(self, phase):
        self.phase = phase
        self.named_contacts = []

    def phase_name_for_contact(self, contact_id):
        self.named_contacts.append(contact_id)
        return self.phase


class _PreparedClient(_RecordingAgentClient):
    def prepare_request(self, messages, **arguments):
        return PreparedCoreRequest(json.dumps({
            "sample_id": arguments["speech_feature_sample_id"],
            "decoding_params": arguments["decoding_params"],
        }).encode(), object())

    def complete_prepared(self, request, *, id_slot=None):
        payload = request.as_payload()
        return self.complete((), speech_feature_sample_id=payload["sample_id"],
            decoding_params={**payload["decoding_params"], "id_slot": id_slot})


class _CountedTask:
    def __init__(self, prepared):
        self.prepared = prepared
        self.index_reads = 0

    @property
    def index(self):
        self.index_reads += 1
        return self.prepared.index

    def __getattr__(self, name):
        return getattr(self.prepared, name)


def _run(indices, sample_ids, *, fit=False, counted=False):
    plan = _plan()
    snapshot = _knowledge_snapshot()
    samples = tuple(_FakeSample(sample_id, AudioRef(f"a-{i}.wav", _SHA_A), 2.0)
        for i, sample_id in enumerate(sample_ids))
    tasks = tuple(runner._PreparedAgentTask(
        index=index, sample=sample, task_contract=f"task-{i}",
        initial_query_state=runner._initial_query_state(
            plan=plan, sample=sample, descriptor={
                "packet_id": f"fake/dev/{i}", "sha256": _SHA_C},
            bootstrap_facet_ids=(), bootstrap_value_ids=(),
        ),
    ) for i, (index, sample) in enumerate(zip(indices, samples, strict=True)))
    if counted:
        tasks = tuple(_CountedTask(task) for task in tasks)
    controllers = tuple(_NamedController(f"node-{i}") for i in range(len(tasks)))
    remaining = iter(controllers)
    client = _PreparedClient()
    batches = []
    outcomes = runner._run_agent_openjiuwen(
        prepared_tasks=tasks, snapshot=snapshot, snapshot_sha256=snapshot.content_hash(),
        protected_material=runner._protected_knowledge_material(snapshot),
        adapter=_FakeAdapter(samples), controller_factory=lambda _sample: next(remaining),
        controller_policy_id="lookup-test", core_client=client, max_contacts=1,
        slot_count=1, queue_depth=2, base_decoding_params={"temperature": 0.25},
        trace_sink=TraceBuffer(plan.run_id), run_id=plan.run_id,
        thinking_policy={f"node-{i}": {"enable_thinking": False,
            "answer_reserve_tokens": 20 + i} for i in range(len(tasks))},
        request_batch_gate=batches.append if fit else None,
        whole_roster_wave_barrier=fit,
    )
    return outcomes, controllers, client, batches, tasks


@pytest.mark.parametrize("fit", [False, True])
def test_sparse_task_indices_keep_each_controllers_thinking_at_fit_and_dispatch(fit):
    outcomes, controllers, client, batches, _ = _run((19, 3), ("s1", "s2"), fit=fit)
    assert len(outcomes) == 2
    assert [options["decoding_params"]["max_tokens"] for _, options in client.calls] == [20, 21]
    assert all(options["decoding_params"]["id_slot"] == 0 for _, options in client.calls)
    assert all(options["decoding_params"]["temperature"] == 0 for _, options in client.calls)
    assert [len(controller.named_contacts) for controller in controllers] == [1 + fit, 1 + fit]
    if fit:
        assert len(batches) == 1 and len(batches[0]) == 2
        assert all("id_slot" not in request.as_payload()["decoding_params"] for request in batches[0])


def test_distinct_tasks_of_one_parent_keep_independent_controllers():
    outcomes, controllers, client, _, _ = _run((19, 3), ("parent", "parent"))
    assert len(outcomes) == 2
    assert [options["decoding_params"]["max_tokens"] for _, options in client.calls] == [20, 21]
    assert [len(controller.named_contacts) for controller in controllers] == [1, 1]


def test_repeated_task_index_preserves_previous_first_match_controller():
    outcomes, controllers, client, _, _ = _run((19, 19), ("s1", "s2"))
    assert len(outcomes) == 2
    assert [options["decoding_params"]["max_tokens"] for _, options in client.calls] == [20, 20]
    assert [len(controller.named_contacts) for controller in controllers] == [2, 0]


def test_repeated_task_index_still_refuses_a_fitted_roster():
    with pytest.raises(runner.RailsRunnerError, match="complete unique roster"):
        _run((19, 19), ("s1", "s2"), fit=True)


def test_full_wave_controller_lookup_does_not_rescan_prior_tasks():
    # Count accesses, not wall-clock time or a synthetic model-cost estimate.
    count = 32
    outcomes, _, _, _, tasks = _run(tuple(range(count)),
        tuple(f"s{i}" for i in range(count)), fit=True, counted=True)
    assert len(outcomes) == count
    assert max(task.index_reads for task in tasks) < 16
