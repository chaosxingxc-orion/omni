from __future__ import annotations

import asyncio
import threading

import pytest
from test_llama_server_transport import _spawn_live_matching_process  # noqa: F401

from speech_aware_evidence_acquisition.reproduction.ojw.node_pipeline import NodePipelineCoordinator
from speech_aware_evidence_acquisition.reproduction.ojw.queue_adapter import QueueCallbackAdapter
from speech_aware_evidence_acquisition.reproduction.ojw.queue_pipeline_bridge import QueuePipelineBridge
from speech_aware_evidence_acquisition.reproduction.ojw.queue_state import QueueStateMachine, QueueStateError


def bridge_at(tmp_path, slots=1):
    path = tmp_path / 'queue.json'
    payloads, events = [], []
    bridge = QueuePipelineBridge.create(path, slots=slots)
    persist = bridge.adapter._persist
    def recorded():
        persist()
        payloads.append(path.read_text())
    bridge.adapter._persist = recorded
    return bridge, path, payloads, events


async def panel(bridge, episode, count=1, slots=1, event_sink=None):
    c = NodePipelineCoordinator(tuple(range(count)), ('p', 's'), slots=slots,
                                group_size=count, prepare_workers=2,
                                queue_bridge=bridge, event_sink=event_sink)
    async def one(member):
        error = None
        try:
            await episode(c, member)
        except BaseException as caught:
            error = caught
            raise
        finally:
            c.finish(member, error)
    driver = asyncio.create_task(c.drive())
    outcomes = await asyncio.gather(*(one(i) for i in range(count)), return_exceptions=True)
    await driver
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    return c


def test_public_path_checkpoint_contract_is_preserved(tmp_path):
    bridge, path, payloads, events = bridge_at(tmp_path)
    async def episode(c, member):
        assert await c.prepare(member, 'p', lambda: 'cpu') == 'cpu'
        assert await c.contact(member, 'p', lambda slot: {'slot': slot}) == {'slot': 0}
    asyncio.run(panel(bridge, episode))
    restored = QueueCallbackAdapter.restore(path)
    assert list(restored.machine.jobs.values())[0].state == 'observed'
    assert payloads[-1] == path.read_text()


def test_ready_panel_exceeds_capacity_without_early_admission(tmp_path):
    bridge, path, _, _ = bridge_at(tmp_path)
    entered, release = threading.Event(), threading.Event()
    async def episode(c, member):
        await c.prepare(member, 'p', lambda: member)
        def contact(slot):
            if member == 0:
                entered.set()
                assert release.wait(5)
            return member
        await c.contact(member, 'p', contact)
    async def run():
        execution = asyncio.create_task(panel(bridge, episode, count=3))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            state = QueueStateMachine.restore_file(path)
            assert len(state.inflight) == 1
            assert len(state.ready_queue) == 2
        finally:
            release.set()
        await execution
    asyncio.run(run())
    assert len(bridge.adapter.machine.jobs) == 3
    assert all(j.state == 'observed' for j in bridge.adapter.machine.jobs.values())


def test_model_tool_model_and_later_round_have_unique_dependencies(tmp_path):
    bridge, path, _, _ = bridge_at(tmp_path)
    seen = []
    async def episode(c, member):
        for role in ('p', 's', 'p'):
            await c.prepare(member, role, lambda: role)
            for kind in ('model', 'tool', 'model'):
                await c.contact(member, role, lambda slot: seen.append((role, kind)), kind)
    asyncio.run(panel(bridge, episode))
    jobs = sorted(QueueStateMachine.restore_file(path).jobs.values(), key=lambda j: j.stage)
    assert len(jobs) == len(seen) == 9
    assert [j.stage for j in jobs] == list(range(9))
    assert all(j.state == 'observed' for j in jobs)
    assert all(jobs[i].depends_on == jobs[i-1].job_id for i in range(1, 9))
    assert 'round-1:node-0' in jobs[-1].job_id


def test_preparation_failure_cancels_unsent_jobs_without_contacts(tmp_path):
    bridge, path, _, _ = bridge_at(tmp_path)
    sent = []
    async def episode(c, member):
        def prepare():
            if member == 0:
                raise RuntimeError('cpu failure')
            return member
        await c.prepare(member, 'p', prepare)
        await c.contact(member, 'p', lambda slot: sent.append(member))
    with pytest.raises(RuntimeError, match='cpu failure'):
        asyncio.run(panel(bridge, episode, count=3))
    state = QueueStateMachine.restore_file(path)
    assert not sent and not state.inflight and not state.ready_queue and not state.prepare_queue


def test_model_failure_drains_successful_peer_and_keeps_response(tmp_path):
    bridge, path, payloads, _ = bridge_at(tmp_path, slots=2)
    both = threading.Barrier(2)
    peer_release = threading.Event()
    async def episode(c, member):
        await c.prepare(member, 'p', lambda: member)
        async def after_abort():
            while c.failure is None:
                await asyncio.sleep(0)
            peer_release.set()
        observer = asyncio.create_task(after_abort()) if member == 1 else None
        def contact(slot):
            both.wait(5)
            if member == 0:
                raise RuntimeError('transport failure')
            assert peer_release.wait(5)
            return {'known': 'response'}
        try:
            await c.contact(member, 'p', contact)
        finally:
            if observer is not None:
                await observer
    with pytest.raises(RuntimeError, match='transport failure'):
        asyncio.run(panel(bridge, episode, count=3, slots=2))
    state = QueueStateMachine.restore_file(path)
    states = {j.sample_id: j.state for j in state.jobs.values()}
    assert states == {'0': 'failed', '1': 'observed', '2': 'cancelled'}
    assert not state.inflight
    assert any('draining' in payload for payload in payloads)


def test_logging_failure_preserves_completed_response_and_stops(tmp_path):
    bridge, path, _, _ = bridge_at(tmp_path)
    def fail_log(member, event):
        if event['kind'] == 'model':
            raise OSError('log unavailable')
    async def episode(c, member):
        await c.prepare(member, 'p', lambda: None)
        await c.contact(member, 'p', lambda slot: {'known': 'response'})
    with pytest.raises(OSError, match='log unavailable'):
        asyncio.run(panel(bridge, episode, event_sink=fail_log))
    state = QueueStateMachine.restore_file(path)
    assert list(state.jobs.values())[0].state == 'observed'
    assert not state.inflight


def test_cancelled_member_waits_for_original_contact_and_keeps_result(tmp_path):
    bridge, path, _, _ = bridge_at(tmp_path)
    entered, release = threading.Event(), threading.Event()
    async def run():
        c = NodePipelineCoordinator((0,), ('p',), slots=1, group_size=1,
                                    prepare_workers=1, queue_bridge=bridge)
        driver = asyncio.create_task(c.drive())
        await c.prepare(0, 'p', lambda: None)
        def contact(slot):
            entered.set()
            assert release.wait(5)
            return 'original response'
        task = asyncio.create_task(c.contact(0, 'p', contact))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert QueueStateMachine.restore_file(path).inflight
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        c.finish(0)
        await driver
    asyncio.run(run())
    assert list(QueueStateMachine.restore_file(path).jobs.values())[0].state == 'observed'


def test_checkpoint_admission_failure_does_not_invoke_contact(tmp_path):
    sent = []
    bridge, path, _, _ = bridge_at(tmp_path)
    persist = bridge.adapter._persist
    def fail_admit():
        if any(j.state == 'generation' for j in bridge.adapter.machine.jobs.values()):
            raise OSError('checkpoint unavailable')
        persist()
    bridge.adapter._persist = fail_admit
    async def episode(c, member):
        await c.prepare(member, 'p', lambda: None)
        await c.contact(member, 'p', lambda slot: sent.append(slot))
    with pytest.raises(OSError, match='checkpoint unavailable'):
        asyncio.run(panel(bridge, episode))
    assert sent == []
    assert list(bridge.adapter.machine.jobs.values())[0].state == 'draining'


def test_restored_or_reused_bridge_refuses_automatic_replay(tmp_path):
    bridge, _, _, _ = bridge_at(tmp_path)
    bridge.bind(slots=1)
    with pytest.raises(QueueStateError, match='reused'):
        bridge.bind(slots=1)
    bridge.prepare_submitted(0, (0, 0), 'p')
    restored = QueueCallbackAdapter.restore(bridge.adapter.checkpoint)
    with pytest.raises(QueueStateError, match='reconcile'):
        QueuePipelineBridge(restored)


def test_mismatched_capacity_refused_before_dispatch(tmp_path):
    bridge, _, _, _ = bridge_at(tmp_path)
    with pytest.raises(QueueStateError, match='capacity'):
        bridge.bind(slots=2)


def test_actual_workflow_runner_receives_bridge_and_preserves_outputs(tmp_path):
    from speech_aware_evidence_acquisition.reproduction.ojw.runner import execute_speech_agent_workflows
    outputs = []
    for enabled in (False, True):
        bridge, path, _, _ = bridge_at(tmp_path / str(enabled), slots=2)
        seen = []
        async def episode(state, contact, session, *, run_prepare):
            for role in ('p', 's'):
                value = await run_prepare(lambda: (state, role), prefix_family=role)
                for kind in ('model', 'tool', 'model'):
                    await contact(lambda slot: seen.append((value, kind)),
                                  prefix_family=role, contact_kind=kind)
        result = execute_speech_agent_workflows(
            list(range(3)), episode_invoker=episode, branch_count=1, slot_count=2,
            queue_depth=3, workflow_timeout_seconds=10, node_pipeline_order=('p', 's'),
            node_pipeline_group_size=3, node_pipeline_prepare_workers=2, node_pipeline_max_jobs=3,
            node_pipeline_queue_bridge=bridge if enabled else None)
        outputs.append((result, sorted(seen)))
        if enabled:
            jobs = QueueStateMachine.restore_file(path).jobs
            assert len(jobs) == 18 and all(j.state == 'observed' for j in jobs.values())
    assert outputs[0] == outputs[1]


def test_runner_bridge_requires_node_pipeline(tmp_path):
    from speech_aware_evidence_acquisition.reproduction.ojw.runner import execute_speech_agent_workflows
    bridge, _, _, _ = bridge_at(tmp_path)
    with pytest.raises(ValueError, match='declared node order'):
        execute_speech_agent_workflows([], branch_count=1, slot_count=1, queue_depth=1,
                                      workflow_timeout_seconds=10, node_pipeline_queue_bridge=bridge)


@pytest.mark.parametrize('workers', [0, 1])
def test_native_tool_workflow_uses_same_metered_contacts_with_bridge(tmp_path, monkeypatch, workers):
    from dataclasses import replace
    from test_native_episode_runner import _ActualEpisodeHarness
    from speech_aware_evidence_acquisition.rails.dispatch_slot_fill import parse_native_node_pipeline
    from speech_aware_evidence_acquisition.reproduction.ojw import node_pipeline

    bridge, path, _, _ = bridge_at(tmp_path, slots=2)
    original = node_pipeline.NodePipelineCoordinator
    def create(*args, **kwargs):
        kwargs['queue_bridge'] = bridge
        return original(*args, **kwargs)
    monkeypatch.setattr(node_pipeline, 'NodePipelineCoordinator', create)
    h = _ActualEpisodeHarness(tmp_path, sample_count=2, max_revisions=0, max_contacts=5,
                             planner_actions=('ACCEPT',), reviewer_task_tool=True)
    h.config = replace(h.config, local_tool_workers=workers)
    h.node_pipeline = parse_native_node_pipeline({'native_node_pipeline': {
        'schema': 'saea-native-node-pipeline-v1',
        'node_order': ['perception', 'solver', 'reviewer', 'planner'],
        'group_size': 2, 'prepare_workers': 2, 'max_jobs': 160}})
    h.slot_count = 2
    try:
        result = h.run()
    finally:
        h.close()
    assert len(result.results) == 2 and len(h.prepared) == 10 and len(h.tool_contacts) == 2
    jobs = QueueStateMachine.restore_file(path).jobs
    assert len(jobs) == (12 if workers == 0 else 10)
    assert all(j.state == 'observed' for j in jobs.values())
