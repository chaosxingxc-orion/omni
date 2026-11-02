from __future__ import annotations

import asyncio
from pathlib import Path

from speech_aware_evidence_acquisition.reproduction.ojw.node_pipeline import NodePipelineCoordinator
from speech_aware_evidence_acquisition.reproduction.ojw.queue_pipeline_bridge import QueuePipelineBridge


def test_opt_in_bridge_persists_prepare_generation_and_resume(tmp_path: Path):
    bridge = QueuePipelineBridge.create(tmp_path / "queue.json", slots=1)
    coordinator = NodePipelineCoordinator(
        (0,), ("perception",), slots=1, group_size=1, prepare_workers=1, queue_bridge=bridge
    )

    async def run():
        driver = asyncio.create_task(coordinator.drive())
        value = await coordinator.prepare(0, "perception", lambda: "prepared")
        assert value == "prepared"
        assert await coordinator.contact(0, "perception", lambda slot: (slot, value)) == (0, "prepared")
        coordinator.finish(0)
        await driver

    asyncio.run(run())
    assert bridge.adapter.machine.jobs["member-0:round-0:stage-0"].state == "observed"
    assert bridge.adapter.restore(tmp_path / "queue.json").machine.jobs["member-0:round-0:stage-0"].state == "observed"


def test_default_coordinator_has_no_queue_side_effect(tmp_path: Path):
    coordinator = NodePipelineCoordinator(
        (0,), ("perception",), slots=1, group_size=1, prepare_workers=1
    )

    async def run():
        driver = asyncio.create_task(coordinator.drive())
        await coordinator.prepare(0, "perception", lambda: "prepared")
        await coordinator.contact(0, "perception", lambda slot: slot)
        coordinator.finish(0)
        await driver

    asyncio.run(run())
    assert not list(tmp_path.iterdir())


def test_second_stage_requires_observed_first_stage(tmp_path: Path):
    bridge = QueuePipelineBridge.create(tmp_path / "queue.json", slots=1)
    bridge.prepare_submitted(0, (0, 0), "perception")
    bridge.prepare_completed(0, ok=True)
    bridge.generation_admitted(0)
    bridge.generation_completed(0, observed=True, result={"ok": 1})
    bridge.prepare_submitted(0, (0, 1), "solver")
    bridge.prepare_completed(0, ok=True)
    assert bridge.adapter.machine.jobs["member-0:round-0:stage-1"].depends_on == "member-0:round-0:stage-0"


def test_later_round_has_monotonic_queue_stage(tmp_path: Path):
    bridge = QueuePipelineBridge.create(tmp_path / "queue.json", slots=1)
    bridge.prepare_submitted(0, (0, 3), "reviewer")
    bridge.prepare_completed(0, ok=True)
    bridge.generation_admitted(0)
    bridge.generation_completed(0, observed=True, result={"round": 0})
    bridge.prepare_submitted(0, (1, 0), "perception")
    assert bridge.adapter.machine.jobs["member-0:round-1:stage-1000000"].stage == 1_000_000


def test_ready_panel_larger_than_capacity_is_admitted_only_on_dispatch(tmp_path):
    import threading
    bridge = QueuePipelineBridge.create(tmp_path / 'queue.json', slots=1)
    c = NodePipelineCoordinator((0, 1, 2), ('p',), slots=1, group_size=3,
                                prepare_workers=2, queue_bridge=bridge)
    entered = threading.Event()
    release = threading.Event()
    observed = []

    async def run():
        async def member(i):
            error = None
            try:
                await c.prepare(i, 'p', lambda: i)
                def contact(slot):
                    if i == 0:
                        entered.set()
                        assert release.wait(3)
                    return i
                observed.append(await c.contact(i, 'p', contact))
            except BaseException as caught:
                error = caught
                raise
            finally:
                c.finish(i, error)
        driver = asyncio.create_task(c.drive())
        members = [asyncio.create_task(member(i)) for i in range(3)]
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            checkpoint = bridge.adapter.restore(tmp_path / 'queue.json').machine
            assert len(checkpoint.inflight) == 1
            assert len(checkpoint.ready_queue) == 2
        finally:
            release.set()
        await asyncio.gather(*members)
        await driver
    asyncio.run(run())
    assert observed == [0, 1, 2]
    assert all(j.state == 'observed' for j in bridge.adapter.machine.jobs.values())
