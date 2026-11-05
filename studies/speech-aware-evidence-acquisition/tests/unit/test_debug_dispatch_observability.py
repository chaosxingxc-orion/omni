"""Real async scheduling waits remain distinct from blocking worker execution."""
import asyncio
import json
import os
import time
from types import SimpleNamespace
import pytest
from speech_aware_evidence_acquisition.core import stage_timing as timing
from speech_aware_evidence_acquisition.reproduction.ojw import runner

@pytest.fixture
def sink(tmp_path, monkeypatch):
    path=tmp_path/'dispatch.jsonl'
    monkeypatch.setenv('SAEA_STAGE_TIMING_PATH',str(path))
    monkeypatch.setenv('SAEA_OBSERVABILITY_MODE','debug')
    monkeypatch.delenv('SAEA_OBSERVABILITY_CONFIG',raising=False)
    monkeypatch.setattr(timing,'_fd',None);monkeypatch.setattr(timing,'_path',None)
    monkeypatch.setattr(timing,'_configuration',None)
    yield path
    if timing._fd is not None:os.close(timing._fd)


def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize('mode',['normal','debug'])
@pytest.mark.parametrize('prefix',[False,True])
def test_real_scheduler_records_wait_and_preserves_results(sink,monkeypatch,mode,prefix):
    monkeypatch.setenv('SAEA_OBSERVABILITY_MODE',mode)
    states=[SimpleNamespace(sample_id='s1'),SimpleNamespace(sample_id='s2')]
    seen=[]
    async def episode(state, contact, session):
        def blocking(slot):
            seen.append((state.sample_id,slot));time.sleep(.01)
            return 'result-'+state.sample_id
        result=await contact(blocking,**({'prefix_family':'p'} if prefix else {}))
        assert result=='result-'+state.sample_id
    assert runner.execute_speech_agent_workflows(states,episode_invoker=episode,branch_count=1,
        slot_count=1,queue_depth=2,workflow_timeout_seconds=10,
        **({'prefix_slot_bindings':{'p':0}} if prefix else {}))==states
    assert sorted(seen)==[('s1',0),('s2',0)]
    assert timing._context.get()=={}
    if mode=='normal':
        assert not sink.exists();return
    records=rows(sink)
    for sample in ('s1','s2'):
        group={e['stage']:e for e in records if e.get('sample_id')==sample}
        assert {'dispatch_contact_lock_wait','dispatch_executor_wait','dispatch_blocking_execute'} <= set(group)
        assert ('dispatch_prefix_slot_wait' if prefix else 'dispatch_trajectory_slot_wait') in group
        wait=group['dispatch_executor_wait'];execute=group['dispatch_blocking_execute']
        assert wait['started_monotonic'] + wait['elapsed_seconds'] <= execute['started_monotonic']
        assert group['dispatch_blocking_execute']['status']=='ok'


def test_worker_failure_is_error_and_original_exception_survives(sink):
    def fail():raise ValueError('private failure')
    with timing.context({'sample_id':'s'}):
        fn,args=runner._observed_worker(fail)
        with pytest.raises(ValueError,match='private failure'):fn(*args)
    assert rows(sink)[-1]['status']=='error'
    assert 'private failure' not in sink.read_text()


def test_lock_released_when_observation_write_fails(sink,monkeypatch):
    def fail(*args):raise OSError('sink failed')
    monkeypatch.setattr(timing,'_record',fail)
    async def run():
        lock=asyncio.Lock()
        with pytest.raises(OSError,match='sink failed'):
            async with runner._observed_lock(lock,'wait'):pass
        assert not lock.locked()
    asyncio.run(run())


def test_cancellation_waiting_for_slot_does_not_release_owner(sink):
    async def run():
        lock=asyncio.Lock();await lock.acquire()
        async def wait():
            async with runner._observed_lock(lock,'wait'):raise AssertionError('acquired')
        task=asyncio.create_task(wait());await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert lock.locked();lock.release()
    asyncio.run(run())
    assert rows(sink)[0]['status']=='error'


def test_slot_is_returned_if_acquisition_event_write_fails(sink,monkeypatch):
    original=timing._record
    queues=[]
    queue_type=asyncio.Queue
    class Queue(queue_type):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);queues.append(self)
    monkeypatch.setattr(runner.asyncio,'Queue',Queue)
    def record(path,event):
        if event['stage']=='dispatch_trajectory_slot_wait':raise OSError('slot event failed')
        return original(path,event)
    monkeypatch.setattr(timing,'_record',record)
    calls=[]
    async def episode(state,contact,session):
        await contact(lambda slot:calls.append(slot))
    with pytest.raises(OSError,match='slot event failed'):
        runner.execute_speech_agent_workflows([SimpleNamespace(sample_id='s')],
            episode_invoker=episode,branch_count=1,slot_count=1,queue_depth=1,
            workflow_timeout_seconds=10)
    assert calls==[]
    assert len(queues)==1 and queues[0].qsize()==1

@pytest.mark.parametrize('mode',['normal','debug'])
@pytest.mark.parametrize('fails',[False,True])
def test_ordinary_runner_scopes_live_maps_without_retaining_connections(sink,monkeypatch,mode,fails):
    from pathlib import Path
    from speech_aware_evidence_acquisition.core.session import vllm_identity as vi
    monkeypatch.setenv('SAEA_OBSERVABILITY_MODE',mode)
    counts={'read':0,'decode':0}; observed=[]
    class Raw(bytes):
        def decode(self,*a,**kw):counts['decode']+=1;return super().decode(*a,**kw)
    def read(*args):counts['read']+=1;return Raw(b'1-2 r-xp 0 0:0 1 /a\n')
    monkeypatch.setattr(vi,'_read',read)
    async def episode(state,contact,session):
        def work(slot):
            vi._mapped_paths(Path('/proc'),1);vi._mapped_paths(Path('/proc'),1)
            cache=vi._probe_local.cache; observed.append(cache)
            if fails:raise ValueError('expected failure')
            return 'done'
        assert await contact(work)=='done'
    def run():return runner.execute_speech_agent_workflows([SimpleNamespace(sample_id='s')],episode_invoker=episode,
          branch_count=1,slot_count=1,queue_depth=1,workflow_timeout_seconds=10)
    if fails:
        with pytest.raises(Exception,match='expected failure'):run()
    else:run()
    assert counts=={'read':2,'decode':1}
    assert len(observed)==1 and not observed[0].maps and not observed[0].connections


def test_contact_scoped_slot_allows_other_ready_sample_while_first_validates(monkeypatch):
    import threading
    entered=threading.Event();second=threading.Event();order=[]
    async def episode(state,contact,session):
        def call(slot):
            order.append((state.sample_id,slot))
            if state.sample_id=='s2':second.set()
            return 'ok'
        await contact(call)
        if state.sample_id=='s1':
            entered.set()
            while not second.is_set():
                await asyncio.sleep(0.001)
    def run():
        return runner.execute_speech_agent_workflows([SimpleNamespace(sample_id='s1'),SimpleNamespace(sample_id='s2')],
            episode_invoker=episode,branch_count=1,slot_count=1,queue_depth=2,
            workflow_timeout_seconds=10,release_slot_after_contact=True)
    from concurrent.futures import ThreadPoolExecutor
    pool=ThreadPoolExecutor(max_workers=1)
    try:
        future=pool.submit(run)
        assert entered.wait(3)
        assert second.wait(3)
        future.result(timeout=5)
    finally:
        second.set();pool.shutdown(wait=True)
    assert sorted(x[0] for x in order)==['s1','s2']


def test_repeated_cancel_drains_actual_worker_before_slot_released(monkeypatch):
    import threading
    entered=threading.Event(); release=threading.Event(); second=threading.Event(); checks=[]
    async def episode(state,contact,session):
        if state.sample_id=='s1':
            def blocked(slot):entered.set();release.wait(5);return 'first'
            task=asyncio.create_task(contact(blocked))
            while not entered.is_set():await asyncio.sleep(.001)
            task.cancel();await asyncio.sleep(.01);task.cancel();await asyncio.sleep(.02)
            checks.append((task.done(),second.is_set()))
            release.set()
            with pytest.raises(asyncio.CancelledError):await task
        else:
            while not entered.is_set():
                await asyncio.sleep(.001)
            await contact(lambda slot:second.set())
    try:
        runner.execute_speech_agent_workflows([SimpleNamespace(sample_id='s1'),SimpleNamespace(sample_id='s2')],episode_invoker=episode,branch_count=1,slot_count=1,queue_depth=2,workflow_timeout_seconds=10,release_slot_after_contact=True)
    finally:release.set()
    assert checks==[(False,False)] and second.is_set()


def test_ready_waiting_for_model_slot_cancel_never_dispatches():
    import threading
    from speech_aware_evidence_acquisition.rails.runner import _SlotFillScheduler
    from speech_aware_evidence_acquisition.rails.dispatch_slot_fill import DispatchSlotFill
    entered, release, prepared, illegal_send = (threading.Event() for _ in range(4))
    scheduler = _SlotFillScheduler(DispatchSlotFill(role_input_offload=True,
        prepare_workers=1, model_inflight=1, ready_request=True))
    async def episode(state, contact, session):
        if state.sample_id == 's1':
            def blocking(slot):
                entered.set()
                if not release.wait(8):
                    raise TimeoutError('test send not released')
            await contact(blocking)
        else:
            while not entered.is_set():
                await asyncio.sleep(.001)
            ready = await scheduler.ready_request_runner(lambda: prepared.set() or object())
            task = asyncio.create_task(contact(lambda slot: illegal_send.set()))
            await asyncio.sleep(.03)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert ready is not None and not illegal_send.is_set()
            release.set()
    try:
        runner.execute_speech_agent_workflows([SimpleNamespace(sample_id='s1'),
            SimpleNamespace(sample_id='s2')], episode_invoker=episode, branch_count=1,
            slot_count=1, queue_depth=2, workflow_timeout_seconds=10,
            release_slot_after_contact=True)
        assert prepared.is_set() and not illegal_send.is_set()
    finally:
        release.set()
        scheduler.close()
