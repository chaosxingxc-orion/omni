"""Ready preparation owns probe reuse through the real imported scheduler.

All inputs are generated in tmp_path. No model, HTTP probe or private fixture.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from speech_aware_evidence_acquisition.core import stage_timing as timing
from speech_aware_evidence_acquisition.core.session import vllm_identity as identity
from speech_aware_evidence_acquisition.rails.dispatch_slot_fill import DispatchSlotFill
from speech_aware_evidence_acquisition.rails.runner import _SlotFillScheduler

# Reuse the observability file's isolated timing sink and teardown discipline.
from test_debug_dispatch_observability import sink  # noqa: F401


@pytest.fixture(params=["normal", "debug"])
def observed_mode(request, sink, monkeypatch):
    monkeypatch.setenv("SAEA_OBSERVABILITY_MODE", request.param)
    return request.param


@pytest.fixture
def scheduler(observed_mode):
    scheduler = _SlotFillScheduler(DispatchSlotFill(
        role_input_offload=True, prepare_workers=1, validation_workers=1,
        ready_request=True,
    ))
    try:
        yield scheduler
    finally:
        scheduler.close()


def _cache():
    return getattr(identity._probe_local, "cache", None)


class _Connection:
    """An owned closeable handle; never opens a socket."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _owned_handle():
    cache = _cache()
    assert cache is not None
    handle = _Connection()
    cache.connections["test-handle"] = handle
    cache.socket_candidates["test-candidate"] = "test-fd"
    return cache, handle


def _assert_closed(cache, handle):
    assert handle.closed
    assert not cache.connections
    assert not cache.maps
    assert not cache.socket_candidates


async def _assert_worker_clean(executor, thread_id):
    # Submit directly to the same one-worker executor: another ready callback
    # would create a fresh scope and could conceal a leaked scope.
    current, cache = await asyncio.get_running_loop().run_in_executor(
        executor, lambda: (threading.get_ident(), _cache()),
    )
    assert current == thread_id
    assert cache is None


def test_ready_callback_reuses_parsing_but_reads_fresh_maps_and_closes(scheduler, tmp_path):
    pid = 17
    proc = tmp_path / "proc"
    maps = proc / str(pid) / "maps"
    maps.parent.mkdir(parents=True)
    library = tmp_path / "library.so"

    def callback():
        cache, handle = _owned_handle()
        raw = f"1000-2000 r-xp 00000000 00:00 0 {library}\n"
        maps.write_text(raw)
        assert identity._mapped_paths(proc, pid) == ([str(library)], [])
        key = (str(proc), pid)
        parsed = cache.maps[key]
        assert identity._mapped_paths(proc, pid) == ([str(library)], [])
        assert cache.maps[key] is parsed
        # Reading current bytes is mandatory even while the scope is active.
        maps.write_text(raw.rstrip() + " (deleted)\n")
        assert identity._mapped_paths(proc, pid) == ([], [str(library)])
        assert cache.maps[key] is not parsed
        with identity.http_probe_connection_scope():
            assert _cache() is cache
        assert _cache() is cache and not handle.closed
        return threading.get_ident(), cache, handle

    async def run():
        assert _cache() is None
        first = await scheduler.ready_request_runner(callback)
        _assert_closed(*first[1:])
        await _assert_worker_clean(scheduler._role_input, first[0])
        second = await scheduler.ready_request_runner(callback)
        assert second[1] is not first[1]
        _assert_closed(*second[1:])
        await _assert_worker_clean(scheduler._role_input, second[0])
        assert _cache() is None

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["role_input", "validation", "unlabelled"])
def test_non_ready_callbacks_do_not_create_scope(scheduler, kind):
    def callback(*args):
        assert _cache() is None
        return threading.get_ident(), args

    async def run():
        if kind == "role_input":
            result = await scheduler.role_input_runner(callback, "phase", "state")
            executor, expected = scheduler._role_input, ("phase", "state")
        elif kind == "validation":
            result = await scheduler.validation_runner(callback, "inputs")
            executor, expected = scheduler._validation, ("inputs",)
        else:
            result = await scheduler._run_bounded(
                scheduler._role_input, scheduler._prepare_capacity, callback, "plain",
            )
            executor, expected = scheduler._role_input, ("plain",)
        assert result[1] == expected
        await _assert_worker_clean(executor, result[0])

    asyncio.run(run())


def test_ready_exception_closes_scope_and_preserves_exception(scheduler):
    failure = ValueError("intentional callback failure")
    seen = []

    def callback():
        seen.append((threading.get_ident(), *_owned_handle()))
        raise failure

    async def run():
        with pytest.raises(ValueError) as caught:
            await scheduler.ready_request_runner(callback)
        assert caught.value is failure
        _assert_closed(*seen[0][1:])
        await _assert_worker_clean(scheduler._role_input, seen[0][0])
        assert not scheduler._prepare_capacity.locked()
        assert await scheduler.ready_request_runner(lambda: "recovered") == "recovered"

    asyncio.run(run())


def test_cancelled_ready_waits_for_worker_then_closes_scope(scheduler):
    entered, release = threading.Event(), threading.Event()
    seen = []

    def callback():
        seen.append((threading.get_ident(), *_owned_handle()))
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test worker was not released")
        assert _cache() is seen[0][1]
        assert not seen[0][2].closed

    async def run():
        task = asyncio.create_task(scheduler.ready_request_runner(callback))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert scheduler._prepare_capacity.locked()
            assert not seen[0][2].closed
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        _assert_closed(*seen[0][1:])
        await _assert_worker_clean(scheduler._role_input, seen[0][0])
        assert not scheduler._prepare_capacity.locked()
        assert await scheduler.ready_request_runner(lambda: "recovered") == "recovered"

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_two_ready_workers_have_distinct_caches(observed_mode):
    scheduler = _SlotFillScheduler(DispatchSlotFill(
        role_input_offload=True, prepare_workers=2, ready_request=True,
    ))
    barrier = threading.Barrier(2, timeout=5)

    def callback(marker):
        cache, handle = _owned_handle()
        # Both threads write the same key before either reads it.
        cache.maps["same-key"] = marker
        barrier.wait()
        assert _cache() is cache
        assert cache.maps["same-key"] == marker
        barrier.wait()
        return threading.get_ident(), cache, handle

    async def run():
        results = await asyncio.gather(*(
            scheduler.ready_request_runner(lambda marker=marker: callback(marker))
            for marker in ("first", "second")
        ))
        assert results[0][0] != results[1][0]
        assert results[0][1] is not results[1][1]
        for _, cache, handle in results:
            _assert_closed(cache, handle)
        assert _cache() is None

        # Pin both workers again to inspect their thread locals after exit.
        def inspect():
            barrier.wait()
            return threading.get_ident(), _cache()

        clean = await asyncio.gather(*(
            asyncio.get_running_loop().run_in_executor(scheduler._role_input, inspect)
            for _ in range(2)
        ))
        assert {row[0] for row in clean} == {row[0] for row in results}
        assert all(cache is None for _, cache in clean)

    try:
        asyncio.run(run())
    finally:
        scheduler.close()
