"""Owned maps parsing reuse never caches liveness or receipt validation."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from _vllm_world import VllmWorld
from speech_aware_evidence_acquisition.core.session import SessionError
from speech_aware_evidence_acquisition.core.session import vllm_identity as m


@pytest.fixture(autouse=True)
def clean_scope():
    m._close_http_probe_connections()
    yield
    m._close_http_probe_connections()


def test_scope_rereads_bytes_but_reuses_parsing_and_returns_private_lists(monkeypatch):
    counts = {'read': 0, 'decode': 0}
    class Raw(bytes):
        def decode(self, *a, **kw):
            counts['decode'] += 1
            return super().decode(*a, **kw)
    def read(*args):
        counts['read'] += 1
        return Raw(b'1-2 r-xp 0 0:0 1 /a\n')
    monkeypatch.setattr(m, '_read', read)
    with m.http_probe_connection_scope():
        first, _ = m._mapped_paths(Path('/proc'), 1)
        first.append('/injected')
        with m.http_probe_connection_scope():
            assert m._mapped_paths(Path('/proc'), 1) == (['/a'], [])
        assert counts == {'read': 2, 'decode': 1}
    assert not hasattr(m._probe_local, 'cache')
    m._mapped_paths(Path('/proc'), 1)
    m._mapped_paths(Path('/proc'), 1)
    assert counts == {'read': 4, 'decode': 3}


@pytest.mark.parametrize('change,match', [('foreign','unpinned interpreter'), ('deleted','deleted mappings'), ('missing','not mapped'), ('utf8','non-UTF-8')])
def test_changed_maps_never_reuses_previous_success(tmp_path, change, match):
    world = VllmWorld(tmp_path)
    pid = 4100001
    world.spawn('tp2-a', pid=pid)
    receipt = world.receipt()
    maps = world.proc_root / str(pid) / 'maps'
    with m.http_probe_connection_scope():
        m._verify_maps(world.proc_root, pid, receipt, str(world.interpreter))
        if change == 'foreign':
            maps.write_bytes(maps.read_bytes() + b'1-2 r-xp 0 0:0 1 /tmp/libpython9.so\n')
        elif change == 'deleted':
            maps.write_bytes(maps.read_bytes() + f'1-2 r-xp 0 0:0 1 {world.trees["vllm"]}/x.py (deleted)\n'.encode())
        elif change == 'missing':
            maps.write_bytes(b'')
        else:
            maps.write_bytes(b'\xff')
        with pytest.raises(SessionError, match=match):
            m._verify_maps(world.proc_root, pid, receipt, str(world.interpreter))


def test_identical_maps_still_revalidates_changed_receipt(tmp_path):
    world = VllmWorld(tmp_path)
    world.spawn('tp2-a', pid=4100001)
    receipt = world.receipt()
    with m.http_probe_connection_scope():
        m._verify_maps(world.proc_root, 4100001, receipt, str(world.interpreter))
        changed = deepcopy(receipt)
        changed['runtime_libraries'].append({'path': '/missing-required.so'})
        with pytest.raises(SessionError, match='not mapped'):
            m._verify_maps(world.proc_root, 4100001, changed, str(world.interpreter))


def test_cache_is_bounded_and_large_maps_are_not_retained(monkeypatch):
    raw = b'1-2 r-xp 0 0:0 1 /a\n'
    monkeypatch.setattr(m, '_read', lambda *args: raw)
    monkeypatch.setattr(m, '_MAPS_CACHE_MAX_BYTES', len(raw) * 2)
    with m.http_probe_connection_scope():
        for pid in range(20):
            m._mapped_paths(Path('/proc'), pid)
        assert len(m._probe_local.cache.maps) == 2
        raw = b'x' * (len(raw) * 3)
        m._mapped_paths(Path('/proc'), 99)
        assert ('/proc', 99) not in m._probe_local.cache.maps


def test_other_thread_and_fork_identity_do_not_inherit_cache(monkeypatch):
    monkeypatch.setattr(m, '_read', lambda *args: b'1-2 r-xp 0 0:0 1 /a\n')
    with m.http_probe_connection_scope():
        m._mapped_paths(Path('/proc'), 1)
        saved = m._probe_local.cache
        def other():
            m._mapped_paths(Path('/proc'), 1)
            return hasattr(m._probe_local, 'cache')
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(other).result() is False
        saved.pid -= 1
        m._mapped_paths(Path('/proc'), 1)
        assert not hasattr(m._probe_local, 'cache')
        assert not saved.maps


def test_cached_maps_does_not_hide_process_death(tmp_path):
    world = VllmWorld(tmp_path)
    world.spawn('tp2-a', pid=4100001)
    session = world.session('tp2-a', 4100001)
    with m.http_probe_connection_scope():
        m.assert_vllm_session_alive(session, world.receipt(), proc_root=world.proc_root)
        (world.proc_root / '4100001' / 'stat').unlink()
        with pytest.raises(SessionError):
            m.assert_vllm_session_alive(session, world.receipt(), proc_root=world.proc_root)
