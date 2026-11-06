"""FD hints avoid scans but never replace current endpoint ownership proof."""
import os
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


def test_live_hint_avoids_scan_and_reused_fd_falls_back(tmp_path, monkeypatch):
    world = VllmWorld(tmp_path)
    pid = 4100001
    world.spawn('tp2-a', pid=pid)
    fd = world.proc_root / str(pid) / 'fd'
    original = Path.iterdir
    scans = []
    def iterdir(path):
        if path == fd:
            scans.append(path)
        return original(path)
    monkeypatch.setattr(Path, 'iterdir', iterdir)
    verify = lambda: m._verify_owns_listening_socket(world.proc_root, pid, '127.0.0.1', 8200)
    with m.http_probe_connection_scope():
        verify()
        verify()
        assert len(scans) == 1
        target = os.readlink(fd / '3')
        (fd / '3').unlink()
        (fd / '3').symlink_to('socket:[999999999]')
        (fd / '7').symlink_to(target)
        verify()
        verify()
        assert len(scans) == 2
        (fd / '7').unlink()
        with pytest.raises(SessionError, match='does not own'):
            verify()
    assert not hasattr(m._probe_local, 'cache')


def test_hint_never_bypasses_changed_listen_table(tmp_path):
    world = VllmWorld(tmp_path)
    pid = 4100001
    world.spawn('tp2-a', pid=pid)
    with m.http_probe_connection_scope():
        m._verify_owns_listening_socket(world.proc_root, pid, '127.0.0.1', 8200)
        table = world.proc_root / 'net' / 'tcp'
        table.write_text(table.read_text().splitlines()[0] + '\n')
        with pytest.raises(SessionError, match='no LISTEN'):
            m._verify_owns_listening_socket(world.proc_root, pid, '127.0.0.1', 8200)


def test_hint_is_bounded_and_fork_discards_it(tmp_path, monkeypatch):
    fd = tmp_path / '1' / 'fd'
    fd.mkdir(parents=True)
    (fd / '3').symlink_to('socket:[123]')
    monkeypatch.setattr(m, '_proc_net_listen_entries',
                        lambda *args: [('127.0.0.1', port, 123) for port in range(8200, 8220)])
    with m.http_probe_connection_scope():
        for port in range(8200, 8220):
            m._verify_owns_listening_socket(tmp_path, 1, '127.0.0.1', port)
        cache = m._probe_local.cache
        assert len(cache.socket_candidates) == m._PROBE_MAX_CONNECTIONS
        cache.pid -= 1
        m._verify_owns_listening_socket(tmp_path, 1, '127.0.0.1', 8219)
        assert not cache.socket_candidates
        assert not hasattr(m._probe_local, 'cache')
