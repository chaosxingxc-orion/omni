"""Real loopback HTTP/1.1 probes reuse sockets, never identity observations."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import importlib.util
import json
import os
from pathlib import Path
import socket
import threading

import pytest

from speech_aware_evidence_acquisition.core.session import SessionError
from speech_aware_evidence_acquisition.core.session import vllm_identity as m
from speech_aware_evidence_acquisition.core import stage_timing


@pytest.fixture
def servers():
    opened = []
    m._close_http_probe_connections()

    def start(actions=()):
        class Server(ThreadingHTTPServer):
            daemon_threads = True
            def get_request(self):
                request, address = super().get_request()
                self.accepted += 1
                return request, address

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args):
                pass
            def do_GET(self):
                server = self.server
                if self.path == '/metrics':
                    raw = (b'vllm:num_requests_running 0\n'
                           b'vllm:num_requests_waiting 0\n')
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(raw)))
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.wfile.write(raw)
                    self.close_connection = True
                    return
                server.requests.append((self.path, self.client_address, self.headers.get('Connection')))
                action = server.actions.popleft() if server.actions else {}
                if action.get('timeout'):
                    server.release.wait(3)
                    self.close_connection = True
                    return
                if action.get('disconnect'):
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.close_connection = True
                    return
                raw = action.get('raw', json.dumps({'data': [{'id': action.get('model', 'served')}],
                    'request': len(server.requests)}).encode())
                self.send_response(action.get('status', 200))
                self.send_header('Content-Type', 'application/json')
                if action.get('chunked'):
                    self.send_header('Transfer-Encoding', 'chunked')
                else:
                    self.send_header('Content-Length', str(action.get('length', len(raw))))
                if action.get('close'):
                    self.send_header('Connection', 'close')
                self.end_headers()
                try:
                    if action.get('chunked'):
                        self.wfile.write(f'{len(raw):x}\r\n'.encode() + raw + b'\r\n0\r\n\r\n')
                    else:
                        self.wfile.write(raw)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                if action.get('close'):
                    self.close_connection = True

        server = Server(('127.0.0.1', 0), Handler)
        server.accepted = 0
        server.requests = []
        server.actions = deque(actions)
        server.release = threading.Event()
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        server.url = f'http://127.0.0.1:{server.server_port}/v1/models'
        opened.append((server, thread))
        return server

    yield start
    m._close_http_probe_connections()
    for server, thread in opened:
        server.release.set()
        server.shutdown()
        server.server_close()
        thread.join(3)


@pytest.fixture
def probe_scope():
    with m.http_probe_connection_scope():
        yield


def test_each_probe_reaches_server_on_same_socket_and_reads_new_body(servers, probe_scope):
    server = servers()
    assert [m._http_get_json(server.url, 1)['request'] for _ in range(3)] == [1, 2, 3]
    assert len(server.requests) == 3 and server.accepted == 1
    assert len({row[1] for row in server.requests}) == 1
    assert all(row[0] == '/v1/models' and row[2] != 'close' for row in server.requests)


@pytest.mark.parametrize('action', [{'close': True}, {'chunked': True}])
def test_complete_close_or_chunked_response_has_exact_reuse_semantics(servers, probe_scope, action):
    server = servers([action])
    assert m._http_get_json(server.url, 1)['request'] == 1
    assert m._http_get_json(server.url, 1)['request'] == 2
    assert server.accepted == (2 if action.get('close') else 1)


@pytest.mark.parametrize('primed', [False, True])
@pytest.mark.parametrize('action', [
    {'status': 500}, {'status': 302}, {'raw': b'bad json'}, {'raw': b'\xff'},
    {'timeout': True}, {'disconnect': True},
    {'raw': b'{}', 'length': 20, 'close': True},
    {'raw': b'{}', 'length': m._PROBE_MAX_RESPONSE_BYTES + 1},
    {'raw': b'x' * (m._PROBE_MAX_RESPONSE_BYTES + 1), 'chunked': True},
])
def test_failure_discards_connection_without_retry_and_next_call_is_fresh(servers, probe_scope, action, primed):
    server = servers([{}, action] if primed else [action])
    if primed:
        m._http_get_json(server.url, .1)
    with pytest.raises((ValueError, OSError, http.client.HTTPException)):
        m._http_get_json(server.url, .1)
    assert len(server.requests) == 1 + primed and server.accepted == 1
    assert not m._probe_local.cache.connections
    server.release.set()
    assert m._http_get_json(server.url, 1)['request'] == 2 + primed
    assert server.accepted == 2


def test_endpoint_and_timeout_connections_are_separate(servers, probe_scope):
    first, second = servers(), servers()
    for server in (first, second, first, second):
        m._http_get_json(server.url, 1)
    assert first.accepted == second.accepted == 1
    m._http_get_json(first.url, 2)
    assert first.accepted == 2


def test_threads_never_share_probe_socket(servers):
    server = servers()
    barrier = threading.Barrier(2)
    def request_pair(_):
        with m.http_probe_connection_scope():
            m._http_get_json(server.url, 1)
            barrier.wait(3)
            m._http_get_json(server.url, 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(request_pair, range(2)))
    assert len(server.requests) == 4 and server.accepted == 2
    assert sorted(sum(row[1] == address for row in server.requests)
        for address in {row[1] for row in server.requests}) == [2, 2]


def test_endpoint_cache_is_bounded_and_evicted_socket_is_closed(servers, probe_scope, monkeypatch):
    monkeypatch.setattr(m, '_PROBE_MAX_CONNECTIONS', 2)
    first, second, third = servers(), servers(), servers()
    m._http_get_json(first.url, 1)
    held = next(iter(m._probe_local.cache.connections.values()))
    for server in (second, third):
        m._http_get_json(server.url, 1)
    assert len(m._probe_local.cache.connections) == 2 and held.sock is None
    m._http_get_json(first.url, 1)
    assert first.accepted == 2


def test_inherited_process_cache_never_reuses_parent_connection(servers, probe_scope):
    server = servers()
    m._http_get_json(server.url, 1)
    held = next(iter(m._probe_local.cache.connections.values()))
    m._probe_local.cache.pid = -1
    m._http_get_json(server.url, 1)
    assert held.sock is None and server.accepted == 2
    assert not hasattr(m._probe_local, 'cache')


@pytest.mark.parametrize('url', [
    'https://127.0.0.1:1/v1/models', 'http://example.com:1/v1/models',
    'http://user@127.0.0.1:1/v1/models', 'http://127.0.0.1:1/other',
    'http://127.0.0.1:1/v1/models?x=1',
])
def test_non_probe_urls_refuse_before_connect(url):
    with pytest.raises((ValueError, SessionError)):
        m._http_get_json(url, 1)


@pytest.mark.parametrize('timeout', [0, -1, True, float('nan'), float('inf')])
def test_invalid_timeout_refuses_before_connect(timeout):
    with pytest.raises(ValueError):
        m._http_get_json('http://127.0.0.1:1/v1/models', timeout)


def test_real_session_rechecks_model_and_process_with_keepalive(servers, probe_scope, tmp_path):
    from _vllm_world import VllmWorld, SERVED
    server = servers([{'model': SERVED}, {'model': 'changed'}])
    world = VllmWorld(tmp_path)
    receipt = world.receipt()
    receipt['deployment'].update(namespace_policy='shared-pid-host',
        harness_hostname=m.current_hostname(world.proc_root))
    receipt['deployment']['instances'][0]['port'] = server.server_port
    world.spawn('tp2-a', pid=4100001, receipt=receipt)
    session = world.session('tp2-a', 4100001, receipt=receipt)
    m.assert_vllm_session_alive(session, receipt, proc_root=world.proc_root)
    with pytest.raises(SessionError, match='not the recorded'):
        m.assert_vllm_session_alive(session, receipt, proc_root=world.proc_root)
    assert server.accepted == 1 and len(server.requests) == 2
    (world.proc_root / '4100001/stat').write_text('4100001 (python) Z 1')
    with pytest.raises(SessionError, match='state Z'):
        m.assert_vllm_session_alive(session, receipt, proc_root=world.proc_root)
    assert len(server.requests) == 2


def test_debug_spans_preserve_identity_and_hide_endpoint(servers, tmp_path, monkeypatch):
    from _vllm_world import VllmWorld, SERVED
    path = tmp_path/'timing.jsonl'
    monkeypatch.setenv('SAEA_STAGE_TIMING_PATH', str(path))
    monkeypatch.setenv('SAEA_OBSERVABILITY_MODE', 'debug')
    monkeypatch.delenv('SAEA_OBSERVABILITY_CONFIG', raising=False)
    monkeypatch.setattr(stage_timing, '_fd', None)
    monkeypatch.setattr(stage_timing, '_path', None)
    monkeypatch.setattr(stage_timing, '_configuration', None)
    server = servers([{'model': SERVED}])
    world = VllmWorld(tmp_path/'world')
    receipt = world.receipt()
    receipt['deployment'].update(namespace_policy='shared-pid-host',
        harness_hostname=m.current_hostname(world.proc_root))
    receipt['deployment']['instances'][0]['port'] = server.server_port
    world.spawn('tp2-a', pid=4100001, receipt=receipt)
    try:
        m.assert_vllm_session_alive(world.session('tp2-a', 4100001, receipt=receipt),
            receipt, proc_root=world.proc_root)
    finally:
        if stage_timing._fd is not None:
            os.close(stage_timing._fd)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert {row['stage'] for row in events} == {'session_total', 'session_declared',
        'session_namespace', 'session_namespace_probe', 'session_image_hash',
        'session_maps', 'session_socket_owner'}
    assert all(row['status'] == 'ok' for row in events)
    assert server.url not in path.read_text() and str(world.interpreter) not in path.read_text()
    assert all(set(row['measurements']) <= stage_timing._METADATA for row in events)


def test_normal_mode_does_not_write_probe_debug_spans(servers, tmp_path, monkeypatch):
    path = tmp_path/'normal.jsonl'
    monkeypatch.setenv('SAEA_STAGE_TIMING_PATH', str(path))
    monkeypatch.setenv('SAEA_OBSERVABILITY_MODE', 'normal')
    monkeypatch.delenv('SAEA_OBSERVABILITY_CONFIG', raising=False)
    server = servers()
    assert m._http_get_json(server.url, 1)['request'] == 1
    assert not path.exists()


def test_default_probes_never_leave_persistent_connections(servers):
    server = servers()
    for index in range(3):
        assert m._http_get_json(server.url, 1)['request'] == index + 1
        assert not hasattr(m._probe_local, 'cache')
    assert server.accepted == 3


@pytest.mark.parametrize('fail', [False, True])
def test_owned_scope_closes_sockets_on_success_or_error(servers, fail):
    server = servers()
    held = []
    def phase():
        with m.http_probe_connection_scope():
            m._http_get_json(server.url, 1)
            held.extend(m._probe_local.cache.connections.values())
            if fail:
                raise ValueError('phase failed')
    if fail:
        with pytest.raises(ValueError, match='phase failed'):
            phase()
    else:
        phase()
    assert len(held) == 1 and held[0].sock is None
    assert not hasattr(m._probe_local, 'cache')
    m._http_get_json(server.url, 1)
    assert server.accepted == 2


def test_nested_scope_preserves_outer_connection_and_closes_at_owner_exit(servers):
    server = servers()
    with m.http_probe_connection_scope():
        m._http_get_json(server.url, 1)
        held = next(iter(m._probe_local.cache.connections.values()))
        with m.http_probe_connection_scope():
            m._http_get_json(server.url, 1)
        assert held.sock is not None
        m._http_get_json(server.url, 1)
    assert held.sock is None and server.accepted == 1
    assert not hasattr(m._probe_local, 'cache')


def test_thread_without_owned_scope_cannot_borrow_another_threads_pool(servers):
    server = servers()
    with m.http_probe_connection_scope():
        m._http_get_json(server.url, 1)
        def request_pair():
            for _ in range(2):
                m._http_get_json(server.url, 1)
                assert not hasattr(m._probe_local, 'cache')
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(request_pair).result(timeout=3)
        m._http_get_json(server.url, 1)
    assert len(server.requests) == 4 and server.accepted == 3


def test_existing_quiescence_rejects_live_scope_and_accepts_owned_cleanup(servers):
    spec = importlib.util.spec_from_file_location('probe_quiescence',
        Path(__file__).parents[2]/'scripts/environment/transition_serving_budget.py')
    transition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transition)
    first, second = servers(), servers()
    instances = [{'host': '127.0.0.1', 'port': server.server_port}
                 for server in (first, second)]
    for server in (first, second):
        m._http_get_json(server.url, 1)
    assert transition.quiescence(instances)['established_connections'] == []
    with m.http_probe_connection_scope():
        for server in (first, second):
            m._http_get_json(server.url, 1)
        with pytest.raises(ValueError, match='established inference connection'):
            transition.quiescence(instances)
    assert transition.quiescence(instances)['established_connections'] == []


@pytest.mark.parametrize('failure', [None, 'render', 'audio_sha', 'tokenizer'])
def test_direct_whole_wave_owns_probe_scope_only_during_render(servers, tmp_path, failure):
    from speech_aware_evidence_acquisition.core.preparation import PreparationError
    from test_direct_only_fit import _Runtime, _Transport, _Tokenizer, _build, _batch, _request, ROWS
    first, second = servers(), servers()
    held = []
    class Transport(_Transport):
        def render_request(self, raw):
            for server in (first, second):
                m._http_get_json(server.url, 1)
            held.extend(m._probe_local.cache.connections.values())
            if failure == 'render' and len(self.calls) == 1:
                raise ValueError('render failed')
            return super().render_request(raw)
    class Tokenizer(_Tokenizer):
        def request_json(self, method, path, payload):
            assert not hasattr(m._probe_local, 'cache')
            assert all(connection.sock is None for connection in held)
            if failure == 'tokenizer':
                raise ValueError('tokenizer failed')
            return super().request_json(method, path, payload)
    runtime = _Runtime(tmp_path)
    runtime._transport = Transport()
    gate, tokenizer = _build(runtime, tokenizer=Tokenizer())
    batch = _batch()
    if failure == 'audio_sha':
        batch = (_request(ROWS[0], raw_speech_query={
            'media_relpath': f'media/{ROWS[0]}.wav', 'sha256': 'f' * 64}), *batch[1:])
    if failure:
        with pytest.raises((ValueError, PreparationError), match='failed|rendered audio differs|multimodal tokenize capability unavailable'):
            gate('direct', batch)
    else:
        gate('direct', batch)
        assert len(tokenizer.calls) == len(ROWS)
    rendered = 1 if failure == 'audio_sha' else 2 if failure == 'render' else len(ROWS)
    assert len(first.requests) == len(second.requests) == rendered
    assert first.accepted == second.accepted == 1
    assert held and all(connection.sock is None for connection in held)
    assert not hasattr(m._probe_local, 'cache')
    if failure in {'render', 'audio_sha'}:
        assert not tokenizer.calls
    for server in (first, second):
        m._http_get_json(server.url, 1)
    assert first.accepted == second.accepted == 2
