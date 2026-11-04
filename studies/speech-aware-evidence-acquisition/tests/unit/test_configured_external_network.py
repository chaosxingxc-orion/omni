import importlib.util
import json
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest

SCRIPT = Path(__file__).parents[2]/'scripts/environment/configured_external_network.py'
spec = importlib.util.spec_from_file_location('network_probe', SCRIPT)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def config(endpoint='https://localhost/v1/chat/completions'):
    return {'schema':'saea-external-network-v1','endpoint':endpoint,'probe_path':'/health',
            'timeout_seconds':2,'total_timeout_seconds':5,'accepted_statuses':[204]}


@pytest.fixture(autouse=True)
def clean_proxy_environment(monkeypatch):
    import os
    for key in list(os.environ):
        if key.lower().endswith('_proxy'):
            monkeypatch.delenv(key, raising=False)


def test_dns_failure_is_pre_dispatch(monkeypatch):
    def fail(*a, **kw): raise socket.gaierror(-3, 'sensitive details')
    monkeypatch.setattr(m.socket, 'getaddrinfo', fail)
    r=m.probe(config())
    assert r['state']=='pre-dispatch-rejected' and r['failure_phase']=='transport-dns'
    assert r['errno']==-3 and r['generation_dispatched'] is False
    assert 'sensitive' not in json.dumps(r)


@pytest.mark.parametrize('change', [{'endpoint':'http://localhost/v1/chat/completions'},
    {'endpoint':'https://secret@localhost/v1/chat/completions'}, {'endpoint':'https://localhost/v1/chat/completions?key=x'},
    {'probe_path':'//outside'}, {'probe_path':'/x?secret=y'}, {'timeout_seconds':0}, {'accepted_statuses':[]}])
def test_invalid_config(change):
    cfg=config();cfg.update(change)
    with pytest.raises(ValueError):m.validate(cfg)


@pytest.fixture
def tls_server(tmp_path):
    cert=tmp_path/'cert.pem';key=tmp_path/'key.pem'
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-keyout',str(key),'-out',str(cert),
                    '-days','1','-subj','/CN=localhost','-addext','subjectAltName=DNS:localhost'],check=True,capture_output=True)
    observed=[]
    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            observed.append({'method':'HEAD','path':self.path,'authorization':self.headers.get('Authorization')})
            self.send_response(204 if self.path=='/health' else 302);self.send_header('Location','/health');self.end_headers()
        def log_message(self,*a):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.load_cert_chain(cert,key)
    server.socket=context.wrap_socket(server.socket,server_side=True)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield server,cert,observed
    server.shutdown();server.server_close();thread.join()


def test_real_tls_head_and_batch_gate(tls_server,tmp_path):
    server,cert,observed=tls_server
    cfg=config(f'https://localhost:{server.server_port}/v1/chat/completions');cfg['ca_file']=str(cert)
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg));receipt=tmp_path/'receipt.json';marker=tmp_path/'started'
    r=subprocess.run([sys.executable,str(SCRIPT),'run','--config',str(path),'--receipt',str(receipt),'--',sys.executable,'-c',f'from pathlib import Path;Path({str(marker)!r}).write_text("yes")'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert marker.exists() and json.loads(receipt.read_text())['state']=='ready'
    assert observed==[{'method':'HEAD','path':'/health','authorization':None}]
    again=subprocess.run([sys.executable,str(SCRIPT),'run','--config',str(path),'--receipt',str(receipt),'--',sys.executable,'-c','raise Exception()'],capture_output=True,text=True)
    assert again.returncode!=0 and len(observed)==1


def test_untrusted_certificate_prevents_http(tls_server):
    server,cert,observed=tls_server
    r=m.probe(config(f'https://localhost:{server.server_port}/v1/chat/completions'))
    assert r['state']=='pre-dispatch-rejected' and r['error_type']=='SSLCertVerificationError'
    assert observed==[]


def test_rejected_status_never_launches_batch(tls_server,tmp_path):
    server,cert,observed=tls_server
    cfg=config(f'https://localhost:{server.server_port}/v1/chat/completions');cfg.update(ca_file=str(cert),probe_path='/redirect')
    p=tmp_path/'config.json';p.write_text(json.dumps(cfg));marker=tmp_path/'intent.json'
    r=subprocess.run([sys.executable,str(SCRIPT),'run','--config',str(p),'--receipt',str(tmp_path/'receipt.json'),'--',sys.executable,'-c',f'from pathlib import Path;Path({str(marker)!r}).write_text("bad")'],capture_output=True,text=True)
    assert r.returncode==4 and not marker.exists()
    assert len(observed)==1 and json.loads(r.stdout)['http_status']==302


def test_probe_total_deadline_blocks_command(tmp_path,monkeypatch):
    p=tmp_path/'config.json';p.write_text(json.dumps(config()))
    def timeout(*a,**kw):raise subprocess.TimeoutExpired('private details',1)
    monkeypatch.setattr(m.subprocess,'run',timeout)
    r=m.bounded_probe(p)
    assert r['state']=='pre-dispatch-rejected' and r['failure_phase']=='probe-total-deadline'
    assert 'private' not in json.dumps(r)


@pytest.fixture
def connect_proxy(tls_server):
    import select
    import socketserver
    server, cert, observed = tls_server
    connects = []
    class Tunnel(socketserver.StreamRequestHandler):
        def handle(self):
            line = self.rfile.readline().decode().strip()
            headers = []
            while (line2 := self.rfile.readline()) not in (b'\r\n', b'\n', b''):
                headers.append(line2.decode().strip())
            connects.append({'request': line, 'headers': headers})
            with socket.create_connection(('127.0.0.1', server.server_port), timeout=3) as upstream:
                self.wfile.write(b'HTTP/1.1 200 Connection established\r\n\r\n'); self.wfile.flush()
                sockets = [self.connection, upstream]
                while True:
                    ready, _, _ = select.select(sockets, [], [], 3)
                    if not ready: return
                    for source in ready:
                        data = source.recv(65536)
                        if not data: return
                        (upstream if source is self.connection else self.connection).sendall(data)
    proxy = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Tunnel)
    proxy.daemon_threads = True
    thread = threading.Thread(target=proxy.serve_forever, daemon=True); thread.start()
    yield proxy, connects
    proxy.shutdown(); proxy.server_close(); thread.join()


def test_environment_connect_delegates_target_dns(tls_server, connect_proxy, monkeypatch):
    server, cert, observed = tls_server
    proxy, connects = connect_proxy
    monkeypatch.setenv('https_proxy', f'http://127.0.0.1:{proxy.server_address[1]}')
    original = socket.getaddrinfo
    def target_dns_unavailable(host, *a, **kw):
        if host == 'localhost': raise socket.gaierror(-3, 'target local DNS unavailable')
        return original(host, *a, **kw)
    monkeypatch.setattr(socket, 'getaddrinfo', target_dns_unavailable)
    cfg = config(f'https://localhost:{server.server_port}/v1/chat/completions')
    cfg['ca_file'] = str(cert)
    result = m.probe(cfg)
    assert result['state'] == 'ready'
    assert result['proxy_mode'] == 'inherit-environment' and result['tls_verified']
    assert len(connects) == 1 and connects[0]['request'].startswith('CONNECT localhost:')
    assert observed == [{'method':'HEAD', 'path':'/health', 'authorization':None}]
    assert '127.0.0.1' not in json.dumps(result)


def test_direct_ignores_environment_and_no_proxy_is_honored(tls_server, monkeypatch):
    server, cert, observed = tls_server
    monkeypatch.setenv('https_proxy', 'http://127.0.0.1:1')
    cfg = config(f'https://localhost:{server.server_port}/v1/chat/completions')
    cfg.update(ca_file=str(cert), proxy_mode='direct')
    assert m.probe(cfg)['state'] == 'ready'
    cfg['proxy_mode'] = 'inherit-environment'
    monkeypatch.setenv('no_proxy', 'localhost')
    assert m.probe(cfg)['state'] == 'ready'
    assert len(observed) == 2


def test_proxy_credentials_not_in_failure_receipt(monkeypatch):
    monkeypatch.setenv('https_proxy', 'http://private-user:private-password@127.0.0.1:1')
    result = m.probe(config('https://not-resolved.invalid/v1/chat/completions'))
    assert result['state'] == 'pre-dispatch-rejected'
    serialized = json.dumps(result)
    assert all(secret not in serialized for secret in ('private-user', 'private-password', '127.0.0.1'))


def test_redirect_never_ready_even_if_status_allowed(tls_server):
    server, cert, observed = tls_server
    cfg = config(f'https://localhost:{server.server_port}/v1/chat/completions')
    cfg.update(ca_file=str(cert), probe_path='/redirect', accepted_statuses=[302])
    assert m.probe(cfg)['state'] == 'pre-dispatch-rejected'
    assert len(observed) == 1
