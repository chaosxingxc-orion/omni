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


def test_dns_failure_is_pre_dispatch(monkeypatch):
    def fail(*a, **kw): raise socket.gaierror(-3, 'sensitive details')
    monkeypatch.setattr(m.socket, 'getaddrinfo', fail)
    r=m.probe(config())
    assert r['state']=='pre-dispatch-rejected' and r['failure_phase']=='dns'
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
