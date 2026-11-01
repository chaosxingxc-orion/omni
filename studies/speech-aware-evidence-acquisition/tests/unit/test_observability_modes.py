"""Operational diagnostics preserve request bytes and cannot invent server times."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
import pytest
from speech_aware_evidence_acquisition.core import stage_timing as m
from speech_aware_evidence_acquisition.core.model import transport

@pytest.fixture
def sink(tmp_path,monkeypatch):
    path=tmp_path/'timing.jsonl'
    monkeypatch.setenv('SAEA_STAGE_TIMING_PATH',str(path))
    monkeypatch.setenv('SAEA_OBSERVABILITY_MODE','debug')
    monkeypatch.delenv('SAEA_OBSERVABILITY_CONFIG',raising=False)
    monkeypatch.setattr(m,'_fd',None);monkeypatch.setattr(m,'_path',None);monkeypatch.setattr(m,'_configuration',None)
    yield path
    if m._fd is not None:os.close(m._fd)


def events(path):return [json.loads(line) for line in path.read_text().splitlines()]


def test_modes_snapshot_and_profile_change_refused(sink,monkeypatch):
    with m.span('fine',{'request_bytes':8}):pass
    cfg=json.loads(sink.with_name(sink.name+'.config.json').read_text())
    assert cfg['effective']['mode']=='debug'
    assert not cfg['capabilities']['ttft'] and not cfg['capabilities']['per_request_server_queue']
    monkeypatch.setenv('SAEA_OBSERVABILITY_MODE','normal')
    @m.measure('normal')
    def f():pass
    with pytest.raises(ValueError,match='profile changed'):f()


def test_normal_preserves_outer_timing_but_skips_details(sink,monkeypatch):
    monkeypatch.setenv('SAEA_OBSERVABILITY_MODE','normal')
    @m.measure('request_total',lambda:{'request_id':'r1'})
    def f():
        with m.span('private_fine',{'request_bytes':4}):return 42
    assert f()==42
    assert [x['stage'] for x in events(sink)]==['request_total']
    assert events(sink)[0]['observability_mode']=='normal'


def test_debug_interleaving_correlates_spans_and_request_ids(sink):
    @m.measure('request_total',lambda rid:{'request_id':rid})
    def f(rid):
        with m.span('outer'):
            with m.span('inner'):pass
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(f,map(str,range(12))))
    for rid in map(str,range(12)):
        group={x['stage']:x for x in events(sink) if x['request_id']==rid}
        assert group['inner']['parent_span_id']==group['outer']['span_id']
        assert group['inner']['started_unix_ns']>0
    assert m._context.get()=={} and m._active_span.get() is None


def test_debug_metadata_rejects_payloads(sink):
    with pytest.raises(ValueError,match='unapproved'):
        with m.span('bad',{'prompt':'secret'}):pass
    assert not sink.exists()


@pytest.mark.parametrize('mode',['normal','debug'])
def test_http_bytes_and_sampling_body_identical_in_both_modes(sink,monkeypatch,mode):
    monkeypatch.setenv('SAEA_OBSERVABILITY_MODE',mode)
    sent=[]
    class Response:
        status=200
        headers={'x-request-id':'internal-request-secret'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return b'{"choices":[]}'
    def open_(request,timeout):
        sent.append((request.data,dict(request.header_items()),timeout))
        return Response()
    monkeypatch.setattr(transport.urllib.request,'urlopen',open_)
    body=b'{"temperature":0,"stream":false,"messages":["private-prompt"]}'
    assert transport._default_post(30)('http://localhost:1234/v1/chat/completions',body)==b'{"choices":[]}'
    assert sent==[(body,{'Content-type':'application/json'},30)]
    rows=events(sink)
    assert 'private-prompt' not in sink.read_text() and 'internal-request-secret' not in sink.read_text()
    assert 'localhost' not in sink.read_text()
    assert {x['stage'] for x in rows}==({'http_request','client_http_open_to_headers','client_http_read_body'} if mode=='debug' else {'http_request'})
    assert all('server_queue_seconds' not in x and 'ttft_seconds' not in x for x in rows)


def test_profile_config_is_exact_and_normal_cadence_fixed(tmp_path,monkeypatch):
    p=tmp_path/'profile.json';monkeypatch.setenv('SAEA_OBSERVABILITY_CONFIG',str(p));monkeypatch.setenv('SAEA_OBSERVABILITY_MODE','normal')
    p.write_text(json.dumps({'mode':'normal','client_detail':False,'server_metrics_interval_seconds':600}))
    assert m.configuration()['server_metrics_interval_seconds']==600
    p=tmp_path/'changed-profile.json';monkeypatch.setenv('SAEA_OBSERVABILITY_CONFIG',str(p))
    p.write_text(json.dumps({'mode':'normal','client_detail':False,'server_metrics_interval_seconds':10}))
    with pytest.raises(ValueError,match='600'):m.configuration()


def test_profile_is_loaded_once_and_snapshot_pins_effective_values(sink,monkeypatch,tmp_path):
    p=tmp_path/'profile.json';p.write_text(json.dumps({'mode':'debug','client_detail':True,'server_metrics_interval_seconds':600}))
    monkeypatch.setenv('SAEA_OBSERVABILITY_CONFIG',str(p))
    with m.span('first'):pass
    p.write_text('invalid edited file after start')
    with m.span('second'):pass
    assert len(events(sink))==2
    snapshot=json.loads(sink.with_name(sink.name+'.config.json').read_text())
    assert snapshot['effective']['server_metrics_interval_seconds']==600


def test_open_response_closed_if_debug_journal_write_fails(sink,monkeypatch):
    closed=[]
    class Response:
        def close(self):closed.append(True)
    monkeypatch.setattr(transport.urllib.request,'urlopen',lambda *a,**kw:Response())
    def fail(*a):raise OSError('sink failed')
    monkeypatch.setattr(m,'_record',fail)
    with pytest.raises(OSError,match='sink failed'):
        transport._default_post(30)('http://localhost:1234',b'{}')
    assert closed==[True]
