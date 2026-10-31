"""Timing preserves outputs/errors and separates concurrent request contexts."""
import json
from concurrent.futures import ThreadPoolExecutor
import os
import pytest
from speech_aware_evidence_acquisition.core import stage_timing as m

@pytest.fixture
def sink(tmp_path,monkeypatch):
    p=tmp_path/'timing.jsonl';monkeypatch.setenv('SAEA_STAGE_TIMING_PATH',str(p))
    monkeypatch.setattr(m,'_fd',None);monkeypatch.setattr(m,'_path',None)
    yield p
    if m._fd is not None:os.close(m._fd)

def test_nested_concurrent_identity_and_return_unchanged(sink):
    @m.measure('nested')
    def nested(value):return value
    @m.measure('request',lambda value:{'request_id':value['id']})
    def request(value):return nested(value)
    items=[{'id':str(i),'content':'not written'} for i in range(16)]
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(request,items))
    assert all(a is b for a,b in zip(results,items))
    events=[json.loads(line) for line in sink.read_text().splitlines()]
    assert len(events)==32
    for item in items:
        pair=[e for e in events if e['request_id']==item['id']]
        assert {e['stage'] for e in pair}=={'nested','request'}
        assert all(e['status']=='ok' and e['elapsed_seconds']>=0 for e in pair)
    assert 'not written' not in sink.read_text()

def test_error_and_context_reset(sink):
    error=ValueError('private text')
    @m.measure('bad',lambda:{'request_id':'failed'})
    def bad():raise error
    with pytest.raises(ValueError) as caught:bad()
    assert caught.value is error and m._context.get()=={}
    assert json.loads(sink.read_text())['status']=='error'
    assert 'private text' not in sink.read_text()

def test_disabled_keeps_no_io(monkeypatch,tmp_path):
    monkeypatch.delenv('SAEA_STAGE_TIMING_PATH',raising=False)
    @m.measure('unused',lambda:(_ for _ in ()).throw(AssertionError('must not inspect')))
    def operation():return 42
    assert operation()==42
