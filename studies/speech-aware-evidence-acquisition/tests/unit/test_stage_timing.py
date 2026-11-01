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


def test_lock_wait_excludes_protected_body_and_keeps_identity(sink, monkeypatch):
    import threading
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(m.time, 'perf_counter', lambda: next(clock))
    lock = threading.Lock()
    with m.measured_lock(lock, 'fit_lock_wait', lambda: {'request_id': 'r1'}):
        assert lock.locked()
        # No timing clock read occurs for the protected operation.
    assert not lock.locked()
    event = json.loads(sink.read_text())
    assert event['elapsed_seconds'] == 2.0
    assert event['request_id'] == 'r1'
    assert event['stage'] == 'fit_lock_wait'


def test_lock_released_if_timing_write_fails(sink, monkeypatch):
    import threading
    lock = threading.Lock()
    def fail(*args):
        raise OSError('sink unavailable')
    monkeypatch.setattr(m, '_record', fail)
    with pytest.raises(OSError, match='sink unavailable'):
        with m.measured_lock(lock, 'fit_lock_wait'):
            pytest.fail('protected operation must not run after sink failure')
    assert not lock.locked()


def test_lock_released_on_body_failure(sink):
    import threading
    lock = threading.Lock()
    with pytest.raises(ValueError, match='operation failed'):
        with m.measured_lock(lock, 'fit_lock_wait'):
            raise ValueError('operation failed')
    assert not lock.locked()
    assert json.loads(sink.read_text())['status'] == 'ok'


def test_disabled_lock_timing_keeps_lock_without_identity_inspection(monkeypatch):
    import threading
    monkeypatch.delenv('SAEA_STAGE_TIMING_PATH', raising=False)
    lock = threading.Lock()
    with m.measured_lock(lock, 'unused', lambda: pytest.fail('must not inspect')):
        assert lock.locked()
    assert not lock.locked()
