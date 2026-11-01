"""Identity debug events distinguish real reads from cache reuse without weakening gates."""
import json
import os
import pytest
from test_runtime_memory_identity_cache import fixture
from speech_aware_evidence_acquisition.core import stage_timing as timing
from speech_aware_evidence_acquisition.contracts import GateClosed


@pytest.fixture
def debug_sink(tmp_path,monkeypatch):
    path=tmp_path/'debug.jsonl'
    monkeypatch.setenv('SAEA_STAGE_TIMING_PATH',str(path))
    monkeypatch.setenv('SAEA_OBSERVABILITY_MODE','debug')
    monkeypatch.delenv('SAEA_OBSERVABILITY_CONFIG',raising=False)
    monkeypatch.setattr(timing,'_fd',None);monkeypatch.setattr(timing,'_path',None)
    monkeypatch.setattr(timing,'_configuration',None)
    yield path
    if timing._fd is not None:os.close(timing._fd)


def records(path):return [json.loads(line) for line in path.read_text().splitlines()]


def test_first_pass_reads_all_bytes_then_memory_hit_is_observed(fixture,debug_sink):
    f=fixture
    first=f.verify();second=f.verify()
    assert first['gguf_bytes_full_hashed']==f.path.stat().st_size
    assert second['gguf_bytes_full_hashed']==0
    assert first['gguf_bytes_verified']==second['gguf_bytes_verified']
    rows=records(debug_sink)
    hashes=[r for r in rows if r['stage']=='runtime_identity_full_hash_process_floor']
    hits=[r for r in rows if r['stage']=='runtime_identity_cache_hit_memory']
    assert len(hashes)==len(hits)==1
    assert hashes[0]['measurements']=={'file_count':1,'verified_bytes':f.path.stat().st_size}
    assert hits[0]['measurements']['cache_hits']==1
    assert len(f.calls)==1
    assert str(f.root) not in debug_sink.read_text() and 'original weights' not in debug_sink.read_text()
    whole=[r for r in rows if r['stage']=='runtime_identity_verify']
    assert len(whole)==2 and hashes[0]['parent_span_id']==whole[0]['span_id']


@pytest.mark.parametrize('change,expected',[
    ('forced','forced'),('metadata','model_identity_changed'),('cache','cache_file_changed')])
def test_full_read_reason_explains_invalidation(fixture,debug_sink,change,expected):
    f=fixture;f.verify()
    if change=='metadata':
        before=f.path.stat();os.utime(f.path,ns=(before.st_atime_ns,before.st_mtime_ns+1000000))
    if change=='cache':(f.root/'.saea-runtime-identity-cache.json').write_text('changed')
    f.verify(full=True if change=='forced' else None)
    assert len(f.calls)==2
    assert any(r['stage']=='runtime_identity_full_hash_'+expected for r in records(debug_sink))


def test_drift_still_fails_and_is_not_cached(fixture,debug_sink):
    f=fixture;f.files[0]['sha256']='0'*64
    with pytest.raises(GateClosed,match='SHA-256 drift'):f.verify()
    rows=records(debug_sink)
    assert next(r for r in rows if r['stage']=='runtime_identity_verify')['status']=='error'
    assert not any(r['stage']=='runtime_identity_cache_persist' for r in rows)


def test_normal_has_same_validation_without_debug_events(fixture,debug_sink,monkeypatch):
    monkeypatch.setenv('SAEA_OBSERVABILITY_MODE','normal')
    f=fixture;f.verify();f.verify()
    assert len(f.calls)==1 and not debug_sink.exists()
