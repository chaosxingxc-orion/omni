"""Read-only model roots must not force repeated full hashes within a process."""
import hashlib
import os
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
import speech_aware_evidence_acquisition.contracts as contracts
from speech_aware_evidence_acquisition.contracts import GateClosed
from speech_aware_evidence_acquisition.contracts import gate_runtime_gguf as module


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_MEMORY_IDENTITIES', {})
    monkeypatch.setattr(module, '_MEMORY_CACHE_FILES', {})
    monkeypatch.setattr(contracts, '_SESSION_FULL_HASH_DONE', set())
    monkeypatch.delenv(contracts.RUNTIME_FULL_HASH_ENV, raising=False)
    # Same behavior as a failed best-effort write on the read-only data root.
    monkeypatch.setattr(contracts, '_write_identity_cache', lambda *args: None)
    root=tmp_path/'data';(root/'model').mkdir(parents=True)
    path=root/'model/weights';path.write_bytes(b'original weights')
    receipt=tmp_path/'receipt.json';receipt.write_text('{}')
    files=[{'path':'model/weights','bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]
    calls=[];original=contracts._measure_file
    def measure(path,label):
        calls.append(str(path));return original(path,label)
    monkeypatch.setattr(contracts,'_measure_file',measure)
    def verify(*,data=root,full=None):
        gate=SimpleNamespace(model_root=None,data_root=data,runtime_receipt=receipt)
        diagnostics={}
        module._GgufIdentityMixin._verify_gguf_files(gate,files,{'size_bytes':files[0]['bytes']},'model',full_hash=full,diagnostics=diagnostics)
        return diagnostics
    return SimpleNamespace(root=root,path=path,receipt=receipt,files=files,calls=calls,verify=verify,measure=measure)


def test_readonly_cache_failure_shared_across_gate_objects(fixture):
    f=fixture
    assert f.verify()['gguf_files_full_hashed']==1
    assert f.verify()['gguf_files_memory_cache_hits']==1
    assert f.verify()['gguf_files_full_hashed']==0
    assert len(f.calls)==1


def test_stale_readonly_disk_cache_does_not_force_repeated_hash(fixture):
    f=fixture;(f.root/contracts.RUNTIME_IDENTITY_CACHE_FILENAME).write_text('{}')
    f.verify();assert f.verify()['gguf_files_memory_cache_hits']==1
    assert len(f.calls)==1


def test_modified_disk_cache_invalidates_memory_fallback(fixture):
    f=fixture;f.verify()
    (f.root/contracts.RUNTIME_IDENTITY_CACHE_FILENAME).write_text('corrupt')
    assert f.verify()['gguf_files_full_hashed']==1
    assert len(f.calls)==2


@pytest.mark.parametrize('mutation',['mtime','inode','ctime'])
def test_metadata_changes_require_fresh_hash(fixture,mutation):
    f=fixture;f.verify();before=f.path.stat()
    if mutation=='mtime':os.utime(f.path,ns=(before.st_atime_ns,before.st_mtime_ns+1000000))
    elif mutation=='inode':
        other=f.path.with_name('replacement');other.write_bytes(f.path.read_bytes());other.replace(f.path)
    else:
        time.sleep(0.02)  # Container filesystem timestamps have finite resolution.
        f.path.write_bytes(f.path.read_bytes())
        os.utime(f.path,ns=(before.st_atime_ns,before.st_mtime_ns))
        assert f.path.stat().st_ctime_ns!=before.st_ctime_ns
    assert f.verify()['gguf_files_full_hashed']==1
    assert len(f.calls)==2


def test_same_size_mtime_tampering_detected_by_memory_ctime(fixture):
    f=fixture;f.verify();before=f.path.stat()
    time.sleep(0.02)
    f.path.write_bytes(b'X'*before.st_size);os.utime(f.path,ns=(before.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(GateClosed,match='SHA-256 drift'):f.verify()


def test_force_full_and_cleared_session_floor(fixture):
    f=fixture;f.verify();f.verify(full=True)
    contracts._SESSION_FULL_HASH_DONE.clear();f.verify()
    assert len(f.calls)==3


def test_pid_change_does_not_inherit_verified_floor(fixture,monkeypatch):
    f=fixture;f.verify();pid=os.getpid()
    monkeypatch.setattr(module.os,'getpid',lambda:pid+1000)
    assert f.verify()['gguf_files_full_hashed']==1
    assert len(f.calls)==2


def test_other_root_and_receipt_need_full_hash(fixture,tmp_path):
    f=fixture;f.verify();other=tmp_path/'other';(other/'model').mkdir(parents=True)
    (other/'model/weights').write_bytes(f.path.read_bytes())
    assert f.verify(data=other)['gguf_files_full_hashed']==1
    f.receipt.write_text('{"revision":2}')
    assert f.verify()['gguf_files_full_hashed']==1
    assert len(f.calls)==3


def test_mutation_during_hash_is_not_cached(fixture,monkeypatch):
    f=fixture
    def changing(path,label):
        result=f.measure(path,label)
        before=path.stat();os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns+1000000))
        return result
    monkeypatch.setattr(contracts,'_measure_file',changing)
    with pytest.raises(GateClosed,match='changed during'):f.verify()
    assert not module._MEMORY_IDENTITIES


def test_failed_hash_does_not_publish_partial_cache(fixture):
    f=fixture;f.files[0]['sha256']='0'*64
    with pytest.raises(GateClosed,match='SHA-256 drift'):f.verify()
    assert not module._MEMORY_IDENTITIES


def test_fork_child_rehashes_even_with_inherited_session_flag(fixture):
    import multiprocessing
    f=fixture;f.verify()
    context=multiprocessing.get_context('fork');parent,child=context.Pipe(False)
    def worker():
        try:child.send(f.verify()['gguf_files_full_hashed'])
        finally:child.close()
    process=context.Process(target=worker);process.start()
    assert parent.poll(10)
    assert parent.recv()==1
    process.join(10)
    assert process.exitcode==0


def test_size_drift_refused(fixture):
    f=fixture;f.verify();f.path.write_bytes(b'longer changed weight bytes')
    with pytest.raises(GateClosed,match='size drift'):f.verify()


def test_valid_readonly_disk_cache_with_old_device_uses_current_memory(fixture,monkeypatch):
    f=fixture
    cached={f.files[0]['path']:contracts._cheap_file_identity(f.path,'fixture')}
    cached[f.files[0]['path']]['st_dev']+=1
    monkeypatch.setattr(contracts,'_load_identity_cache',lambda *a:cached)
    (f.root/contracts.RUNTIME_IDENTITY_CACHE_FILENAME).write_text('{}')
    assert f.verify()['gguf_files_full_hashed']==1
    assert f.verify()['gguf_files_memory_cache_hits']==1
    assert len(f.calls)==1
