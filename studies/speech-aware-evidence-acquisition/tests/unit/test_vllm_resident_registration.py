"""External deployment registration proves every process and recovers partial publication."""
import hashlib
import json
from pathlib import Path

import pytest

from _vllm_world import VllmWorld, PROTOCOL_ARGS
from speech_aware_evidence_acquisition.core.session import SessionError
from speech_aware_evidence_acquisition.core.session import vllm_resident as vr
from speech_aware_evidence_acquisition.core.session.serving import find_resident_session
from speech_aware_evidence_acquisition.core.session.vllm_identity import attach_vllm_session


@pytest.fixture
def registered_world(tmp_path):
    w=VllmWorld(tmp_path)
    w.spawn('tp2-a',pid=4200001)
    w.spawn('tp2-b',pid=4200002)
    receipt=tmp_path/'runtime.json'
    receipt.write_text(json.dumps(w.receipt()))
    fragment={'transport':'vllm','server_args':PROTOCOL_ARGS,
              'model_materialization_root':str(w.materialization_root),
              'runtime_receipt':'receipts/runtime.json','decoding_params':{'max_tokens':4096}}
    observations={name:{'pid':pid,'started_at':'2026-10-30T00:00:00Z',
                         'startup_stdout_sha256':hashlib.sha256(b'').hexdigest()}
                  for name,pid in [('tp2-a',4200001),('tp2-b',4200002)]}
    return w,dict(data_root=w.materialization_root,fragment=fragment,model_fragment='m',receipt_path=receipt,
                  instances=observations,session_id='deployment-v3',proc_root=w.proc_root)


def test_registers_one_coordinator_and_attachable_members(registered_world):
    w,kw=registered_world
    paths=vr.register_deployment(**kw)
    assert len(paths)==6
    resident=find_resident_session(w.materialization_root)
    assert resident[1]['session_id']=='deployment-v3'
    coordinator = attach_vllm_session(resident[0]/'session-receipt.json',
                                     runtime_receipt=w.receipt(),proc_root=w.proc_root)
    assert coordinator.session_id == 'deployment-v3'
    assert coordinator.pid == kw['instances']['tp2-a']['pid']
    for name in kw['instances']:
        attached=attach_vllm_session(w.materialization_root/'serving'/name/'session-receipt.json',
                                    runtime_receipt=w.receipt(),proc_root=w.proc_root)
        assert attached.pid==kw['instances'][name]['pid']
    before={p:p.read_bytes() for p in paths}
    assert vr.register_deployment(**kw)==paths
    assert all(p.read_bytes()==data for p,data in before.items())


def test_partial_publication_resumes_without_exposing_coordinator(registered_world,monkeypatch):
    w,kw=registered_world
    publish=vr.publish_new
    count=0
    def interrupt(path,data):
        nonlocal count
        count+=1
        if count==3:
            raise OSError('interrupted publication')
        publish(path,data)
    monkeypatch.setattr(vr,'publish_new',interrupt)
    with pytest.raises(OSError,match='interrupted'):
        vr.register_deployment(**kw)
    assert find_resident_session(w.materialization_root) is None
    monkeypatch.setattr(vr,'publish_new',publish)
    vr.register_deployment(**kw)
    assert find_resident_session(w.materialization_root)[1]['session_id']=='deployment-v3'


def test_one_dead_member_refuses_entire_registration(registered_world):
    w,kw=registered_world
    (w.proc_root/'4200002/stat').unlink()
    with pytest.raises(SessionError,match='not a live process'):
        vr.register_deployment(**kw)
    assert not (w.materialization_root/'serving').exists()


def test_restarted_member_cannot_overwrite_old_receipt(registered_world):
    w,kw=registered_world
    vr.register_deployment(**kw)
    w.spawn('tp2-a',pid=4200003)
    kw['instances']['tp2-a']['pid']=4200003
    with pytest.raises(SessionError,match='conflicting resident'):
        vr.register_deployment(**kw)


def test_fragment_protocol_drift_refused(registered_world):
    w,kw=registered_world
    kw['fragment']['server_args']=[*PROTOCOL_ARGS,'--enforce-eager']
    with pytest.raises(Exception,match='protocol'):
        vr.register_deployment(**kw)
    assert not (w.materialization_root/'serving').exists()


def test_another_coordinator_cannot_claim_same_members(registered_world):
    w, kw = registered_world
    vr.register_deployment(**kw)
    kw['session_id'] = 'other-coordinator'
    with pytest.raises(SessionError, match='another coordinator'):
        vr.register_deployment(**kw)
    assert not (w.materialization_root/'serving'/'other-coordinator').exists()


def test_runtime_required_identity_uses_vllm_protocol(registered_world, monkeypatch):
    from speech_aware_evidence_acquisition.core.flight.runtime import FlightRuntime
    from speech_aware_evidence_acquisition.core import config
    w, kw = registered_world
    repo = w.root / 'repo'
    receipt = repo / 'docs' / 'receipts' / 'runtime.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(kw['receipt_path'].read_bytes())
    runtime = object.__new__(FlightRuntime)
    runtime._repo_root, runtime._model = repo, 'm'
    runtime._host, runtime._port = '127.0.0.1', w.receipt()['deployment']['instances'][0]['port']
    monkeypatch.setattr(config, 'read_fragment', lambda *a: kw['fragment'])
    result = runtime._required_serving_identity(values=kw['fragment'], runtime_receipt=w.receipt(),
        geometry={}, extra_argv=[], extra_env={}, configured_model_root=str(w.materialization_root))
    assert result == vr.required_identity(fragment=kw['fragment'], model_fragment='m',
        runtime_receipt=w.receipt(), receipt_path=receipt, host=runtime._host, port=runtime._port)


def test_identity_hash_does_not_inherit_container_data_root(registered_world, monkeypatch):
    w, kw = registered_world
    endpoint = w.receipt()['deployment']['instances'][0]
    def identity():
        return vr.required_identity(fragment=kw['fragment'], model_fragment='m',
            runtime_receipt=w.receipt(), receipt_path=kw['receipt_path'],
            host=endpoint['host'], port=endpoint['port'])
    monkeypatch.setenv('SAEA_DATA_ROOT', '/stale-image-default')
    monkeypatch.delenv('SPEECHRL_DATA_DIR', raising=False)
    stale = identity()
    monkeypatch.setenv('SAEA_DATA_ROOT', str(w.materialization_root))
    assert identity() == stale
