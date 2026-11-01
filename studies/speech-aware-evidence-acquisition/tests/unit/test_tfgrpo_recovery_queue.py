import copy
from dataclasses import asdict,replace
import json
from pathlib import Path
import pytest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay,canonical,digest
from speech_aware_evidence_acquisition.rails.tfgrpo_runtime import learning_source_plan
from test_tfgrpo_runtime import setup
import tfgrpo_recovery_queue as queue


def empty_chain(tmp_path):
    caller,client,_,_,_,_=setup(tmp_path)
    original=client._adapter._plan;data=client._binding.data_root
    client.close();client._adapter._attempt_handle.mark_aborted('zero-contact setup failure')
    first=replace(original,attempt_id='first-recovery')
    replay=NativeContactReplay(data,original,first,'prepared-context-text')
    receipt=data/'receipt.json';receipt.write_bytes(canonical({'schema':'saea-native-recovery-v1',
        'source':replay.source_binding(),'target_plan':asdict(first)}))
    run=data/'runs'/first.run_id/first.attempt_id;run.mkdir(parents=True)
    (run/'resume-declaration.json').write_bytes(canonical({'schema':'saea-resume-declaration-v1',
        'run_id':first.run_id,'attempt_id':first.attempt_id,'predecessors':[{'attempt_id':original.attempt_id,
        'mode':'native-contact-journal','pin_sha256':replay.source_binding()['attempt_sha256']}],
        'native_recovery_receipt':{'path':str(receipt),'sha256':digest(receipt.read_bytes())}}))
    state=json.loads(client._adapter._attempt_handle.path.read_bytes());state['attempt_id']=first.attempt_id
    path=data/'attempts'/first.run_id/(first.attempt_id+'.json');path.write_bytes(canonical(state))
    return data,original,first,receipt


def test_empty_ancestor_is_pinned_and_resolves_original_scientific_registration(tmp_path):
    data,original,first,receipt=empty_chain(tmp_path)
    target=replace(first,attempt_id='second-recovery')
    replay=NativeContactReplay(data,first,target,'prepared-context-text')
    assert replay.source_plan==first
    assert learning_source_plan(target,replay=replay,data_root=data)==original
    assert replay.source_binding()['empty_ancestry']['receipt']['sha256']==digest(receipt.read_bytes())
    assert replay.source_actual_usage=={'calls':0,'audio_seconds':0.0}


def test_tampered_or_spent_ancestor_is_rejected(tmp_path):
    data,original,first,receipt=empty_chain(tmp_path)
    path=data/'attempts'/original.run_id/(original.attempt_id+'.json')
    state=json.loads(path.read_bytes());state['actual']['calls']=1;path.write_bytes(canonical(state))
    with pytest.raises(ValueError):NativeContactReplay(data,first,replace(first,attempt_id='next'),'prepared-context-text')


def test_ancestor_receipt_tampering_refused(tmp_path):
    data,original,first,receipt=empty_chain(tmp_path)
    receipt.write_bytes(receipt.read_bytes()+b' ')
    with pytest.raises(ValueError,match='ancestor receipt changed'):
        NativeContactReplay(data,first,replace(first,attempt_id='next'),'prepared-context-text')


def test_unknown_fit_has_no_automatic_retry(tmp_path):
    spec={'run_id':'fit','attempt_id':'one'};root=tmp_path/'prepared/fit/one';root.mkdir(parents=True)
    (root/'contact.endpoint.json').write_text('{}')
    with pytest.raises(ValueError,match='unknown tokenizer'):queue.inspect_fit(tmp_path,spec)
    (root/'contact.request-fit.json').write_text(json.dumps({'output':{'every_request_fits':True}}))
    assert len(queue.inspect_fit(tmp_path,spec))==1


def test_live_worker_never_reconciled(tmp_path,monkeypatch):
    identity={'pid':123,'start_ticks':1,'boot_id':'boot','state':'S'}
    monkeypatch.setattr('a0_recovery_queue.process_identity',lambda _:identity)
    with pytest.raises(ValueError,match='still live'):
        queue.reconcile(tmp_path,None,'next',tmp_path/'out',identity,None,{},observe=lambda _:pytest.fail('no server probe'))


def test_quiescence_required_before_attempt_or_lease_mutation(tmp_path,monkeypatch):
    data,original,first,receipt=empty_chain(tmp_path)
    monkeypatch.setattr(queue,'require_terminal',lambda i:{'terminal':True})
    guard=data/'locks';guard.mkdir(exist_ok=True)
    lease={'run_id':first.run_id,'attempt_id':first.attempt_id,'pid':123}
    (guard/'flight.lock').write_bytes(canonical(lease))
    before=(data/'attempts'/first.run_id/(first.attempt_id+'.json')).read_bytes()
    def busy(_):raise ValueError('server busy')
    with pytest.raises(ValueError,match='server busy'):
        queue.reconcile(data,first,'next',data/'out',{'pid':123},None,{'run_id':'FIT','attempt_id':'one'},observe=busy)
    assert (guard/'flight.lock').read_bytes()==canonical(lease)
    assert (data/'attempts'/first.run_id/(first.attempt_id+'.json')).read_bytes()==before


def test_guarded_zero_contact_reconcile_preserves_lock_and_source(tmp_path,monkeypatch):
    data,original,first,receipt=empty_chain(tmp_path)
    monkeypatch.setattr(queue,'require_terminal',lambda i:{'terminal':True})
    guard=data/'locks';guard.mkdir(exist_ok=True)
    lease={'run_id':first.run_id,'attempt_id':first.attempt_id,'pid':123}
    (guard/'flight.lock').write_bytes(canonical(lease))
    out=data/'queue';out.mkdir()
    declaration=queue.reconcile(data,first,'next',out,{'pid':123},None,{'run_id':'FIT','attempt_id':'one'},observe=lambda _: {'quiet':True})
    assert not (guard/'flight.lock').exists()
    assert (out/'preserved-flight.lock.json').read_bytes()==canonical(lease)
    assert read_json(Path(declaration['path']))['source']['empty_ancestry']
    assert not (data/'attempts'/first.run_id/'next.json').exists()


def read_json(path):return json.loads(path.read_bytes())


def test_nonzero_immediate_source_retains_every_contact_after_empty_ancestor(tmp_path,monkeypatch):
    from test_tfgrpo_recovery import interrupted
    runtime,transport,old_replay=interrupted(tmp_path,monkeypatch)
    data=runtime._data_root;source=old_replay.source_plan
    ancestor=replace(source,attempt_id='zero-contact-ancestor')
    source_path=data/'attempts'/source.run_id/(source.attempt_id+'.json')
    state=read_json(source_path);state['attempt_id']=ancestor.attempt_id;state['actual']={'calls':0,'audio_seconds':0.0}
    (source_path.parent/(ancestor.attempt_id+'.json')).write_bytes(canonical(state))
    zero=NativeContactReplay(data,ancestor,source,'prepared-context-text')
    receipt=data/'zero-ancestor.json';receipt.write_bytes(canonical({'schema':'saea-native-recovery-v1',
        'source':zero.source_binding(),'target_plan':asdict(source)}))
    path=data/'runs'/source.run_id/source.attempt_id/'resume-declaration.json'
    path.write_bytes(canonical({'schema':'saea-resume-declaration-v1','run_id':source.run_id,'attempt_id':source.attempt_id,
        'predecessors':[{'attempt_id':ancestor.attempt_id,'mode':'native-contact-journal','pin_sha256':zero.source_binding()['attempt_sha256']}],
        'native_recovery_receipt':{'path':str(receipt),'sha256':digest(receipt.read_bytes())}}))
    replay=NativeContactReplay(data,source,runtime.plan,'prepared-context-text')
    assert replay.source_actual_usage['calls']==3 and len(replay.request_ids)==3
    assert replay.scientific_source_plan==ancestor
    for rid in replay.request_ids:
        assert replay.receipt(rid)==old_replay.receipt(rid)
    assert not transport.calls
    # Changing an empty ancestor after indexing must invalidate response reuse.
    ap=source_path.parent/(ancestor.attempt_id+'.json');ap.write_bytes(ap.read_bytes()+b' ')
    with pytest.raises(ValueError,match='ancestor changed'):replay.receipt(next(iter(replay.request_ids)))
