import copy
import json
import multiprocessing
import os
import pytest
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal, ContactJournalError, digest, canonical
from test_frozen_core_adapter import _plan, _payload, _binding, _OpenGate


def payload(request_id='req-1'):
    return _payload(request_id=request_id,history=[{'role':'control','contact_id':'solver:row:arm-b'}])


def test_reopened_journal_preserves_exact_bytes_and_refuses_duplicate(tmp_path):
    j=ContactJournal(tmp_path,_plan(),'bare-core');p=payload()
    assert j.inspect('req-1')['state']=='absent'
    j.intent(p)
    assert j.inspect('req-1')['state']=='unknown'
    raw=b'\xffmalformed-but-observed'
    j.response('req-1',digest(canonical(p)),raw,.25)
    again=ContactJournal(tmp_path,_plan(),'bare-core')
    assert again.inspect('req-1')['response_bytes']==raw
    with pytest.raises(ContactJournalError):again.intent(p)
    with pytest.raises(ContactJournalError):again.response('req-1',digest(canonical(p)),raw,.25)


@pytest.mark.parametrize('fault',['request','response','identity','intent_hash','latency','orphan'])
def test_tampering_never_becomes_a_reusable_response(tmp_path,fault):
    j=ContactJournal(tmp_path,_plan(),'bare-core');p=payload();j.intent(p)
    j.response('req-1',digest(canonical(p)),b'{}',1)
    path=j._path('req-1','intent' if fault=='request' else 'response')
    doc=json.loads(path.read_bytes())
    if fault=='request':doc['payload']['task_instruction']='changed'
    elif fault=='response':doc['response_b64']='YQ=='
    elif fault=='identity':doc['identity']['protocol_hash']='c'*64
    elif fault=='intent_hash':doc['intent_sha256']='0'*64
    elif fault=='latency':doc['latency_seconds']=-1
    else:j._path('req-1','intent').unlink()
    path.write_bytes(canonical(doc))
    with pytest.raises(ContactJournalError):j.inspect('req-1')


def test_changed_protocol_cannot_read_previous_contact(tmp_path):
    j=ContactJournal(tmp_path,_plan(),'bare-core');j.intent(payload())
    with pytest.raises(ContactJournalError):ContactJournal(tmp_path,_plan(protocol_hash='c'*64),'bare-core').inspect('req-1')


def crash_writer(root,phase):
    j=ContactJournal(root,_plan(),'bare-core');p=payload();j.intent(p)
    if phase=='response':j.response('req-1',digest(canonical(p)),b'known response',1)
    os._exit(73)


@pytest.mark.parametrize('phase,state',[('intent','unknown'),('response','observed')])
def test_abrupt_process_exit_keeps_custody(tmp_path,phase,state):
    process=multiprocessing.get_context('fork').Process(target=crash_writer,args=(tmp_path,phase))
    process.start();process.join(10)
    assert not process.is_alive() and process.exitcode==73
    assert ContactJournal(tmp_path,_plan(),'bare-core').inspect('req-1')['state']==state


def test_fallback_same_contact_keeps_separate_observed_requests(tmp_path):
    j=ContactJournal(tmp_path,_plan(),'bare-core')
    for rid in ('req-1','req-2'):
        p=payload(rid);j.intent(p);j.response(rid,digest(canonical(p)),rid.encode(),1)
    assert j.inspect('req-1')['response_bytes']!=j.inspect('req-2')['response_bytes']


def test_directory_and_records_are_fsynced(tmp_path,monkeypatch):
    import speech_aware_evidence_acquisition.core.contact_journal as module
    calls=[]
    original=module.os.fsync
    def sync(fd):
        calls.append(fd)
        original(fd)
    monkeypatch.setattr(module.os,'fsync',sync)
    j=ContactJournal(tmp_path,_plan(),'bare-core');p=payload();j.intent(p)
    # Three new directories, intent file, and containing directory.
    assert len(calls)>=5
    before=len(calls)
    j.response('req-1',digest(canonical(p)),b'known response',1)
    assert len(calls)>=before+2


def test_concurrent_intent_writers_publish_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    j=ContactJournal(tmp_path,_plan(),'bare-core')
    def write(_):
        try:j.intent(payload());return True
        except ContactJournalError:return False
    with ThreadPoolExecutor(max_workers=8) as executor:
        results=list(executor.map(write,range(8)))
    assert sum(results)==1 and j.inspect('req-1')['state']=='unknown'


def test_aborted_zero_spend_native_startup_can_resume_without_inventing_contacts(tmp_path):
    from dataclasses import replace
    from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
    source=_plan();target=replace(source,attempt_id='native-startup-recovery')
    ap=tmp_path/'attempts'/source.run_id/(source.attempt_id+'.json');ap.parent.mkdir(parents=True)
    doc={'schema':'saea-attempt-v1','run_id':source.run_id,'attempt_id':source.attempt_id,
         'carrier_lock_key':source.carrier_lock_key,'state':'ABORTED','actual':{'calls':0,'audio_seconds':0.0}}
    ap.write_bytes(canonical(doc))
    replay=NativeContactReplay(tmp_path,source,target,'raw-speech-agent-native-tools')
    assert not replay.request_ids and replay.source_actual_usage=={'calls':0,'audio_seconds':0.0}
    for fault in ('state','spend','orphan'):
        changed=copy.deepcopy(doc)
        if fault=='state':changed['state']='OPENED'
        elif fault=='spend':changed['actual']['calls']=1
        else:
            journal=ContactJournal(tmp_path,source,'raw-speech-agent-native-tools')
            journal.root.mkdir(parents=True,exist_ok=True)
            journal._path('orphan','response').write_bytes(b'{}')
        ap.write_bytes(canonical(changed))
        with pytest.raises(ContactJournalError):NativeContactReplay(tmp_path,source,target,'raw-speech-agent-native-tools')
