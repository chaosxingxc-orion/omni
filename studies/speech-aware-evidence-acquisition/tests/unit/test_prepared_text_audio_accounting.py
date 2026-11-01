"""Prepared reflection preserves speech custody but contacts no audio."""
from dataclasses import replace
import json
import pytest
from speech_aware_evidence_acquisition.core.model import FrozenCoreError
from speech_aware_evidence_acquisition.core.attempts import AttemptStore, AttemptError
from speech_aware_evidence_acquisition.core.attempts.constants import CONTACT_KIND_PREPARED_TEXT
from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
from test_tfgrpo_runtime import setup,request
from test_rails_frozen_core_client import _World,_client,_plan,_zero_supply_messages


def test_actual_prepared_contact_zero_audio_preserves_bound_identity(tmp_path):
    caller,client,_,transport,_,_=setup(tmp_path)
    # Real production plan has a positive 1s schema cap; its audio anchor is 2s.
    client._adapter._plan=replace(client._adapter._plan,planned_speech_audio_seconds=1)
    observed=caller(request())
    assert observed['usage']['calls']==1
    assert transport.calls[0]['audio_seconds']==2.0
    assert transport.calls[0]['media_relpath'].endswith('s1.wav')
    actual=json.loads(client._adapter._attempt_handle.path.read_text())
    assert actual['contact_kind']==CONTACT_KIND_PREPARED_TEXT
    assert actual['actual']=={'calls':1,'audio_seconds':0.0}
    assert client._adapter.cost_summary()['audio_seconds_used']==0
    trace=json.loads(next(caller.root.glob('*.trace.json')).read_text())
    assert [record[2]['audio_seconds'] for record in trace if record[0]=='COST']==[0.0]
    client.close()


def test_prepared_audio_identity_tampering_still_refused(tmp_path):
    caller,client,_,transport,_,_=setup(tmp_path)
    _,prepared=caller.prepare(request())
    payload=prepared.as_payload();payload['audio_seconds']=0
    with pytest.raises(FrozenCoreError):client._adapter.request(payload)
    assert not transport.calls
    client.close()


def test_raw_speech_metering_and_budget_unchanged(tmp_path):
    world=_World(tmp_path)
    client,_,transport=_client(world,tmp_path,plan=_plan(planned_speech_audio_seconds=2))
    client.complete(_zero_supply_messages(world.media_relpath('s1')))
    assert client._adapter.cost_summary()['audio_seconds_used']==2
    with pytest.raises(FrozenCoreError,match='audio budget exhausted'):
        client.complete(_zero_supply_messages(world.media_relpath('s1')))
    assert len(transport.calls)==1
    client.close()


def test_prepared_attempt_rejects_nonzero_audio_and_speech_rejects_zero(tmp_path):
    store=AttemptStore(tmp_path/'attempts')
    prepared=store.open_attempt('r1','a1',carrier_lock_key='fake',reserved_calls=2,
        reserved_audio_seconds=1,contact_kind=CONTACT_KIND_PREPARED_TEXT)
    assert prepared.record_usage(calls=1,audio_seconds=0)==(1,0)
    with pytest.raises(AttemptError):prepared.record_usage(calls=1,audio_seconds=2)
    speech=store.open_attempt('r2','a1',carrier_lock_key='fake',reserved_calls=2,reserved_audio_seconds=1)
    with pytest.raises(AttemptError):speech.record_usage(calls=1,audio_seconds=0)


def test_prepared_journal_recovery_retains_zero_audio(tmp_path):
    from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal,ContactJournalError
    caller,client,_,_,_,_=setup(tmp_path)
    plan=client._adapter._plan
    data=client._binding.data_root
    client._adapter._contact_journal=ContactJournal(data,plan,client.arm)
    caller(request())
    client.close()
    client._adapter._attempt_handle.mark_aborted('test interruption')
    target=replace(plan,attempt_id='recovery')
    replay=NativeContactReplay(data,plan,target,client.arm)
    assert replay.source_actual_usage['audio_seconds']==0
    records={}
    source=next(iter(replay._records.values()))
    rid=source['intent']['request_id'];payload=source['intent']['payload']
    raw=json.loads(source['response_bytes']);response=source['response']
    from speech_aware_evidence_acquisition.core.contact_journal import digest
    records[rid+'/request']={'arm':client.arm,'payload':payload,'payload_sha256':source['intent']['request_sha256']}
    records[rid+'/response']={'text':raw['text'],'usage':raw['usage'],'response_sha256':response['response_sha256'],
                            'response_text_sha256':digest(raw['text'].encode())}
    cost={'calls':1,'audio_seconds':0.,'latency_seconds':response['latency_seconds'],
          'prompt_tokens':raw['usage']['prompt_tokens'],'completion_tokens':raw['usage']['completion_tokens']}
    records[rid+'/cost']=cost;records[rid+'/contact-replay']=replay.receipt(rid)
    from speech_aware_evidence_acquisition.core.bundle.native_replay import verify_native_replay
    manifest={'arm':client.arm,'attempt':{'attempt_id':target.attempt_id,
        'state_path':str(data/'attempts'/plan.run_id/(target.attempt_id+'.json'))},
        'cost':{'native_replay_source':replay.source_binding(),'replayed_predecessor_cost':cost}}
    verify_native_replay(manifest,records,target)
    path=client._adapter._attempt_handle.path
    doc=json.loads(path.read_text());doc.pop('contact_kind');path.write_text(json.dumps(doc))
    with pytest.raises(ContactJournalError,match='contact kind'):
        NativeContactReplay(data,plan,target,client.arm)
