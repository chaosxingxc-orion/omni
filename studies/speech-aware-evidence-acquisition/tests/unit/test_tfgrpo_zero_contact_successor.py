from dataclasses import replace
import json
import pytest
from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay,ContactJournalError
from test_tfgrpo_runtime import setup

def test_zero_contact_aborted_prepared_attempt_has_explicit_recovery_custody(tmp_path):
    caller,client,_,transport,_,_=setup(tmp_path)
    plan=client._adapter._plan;data=client._binding.data_root
    client.close();client._adapter._attempt_handle.mark_aborted('precontact router failure')
    target=replace(plan,attempt_id='recovery-zero')
    replay=NativeContactReplay(data,plan,target,client.arm)
    assert replay.request_ids==frozenset()
    assert replay.source_binding()['actual_usage']=={'calls':0,'audio_seconds':0.0}
    replay.assert_target(data,target,client.arm)
    assert not transport.calls
    path=client._adapter._attempt_handle.path
    doc=json.loads(path.read_bytes());doc['actual']['calls']=1;path.write_text(json.dumps(doc))
    with pytest.raises(ContactJournalError):NativeContactReplay(data,plan,target,client.arm)

def test_unaborted_empty_prepared_attempt_refused(tmp_path):
    caller,client,_,_,_,_=setup(tmp_path)
    with pytest.raises(ContactJournalError,match='explicitly aborted'):
        NativeContactReplay(client._binding.data_root,client._adapter._plan,
            replace(client._adapter._plan,attempt_id='recovery-zero'),client.arm)
    client.close()
