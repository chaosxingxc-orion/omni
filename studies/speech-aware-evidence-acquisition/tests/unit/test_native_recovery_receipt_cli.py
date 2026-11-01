"""Operational publication uses real journal custody without sending requests."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
from native_recovery_receipt import prepare
from test_native_contact_replay import interrupt_after_known_contacts
from test_llama_server_transport import _spawn_live_matching_process
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournalError


def test_published_receipt_selects_exact_new_attempt_and_rejects_overwrite(tmp_path):
    old,_=interrupt_after_known_contacts(tmp_path,2)
    data=old.world.data_root
    out=data/'work'/'recovery.json'
    result=prepare(data,old.plan,'resume-v1',old.client.arm,out)
    assert result['contacts']=={'observed':2,'unknown':0}
    assert not result['generation_started']
    document=json.loads(out.read_bytes())
    assert document['target_plan']['attempt_id']=='resume-v1'
    assert document['target_plan']['protocol_hash']==old.plan.protocol_hash
    with pytest.raises(FileExistsError):prepare(data,old.plan,'resume-v1',old.client.arm,out)


def test_unknown_is_reported_not_hidden_or_resent(tmp_path):
    old,_=interrupt_after_known_contacts(tmp_path,unknown=True)
    result=prepare(old.world.data_root,old.plan,'resume-v1',old.client.arm,
                   old.world.data_root/'work'/'recovery.json')
    assert result['contacts']=={'observed':4,'unknown':1}
    assert not result['unknown_auto_retry']


def test_open_source_and_used_target_refused(tmp_path):
    old,_=interrupt_after_known_contacts(tmp_path,1)
    data=old.world.data_root;out=data/'work'/'recovery.json'
    target=data/'runs'/old.plan.run_id/'resume-v1';target.mkdir(parents=True)
    with pytest.raises(ValueError,match='already has state'):
        prepare(data,old.plan,'resume-v1',old.client.arm,out)
    target.rmdir()
    path=old.client._adapter._attempt_handle.path
    doc=json.loads(path.read_bytes());doc['state']='OPENED';path.write_text(json.dumps(doc))
    with pytest.raises(ContactJournalError,match='aborted'):
        prepare(data,old.plan,'resume-v1',old.client.arm,out)
    assert not out.exists()


def test_output_escape_refused(tmp_path):
    old,_=interrupt_after_known_contacts(tmp_path,1)
    with pytest.raises(ValueError,match='confined'):
        prepare(old.world.data_root,old.plan,'resume-v1',old.client.arm,tmp_path/'outside.json')
