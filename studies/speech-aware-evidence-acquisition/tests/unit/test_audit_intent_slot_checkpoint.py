import importlib.util,json
from pathlib import Path
import pytest
from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter,RowCheckpointError
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('audit_checkpoint',ROOT/'scripts/analysis/audit_intent_slot_checkpoint.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def writer(tmp_path):return RowCheckpointWriter(tmp_path,'RUN','attempt',identity={},fsync=False)
def refusal():return {'native_episode_receipt':{'role_visits':[],'terminal_disposition':{'reason':'seal_refusal'}},'text':''}
def test_refusal_then_complete_and_only_refusal(tmp_path,monkeypatch):
    w=writer(tmp_path);w.append_row(sample_id='s1',complete=True,output_rows=[refusal()],trace_lines=[])
    assert m.audit(w.path)['status']=='waiting'
    visits=[{'stage':s,'execution_receipt':{'final_contact':{'request_id':s,'request_sha256':'a','response_text_sha256':__import__('hashlib').sha256(b'answer').hexdigest()}}} for s in ('perception','slot','solver')]
    trace=[]
    for visit in visits:
        source=visit['execution_receipt']['final_contact'];rid=source['request_id']
        trace.extend([json.dumps({'record_id':rid+'/request','channel':'MODEL_REQUEST','payload':{'payload_sha256':'a','payload':{'sample_id':'s2'}}}),json.dumps({'record_id':rid+'/response','channel':'MODEL_RESPONSE','payload':{'response_text_sha256':source['response_text_sha256'],'text':'answer'}})])
    w.append_row(sample_id='s2',complete=True,output_rows=[{'native_episode_receipt':{'role_visits':visits,'terminal_disposition':None},'text':'answer'}],trace_lines=trace);w.close()
    result=m.audit(w.path);assert result['status']=='complete' and result['classifications']=={'refusal':1,'complete':1} and result['first_complete']['raw_equals_delivered_bytes']

def test_empty_torn_and_malformed(tmp_path):
    w=writer(tmp_path);w.close();assert m.audit(w.path)['status']=='waiting'
    with w.path.open('ab') as f:f.write(b'{"partial"')
    assert m.audit(w.path)['torn_tail_bytes']>0
    with w.path.open('ab') as f:f.write(b'\n')
    with pytest.raises(RowCheckpointError):m.audit(w.path)

def test_malformed_receipt_refused(tmp_path):
    w=writer(tmp_path);w.append_row(sample_id='s1',complete=True,output_rows=[{'native_episode_receipt':{'role_visits':'bad'}}],trace_lines=[]);w.close()
    with pytest.raises(ValueError,match='malformed'):m.audit(w.path)
