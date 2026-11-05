import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter, RowCheckpointError
from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import canonical, digest, slot_results
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('audit_checkpoint',ROOT/'scripts/analysis/audit_intent_slot_checkpoint.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def writer(tmp_path):return RowCheckpointWriter(tmp_path,'RUN','attempt',identity={},fsync=False)
def refusal(reason='seal_refusal'):return {'native_episode_receipt':{'role_visits':[],'terminal_disposition':{'reason':reason}},'text':''}

def complete():
    texts=[canonical({'candidates':[{'intent_id':'general_joke'}]}), canonical({'candidates':[{'candidate_id':'general_joke','entities':[],'evidence_refs':[],'unresolved_fields':[]}]}),canonical({'intent':'general_joke','entities':[]})]
    trace=[];visits=[]
    for stage,text in zip(('perception','slot','solver'),texts):
        payload={'sample_id':'s2','request_id':stage,'messages':[]}
        source={'request_id':stage,'request_sha256':digest(payload),'response_sha256':hashlib.sha256(('wire:'+text).encode()).hexdigest(),'response_text_sha256':hashlib.sha256(text.encode()).hexdigest()}
        visits.append({'stage':stage,'execution_receipt':{'final_contact':source}})
        trace.extend([{'record_id':stage+'/request','channel':'MODEL_REQUEST','payload':{'payload_sha256':digest(payload),'payload':payload}}, {'record_id':stage+'/response','channel':'MODEL_RESPONSE','payload':{'response_sha256':source['response_sha256'],'response_text_sha256':source['response_text_sha256'],'text':text}}])
    block={'schema':'saea-intent-slot-pipeline-v1','slot_mode':'batched','candidate_ids':['general_joke'],'perception_source':visits[0]['execution_receipt']['final_contact'],'slot_sources':[{'candidate_ids':['general_joke'],'source':visits[1]['execution_receipt']['final_contact'],'result_sha256':digest(slot_results(texts[1],['general_joke']))}], 'selected_candidate_id':'general_joke','decision_source':visits[2]['execution_receipt']['final_contact']}
    return {'native_episode_receipt':{'role_visits':visits,'terminal_disposition':None,'intent_slot_pipeline':block},'text':texts[-1]},trace

def append(w,out,trace):w.append_row(sample_id='s2',complete=True,output_rows=[out],trace_lines=[json.dumps(t) for t in trace])

def test_refusal_then_complete_and_only_refusal(tmp_path):
    w=writer(tmp_path);w.append_row(sample_id='s1',complete=True,output_rows=[refusal()],trace_lines=[])
    assert m.audit(w.path)['status']=='waiting'
    append(w,*complete());w.close();result=m.audit(w.path)
    assert result['status']=='complete' and result['classifications']=={'refusal':1,'complete':1}
    assert result['first_complete']['raw_equals_delivered_bytes']

@pytest.mark.parametrize('reason,kind',[('finite_completion','finite_completion'),('seal_refusal','refusal'),('other','terminal_other')])
def test_dispositions_keep_reason_classification(tmp_path,reason,kind):
    w=writer(tmp_path);w.append_row(sample_id='s1',complete=True,output_rows=[refusal(reason)],trace_lines=[]);w.close()
    assert m.audit(w.path)['classifications']=={kind:1}

def test_unrendered_excluded_from_terminal_denominator(tmp_path):
    w=writer(tmp_path);w.append_row(sample_id='s1',complete=False,reason='interrupted',output_rows=[],trace_lines=[]);w.close()
    result=m.audit(w.path);assert result['terminal_rows']==0 and result['checkpoint_rows']==1

@pytest.mark.parametrize('fault',['payload','pipeline','candidate','response','sample'])
def test_actual_source_payload_and_pipeline_tampering_refused(tmp_path,fault):
    out,trace=complete()
    if fault=='payload':trace[0]['payload']['payload']['messages']=[{'role':'user','content':'tampered'}]
    elif fault=='pipeline':out['native_episode_receipt'].pop('intent_slot_pipeline')
    elif fault=='candidate':out['native_episode_receipt']['intent_slot_pipeline']['candidate_ids']=['other']
    elif fault=='sample':
        trace[0]['payload']['payload']['sample_id']='other'
        sha=digest(trace[0]['payload']['payload']);trace[0]['payload']['payload_sha256']=sha
        out['native_episode_receipt']['role_visits'][0]['execution_receipt']['final_contact']['request_sha256']=sha
    else:trace[-1]['payload']['text']='tampered'
    w=writer(tmp_path);append(w,out,trace);w.close()
    with pytest.raises(ValueError):m.audit(w.path)

def test_empty_torn_and_malformed(tmp_path):
    blank=tmp_path/'blank';blank.write_bytes(b'');assert m.audit(blank)['status']=='waiting'
    w=writer(tmp_path);w.close();assert m.audit(w.path)['status']=='waiting'
    with w.path.open('ab') as f:f.write(b'{"partial"')
    assert m.audit(w.path)['torn_tail_bytes']>0
    with w.path.open('ab') as f:f.write(b'\n')
    with pytest.raises(RowCheckpointError):m.audit(w.path)

def test_malformed_receipt_refused(tmp_path):
    w=writer(tmp_path);w.append_row(sample_id='s1',complete=True,output_rows=[{'native_episode_receipt':{'role_visits':'bad'}}],trace_lines=[]);w.close()
    with pytest.raises(ValueError,match='malformed'):m.audit(w.path)
