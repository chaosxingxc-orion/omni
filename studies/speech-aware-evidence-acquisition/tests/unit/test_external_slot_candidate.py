"""External knowledge construction preserves source custody and unrelated values."""
import copy
from pathlib import Path
import sys
import pytest

STUDY=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(STUDY/'scripts'/'analysis'))
from external_slot_candidate import compile_candidate, verify_response
from speech_aware_evidence_acquisition.rails.errors import RailsError
from test_semantic_knowledge_edits import rows, proposal
from speech_aware_evidence_acquisition.rails.semantic_knowledge_edits import slot_value_edit_contract
from speech_aware_evidence_acquisition.rails.semantic_knowledge_build import AuthoredSemanticKnowledgeSpec, build_slot_contract_snapshot


def fixture():
    base=rows()
    other=copy.deepcopy(base[0]);other['slot_id']='time'
    original=build_slot_contract_snapshot(AuthoredSemanticKnowledgeSpec('fixture','2'*64,('slurp:train:base',)),rows=base+[other]).frozen_json()
    contract=slot_value_edit_contract(base,slot_ids=['date'],fields=['extraction_cues'],evidence_groups={'slurp:train:new':['slurp:train:new:audio-1']},max_edits=1,max_field_chars=300)
    p=proposal(base);p['edits'][0]['evidence_group_ids']=['slurp:train:new']
    dynamic={'base_rows':base,'edit_contract':contract,'records':[{'source_id':'slurp:train:new'}],'pool':{'file_sha256':'3'*64}}
    return original,base,contract,dynamic,{'slot_id':'date','proposal':p},{'response_sha256':'4'*64},[]


def test_preserves_other_cards_and_routing_but_adds_actual_ancestry():
    args=fixture();before=copy.deepcopy(args)
    result=compile_candidate(*args)
    assert args==before
    assert result['admission']=='candidate-only' and not result['promotion_allowed']
    original=args[0];after=result['snapshot']
    assert original['values'][1]==after['values'][1]
    assert original['facets'][1]==after['facets'][1]
    assert original['facets'][0]['representation']==after['facets'][0]['representation']
    assert 'slurp:train:new:audio-1' in after['source_sample_ids']
    assert after['snapshot_sha256']!=original['snapshot_sha256']


@pytest.mark.parametrize('kind',['different-evidence','different-base','wrong-slot','forbidden','routing'])
def test_rejects_custody_and_isolation_violations(kind):
    args=list(fixture())
    if kind=='different-evidence':args[3]['records']=[]
    elif kind=='different-base':args[1]=copy.deepcopy(args[1]);args[1][0]['missing_behavior']='Different'
    elif kind=='wrong-slot':args[4]['slot_id']='time'
    elif kind=='forbidden':args[-1]=['slurp:train:new:audio-1']
    else:args[4]['proposal']['edits'][0]['field']='intent_bindings'
    with pytest.raises((ValueError, RailsError)):compile_candidate(*args)


def teacher_fixture(tmp_path):
    import hashlib,json
    import configured_external_learner as learner
    def write(name, data):
        p=tmp_path/name;p.write_text(json.dumps(data));return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    dynamic={'records':[]}
    messages=[{'role':'user','content':json.dumps(dynamic)}]
    pin=write('messages.json',messages)
    prompt=tmp_path/'template.md';prompt.write_text('Analyze construction evidence.')
    cfg={'output_root':str(tmp_path),'model':'teacher','messages':pin,
         'research_prompt':{'path':str(prompt),'sha256':hashlib.sha256(prompt.read_bytes()).hexdigest()},
         'max_tokens':131072,'stream':True}
    cfg_pin=write('config.json',cfg)
    request={'model':'teacher','messages':[{'role':'system','content':prompt.read_text()+learner.analysis_budget_prompt(cfg)},*messages],'max_tokens':131072,'stream':True,'stream_options':{'include_usage':True}}
    write('request.json',request);write('intent.json',{'config':cfg,'messages_sha256':pin['sha256']})
    response={'id':'r1','model':'teacher','usage':{'total_tokens':7},'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'{"candidates":[]}', 'reasoning_content':''}}]}
    event={'id':'r1','model':'teacher','usage':response['usage'],'choices':[{'index':0,'finish_reason':'stop','delta':{'content':'{"candidates":[]}'}}]}
    wire=('data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n').encode();(tmp_path/'response.sse').write_bytes(wire)
    write('response.json',response)
    write('result.json',{'state':'response_received','http_status':200,'response_sha256':hashlib.sha256(wire).hexdigest(),'usage':response['usage'],'model':'teacher'})
    return cfg_pin,dynamic


def test_actual_sse_replayed_before_candidate_acceptance(tmp_path):
    cfg,dynamic=teacher_fixture(tmp_path)
    proposals,custody=verify_response(cfg,dynamic)
    assert proposals=={'candidates':[]}
    assert custody['usage']=={'total_tokens':7}


@pytest.mark.parametrize('file',['request.json','response.json','response.sse','messages.json'])
def test_tampered_wire_request_or_reconstruction_is_rejected(tmp_path,file):
    import json
    cfg,dynamic=teacher_fixture(tmp_path)
    path=tmp_path/file
    if file=='response.sse':path.write_text(path.read_text().replace('[DONE]','BROKEN'))
    else:
        value=json.loads(path.read_text())
        if file=='messages.json':value[0]['content']='{}'
        else:value['unapproved']='changed'
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):verify_response(cfg,dynamic)


def test_inherited_overlap_reported_without_erasing_old_source():
    args=list(fixture());args[-1]=["slurp:train:base"]
    result=compile_candidate(*args)
    assert result["isolation"]["inherited_snapshot_forbidden_overlap"]==["slurp:train:base"]
    assert result["isolation"]["new_optimizer_forbidden_overlap"]==[]


def test_other_recording_of_same_forbidden_utterance_rejected():
    args=list(fixture());args[-1]=["slurp:train:new:audio-2"]
    with pytest.raises(ValueError):compile_candidate(*args)
