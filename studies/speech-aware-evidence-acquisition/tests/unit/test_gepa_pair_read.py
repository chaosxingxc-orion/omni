import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/analysis'))
import gepa_pair_read as m
from test_tfrl_h1_observation_read import inputs


def fixture():
    result,items=inputs();result['per_sample']=result['per_sample'][:4];result['aggregate']['samples']=4;result['aggregate']['official']['slu_f1']=1.0
    for row in result['per_sample']:row['slot_f1_exact_tp']=1
    return {a:copy.deepcopy(result) for a in m.ARMS},items[:4]


def test_complete_pairs_keep_official_aggregates_and_never_adopt():
    results,items=fixture();r=m.summarize(results,items)
    assert r['rows']==4 and not r['candidate_accepted'] and not r['claims_stable_gain']
    assert r['proxy_scores']['child']==[1.0]*4 and not r['minibatch_strictly_improved']
    assert r['comparisons']['child_vs_parent']['strict_repairs']==0


def test_pairing_uses_current_parent_not_original_lock():
    results,items=fixture();row=results['parent']['per_sample'][0];row['scenario_action_exact']=0
    r=m.summarize(results,items)
    assert r['comparisons']['child_vs_parent']['strict_repairs']==1
    assert r['comparisons']['child_vs_lock']['strict_repairs']==0
    assert r['minibatch_strictly_improved'] and not r['candidate_accepted']


@pytest.mark.parametrize('fault',['missing','duplicate','other_arm','scorer','nan','negative','bool','intent'])
def test_unsafe_score_inputs_rejected(fault):
    results,items=fixture()
    if fault=='missing':results['child']['per_sample'].pop()
    elif fault=='duplicate':items[1]=items[0]
    elif fault=='other_arm':results['other']=results.pop('parent')
    elif fault=='scorer':results['child']['aggregate']['scorer_version']='other'
    elif fault=='intent':results['child']['per_sample'][0]['scenario_action_exact']=2
    else:results['child']['per_sample'][0]['slot_f1_exact_tp']={'nan':float('nan'),'negative':-1,'bool':True}[fault]
    with pytest.raises(ValueError):m.summarize(results,items)


def test_native_perception_text_hash_is_bound_not_response_envelope(tmp_path,monkeypatch):
    import hashlib,json
    from types import SimpleNamespace
    from test_gepa_pair_request_audit import fixture as request_fixture
    from speech_aware_evidence_acquisition.rails import gepa_prompt_material as material
    contacts,requests,_=request_fixture();original='Lock instruction'
    paths={}
    for arm in m.ARMS:
        text=original if arm=='lock' else arm+' instruction'
        path=tmp_path/'configs'/(arm+'.txt');path.parent.mkdir(exist_ok=True);path.write_text(text)
        paths[arm]={'relpath':arm+'.txt','sha256':hashlib.sha256(text.encode()).hexdigest()}
        instruction=text+'\n\nNative episode limits: '+json.dumps({'calls':4},sort_keys=True)
        requests[arm]['history'][0]['source_response_sha256']='a'*64
        control=json.loads(requests[arm]['history'][-1]['content']);control['messages'][1]['content']=instruction
        content=json.dumps(control);requests[arm]['history'][-1].update(content=content,content_sha256=hashlib.sha256(content.encode()).hexdigest())
    values={'native_episode':{'consumption_module':{}},'phase_material':{'solver':{'instruction_template':'lock.txt'}}}
    declaration={'frozen_sha256':material.pair_frozen_identity(values),'parent_template':paths['parent'],'child_template':paths['child']}
    monkeypatch.setattr(material,'validate_pair_declaration',lambda block:declaration)
    pc={'response_text_sha256':'a'*64,'response_sha256':'b'*64}
    bundle=SimpleNamespace(outputs_rows=[{'sample_id':'slurp:train:1','native_episode_receipt':{'role_visits':[
        {'stage':'perception','execution_receipt':{'final_contact':pc}},
        {'stage':'solver','execution_receipt':{'contacts':[contacts[a] for a in m.ARMS]}}]}}],
        trace_records={contacts[a]['request_id']+'/request':{'payload':requests[a]} for a in m.ARMS})
    assert m.audit_inputs(tmp_path,values,bundle,['slurp:train:1'])['audited_rows']==1
    for arm in m.ARMS:requests[arm]['history'][0]['source_response_sha256']='b'*64
    with pytest.raises(ValueError,match='exact native P'):m.audit_inputs(tmp_path,values,bundle,['slurp:train:1'])
