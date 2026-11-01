import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
from tfgrpo_seed_groups import group_from_row, text_projection, sha
from speech_aware_evidence_acquisition.rails.hashing import content_hash
from test_tfrl_uncertain_credit import score


def fixture():
    sid='slurp:train:000001:real'; contacts=[]; trace={}; scores={}
    for i,arm in enumerate(('s1','s2','s3','s4','aa','lock')):
        rid='r'+str(i); answer='{"intent":"alarm_set"}'
        payload={'sample_id':sid,'task_instruction':'Extract intent and slots.',
                 'history':[{'role':'perception','content':'set an alarm'},{'role':'control','content':'solve'}],
                 'supplied_evidence':[{'evidence_id':'k','content':'Alarm setting'}],
                 'raw_speech_query':{'base64':'fakebytes'},'decoding_params':{'seed':i}}
        reqsha=content_hash(payload); textsha=hashlib.sha256(answer.encode()).hexdigest()
        trace[rid+'/request']={'payload':payload,'payload_sha256':reqsha}
        trace[rid+'/response']={'text':answer,'response_text_sha256':textsha}
        contacts.append({'request_id':rid,'request_sha256':reqsha,'response_text_sha256':textsha})
        scores[arm]={sid:score(intent=int(i==0))}
    row={'sample_id':sid,'native_episode_receipt':{'role_visits':[{'stage':'solver','execution_receipt':{'contacts':contacts}}]}}
    return row,trace,scores


def build(row,trace,scores):
    return group_from_row(row,trace,scores,{'scenario':'alarm','action':'set','entities':[]},
                          model_id='model',policy_sha256='a'*64,experience_sha256='b'*64,namespace='source')


def test_exact_projection_reward_and_custody():
    row,trace,scores=fixture();before=copy.deepcopy(trace)
    g,pins=build(row,trace,scores)
    assert [r['reward'] for r in g['rollouts']]==[1,0,0,0]
    assert len(pins)==4 and len({r['trajectory_id'] for r in g['rollouts']})==4
    assert 'fakebytes' not in json.dumps(g)
    assert 'Alarm setting' in g['rollouts'][0]['trajectory']
    assert json.loads(g['answer'])['action']=='set'
    assert trace==before
    projection=json.loads(g['rollouts'][0]['trajectory'])['request']['raw_speech_query']
    assert projection['original_sha256']==sha({'base64':'fakebytes'})


@pytest.mark.parametrize('mutation',['request','response','contact','missing'])
def test_corrupt_source_refused(mutation):
    row,trace,scores=fixture()
    if mutation=='request':trace['r0/request']['payload']['task_instruction']='changed'
    if mutation=='response':trace['r0/response']['text']='changed'
    if mutation=='contact':row['native_episode_receipt']['role_visits'][0]['execution_receipt']['contacts'][0]['request_sha256']='x'
    if mutation=='missing':del trace['r0/response']
    with pytest.raises((ValueError,KeyError)):build(row,trace,scores)


def test_tied_group_not_exported_and_namespace_disambiguates():
    row,trace,scores=fixture();g,_=build(row,trace,scores)
    other,_=group_from_row(row,trace,scores,{},model_id='model',policy_sha256='a'*64,experience_sha256='b'*64,namespace='different')
    assert g['rollouts'][0]['trajectory_id']!=other['rollouts'][0]['trajectory_id']
    for s in scores.values():s[row['sample_id']]=score()
    with pytest.raises(ValueError,match='mixed'):build(row,trace,scores)


def test_dense_export_preserves_actual_scored_answer_and_partial_reward():
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import DENSE_REWARD
    import hashlib,json
    row,trace,scores=fixture();sid=row['sample_id']
    for i,table in enumerate(scores.values()):
        table[sid]={**table[sid],'scenario_action_exact':i%2,'slot_f1_exact_tp':0,'slot_f1_exact_fp':1,'slot_f1_exact_fn':1}
    answers={arm:{sid:'{"scenario":"normalized"}'} for arm in scores}
    kwargs=dict(model_id='model',policy_sha256='a'*64,experience_sha256='b'*64,namespace='source',reward_mode=DENSE_REWARD)
    with pytest.raises(ValueError,match='exact normalized scored answer'):
        group_from_row(row,trace,scores,{},**kwargs)
    g,pins=group_from_row(row,trace,scores,{},scored_answers=answers,**kwargs)
    assert {r['reward'] for r in g['rollouts']}=={0,.5}
    for r,pin in zip(g['rollouts'],pins):
        trajectory=json.loads(r['trajectory']);crit=json.loads(r['critique'])
        assert trajectory['scored_answer_after_fixed_normalization']=='{"scenario":"normalized"}'
        assert crit['strict_task_success']==0 and crit['partial_credit_is_not_success']
        assert crit['scored_answer_sha256']==pin['scored_answer_sha256']==hashlib.sha256(b'{"scenario":"normalized"}').hexdigest()
