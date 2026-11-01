import copy
import json
import pytest
from speech_aware_evidence_acquisition.rails.tfgrpo_projection import summary_trajectory,restore,VERSION


def source():
    q={'task_instruction':'instruction '*300,'supplied_evidence':['evidence '*1000],
       'sample_id':'slurp:train:1','history':[{'role':'user','content':'observed perception'}]}
    t={'request':{**copy.deepcopy(q),'history':q['history']+[{'role':'control','content':'solve'}],
                  'decoding_params':{'seed':1},'raw_speech_query':{'original_sha256':'a'*64}},
       'response_text':'Observed answer and reasoning'}
    return q,t


def test_lossless_reference_dedup_preserves_response_and_actual_conditioning():
    q,t=source();original=json.dumps(t)
    result=summary_trajectory(json.dumps(q),original);doc=json.loads(result)
    assert doc['schema']==VERSION
    assert restore(q,doc)==t
    assert doc['trajectory']['response_text']==t['response_text']
    assert doc['trajectory']['request']['history']==t['request']['history'][-1:]
    assert len(result)<len(original)/3


def test_unequal_fields_are_never_replaced():
    q,t=source();t['request']['task_instruction']='different';t['request']['history']=[]
    doc=json.loads(summary_trajectory(json.dumps(q),json.dumps(t)))
    assert doc['trajectory']['request']['task_instruction']=='different'
    assert not doc['history_prefix_from_question']
    assert restore(q,doc)==t


@pytest.mark.parametrize('changed',['question','response','reference'])
def test_reconstruction_rejects_changed_context(changed):
    q,t=source();doc=json.loads(summary_trajectory(json.dumps(q),json.dumps(t)))
    if changed=='question':q['sample_id']='another'
    elif changed=='response':doc['trajectory']['response_text']='another'
    else:doc['request_fields_from_question'].append('audio')
    with pytest.raises(ValueError):restore(q,doc)


def test_generic_trajectories_unchanged():
    assert summary_trajectory('plain question','plain trajectory')=='plain trajectory'
    assert summary_trajectory('{}','{"response": "value"}')=='{"response": "value"}'


def test_full_learner_changes_summary_only(tmp_path):
    from test_tfgrpo_learning import group,learn,Model
    q,t=source();g=group();g['query']=json.dumps(q)
    for r in g['rollouts']:r['trajectory']=json.dumps(t)
    model=Model();learn(tmp_path,model,[g])
    summaries=[r for r in model.requests if r['stage']=='summary']
    assert all(VERSION in r['messages'][1]['content'] for r in summaries)
    advantage=next(r for r in model.requests if r['stage']=='advantage')
    assert g['query'] in advantage['messages'][1]['content']
    assert VERSION not in advantage['messages'][1]['content']
