import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import tfrl_h1_observation_read as m
from test_tfrl_current_half_result import result


def inputs():
    rows=[]
    for i in range(313):
        r=result(i<200)['per_sample'][0]
        r.update(sample_id=f'row{i}',slot_f1_exact_fp=0,slot_f1_exact_fn=int(i>=190))
        rows.append(r)
    a=result()['aggregate'];a['samples']=313;a['official']['slu_f1']=200/313
    return {'aggregate':a,'per_sample':rows},[{'sample_id':r['sample_id']} for r in rows]


def test_task_reward_and_official_metric_are_separate():
    result,items=inputs();out=m.summarize_scores(result,items)
    assert out['task_correct_rows']==190 and out['task_error_rows']==123
    assert out['intent_accuracy']==200/313
    assert not out['hardness_confirmed'] and not out['independent_stochastic_draws_claimed']


@pytest.mark.parametrize('fault',['partial','duplicate','counts','nan','inf','micro'])
def test_refuse_incomplete_or_invalid_scores(fault):
    result,items=inputs()
    if fault=='partial':result['per_sample'].pop()
    elif fault=='duplicate':items[-1]=items[0]
    elif fault=='micro':result['aggregate']['official']['slu_f1']=.5
    else:result['per_sample'][0]['slot_f1_exact_fp']={'counts':-1,'nan':float('nan'),'inf':float('inf')}[fault]
    with pytest.raises(ValueError):m.summarize_scores(result,items)


def test_roles_use_receipts_and_reject_double_counting():
    rows=[{'native_episode_receipt':{'role_visits':[
        {'role_id':'perception','execution_receipt':{'contacts':[{'request_id':'p'}]}},
        {'role_id':'solver','execution_receipt':{'contacts':[{'request_id':'s1'},{'request_id':'s2'}]}}]}}]
    assert m.role_costs(rows)=={'perception':{'calls':1},'solver':{'calls':2}}
    with pytest.raises(ValueError):m.role_costs(rows+copy.deepcopy(rows))
