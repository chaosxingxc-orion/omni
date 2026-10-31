import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/flights/nx'))
import fold_a_c2_resident as m


def fixture():
    exp={'cells':[{'cell_id':m.baseline.CELL,'run_id':m.baseline.RUN,
        'readiness':{'state':'held'},'allowed_execution_stages':['check'],'model_fragment':'old'}],
        'stage3_algorithm':{'id':'none','operators_used':[]},'population':{'count':1080},
        'populations':[{'ids':['real1','real2']}],'execution_plan':{'planned_model_calls':4320}}
    old={'runtime_receipt':'old','decoding':{'temperature':0.6,'max_tokens':4096}}
    new={**copy.deepcopy(old),'runtime_receipt':m.RECEIPT}
    return exp,old,new


def test_only_serving_fields_change():
    exp,old,new=fixture();before=copy.deepcopy((exp,old,new));result=m.documents(exp,old,new)
    assert (exp,old,new)==before
    assert result['stage3_algorithm']==exp['stage3_algorithm']
    assert result['populations']==exp['populations'] and result['execution_plan']==exp['execution_plan']
    assert result['cells'][0]['readiness']['state']=='held'
    assert result['cells'][0]['model_fragment']==m.MODEL and result['service_session']==m.SESSION


@pytest.mark.parametrize('fault',['decode','receipt','ready','algorithm','count','run'])
def test_invalid_stage_refused(fault):
    exp,old,new=fixture()
    if fault=='decode':new['decoding']['max_tokens']=2048
    elif fault=='receipt':new['runtime_receipt']='other'
    elif fault=='ready':exp['cells'][0]['readiness']['state']='ready'
    elif fault=='algorithm':exp['stage3_algorithm']['operators_used']=['rollout']
    elif fault=='count':exp['population']['count']=1079
    else:exp['cells'][0]['run_id']='other'
    with pytest.raises(ValueError):m.documents(exp,old,new)
