import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/flights/nx'))
import fold_a_c2_supply_binding as m


def docs():
    b={};e={'cells':[{'cell_id':m.baseline.CELL,'run_id':m.baseline.RUN,
        'readiness':{'state':'held','requirements':[]},'allowed_execution_stages':['check']}]}
    p={'schema':'saea-fold-a-c2-supply-v1','rows':1080,'model_calls':0,'readiness':'held','registered':False,
        'cell':m.baseline.CELL,'run_id':m.baseline.RUN,
        'supply':{'relpath':f'prepared/{m.baseline.RUN}/phase-supply-v1.json','sha256':'a'*64}}
    return b,e,p


def test_binding_remains_held_and_does_not_mutate_source():
    b,e,p=docs();before=copy.deepcopy((b,e,p));x,y=m.bound_documents(b,e,p)
    assert (b,e,p)==before and x['prepared_phase_supply']==p['supply']
    assert y['cells'][0]['readiness']['state']=='held'
    assert y['cells'][0]['allowed_execution_stages']==['check']
    assert y['cells'][0]['readiness']['requirements']==[{'kind':'preparation',**p['supply']}]


@pytest.mark.parametrize('fault',['rows','calls','run','path','bound','ready'])
def test_wrong_or_spent_binding_refused(fault):
    b,e,p=docs()
    if fault=='rows':p['rows']=1079
    elif fault=='calls':p['model_calls']=1
    elif fault=='run':p['run_id']='other'
    elif fault=='path':p['supply']['relpath']='prepared/other/phase-supply-v1.json'
    elif fault=='bound':b['prepared_phase_supply']=p['supply']
    else:e['cells'][0]['readiness']['state']='ready'
    with pytest.raises(ValueError):m.bound_documents(b,e,p)
