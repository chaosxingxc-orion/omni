from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/flights/nx'))
from tfrl_current_supply_binding import bound_documents


@pytest.mark.parametrize('fault',[None,'cross_half','admitted','spent','wrong_rows','not_complete','already_bound'])
def test_exact_unspent_half_binding(fault):
    name='slurp-tfrl-current-seed-i1-h1-v1';run='SAEA-SLURP-TFRL-CURRENT-SEED-I1-H1-V1'
    base={};exp={'population':{'count':313},'cells':[{'cell_id':name,'run_id':run,'readiness':{'state':'held','requirements':[]},'allowed_execution_stages':['check']}]}
    prep={'schema':'saea-tfrl-current-supply-v1','rows':313,'cell':name,'run_id':run,'model_calls':0,'readiness':'held','registered':False,'supply':{'relpath':f'prepared/{run}/phase-supply-v1.json','sha256':'a'*64}}
    if fault=='cross_half':prep['supply']['relpath']=prep['supply']['relpath'].replace('H1','H2')
    elif fault=='admitted':exp['cells'][0]['readiness']['state']='ready'
    elif fault=='spent':prep['model_calls']=1
    elif fault=='wrong_rows':prep['rows']=312
    elif fault=='not_complete':prep['registered']=True
    elif fault=='already_bound':base['prepared_phase_supply']=prep['supply']
    if fault:
        with pytest.raises(ValueError):bound_documents(base,exp,prep)
    else:
        b,e=bound_documents(base,exp,prep)
        assert b['prepared_phase_supply']==prep['supply']
        assert e['cells'][0]['readiness']['requirements']==[{'kind':'preparation',**prep['supply']}]
        assert base=={} and exp['cells'][0]['readiness']['requirements']==[]
