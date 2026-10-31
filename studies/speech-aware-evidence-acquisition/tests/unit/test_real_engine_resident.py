import copy
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('resident',Path(__file__).resolve().parents[2]/'scripts/analysis/register_real_engine_resident.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    plan={'source':'tp4','before':[{'name':'i0'}],'old_names':{'i0':'i0-old'},'original_ids':{'i0':'original'}}
    original={'Name':'/i0','Id':'original','State':{'Pid':123},'Image':'image'}
    archive={'Name':'/i0-old','Id':'original','State':{'Running':False},'RestartCount':0,'Image':'image','Config':{'Labels':{'saea.deployment.profile':'tp4'}}}
    session={'session_id':'parent','deployment_instance_ids':['i0'],'pid':123,'identity':{'model_fragment':'model'}}
    previous={'profile':'tp4','service_session':'parent','model_fragment':'model'}
    return plan,[original],[archive],{'parent':session,'i0':{**session,'session_id':'i0'}},previous


def test_exact_stopped_owned_predecessor_only():
    args=fixture();m.validate_predecessor(*args)
    missing=copy.deepcopy(args);missing[2].clear()
    with pytest.raises(ValueError,match='count'):m.validate_predecessor(*missing)
    for field,value in [('Id','foreign'),('Name','/other'),('State',{'Running':True}),('RestartCount',1),('Image','other'),('Config',{'Labels':{}})]:
        changed=copy.deepcopy(args);changed[2][0][field]=value
        with pytest.raises(ValueError):m.validate_predecessor(*changed)


def test_session_or_topology_mismatch_refused():
    for mutation in ('pid','model','members','source'):
        args=fixture()
        if mutation=='pid':args[3]['i0']['pid']=999
        elif mutation=='model':args[3]['i0']['identity']={'model_fragment':'foreign'}
        elif mutation=='members':args[3]['parent']['deployment_instance_ids']=['unknown']
        else:args[0]['source']='foreign'
        with pytest.raises(ValueError):m.validate_predecessor(*args)
