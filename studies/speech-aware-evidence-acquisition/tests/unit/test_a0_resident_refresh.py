import copy
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
import refresh_a0_resident_binding as m


def fixture():
    spec={'profile_selection_state':'selected','runtime_receipt':'old.json','profile':'tp4-s128'}
    cell={'readiness':{'state':'held'}}
    old={'container':{'image_id':'image'},'deployment':{'instances':[{'instance_id':'i0','max_num_seqs':128}]},'protocol_sha256':'old','model':'fixed'}
    new=copy.deepcopy(old);new['protocol_sha256']='fresh'
    plan={'target':'tp4-s128','after':[{'name':'i0'}]}
    runtime={'observations':{'i0':{'pid':123}},'container_ids':{'i0':'fresh-id'},'docker_identity':[
        {'Name':'/i0','Id':'fresh-id','State':{'Running':True,'Pid':123},'RestartCount':0,'Image':'image',
         'Config':{'Labels':{'saea.deployment.profile':'tp4-s128'}}}]}
    return spec,cell,old,new,plan,runtime


def test_same_measured_profile_fresh_process_allowed():
    m.validate_refresh(*fixture())


@pytest.mark.parametrize('fault',['unselected','ready','duplicate','model','image','geometry','profile','members','pid','container','stopped','restart','foreign'])
def test_refuse_unproven_or_changed_runtime(fault):
    spec,cell,old,new,plan,runtime=fixture()
    if fault=='unselected':spec['profile_selection_state']='pending'
    elif fault=='ready':cell['readiness']['state']='ready'
    elif fault=='duplicate':spec['resident_refresh']={'already':True}
    elif fault=='model':new['model']='different'
    elif fault=='image':new['container']['image_id']='other'
    elif fault=='geometry':new['deployment']['instances'][0]['max_num_seqs']=96
    elif fault=='profile':plan['target']='other'
    elif fault=='members':plan['after'].append({'name':'extra'})
    elif fault=='pid':runtime['observations']['i0']['pid']=321
    elif fault=='container':runtime['container_ids']['i0']='old-id'
    elif fault=='stopped':runtime['docker_identity'][0]['State']['Running']=False
    elif fault=='restart':runtime['docker_identity'][0]['RestartCount']=1
    else:runtime['docker_identity'][0]['Config']['Labels'].clear()
    with pytest.raises(ValueError):m.validate_refresh(spec,cell,old,new,plan,runtime)
