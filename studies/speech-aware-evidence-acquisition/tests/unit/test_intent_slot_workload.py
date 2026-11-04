import copy,json
from pathlib import Path
import pytest
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
from intent_slot_workload import NodeWorkload,ARMS
from registered_workload import assert_execution_binding

def config():
    return {'schema':'saea-intent-slot-workload-v1','name':'node-test-v1','run_id':'NODE-TEST-V1',
      'sample_ids':[f'slurp:train:{i:06}:audio-a' for i in range(80)],'runtime_binding':{},
      'execution':{'admitted_level':32},'material':{},'node_arm':ARMS[0]}

@pytest.mark.parametrize('arm,contacts',[(ARMS[0],2),(ARMS[1],2),(ARMS[2],3)])
def test_registered_graph_contacts_and_width(arm,contacts):
    c=config();c['node_arm']=arm;j=NodeWorkload(c);assert j.CONTACTS==contacts
    spec={'workload_kind':j.KIND,'job_config':c,'execution':c['execution'],'profile':'b'}
    assert_execution_binding(spec,{'SAEA_ADMITTED_LEVEL':'32','SAEA_DEPLOYMENT_PROFILE':'b'})
    with pytest.raises(ValueError):assert_execution_binding(spec,{'SAEA_ADMITTED_LEVEL':'96','SAEA_DEPLOYMENT_PROFILE':'b'})

@pytest.mark.parametrize('fault',['arm','subset','test','order'])
def test_no_alternate_data_or_graph(fault):
    c=config()
    if fault=='arm':c['node_arm']='free-form'
    if fault=='subset':c['sample_ids'].pop()
    if fault=='test':c['sample_ids'][0]='slurp:test:000000:audio-a'
    if fault=='order':c['sample_ids'].reverse()
    with pytest.raises(ValueError):NodeWorkload(c)
