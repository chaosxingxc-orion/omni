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
    c=config();c['node_arm']=arm;j=NodeWorkload(c);assert j.CONTACTS==contacts;assert j.DELIVERED_ARM==arm
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

def test_comparison_rejects_different_roster_or_scorer():
    from intent_slot_read import compare
    def result(arm):return {'node_arm':arm,'rows':80,'scorer_identity':{'scorer':'same'},
      'candidate_coverage':{'rows':[{'sample_id':sid} for sid in config()['sample_ids']]},
      'official':{'slu_f1':.7},'intent_accuracy':.8,'task_correct_rows':30}
    rs={a:result(a) for a in ARMS}
    assert compare(rs)['contrasts'][ARMS[2]]['strict_delta_rows']==0
    rs[ARMS[1]]['candidate_coverage']['rows'].reverse()
    with pytest.raises(ValueError):compare(rs)


def test_node_policy_old_config_is_unchanged_and_joint_rejects_slot(tmp_path, monkeypatch):
    c=config(); job=NodeWorkload(c)
    assert 'perception_policy' not in job.config and 'slot_policy' not in job.config
    c['slot_policy']={'thinking':{'enable_thinking':False,'answer_reserve_tokens':4096},'instruction_template':{'relpath':'configs/probes/any.txt','sha256':'0'*64}}
    job=NodeWorkload(c);job.material_checked=lambda *args: ({'source_roster':{}},{'roster':{'items':[],'split_identity_hash':'a'*64},'view':{}});job.runtime_binding=lambda *args: {'model_fragment':'unused'}
    with pytest.raises(ValueError,match='Slot policy requires'):
        job.derive(ROOT,tmp_path,{})

@pytest.mark.parametrize('relpath,digest,error',[
    ('configs/probes/slurp_native_candidate_perception_off_v1.txt','0'*64,'template pin differs'),
    ('configs/probes/../../docs/exposure-ledger.md','0'*64,'under configs/probes'),
    ('docs/exposure-ledger.md','0'*64,'under configs/probes'),
])
def test_perception_policy_rejects_bad_template_pin_and_escape(tmp_path,relpath,digest,error):
    c=config();c['perception_policy']={'thinking':{'enable_thinking':False,'answer_reserve_tokens':2048},'instruction_template':{'relpath':relpath,'sha256':digest}}
    job=NodeWorkload(c);job.material_checked=lambda *args: ({'source_roster':{}},{'roster':{'items':[],'split_identity_hash':'a'*64},'view':{}});job.runtime_binding=lambda *args: {'model_fragment':'unused'}
    with pytest.raises(ValueError,match=error):job.derive(ROOT,tmp_path,{})


def test_node_width_ceiling_is_128():
    c=config();c['execution']['admitted_level']=128
    assert NodeWorkload(c).config['execution']['admitted_level']==128
    c['execution']['admitted_level']=129
    with pytest.raises(ValueError,match='bounded client width'):
        NodeWorkload(c)


def test_v2_sdk_rails_selection_is_explicit_and_restricted():
    c=config();c['node_arm']=ARMS[2]
    c['pipeline']={'protocol_version':'v2','node_rails':'sdk-handoff-v1','answer_only_fallback':'budgeted-v1'}
    assert NodeWorkload(c).config['pipeline']==c['pipeline']
    for bad in ({'protocol_version':'v2','node_rails':'sdk-handoff-v1'},
                {'protocol_version':'v2','node_rails':'sdk-handoff-v1','answer_only_fallback':'off'}):
        c['pipeline']=bad
        with pytest.raises(ValueError,match='explicit supported'):NodeWorkload(c)
    c['pipeline']={'protocol_version':'v2','node_rails':'sdk-handoff-v1','answer_only_fallback':'budgeted-v1'}
    c['node_arm']=ARMS[1]
    with pytest.raises(ValueError,match='explicit supported'):NodeWorkload(c)
