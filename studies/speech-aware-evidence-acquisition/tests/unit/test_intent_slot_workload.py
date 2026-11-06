import copy,json
from pathlib import Path
import pytest
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
from intent_slot_workload import NodeWorkload,ARMS
from registered_workload import assert_execution_binding
from test_gepa_probe_workload import repo  # noqa: F401

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


def test_recovery_workload_reserves_six_contacts():
    c=config();c['node_arm']=ARMS[2]
    c['pipeline']={'protocol_version':'v2','node_rails':'sdk-handoff-v1','answer_only_fallback':'off',
                   'coverage_recovery':{'strategy':'complete-original-candidate-set-v1','max_extra_contacts':3}}
    assert NodeWorkload(c).CONTACTS == 6
    assert NodeWorkload(c).contact_budgets() == {
        'normal_generation_calls': 240, 'fallback_inclusive_cap': 480, 'fit_calls': 480}


@pytest.mark.parametrize('extra', [1, 2, 3])
def test_recovery_workload_budget_comes_from_config(extra):
    c = config(); c['node_arm'] = ARMS[2]
    c['pipeline'] = {'protocol_version': 'v2', 'node_rails': 'sdk-handoff-v1',
                     'answer_only_fallback': 'off', 'coverage_recovery': {
                         'strategy': 'complete-original-candidate-set-v1', 'max_extra_contacts': extra}}
    job = NodeWorkload(c)
    assert job.CONTACTS == 3 + extra
    assert job.contact_budgets()['fit_calls'] == 80 * (3 + extra)


def test_recovery_workload_derive_binds_fit_and_generation_caps(tmp_path, monkeypatch):
    import intent_slot_workload as workload
    c = config(); c['node_arm'] = ARMS[2]
    c['pipeline'] = json.loads((ROOT / 'configs/examples/psd-slot-coverage-recovery-v1.json').read_text())
    read_asset = workload.read_asset
    monkeypatch.setattr(workload, 'read_asset', lambda repo, relative: read_asset(ROOT, relative))
    job = NodeWorkload(c)
    job.material_checked = lambda *args: ({'source_roster': {}}, {
        'roster': {'roster_id': 'fixture', 'items': [{'audio_seconds': 1}] * 80,
                   'split_identity_hash': 'a' * 64}, 'view': {}})
    job.runtime_binding = lambda *args: {
        'model_fragment': 'bf16-thinking-tfgrpo-tp4-ctx48k-v1', 'service_session': 'fixture'}
    base, experiment, _, _, _ = job.derive(tmp_path, tmp_path, {})
    assert base['policy'] == 'phase-native-intent-slot-batched-recovery-v1'
    assert base['native_contact_fit']['contact_cap'] == 480
    assert experiment['execution_plan']['planned_model_calls'] == 480
    assert base['native_episode']['intent_slot_pipeline']['coverage_recovery'] == c['pipeline']['coverage_recovery']


def test_decision_policy_requires_v2():
    c=config();c['node_arm']=ARMS[2];c['decision_policy']={}
    with pytest.raises(ValueError,match='Decision policy requires'):
        NodeWorkload(c)


def test_decision_policy_changes_only_decision_material_and_renders(tmp_path,monkeypatch):
    import hashlib
    import intent_slot_workload as workload
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.phase_material import load_phase_material_specs,render_static_instruction
    c=config();c['node_arm']=ARMS[2]
    c['pipeline']={'protocol_version':'v2','node_rails':'sdk-handoff-v1','answer_only_fallback':'budgeted-v1'}
    read_asset=workload.read_asset
    monkeypatch.setattr(workload,'read_asset',lambda repo,relative:read_asset(ROOT,relative))
    def derive(cfg):
        job=NodeWorkload(cfg)
        job.material_checked=lambda *args: ({'source_roster':{}},{
            'roster':{'roster_id':'fixture','items':[{'audio_seconds':1}], 'split_identity_hash':'a'*64},'view':{}})
        job.runtime_binding=lambda *args: {'model_fragment':'bf16-thinking-tfgrpo-tp4-ctx48k-v1','service_session':'fixture'}
        return job.derive(tmp_path,tmp_path,{})[0]
    baseline=derive(c)
    path=tmp_path/'configs/probes/decision_candidate.txt'
    path.parent.mkdir(parents=True)
    path.write_text('Compare all supported hypotheses; return only the selected candidate_id.\n')
    c['decision_policy']={'thinking':{'enable_thinking':False,'answer_reserve_tokens':4096},
        'instruction_template':{'relpath':path.relative_to(tmp_path).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
    candidate=derive(c)
    expected=copy.deepcopy(baseline)
    expected['thinking_policy']['solver']=c['decision_policy']['thinking']
    expected['phase_material']['solver']['instruction_template']='probes/decision_candidate.txt'
    assert candidate==expected
    specs=load_phase_material_specs(candidate,default_phase_program(candidate['policy']))
    assert render_static_instruction(specs['solver'],tmp_path,{},tmp_path)==path.read_text()
    c['decision_policy']['instruction_template']['sha256']='0'*64
    with pytest.raises(ValueError,match='template pin differs'):
        derive(c)


def test_recovery_workload_combines_fallback_and_configured_budget(tmp_path, monkeypatch):
    import intent_slot_workload as workload
    c = config(); c['node_arm'] = ARMS[2]
    c['pipeline'] = {'protocol_version': 'v2', 'node_rails': 'sdk-handoff-v1',
        'answer_only_fallback': 'budgeted-v1', 'coverage_recovery': {
            'strategy': 'complete-original-candidate-set-v1', 'max_extra_contacts': 2}}
    read_asset = workload.read_asset
    monkeypatch.setattr(workload, 'read_asset', lambda repo, relative: read_asset(ROOT, relative))
    job = NodeWorkload(c)
    job.material_checked = lambda *args: ({'source_roster': {}}, {
        'roster': {'roster_id':'fixture', 'items':[{'audio_seconds':1}] * 80,
                   'split_identity_hash':'a' * 64}, 'view': {}})
    job.runtime_binding = lambda *args: {'model_fragment':'bf16-thinking-tfgrpo-tp4-ctx48k-v1', 'service_session':'fixture'}
    base, experiment, _, _, _ = job.derive(tmp_path, tmp_path, {})
    pipeline = base['native_episode']['intent_slot_pipeline']
    assert pipeline['answer_only_fallback'] == 'budgeted-v1'
    assert pipeline['coverage_recovery']['max_extra_contacts'] == 2
    assert base['native_contact_fit']['contact_cap'] == experiment['execution_plan']['planned_model_calls'] == 400


def test_combined_recovery_example_is_accepted():
    c = config(); c['node_arm'] = ARMS[2]
    c['pipeline'] = json.loads((ROOT / 'configs/examples/psd-slot-coverage-answer-only-v1.json').read_text())
    job = NodeWorkload(c)
    assert job.contact_budgets() == {'normal_generation_calls': 240,
        'fallback_inclusive_cap': 480, 'fit_calls': 480}


def test_node_stage_prepare_register_shared_budget(repo, monkeypatch):
    from types import SimpleNamespace
    import gepa_pair_workload as pair
    import intent_slot_workload as workload
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    from speech_aware_evidence_acquisition.core import config as composition
    from speech_aware_evidence_acquisition.core.flight import factory
    import tfrl_odd40_cells
    path, data, _ = repo
    c = config(); c['node_arm'] = ARMS[2]
    c['pipeline'] = {'protocol_version':'v2', 'node_rails':'sdk-handoff-v1',
        'answer_only_fallback':'budgeted-v1', 'coverage_recovery': {
            'strategy':'complete-original-candidate-set-v1', 'max_extra_contacts':2}}
    job = NodeWorkload(c)
    read_asset = workload.read_asset
    monkeypatch.setattr(workload, 'read_asset', lambda root, rel: read_asset(ROOT if not (root / rel).exists() else root, rel))
    job.material_checked = lambda *args: ({'source_roster':{}}, {
        'roster': {'roster_id':'fixture', 'items':[{'sample_id': sid, 'audio_seconds':1} for sid in c['sample_ids']],
                   'split_identity_hash':'a'*64}, 'view': {}})
    binding = {'model_fragment':'bf16-thinking-tfgrpo-tp4-ctx48k-v1','service_session':'fixture',
               'profile':'fixture','runtime_receipt':'fixture'}
    job.runtime_binding = lambda *args: binding
    monkeypatch.setattr(contracts, 'assert_execution_environment', lambda: None)
    monkeypatch.setattr(git_commit, 'require_clean_study_repo', lambda *_: 'a'*40)
    monkeypatch.setattr(composition, 'compose', lambda *_: SimpleNamespace(config_hash='b'*64,
        values={'arm':'raw-speech-agent-native-tools'}))
    monkeypatch.setattr(job, 'validate', lambda *args:80)
    monkeypatch.setattr(tfrl_odd40_cells, 'verify_output', lambda *args:None)
    class Runtime:
        def prepare_phase_supply(self, target, max_workers):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({'output': {}}))
    monkeypatch.setattr(factory, 'build_flight_runtime', lambda *args, **kwargs: Runtime())
    lock=path/pair.observation.LOCK; lock.parent.mkdir(parents=True,exist_ok=True); lock.write_text('{}')
    (path/'docs/exposure-ledger.md').write_text('history\n')
    stage=data/'work/codex/node-stage'
    job.stage(path,data,c['material'],stage)
    prepared=data/'work/codex/node-prepared'
    job.prepare(path,data,stage/'complete.json',prepared)
    job.register(path,data,stage/'complete.json',prepared/'complete.json')
    package=path/'docs/readiness'/job.NAME
    assert pair.read(package/'generation-plan-registered-v1.json')['execution_plan']['planned_model_calls']==400
    assert pair.read(package/'fit-plan-registered-v1.json')['execution_plan']['planned_model_calls']==400
    assert pair.read(path/f'configs/baseline/{job.NAME}.json')['native_episode']['intent_slot_pipeline']['answer_only_fallback']=='budgeted-v1'


def test_reference_normalizer_is_explicit_and_composes_with_coverage():
    c=config(); c['node_arm']=ARMS[2]
    c['pipeline']={'protocol_version':'v2','node_rails':'sdk-handoff-v1','answer_only_fallback':'budgeted-v1',
                   'reference_normalizer':'quarantine-unsupplied-v1'}
    assert NodeWorkload(c).config['pipeline']==c['pipeline']
    c['pipeline']['coverage_recovery']={'strategy':'complete-original-candidate-set-v1','max_extra_contacts':3}
    assert NodeWorkload(c).CONTACTS==6
    c['pipeline']['reference_normalizer']='unknown'
    with pytest.raises(ValueError): NodeWorkload(c)
