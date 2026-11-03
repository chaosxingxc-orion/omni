import copy
from pathlib import Path
import shutil
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
import gepa_probe_workload as m
from speech_aware_evidence_acquisition.rails.gepa_partition import digest


@pytest.fixture
def repo(tmp_path):
    repo=tmp_path/'repo'; data=tmp_path/'data'; repo.mkdir(); data.mkdir()
    paths=[m.observation.H1,m.observation.H2,'docs/receipts/rosters/slurp-stage3-construction-r01c.json',
        f'configs/dataset/{m.observation.DATASET}.json','configs/dataset/slurp-speech-native-test-full.json',
        m.observation.SOURCE_EXP,m.observation.SOURCE_BASE]
    for rel in paths:
        p=repo/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
    ids=sorted(r['sample_id'] for r in m.read(repo/m.observation.H1)['items'])
    part={'schema':'saea-gepa-partition-v1','feedback_ids':ids[:233],'selection_ids':ids[233:]}
    part['partition_sha256']=digest(part)
    inputs=(m.observation.H1,m.observation.H2,f'configs/dataset/{m.observation.DATASET}.json','configs/dataset/slurp-speech-native-test-full.json')
    part['input_files_sha256']={p:m.sha(repo/p) for p in inputs}
    path=repo/'partition.json';path.write_bytes(m.raw(part))
    return repo,data,m.pin(repo,'partition.json')


def test_fixed_probe_population_is_existing_feedback_prefix(repo):
    repo,_,pin=repo
    roster,view=m.population(repo,pin)
    assert len(roster['items'])==16 and view['count']==625
    assert [r['sample_id'] for r in roster['items']]==m.pinned(repo,pin)['feedback_ids'][:16]
    assert roster['claims_benefit'] is False
    from speech_aware_evidence_acquisition.core.roster import verify_roster
    verify_roster(roster)


@pytest.mark.parametrize('fault',['partition_hash','overlap','order','population_pin','selection_size'])
def test_partition_drift_or_leakage_refused(repo,fault):
    repo,_,pin=repo;part=m.pinned(repo,pin)
    if fault=='partition_hash':part['partition_sha256']='0'*64
    elif fault=='overlap':part['feedback_ids'][0]=part['selection_ids'][0]
    elif fault=='order':part['feedback_ids'].reverse()
    elif fault=='population_pin':part['input_files_sha256'][m.observation.H1]='0'*64
    else:part['selection_ids'].pop()
    if fault!='partition_hash':part['partition_sha256']=digest({k:v for k,v in part.items() if k not in ('partition_sha256','input_files_sha256')})
    (repo/'partition.json').write_bytes(m.raw(part));pin=m.pin(repo,'partition.json')
    with pytest.raises(ValueError):m.population(repo,pin)


def fake_material(repo,partition):
    source=m.read(repo/m.observation.SOURCE_BASE)
    source['thinking_policy']={role:copy.deepcopy(m.observation.THINKING_ARMS['baseline']) for role in ('perception','solver')}
    compiled=copy.deepcopy(source);compiled['policy']='phase-native-module-v1-c3'
    for name,doc in [('source.json',source),('compiled.json',compiled)]:
        (repo/name).write_bytes(m.raw(doc))
    return {'schema':'saea-gepa-seed-material-v1','source_pins':{'partition':partition},
        'source_baseline':m.pin(repo,'source.json'),'compiled_baseline':m.pin(repo,'compiled.json'),
        'runtime_binding':{'model_fragment':m.observation.MODEL,'profile':'profile',
            'service_session':'session','runtime_receipt':'receipt'}}


def test_derive_registers_three_contacts_per_sample_and_preserves_science(repo,monkeypatch):
    repo,data,partition=repo;material=fake_material(repo,partition)
    monkeypatch.setattr(m,'material_checked',lambda *_:(material, {
        'runtime_binding':material['runtime_binding'], 'partition':m.pinned(repo,partition),
        'source_baseline':m.pinned(repo,material['source_baseline']),
        'compiled_baseline':m.pinned(repo,material['compiled_baseline'])}))
    monkeypatch.setattr(m.observation,'runtime_binding',lambda *_:None)
    base,exp,roster,view,checked=m.derive(repo,data,{})
    assert base['native_contact_fit']['contact_cap']==48
    assert base['native_contact_fit']['run_id']==m.RUN+'-FIT'
    assert exp['execution_plan']['planned_model_calls']==96
    assert exp['execution_plan']['planned_speech_audio_seconds']>=6*sum(r['audio_seconds'] for r in roster['items'])
    assert exp['population']['count']==16
    assert exp['populations'][0]['ids']==[r['sample_id'] for r in roster['items']]
    assert exp['stage3_measurement']['claims_benefit'] is False
    assert checked==material['runtime_binding'] and view['count']==625


@pytest.mark.parametrize('key',['knowledge_stores','phase_material','thinking_policy','phase_tools'])
def test_derive_rejects_frozen_field_change(repo,monkeypatch,key):
    repo,data,partition=repo;material=fake_material(repo,partition)
    base=m.read(repo/'compiled.json');base[key]={'changed':True}
    (repo/'compiled.json').write_bytes(m.raw(base));material['compiled_baseline']=m.pin(repo,'compiled.json')
    monkeypatch.setattr(m,'material_checked',lambda *_:(material, {
        'runtime_binding':material['runtime_binding'], 'partition':m.pinned(repo,partition),
        'source_baseline':m.pinned(repo,material['source_baseline']),
        'compiled_baseline':m.pinned(repo,material['compiled_baseline'])}))
    monkeypatch.setattr(m.observation,'runtime_binding',lambda *_:None)
    with pytest.raises(ValueError,match='frozen'):m.derive(repo,data,{})


def test_execution_binding_requires_registered_width_and_profile():
    from registered_workload import assert_execution_binding
    spec={'workload_kind':m.KIND,'execution':{'admitted_level':96},'profile':'profile'}
    env={'SAEA_ADMITTED_LEVEL':'96','SAEA_DEPLOYMENT_PROFILE':'profile'}
    assert_execution_binding(spec,env)
    for bad in ({}, {**env,'SAEA_ADMITTED_LEVEL':'64'}, {**env,'SAEA_DEPLOYMENT_PROFILE':'other'}):
        with pytest.raises(ValueError,match='C96'):assert_execution_binding(spec,bad)


def test_stage_emits_exact_files_and_refuses_repeat(repo,monkeypatch):
    repo,data,partition=repo;material=fake_material(repo,partition)
    monkeypatch.setattr(m,'material_checked',lambda *_:(material, {
        'runtime_binding':material['runtime_binding'],'partition':m.pinned(repo,partition),
        'source_baseline':m.pinned(repo,material['source_baseline']),
        'compiled_baseline':m.pinned(repo,material['compiled_baseline'])}))
    monkeypatch.setattr(m.observation,'runtime_binding',lambda *_:None)
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda *_:'a'*40)
    (repo/'docs/exposure-ledger.md').write_text('old spend\n')
    out=data/'work/codex/stage'
    result=m.stage(repo,data,{'relpath':'material.json','sha256':'a'*64},out)
    assert result['normal_generation_calls']==result['fit_calls']==48
    assert result['fallback_inclusive_cap']==96
    assert not result['generation_authorized']
    assert m.verify_stage(repo,data,out/'complete.json')==result
    assert (repo/'docs/exposure-ledger.md').read_text()=='old spend\n'
    with pytest.raises(ValueError):m.stage(repo,data,{},data/'work/codex/repeated')
    base=m.read(repo/f'configs/baseline/{m.NAME}.json');base['native_contact_fit']['contact_cap']=32
    (repo/f'configs/baseline/{m.NAME}.json').write_bytes(m.raw(base))
    with pytest.raises(ValueError,match='changed'):m.verify_stage(repo,data,out/'complete.json')


def test_registered_validation_rejects_budget_and_fixed_identity_drift(repo,monkeypatch):
    repo,data,partition=repo;material=fake_material(repo,partition)
    monkeypatch.setattr(m,'material_checked',lambda *_:(material, {
        'runtime_binding':material['runtime_binding'],'partition':m.pinned(repo,partition),
        'source_baseline':m.pinned(repo,material['source_baseline']),
        'compiled_baseline':m.pinned(repo,material['compiled_baseline'])}))
    monkeypatch.setattr(m.observation,'runtime_binding',lambda *_:None)
    base,exp,roster,view,binding=m.derive(repo,data,{})
    (repo/m.ROSTER).write_bytes(m.raw(roster))
    (repo/f'configs/baseline/{m.NAME}.json').write_bytes(m.raw(base))
    spec={'cell':m.NAME,'experiment':m.NAME,'baseline':m.NAME,'run_id':m.RUN,'attempt_id':'generation-v1',
        'fit_run_id':m.RUN+'-FIT','fit_attempt_id':'fit-v1','rows':16,'workload_kind':m.KIND,
        'model_fragment':binding['model_fragment'],'runtime_binding':binding,'profile':binding['profile'],
        'service_session':binding['service_session'],'runtime_receipt':binding['runtime_receipt'],
        'split_identity_hash':roster['split_identity_hash'],'execution':{'admitted_level':96},
        'material':{},'source_experiment':m.pin(repo,m.observation.SOURCE_EXP),
        'dataset':m.pin(repo,f'configs/dataset/{m.observation.DATASET}.json'),
        'roster':m.pin(repo,m.ROSTER),'scientific_baseline_sha256':m.scientific_hash(base)}
    bad_exp=copy.deepcopy(exp);bad_exp['execution_plan']['planned_model_calls']=64
    with pytest.raises(ValueError,match='budget'):m.validate(repo,data,m.NAME,spec,bad_exp,view)
    for key,value in [('rows',15),('fit_run_id','other'),('profile','other')]:
        with pytest.raises(ValueError,match='identity'):
            m.validate(repo,data,m.NAME,{**spec,key:value},exp,view)
    broken=copy.deepcopy(base);broken['native_contact_fit']['contact_cap']=32
    (repo/f'configs/baseline/{m.NAME}.json').write_bytes(m.raw(broken))
    with pytest.raises(ValueError,match='program'):
        m.validate(repo,data,m.NAME,spec,exp,view)


def test_register_emits_budget_plans_and_both_arm_recipe(repo,monkeypatch):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    from speech_aware_evidence_acquisition.core import config
    repo,data,partition=repo;material=fake_material(repo,partition)
    monkeypatch.setattr(m,'material_checked',lambda *_:(material, {
        'runtime_binding':material['runtime_binding'],'partition':m.pinned(repo,partition),
        'source_baseline':m.pinned(repo,material['source_baseline']),
        'compiled_baseline':m.pinned(repo,material['compiled_baseline'])}))
    monkeypatch.setattr(m.observation,'runtime_binding',lambda *_:None)
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda *_:'a'*40)
    monkeypatch.setattr(config,'compose',lambda *_:SimpleNamespace(config_hash='b'*64,values={'arm':'raw-speech-agent-native-tools'}))
    monkeypatch.setattr(m,'validate',lambda *args:16) # Separate tests exercise scientific validation.
    lock=repo/m.observation.LOCK;lock.parent.mkdir(parents=True,exist_ok=True);lock.write_bytes(b'{}\n')
    (repo/'docs/exposure-ledger.md').write_text('historical spend\n')
    stage=data/'work/codex/stage';m.stage(repo,data,{'relpath':'material.json','sha256':'a'*64},stage)
    supply=data/'prepared'/m.RUN/'phase-supply-v1.json';supply.parent.mkdir(parents=True);supply.write_bytes(b'{}\n')
    prepared=data/'work/codex/prepared.json';prepared.write_bytes(m.raw({'schema':'saea-gepa-probe-prepared-v1',
        'stage_sha256':m.sha(stage/'complete.json'),'rows':16,'run_id':m.RUN,'new_model_calls':0,
        'supply':m.pin(data,str(supply.relative_to(data)))}))
    result=m.register(repo,data,stage/'complete.json',prepared)
    pkg=repo/'docs/readiness'/m.NAME
    assert result['readiness']=='held' and result['new_model_calls']==0
    assert m.read(pkg/'generation-plan-registered-v1.json')['execution_plan']['planned_model_calls']==96
    assert m.read(pkg/'fit-plan-registered-v1.json')['execution_plan']['planned_model_calls']==48
    recipe=m.read(pkg/'analysis-recipe-v1.json')
    assert recipe['arms']==['gepa','lock'] and recipe['delivered_arm']=='lock'
    assert recipe['rows']==16 and recipe['schema']=='saea-gepa-probe-read-v1'
    assert (repo/'docs/exposure-ledger.md').read_text().startswith('historical spend\n')
    with pytest.raises(ValueError,match='registration|spend'):
        m.register(repo,data,stage/'complete.json',prepared)
