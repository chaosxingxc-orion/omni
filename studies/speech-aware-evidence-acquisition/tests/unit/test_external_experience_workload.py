import copy,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/analysis'))
import external_experience_workload as m
from test_gepa_pair_workload import config,derived,repo


@pytest.mark.parametrize('kind', ['external-tfgrpo-reference-v1', 'external-tfgrpo-v1',
                                 'external-ace-v1', 'external-reflexion-v1'])
def test_generated_algorithm_label_is_accepted_by_official_reader(kind):
    from speech_aware_evidence_acquisition.core.stage3_algorithms import validate_stage3_algorithm_label
    label = m.algorithm_metadata(kind)
    record, used = validate_stage3_algorithm_label(label)
    assert used == ('context_application',)
    assert record.id == label['id']


def test_explicit_external_identity_and_registered_width():
    from registered_workload import assert_execution_binding
    cfg={**config(),'schema':m.SCHEMA};job=m.ExternalExperienceWorkload(cfg)
    assert job.KIND=='external-experience-pair' and job.CONTACTS==3 and job.ARMS==['learned','lock']
    spec={'workload_kind':job.KIND,'job_config':cfg,'execution':cfg['execution'],'profile':'profile'}
    assert_execution_binding(spec,{'SAEA_ADMITTED_LEVEL':'16','SAEA_DEPLOYMENT_PROFILE':'profile'})
    with pytest.raises(ValueError):assert_execution_binding(spec,{'SAEA_ADMITTED_LEVEL':'32','SAEA_DEPLOYMENT_PROFILE':'profile'})
    for field,value in [('schema','saea-gepa-pair-workload-v1'),('sample_ids',['slurp:test:1'])]:
        with pytest.raises(ValueError):m.ExternalExperienceWorkload({**cfg,field:value})


def test_shared_registration_uses_two_solver_budget_and_external_recipe(repo,monkeypatch):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    from speech_aware_evidence_acquisition.core import config as composition
    old,_=derived(repo,monkeypatch);path,data,_=repo
    base,exp,roster,view,binding=old.derive(path,data,old.config['material'])
    cfg={**old.config,'schema':m.SCHEMA};job=m.ExternalExperienceWorkload(cfg)
    base['native_contact_fit']['contact_cap']=48;exp['execution_plan']['planned_model_calls']=96
    monkeypatch.setattr(job,'derive',lambda *_:(base,exp,roster,view,binding))
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda *_:'a'*40)
    monkeypatch.setattr(composition,'compose',lambda *_:SimpleNamespace(config_hash='b'*64,values={'arm':'raw-speech-agent-native-tools'}))
    monkeypatch.setattr(job,'validate',lambda *args:16)
    lock=path/m.observation.LOCK;lock.parent.mkdir(parents=True,exist_ok=True);lock.write_text('{}\n')
    (path/'docs/exposure-ledger.md').write_text('old cost\n')
    stage=data/'work/codex/stage';record=job.stage(path,data,cfg['material'],stage)
    assert record['normal_generation_calls']==record['fit_calls']==48 and record['fallback_inclusive_cap']==96
    supply=data/'prepared'/job.RUN/'phase-supply-v1.json';supply.parent.mkdir(parents=True);supply.write_text('{}\n')
    prepared=data/'prepared.json';prepared.write_text(json.dumps({'schema':'saea-gepa-pair-prepared-v1','stage_sha256':m.sha(stage/'complete.json'),'rows':16,'run_id':job.RUN,'supply':m.pin(data,str(supply.relative_to(data))),'new_model_calls':0}))
    job.register(path,data,stage/'complete.json',prepared)
    pkg=path/'docs/readiness'/job.NAME
    assert m.read(pkg/'candidate.json')['workload_kind']==m.KIND
    assert m.read(pkg/'generation-plan-registered-v1.json')['execution_plan']['planned_model_calls']==96
    assert m.read(pkg/'fit-plan-registered-v1.json')['execution_plan']['planned_model_calls']==48
    recipe=m.read(pkg/'analysis-recipe-v1.json')
    assert recipe['arms']==['learned','lock'] and recipe['schema']==job.RESULT_SCHEMA
    assert 'External reviewed experience' in (path/'docs/exposure-ledger.md').read_text()


def test_external_source_kind_reaches_component_compiler():
    from speech_aware_evidence_acquisition.rails.consumption_module import COMPONENT_CATALOGUE
    from speech_aware_evidence_acquisition.rails.tfgrpo_consumption import validate_params
    component=next(c for c in COMPONENT_CATALOGUE if c.component_id=='tfgrpo-solver-experience')
    assert 'source_kind' in component.optional_params


def test_target_grouping_rejected_before_source_or_runtime_contact(tmp_path):
    import hashlib
    from speech_aware_evidence_acquisition.rails.adapters.runtime_view import load_runtime_view
    ids=['slurp:train:000090:audio-b', 'slurp:train:000001:audio-a']
    cfg={**config(),'schema':m.SCHEMA,'sample_ids':ids}
    with pytest.raises(ValueError,match='runtime sample_id order'):
        m.ExternalExperienceWorkload(cfg)
    # Real label-free runtime loader defines the order, not a test-only mock.
    items=[]
    for n,sid in enumerate(ids):
        p=tmp_path/f'{n}.flac';p.write_bytes(bytes([n]))
        items.append({'sample_id':sid,'media_relpath':p.name,
                      'audio_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'audio_seconds':1.0})
    actual=load_runtime_view({'schema':'saea-speech-native-runtime-view-v1','count':2,'items':items},tmp_path,sample_ids=ids)
    canonical=[row['sample_id'] for row in actual]
    assert canonical==sorted(ids) and set(canonical)==set(ids)
    assert m.ExternalExperienceWorkload({**cfg,'sample_ids':canonical}).config['sample_ids']==canonical
