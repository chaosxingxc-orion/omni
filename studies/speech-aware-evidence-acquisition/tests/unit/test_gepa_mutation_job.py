import copy,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/analysis'))
import gepa_mutation_job as m
from test_gepa_mutation_runtime import setup
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical,sha


def prepared(tmp_path,monkeypatch):
    rt,_,_,_,source,_=setup(tmp_path,monkeypatch)
    repo=tmp_path/'repo';path=repo/'configs/model/frozen-model.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'model_lock_key':'frozen','decoding_params':{'max_tokens':100,'temperature':0}}))
    identity=rt.session_required_identity();(tmp_path/'identity.json').write_bytes(canonical(identity))
    monkeypatch.setattr(m.evidence,'seed_contract',lambda *a,**k:{'model':{'model_lock_key':'frozen'}})
    monkeypatch.setattr(m.base,'deployment_identity',lambda *a:identity)
    monkeypatch.setattr(m,'assert_execution_environment',lambda:None)
    cfg={'schema':m.CONFIG_SCHEMA,'name':'gepa-test','run_id':'GEPA-TEST','model_fragment':'frozen-model',
        'service_session':'resident','runtime_identity':{'root':'data','relpath':'identity.json','sha256':sha(identity)},
        'source':{'focus':'evidence-use','seed_material':{}},'bounds':{'resident_context':4096,'reserve_tokens':64,'max_context_bytes':2000000}}
    return repo,tmp_path,tmp_path/'work/codex/gepa',cfg


def test_prepare_and_all_shared_verifiers_bind_single_native_contact(tmp_path,monkeypatch):
    repo,data,out,cfg=prepared(tmp_path,monkeypatch);complete=m.prepare(repo,data,cfg,out)
    verified=m.verify_job(repo,data,out)
    import tfgrpo_final_job as final,grounded_synthesis_job as shared,tfgrpo_learning_job as recovery
    assert complete==verified[0] and complete['generation_calls']==complete['fit_calls']==1
    assert complete['historical_usage']['calls']==48 and complete['new_model_calls']==0
    assert final.verify_job(repo,data,out)==shared.verify_job(repo,data,out)==recovery.verify_job(repo,data,out)==verified
    assert verified[1]['decoding_params']['response_format']=={'type':'json_object'}


@pytest.mark.parametrize('fault',['calls','cost','source','roster','registration'])
def test_prepared_mutation_cannot_change_source_budget_or_custody(tmp_path,monkeypatch,fault):
    repo,data,out,cfg=prepared(tmp_path,monkeypatch);m.prepare(repo,data,cfg,out)
    manifest=m.read(out/'complete.json')
    if fault=='calls':manifest['generation_calls']=2
    elif fault=='cost':manifest['historical_usage']['calls']=0
    elif fault=='source':manifest['source_custody_sha256']='0'*64
    else:
        name='construction-roster.json' if fault=='roster' else 'registration.json'
        doc=m.read(out/name)
        if fault=='roster':doc['count']=2
        else:doc['contacts'][0]['sample_id']='slurp:test:1'
        (out/name).write_bytes(m.raw(doc));manifest['artifacts'][name]=m.file_sha(out/name)
    (out/'complete.json').write_bytes(m.raw(manifest))
    with pytest.raises(ValueError):m.verify_job(repo,data,out)


def test_shared_registration_binds_gepa_label_and_exact_single_contact(tmp_path,monkeypatch):
    import shutil
    import tfgrpo_final_job as final
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    from speech_aware_evidence_acquisition.core.stage3_algorithms import resolve_stage3_algorithm
    repo,data,out,cfg=prepared(tmp_path,monkeypatch)
    model=m.read(m.ROOT/'configs/model/bf16-thinking-tfgrpo-tp4-ctx48k-v1.json')
    model['model_lock_key']='frozen';model['decoding_params']['max_tokens']=100
    model['reasoning_budget_tokens']=50;model['answer_reserve_tokens']=50
    (repo/'configs/model/frozen-model.json').write_bytes(m.raw(model));m.prepare(repo,data,cfg,out)
    monkeypatch.setattr('speech_aware_evidence_acquisition.contracts.assert_execution_environment',lambda:None)
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda _: 'a'*40)
    for rel in ('configs/baseline/system-group-update-text-v1.json',
                'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json',
                'configs/experiment/slurp-tfrl-hardset-a0-observation-r1-v1.json',
                'configs/dataset/slurp-speech-native-stage3-construction-r01c.json'):
        target=repo/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(m.ROOT/rel,target)
    (repo/'docs').mkdir();(repo/'docs/exposure-ledger.md').write_text('')
    result=final.register(repo,data,out)
    assert result['calls']==result['fit_calls']==result['rows']==1
    exp=m.read(repo/'configs/experiment/gepa-test.json')
    assert exp['prepared_training_feedback']['schema']==m.runtime.SOURCE_SCHEMA
    assert exp['stage3_algorithm']['id']=='gepa-reflective-pareto-v1'
    algorithm=resolve_stage3_algorithm(exp['stage3_algorithm']['id'])
    assert set(exp['stage3_algorithm']['operators_used'])<=set(algorithm.operators)
    assert 'no Pareto survivor search' in exp['stage3_algorithm']['note']
    assert exp['tfgrpo_contact_fit']['contact_cap']==1
