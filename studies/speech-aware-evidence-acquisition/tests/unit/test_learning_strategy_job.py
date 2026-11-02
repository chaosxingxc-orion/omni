"""Reusable strategy preparation binds exact source and truthful phase readiness."""
import copy
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/analysis'))
import learning_strategy_job as job
import tfgrpo_final_job as final
import grounded_synthesis_job as shared
from test_ace_runtime import setup
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical


def prepared(tmp_path,monkeypatch):
    runtime,_,_,source,spec,_ = setup(tmp_path,monkeypatch)
    repo = tmp_path/'repo'; path = repo/'configs/model/frozen-model.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'decoding_params':{'max_tokens':100,'temperature':0}}))
    files = {'spec.json':{},'roster.json':{'items':[{'sample_id':runtime._samples[0].sample_id}]},
             'identity.json':runtime.session_required_identity()}
    for name,value in files.items():(tmp_path/name).write_bytes(canonical(value))
    monkeypatch.setattr(job,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(job.base,'deployment_identity',lambda *a:runtime.session_required_identity())
    out = tmp_path/'work/codex/ace-job'
    args = dict(strategy=job.ace.STRATEGY,source_spec=tmp_path/'spec.json',source_roster=tmp_path/'roster.json',
        tokenizer_relpath='tokenizer',runtime_identity=tmp_path/'identity.json',
        bounds={k:v for k,v in spec['document']['bounds'].items() if k != 'max_calls'},
        out=out,job_name='ace-test',run_id='ACE-TEST',service_session='resident',planned_gpu_hours=1)
    return repo,tmp_path,out,args


def test_shared_prepare_verify_has_no_fit_claim_and_routes_register_run_verifier(tmp_path,monkeypatch):
    repo,data,out,args = prepared(tmp_path,monkeypatch)
    complete = job.prepare(repo,data,**args)
    verified,reg,roster = job.verify_job(repo,data,out)
    assert complete == verified and complete['generation_calls'] == complete['fit_calls'] == 2
    assert complete['runtime_rows'] == complete['groups'] == 1
    assert complete['phase_readiness']['later'] == 'AWAITING_OBSERVED_PREDECESSOR'
    assert complete['new_model_calls'] == complete['new_tokenizer_calls'] == 0
    assert final.verify_job(repo,data,out) == shared.verify_job(repo,data,out) == (verified,reg,roster)
    assert job.base.verify_job(repo,data,out)==(verified,reg,roster)  # Shared recovery worker verifier.


@pytest.mark.parametrize('fault',['calls','cost','phase','source','tokenizer','roster'])
def test_changed_strategy_job_refused(tmp_path,monkeypatch,fault):
    repo,data,out,args = prepared(tmp_path,monkeypatch);job.prepare(repo,data,**args)
    complete = job.read(out/'complete.json')
    if fault == 'calls':complete['generation_calls'] += 1
    elif fault == 'cost':complete['historical_usage']['calls'] += 1
    elif fault == 'phase':complete['phase_readiness']['later'] = 'READY'
    elif fault == 'source':(data/'roster.json').write_text('{}')
    else:
        name = 'source.json' if fault == 'tokenizer' else 'construction-roster.json'
        doc = job.read(out/name)
        if fault == 'tokenizer':doc['tokenizer_identity'] = {'changed':True}
        else:doc['count'] += 1
        (out/name).write_bytes(job.raw(doc));complete['artifacts'][name] = job.file_sha(out/name)
    (out/'complete.json').write_bytes(job.raw(complete))
    with pytest.raises((ValueError,KeyError)):job.verify_job(repo,data,out)


def test_unknown_strategy_and_cross_parent_are_not_silent_fallbacks(tmp_path,monkeypatch):
    repo,data,out,args = prepared(tmp_path,monkeypatch)
    args['strategy'] = 'not-implemented'
    with pytest.raises(ValueError,match='implemented'):job.prepare(repo,data,**args)
    assert not out.exists()


def test_shared_registration_emits_real_ace_label_and_exact_two_phase_budget(tmp_path,monkeypatch):
    import shutil
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    from speech_aware_evidence_acquisition.core.stage3_algorithms import resolve_stage3_algorithm
    repo,data,out,args = prepared(tmp_path,monkeypatch)
    model = job.read(job.ROOT/'configs/model/bf16-thinking-tfgrpo-tp4-ctx48k-v1.json')
    model['decoding_params']['max_tokens']=100
    model['reasoning_budget_tokens']=50;model['answer_reserve_tokens']=50
    (repo/'configs/model/frozen-model.json').write_bytes(job.raw(model))
    job.prepare(repo,data,**args)
    monkeypatch.setattr('speech_aware_evidence_acquisition.contracts.assert_execution_environment',lambda:None)
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda _: 'a'*40)
    for rel in ('configs/baseline/system-group-update-text-v1.json',
                'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json',
                'configs/experiment/slurp-tfrl-hardset-a0-observation-r1-v1.json',
                'configs/dataset/slurp-speech-native-stage3-construction-r01c.json'):
        target=repo/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(job.ROOT/rel,target)
    (repo/'docs').mkdir();(repo/'docs/exposure-ledger.md').write_text('')
    result=job.register(repo,data,out)
    assert result['calls']==result['fit_calls']==2 and result['rows']==1
    exp=job.read(repo/'configs/experiment/ace-test.json')
    assert exp['prepared_training_feedback']['schema']==job.runtime.SOURCE_SCHEMA
    assert exp['stage3_algorithm']['id']=='ace-grow-refine-v1'
    record=resolve_stage3_algorithm(exp['stage3_algorithm']['id'])
    assert set(exp['stage3_algorithm']['operators_used']) <= set(record.operators)
    assert job.ace.STRATEGY in exp['stage3_algorithm']['note']
    assert exp['tfgrpo_contact_fit']['contact_cap']==2
    assert exp['cells'][0]['readiness']['state']=='held'


def test_json_config_prepare_and_verify_without_inline_python(tmp_path,monkeypatch):
    repo,data,out,args=prepared(tmp_path,monkeypatch)
    args['max_context_bytes']=2000000
    config={'schema':job.CONFIG_SCHEMA,'mode':'prepare','repo':str(repo),'data':str(data),
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in args.items()}}
    path=tmp_path/'command.json';path.write_bytes(job.raw(config))
    prepared_result=job.execute_config(path)
    config.update(mode='verify',arguments={'job':str(out)});path.write_bytes(job.raw(config))
    assert job.execute_config(path)==prepared_result


@pytest.mark.parametrize('mode',['register','admit','run'])
def test_json_config_routes_to_shared_api(tmp_path,monkeypatch,mode):
    calls=[]
    monkeypatch.setattr(job,mode,lambda repo,data,**args:calls.append((repo,data,args)) or {'ok':True})
    args={'job':str(tmp_path/'job')}
    if mode=='run':args.update(out=str(tmp_path/'out'),after=str(tmp_path/'release.json'),predecessor_run='PREVIOUS')
    path=tmp_path/'command.json'
    path.write_bytes(job.raw({'schema':job.CONFIG_SCHEMA,'mode':mode,'repo':str(tmp_path/'repo'),
        'data':str(tmp_path/'data'),'arguments':args}))
    assert job.execute_config(path)=={'ok':True}
    assert calls[0][0]==tmp_path/'repo' and calls[0][2]['job']==tmp_path/'job'


@pytest.mark.parametrize('fault',['mode','extra','path','missing','schema'])
def test_json_command_refuses_ambiguous_config(tmp_path,fault):
    config={'schema':job.CONFIG_SCHEMA,'mode':'verify','repo':str(tmp_path/'repo'),
        'data':str(tmp_path/'data'),'arguments':{'job':str(tmp_path/'job')}}
    if fault=='mode':config['mode']='custom-launch'
    elif fault=='extra':config['arguments']['code']='print(1)'
    elif fault=='path':config['repo']='relative'
    elif fault=='missing':config['arguments']={}
    else:config['schema']='unknown'
    path=tmp_path/'command.json';path.write_bytes(job.raw(config))
    with pytest.raises(ValueError):job.execute_config(path)
