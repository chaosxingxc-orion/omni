import copy,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/analysis'))
import gepa_pair_workload as m


def config():
    return {'schema':m.SCHEMA,'name':'pair-a','run_id':'PAIR-A','sample_ids':['slurp:train:1'],
        'runtime_binding':{'model_fragment':'model','profile':'profile','service_session':'session','runtime_receipt':'receipts/runtime.json'},
        'execution':{'admitted_level':16},'material':{'relpath':'material.json','sha256':'a'*64}}


@pytest.mark.parametrize('fault',[None,'test','duplicate','width','run','extra'])
def test_parameterized_pairs_have_distinct_identities(fault):
    value=config()
    if fault=='test':value['sample_ids']=['slurp:test:1']
    elif fault=='duplicate':value['sample_ids']*=2
    elif fault=='width':value['execution']['admitted_level']=True
    elif fault=='run':value['run_id']='../bad'
    elif fault=='extra':value['extra']=True
    if fault:
        with pytest.raises(ValueError):m.PairWorkload(value)
    else:
        a=m.PairWorkload(value);value.update(name='pair-b',run_id='PAIR-B')
        b=m.PairWorkload(value)
        assert a.NAME!=b.NAME and a.RUN!=b.RUN and a.ROSTER!=b.ROSTER
        assert a.config['name']=='pair-a'


def test_one_two_card_profile_is_explicit_not_old_eight_card_requirement(tmp_path):
    value=config();job=m.PairWorkload(value)
    docs={'configs/model/model.json':{'runtime_receipt':'receipts/runtime.json'},
        'configs/deployment/profile.json':{'model_fragment':'model','admitted_level':48,'instances':[{'name':'b','cards':[6,7]}]},
        'docs/receipts/runtime.json':{'deployment':{'instances':[{'instance_id':'b'}]}}}
    for path,doc in docs.items():
        target=tmp_path/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(doc))
    assert job.runtime_binding(tmp_path)==value['runtime_binding']
    value['execution']['admitted_level']=96
    with pytest.raises(ValueError,match='width exceeds'):m.PairWorkload(value).runtime_binding(tmp_path)

from test_gepa_probe_workload import repo  # noqa: F401


def derived(repo,monkeypatch):
    path,data,partition=repo
    ids=m.pinned(path,partition)['feedback_ids'][:16]
    cfg=config();cfg['sample_ids']=ids;job=m.PairWorkload(cfg)
    source=m.read(path/m.observation.SOURCE_BASE)
    source['thinking_policy']={role:copy.deepcopy(m.observation.THINKING_ARMS['baseline']) for role in ('perception','solver')}
    compiled=copy.deepcopy(source);compiled['policy']='phase-native-module-v1-c4'
    model={'server_args':['--max-model-len','49152'],'decoding_params':{'max_tokens':4096},'model_lock_key':'frozen'}
    model_path=path/'configs/model/model.json';model_path.parent.mkdir(parents=True,exist_ok=True);model_path.write_bytes(m.raw(model))
    checked={'source_pins':{'partition':partition},'partition':m.pinned(path,partition),
             'source_baseline':source,'compiled_baseline':compiled,'model':model}
    monkeypatch.setattr(job,'material_checked',lambda *_:({'schema':'saea-gepa-pair-material-v1'},checked))
    monkeypatch.setattr(job,'runtime_binding',lambda *_:cfg['runtime_binding'])
    return job,checked


def test_real_fixed_population_derivation_and_four_contact_budget(repo,monkeypatch):
    job,checked=derived(repo,monkeypatch);path,data,_=repo
    base,exp,roster,view,binding=job.derive(path,data,job.config['material'])
    assert base['native_contact_fit']['contact_cap']==64
    assert exp['execution_plan']['planned_model_calls']==128
    assert len(roster['items'])==16 and view['count']==625
    assert base['knowledge_stores']==checked['source_baseline']['knowledge_stores']


@pytest.mark.parametrize('fault',['decoding','knowledge','shrunk','rope','too_large'])
def test_pair_frozen_science_and_native_context_boundary(repo,monkeypatch,fault):
    job,checked=derived(repo,monkeypatch);path,data,_=repo
    model=m.read(path/'configs/model/model.json')
    if fault=='decoding':model['decoding_params']['max_tokens']=16384
    elif fault=='knowledge':checked['compiled_baseline']['knowledge_stores']={}
    elif fault=='shrunk':model['server_args'][1]='24576'
    elif fault=='rope':model['server_args']+=['--rope-scaling','{}']
    else:model['server_args'][1]='81920'
    (path/'configs/model/model.json').write_bytes(m.raw(model))
    with pytest.raises(ValueError):job.derive(path,data,job.config['material'])


def test_pair_native_64k_context_is_explicit_operational_change_only(repo,monkeypatch):
    job,_=derived(repo,monkeypatch);path,data,_=repo
    model=m.read(path/'configs/model/model.json');model['server_args'][1]='65536'
    (path/'configs/model/model.json').write_bytes(m.raw(model))
    assert job.derive(path,data,job.config['material'])[1]['execution_plan']['planned_model_calls']==128


def test_pair_stage_and_register_use_shared_contract_and_complete_three_arm_recipe(repo,monkeypatch):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    from speech_aware_evidence_acquisition.core import config as composition
    job,_=derived(repo,monkeypatch);path,data,_=repo
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda *_:'a'*40)
    monkeypatch.setattr(composition,'compose',lambda *_:SimpleNamespace(config_hash='b'*64,values={'arm':'raw-speech-agent-native-tools'}))
    monkeypatch.setattr(job,'validate',lambda *args:16)
    lock=path/m.observation.LOCK;lock.parent.mkdir(parents=True,exist_ok=True);lock.write_bytes(b'{}\n')
    (path/'docs/exposure-ledger.md').write_text('old cost\n')
    stage=data/'work/codex/stage'
    record=job.stage(path,data,job.config['material'],stage)
    assert record['normal_generation_calls']==record['fit_calls']==64 and record['fallback_inclusive_cap']==128
    assert job.verify_stage(path,data,stage/'complete.json')==record
    supply=data/'prepared'/job.RUN/'phase-supply-v1.json';supply.parent.mkdir(parents=True);supply.write_bytes(b'{}\n')
    prep=data/'work/codex/prepared.json';prep.write_bytes(m.raw({'schema':'saea-gepa-pair-prepared-v1',
      'stage_sha256':m.sha(stage/'complete.json'),'rows':16,'run_id':job.RUN,'new_model_calls':0,
      'supply':m.pin(data,str(supply.relative_to(data)))}))
    result=job.register(path,data,stage/'complete.json',prep)
    pkg=path/'docs/readiness'/job.NAME
    assert result['readiness']=='held'
    assert m.read(pkg/'generation-plan-registered-v1.json')['execution_plan']['planned_model_calls']==128
    assert m.read(pkg/'fit-plan-registered-v1.json')['execution_plan']['planned_model_calls']==64
    assert m.read(pkg/'analysis-recipe-v1.json')['arms']==['parent','child','lock']
    assert (path/'docs/exposure-ledger.md').read_text().startswith('old cost\n')


def test_pair_registered_width_uses_config_not_historical96():
    from registered_workload import assert_execution_binding
    cfg=config();spec={'workload_kind':m.KIND,'job_config':cfg,'execution':cfg['execution'],'profile':'profile'}
    assert_execution_binding(spec,{'SAEA_ADMITTED_LEVEL':'16','SAEA_DEPLOYMENT_PROFILE':'profile'})
    with pytest.raises(ValueError):assert_execution_binding(spec,{'SAEA_ADMITTED_LEVEL':'96','SAEA_DEPLOYMENT_PROFILE':'profile'})
