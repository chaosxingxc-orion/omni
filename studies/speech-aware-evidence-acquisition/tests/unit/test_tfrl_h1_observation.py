import copy
from pathlib import Path
import shutil
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
import tfrl_h1_observation as m

BINDING={'model_fragment':m.MODEL,'service_session':'serving-a0-remaining353-selected-resident-v1',
         'profile':'qwen3-omni-real500-2xtp4-s128-b2048-v1','runtime_receipt':'receipts/runtime-a0-remaining353-selected-resident-v1.json'}
if (ROOT/'configs/evaluation/tfgrpo-h2-runtime-v1.json').exists():
    BINDING=m.read(ROOT/'configs/evaluation/tfgrpo-h2-runtime-v1.json')


def test_single_round_reuses_exact_h1_dataset_without_learning_or_h2_scores(monkeypatch):
    monkeypatch.setattr(m,'runtime_binding',lambda *args:m.read(ROOT/f'configs/model/{m.MODEL}.json'))
    before=m.sha(ROOT/f'configs/dataset/{m.DATASET}.json')
    docs=[m.derive(ROOT,r,BINDING) for r in (1,)]
    for r,(base,exp,h1,view) in enumerate(docs,1):
        name,run=m.identity(r)
        assert len(exp['populations'][0]['ids'])==313
        assert exp['populations'][0]['ids']==[x['sample_id'] for x in m.read(ROOT/m.H1)['items']]
        assert base['native_contact_fit']=={**base['native_contact_fit'],'contact_cap':626,'run_id':run+'-FIT'}
        assert exp['execution_plan']['planned_model_calls']==1252
        assert base['native_episode']['consumption_module']['columns']==[]
        assert 'decoding_params' not in base
        assert view==m.read(ROOT/f'configs/dataset/{m.DATASET}.json')
        assert exp['cells'][0]['readiness']['state']=='held'
    assert m.sha(ROOT/f'configs/dataset/{m.DATASET}.json')==before
    for r in (0,2,3,True):
        with pytest.raises(ValueError):m.identity(r)


@pytest.fixture
def repo(tmp_path):
    repo=tmp_path/'repo';data=tmp_path/'data';repo.mkdir();data.mkdir()
    rels=[m.H1,m.H2,'docs/receipts/rosters/slurp-stage3-construction-r01c.json',
          f'configs/dataset/{m.DATASET}.json',m.SOURCE_BASE,m.SOURCE_EXP,m.LOCK,
          f'configs/model/{m.MODEL}.json',f"configs/deployment/{BINDING['profile']}.json",
          'docs/'+BINDING['runtime_receipt']]
    for rel in rels:
        p=repo/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
    profile=m.read(repo/f"configs/deployment/{BINDING['profile']}.json");profile['model_fragment']=m.MODEL
    (repo/f"configs/deployment/{BINDING['profile']}.json").write_bytes(m.raw(profile))
    # Module compilation binds immutable profiles/probes from the real tree.
    for directory in ('configs/probes','configs/modules','configs/consumption'):
        source=ROOT/directory
        if source.exists():shutil.copytree(source,repo/directory,dirs_exist_ok=True)
    (repo/'docs/exposure-ledger.md').write_text('historical costs retained\n')
    return repo,data


@pytest.mark.parametrize('fault',['h1_order','h2_overlap','view_missing','view_audio','h2_audio'])
def test_population_refuses_changed_or_leaking_membership(repo,fault):
    repo,_=repo
    rel=m.H1 if fault=='h1_order' else m.H2 if fault=='h2_overlap' else f'configs/dataset/{m.DATASET}.json'
    d=m.read(repo/rel)
    if fault=='h1_order':d['items'].reverse()
    elif fault=='h2_overlap':d['items'][0]=m.read(repo/m.H1)['items'][0]
    elif fault=='view_missing':d['items'].pop()
    else:
        h1=m.read(repo/m.H1)['items'][0]['sample_id'];h2=m.read(repo/m.H2)['items'][0]['sample_id']
        by={x['sample_id']:x for x in d['items']}
        if fault=='view_audio':by[h1]['audio_seconds']+=1
        else:by[h2]['audio_sha256']=by[h1]['audio_sha256']
    (repo/rel).write_bytes(m.raw(d))
    with pytest.raises(ValueError):m.population(repo)


def test_stage_does_not_consult_scores_or_answers(repo,monkeypatch):
    repo,data=repo
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda p:'a'*40)
    real_derive=m.derive
    monkeypatch.setattr(m,'runtime_binding',lambda *args:m.read(ROOT/f'configs/model/{m.MODEL}.json'))
    # Actual compiled material uses repo source; only parent roots differ.
    monkeypatch.setattr(m,'derive',lambda repo,round_index,binding:real_derive(ROOT,round_index,binding))
    result=m.stage(repo,data,1,BINDING,data/'work/codex/stage')
    assert result['test_labels_read'] is False and result['h2_scores_read'] is False
    assert result['new_model_calls']==0 and result['prior_costs_reused_or_reset'] is False
    assert result['normal_generation_calls']==result['fit_calls']==626
    assert result['fallback_inclusive_cap']==1252
    assert (repo/'docs/exposure-ledger.md').read_text()=='historical costs retained\n'
    with pytest.raises(ValueError):m.stage(repo,data,1,BINDING,data/'work/codex/again')


def test_reject_existing_spend_instead_of_restarting(repo):
    repo,data=repo;_,run=m.identity(1)
    (data/'attempts'/run).mkdir(parents=True)
    with pytest.raises(ValueError,match='spend'):m.require_unspent(repo,data,run)


def test_runtime_binding_rejects_model_or_receipt_drift(repo):
    repo,_=repo
    assert m.runtime_binding(repo,BINDING)['decoding_params']['max_tokens']==4096
    for key,value in [('model_fragment','other'),('runtime_receipt','receipts/other.json'),('profile','../other')]:
        with pytest.raises(ValueError):m.runtime_binding(repo,{**BINDING,key:value})


@pytest.fixture
def registered(repo,monkeypatch):
    repo,data=repo
    real_derive=m.derive
    monkeypatch.setattr(m,'runtime_binding',lambda *args:m.read(ROOT/f'configs/model/{m.MODEL}.json'))
    base,exp,roster,view=real_derive(ROOT,1,BINDING)
    expected=copy.deepcopy((base,exp,roster,view))
    monkeypatch.setattr(m,'derive',lambda *args:copy.deepcopy(expected))
    name,run=m.identity(1);actual=copy.deepcopy(base)
    supply=data/'prepared/supply.json';supply.parent.mkdir(parents=True)
    supply.write_bytes(m.raw({'output':{'facets':{},'values':{'perception':{r['sample_id']:[] for r in roster['items']}}}}))
    actual['prepared_phase_supply']=m.pin(data,'prepared/supply.json')
    for kind,doc in [('baseline',actual),('experiment',exp)]:
        (repo/f'configs/{kind}/{name}.json').write_bytes(m.raw(doc))
    spec={'round':1,'cell':name,'experiment':name,'workload_kind':m.KIND,'run_id':run,'attempt_id':'generation-v1',
          'fit_run_id':run+'-FIT','fit_attempt_id':'fit-v1','rows':313,'runtime_binding':BINDING,'model_fragment':m.MODEL,
          'service_session':BINDING['service_session'],'profile':BINDING['profile'],'runtime_receipt':BINDING['runtime_receipt'],
          'dataset':m.pin(repo,f'configs/dataset/{m.DATASET}.json'),'roster':m.pin(repo,m.H1),
          'split_identity_hash':roster['split_identity_hash'],'scientific_baseline_sha256':m.scientific_hash(actual),'gain_lock':m.pin(repo,m.LOCK)}
    return repo,data,name,spec,exp,view,base


def test_shared_validator_routes_complete_h1(registered):
    import registered_workload
    repo,data,name,spec,exp,view,_=registered
    assert m.validate(repo,data,name,spec,exp,view)==313
    assert registered_workload.validate_workload(repo,data,name,spec)==313


@pytest.mark.parametrize('fault',['subset','order','budget','round','split','model','benefit','baseline','supply'])
def test_validator_rejects_drift(registered,fault):
    repo,data,name,spec,exp,view,_=registered
    if fault=='subset':exp['populations'][0]['ids'].pop()
    elif fault=='order':exp['populations'][0]['ids'].reverse()
    elif fault=='budget':exp['execution_plan']['planned_model_calls']=1756
    elif fault=='round':spec['round']=2
    elif fault=='split':spec['split_identity_hash']='x'
    elif fault=='model':spec['model_fragment']='other'
    elif fault=='benefit':exp['stage3_measurement']['claims_benefit']=True
    elif fault=='baseline':
        p=repo/f'configs/baseline/{name}.json';base=m.read(p);base['native_contact_fit']['contact_cap']=878;p.write_bytes(m.raw(base))
    else:(data/'prepared/supply.json').write_text('{}')
    with pytest.raises(ValueError):m.validate(repo,data,name,spec,exp,view)


def test_registration_keeps_old_costs_and_held_bounded_plans(registered,monkeypatch):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    from speech_aware_evidence_acquisition.core import config
    repo,data,name,spec,exp,view,base=registered
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda p:'a'*40)
    monkeypatch.setattr(config,'compose',lambda *a:SimpleNamespace(config_hash='c'*64,values={'arm':base['arm']}))
    monkeypatch.setattr(contracts.ExecutionPlan,'validate',lambda self,**kwargs:self)
    bp=repo/f'configs/baseline/{name}.json';bp.write_bytes(m.raw(base))
    ep=repo/f'configs/experiment/{name}.json'
    work=data/'work/codex/job';work.mkdir(parents=True)
    stage=work/'stage.json';stage.write_bytes(m.raw({'schema':'saea-h1-observation-stage-v1','round':1,'runtime_binding':BINDING,
        'dataset':m.pin(repo,f'configs/dataset/{m.DATASET}.json'),'h1':m.pin(repo,m.H1),'h2_metadata':m.pin(repo,m.H2),
        'source_baseline':m.pin(repo,m.SOURCE_BASE),'output_hashes':{str(p.relative_to(repo)):m.sha(p) for p in (bp,ep)}}))
    prep=work/'prepared.json';prep.write_bytes(m.raw({'schema':'saea-h1-observation-prepared-v1','stage_sha256':m.sha(stage),
        'rows':313,'run_id':spec['run_id'],'supply':m.pin(data,'prepared/supply.json')}))
    out=m.register(repo,data,stage,prep)
    assert out['readiness']=='held'
    pkg=repo/'docs/readiness'/name
    assert m.read(pkg/'generation-plan-registered-v1.json')['execution_plan']['planned_model_calls']==1252
    assert m.read(pkg/'fit-plan-registered-v1.json')['execution_plan']['planned_model_calls']==626
    assert m.read(pkg/'protocol-registered-v1.json')['total_scheduled_rounds']==1
    assert (repo/'docs/exposure-ledger.md').read_text().startswith('historical costs retained\n')
    assert m.read(ep)['cells'][0]['readiness']['state']=='held'
    with pytest.raises(ValueError,match='existing'):m.register(repo,data,stage,prep)


def test_current_context_survives_legacy_baseline_override(monkeypatch):
    from speech_aware_evidence_acquisition.rails.adapters.runtime_view import configured_geometry
    model=copy.deepcopy(m.read(ROOT/f'configs/model/{m.MODEL}.json'))
    model['serving_slot_tokens']=49152
    idx=model['server_args'].index('--max-model-len')+1
    model['server_args'][idx]='49152'
    monkeypatch.setattr(m,'runtime_binding',lambda *args:model)
    base,_,_,_=m.derive(ROOT,1,BINDING)
    assert configured_geometry({**model,**base},defaults=(8,8192,64))['slot_tokens']==49152
    original=m.read(ROOT/m.SOURCE_BASE)
    assert base['native_episode']==original['native_episode']
    assert base['scorers']==original['scorers']
