import copy
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
import tfgrpo_h2_job as m
from test_tfgrpo_consumption import release


@pytest.fixture
def workload(release,monkeypatch):
    repo,data,params=release
    roster=m.read(ROOT/m.ROSTER);params['consumer_ids']=[r['sample_id'] for r in roster['items']]
    original=m.read(ROOT/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json')
    params['base_template']['relpath']=original['phase_material']['solver']['instruction_template']
    monkeypatch.setattr(m.consumption,'verify',lambda *a,**kw:None)
    monkeypatch.setattr(m.consumption,'verify_for_flight',lambda *a,**kw:'a'*64)
    derived=m.paired.derive(ROOT,data,params,fit_estimate={'slot_tokens':24576,'completion_ceiling':4096,
        'base_prompt_tokens':2000,'tokens_per_supplied_item':235},base_template_tokens=400,learned_template_tokens=500)
    base=derived['baseline'];exp=derived['experiment'];exp['service_session']=m.SESSION
    exp['cells'][0]['allowed_execution_stages']=['check','prepare-supply','worker','campaign']
    rels=[m.ROSTER,m.LOCK,'configs/dataset/slurp-speech-native-stage3-construction-r01c.json',m.AUTH,
        'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json']
    for rel in rels:
        p=repo/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
    ids=exp['populations'][0]['ids']
    supply=data/'prepared/supply.json';supply.parent.mkdir(parents=True)
    supply.write_bytes(m.raw({'output':{'facets':{},'values':{'perception':{i:[] for i in ids}}}}))
    base['prepared_phase_supply']=m.pin(data,'prepared/supply.json')
    for kind,doc in [('baseline',base),('experiment',exp)]:
        p=repo/f'configs/{kind}/{m.NAME}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(m.raw(doc))
    (repo/'docs/exposure-ledger.md').write_text('')
    spec={'cell':m.NAME,'experiment':m.NAME,'run_id':m.RUN,'attempt_id':'generation-v1',
        'fit_run_id':m.RUN+'-FIT','fit_attempt_id':'fit-v1','rows':312,
        'service_session':m.SESSION,'profile':m.PROFILE,'runtime_receipt':m.RECEIPT,
        'roster':m.pin(repo,m.ROSTER),'gain_lock':m.pin(repo,m.LOCK),'dataset':m.pin(repo,rels[2]),
        'split_identity_hash':roster['split_identity_hash'],'scientific_baseline_sha256':m.scientific_hash(base)}
    return repo,data,spec,exp,m.read(repo/rels[2]),derived


def test_full_h2_registers_narrow_workload(workload):
    import registered_workload
    repo,data,spec,exp,view,_=workload
    assert m.validate(repo,data,m.NAME,spec,exp,view)==312
    spec['workload_kind']=m.KIND
    assert registered_workload.validate_workload(repo,data,m.NAME,spec)==312


@pytest.mark.parametrize('fault',['subset','order','view','split','model','budget','fit','policy','seal','supply'])
def test_h2_refuses_changed_population_or_program(workload,monkeypatch,fault):
    repo,data,spec,exp,view,_=workload
    if fault=='subset':exp['populations'][0]['ids'].pop()
    elif fault=='order':exp['populations'][0]['ids'].reverse()
    elif fault=='view':view['items'].pop()
    elif fault=='split':spec['split_identity_hash']='0'*64
    elif fault=='model':exp['cells'][0]['model_fragment']='another'
    elif fault=='budget':exp['execution_plan']['planned_model_calls']=936
    elif fault=='seal':
        def reject(*a,**kw):raise ValueError('unsealed candidate')
        monkeypatch.setattr(m.consumption,'verify_for_flight',reject)
    elif fault=='supply':(data/'prepared/supply.json').write_text('{}')
    else:
        p=repo/f'configs/baseline/{m.NAME}.json';base=m.read(p)
        if fault=='fit':base['native_contact_fit']['contact_cap']=1
        else:base['native_episode']['roles']['solver']['template_variants']['arms'][1]['deliver']=False
        p.write_bytes(m.raw(base))
    with pytest.raises(ValueError):m.validate(repo,data,m.NAME,spec,exp,view)


def test_algorithm_resolves_registered_semantic_advantage(workload):
    from speech_aware_evidence_acquisition.core import stage3_algorithms
    from speech_aware_evidence_acquisition.core.base import STAGE3_ALGORITHMS
    exp=workload[3]
    record=STAGE3_ALGORITHMS.resolve(exp['stage3_algorithm']['id'])
    assert set(exp['stage3_algorithm']['operators_used']) <= set(record.operators)


def test_registration_binds_two_budgets_and_prespecified_read(workload,monkeypatch):
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    repo,data,spec,exp,view,derived=workload
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda _: 'a'*40)
    model=m.read(ROOT/'configs/model/bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-closemarker-port-v2.json')
    model['runtime_receipt']=m.RECEIPT
    mp=repo/f'configs/model/{m.MODEL}.json';mp.parent.mkdir(parents=True,exist_ok=True);mp.write_bytes(m.raw(model))
    bp=repo/f'configs/baseline/{m.NAME}.json';base=m.read(bp);supply=base.pop('prepared_phase_supply');bp.write_bytes(m.raw(base))
    staged={'evaluation':derived['evaluation'],'measurement':{'relpath':'work/codex/measurement.json','sha256':'c'*64}}
    monkeypatch.setattr(m,'verify_stage',lambda *a:staged)
    work=data/'work/codex/job';work.mkdir(parents=True)
    stage=work/'stage.json';stage.write_text('{}')
    prep=work/'prep.json';prep.write_bytes(m.raw({'schema':'saea-tfgrpo-h2-first-p-supply-v1','rows':312,
        'run_id':m.RUN,'stage_sha256':m.sha(stage),'supply':supply,'new_model_calls':0,'new_tokenizer_endpoint_calls':0}))
    result=m.register(repo,data,stage,prep)
    assert result['readiness']=='held' and result['rows']==312
    pkg=repo/'docs/readiness'/m.NAME
    assert m.read(pkg/'generation-plan-registered-v1.json')['execution_plan']['planned_model_calls']==1872
    assert m.read(pkg/'fit-plan-registered-v1.json')['execution_plan']['planned_model_calls']==936
    recipe=m.read(pkg/'analysis-recipe-v1.json')
    assert recipe['arms']==['learned','lock'] and recipe['delivered_arm']=='lock'
    assert '+1 percentage point' in recipe['evaluation']['acceptance']
    assert m.read(repo/f'configs/experiment/{m.NAME}.json')['cells'][0]['readiness']['state']=='held'
    assert ('| '+m.RUN+' |') in (repo/'docs/exposure-ledger.md').read_text()
    with pytest.raises(ValueError,match='existing H2'):m.register(repo,data,stage,prep)


def test_stage_rejects_unsealed_candidate_before_template_creation(workload,monkeypatch):
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    repo,data,*_=workload
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda _:'a'*40)
    for kind in ('baseline','experiment'):(repo/f'configs/{kind}/{m.NAME}.json').unlink()
    def reject(**kw):raise ValueError('candidate not FINALIZED')
    monkeypatch.setattr(m.consumption,'prepare',reject)
    with pytest.raises(ValueError,match='FINALIZED'):
        m.stage(repo,data,data/'candidate.json',data/'manifest.json',data/'work/codex/stage')
    assert not (repo/'configs'/m.TEMPLATE).exists()


def test_template_measurement_uses_pinned_local_tokenizer(tmp_path,monkeypatch):
    import transformers
    repo=tmp_path/'repo';data=tmp_path/'data';model=data/'models/tokenizer';model.mkdir(parents=True)
    files=[]
    for name in ('config.json','tokenizer_config.json','vocab.json','merges.txt'):
        path=model/name;path.write_text('{}')
        files.append({'path':str(path.relative_to(data)),'sha256':m.sha(path)})
    receipt=repo/'docs'/m.RECEIPT;receipt.parent.mkdir(parents=True)
    receipt.write_bytes(m.raw({'model':{'model_dir':str(model),'files':files}}))
    source=repo/m.ESTIMATE_SOURCE;source.parent.mkdir(parents=True)
    estimate={'slot_tokens':24576,'completion_ceiling':4096,'base_prompt_tokens':2000,'tokens_per_supplied_item':235}
    source.write_bytes(m.raw({'native_episode':{'consumption_module':{'supply_frame':{'fit_estimate':estimate}}}}))
    mp=repo/f'configs/model/{m.MODEL}.json';mp.parent.mkdir(parents=True)
    mp.write_bytes(m.raw({'server_args':['--max-model-len','24576'],'decoding_params':{'max_tokens':4096}}))
    params={}
    for key,text in [('base_template','a b'),('template','a b c')]:
        path=repo/'configs/probes'/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
        params[key]=m.pin(repo/'configs','probes/'+key)
    calls=[]
    class Tokenizer:
        def encode(self,text,*,add_special_tokens):
            assert add_special_tokens is False
            return text.split()
    def load(path,**kwargs):
        calls.append((path,kwargs));return Tokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',load)
    result=m.measure_templates(repo,data,params)
    assert result['token_counts']=={'base_template':2,'template':3}
    assert calls==[(model,{'local_files_only':True,'trust_remote_code':False})]
    assert result['full_prompt_measured'] is False and result['new_tokenizer_endpoint_calls']==0
    (model/'vocab.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='differs from resident'):m.measure_templates(repo,data,params)
    assert len(calls)==1


def test_explicit_h2_runtime_binding_preserves_historical_default(tmp_path):
    assert m.runtime_binding(tmp_path)['service_session']==m.SESSION
    spec={'model_fragment':m.MODEL,'service_session':'serving-new','profile':'new-profile','runtime_receipt':'receipts/new.json'}
    for rel,doc in [(m.RUNTIME_BINDING,spec),(f'configs/model/{m.MODEL}.json',{'runtime_receipt':spec['runtime_receipt']}),('configs/deployment/new-profile.json',{'model_fragment':m.MODEL})]:
        p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(m.raw(doc))
    assert m.runtime_binding(tmp_path)==spec
    p=tmp_path/'configs/deployment/new-profile.json';p.write_bytes(m.raw({'model_fragment':'different'}))
    with pytest.raises(ValueError,match='differs'):m.runtime_binding(tmp_path)


def test_successor_paired_geometry_is_compiled_from_measured_window(release,monkeypatch):
    repo,data,params=release
    params['consumer_ids']=[r['sample_id'] for r in m.read(ROOT/m.ROSTER)['items']]
    original=m.read(ROOT/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json')
    params['base_template']['relpath']=original['phase_material']['solver']['instruction_template']
    monkeypatch.setattr(m.consumption,'verify',lambda *a,**kw:None)
    derived=m.paired.derive(ROOT,data,params,fit_estimate={'slot_tokens':49152,'completion_ceiling':4096,'base_prompt_tokens':2000,'tokens_per_supplied_item':235},base_template_tokens=400,learned_template_tokens=500)
    assert derived['baseline']['serving_slot_tokens']==49152
    assert derived['evaluation']['rows']==312
