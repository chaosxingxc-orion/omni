import copy
from dataclasses import replace
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import tfgrpo_learning_job as m
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import ExperienceLearner,empty_experience


def source():
    parent=empty_experience('model');policy={'id':'actual-policy'};groups=[];custody=[]
    for i in range(3):
        q=f'slurp:train:{i}:audio';rolls=[]
        for k in range(4):
            rid=f'{q}-{k}';custody.append({'trajectory_id':rid})
            rolls.append({'trajectory_id':rid,'trajectory':'sealed response','critique':'official counts','reward':int(k==0),
                'model_id':'model','policy_sha256':m.sha(policy),'experience_sha256':parent['sha256']})
        groups.append({'query_id':q,'query':'fixed P context','answer':'gold','rollouts':rolls})
    half={'schema':'saea-tfgrpo-seed-half-v1','source_half':'h1','evaluation_half':'h2','rollout_kind':'shared-perception-adaptation',
        'model_id':'model','parent':parent,'policy':policy,'policy_sha256':m.sha(policy),'groups':groups,'custody':custody}
    roster={'items':[{'sample_id':groups[i]['query_id'],'stratum':'alarm'} for i in (0,2)]}
    return half,roster


def test_exact_error_intersection_contacts_and_no_parent_mutation():
    h,r=source();before=copy.deepcopy(h)
    roster,src,reg,plan=m.documents(h,r,runtime_identity={'model_fragment':'model'},roster_sha256='a'*64,provenance=[{'source':'test'}])
    assert len(src['batch']['groups'])==2 and len(reg['contacts'])==13
    assert plan.planned_model_calls==13 and h==before
    assert [g['query_id'] for g in src['batch']['groups']]==[x['sample_id'] for x in r['items']]
    assert reg['batch_sha256']==m.sha(src['batch'])
    assert 'protocol_hash' not in reg['plan']


@pytest.mark.parametrize('fault',['half','reward','custody','policy','model'])
def test_rejects_changed_source(fault):
    h,r=source()
    if fault=='half':h['source_half']='h2'
    elif fault=='reward':h['groups'][0]['rollouts'][0]['reward']=0
    elif fault=='custody':h['custody']=[]
    elif fault=='policy':h['policy']['id']='changed'
    else:h['groups'][0]['rollouts'][0]['model_id']='another'
    with pytest.raises(ValueError):m.select(h,r)


def test_job_batch_is_exact_real_learner_binding(tmp_path):
    h,r=source();_,src,reg,plan=m.documents(h,r,runtime_identity={'model_fragment':'model'},roster_sha256='a'*64,provenance=[{'source':'test'}])
    batch=src['batch'];calls=[]
    def caller(req):
        assert req['batch_sha256']==m.sha(batch)
        calls.append(req)
        text='<Experiences>Use evidence.</Experiences>' if req['stage']=='advantage' else '[]' if req['stage'].endswith('update') else 'Evidence summary.'
        return {'text':text,'model_id':'model','receipt_id':str(len(calls)),'usage':{'calls':1,'prompt_tokens':10,'completion_tokens':10}}
    learner=ExperienceLearner(tmp_path,model_id='model',metered_call=caller,learning_config_sha256=reg['learning_config_sha256'],**batch['bounds'])
    candidate=learner.learn(batch['groups'],batch['parent'],policy_sha256=batch['policy_sha256'],rollout_kind=batch['rollout_kind'],**batch['common'])
    assert candidate['planned_learning_calls']==13 and candidate['mixed_groups']==2
    assert m.read(tmp_path/'batch.json')==batch


def test_register_exact_protocol_budgets_and_source(tmp_path,monkeypatch):
    import shutil
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda _: 'a'*40)
    repo=tmp_path/'repo';data=tmp_path/'data';job=data/'work/codex/job';job.mkdir(parents=True)
    h,r=source();h['model_id']='bf16-thinking-real500-tp4-s128-v1';h['parent']=empty_experience(h['model_id'])
    for g in h['groups']:
        for x in g['rollouts']:x.update(model_id=h['model_id'],experience_sha256=h['parent']['sha256'])
    _,roster=m.select(h,r);roster_sha=__import__('hashlib').sha256(m.raw(roster)).hexdigest()
    roster,src,reg,plan=m.documents(h,r,runtime_identity={'model_fragment':h['model_id']},roster_sha256=roster_sha,provenance=[{'source':'test'}])
    docs={'construction-roster.json':roster,'source.json':src,'registration.json':reg}
    for name,doc in docs.items():(job/name).write_bytes(m.raw(doc))
    (job/'complete.json').write_bytes(m.raw({'artifacts':{n:m.file_sha(job/n) for n in docs},'batch_sha256':m.sha(src['batch']),
        'groups':2,'generation_calls':13,'fit_calls':13}))
    rels=['configs/baseline/system-group-update-text-v1.json','configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json','configs/experiment/slurp-tfrl-hardset-a0-observation-r1-v1.json',
        'configs/dataset/slurp-speech-native-stage3-construction-r01c.json']
    for rel in rels:
        p=repo/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(m.ROOT/rel,p)
    model=m.read(m.ROOT/'configs/model/bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-closemarker-port-v2.json')
    p=repo/f"configs/model/{h['model_id']}.json";p.parent.mkdir(parents=True);p.write_bytes(m.raw(model))
    (repo/'docs').mkdir();(repo/'docs/exposure-ledger.md').write_text('')
    monkeypatch.setattr(m,'verify_job',lambda *_:(m.read(job/'complete.json'),reg,roster))
    result=m.register(repo,data,job)
    assert result['calls']==13 and result['readiness']=='held'
    exp=m.read(repo/f'configs/experiment/{m.NAME}.json')
    assert exp['tfgrpo_learning_registration']['batch_sha256']==m.sha(src['batch'])
    assert exp['tfgrpo_contact_fit']['contact_cap']==13
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.rails.adapters.runtime_view import configured_geometry
    config=compose(repo,h['model_id'],'slurp-speech-native-stage3-construction-r01c',m.NAME,m.NAME)
    assert configured_geometry(config.values,defaults=(8,8192,64))['slot_tokens']==24576
    assert m.RUN+'-FIT' in (repo/'docs/exposure-ledger.md').read_text()
    with pytest.raises(ValueError,match='existing'):m.register(repo,data,job)


def prepared_fixture(tmp_path,monkeypatch):
    repo=tmp_path/'repo';data=tmp_path/'data';job=data/'work/codex/job'
    repo.mkdir();job.mkdir(parents=True)
    model_path=repo/'configs/model/model.json';model_path.parent.mkdir(parents=True)
    model_path.write_bytes(m.raw({'decoding_params':{'max_tokens':4096,'temperature':0}}))
    h,r=source();identity={'model_fragment':'model','host':'127.0.0.1','port':8200}
    exports=data/'exports';exports.mkdir()
    (exports/'h1.json').write_bytes(m.raw(h));(repo/'roster.json').write_bytes(m.raw(r))
    (exports/'complete.json').write_bytes(m.raw({'schema':'saea-tfgrpo-seed-groups-v1','artifacts':{'h1.json':m.file_sha(exports/'h1.json')}}))
    provenance=[m.pin(exports/'complete.json',data,'data'),m.pin(exports/'h1.json',data,'data'),m.pin(repo/'roster.json',repo,'repository')]
    _,roster=m.select(h,r);roster_sha=__import__('hashlib').sha256(m.raw(roster)).hexdigest()
    roster,src,reg,_=m.documents(h,r,runtime_identity=identity,roster_sha256=roster_sha,provenance=provenance)
    docs={'construction-roster.json':roster,'source.json':src,'registration.json':reg,'runtime-identity.json':identity}
    for n,d in docs.items():(job/n).write_bytes(m.raw(d))
    complete={'artifacts':{n:m.file_sha(job/n) for n in docs},'inputs':provenance,
        'runtime_identity_sha256':m.sha(identity),'batch_sha256':m.sha(src['batch']),'groups':2,'generation_calls':13,'fit_calls':13}
    (job/'complete.json').write_bytes(m.raw(complete))
    monkeypatch.setattr(m,'deployment_identity',lambda *_:identity)
    return repo,data,job


def test_verify_exact_derivation(tmp_path,monkeypatch):
    repo,data,job=prepared_fixture(tmp_path,monkeypatch)
    assert m.verify_job(repo,data,job)[0]['groups']==2


def test_learning_retains_real_fragment_thinking_budget_and_answer_reserve():
    model=m.read(m.ROOT/'configs/model/bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-closemarker-port-v2.json')
    decoding=m.learning_decoding(model)
    assert decoding['thinking_budget_tokens']==3072
    assert decoding['chat_template_kwargs']=={'enable_thinking':True}
    assert decoding['max_tokens']-decoding['thinking_budget_tokens']==1024
    h,r=source()
    _,src,reg,plan=m.documents(h,r,runtime_identity={'model_fragment':'model'},roster_sha256='a'*64,
        provenance=[{'source':'test'}],model_values=model)
    assert reg['decoding_params']==decoding
    assert src['batch']['learning_config_sha256']==reg['learning_config_sha256']


@pytest.mark.parametrize('fault',['source','deployment','budget','missing_artifact','provenance'])
def test_verify_rejects_self_consistent_forgery(tmp_path,monkeypatch,fault):
    repo,data,job=prepared_fixture(tmp_path,monkeypatch)
    complete=m.read(job/'complete.json');reg=m.read(job/'registration.json');src=m.read(job/'source.json')
    if fault=='source':
        src['batch']['groups'][0]['rollouts'][0]['trajectory']='invented successful trace'
        reg['batch_sha256']=m.sha(src['batch']);complete['batch_sha256']=reg['batch_sha256']
        (job/'source.json').write_bytes(m.raw(src));(job/'registration.json').write_bytes(m.raw(reg))
        for n in ('source.json','registration.json'):complete['artifacts'][n]=m.file_sha(job/n)
    elif fault=='deployment':monkeypatch.setattr(m,'deployment_identity',lambda *_:{'model_fragment':'changed'})
    elif fault=='budget':complete['fit_calls']=12
    elif fault=='missing_artifact':complete['artifacts'].pop('runtime-identity.json')
    else:(data/'exports/h1.json').write_text('{}')
    (job/'complete.json').write_bytes(m.raw(complete))
    with pytest.raises(ValueError):m.verify_job(repo,data,job)


def test_generation_release_requires_sealed_exact_a0(tmp_path):
    artifact=tmp_path/'manifest.json';artifact.write_text('{}')
    release=tmp_path/'release.json'
    doc={'schema':'saea-generation-released-v1','generation_sealed':True,'generation_lease_released':True,
        'run_id':'SAEA-SLURP-TFRL-HARDSET-A0-OBSERVATION-R1-V1','artifacts':{'manifest':{'path':str(artifact),'sha256':m.file_sha(artifact)}}}
    release.write_bytes(m.raw(doc));assert m.verify_release(release,tmp_path)==doc
    artifact.write_text('changed')
    with pytest.raises(ValueError):m.verify_release(release,tmp_path)


def test_admission_binds_decision_and_refreshes_final_protocol(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.core import config
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    from speech_aware_evidence_acquisition.core.session import serving
    from tfrl_current_generation import SESSION
    repo,data,job=prepared_fixture(tmp_path,monkeypatch)
    complete,reg,roster=m.verify_job(repo,data,job)
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda _: 'a'*40)
    monkeypatch.setattr(m,'verify_job',lambda *_:(complete,reg,roster))
    pkg=repo/'docs/readiness'/m.NAME;pkg.mkdir(parents=True)
    exp={'cells':[{'cell_id':m.NAME,'dataset_fragment':'dataset','readiness':{'state':'held','requirements':[]}}],
        'tfgrpo_learning_registration':reg}
    ep=repo/f'configs/experiment/{m.NAME}.json';ep.parent.mkdir(parents=True);ep.write_bytes(m.raw(exp))
    for kind in ('generation','fit'):
        (pkg/(kind+'-plan.json')).write_bytes(m.raw({'execution_plan':{'protocol_hash':'old'},'config_sha256':'old'}))
    ledger=repo/'docs/exposure-ledger.md';ledger.write_text(m.file_sha(pkg/'generation-plan.json'))
    authority=repo/m.AUTH;authority.parent.mkdir(parents=True);authority.write_text('Authorized finite learning.')
    receipt=repo/'receipt.json';receipt.write_bytes(m.raw({'deployment':{'instances':[]}}))
    identity=m.read(job/'runtime-identity.json')
    monkeypatch.setattr(serving,'find_resident_session',lambda _:(data,{'session_id':SESSION,**identity}))
    monkeypatch.setattr(serving,'assert_identity_matches',lambda *_:None)
    monkeypatch.setattr(config,'read_fragment',lambda *_:{})
    monkeypatch.setattr(config,'resolve_runtime_receipt_path',lambda *_:receipt)
    monkeypatch.setattr(config,'compose',lambda *_:SimpleNamespace(config_hash='b'*64,values={'arm':'prepared-context-text'}))
    def mint(root,data,coordinator,dry,*,family,tool):
        draft=m.read(pkg/'root-admission-draft-v1.json');draft['status']='decided'
        target=Path(data)/family.DECISION_RELPATH;target.parent.mkdir(parents=True);target.write_bytes(m.raw(draft))
        return {'out':str(target)}
    monkeypatch.setitem(sys.modules,'thinking_system_mint_receipt',SimpleNamespace(mint=mint))
    result=m.admit(repo,data,job)
    assert result['new_model_calls']==0
    ready=m.read(ep)['cells'][0]['readiness']
    assert ready['state']=='ready' and ready['requirements'][0]['kind']=='decision'
    plan=m.read(pkg/'generation-plan.json')
    assert plan['config_sha256']=='b'*64 and plan['execution_plan']['protocol_hash']!='old'
    assert m.file_sha(pkg/'generation-plan.json') in ledger.read_text()


def test_successor_run_identity_rederived_without_changing_source_policy():
    h,r=source();before=copy.deepcopy(h)
    run='SAEA-TFGRPO-CORRECTED-V1'
    roster,src,reg,plan=m.documents(h,r,runtime_identity={'model_fragment':'model'},roster_sha256='a'*64,provenance=[{'source':'test'}],run_id=run)
    assert plan.run_id==run and reg['plan']['run_id']==run and roster['roster_id']==run
    assert h==before and src['batch']['policy_sha256']==h['policy_sha256']


def test_job_import_uses_verified_ids_and_rederives_all_budgets(monkeypatch):
    from speech_aware_evidence_acquisition.rails import tfgrpo_summary_import as module
    h,r=source();tid=h['groups'][0]['rollouts'][0]['trajectory_id'];pin={'relpath':'import.json','sha256':'b'*64}
    def verified(spec,**kwargs):
        assert spec==pin and kwargs['data_root']==Path('/data')
        assert kwargs['groups']==[h['groups'][0],h['groups'][2]]
        return {tid:'Observed summary'},{}
    monkeypatch.setattr(module,'load_import',verified)
    _,src,reg,plan=m.documents(h,r,runtime_identity={'model_fragment':'model'},roster_sha256='a'*64,provenance=[{'source':'test'}],summary_import=pin,data_root=Path('/data'))
    assert plan.planned_model_calls==12 and len(reg['contacts'])==12 and src['batch']['bounds']['max_calls']==12
    assert src['batch']['summary_import']==pin and reg['batch_sha256']==m.sha(src['batch'])
    assert not any(x['stage']=='summary' and x['identity']==tid for x in reg['contacts'])


def test_missing_multi_instance_profile_rejected_before_dispatch(tmp_path,monkeypatch):
    from speech_aware_evidence_acquisition.core import config
    monkeypatch.setattr(config,'read_fragment',lambda *a:{'model_materialization_root':str(tmp_path)})
    receipt=tmp_path/'receipt.json';receipt.write_bytes(m.raw({'deployment':{'instances':[{'instance_id':'one'},{'instance_id':'two'}]}}))
    monkeypatch.setattr(config,'resolve_runtime_receipt_path',lambda *a:receipt)
    monkeypatch.delenv('SAEA_DEPLOYMENT_PROFILE',raising=False)
    with pytest.raises(ValueError,match='SAEA_DEPLOYMENT_PROFILE'):m.require_learning_deployment(tmp_path,'model')
