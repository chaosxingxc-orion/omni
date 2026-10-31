import copy
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/analysis'),str(ROOT/'scripts/flights/nx')]
import tfrl_current_generation as m
import tfrl_v2_sampler as sampler
from registered_workload import validate_workload
from test_decoding_variants import _decoding_block
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks


def put(root,rel,obj):
    p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj));return {'relpath':rel,'sha256':m.sha(p)}


@pytest.mark.parametrize('fault',[None,'decoding','ready','stages'])
def test_stage_changes_only_resident_and_algorithm(fault):
    exp={'cells':[{'model_fragment':'old','readiness':{'state':'held'},'allowed_execution_stages':['check']}],
        'service_session':'old','stage3_algorithm':{'id':'none'},'population':{'count':313}}
    old={'runtime_receipt':'old','temperature':.7};new={'runtime_receipt':m.RECEIPT,'temperature':.7}
    original=copy.deepcopy(exp)
    if fault=='decoding':new['temperature']=.8
    elif fault=='ready':exp['cells'][0]['readiness']['state']='ready'
    elif fault=='stages':exp['cells'][0]['allowed_execution_stages'].append('worker')
    if fault:
        with pytest.raises(ValueError):m.stage_documents(exp,old,new)
    else:
        doc=m.stage_documents(exp,old,new)
        assert exp==original and doc['population']==exp['population']
        assert doc['stage3_algorithm']['id']=='shared-rollout'
        assert doc['cells'][0]['readiness']['state']=='held'


@pytest.mark.parametrize('half,n',[('h1',313),('h2',312)])
@pytest.mark.parametrize('fault',[None,'duplicate','outside','media','seed','columns','calls','algorithm','pin','run'])
def test_real_half_guards(tmp_path,monkeypatch,half,n,fault):
    rel='configs/baseline/slurp-native-default-plan-lock-baseline-v1.json'
    put(tmp_path,rel,json.loads((ROOT/rel).read_bytes()))
    name=f'slurp-tfrl-current-seed-i1-{half}-v1';run=f'SAEA-SLURP-TFRL-CURRENT-SEED-I1-{half.upper()}-V1'
    rows=[{'sample_id':f'slurp:train:fixture{i}','audio_seconds':2.,'audio_sha256':str(i),'media_relpath':f'a/slurp_real/{i}.wav'} for i in range(625)]
    selected=rows[:n];roster={'roster_id':sampler.cm.HALF_ROSTER_ID['R01C-'+half.upper()],'split_identity_hash':'a'*64,'items':selected}
    rp=put(tmp_path,'roster.json',roster)
    plan=sampler.build_plan(roster,'R01C-'+half.upper(),1,roster_file_sha256=rp['sha256'],execution_run_id=run,execution_attempt_id='generation-v1')
    block=_decoding_block();block['columns']=sampler.execution_columns(plan)
    base=cm.compile_module(block,carrier_module_hooks()).values();base['native_episode']['consumption_module']=block
    base['native_contact_fit']={'run_id':run+'-FIT','attempt_id':'fit-v1','contact_cap':7*n}
    lock=put(tmp_path,m.LOCK,{'status':'locked_full_train_fold_b'});monkeypatch.setattr(m,'LOCK_SHA',lock['sha256'])
    exp={'cells':[{'run_id':run,'attempt_id':'generation-v1','dataset_fragment':'view','baseline_fragment':name}],
        'population':{'count':n,'dataset_fragment':'view','roster_receipt':'roster.json','split_identity_hash':'a'*64},
        'populations':[{'dataset_fragment':'view','ids':[r['sample_id'] for r in selected]}],
        'execution_plan':{'planned_model_calls':14*n,'planned_speech_audio_seconds':28*n},
        'stage3_algorithm':{'id':'shared-rollout','operators_used':['rollout','scoring']}}
    spec={'cell':name,'experiment':name,'workload_kind':'tfrl-real-seed-half','run_id':run,'attempt_id':'generation-v1','rows':n,
        'split_identity_hash':'a'*64,'gain_lock':lock,'roster':rp,'fit_run_id':run+'-FIT','fit_attempt_id':'fit-v1'}
    if fault=='duplicate':exp['populations'][0]['ids'][0]=exp['populations'][0]['ids'][1]
    elif fault=='outside':rows[0]=dict(rows[0],sample_id='slurp:train:other')
    elif fault=='media':rows[0]=dict(rows[0],media_relpath='synthetic/file.wav')
    elif fault=='seed':plan['row_plan'][0]['contacts'][1]['seed']+=1
    elif fault=='columns':block['columns'][0]['decoding']['temperature']=.8
    elif fault=='calls':base['native_contact_fit']['contact_cap']+=1
    elif fault=='algorithm':exp['stage3_algorithm']['id']='none'
    elif fault=='pin':spec['roster']['sha256']='0'*64
    elif fault=='run':plan['run_id']='other'
    spec['sampling_plan']=put(tmp_path,'plan.json',plan)
    put(tmp_path,f'configs/baseline/{name}.json',base)
    put(tmp_path,f'configs/experiment/{name}.json',exp)
    spec['dataset']=put(tmp_path,'configs/dataset/view.json',{'count':625,'items':rows})
    if fault:
        with pytest.raises((ValueError,sampler.SamplerError)):validate_workload(tmp_path,tmp_path,name,spec)
    else:assert validate_workload(tmp_path,tmp_path,name,spec)==n


@pytest.mark.parametrize('iteration',[2,3])
@pytest.mark.parametrize('fault',[None,'bank_file','self_bank','allocation','decision','source_result','wrong_label'])
def test_banked_workload_full_roster_bank_and_first_gate(tmp_path,monkeypatch,iteration,fault):
    from test_tfrl_v2_consumption import make_bank,SCOPE,H1,H2,rail
    from speech_aware_evidence_acquisition.rails.tfrl_supply_allocation import declaration,KEY
    from tfrl_current_next_plan import first_construction_decision
    from tfrl_recovered_consumers import POOL
    rel='configs/baseline/slurp-native-default-plan-lock-baseline-v1.json'
    put(tmp_path,rel,json.loads((ROOT/rel).read_bytes()))
    n=313;name=f'slurp-tfrl-current-bank-i{iteration}-h1-v1';run=f'SAEA-SLURP-TFRL-CURRENT-BANK-I{iteration}-H1-V1'
    rows=[{'sample_id':f'slurp:train:fixture{i}','audio_seconds':2.,'audio_sha256':str(i),'media_relpath':f'a/slurp_real/{i}.wav'} for i in range(625)]
    roster={'roster_id':sampler.cm.HALF_ROSTER_ID[H1],'split_identity_hash':SCOPE[H1],'items':rows[:n]}
    rp=put(tmp_path,'roster.json',roster)
    path,receipt=make_bank(tmp_path,kind='half',sources=(H2,),source_rows=('slurp:train:fixture624',),iteration=iteration-1)
    plan=sampler.build_plan(roster,H1,iteration,bank_receipt=receipt,roster_file_sha256=rp['sha256'],execution_run_id=run,execution_attempt_id='generation-v1')
    block=_decoding_block();block['columns']=sampler.execution_columns(plan)
    allocation=declaration(block['decoding_frame']['max_supplied_items'])
    block[KEY]=allocation
    block['tfrl_v2_bank']=rail.declaration_block(iteration=iteration-1,consumer=H1,bank_kind='half',
        receipt_relpath=str(path.relative_to(tmp_path)),receipt_sha=rail.receipt_sha256(receipt),source_scope=SCOPE)
    plan['banks'].update(supply_allocation=allocation,supply_bound_items=allocation['candidate_max_items'])
    base=cm.compile_module(block,carrier_module_hooks()).values();base['native_episode']['consumption_module']=block
    base['native_contact_fit']={'run_id':run+'-FIT','attempt_id':'fit-v1','contact_cap':7*n}
    lock=put(tmp_path,m.LOCK,{'status':'locked_full_train_fold_b'});monkeypatch.setattr(m,'LOCK_SHA',lock['sha256'])
    sources={}
    for h in ('h1','h2'):
        pin=put(tmp_path,f'analysis/{h}/complete.json',{'source':h})
        sources[h]={'complete_relpath':pin['relpath'],'complete_sha256':pin['sha256']}
    pooled={'rows':625,'iteration':1,'vote_minus_lock_pp':1.1,'threshold_pp':1.,'construction_gate_passed':True,'sources':sources}
    pp=put(tmp_path,f'analysis/{POOL}/complete.json',pooled)
    decision={'schema':'saea-tfrl-first-construction-decision-v1','rows':625,'generation_authorized':False,
        **first_construction_decision(pooled),'pool_sha256':pp['sha256'],'sources':sources}
    exp={'cells':[{'run_id':run,'attempt_id':'generation-v1','dataset_fragment':'view','baseline_fragment':name}],
        'population':{'count':n,'dataset_fragment':'view','roster_receipt':'roster.json','split_identity_hash':SCOPE[H1]},
        'populations':[{'dataset_fragment':'view','ids':[r['sample_id'] for r in rows[:n]]}],
        'execution_plan':{'planned_model_calls':14*n,'planned_speech_audio_seconds':28*n},
        'stage3_algorithm':{'id':'tfrl-v2','operators_used':['rollout','scoring','context_application']}}
    spec={'cell':name,'experiment':name,'workload_kind':'tfrl-real-banked-half','iteration':iteration,
        'run_id':run,'attempt_id':'generation-v1','rows':n,'split_identity_hash':SCOPE[H1],
        'gain_lock':lock,'roster':rp,'fit_run_id':run+'-FIT','fit_attempt_id':'fit-v1'}
    if fault=='bank_file':(path.parent/'rule-bank.json').write_text('{}')
    elif fault=='self_bank':block['tfrl_v2_bank']['consumer']=H2
    elif fault=='allocation':plan['banks']['supply_bound_items']+=1
    elif fault=='decision':decision['preparation_gate_passed']=False
    elif fault=='source_result':(tmp_path/'analysis/h1/complete.json').write_text('{}')
    elif fault=='wrong_label':exp['stage3_algorithm']['id']='shared-rollout'
    spec['first_construction_decision']=put(tmp_path,'work/codex/decision.json',decision)
    spec['sampling_plan']=put(tmp_path,'plan.json',plan)
    put(tmp_path,f'configs/baseline/{name}.json',base)
    put(tmp_path,f'configs/experiment/{name}.json',exp)
    spec['dataset']=put(tmp_path,'configs/dataset/view.json',{'count':625,'items':rows})
    if fault:
        with pytest.raises((ValueError,rail.TfrlV2BankRefusal)):validate_workload(tmp_path,tmp_path,name,spec)
    else:
        assert validate_workload(tmp_path,tmp_path,name,spec)==n
        # Exercise the real registration writer after the same bank/roster guard.
        import speech_aware_evidence_acquisition.contracts as contracts
        import speech_aware_evidence_acquisition.core.config as config
        import speech_aware_evidence_acquisition.core.bundle.git_commit as git
        from types import SimpleNamespace
        monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
        monkeypatch.setattr(git,'require_clean_study_repo',lambda p:'a'*40)
        monkeypatch.setattr(config,'compose',lambda *a:SimpleNamespace(config_hash='a'*64,values={'arm':'arm'}))
        class ExecutionPlan:
            def __init__(self,**kw):self.values=kw
            def validate(self,**kw):assert self.values['planned_model_calls'] in (7*n,14*n)
        monkeypatch.setattr(contracts,'ExecutionPlan',ExecutionPlan)
        c=exp['cells'][0];c.update(model_fragment=m.MODEL,readiness={'state':'held'})
        exp['service_session']=m.SESSION;exp['execution_plan']['planned_gpu_hours']=16
        put(tmp_path,f'configs/experiment/{name}.json',exp)
        # Prepare this banked half through the real guarded entry before registration.
        import tfrl_banked_supply as supply
        import speech_aware_evidence_acquisition.core.flight.factory as factory
        c.update(cell_id=name,allowed_execution_stages=['check'])
        c['readiness']['requirements']=[]
        exp['population']['roster_receipt']='roster.json'
        put(tmp_path,f'configs/experiment/{name}.json',exp)
        put(tmp_path,f'docs/readiness/{name}/sampling-plan.json',plan)
        monkeypatch.setattr(supply,'LOCK_SHA',lock['sha256'])
        prepares=[]
        class Runtime:
            def prepare_phase_supply(self,path,*,max_workers):
                prepares.append(path)
                put(tmp_path,str(path.relative_to(tmp_path)),{'output':{
                    'values':{'first':{r['sample_id']:[] for r in rows[:n]}}}})
        monkeypatch.setattr(factory,'build_flight_runtime',lambda *a,**kw:Runtime())
        out=tmp_path/'work/codex/prepare'
        result=supply.prepare(tmp_path,tmp_path,name,iteration,tmp_path/'work/codex/decision.json',out)
        assert result['rows']==313 and result['iteration']==iteration and len(prepares)==1
        assert result['model_calls']==result['tokenizer_calls']==result['scoring_jobs']==0
        from tfrl_current_supply_binding import bound_documents
        bound,held=bound_documents(base,exp,result,iteration=iteration)
        assert bound['prepared_phase_supply']==result['supply']
        assert held['cells'][0]['readiness']['state']=='held'
        with pytest.raises(ValueError,match='fresh owned'):
            supply.prepare(tmp_path,tmp_path,name,iteration,tmp_path/'work/codex/decision.json',out)
        assert len(prepares)==1
        base['prepared_phase_supply']={'relpath':'prepared/own.json','sha256':'f'*64}
        put(tmp_path,f'configs/baseline/{name}.json',base)
        put(tmp_path,f'docs/readiness/{name}/sampling-plan.json',plan)
        (tmp_path/'docs/exposure-ledger.md').write_text('')
        registered=m.register(tmp_path,tmp_path,'h1',iteration,tmp_path/'work/codex/decision.json')
        assert registered['readiness']=='held'
        written=m.read(tmp_path/f'docs/readiness/{name}/candidate.json')
        assert written['workload_kind']=='tfrl-real-banked-half' and written['iteration']==iteration
        assert validate_workload(tmp_path,tmp_path,name,written)==n
        with pytest.raises(ValueError,match='existing registration'):m.register(tmp_path,tmp_path,'h1',iteration,tmp_path/'work/codex/decision.json')


@pytest.mark.parametrize('iteration',[2,3])
@pytest.mark.parametrize('fault',[None,'label','operators','decoding'])
def test_banked_stage_preserves_scientific_algorithm(iteration,fault):
    exp={'cells':[{'model_fragment':'old','readiness':{'state':'held'},'allowed_execution_stages':['check']}],
        'stage3_algorithm':{'id':'tfrl-v2','operators_used':['rollout','scoring','context_application']}}
    old={'runtime_receipt':'old','temperature':.7};new={'runtime_receipt':m.RECEIPT,'temperature':.7}
    if fault=='label':exp['stage3_algorithm']['id']='shared-rollout'
    if fault=='operators':exp['stage3_algorithm']['operators_used']=['rollout','scoring']
    if fault=='decoding':new['temperature']=.8
    if fault:
        with pytest.raises(ValueError):m.stage_documents(exp,old,new,iteration=iteration)
    else:
        before=copy.deepcopy(exp);doc=m.stage_documents(exp,old,new,iteration=iteration)
        assert exp==before and doc['stage3_algorithm']==exp['stage3_algorithm']
        assert doc['cells'][0]['allowed_execution_stages']==['check','prepare-supply','worker','campaign']
        assert doc['cells'][0]['readiness']['state']=='held'


@pytest.mark.parametrize('iteration',[2,3])
@pytest.mark.parametrize('fault',[None,'gate','supply','binding'])
def test_banked_stage_writer_requires_verified_inputs_before_mutation(tmp_path,monkeypatch,iteration,fault):
    import speech_aware_evidence_acquisition.contracts as contracts
    import speech_aware_evidence_acquisition.core.bundle.git_commit as git
    import tfrl_banked_supply as supply
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(git,'require_clean_study_repo',lambda p:'a'*40)
    repo=tmp_path/'repo';data=tmp_path/'data';source=tmp_path/'source'
    lock=put(repo,m.LOCK,{'status':'locked_full_train_fold_b'});monkeypatch.setattr(m,'LOCK_SHA',lock['sha256'])
    put(repo,'configs/model/old.json',{'runtime_receipt':'old','temperature':.7})
    put(source,f'configs/model/{m.MODEL}.json',{'runtime_receipt':m.RECEIPT,'temperature':.7})
    put(source,'docs/'+m.RECEIPT,{'receipt':'fixture'})
    pins={};originals={}
    for h in ('h1','h2'):
        name=f'slurp-tfrl-current-bank-i{iteration}-{h}-v1'
        run=f'SAEA-SLURP-TFRL-CURRENT-BANK-I{iteration}-{h.upper()}-V1'
        ep=f'configs/experiment/{name}.json';bp=f'configs/baseline/{name}.json'
        exp={'cells':[dict(run_id=run,baseline_fragment=name,model_fragment='old',
            readiness={'state':'held'},allowed_execution_stages=['check'])],
            'stage3_algorithm':{'id':'tfrl-v2','operators_used':['rollout','scoring','context_application']}}
        sp=put(data,f'prepared/{run}/phase-supply-v1.json',{'output':'fixture'})
        for rel,doc in ((ep,exp),(bp,{'prepared_phase_supply':sp})):
            pins[rel]=put(repo,rel,doc)['sha256'];originals[rel]=(repo/rel).read_bytes()
        if fault=='supply':(data/sp['relpath']).write_text('{}')
    if fault=='binding':pins={}
    binding=data/'work/codex/binding.json';put(data,str(binding.relative_to(data)),{
        'schema':'saea-tfrl-current-supply-binding-v1','rows':625,'readiness':'held','new_model_calls':0,'output_hashes':pins})
    def guard(*a,**kw):
        assert kw['allow_bound']
        if fault=='gate':raise ValueError('failed decision')
    monkeypatch.setattr(supply,'validate_preparation_inputs',guard)
    out=data/'work/codex/stage'
    call=lambda:m.stage(repo,data,source,binding,out,iteration,data/'decision')
    if fault:
        with pytest.raises(ValueError):call()
        assert not out.exists()
        assert all((repo/rel).read_bytes()==raw for rel,raw in originals.items())
    else:
        result=call();assert result['new_model_calls']==0 and result['ready'] is False
        for rel in result['output_hashes']:
            doc=m.read(repo/rel)
            assert doc['stage3_algorithm']['id']=='tfrl-v2'
            assert doc['cells'][0]['allowed_execution_stages'][-1]=='campaign'
        with pytest.raises(ValueError,match='fresh owned'):call()
