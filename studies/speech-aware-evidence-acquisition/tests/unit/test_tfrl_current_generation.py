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
