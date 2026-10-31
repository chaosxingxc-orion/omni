import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
import fold_a_c2_registration as m


def fixture(tmp_path):
    repo=tmp_path/'repo';data=tmp_path/'data'
    def put(root,rel,obj):
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(m.raw(obj))
        return {'relpath':rel,'sha256':m.sha(p)}
    ids=[f'slurp:train:{i}:audio' for i in range(1080)];split='a'*64
    view={'count':1080,'split_identity_hash':split,'items':[{'sample_id':i,'media_relpath':'datasets/slurp_real/'+i,
        'audio_sha256':'b'*64,'audio_seconds':1.} for i in ids]}
    roster=put(repo,'roster.json',{'items':[{'sample_id':i} for i in ids],'split_identity_hash':split})
    dataset=put(repo,'dataset.json',view)
    supply=put(data,f'prepared/{m.RUN}/phase-supply-v1.json',{'output':{'facets':{},'values':{'perception':dict.fromkeys(ids,{})}}})
    row=['date','encoder','profile','scope','tool','split','role',split,'yes','1080','1080','0','old','artifact','complete']
    closure=put(repo,m.CLOSURE,{'rows':1080,'generation_admitted':False,'derivatives_values_unchanged':True,'pins':{},
        'encoder_ledger_row':row,'derivative_ledger_rows':[],'closed_agreement_line':'agreement-completed'})
    p=repo/'docs/exposure-ledger.md';p.write_text('| '+' | '.join(row)+' |\nagreement-completed\n')
    base={'policy':'phase-native-module-v1-c2','native_episode':{'consumption_module':{}},'prepared_phase_supply':supply,
        'native_contact_fit':{'contact_cap':2160,'run_id':m.RUN+'-FIT','attempt_id':'fit-v1'}}
    put(repo,f'configs/baseline/{m.NAME}.json',base)
    model=put(repo,f'configs/model/{m.resident.MODEL}.json',{'runtime_receipt':m.resident.RECEIPT})
    spec={'cell':m.NAME,'experiment':m.NAME,'run_id':m.RUN,'attempt_id':'generation-v1','rows':1080,
        'profile':m.PROFILE,'service_session':m.resident.SESSION,'runtime_receipt':m.resident.RECEIPT,
        'roster':roster,'dataset':dataset,'split_identity_hash':split,'scientific_baseline_sha256':m.scientific_baseline_hash(base),
        'fit_run_id':m.RUN+'-FIT','fit_attempt_id':'fit-v1','model':model,'preparation_closure':closure,
        'gain_lock':{'relpath':m.LOCK,'sha256':m.LOCK_SHA}}
    exp={'cells':[{'cell_id':m.NAME,'run_id':m.RUN,'attempt_id':'generation-v1','baseline_fragment':m.NAME,
        'model_fragment':m.resident.MODEL,'dataset_fragment':'dataset'}],'service_session':m.resident.SESSION,
        'populations':[{'ids':ids,'dataset_fragment':'dataset'}],
        'population':{'count':1080,'roster_receipt':'roster.json','split_identity_hash':split,'dataset_fragment':'dataset'},
        'stage3_algorithm':{'id':'none','operators_used':[]},'stage3_measurement':{'claims_benefit':False},
        'execution_plan':{'planned_model_calls':4320,'planned_speech_audio_seconds':4320}}
    return repo,data,spec,exp,view


@pytest.mark.parametrize('fault',[None,'duplicate','subset','budget','algorithm','audio','base','supply','closure','model','program'])
def test_full_workload_validation(tmp_path,monkeypatch,fault):
    repo,data,spec,exp,view=fixture(tmp_path)
    real=m.pinned
    monkeypatch.setattr(m,'pinned',lambda root,pin: {'status':'locked_full_train_fold_b'} if pin==spec['gain_lock'] else real(root,pin))
    monkeypatch.setattr(m.baseline.cm,'verify_declared_module',lambda *a,**k:None)
    monkeypatch.setattr(m.baseline,'carrier_module_hooks',lambda **k:None)
    monkeypatch.setattr(m.baseline.cm,'compile_module',lambda *a,**k:SimpleNamespace(program_ceiling=3 if fault=='program' else 2))
    if fault=='duplicate':exp['populations'][0]['ids'][-1]=exp['populations'][0]['ids'][0]
    elif fault=='subset':view['items'].pop()
    elif fault=='budget':exp['execution_plan']['planned_model_calls']=2160
    elif fault=='algorithm':exp['stage3_algorithm']['id']='tfrl-v2'
    elif fault=='audio':view['items'][0]['media_relpath']='synthetic.wav'
    elif fault=='base':
        p=repo/f'configs/baseline/{m.NAME}.json';b=m.read(p);b['policy']='other';p.write_bytes(m.raw(b))
    elif fault=='supply':(data/f'prepared/{m.RUN}/phase-supply-v1.json').write_text('{}')
    elif fault=='closure':(repo/'docs/exposure-ledger.md').write_text('')
    elif fault=='model':(repo/spec['model']['relpath']).write_text('{}')
    if fault:
        with pytest.raises(ValueError):m.validate(repo,data,m.NAME,spec,exp,view)
    else:assert m.validate(repo,data,m.NAME,spec,exp,view)==1080


def test_supply_repin_excluded_but_method_changes_detected():
    a={'prepared_phase_supply':{'sha256':'a'},'native_episode':{'policy':'c2'}}
    b=copy.deepcopy(a);b['prepared_phase_supply']['sha256']='b'
    assert m.scientific_baseline_hash(a)==m.scientific_baseline_hash(b)
    b['native_episode']['policy']='c7'
    assert m.scientific_baseline_hash(a)!=m.scientific_baseline_hash(b)


def test_shared_workload_routes_fold_a_and_refuses_unknown(tmp_path,monkeypatch):
    import registered_workload
    repo,data,spec,exp,view=fixture(tmp_path)
    spec['workload_kind']='a0-fold-a-c2'
    p=repo/f'configs/experiment/{m.NAME}.json';p.parent.mkdir(parents=True);p.write_bytes(m.raw(exp))
    p=repo/'configs/dataset/dataset.json';p.parent.mkdir(parents=True);p.write_bytes(m.raw(view))
    seen=[]
    monkeypatch.setattr(m,'validate',lambda *args:seen.append(args) or 1080)
    assert registered_workload.validate_workload(repo,data,m.NAME,spec)==1080
    assert seen[0][2]==m.NAME
    spec['workload_kind']='unsupported'
    with pytest.raises(ValueError,match='unsupported'):
        registered_workload.validate_workload(repo,data,m.NAME,spec)
