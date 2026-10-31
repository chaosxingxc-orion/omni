"""Fold A admission must use its own roster, spend and supply identities."""
import json
from pathlib import Path
import shutil
import sys

import pytest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/flights/nx'))
import fold_a_current_registration as registration
from test_fold_a_current_comparator import source
from test_fold_a_current_agreement import put,table


@pytest.fixture
def prepared(source,tmp_path,monkeypatch):
    b,e,pins=source;v=registration.evaluation
    repo,src,data=tmp_path/'repo',tmp_path/'source',tmp_path/'data'
    data.mkdir();shutil.copytree(REPO/'configs',repo/'configs')
    for n in (v.comparator.inputs.ROSTER,'docs/results/2026-09-30-slurp-a2-comp1b-dev2000-read.md','scripts/flights/nx/l338_port_pins.json'):
        dest=repo/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/n,dest)
    base_rel=f'configs/baseline/{v.CELL}.json';exp_rel=f'configs/experiment/{v.CELL}.json'
    spec=registration.read(REPO/v.comparator.SPEC)
    put(src/v.comparator.SPEC,spec)
    source_package=(src/v.comparator.SPEC).parent
    oldfacts=registration.read(REPO/'docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/registered-facts-v1.json')
    measure=next(iter(oldfacts['stage3_measurement'].values()))
    # Use the real full arm measurement, not the four-row proof block.
    measure=oldfacts['stage3_measurement']['SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-NPU-PORT-V1']
    put(source_package/'registered-facts-v1.json',{'stage3_measurement':{spec['run_id']:measure}})
    c=v.comparator
    # Materialize every data receipt under private roots; keep real module and
    # roster semantics, with synthetic sealed file bytes and their real hashes.
    for pin in b['preparation_source_receipts']:
        root=data if pin['root']=='data' else repo
        path=root/pin['relpath']
        if not path.exists():
            original=REPO/pin['relpath']
            if pin['root']=='repository' and original.exists():
                path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,path)
            else:put(path,{'fixture':pin['relpath']})
        pin['sha256']=registration.sha(path)
    pins={}
    for n in (c.inputs.RECEIPT,c.read(repo/c.inputs.PROTOCOL)['query_rows_relpath'],
              c.inputs.ALIAS_RECEIPT,c.inputs.SEGMENTS,c.inputs.CONTENT,c.inputs.PROVENANCE):
        put(data/n,{'fixture':n});pins[n]=registration.sha(data/n)
    ids=[row['sample_id'] for row in registration.read(repo/c.inputs.ROSTER)['items']]
    _,receipt=table(data,ids);rp=data/v.agreement.OUTPUT/'receipt.json';put(rp,receipt)
    base,exp=v.derive(REPO,b,e,pins,receipt['artifact']['sha256'],registration.sha(rp))
    put(repo/base_rel,base);put(repo/exp_rel,exp)
    sb=f"configs/baseline/{spec['baseline']}.json";se=f"configs/experiment/{spec['cell']}.json"
    model='configs/model/'+e['cells'][0]['model_fragment']+'.json'
    put(src/sb,b);put(src/se,e)
    (src/model).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/model,src/model)
    completed=data/'parent/complete.json'
    put(completed,{'rows':1080,'score_surface':'sealed-final-delivery','commit':'source-head','run_id':spec['run_id']})
    retrieval=data/'retrieval/complete.json';put(retrieval,{'rows':1080,'values_unchanged':True,'pins':pins})
    comp=data/'comparator/complete.json';put(comp,{'source_commit':'source-head','source_run':spec['run_id'],
        'agent_completion_sha256':registration.sha(completed),'retrieval_completion_sha256':registration.sha(retrieval)})
    agree=data/'agreement/complete.json';put(agree,{'status':'complete','rows':1080,'comparator_completion_sha256':registration.sha(comp),
        'receipt_sha256':registration.sha(rp),'artifact_sha256':receipt['artifact']['sha256']})
    put(repo/v.PACKAGE/'derivation.json',{'launchable':False,'model_contacts':0,'source_commit':'source-head','source_run':spec['run_id'],
        'source_files':{n:registration.sha(src/n) for n in (sb,se,model)},
        'outputs':{n:registration.sha(repo/n) for n in (base_rel,exp_rel)},
        'parent_binding':{'source_spec':v.comparator.SPEC},
        'comparator_complete':'comparator/complete.json','agreement_complete':'agreement/complete.json',
        'comparator_completion_sha256':registration.sha(comp),'agreement_completion_sha256':registration.sha(agree)})
    (repo/'docs/exposure-ledger.md').write_text('# Prior ledger\n')
    binding={'session_id':exp['service_session'],'model_fragment':{'relpath':'configs/model/'+exp['cells'][0]['model_fragment']+'.json'},
             'lock_rows_per_minute_per_instance':6.,'tensor_parallel_size':2}
    bp=data/'binding.json';put(bp,binding)
    monkeypatch.setattr(registration,'validate_binding',lambda *args:binding)
    monkeypatch.setattr(registration,'accepted_proof',lambda *args:({'run_id':'PROOF'},{'proof':'a'*64}))
    monkeypatch.setattr(v.comparator,'bound_parents',lambda *args:(src/v.comparator.SPEC,completed,retrieval))
    monkeypatch.setattr(registration.subprocess,'check_output',lambda args,**kwargs:'source-head' if 'rev-parse' in args else b'')
    return repo,src,data,bp


def test_registration_and_actual_structural_gate(prepared):
    repo,src,data,binding=prepared;v=registration.evaluation
    result=registration.register(repo,src,data,binding,data/'archive')
    assert result['held'] and result['audio_seconds']==104874
    gen=registration.read(repo/registration.PKG/'generation-plan-registered-v1.json')
    fit=registration.read(repo/registration.PKG/'fit-plan-registered-v1.json')
    assert gen['execution_plan']['planned_model_calls']==36720
    assert fit['execution_plan']['planned_model_calls']==18360
    assert gen['execution_plan']['planned_speech_audio_seconds']==104874
    assert gen['prepared_phase_supply']['relpath']==f'prepared/{v.RUN}/phase-supply-v1.json'
    assert registration.read(repo/registration.PKG/'root-admission-draft-v1.json')['eligible_run_ids']==[v.RUN,v.FIT]
    shutil.copytree(REPO/'docs/receipts',repo/'docs/receipts',dirs_exist_ok=True)
    shutil.copytree(REPO/'scripts/checks',repo/'scripts/checks')
    sys.path.insert(0,str(REPO/'scripts/checks'))
    import flight_registration_check as gate
    cell,=gate.evaluate(repo,data,Path('docs/exposure-ledger.md'),cells=[v.CELL])['cells']
    assert cell['read_run_id']==registration.READ
    kinds={f['kind'] for f in cell['findings']}
    assert kinds=={'not-ready','receipt-missing','registered-supply-binding'},json.dumps(cell['findings'],indent=2)


def test_dry_run_no_mutation(prepared):
    repo,src,data,binding=prepared
    before={str(p.relative_to(repo)):p.read_bytes() for p in repo.rglob('*') if p.is_file()}
    result=registration.register(repo,src,data,binding,data/'archive',True)
    assert result['dry_run'] and result['audio_seconds']==104874
    assert not (data/'archive').exists()
    assert before=={str(p.relative_to(repo)):p.read_bytes() for p in repo.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault',['source','derived','spent','decision','roster','audio','serving','proof'])
def test_refuse_before_mutation(prepared,monkeypatch,fault):
    repo,src,data,binding=prepared;v=registration.evaluation
    ep=repo/f'configs/experiment/{v.CELL}.json'
    if fault=='source':
        spec=registration.read(src/v.comparator.SPEC);(src/f"configs/baseline/{spec['baseline']}.json").write_text('{}')
    elif fault=='derived':ep.write_bytes(ep.read_bytes()+b' ')
    elif fault=='spent':(data/'attempts'/v.RUN).mkdir(parents=True)
    elif fault=='decision':put(data/v.DECISION_PATH,{})
    elif fault in ('roster','audio','serving'):
        exp=registration.read(ep)
        if fault=='roster':exp['populations'][0]['ids'].pop()
        elif fault=='audio':exp['execution_plan']['planned_speech_audio_seconds']=107264
        else:exp['service_session']='wrong'
        put(ep,exp)
        dp=repo/v.PACKAGE/'derivation.json';d=registration.read(dp);d['outputs'][str(ep.relative_to(repo))]=registration.sha(ep);put(dp,d)
    else:
        def fail(*args):raise ValueError('no proof')
        monkeypatch.setattr(registration,'accepted_proof',fail)
    with pytest.raises(ValueError):registration.register(repo,src,data,binding,data/'archive')
    assert not (data/'archive').exists()
    assert (repo/'docs/exposure-ledger.md').read_text()=='# Prior ledger\n'


@pytest.mark.parametrize('fault',['vectors','agreement','completion','receipt','method','omitted-source'])
def test_registration_rechecks_material_inputs_and_rederives_method(prepared,fault):
    repo,src,data,binding=prepared;v=registration.evaluation
    if fault=='vectors':(data/v.comparator.inputs.CONTENT).write_text('changed')
    elif fault=='agreement':(data/v.agreement.OUTPUT/'agreement.jsonl').write_text('changed')
    elif fault=='completion':(data/'agreement/complete.json').write_text('{}')
    elif fault=='receipt':
        b=registration.read(repo/f'configs/baseline/{v.CELL}.json')
        pin=next(p for p in b['preparation_source_receipts'] if p['root']=='data' and p['relpath'].startswith('knowledge/'))
        (data/pin['relpath']).write_text('changed')
    else:
        dp=repo/v.PACKAGE/'derivation.json';d=registration.read(dp)
        if fault=='method':
            n=f'configs/baseline/{v.CELL}.json';b=registration.read(repo/n);b['what_this_is']='unreviewed';put(repo/n,b)
            d['outputs'][n]=registration.sha(repo/n)
        else:d['source_files']={}
        put(dp,d)
    with pytest.raises(ValueError):registration.register(repo,src,data,binding,data/'archive')
    assert not (data/'archive').exists()
