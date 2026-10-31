"""The real Fold A arm restores all current A0 method semantics, held for admission."""
import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/flights/nx'))
import fold_a_current_evaluation as evaluation
from speech_aware_evidence_acquisition.core.config import compose
from test_fold_a_current_comparator import source
from test_fold_a_current_agreement import table,put


def test_restore_actual_a0_component_and_all_sampling_columns(source,tmp_path):
    base,exp,pins=source
    original=copy.deepcopy((base,exp,pins))
    b,e=evaluation.derive(REPO,base,exp,pins,'a'*64,'b'*64)
    assert (base,exp,pins)==original
    m=b['native_episode']['consumption_module'];old=base['native_episode']['consumption_module']
    assert m['columns']==old['columns'] and m['sampling_frame']==old['sampling_frame']
    for new,prev in zip(m['components'],old['components']):
        if prev['component_id']!='agree-vote-seed': assert new==prev
        else:
            assert new['state']=='on'
            assert new['params']['k']==prev['params']['k']==60
            assert new['params']['w_agree']==prev['params']['w_agree']==1
            assert new['params']['agreement_source']['sha256']=='a'*64
    phase=b['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
    assert phase['k']==60 and phase['rank_weights']==base['knowledge_stores']['intent-boundaries']['phase_supply']['perception']['rank_weights']
    assert phase['agreement_source']['relpath']==evaluation.agreement.OUTPUT+'/agreement.jsonl'
    assert e['service_session']==exp['service_session']
    assert e['cells'][0]['model_fragment']==exp['cells'][0]['model_fragment']
    assert e['cells'][0]['allowed_execution_stages']==['check']
    assert e['cells'][0]['readiness']['state']=='held'
    assert e['generation_budget']['accelerator_bound']['status']=='PENDING-FOLD-A-ADMISSION'
    assert len(e['populations'][0]['ids'])==1080
    assert e['execution_plan']['planned_speech_audio_seconds']==104874
    assert not b.get('prepared_phase_supply')
    assert 'FOLD-B' not in json.dumps(b['preparation_source_receipts'])
    shutil.copytree(REPO/'configs',tmp_path/'configs')
    put(tmp_path/f'configs/baseline/{evaluation.CELL}.json',b)
    put(tmp_path/f'configs/experiment/{evaluation.CELL}.json',e)
    c=e['cells'][0]
    cfg=compose(tmp_path,c['model_fragment'],c['dataset_fragment'],evaluation.CELL,evaluation.CELL)
    evaluation.cm.verify_declared_module(cfg.values['native_episode']['consumption_module'],cfg.values,repo_root=REPO)


@pytest.mark.parametrize('fault',['zero','nonhex','missing','disabled','recall'])
def test_refuse_non_a0_or_unsealed_agreement(source,fault):
    b,e,p=source
    digest='a'*64
    if fault=='zero':digest='0'*64
    elif fault=='nonhex':digest='z'*64
    elif fault=='missing':digest=None
    elif fault=='disabled':
        for c in b['native_episode']['consumption_module']['components']:
            if c['component_id']=='agree-vote-seed':c['state']='off'
    else:b['knowledge_stores']['train-exemplars']['phase_supply']['solver']['k']=3
    with pytest.raises((ValueError,evaluation.cm.ConsumptionModuleConfigError)):
        evaluation.derive(REPO,b,e,p,digest,'b'*64)


@pytest.mark.parametrize('fault',[None,'ancestry','artifact','model','source','rowcount'])
def test_writer_pins_real_parent_and_refuses_drift(source,tmp_path,monkeypatch,fault):
    base,exp,_=source
    repo,src,data=tmp_path/'repo',tmp_path/'source',tmp_path/'data'
    shutil.copytree(REPO/'configs',repo/'configs')
    c=evaluation.comparator
    for n in (c.inputs.ROSTER,'docs/results/2026-09-30-slurp-a2-comp1b-dev2000-read.md','scripts/flights/nx/l338_port_pins.json'):
        dest=repo/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/n,dest)
    spec=c.read(REPO/c.SPEC)
    put(src/c.SPEC,spec)
    put(src/f"configs/baseline/{spec['baseline']}.json",base)
    put(src/f"configs/experiment/{spec['cell']}.json",exp)
    model='configs/model/'+exp['cells'][0]['model_fragment']+'.json'
    dest=src/model;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/model,dest)
    pins={}
    for n in (c.inputs.RECEIPT,c.read(repo/c.inputs.PROTOCOL)['query_rows_relpath'],
              c.inputs.ALIAS_RECEIPT,c.inputs.SEGMENTS,c.inputs.CONTENT,c.inputs.PROVENANCE):
        put(data/n,{'fixture':n});pins[n]=c.sha(data/n)
    completed=data/'parent/complete.json';retrieval=data/'retrieval/complete.json'
    put(completed,{'commit':'actual-head','run_id':spec['run_id'],'rows':1080,'score_surface':'sealed-final-delivery'})
    put(retrieval,{'rows':1080,'values_unchanged':True,'pins':pins})
    comp=data/'comparator/complete.json'
    put(comp,{'source_commit':'actual-head','source_run':spec['run_id'],
              'agent_completion_sha256':c.sha(completed),'retrieval_completion_sha256':c.sha(retrieval)})
    ids=[x['sample_id'] for x in c.read(repo/c.inputs.ROSTER)['items']]
    _,receipt=table(data,ids)
    receipt_path=data/evaluation.agreement.OUTPUT/'receipt.json';put(receipt_path,receipt)
    ac=data/'agreement/complete.json'
    put(ac,{'status':'complete','rows':1080,'comparator_completion_sha256':c.sha(comp),
            'receipt_sha256':c.sha(receipt_path),'artifact_sha256':receipt['artifact']['sha256']})
    monkeypatch.setattr(c,'bound_parents',lambda *args:(src/c.SPEC,completed,retrieval))
    monkeypatch.setattr(evaluation.subprocess,'check_output',lambda args,**kw:'actual-head' if 'rev-parse' in args else b'')
    if fault=='ancestry':put(comp,{**c.read(comp),'source_run':'wrong'})
    elif fault=='artifact':(data/evaluation.agreement.OUTPUT/'agreement.jsonl').write_text('changed')
    elif fault=='model':(repo/model).write_bytes((repo/model).read_bytes()+b' ')
    elif fault=='source':put(completed,{**c.read(completed),'commit':'wrong'})
    elif fault=='rowcount':put(completed,{**c.read(completed),'rows':1079})
    if fault:
        with pytest.raises(ValueError):evaluation.write(repo,src,data,{},comp,ac,data/'out')
        assert not (data/'out').exists()
    else:
        result=evaluation.write(repo,src,data,{},comp,ac,data/'out')
        assert result['launchable'] is False and result['model_contacts']==0
        assert result['source_files'][model]==c.sha(src/model)
        assert c.read(repo/f'configs/experiment/{evaluation.CELL}.json')['cells'][0]['readiness']['state']=='held'
        with pytest.raises(ValueError,match='fresh derivation'):
            evaluation.write(repo,src,data,{},comp,ac,data/'out-again')
