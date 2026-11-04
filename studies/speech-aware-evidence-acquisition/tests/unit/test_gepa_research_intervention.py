import copy,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/analysis'))
import external_exploration as e
import gepa_research_intervention as m
from test_gepa_external_mutation import material
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha


def setup(tmp_path,monkeypatch):
    source,contract,parent,*_=material(tmp_path,monkeypatch)
    verified=m.verify_external_mutation(source,contract=contract,parent=parent)
    original=json.loads(verified['mutation_text']);mutation=copy.deepcopy(original)
    mutation['solver_instruction']='Use supplied slot explanations for separate clock and part-of-day mentions.'
    doc={'schema':'saea-gepa-research-intervention-document-v1','source':source,
        'parent_sha256':parent['candidate_sha256'],'contract_sha256':contract['contract_sha256'],
        'mutation':mutation,'review':{'author':'researcher','kind':'researcher-authored-factor-ablation',
        'factor':'time-only','rationale':'Separate two factors without claiming a new teacher response.',
        'source_mutation_sha256':sha(original),'mutation_sha256':sha(mutation),'decision':'eligible-for-feedback-only'}}
    path=tmp_path/'intervention.json';path.write_text(json.dumps(doc))
    return {'schema':'saea-gepa-research-intervention-v1','document':e.pin(path)},contract,parent,doc,path


def test_authentic_source_and_new_research_identity(tmp_path,monkeypatch):
    spec,c,p,doc,path=setup(tmp_path,monkeypatch);r=m.verify_intervention(spec,contract=c,parent=p)
    assert r['receipt_kind']=='researcher-authored-factor-ablation-v1'
    assert r['response_sha256']==e.pin(path)['sha256']
    assert r['source_external_response_sha256'] != r['response_sha256']
    assert r['new_model_calls']==0 and r['candidate_accepted'] is False


@pytest.mark.parametrize('fault',['bytes','parent','source','review','mutation','decision'])
def test_changed_intervention_rejected(tmp_path,monkeypatch,fault):
    spec,c,p,d,path=setup(tmp_path,monkeypatch)
    if fault=='bytes':path.write_text('{}')
    else:
        if fault=='parent':d['parent_sha256']='f'*64
        elif fault=='source':d['source']['candidate_index']=999
        elif fault=='review':d['review']['source_mutation_sha256']='f'*64
        elif fault=='mutation':d['mutation']['solver_instruction']='changed without review'
        else:d['review']['decision']='production'
        path.write_text(json.dumps(d));spec['document']=e.pin(path)
    with pytest.raises(ValueError):m.verify_intervention(spec,contract=c,parent=p)
