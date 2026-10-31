"""Source custody, full-roster validation and no replay after partial output."""
import copy
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/flights/nx'))
import fold_a_current_agreement as agreement


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(agreement.raw(value))


def table(data, ids):
    rows = [{'schema': agreement.SEALED_AGREEMENT_SCHEMA, 'dimension': agreement.EXEMPLAR_INTENT_AGREEMENT_DIMENSION,
             'id': sid, 'scores': {'intent-boundary:a': .75, 'intent-boundary:b': .25},
             'vote_total': 20, 'self_excluded': 0} for sid in ids]
    path = data / agreement.OUTPUT / 'agreement.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    receipt = {'schema': 'saea-phase-agreement-receipt-v1', 'artifact_id': agreement.ARTIFACT,
        'contact_class': 'model-free', 'contacts': {k:0 for k in
            ('model','server','encoder','embedder','scorer','network','gpu','paid_api')},
        'artifact': {'relpath': agreement.OUTPUT+'/agreement.jsonl', 'sha256': agreement.sha(path), 'rows': len(ids)},
        'derived_from': {'experiment': agreement.comparator.CELL, 'cell': agreement.comparator.CELL,
            'vote_store': 'train-exemplars', 'vote_phase': 'solver', 'target_store': 'intent-boundaries',
            'self_exclusion': 'sample-id utterance identity (first three fields)'}}
    return rows, receipt


@pytest.mark.parametrize('fault', [None,'duplicate','missing','nan','bool','total','fraction','self-vote',
    'negative','extra-contact','missing-contact','path-escape','recall','exclusion','hash'])
def test_complete_probability_table(tmp_path, fault):
    ids = [f'slurp:train:{i:06}:r' for i in range(1080)]
    rows, receipt = table(tmp_path, ids)
    if fault=='duplicate': rows[-1]=copy.deepcopy(rows[0])
    elif fault=='missing': rows.pop()
    elif fault=='nan': rows[0]['scores']['intent-boundary:a']=float('nan')
    elif fault=='bool': rows[0]['vote_total']=True
    elif fault=='total': rows[0]['scores']['intent-boundary:a']=.5
    elif fault=='fraction': rows[0]['scores']={'intent-boundary:a':.73,'intent-boundary:b':.27}
    elif fault=='self-vote': rows[0]['self_excluded']=1
    elif fault=='negative': rows[0]['scores']['intent-boundary:a']=-.75
    elif fault=='extra-contact': receipt['contacts']['model']=1
    elif fault=='missing-contact': del receipt['contacts']['encoder']
    elif fault=='path-escape': receipt['artifact']['relpath']='../outside'
    elif fault=='recall': receipt['derived_from']['vote_phase']='perception'
    elif fault=='exclusion': receipt['derived_from']['self_exclusion']='none'
    path=tmp_path/agreement.OUTPUT/'agreement.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    receipt['artifact']['sha256']=agreement.sha(path) if fault!='hash' else '0'*64
    if fault:
        with pytest.raises(ValueError): agreement.verify_artifact(tmp_path,receipt,ids)
    else:
        assert len(agreement.verify_artifact(tmp_path,receipt,ids))==1080


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    repo, data = tmp_path/'repo', tmp_path/'data'
    ids=[f'slurp:train:{i:06}:r' for i in range(1080)]
    put(repo/agreement.comparator.inputs.ROSTER, {'items':[{'sample_id':sid} for sid in ids]})
    names=[f'configs/{kind}/{agreement.comparator.CELL}.json' for kind in ('baseline','experiment')]
    put(repo/names[0],{})
    put(repo/names[1],{'cells':[{'allowed_execution_stages':['check'],'readiness':{'state':'held'}}]})
    complete=data/'comparator/complete.json'
    put(complete,{'allowed_execution_stages':['check'],'model_contacts':0,'claims_system_lock':False,
                  'output_hashes':{n:agreement.sha(repo/n) for n in names}})
    builder=repo/'scripts/build_slurp_intent_exemplar_agreement.py';put(builder,{})
    lock=data/'lock.json';put(lock,{})
    monkeypatch.setattr(agreement.subprocess,'check_output',lambda args,**kw: 'source-head' if 'rev-parse' in args else b'')
    calls=[]
    def run(command, **kw):
        calls.append(command)
        _,receipt=table(data,ids)
        receipt['builder_sha256']=agreement.sha(builder)
        put(data/agreement.OUTPUT/'receipt.json',receipt)
    monkeypatch.setattr(agreement.subprocess,'run',run)
    return repo,data,lock,complete,calls


def test_complete_then_refuse_replay(prepared):
    repo,data,lock,complete,calls=prepared
    result=agreement.prepare(repo,data,lock,complete,data/'out','source-head')
    assert result['rows']==1080 and len(calls)==1
    assert '--vote-phase' in calls[0] and 'solver' in calls[0]
    assert (data/'out/intent.json').exists() and (data/'out/complete.json').exists()
    with pytest.raises(ValueError,match='fresh separate'):
        agreement.prepare(repo,data,lock,complete,data/'other','source-head')
    assert len(calls)==1


@pytest.mark.parametrize('fault',['head','config','missing-config','unheld','output','workers'])
def test_precontact_refusal(prepared,fault):
    repo,data,lock,complete,calls=prepared
    if fault=='config': (repo/f'configs/baseline/{agreement.comparator.CELL}.json').write_text('{} ')
    if fault in ('missing-config','unheld'):
        d=agreement.read(complete)
        if fault=='missing-config': d['output_hashes']={}
        else:
            n=f'configs/experiment/{agreement.comparator.CELL}.json'
            put(repo/n,{'cells':[{'allowed_execution_stages':['check'],'readiness':{'state':'ready'}}]})
            d['output_hashes'][n]=agreement.sha(repo/n)
        put(complete,d)
    if fault=='output': (data/agreement.OUTPUT).mkdir(parents=True)
    with pytest.raises(ValueError):
        agreement.prepare(repo,data,lock,complete,data/'out', 'wrong' if fault=='head' else 'source-head',
                          0 if fault=='workers' else 8)
    assert not calls and not (data/'out').exists()


def test_interrupted_builder_retains_evidence_and_prevents_retry(prepared,monkeypatch):
    repo,data,lock,complete,calls=prepared
    def fail(*args,**kw):
        (data/agreement.OUTPUT/'agreement.jsonl').write_text('partial')
        raise RuntimeError('interrupted')
    monkeypatch.setattr(agreement.subprocess,'run',fail)
    with pytest.raises(RuntimeError,match='interrupted'):
        agreement.prepare(repo,data,lock,complete,data/'out','source-head')
    assert agreement.read(data/'out/failed.json')['automatic_retry'] is False
    assert (data/agreement.OUTPUT/'agreement.jsonl').read_text()=='partial'
    assert not (data/'out/complete.json').exists()
    with pytest.raises(ValueError,match='fresh separate'):
        agreement.prepare(repo,data,lock,complete,data/'retry','source-head')


@pytest.mark.parametrize('changed', ['completion', 'lock', 'config'])
def test_midrun_parent_drift_preserves_artifact_but_refuses_completion(prepared, monkeypatch, changed):
    repo,data,lock,complete,calls=prepared
    original=agreement.subprocess.run
    def mutate(*args,**kwargs):
        original(*args,**kwargs)
        path={'completion':complete, 'lock':lock,
              'config':repo/f'configs/baseline/{agreement.comparator.CELL}.json'}[changed]
        path.write_bytes(path.read_bytes()+b' ')
    monkeypatch.setattr(agreement.subprocess,'run',mutate)
    with pytest.raises(ValueError,match='changed during'):
        agreement.prepare(repo,data,lock,complete,data/'out','source-head')
    assert (data/'out/failed.json').exists()
    assert not (data/'out/complete.json').exists()
    assert (data/agreement.OUTPUT/'agreement.jsonl').exists()
