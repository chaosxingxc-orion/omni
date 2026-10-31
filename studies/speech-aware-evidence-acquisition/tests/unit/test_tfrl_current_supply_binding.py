from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/flights/nx'))
from tfrl_current_supply_binding import bound_documents


@pytest.mark.parametrize('iteration',[1,2,3])
@pytest.mark.parametrize('fault',[None,'cross_half','admitted','spent','wrong_rows','not_complete','already_bound'])
def test_exact_unspent_half_binding(fault,iteration):
    prefix='seed' if iteration==1 else 'bank'
    name=f'slurp-tfrl-current-{prefix}-i{iteration}-h1-v1';run=f'SAEA-SLURP-TFRL-CURRENT-{prefix.upper()}-I{iteration}-H1-V1'
    base={};exp={'population':{'count':313},'cells':[{'cell_id':name,'run_id':run,'readiness':{'state':'held','requirements':[]},'allowed_execution_stages':['check']}]}
    prep={'schema':'saea-tfrl-current-supply-v1','rows':313,'cell':name,'run_id':run,'model_calls':0,'readiness':'held','registered':False,'supply':{'relpath':f'prepared/{run}/phase-supply-v1.json','sha256':'a'*64}}
    if iteration>1:prep['iteration']=iteration
    if fault=='cross_half':prep['supply']['relpath']=prep['supply']['relpath'].replace('H1','H2')
    elif fault=='admitted':exp['cells'][0]['readiness']['state']='ready'
    elif fault=='spent':prep['model_calls']=1
    elif fault=='wrong_rows':prep['rows']=312
    elif fault=='not_complete':prep['registered']=True
    elif fault=='already_bound':base['prepared_phase_supply']=prep['supply']
    if fault:
        with pytest.raises(ValueError):bound_documents(base,exp,prep,iteration=iteration)
    else:
        b,e=bound_documents(base,exp,prep,iteration=iteration)
        assert b['prepared_phase_supply']==prep['supply']
        assert e['cells'][0]['readiness']['requirements']==[{'kind':'preparation',**prep['supply']}]
        assert base=={} and exp['cells'][0]['readiness']['requirements']==[]


@pytest.mark.parametrize('umbrella',[False,True])
def test_binding_reads_exact_preparation_commit_in_both_repo_layouts(tmp_path,monkeypatch,umbrella):
    import json,subprocess
    import tfrl_current_supply_binding as m
    import tfrl_banked_supply as supply
    import speech_aware_evidence_acquisition.contracts as contracts
    import speech_aware_evidence_acquisition.core.flight.factory as factory
    root=tmp_path/'git';root.mkdir()
    repo=root/'studies/speech-aware-evidence-acquisition' if umbrella else root
    repo.mkdir(parents=True,exist_ok=True);data=tmp_path/'data';preps=[];sources={}
    def put(path,doc):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(doc))
    def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    git('init','-q');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
    for h,n in [('h1',313),('h2',312)]:
        name=f'slurp-tfrl-current-bank-i2-{h}-v1';run=f'SAEA-SLURP-TFRL-CURRENT-BANK-I2-{h.upper()}-V1'
        ids=[str(i) for i in range(n)]
        base={};exp={'population':{'count':n},'populations':[{'ids':ids}],
            'cells':[dict(cell_id=name,run_id=run,attempt_id='a',model_fragment='m',dataset_fragment='d',
                baseline_fragment=name,readiness={'state':'held','requirements':[]},allowed_execution_stages=['check'])]}
        bp=repo/f'configs/baseline/{name}.json';ep=repo/f'configs/experiment/{name}.json'
        put(bp,base);put(ep,exp)
        sources[name]={str(p.relative_to(repo)):m.sha(p) for p in (bp,ep)}
        target=data/f'prepared/{run}/phase-supply-v1.json';put(target,{'output':{'values':{'p':{i:[] for i in ids}}}})
        decision=data/'work/codex/decision.json';put(decision,{'decision':'fixture'})
        prep={'schema':'saea-tfrl-current-supply-v1','iteration':2,'rows':n,'cell':name,'run_id':run,
            'model_calls':0,'readiness':'held','registered':False,'source_hashes':sources[name],
            'supply':{'relpath':str(target.relative_to(data)),'sha256':m.sha(target)},
            'first_construction_decision':{'relpath':str(decision.relative_to(data)),'sha256':m.sha(decision)}}
        preps.append(prep)
    git('add','.');git('commit','-qm','fixture sources');head=git('rev-parse','HEAD')
    paths=[]
    for i,prep in enumerate(preps):
        prep['source_commit']=head;p=data/f'work/codex/prep{i}.json';put(p,prep);paths.append(p)
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(supply,'validate_preparation_inputs',lambda repo,data,name,*a:(None,None,sources[name],None))
    class Runtime:
        def verify_prepared_artifacts(self):return [{'identity_ok':True,'seed_ok':True}]
    monkeypatch.setattr(factory,'build_flight_runtime',lambda *a,**kw:Runtime())
    out=data/'work/codex/bind'
    result=m.bind(repo,data,paths,out,iteration=2)
    assert result['rows']==625 and result['readiness']=='held'
    assert len(result['halves'])==2 and (out/'complete.json').exists()
    for prep in preps:
        base=json.loads((repo/f"configs/baseline/{prep['cell']}.json").read_bytes())
        assert base['prepared_phase_supply']==prep['supply']
