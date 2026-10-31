from pathlib import Path
import sys,copy
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/analysis'))
import tfrl_recovered_consumers as m


def test_registered_consumer_requires_both_countersigned_source_bound_reads(tmp_path,monkeypatch):
    import flight_registration_check as reg
    recipe={'schema':'saea-tfrl-recovered-consumers-v1','read_run_id':m.POOL,'bank_run_id':m.BANK,
        'rows':625,'new_model_calls':0,'official_scoring_jobs':0,'tokenizer_contact_cap':m.bank.CAP,
        'threshold_pp':1.0,'claims_benefit':False,'next_iteration_authorized':False,
        'exemplar_token_cap':2000,'rule_token_cap':400,'rule_support_rows':3,'sources':{'h1':{},'h2':{}}}
    monkeypatch.setattr(m,'read',lambda p:recipe);monkeypatch.setattr(m,'sha',lambda p:'a'*64)
    rows={rid:[{'artifacts':m.REL,'notes':'a'*64}] for rid in (m.POOL,m.BANK)}
    monkeypatch.setattr(reg,'parse_ledger',lambda *a:rows);monkeypatch.setattr(reg,'is_countersigned',lambda r:True)
    assert m.validate(tmp_path)==recipe
    rows.pop(m.BANK)
    with pytest.raises(ValueError,match='registered'):m.validate(tmp_path)
    rows[m.BANK]=[{'artifacts':m.REL,'notes':'a'*64}];recipe['rows']=624
    with pytest.raises(ValueError,match='scope'):m.validate(tmp_path)


def test_bank_producer_rejects_unowned_output_before_token_contacts(tmp_path):
    with pytest.raises(ValueError,match='owned bank'):
        m.bank.produce(tmp_path,tmp_path,tmp_path,{}, {},out=tmp_path/'outside',recipe_path=tmp_path/'recipe')


@pytest.mark.parametrize('iteration',[0,4,True,2.0])
def test_consumer_rejects_invalid_iteration(iteration):
    with pytest.raises(ValueError):m.identities(iteration)


@pytest.mark.parametrize('iteration',[2,3])
def test_banked_registration_pins_distinct_sources_and_loads_same_iteration(tmp_path,monkeypatch,iteration):
    import json
    import speech_aware_evidence_acquisition.contracts as contracts
    import speech_aware_evidence_acquisition.core.bundle.git_commit as git
    import tfrl_current_half_result as result
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(git,'require_clean_study_repo',lambda p:'a'*40)
    repo=tmp_path/'consumer';(repo/'docs').mkdir(parents=True)
    (repo/'docs/exposure-ledger.md').write_text('')
    roots={h:tmp_path/h for h in ('h1','h2')}
    def write(root,rel,doc):
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(doc))
    for h,root in roots.items():
        name=f'slurp-tfrl-current-bank-i{iteration}-{h}-v1'
        write(root,f'docs/readiness/{name}/analysis-recipe-v1.json',{
            'rows':313 if h=='h1' else 312,'run_id':f'SAEA-SLURP-TFRL-CURRENT-BANK-I{iteration}-{h.upper()}-V1','attempt_id':'generation-v1'})
        write(root,f'configs/experiment/{name}.json',{'cells':[{'model_fragment':'model','baseline_fragment':name}],'service_session':'session'})
        write(root,'configs/model/model.json',{'runtime_receipt':'receipts/runtime.json'})
        write(root,'docs/receipts/runtime.json',{'runtime':'same'})
        write(root,f'configs/baseline/{name}.json',{'knowledge_stores':['same']})
    monkeypatch.setattr(result,'validate_recipe',lambda root,p:m.read(p))
    recipe=m.register(repo,tmp_path,roots['h1'],roots['h2'],iteration)
    pool,bank,rel=m.identities(iteration)
    assert recipe['iteration']==iteration and recipe['bank_run_id']==bank
    assert bank!=m.BANK and pool!=m.POOL
    assert not (repo/m.REL).exists()
    assert recipe['sources']['h1']['commit']=='a'*40
    assert m.read(repo/rel)==recipe
    # Keep source pin validation real; mock only the heavy verified loader.
    monkeypatch.setattr(m,'validate',lambda root,it:recipe if it==iteration else pytest.fail('wrong consumer iteration'))
    calls=[]
    def load(root,data,half,pin,**kwargs):
        calls.append((half,kwargs));return {'verified':half}
    monkeypatch.setattr(m,'load_half',load)
    monkeypatch.setattr(m.h1union,'run',lambda *a,**k:pytest.fail('banked half must not use original union'))
    _,_,loaded=m.load(repo,tmp_path,iteration)
    assert set(loaded)=={'h1','h2'}
    assert calls==[('h1',{'iteration':iteration}),('h2',{'iteration':iteration})]
    (roots['h1']/'configs/model/model.json').write_text('{}')
    with pytest.raises(ValueError,match='pin differs'):m.load(repo,tmp_path,iteration)


def test_later_pool_uses_new_output_and_passes_iteration_to_loader(tmp_path,monkeypatch):
    import speech_aware_evidence_acquisition.contracts as contracts
    import speech_aware_evidence_acquisition.core.bundle.git_commit as git
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(git,'require_clean_study_repo',lambda p:'a'*40)
    monkeypatch.setattr(m,'sha',lambda p:'b'*64)
    def load(repo,data,iteration):
        assert iteration==2
        return {},{}, {h:{'plan':{'half':'R01C-'+h.upper()},'results':{},'source':h} for h in ('h1','h2')}
    monkeypatch.setattr(m,'load',load)
    monkeypatch.setattr(m,'pool',lambda halves:{'iteration':2,'rows':625})
    monkeypatch.setattr(m.bank,'produce',lambda *a,**k:pytest.fail('pool must not tokenize'))
    result=m.run(tmp_path,tmp_path,'pool',2)
    assert result['read_run_id']==m.identities(2)[0]
    assert (tmp_path/'analysis'/m.identities(2)[0]/'complete.json').exists()
    assert not (tmp_path/'analysis'/m.POOL).exists()
    monkeypatch.setattr(m,'pool',lambda halves:{'iteration':1,'rows':625})
    with pytest.raises(ValueError,match='pooled iteration'):m.run(tmp_path,tmp_path,'pool',2)
