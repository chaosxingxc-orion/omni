import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/analysis')]
import tfrl_current_bank_run as m


def put(root,rel,obj):
    p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj));return p


def test_bank_registration_pins_sources_and_token_only_budget(tmp_path,monkeypatch):
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    import tfrl_current_half_result
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda _: 'head')
    monkeypatch.setattr(tfrl_current_half_result,'validate_recipe',lambda *a: {})
    repo=tmp_path/'repo';source=tmp_path/'source';(repo/'docs').mkdir(parents=True);(repo/'docs/exposure-ledger.md').write_text('# Ledger\n')
    for half in ('h1','h2'):
        name=f'slurp-tfrl-current-seed-i1-{half}-v1'
        put(source,f'docs/readiness/{name}/analysis-recipe-v1.json',{'half':half})
        put(source,f'configs/experiment/{name}.json',{'cells':[{'baseline_fragment':name,'model_fragment':'model'}],'service_session':'resident'})
        put(source,f'configs/baseline/{name}.json',{'knowledge_stores':{'same':'binding'}})
    put(source,'configs/model/model.json',{'runtime_receipt':'receipt.json'});put(source,'docs/receipt.json',{})
    recipe=m.register(repo,source)
    assert m.validate_recipe(repo)==recipe
    assert recipe['generation_calls']==0 and recipe['tokenizer_contact_cap']==20000
    with pytest.raises(ValueError,match='already registered'):m.register(repo,source)


def test_bank_run_requires_complete_verified_sources_before_endpoint(tmp_path,monkeypatch):
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core.session import serving
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(m,'validate_recipe',lambda _: {'source_recipes':{'h1':{},'h2':{}}})
    def refuse(*a):raise ValueError('half incomplete')
    def no_server(*a):pytest.fail('resident touched before source check')
    monkeypatch.setattr(m,'load_half',refuse);monkeypatch.setattr(serving,'find_resident_session',no_server)
    with pytest.raises(ValueError,match='incomplete'):m.run(tmp_path,tmp_path,tmp_path)
    assert not (tmp_path/'analysis').exists()
