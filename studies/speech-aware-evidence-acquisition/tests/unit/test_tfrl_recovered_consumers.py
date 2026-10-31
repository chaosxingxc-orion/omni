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
