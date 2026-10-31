from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
from export_real_engine_result import generation_duration, reconcile_cost


def test_duration_excludes_score_wait_and_refuses_wrong_start():
    spec={'cell':'c','profile':'p','run_id':'r'}
    start={'candidate':'c','profile':'p','rows':500,'time':100}
    generation={'status':'COMPLETED','run_id':'r'}
    assert generation_duration(start,generation,130,spec)==30
    with pytest.raises(ValueError,match='identity'):
        generation_duration({**start,'candidate':'other'},generation,130,spec)
    with pytest.raises(ValueError,match='duration'):
        generation_duration(start,generation,90,spec)


def test_cost_reconciliation_retains_every_generation_contact():
    rows=[SimpleNamespace(cost={'calls':2,'prompt_tokens':15,'completion_tokens':7,'audio_seconds':1.1})]*2
    cost={'calls_used':4,'prompt_tokens_total':30,'completion_tokens_total':14,'audio_seconds_used':2.2}
    assert reconcile_cost(rows,cost)['calls']==4
    with pytest.raises(ValueError,match='cost mismatch'):
        reconcile_cost(rows,{**cost,'calls_used':5})
