import pytest
from tfrl_current_pool_queue import ready


def complete(h,n):
    return {'run_id':f'SAEA-SLURP-TFRL-CURRENT-SEED-I1-{h}-V1','rows':n,'official_scoring_jobs':7,'new_model_calls':0}


def test_partial_does_not_trigger_analysis():
    assert not ready([complete('H1',313),None],{'state':'generating'})
    with pytest.raises(ValueError):ready([complete('H1',313),None],{'state':'failed'})


def test_exact_two_sources_required():
    assert ready([complete('H1',313),complete('H2',312)],{})
    with pytest.raises(ValueError):ready([complete('H1',313),complete('H1',312)],{})
    with pytest.raises(ValueError):ready([complete('H1',313),complete('H2',311)],{})
