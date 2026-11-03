import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import gepa_probe_read as m
from test_tfrl_h1_observation_read import inputs


def fixture():
    result, items = inputs()
    result['per_sample'] = result['per_sample'][:16]
    result['aggregate']['samples'] = 16
    result['aggregate']['official']['slu_f1'] = 1.0
    return {a: copy.deepcopy(result) for a in m.ARMS}, items[:16]


def test_equal_probe_arms_do_not_claim_adoption():
    results, items = fixture()
    result = m.summarize(results, items)
    assert result['strict_success'] == {'gepa': 16, 'lock': 16}
    assert result['task_repairs'] == result['task_breaks'] == 0
    assert not result['candidate_accepted'] and not result['claims_stable_gain']


@pytest.mark.parametrize('fault', ['rows', 'arm', 'scorer', 'partial', 'count'])
def test_refuses_noncomparable_probe(fault):
    results, items = fixture()
    if fault == 'rows': items.pop()
    elif fault == 'arm': results['other'] = results.pop('gepa')
    elif fault == 'scorer': results['gepa']['aggregate']['evaluator_commit'] = 'changed'
    elif fault == 'partial': results['gepa']['per_sample'].pop()
    else: results['gepa']['per_sample'][0]['slot_f1_exact_fn'] = -1
    with pytest.raises(ValueError): m.summarize(results, items)


def test_recipe_binds_attempt_roster_and_delivering_arm():
    spec = {'run_id': 'RUN', 'attempt_id': 'generation-v1', 'rows': 16, 'roster': {'sha256': 'a'*64}}
    recipe = {'schema': 'saea-gepa-probe-read-v1', 'rows': 16, 'generation_run_id': 'RUN',
        'attempt_id': 'generation-v1', 'roster': spec['roster'], 'arms': ['gepa', 'lock'], 'delivered_arm': 'lock'}
    m.validate_recipe(recipe, spec)
    for field, value in [('attempt_id', 'other'), ('roster', {}), ('delivered_arm', 'gepa'), ('rows', 15)]:
        with pytest.raises(ValueError): m.validate_recipe({**recipe, field: value}, spec)
