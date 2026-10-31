import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import tfrl_recovered_cost_audit as m


def parts():
    return {name:{'rows':n,'contacts':7*n,'groups':{'s1':{'calls':n,'completion_tokens':2*n}},
        'cost_by_credit_disposition':{'positive':{'rows':n,'calls':7*n}},
        'answer_diversity':{'normalized:k4_all_equal':n},'contacts_per_row_distribution':{'7':n},
        'missing_request_timing':21 if n==3 else 0}
        for name,n in [('original-checkpoint',3),('repaired310',310),('h2',312)]}


def test_complete_source_union_counts_and_missing_timing():
    x=parts();before=copy.deepcopy(x);r=m.combine(x)
    assert x==before
    assert (r['rows'],r['contacts'],r['missing_request_timing'])==(625,4375,21)
    assert r['groups']['s1']=={'calls':625,'completion_tokens':1250}
    assert r['cost_by_credit_disposition']['positive']=={'rows':625,'calls':4375}
    assert r['contacts_per_row_distribution']=={'7':625}
    assert r['answer_diversity']['normalized:k4_all_equal']==625


@pytest.mark.parametrize('fault',['missing','extra','rows','contacts'])
def test_refuse_incomplete_union(fault):
    x=parts()
    if fault=='missing':del x['original-checkpoint']
    elif fault=='extra':x['other']=copy.deepcopy(x['h2'])
    elif fault=='rows':x['repaired310']['rows']=309
    else:x['h2']['contacts']=2183
    with pytest.raises(ValueError):m.combine(x)
