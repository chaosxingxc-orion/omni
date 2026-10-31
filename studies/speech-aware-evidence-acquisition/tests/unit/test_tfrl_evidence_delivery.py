import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import tfrl_evidence_delivery as m


def answer(value='one'):
    return {'scenario':'s','action':'a','entities':[{'type':'n','filler':value}]}


def test_selects_whole_original_without_mutating():
    a,b=answer(),answer('two');before=copy.deepcopy((a,b))
    out,receipt=m.deliver(a,b,'{"choice":"B"}')
    assert out==b and out is not b and receipt['selected']=='B' and (a,b)==before


@pytest.mark.parametrize('text',['','B','{"choice":"C"}','{"choice":"B","entities":[]}',
    '{"choice":"A","choice":"B"}','["B"]','{"choice":null}'])
def test_invalid_selection_falls_back_to_original(text):
    a=answer();out,r=m.deliver(a,answer('two'),text)
    assert out==a and r['reason']=='invalid-selection-fallback'


def test_entity_order_is_equivalent_but_duplicates_are_not():
    a=answer();a['entities'].append({'type':'other','filler':'x'});b=copy.deepcopy(a);b['entities'].reverse()
    assert not m.needs_adjudication(a,b)
    assert m.deliver(a,b)[0]==a
    b['entities'].append(copy.deepcopy(b['entities'][0]))
    assert m.needs_adjudication(a,b)
    with pytest.raises(ValueError):m.deliver(a,b)


def test_equal_candidates_reject_unnecessary_contact():
    with pytest.raises(ValueError):m.deliver(answer(),answer(),'{}')


def test_budget_counts_reused_control_and_conditional_adjudicator():
    assert m.budget(625)['normal_generation_min']==2500
    assert m.budget(625)['normal_generation_max']==3125
    assert m.budget(625)['fallback_inclusive_cap']==6250
    with pytest.raises(ValueError):m.budget(True)
