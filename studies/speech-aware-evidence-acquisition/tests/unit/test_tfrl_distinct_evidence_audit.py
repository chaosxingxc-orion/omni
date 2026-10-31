from pathlib import Path
import sys
from types import SimpleNamespace as Obj
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import tfrl_distinct_evidence_audit as m


def test_rank_order_is_not_identifier_order():
    entries=[Obj(entry_id='f1',score=.9),Obj(entry_id='f2',score=.8)]
    values=[Obj(value_id='a',associated_facet_ids=['f2']),Obj(value_id='z',associated_facet_ids=['f1'])]
    assert m.ordered_values(entries,values,['a','z'])==['z','a']
    with pytest.raises(ValueError):m.ordered_values(entries[:1],values,['a'])


def test_odd_rank_selection_changes_ten_items_without_increasing_budget():
    ranked=[f'v{i}' for i in range(40)]
    chosen,fallback=m.variant(ranked[:20],ranked)
    assert chosen==ranked[::2] and not fallback
    assert len(chosen)==20 and len(set(chosen)&set(ranked[:20]))==10


def test_insufficient_distinct_evidence_falls_back():
    ranked=[f'v{i}' for i in range(39)]
    chosen,fallback=m.variant(ranked[:20],ranked)
    assert chosen==ranked[:20] and fallback


@pytest.mark.parametrize('fault',['duplicate','prefix'])
def test_ambiguous_or_changed_original_refused(fault):
    ranked=[f'v{i}' for i in range(40)];original=ranked[:20]
    if fault=='duplicate':ranked[-1]=ranked[0]
    else:original.reverse()
    with pytest.raises(ValueError):m.variant(original,ranked)
