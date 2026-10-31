import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('coverage',Path(__file__).resolve().parents[2]/'scripts/analysis/audit_a0_roster_coverage.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_disjoint_union_and_historical_precedence():
    d=m.partition(['a','b','c','d'],['a','b'],['b','c'])
    assert d['historical']==['a','b']
    assert d['engineering_additional_if_compatible']==['c']
    assert d['remaining_if_all_engineering_complete_and_compatible']==['d']
    assert d['remaining_without_engineering_reuse']==['c','d']


def test_duplicate_or_foreign_rows_refused():
    with pytest.raises(ValueError,match='duplicate'):m.partition(['a'],['a','a'],[])
    with pytest.raises(ValueError,match='outside'):m.partition(['a'],[],['b'])
