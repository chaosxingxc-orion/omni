import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/flights/nx'))
from r01c_pool_audit import verify_membership


def fixture():
    queries=[f'slurp:train:{i}:headset' for i in range(625)]
    sid='slurp:train:9999:farfield'
    snap=NS(source_sample_ids=[sid],facets=[NS(facet_id='f',source_sample_ids=[sid])],
        values=[NS(source_sample_ids=[sid],associated_facet_ids=['f'],task_target='intent')])
    return queries,snap


def test_complete_utterance_disjoint_pool():
    ids,s=fixture();result=verify_membership(ids,s)
    assert result['utterance_overlap']==0 and result['query_rows']==625


@pytest.mark.parametrize('fault',['same-recording','other-recording','hidden-facet','hidden-value','unknown-facet','missing-target','duplicate-query'])
def test_reject_self_retrieval_and_inconsistent_membership(fault):
    ids,s=fixture()
    if fault=='same-recording':s.source_sample_ids.append(ids[0])
    elif fault=='other-recording':s.source_sample_ids.append('slurp:train:0:farfield')
    elif fault=='hidden-facet':s.facets[0].source_sample_ids.append(ids[0])
    elif fault=='hidden-value':s.values[0].source_sample_ids.append(ids[0])
    elif fault=='unknown-facet':s.values[0].associated_facet_ids=['wrong']
    elif fault=='missing-target':s.values[0].task_target=None
    else:ids[-1]=ids[0]
    with pytest.raises(ValueError):verify_membership(ids,s)
