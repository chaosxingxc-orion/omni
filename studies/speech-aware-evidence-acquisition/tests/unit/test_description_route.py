import math
import pytest
from speech_aware_evidence_acquisition.analysis.description_route import DescriptionCandidate, merge_description


def card(label, score, source='bank', complete=True):
    return DescriptionCandidate(label, 'id::'+label, score, frozenset([source]) if source else frozenset(), complete)


def test_merge_uses_capped_parent_and_deterministic_ties():
    parent = ['a', 'b', 'c', 'd', 'e']
    output, _ = merge_description(parent, [card('e', .99), card('z', .8), card('f', .8), card('f', .7)], {'query'}, threshold=.7)
    assert output == ['a', 'b', 'c', 'f', 'z']
    assert merge_description(['a'], [card('b', .8)], {'query'}, threshold=.7)[0] == ['a', 'b']
    assert merge_description(parent, [card('f', .8)], {'query'}, threshold=math.inf)[0] == parent


def test_sources_and_bad_scores_fail_closed():
    result, audit = merge_description(['a'], [card('b', .9, 'query'), card('c', .9, complete=False), card('d', float('nan')), card('e', .9, '')], {'query'}, threshold=.5)
    assert result == ['a']
    assert audit == {'unresolved': 2, 'excluded': 1, 'invalid_score': 1}
    with pytest.raises(ValueError):
        merge_description([], [], set(), threshold=.5)


def test_fill_preserves_parent_order_and_respects_capacity():
    candidates = [card('z', .9), card('y', .8), card('x', .7)]
    for parent in ([], ['a'], ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd', 'e']):
        result, _ = merge_description(parent, candidates, {'query'}, threshold=.75,
                                      merge_policy='fill_vacancies_only')
        assert result == parent + ['z', 'y'][:5-len(parent)]
    with pytest.raises(ValueError):
        merge_description([], candidates, {'query'}, threshold=.5, merge_policy='dominance')
