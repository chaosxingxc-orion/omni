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


def test_strict_dominance_append_then_compare_and_keep_original_order():
    policy = dict(threshold=.2, merge_policy='strict_description_dominance')
    parent = ['a', 'b', 'c', 'd']
    cards = [card('d', -.1), card('x', .9), card('y', .8)]
    assert merge_description(parent, iter(cards), {'query'}, **policy)[0] == ['a', 'b', 'c', 'x', 'y']
    assert merge_description(['a', 'b', 'c'], cards, {'query'}, **policy)[0] == ['a', 'b', 'c', 'x', 'y']


def test_strict_dominance_does_not_skip_strong_or_unknown_victim():
    policy = dict(threshold=.2, merge_policy='strict_description_dominance')
    parent = ['a', 'b', 'c', 'd', 'e']
    for last in [card('e', .9), card('e', float('nan')), card('e', .1, 'query'), card('e', .1, complete=False)]:
        cards = [card('d', -.5), last, card('x', .8), card('y', .7)]
        assert merge_description(parent, cards, {'query'}, **policy)[0] == parent
    assert merge_description(parent, [card('d', -.5), card('x', .8)], {'query'}, **policy)[0] == parent
    assert merge_description(parent, [card('e', .8), card('x', .8)], {'query'}, **policy)[0] == parent


def test_strict_dominance_two_swaps_and_only_original_top_two():
    parent = ['a', 'b', 'c', 'd', 'e']
    cards = [card('d', .6), card('e', .1), card('x', .9), card('y', .7), card('z', .65)]
    result, _ = merge_description(parent, cards, {'query'}, threshold=.2, merge_policy='strict_description_dominance')
    assert result == ['a', 'b', 'c', 'x', 'y']
