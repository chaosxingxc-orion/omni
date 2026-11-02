import random
import json
from dataclasses import FrozenInstanceError
import pytest
from speech_aware_evidence_acquisition.rails.gepa_search import (
    ScoredCandidate, cover_prune, select_parent, random_from_state,
    best_candidate, compare_minibatch,
)

IDS = ('a', 'b', 'c', 'd')
def c(i, scores, ids=IDS):
    return ScoredCandidate(f'c{i}', f'{i:064x}', ids, tuple(scores))

def test_complementary_winners_and_redundant_cover_are_pruned():
    result = cover_prune((c(0, (1, 1, .6, 0)), c(1, (0, .6, 1, 1)),
                          c(2, (1, 0, 1, 0))), IDS)
    assert result.instance_winners == ((0, 2), (0,), (1, 2), (1,))
    assert result.retained_winners == ((0,), (0,), (1,), (1,))
    assert result.removed_indices == (2,)
    assert result.sampling_indices == (0, 1)
    assert result.sampling_weights == (2, 2)

def test_ties_keep_earliest_pool_candidate_when_best_and_cover_is_deterministic():
    pool = (c(0, (1, 1, 1, 1)), c(1, (1, 1, 1, 1)))
    result = cover_prune(pool, IDS)
    assert result.retained_winners == ((1,),) * 4
    assert best_candidate(pool, IDS).candidate_id == 'c0'

def test_parent_selection_is_replayable_and_uses_survivor_frequency():
    pool = (c(0, (1, 1, 0, 0)), c(1, (0, 0, 1, 1)))
    first = select_parent(pool, IDS, rng=random.Random(17))
    replay = random_from_state(first.rng_before)
    again = select_parent(pool, IDS, rng=replay)
    assert first.candidate_id == again.candidate_id
    assert first.rng_after == again.rng_after

def test_bad_shape_nonfinite_duplicate_ids_and_partial_scores_rejected():
    with pytest.raises(ValueError): ScoredCandidate('x', '0' * 64, ('a', 'a'), (1, 2))
    with pytest.raises(ValueError): ScoredCandidate('x', '0' * 64, IDS, (1, 2, 3))
    with pytest.raises(ValueError): ScoredCandidate('x', '0' * 64, IDS, (1, 2, 3, float('nan')))
    with pytest.raises(ValueError): cover_prune((c(0, (1, 1, 1, 1)), c(0, (1, 1, 1, 1))), IDS)

def test_minibatch_requires_same_order_and_strict_improvement():
    parent = c(0, (0.5, 0.5), ('x', 'y'))
    child = c(1, (0.6, 0.5), ('x', 'y'))
    assert compare_minibatch(parent, child, ('x', 'y')).accepted
    assert not compare_minibatch(parent, c(2, (.5, .5), ('x', 'y')), ('x', 'y')).accepted
    with pytest.raises(ValueError): compare_minibatch(parent, child, ('y', 'x'))

def test_noop_is_rejected_even_if_candidate_id_differs():
    parent = c(0, (0.5, 0.5), ('x', 'y'))
    child = ScoredCandidate('other', parent.instruction_sha256, ('x', 'y'), (.9, .9))
    assert compare_minibatch(parent, child, ('x', 'y')).reason == 'rejected-noop'


@pytest.mark.parametrize('value', [None, True, '1', float('nan'), float('inf'), -float('inf'), 10**400])
def test_invalid_score_values(value):
    with pytest.raises(ValueError):
        c(0, (value, 0, 0, 0))


def test_nonfinite_aggregate_rejected():
    with pytest.raises(ValueError):
        c(0, (1e308, 1e308, 0, 0))


@pytest.mark.parametrize('ids', [(), ['x', 'x'], ['x', ''], {'x', 'y'}, 'xy'])
def test_invalid_sample_ids(ids):
    with pytest.raises(ValueError):
        ScoredCandidate('c0', '0' * 64, ids, (1, 1))


@pytest.mark.parametrize('digest', ['', 'x' * 64, 'A' * 64, None])
def test_invalid_instruction_sha(digest):
    with pytest.raises(ValueError):
        ScoredCandidate('c0', digest, ('x',), (1,))


def test_complete_roster_is_required_even_for_single_candidate():
    with pytest.raises(ValueError):
        cover_prune((c(0, (1, 1), ('a', 'b')),), IDS)
    with pytest.raises(ValueError):
        cover_prune((c(0, (1, 1, 1, 1), tuple(reversed(IDS))),), IDS)
    with pytest.raises(ValueError):
        best_candidate((), IDS)


def test_duplicate_instruction_cannot_enter_pool():
    original = c(0, (1, 1, 1, 1))
    duplicate = ScoredCandidate('another-id', original.instruction_sha256, IDS, (0, 0, 0, 0))
    with pytest.raises(ValueError):
        cover_prune((original, duplicate), IDS)


def test_input_vectors_are_copied_and_records_are_immutable():
    ids = ['x', 'y']
    scores = [0, 1]
    record = ScoredCandidate('c0', '0' * 64, ids, scores)
    ids[0] = 'changed'
    scores[0] = 10
    assert record.sample_ids == ('x', 'y')
    assert record.scores == (0, 1)
    with pytest.raises(FrozenInstanceError):
        record.scores = (10, 10)


def test_rng_json_roundtrip_and_invalid_input_does_not_advance_rng():
    pool = (c(0, (1, 1, 1, 0)), c(1, (0, 0, 0, 1)))
    rng = random.Random(42)
    result = select_parent(pool, IDS, rng=rng)
    assert result.coverage.sampling_weights == (3, 1)
    document = json.loads(json.dumps(result.to_dict()))
    restored = random_from_state(document['rng_before'])
    replay = select_parent(pool, IDS, rng=restored)
    assert replay == result
    assert random_from_state(document['rng_after']).getstate() == rng.getstate()
    before = rng.getstate()
    with pytest.raises(ValueError):
        select_parent(pool, ('a',), rng=rng)
    assert rng.getstate() == before


@pytest.mark.parametrize('state', [None, [], [2, [0] * 625, None],
                                  [3, [0] * 624, None],
                                  [3, [0] * 624 + [625], None],
                                  [3, [True] * 624 + [0], None],
                                  [3, [0] * 625, float('nan')]])
def test_invalid_rng_state_rejected(state):
    with pytest.raises(ValueError):
        random_from_state(state)


def test_first_winner_appearance_breaks_equal_mean_pruning_ties():
    # c1 appears before c0 because it wins the first sample. Equal means
    # therefore remove c1 first, preserving c0 and c2 for complete coverage.
    pool = (c(0, (0, 1, 1, 0)), c(1, (1, 0, 1, 0)), c(2, (1, 0, 0, 1)))
    result = cover_prune(pool, IDS)
    assert result.prune_order == (1, 2, 0)
    assert result.removed_indices == (1,)
    assert result.retained_winners == ((2,), (0,), (0,), (2,))


def test_nonwinning_high_average_candidate_is_not_sampling_fallback():
    pool = (c(0, (1, 0, 1, 0)), c(1, (0, 1, 0, 1)), c(2, (.9, .9, .9, .9)))
    assert best_candidate(pool, IDS).candidate_id == 'c2'
    result = cover_prune(pool, IDS)
    assert result.sampling_indices == (0, 1)


def test_minibatch_no_tolerance_or_partial_vector_acceptance():
    parent = c(0, (0, 0), ('x', 'y'))
    assert compare_minibatch(parent, c(1, (1e-15, 0), ('x', 'y')), ('x', 'y')).accepted
    assert not compare_minibatch(parent, c(1, (-1, 0), ('x', 'y')), ('x', 'y')).accepted
    with pytest.raises(ValueError):
        compare_minibatch(parent, c(1, (1,), ('x',)), ('x', 'y'))
    with pytest.raises(ValueError):
        compare_minibatch(parent, c(1, (1, 0), ('y', 'x')), ('x', 'y'))
    with pytest.raises(ValueError):
        compare_minibatch(parent, parent, ('x', 'y'))


def test_minibatch_sum_comparison_preserves_smallest_positive_improvement():
    parent = c(0, (0, 0), ('x', 'y'))
    child = c(1, (5e-324, 0), ('x', 'y'))
    # Division can round these means to the same float; the reference strict
    # criterion compares sums over the identical batch, without a tolerance.
    assert compare_minibatch(parent, child, ('x', 'y')).accepted


def test_pruning_preserves_cover_and_leaves_no_redundant_survivor_exhaustively():
    from itertools import product
    ids = ('x', 'y', 'z')
    for flat in product((0, 1), repeat=9):
        pool = tuple(c(i, flat[i * 3:(i + 1) * 3], ids) for i in range(3))
        result = cover_prune(pool, ids)
        assert all(result.retained_winners)
        for before, after in zip(result.instance_winners, result.retained_winners):
            assert set(after) <= set(before)
        survivors = set(result.sampling_indices)
        for survivor in survivors:
            assert (survivor,) in result.retained_winners
        assert sum(result.sampling_weights) == sum(map(len, result.retained_winners))
