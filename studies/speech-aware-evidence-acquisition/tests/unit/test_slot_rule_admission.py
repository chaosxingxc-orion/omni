"""Slot-presence evidence can refute universal rules, not certify transfer."""
import copy

import pytest

from speech_aware_evidence_acquisition.rails.slot_rule_admission import (
    CANDIDATE_SCHEMA, POOL_SCHEMA, review_slot_rule,
)
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha


def evidence(key, slots, *, scenario='news', action='query', complete=True):
    return {'evidence_id': key, 'source_sha256': sha(key),
        'reference_complete': complete,
        'reference': {'scenario': scenario, 'action': action,
                      'entities': [{'type': slot, 'filler': value} for slot, value in slots]}}


def rule(slot='news_topic', operation='always_drop', refs=None):
    return {'schema': CANDIDATE_SCHEMA,
        'intent_scope': [{'scenario': 'news', 'action': 'query'}],
        'condition': None, 'operation': operation, 'slot': slot,
        'support_refs': ['support'] if refs is None else refs}


def check(candidate, rows):
    pool = {'schema': POOL_SCHEMA, 'split': 'train', 'rows': rows}
    before = copy.deepcopy((candidate, pool))
    result = review_slot_rule(candidate, pool, expected_pool_sha256=sha(pool))
    assert (candidate, pool) == before
    assert result['deployment_accepted'] is False
    assert result['generalization_verified'] is False
    assert result['new_model_calls'] == 0
    return result


@pytest.mark.parametrize('slot', ['news_topic', 'song_name', 'arbitrary_slot'])
def test_support_itself_refutes_deleting_required_slot(slot):
    r = check(rule(slot), [evidence('support', [(slot, 'reference value')])])
    assert r['status'] == 'rejected'
    assert r['support_conflict_refs'] == r['counterexample_refs'] == ['support']


def test_missing_slot_counterexample_refutes_universal_required_rule():
    r = check(rule('media_type', 'always_require'), [
        evidence('support', [('media_type', 'present')]), evidence('counterexample', [])])
    assert r['status'] == 'rejected' and r['support_conflict_refs'] == []
    assert r['counterexample_refs'] == ['counterexample']


def test_one_consistent_case_is_never_release_authority():
    r = check(rule('person', 'always_require'), [evidence('support', [('person', 'name')])])
    assert r['status'] == 'static_consistency_only'
    assert 'independent_development_validation_required' in r['reasons']


def test_scan_all_scope_evidence_not_only_cited_support():
    r = check(rule(), [evidence('support', []), evidence('other', [('news_topic', 'topic')])])
    assert r['counterexample_refs'] == ['other']


def test_different_intent_is_not_a_counterexample_to_local_scope():
    r = check(rule(), [evidence('support', []),
        evidence('outside', [('news_topic', 'topic')], action='other')])
    assert r['status'] == 'static_consistency_only' and r['checked_refs'] == ['support']


def test_cited_outside_scope_cannot_supply_support():
    r = check(rule(), [evidence('support', [], scenario='outside')])
    assert r['status'] == 'rejected'
    assert r['outside_scope_support_refs'] == ['support']


def test_unknown_condition_is_neither_confirmed_nor_refuted():
    candidate = rule(); candidate['condition'] = 'when the request concerns a person'
    r = check(candidate, [evidence('support', [('news_topic', 'topic')])])
    assert r['status'] == 'unverified' and not r['counterexample_refs']


@pytest.mark.parametrize('complete,reference', [(False, None), (True, None), (False, [])])
def test_unavailable_reference_never_becomes_empty_gold(complete, reference):
    row = evidence('support', [], complete=complete)
    if reference is None:
        row['reference'] = None
    r = check(rule(), [row])
    assert r['status'] == 'unverified' and r['unavailable_refs'] == ['support']


def test_known_counterexample_still_rejects_when_other_evidence_missing():
    r = check(rule(), [evidence('support', [('news_topic', 'topic')]),
                       evidence('incomplete', [], complete=False)])
    assert r['status'] == 'rejected'


def test_missing_support_and_legacy_text_stay_quarantined():
    rows = [evidence('support', [])]
    assert check(rule(refs=[]), rows)['status'] == 'unverified'
    assert check('never emit news_topic', rows)['reasons'] == [
        'legacy_free_text_requires_structured_hypothesis']


def test_duplicate_entities_still_establish_slot_presence():
    r = check(rule(), [evidence('support', [('news_topic', 'x'), ('news_topic', 'x')])])
    assert r['counterexample_refs'] == ['support']


def test_test_split_and_changed_pool_are_rejected_before_reading_rule():
    pool = {'schema': POOL_SCHEMA, 'split': 'test', 'rows': []}
    with pytest.raises(ValueError, match='TRAIN'):
        review_slot_rule(rule(), pool, expected_pool_sha256=sha(pool))
    pool['split'] = 'train'; digest = sha(pool)
    pool['rows'].append(evidence('support', []))
    with pytest.raises(ValueError, match='changed'):
        review_slot_rule(rule(), pool, expected_pool_sha256=digest)


def test_unknown_and_duplicate_support_cannot_impersonate_evidence():
    for refs in [['missing'], ['support', 'support']]:
        with pytest.raises(ValueError):
            check(rule(refs=refs), [evidence('support', [])])


def test_new_rule_field_cannot_hide_unvalidated_condition():
    candidate = rule(); candidate['only_if'] = 'secret condition'
    with pytest.raises(ValueError, match='structured'):
        check(candidate, [evidence('support', [])])
