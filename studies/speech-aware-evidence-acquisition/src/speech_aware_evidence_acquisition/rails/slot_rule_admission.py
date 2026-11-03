"""CPU evidence checks for structured TRAIN slot-rule hypotheses.

This is a static gate, not a natural-language verifier or deployment authority.
The caller supplies an independently verified, pinned TRAIN evidence pool;
matching its digest does not establish native journal custody. Existing free
text is never silently interpreted as an executable rule.
"""
from __future__ import annotations

import re

from .slot_error_feedback import _frame
from .tfgrpo_learning import sha

CANDIDATE_SCHEMA = 'saea-slot-rule-candidate-v1'
POOL_SCHEMA = 'saea-slot-rule-evidence-v1'
RESULT_SCHEMA = 'saea-slot-rule-admission-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def evidence_rows(pool, expected_pool_sha256):
    require(isinstance(expected_pool_sha256, str)
            and re.fullmatch('[a-f0-9]{64}', expected_pool_sha256) is not None,
            'externally pinned evidence digest required')
    require(sha(pool) == expected_pool_sha256, 'evidence pool changed')
    require(isinstance(pool, dict) and set(pool) == {'schema', 'split', 'rows'}
            and pool['schema'] == POOL_SCHEMA and pool['split'] == 'train',
            'exact TRAIN evidence pool required')
    require(isinstance(pool['rows'], list), 'evidence rows must be a list')
    rows = {}
    for row in pool['rows']:
        require(isinstance(row, dict)
                and set(row) == {'evidence_id', 'source_sha256', 'reference_complete', 'reference'},
                'exact evidence row required')
        key = row['evidence_id']
        require(text(key) and key not in rows, 'unique evidence IDs required')
        require(isinstance(row['source_sha256'], str)
                and re.fullmatch('[a-f0-9]{64}', row['source_sha256']) is not None,
                'source digest required')
        require(type(row['reference_complete']) is bool, 'explicit reference completeness required')
        rows[key] = row
    return rows


def review_slot_rule(candidate, pool, *, expected_pool_sha256):
    """Return rejection, unverified, or static-consistency-only; never release.

    Supported scope is a set of explicit scenario/action pairs, with no other
    condition. References use the same literal entity multiset parser as slot
    error feedback. Missing/incomplete evidence is never an empty gold frame.
    A complete pool scan is recorded, but coverage outside the supplied pool
    and independent development performance remain unverified.
    """
    rows = evidence_rows(pool, expected_pool_sha256)
    result = {'schema': RESULT_SCHEMA, 'candidate_sha256': sha(candidate),
        'evidence_pool_sha256': expected_pool_sha256, 'status': 'unverified',
        'reasons': [], 'support_refs': [], 'checked_refs': [],
        'counterexample_refs': [], 'support_conflict_refs': [],
        'unavailable_refs': [], 'deployment_accepted': False,
        'generalization_verified': False, 'native_custody_verified': False,
        'new_model_calls': 0}
    if isinstance(candidate, str):
        result['reasons'] = ['legacy_free_text_requires_structured_hypothesis']
        return result
    require(isinstance(candidate, dict) and set(candidate) == {
        'schema', 'intent_scope', 'condition', 'operation', 'slot', 'support_refs'}
        and candidate['schema'] == CANDIDATE_SCHEMA, 'exact structured candidate required')
    require(candidate['operation'] in ('always_drop', 'always_require')
            and text(candidate['slot']), 'supported slot operation required')
    scope = candidate['intent_scope']
    require(isinstance(scope, list) and bool(scope), 'nonempty explicit intent scope required')
    pairs = []
    for intent in scope:
        require(isinstance(intent, dict) and set(intent) == {'scenario', 'action'}
                and all(text(value) for value in intent.values()), 'exact intent pair required')
        pairs.append((intent['scenario'], intent['action']))
    require(len(set(pairs)) == len(pairs), 'duplicate intent scope')
    refs = candidate['support_refs']
    require(isinstance(refs, list) and all(text(ref) for ref in refs)
            and len(set(refs)) == len(refs), 'unique support references required')
    require(all(ref in rows for ref in refs), 'support must name supplied TRAIN evidence')
    result['support_refs'] = list(refs)
    require(candidate['condition'] is None or text(candidate['condition']),
            'condition must be null or nonempty text')
    if candidate['condition'] is not None:
        result['reasons'] = ['natural_language_condition_not_machine_verified']
        return result
    if not refs:
        result['reasons'] = ['missing_per_rule_support']
        return result
    outside = []
    for key, row in rows.items():
        frame = _frame(row['reference'])
        if frame is None or not row['reference_complete']:
            result['unavailable_refs'].append(key)
            continue
        if frame[:2] not in pairs:
            if key in refs:
                outside.append(key)
            continue
        result['checked_refs'].append(key)
        has_slot = any(kind == candidate['slot'] for kind, _ in frame[2])
        contradicts = has_slot if candidate['operation'] == 'always_drop' else not has_slot
        if contradicts:
            result['counterexample_refs'].append(key)
            if key in refs:
                result['support_conflict_refs'].append(key)
    if result['counterexample_refs'] or outside:
        result['status'] = 'rejected'
        if result['counterexample_refs']:
            result['reasons'].append('complete_reference_contradicts_rule')
        if outside:
            result['reasons'].append('support_outside_declared_scope')
            result['outside_scope_support_refs'] = outside
    elif result['unavailable_refs']:
        result['reasons'] = ['incomplete_evidence_prevents_static_clearance']
    else:
        result['status'] = 'static_consistency_only'
        result['reasons'] = ['no_counterexample_in_supplied_pool',
                             'independent_development_validation_required']
    return result
