"""Lossless literal slot diagnostics for TRAIN reflection, not a new scorer.

Compare the supplied reference with the already-normalized scored prediction.
No additional normalization, fuzzy matching or label inference occurs here.
Official feedback stays authoritative; multiset differences are supplementary
evidence preventing a high partial score from hiding missing correct slots.
"""
from __future__ import annotations

from collections import Counter

from .tfgrpo_learning import sha


def _frame(value):
    if not isinstance(value, dict) or not {'scenario', 'action', 'entities'} <= set(value):
        return None
    if not all(isinstance(value[k], str) and value[k] for k in ('scenario', 'action')):
        return None
    if not isinstance(value['entities'], list):
        return None
    slots = []
    for row in value['entities']:
        if (not isinstance(row, dict) or set(row) != {'type', 'filler'}
                or not all(isinstance(row[k], str) and row[k] for k in ('type', 'filler'))):
            return None
        slots.append((row['type'], row['filler']))
    return value['scenario'], value['action'], Counter(slots)


def _slots(counter):
    return [{'type': kind, 'filler': filler, 'count': count}
            for (kind, filler), count in sorted(counter.items())]


def group_error_feedback(group):
    """Expose every trajectory's omissions and additions, including duplicates.

    Missing or unparseable predictions produce explicit unavailable diagnostics,
    never a fabricated empty prediction or perfect score. Identity is bound to
    original evidence; callers must independently verify TRAIN/native custody.
    """
    evidence = group['evidence']; reference = evidence['reference']
    expected = _frame(reference); rows = []
    for index, row in enumerate(evidence['rollouts']):
        prediction = row.get('scored_prediction'); actual = _frame(prediction)
        entry = {'ref': f'r{index}', 'trajectory_id': row['trajectory_id'],
                 'prediction_sha256': sha(prediction)}
        if expected is None or actual is None or row.get('prediction_observed') is not True:
            entry.update(available=False, reason='complete reference and observed scored frame required')
        else:
            matches = expected[2] & actual[2]
            missing = expected[2] - actual[2]; extra = actual[2] - expected[2]
            entry.update(available=True, intent_exact=expected[:2] == actual[:2],
                matched=_slots(matches), missing=_slots(missing), extra=_slots(extra),
                complete_frame_exact=expected == actual)
        rows.append(entry)
    return {'schema': 'saea-literal-slot-errors-v1', 'group_sha256': group['group_sha256'],
            'reference_sha256': sha(reference), 'official_score_recomputed': False,
            'additional_normalization': 'none', 'trajectories': rows}
