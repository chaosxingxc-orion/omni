"""Deterministic additive description route; caller supplies calibrated scores and provenance."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DescriptionCandidate:
    label: str
    knowledge_id: str
    score: float
    source_groups: frozenset[str]
    provenance_complete: bool


def merge_description(parent, candidates, excluded_groups, *, threshold, max_additions=2, cap=5, merge_policy='replace_tail'):
    """Return labels and audit; parent is the already capped ordered question list."""
    if cap != 5 or not 0 <= max_additions <= 2:
        raise ValueError('unsupported cap or additions')
    if merge_policy not in {'replace_tail', 'fill_vacancies_only'}:
        raise ValueError('unsupported merge policy')
    if len(parent) > cap or len(set(parent)) != len(parent):
        raise ValueError('parent must be a unique cap5 list')
    if math.isnan(threshold) or threshold == -math.inf:
        raise ValueError('invalid threshold')
    excluded = frozenset(excluded_groups)
    if not excluded or any(not x for x in excluded):
        raise ValueError('query source groups must be resolved')
    labels = set(parent)
    eligible = []
    audit = {'unresolved': 0, 'excluded': 0, 'invalid_score': 0}
    for c in candidates:
        if not c.label or not c.knowledge_id or not c.provenance_complete or not c.source_groups or any(not x for x in c.source_groups):
            audit['unresolved'] += 1
        elif c.source_groups & excluded:
            audit['excluded'] += 1
        elif not math.isfinite(c.score) or not -1 <= c.score <= 1:
            audit['invalid_score'] += 1
        elif c.label not in labels and c.score >= threshold:
            eligible.append(c)
    eligible.sort(key=lambda c: (-c.score, c.knowledge_id, c.label))
    added = []
    for c in eligible:
        if c.label not in labels and len(added) < max_additions:
            labels.add(c.label)
            added.append(c.label)
    if merge_policy == 'fill_vacancies_only':
        return list(parent) + added[:cap-len(parent)], audit
    return list(parent[:cap-len(added)]) + added, audit
