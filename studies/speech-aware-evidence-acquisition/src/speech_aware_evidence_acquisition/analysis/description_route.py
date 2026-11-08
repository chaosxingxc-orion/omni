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
    if merge_policy not in {'replace_tail', 'fill_vacancies_only', 'strict_description_dominance'}:
        raise ValueError('unsupported merge policy')
    if len(parent) > cap or len(set(parent)) != len(parent):
        raise ValueError('parent must be a unique cap5 list')
    if math.isnan(threshold) or threshold == -math.inf:
        raise ValueError('invalid threshold')
    excluded = frozenset(excluded_groups)
    if not excluded or any(not x for x in excluded):
        raise ValueError('query source groups must be resolved')
    candidates = list(candidates)
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
    if merge_policy == 'strict_description_dominance':
        # The complete candidate bank supplies parent scores as well as novel scores.
        # A missing, masked or ambiguous parent card cannot authorize replacement.
        by_label = {}
        for c in candidates:
            by_label.setdefault(c.label, []).append(c)
        remaining = list(parent)
        selected = []
        for label in added:
            new = next(c for c in eligible if c.label == label)
            if len(remaining) + len(selected) < cap:
                selected.append(label)
                continue
            if len(remaining) <= min(3, len(parent)):
                continue
            victim = remaining[-1]
            matches = by_label.get(victim, [])
            if len(matches) != 1:
                continue
            old = matches[0]
            if (old.knowledge_id and old.provenance_complete and old.source_groups
                    and all(old.source_groups) and not old.source_groups & excluded
                    and math.isfinite(old.score) and -1 <= old.score <= 1
                    and new.score > old.score):
                remaining.pop()
                selected.append(label)
        return remaining + selected, audit
    return list(parent[:cap-len(added)]) + added, audit
