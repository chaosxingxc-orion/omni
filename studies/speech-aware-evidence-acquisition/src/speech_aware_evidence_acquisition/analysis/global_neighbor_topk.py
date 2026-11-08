"""Global cosine-neighbor aggregation; no gold enters this module."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Config:
    k: int = 5
    source_distinct: bool = True
    include_sum: bool = True
    high_confidence_enabled: bool = False


def aggregate(scores, members, config, excluded_groups=()):
    if config.high_confidence_enabled:
        raise NotImplementedError('TRAIN source-held-out calibration is required')
    if config.k < 1 or len(scores) != len(members):
        raise ValueError('invalid K or member count')
    s = np.asarray(scores, dtype=float)
    if not np.isfinite(s).all() or np.any(np.abs(s) > 1.000001):
        raise ValueError('expected finite cosine scores')
    # Stable ties follow original pinned matrix column order.
    order = np.argsort(-s, kind='stable')
    seen, neighbors = set(), []
    excluded = set(excluded_groups)
    for i in order:
        m = members[int(i)]; group = m['source_group']
        if group in excluded or (config.source_distinct and group in seen):
            continue
        seen.add(group)
        neighbors.append({'member_index': int(i), 'label': m['intent'],
                          'source_group': group, 'cosine': float(s[i])})
        if len(neighbors) == config.k:
            break
    if len(neighbors) < config.k:
        raise ValueError('fewer than K legal neighbors')
    labels = {}
    for neighbor in neighbors:
        label = neighbor['label']
        if label not in labels:
            labels[label] = {'label': label, 'max_similarity': neighbor['cosine'],
                             'count': 0, 'evidence': []}
        entry = labels[label]; entry['count'] += 1; entry['evidence'].append(neighbor)
    result = list(labels.values())  # insertion order = max cosine, stable ties
    if config.include_sum:
        for entry in result:
            entry['sum_similarity'] = sum(x['cosine'] for x in entry['evidence'])
    return {'neighbors': neighbors, 'labels': result, 'direct_decision': None}
