"""TRAIN-only S1/S2 construction. No intent-indexed knowledge consumption."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def split_folds(types, edges, maximum):
    """Deterministic capacitated graph coloring: confusable pairs stay apart."""
    if maximum < 1:
        raise ValueError('max_types_per_fold must be positive')
    neighbors = {t: set() for t in types}
    for a, b in edges:
        if a != b and a in neighbors and b in neighbors:
            neighbors[a].add(b)
            neighbors[b].add(a)
    assignment = {}
    folds = []
    while len(assignment) < len(types):
        pending = set(types) - assignment.keys()
        t = min(pending, key=lambda x: (-len({assignment[n] for n in neighbors[x] if n in assignment}),
                                        -len(neighbors[x]), x))
        valid = [i for i, f in enumerate(folds) if len(f) < maximum and not neighbors[t].intersection(f)]
        i = min(valid, key=lambda k: (len(folds[k]), k)) if valid else len(folds)
        if i == len(folds):
            folds.append([])
        folds[i].append(t)
        assignment[t] = i
    return [sorted(f) for f in folds]


def definition_view(bundle, types):
    """Project locked extraction cues; intent bindings and cross-type rules stay out."""
    out = {}
    for t in types:
        candidates = []
        for v in bundle['snapshot']['values']:
            if v['value_id'].endswith('slot-contract:' + t):
                match = re.search(r'Extraction cues:\s*\n([^\n]+)', v['content'])
                if match:
                    candidates.append((match.group(1).lstrip('- ').strip(), v['value_id']))
        # Keep differing definitions visible for review rather than silently voting.
        if candidates:
            text, source = candidates[0]
            source_ids = bundle['original_sources'].get(source, [])
            components = sorted({bundle['source_components'][str(x)] for x in source_ids})
            out[t] = {'definition': text, 'definition_source': source,
                      'definition_variants': sorted({x[0] for x in candidates}),
                      'definition_status': 'projected_locked_card', 'definition_components': components}
        else:
            out[t] = {'definition': t.replace('_', ' '), 'definition_source': 'train_label',
                      'definition_variants': [], 'definition_status': 'needs_semantic_definition', 'definition_components': []}
    return out


def build(rows, bundle, policy, *, forbidden_components=()):
    """Build full TRAIN statistics and separately eligible serving memory.

    Component quarantine and held development-source exclusions are inherited.
    Query self-source exclusion is additionally applied by consumption rails.
    """
    if policy['construction_split'] != 'train':
        raise ValueError('knowledge construction requires TRAIN')
    types = sorted({e['type'] for r in rows for e in r['entities']})
    if set(types) != set(policy['kinds']):
        raise ValueError('all observed types require an explicit kind')
    blocked = set(bundle['bad_components']) | set(bundle['dev_components']) | set(forbidden_components)
    sources = bundle['source_components']
    values = {t: defaultdict(list) for t in types}
    examples = {t: [] for t in types}
    census = {t: Counter() for t in types}
    overlap = defaultdict(set)
    skipped = Counter()
    value_equals_type = 0
    for r in rows:
        tok = {x['id']: x['surface'] for x in r['tokens']}
        component = sources.get(str(r['slurp_id']))
        eligible = component is not None and component not in blocked
        if not eligible:
            skipped['blocked_or_unmapped_records'] += 1
        for e in r['entities']:
            t = e['type']
            raw = ' '.join(tok[i] for i in e['span'])
            key = normalized(raw)
            census[t][key] += 1
            if not key:
                skipped['empty_observed_values'] += 1
                continue
            overlap[key].add(t)
            value_equals_type += key in {t, t.replace('_', ' ')}
            if not eligible:
                continue
            record = {'source_id': r['slurp_id'], 'component': component, 'surface': raw}
            values[t][key].append(record)
            examples[t].append({'source_id': r['slurp_id'], 'component': component,
                                'query': r['sentence'], 'value': raw})
    edges = set()
    reasons = defaultdict(set)
    for group in policy['separation_groups']:
        for pair in itertools.combinations(sorted(set(group) & set(types)), 2):
            edges.add(pair)
            reasons[pair].add('configured_semantic_overlap')
    if policy['train_value_overlap_edges']:
        for ts in overlap.values():
            for pair in itertools.combinations(sorted(ts), 2):
                edges.add(pair)
                reasons[pair].add('same_normalized_train_surface')
    folds = split_folds(types, edges, policy['max_types_per_fold'])
    definitions = definition_view(bundle, types)
    for t, supplied in policy.get('semantic_definitions', {}).items():
        if t not in definitions or not supplied.get('definition', '').strip() or not supplied.get('source'):
            raise ValueError('semantic definition requires an observed type, text and provenance')
        source_ids = supplied.get('source_ids', [])
        current_ids = {str(r['slurp_id']) for r in rows}
        if any(str(sid) not in current_ids or str(sid) not in sources for sid in source_ids):
            raise ValueError('semantic definition references a missing TRAIN source')
        components = sorted({sources[str(sid)] for sid in source_ids})
        definitions[t] = {
            'definition': supplied['definition'], 'definition_source': supplied['source'],
            'definition_variants': [supplied['definition']],
            'definition_status': 'configured_reviewed_card',
            'definition_basis': supplied.get('basis', ''),
            'support_source_ids': source_ids,
            'definition_components': components if supplied.get('definition_depends_on_examples', True) else [],
        }
    for t, definition in definitions.items():
        if blocked.intersection(definition['definition_components']):
            definition.update(definition=t.replace('_', ' '), definition_status='quarantined_semantic_definition')
    cards = {}
    for t in types:
        records = [{'normalized': k, 'surfaces': sorted({x['surface'] for x in vs}),
                    'sources': vs} for k, vs in sorted(values[t].items())]
        # Aggregate all same-type mentions in each source query. A demonstration
        # must not teach omission of a second mention or collapse repetitions.
        per_source = {}
        for e in examples[t]:
            item = per_source.setdefault(e['source_id'], {k: e[k] for k in ('source_id', 'component', 'query')})
            item.setdefault('values', []).append(e['value'])
        distinct = {}
        for e in per_source.values():
            distinct.setdefault((normalized(e['query']), tuple(normalized(v) for v in e['values'])), e)
        cards[t] = {**definitions[t], 'type': t, 'kind': policy['kinds'][t],
                    'closed_world': t in policy['closed_world_types'],
                    'value_rules': policy.get('value_rules_by_kind', {}).get(policy['kinds'][t], [])
                                   + policy.get('value_rules_by_type', {}).get(t, []),
                    'vocabulary': records if policy['kinds'][t] == 'enumerated_observed' else [],
                    'examples': list(distinct.values()),
                    'observed_unique_values': len(census[t]), 'eligible_unique_values': len(records),
                    'missing_value_policy': 'evidenced type with unrecoverable value => null; no type evidence => []'}
    report = {'train_records': len(rows), 'types': len(types), 'folds': len(folds),
              'fold_sizes': [len(f) for f in folds], 'separation_edges': len(edges),
              'within_fold_constraint_violations': sum(a in f and b in f for a, b in edges for f in folds),
              'semantic_orthogonality_proven': False,
              'limitation': 'No configured/observed conflict edge within fold; unseen semantic confusion remains possible.',
              'kind_counts': dict(Counter(policy['kinds'].values())),
              'value_equals_type_mentions_retained': value_equals_type,
              'definition_gaps': [t for t in types if cards[t]['definition_status'] not in ('projected_locked_card', 'configured_reviewed_card')],
              'definition_conflicts': [t for t in types if len(cards[t]['definition_variants']) > 1],
              'empty_eligible_knowledge_types': [t for t in types if not cards[t]['examples']],
              'filtering': dict(skipped), 'test_data_read': False,
              'per_type': {t: {'kind': cards[t]['kind'], 'mentions': sum(census[t].values()),
                              'observed_values': len(census[t]), 'eligible_values': cards[t]['eligible_unique_values'],
                              'examples': len(cards[t]['examples'])} for t in types}}
    artifact = {'schema': 's-stages-knowledge-v1', 'construction_split': 'train',
                'folds': [{'fold_id': f'fold-{i:02d}', 'types': f} for i, f in enumerate(folds)],
                'cards': cards, 'source_components': sources,
                'separation_edges': [{'types': list(p), 'reasons': sorted(reasons[p])} for p in sorted(edges)],
                'policy': policy, 'report': report}
    lexicon_audit = {}
    for t, card in cards.items():
        if card['kind'] != 'enumerated_observed':
            continue
        lexicon_audit[t] = {
            'observed_vocabulary_not_closed_world': not card['closed_world'],
            'entries': [{'normalized': v['normalized'], 'surfaces': v['surfaces'],
                         'source_ids': sorted({s['source_id'] for s in v['sources']}),
                         'source_count': len({s['source_id'] for s in v['sources']}),
                         'other_types_same_surface': sorted(overlap[v['normalized']] - {t}),
                         'review_flags': (['single_source'] if len({s['source_id'] for s in v['sources']}) == 1 else [])
                                        + (['cross_type_surface'] if overlap[v['normalized']] - {t} else [])}
                        for v in card['vocabulary']]}
    artifact['lexicon_audit'] = lexicon_audit
    report['enum_vocabulary_entries'] = sum(len(c['vocabulary']) for c in cards.values())
    report['enum_surface_variants'] = sum(len(v['surfaces']) for c in cards.values() for v in c['vocabulary'])
    report['enum_single_source_entries'] = sum('single_source' in e['review_flags'] for a in lexicon_audit.values() for e in a['entries'])
    report['enum_cross_type_entries'] = sum('cross_type_surface' in e['review_flags'] for a in lexicon_audit.values() for e in a['entries'])
    artifact['knowledge_id'] = digest(artifact)
    return artifact


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True, type=Path)
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--policy', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    rows = [json.loads(x) for x in a.train.read_text().splitlines() if x.strip()]
    bundle = json.loads(a.bundle.read_text())
    # Extend old component quarantine with every source removed in current clean TRAIN.
    current = {str(r['slurp_id']) for r in rows}
    removed = {c for sid, c in bundle['source_components'].items() if sid not in current}
    artifact = build(rows, bundle, json.loads(a.policy.read_text()), forbidden_components=removed)
    artifact['input_pins'] = {k: hashlib.sha256(v.read_bytes()).hexdigest()
                              for k, v in {'train': a.train, 'bundle': a.bundle, 'policy': a.policy}.items()}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'knowledge.private.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + '\n')
    (a.output / 'LEXICON-AUDIT.private.json').write_text(json.dumps(artifact['lexicon_audit'], ensure_ascii=False, indent=2) + '\n')
    (a.output / 'SUMMARY.json').write_text(json.dumps(artifact['report'], ensure_ascii=False, indent=2) + '\n')
    (a.output / 'FOLDS.json').write_text(json.dumps(artifact['folds'], ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in artifact['report'].items() if k != 'per_type'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
