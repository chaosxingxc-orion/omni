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



def source_document(root, reference, cache=None):
    """Resolve a byte-pinned local JSON/JSONL source; never fetch a source."""
    cache = {} if cache is None else cache
    rel = Path(reference['path'])
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('source path must be workspace relative')
    key = (str(rel), reference['sha256'])
    if key not in cache:
        raw = (Path(root) / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference['sha256']:
            raise ValueError('source SHA mismatch: ' + str(rel))
        cache[key] = ([json.loads(line) for line in raw.splitlines() if line.strip()]
                      if rel.suffix == '.jsonl' else json.loads(raw))
    value = cache[key]
    pointer = reference.get('pointer', '')
    if pointer and not pointer.startswith('/'):
        raise ValueError('invalid JSON source pointer')
    for part in pointer.split('/')[1:]:
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def build_s1_supply(knowledge, bundle, rows, policy, root):
    """Compile source-bound S1 cards/rules/whole-sentence retrieval inputs.

    Does not access query transcripts, labels, embeddings, transports or TEST.
    The caller supplies clean TRAIN construction rows and pinned source policy.
    Consumer owns query-component exclusion; dependencies are never erased.
    """
    import copy
    if knowledge['construction_split'] != 'train' or policy['construction_split'] != 'train':
        raise ValueError('S1 supply construction requires TRAIN')
    if policy['schema'] != 's1-knowledge-supply-policy-v1':
        raise ValueError('unknown S1 supply policy')
    cache = {}
    def resolve(ref):
        return source_document(root, ref, cache)
    refs = policy['sources']
    locked = resolve(refs['bundle'])
    train = resolve(refs['train'])
    support = resolve(refs['author_support'])['output']['slot_support']
    if locked != bundle or train != rows:
        raise ValueError('construction rows/bundle differ from pinned sources')
    glossary = resolve(refs['generic'])
    if glossary.get('schema') != 'saea-asrbias-slot-type-descriptions-v1':
        raise ValueError('generic source is not the fixed type glossary')
    types = set(knowledge['cards'])
    root_dependencies = {t: set() for t in types}
    for r in support:
        uid = str(r['utterance_id'])
        component = bundle['source_components'].get(uid)
        if component is None:
            raise ValueError('unmapped original author support')
        for e in r['entities']:
            if e['entity_type'] in types:
                root_dependencies[e['entity_type']].add(component)
    current = {str(r['slurp_id']) for r in rows}
    removed = {c for sid,c in bundle['source_components'].items() if sid not in current}
    blocked = set(bundle['bad_components']) | set(bundle['dev_components']) | removed
    out = copy.deepcopy(knowledge)
    rules = []
    rule_audit = []
    reviewed = policy.get('rule_scope_reviews', {})
    variants = {}
    for t in sorted(types):
        selected = []
        deps = set(root_dependencies[t])
        for i,v in enumerate(bundle['snapshot']['values']):
            if v['value_id'].endswith('slot-contract:' + t):
                selected.append((i,v))
                deps.update(bundle['source_components'][u] for u in bundle['original_sources'][v['value_id']])
        variants[t] = len(selected)
        card = out['cards'][t]
        if policy.get('independent_definition_enabled', True) and t in glossary['types']:
            ref = dict(refs['generic'], pointer='/types/' + t)
            text = resolve(ref)
            if not isinstance(text,str) or not text.strip():
                raise ValueError('generic type meaning must be nonempty text')
            card['independent_definition'] = dict(definition=text, source_refs=[ref],
                source_components=[], scope='generic_type_meaning')
        if not policy.get('discriminative_rules_enabled', True) or not selected:
            continue
        # First card is the original builder's selected source. Retain every
        # original/root dependency, even if another card has the same text.
        i,v = selected[0]
        content = v['content']
        cues = content.split('Extraction cues:\n',1)[1].split('\nMissing behavior:',1)[0].splitlines()
        statements = [('extraction_boundary',line[2:]) for line in cues[1:] if line.startswith('- ')]
        for label,kind in [('Ambiguity behavior:','ambiguity'),('Missing behavior:','presence')]:
            if label in content:
                statements.append((kind,content.split(label,1)[1].split('\n',1)[0].strip()))
        seen = set()
        for kind,text in statements:
            if not text or text in seen:
                continue
            seen.add(text)
            ref = dict(refs['bundle'], pointer='/snapshot/values/' + str(i) + '/content')
            if text not in resolve(ref):
                raise ValueError('rule text is not literal pinned source text')
            related = []
            for other in sorted(types - {t}):
                forms = [other, other.replace('_',' ')]
                if any(re.search(r'(?<![a-z_])' + re.escape(f) + r'(?![a-z_])',text,flags=re.I) for f in forms):
                    related.append(other)
            rule_id = 's1-rule-' + digest([t,kind,text])[:20]
            review = reviewed.get(rule_id)
            if review is None or review.get('description_sha256') != hashlib.sha256(text.encode()).hexdigest():
                raise ValueError('every extracted rule requires exact-text task scope review')
            if review.get('status') not in ('applicable', 'inapplicable') or not review.get('reason'):
                raise ValueError('scope review requires explicit status and reason')
            rule_audit.append(dict(rule_id=rule_id,target_types=[t],description=text,kind=kind,
                source_refs=[ref],source_components=sorted(deps),scope_review=copy.deepcopy(review)))
            if review['status'] != 'applicable':
                continue
            rules.append(dict(rule_id=rule_id,target_types=[t],
                related_types=related,description=text,source_refs=[ref],source_components=sorted(deps),
                scope=dict(stage='S1',intent_independent=True,multi_candidate=True,condition='current_audio_evidence',final_adjudication=False),
                kind=kind,source_mode='literal_locked_card_with_conservative_author_dependency_union'))
    bank = []
    if policy.get('retrieval_examples_enabled', True):
        for i,r in enumerate(rows):
            component = bundle['source_components'].get(str(r['slurp_id']))
            if component is None or component in blocked:
                continue
            observed = sorted({e['type'] for e in r['entities']})
            if not observed:
                continue
            if not set(observed) <= types:
                raise ValueError('retrieval example outside global type inventory')
            ref = dict(refs['train'], pointer='/' + str(i))
            if resolve(ref) != r:
                raise ValueError('retrieval example differs from pinned TRAIN source')
            bank.append(dict(sample_id='slurp:train:' + str(r['slurp_id']).zfill(6),
                source_id=r['slurp_id'],source_component=component,audio_sha256=None,
                query=r['sentence'],observed_types=observed,source_refs=[ref],
                recordings=[x['file'] for x in r.get('recordings',[])],
                annotation_scope='observed positive types only; unmentioned types are not negatives'))
    out['s1_discriminative_rules'] = rules
    out['s1_rule_scope_audit'] = rule_audit
    out['retrieval_examples'] = bank
    procedural = []
    for key in ('coexistence','inventory','demonstration','presence'):
        ref = dict(refs['bundle'], pointer='/policy/'+key)
        procedural.append(dict(description=resolve(ref),source_refs=[ref],source_components=[]))
    out['knowledge_supply'] = dict(schema='s1-knowledge-supply-v1',policy=copy.deepcopy(policy),
        blocked_components=sorted(blocked),procedural_rules=procedural,
        fold_boundary_instruction='Select only types supported by the current audio in this Fold. Cross-Fold references do not impose mutual exclusion. When no business type is supported, use the configured empty-selection protocol. Its control label is not a business slot type.',
        source_validation=dict(status='PASS_BYTE_SHA_AND_JSON_POINTER',files=len(cache),
            rule_texts=len(rules),retrieval_rows=len(bank),test_data_read=False),
        gaps=dict(independent_definition_missing=sorted(types-set(glossary['types'])),
            locked_rules_missing=sorted(t for t,n in variants.items() if not n),
            audio_hashes_pending=len(bank),minimal_sentence_dependencies_not_proven=True))
    out.pop('knowledge_id', None)
    out['knowledge_id'] = digest(out)
    return out


def supply_main(argv=None):
    """Separate optional CPU compiler; the original build CLI is unchanged."""
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', required=True, type=Path)
    p.add_argument('--knowledge', required=True, type=Path)
    p.add_argument('--policy', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args(argv)
    policy = json.loads(a.policy.read_bytes())
    bundle = source_document(a.workspace, policy['sources']['bundle'])
    rows = source_document(a.workspace, policy['sources']['train'])
    knowledge = json.loads(a.knowledge.read_bytes())
    expected = policy['parent_knowledge']
    if hashlib.sha256(a.knowledge.read_bytes()).hexdigest() != expected['sha256']:
        raise ValueError('parent knowledge SHA mismatch')
    result = build_s1_supply(knowledge,bundle,rows,policy,a.workspace)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status='S1_SUPPLY_COMPILED',types=len(result['cards']),folds=len(result['folds']),
        rules=len(result['s1_discriminative_rules']),retrieval_examples=len(result['retrieval_examples']),
        model_calls=0,source_validation=result['knowledge_supply']['source_validation'])))

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
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'supply':
        supply_main(sys.argv[2:])
    else:
        main()
