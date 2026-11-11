"""Source-bound S1 knowledge consumption; scores never decide output types.

No encoder, query transcript, intent, labels of the query, transport or retry is
available here. Cached similarity is audio-query to complete TRAIN sentences.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


class S1KnowledgeConsumer:
    """Compile reference-only rules and ranked full-sentence examples per Fold.

    A cache is an immutable, source-pinned artifact loaded by the caller on CPU.
    Its version participates in request identity through this consumer's digest.
    A missing cache/query explicitly degrades to the existing card examples.
    An identity mismatch is a deployment error and is never silently downgraded.
    """

    def __init__(self, knowledge, policy, cache=None, *, provenance_root=None):
        self.knowledge = knowledge
        self.policy = copy.deepcopy(policy)
        if policy.get('schema') != 's1-consumption-policy-v1':
            raise ValueError('unsupported S1 consumption policy')
        for k in ('max_examples', 'max_example_bytes', 'max_rules', 'max_rule_bytes'):
            if type(policy.get(k)) is not int or policy[k] < 0:
                raise ValueError('nonnegative S1 budget required: ' + k)
        if policy.get('coverage_policy', 'score_order') not in ('score_order', 'type_round_robin', 'fold_supported_score_order'):
            raise ValueError('unknown retrieval coverage policy')
        self.blocked = set(knowledge.get('excluded_source_components', []))
        self.blocked.update(knowledge.get('bad_components', []))
        self.blocked.update(knowledge.get('dev_components', []))
        self.blocked.update(knowledge.get('knowledge_supply', {}).get('blocked_components', []))
        self.cache = copy.deepcopy(cache)
        self.version = digest({'policy': policy, 'cache': self.cache})
        self.queries = {}
        self.bank = {}
        self.rules = knowledge.get('s1_discriminative_rules', [])
        self.provenance_root = Path(provenance_root) if provenance_root is not None else None
        self._source_documents = {}
        self._last_audio = None
        self._validate_rules()
        for card in knowledge['cards'].values():
            if card.get('independent_definition'):
                supplied = card['independent_definition']
                self._validate_refs(supplied['source_refs'], supplied['definition'])
        self.procedural = knowledge.get('knowledge_supply', {}).get('procedural_rules', [])
        for rule in self.procedural:
            self._validate_refs(rule['source_refs'], rule['description'])
        if cache is not None:
            self._load_cache()

    def _validate_rules(self):
        seen = set()
        for rule in self.rules:
            if not rule.get('rule_id') or rule['rule_id'] in seen:
                raise ValueError('missing or duplicate discrimination rule id')
            seen.add(rule['rule_id'])
            if (not rule.get('target_types') or
                    not set(rule['target_types']) <= self.knowledge['cards'].keys() or
                    not set(rule.get('related_types', [])) <= self.knowledge['cards'].keys() or
                    not rule.get('description') or not rule.get('source_refs') or
                    'source_components' not in rule):
                raise ValueError('rule requires legal types, text and explicit source provenance')
            scope = rule.get('scope', {})
            if (scope.get('stage') != 'S1' or scope.get('intent_independent') is not True
                    or scope.get('multi_candidate') is not True or scope.get('final_adjudication') is not False
                    or scope.get('condition') != 'current_audio_evidence'):
                raise ValueError('rule requires reviewed intent-independent multi-candidate S1 scope')
            self._validate_refs(rule['source_refs'], rule['description'])

    def _resolve_ref(self, ref):
        if self.provenance_root is None:
            raise ValueError('source references require an explicit provenance root')
        if not isinstance(ref, dict) or not {'path', 'sha256', 'pointer'} <= ref.keys():
            raise ValueError('source reference requires path/byte SHA/JSON pointer')
        source = (self.provenance_root / ref['path']).resolve()
        if not source.is_relative_to(self.provenance_root.resolve()):
            raise ValueError('source reference leaves provenance root')
        key = (str(source), ref['sha256'])
        if key not in self._source_documents:
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != ref['sha256']:
                raise ValueError('knowledge source byte SHA mismatch')
            doc = [json.loads(l) for l in raw.splitlines() if l.strip()] if source.suffix == '.jsonl' else json.loads(raw)
            self._source_documents[key] = doc
        node = self._source_documents[key]
        pointer = ref['pointer']
        if pointer:
            if not pointer.startswith('/'):
                raise ValueError('invalid JSON source pointer')
            for segment in pointer[1:].split('/'):
                token = segment.replace('~1', '/').replace('~0', '~')
                node = node[int(token)] if isinstance(node, list) else node[token]
        return node

    def _validate_refs(self, refs, text):
        if not refs:
            raise ValueError('missing source references')
        for ref in refs:
            node = self._resolve_ref(ref)
            if not isinstance(node, str) or text not in node:
                raise ValueError('knowledge reference does not contain literal supplied text')

    def _validate_bank_row(self, row):
        for ref in row['source_refs']:
            original = self._resolve_ref(ref)
            if (not isinstance(original, dict) or str(original.get('slurp_id')) != str(row['source_id'])
                    or original.get('sentence') != row['query']
                    or set(e['type'] for e in original['entities']) != set(row['observed_types'])):
                raise ValueError('retrieval sentence/types differ from original TRAIN row')

    def _load_cache(self):
        cache = self.cache
        if cache.get('schema') != 's1-audio-sentence-retrieval-cache-v1':
            raise ValueError('unsupported audio sentence cache')
        content = {k: v for k, v in cache.items() if k != 'cache_id'}
        if cache.get('cache_id') != digest(content):
            raise ValueError('retrieval cache digest mismatch')
        if self.policy.get('cache_id') != cache['cache_id']:
            raise ValueError('runtime policy must pin exact retrieval cache')
        if cache.get('knowledge_id') != self.knowledge['knowledge_id']:
            raise ValueError('retrieval cache knowledge version mismatch')
        self.blocked.update(cache.get('excluded_source_components', []))
        encoder = cache.get('encoder', {})
        if (encoder.get('query_modality') != 'audio' or encoder.get('key_unit') != 'full_train_sentence'
                or not encoder.get('identity_sha256') or not cache.get('input_pins')):
            raise ValueError('cache lacks audio/full-sentence encoder source identity')
        expected_identity = self.policy.get('encoder_identity_sha256')
        if expected_identity != encoder['identity_sha256']:
            raise ValueError('encoder identity mismatch')
        # The only model-visible text comes from explicitly bound TRAIN bank rows.
        allowed = {'source_id', 'source_component', 'query', 'observed_types', 'source_refs', 'sentence_sha256'}
        for row in cache.get('bank', []):
            if set(row) - allowed or not allowed <= row.keys():
                raise ValueError('retrieval bank schema contains unauthorized fields')
            sid = str(row['source_id'])
            if sid in self.bank or not row['source_refs'] or not row['source_component']:
                raise ValueError('duplicate/unbound retrieval source')
            if self.knowledge['source_components'].get(sid) != row['source_component']:
                raise ValueError('retrieval source component mismatch')
            if row['source_component'] in self.blocked:
                raise ValueError('blocked source retained in retrieval bank')
            if row['sentence_sha256'] != hashlib.sha256(row['query'].encode()).hexdigest():
                raise ValueError('retrieval sentence hash mismatch')
            if not set(row['observed_types']) <= self.knowledge['cards'].keys():
                raise ValueError('unknown retrieval support type')
            self._validate_bank_row(row)
            self.bank[sid] = row
        query_fields = {'sample_id', 'source_component', 'audio_sha256', 'audio_object_sha256', 'excluded_source_components', 'neighbors'}
        for row in cache.get('queries', []):
            if set(row) != query_fields or row['sample_id'] in self.queries:
                raise ValueError('query cache must contain identity/audio/ranking only')
            if not row['audio_sha256'] or not row['audio_object_sha256']:
                raise ValueError('cache query lacks audio identity')
            neighbors = row['neighbors']
            seen = set()
            previous = None
            for neighbor in neighbors:
                if set(neighbor) != {'source_id', 'score'}:
                    raise ValueError('ranking fields must be source identity and score only')
                sid = str(neighbor['source_id'])
                score = neighbor['score']
                if (sid not in self.bank or sid in seen or isinstance(score, bool)
                        or not isinstance(score, (int, float)) or not math.isfinite(score)):
                    raise ValueError('invalid cached neighbor identity/score')
                key = (-score, sid)
                if previous is not None and key < previous:
                    raise ValueError('cache neighbors must be signed-score ordered')
                previous = key
                seen.add(sid)
            self.queries[row['sample_id']] = row

    def rules_for(self, types, component):
        selected, audit = [], {'eligible': 0, 'source_excluded': 0, 'budget_excluded': 0}
        for rule in self.rules:
            if not set(rule['target_types']).intersection(types):
                continue
            if set(rule['source_components']).intersection(self.blocked | {component}):
                audit['source_excluded'] += 1
                continue
            audit['eligible'] += 1
            view = {k: copy.deepcopy(rule[k]) for k in ('rule_id', 'target_types', 'related_types', 'description', 'source_refs')}
            # Cross-Fold references describe evidence; output scope remains enum.
            view['target_types'] = [t for t in rule['target_types'] if t in types]
            if len(selected) >= self.policy['max_rules'] or len(canonical(selected + [view])) > self.policy['max_rule_bytes']:
                audit['budget_excluded'] += 1
                continue
            selected.append(view)
        return selected, audit

    def examples_for(self, query, types, component):
        audit = {'cache_version': self.cache.get('cache_id') if self.cache else None,
                 'status': 'DEGRADED_NO_CACHE', 'selected': 0, 'covered_types': [],
                 'candidate_pool_count': 0, 'fold_supported_count': 0,
                 'uncovered_types': list(types), 'source_excluded': 0,
                 'duplicate_excluded': 0, 'budget_excluded': 0}
        if self.cache is None:
            return [], audit
        cached = self.queries.get(query.sample_id)
        if cached is None:
            audit['status'] = 'DEGRADED_QUERY_NOT_CACHED'
            return [], audit
        token = (query.sample_id, tuple(sorted((k, tuple(sorted(v.items())) if isinstance(v, dict) else v)
                                               for k, v in query.audio.items())))
        if self._last_audio is None or self._last_audio[0] != token:
            self._last_audio = (token, digest(query.audio))
        if (cached['source_component'] != component or self._last_audio[1] != cached['audio_object_sha256']
                or (getattr(query, 'audio_sha256', None) is not None and
                    query.audio_sha256 != cached['audio_sha256'])):
            raise ValueError('query audio/sample/source does not match pinned retrieval cache')
        selected, seen_components, seen_sentences = [], set(), set()
        candidates = []
        fold_supported = self.policy.get('coverage_policy') == 'fold_supported_score_order'
        # Never search beyond the original cached top32, even for rare types.
        neighbors = cached['neighbors'][:32] if fold_supported else cached['neighbors']
        for neighbor in neighbors:
            row = self.bank[str(neighbor['source_id'])]
            if row['source_component'] in self.blocked | {component} | set(cached['excluded_source_components']):
                audit['source_excluded'] += 1
                continue
            support = [t for t in types if t in row['observed_types']]
            audit['candidate_pool_count'] += 1
            audit['fold_supported_count'] += bool(support)
            # Selecting positive reference evidence does not make unlisted labels
            # negative. The legacy score_order path keeps its original wire.
            if fold_supported and not support:
                continue
            view = {'query': row['query'], 'observed_support_types': support,
                    'annotation_scope': 'Only observed TRAIN labels are shown; unlisted types are not negative evidence.'}
            candidates.append((neighbor, row, view))
        if self.policy.get('coverage_policy', 'score_order') == 'type_round_robin':
            loss = self.policy.get('coverage_max_score_loss')
            if not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
                raise ValueError('type coverage needs a finite nonnegative relevance bound')
            ordered, used = [], set()
            start = int(hashlib.sha256(query.sample_id.encode()).hexdigest()[:8], 16) % len(types)
            rotated = list(types[start:]) + list(types[:start])
            best = candidates[0][0]['score'] if candidates else 0
            for t in rotated:
                first = next((i for i, (n, _, v) in enumerate(candidates) if i not in used and
                              n['score'] >= best-loss and t in v['observed_support_types']), None)
                if first is not None:
                    ordered.append(candidates[first]); used.add(first)
            candidates = ordered + [v for i, v in enumerate(candidates) if i not in used]
        provenance = []
        for neighbor, row, view in candidates:
            sentence = ' '.join(row['query'].casefold().split())
            if row['source_component'] in seen_components or sentence in seen_sentences:
                audit['duplicate_excluded'] += 1
                continue
            if len(selected) >= self.policy['max_examples'] or len(canonical(selected + [view])) > self.policy['max_example_bytes']:
                audit['budget_excluded'] += 1
                continue
            selected.append(view)
            seen_components.add(row['source_component']); seen_sentences.add(sentence)
            provenance.append({'source_id': row['source_id'], 'source_component': row['source_component'],
                               'source_refs': row['source_refs'], 'score': neighbor['score']})
        covered = sorted({t for v in selected for t in v['observed_support_types']})
        audit.update(status='READY' if selected else 'DEGRADED_NO_ELIGIBLE_EXAMPLES', selected=len(selected),
                     covered_types=covered, uncovered_types=[t for t in types if t not in covered],
                     provenance=provenance, query_audio_sha256=cached['audio_sha256'])
        if fold_supported and not audit['fold_supported_count']:
            audit['status'] = 'DEGRADED_NO_FOLD_SUPPORT'
        return selected, audit

    def retain_static_examples(self, cards, reference, receipt):
        """Retain uncovered static evidence, with bounded, sentence-deduped text.

        Static sentences keep their original support labels and source isolation.
        Repeated static sentences are printed once; supports_types preserves all
        their original positive labels. No support label is inferred from absence.
        """
        audit = receipt['retrieval']
        original = [copy.deepcopy(e) for card in cards for e in card['examples']]
        covered = set(audit['covered_types'])
        static = {}
        retained = set()
        for card in cards:
            examples = card['examples'] if card['type'] not in covered else []
            card['examples'] = []
            for example in examples:
                retained.add(card['type'])
                key = ' '.join(example['query'].casefold().split())
                if key in static:
                    first = static[key]
                    first['supports_types'] = sorted(set(first.get('supports_types', [first['supports_type']])) | {card['type']})
                else:
                    static[key] = example
                    card['examples'].append(example)
        # A dynamic sentence already contains all its observed Fold labels.
        # If it duplicates a retained static sentence, display once and preserve
        # both sets of positive supports (the static provenance was pre-isolated).
        dynamic = []
        duplicates = 0
        for example in reference['retrieved_examples']:
            key = ' '.join(example['query'].casefold().split())
            if key in static:
                first = static[key]
                first['supports_types'] = sorted(set(first.get('supports_types', [first['supports_type']])) |
                                                set(example['observed_support_types']))
                duplicates += 1
            else:
                dynamic.append(example)
        reference['retrieved_examples'] = dynamic
        # Total context is bounded by the original static inventory plus the
        # configured dynamic budget; no fallback fetches or additional examples.
        final_static = [e for card in cards for e in card['examples']]
        audit.update(static_retained_types=sorted(retained),
                     static_example_count=len(final_static), dynamic_displayed_count=len(dynamic),
                     sentence_duplicates_merged=len(original) - len(final_static) -
                         sum(1 for e in original if e['supports_type'] in covered) + duplicates,
                     total_example_count=len(final_static) + len(dynamic),
                     total_example_limit=len(original) + self.policy['max_examples'],
                     total_example_bytes=len(canonical(final_static)) + len(canonical(dynamic)),
                     total_example_byte_limit=len(canonical(original)) + self.policy['max_example_bytes'] +
                         len(original) * (len(canonical([c['type'] for c in cards])) + 64) + 2,
                     budget_policy='original_static_inventory_plus_dynamic_budget; merge_equal_sentences_preserving_supports')

    def before(self, query, types, component):
        rules, rule_audit = self.rules_for(types, component)
        examples, retrieval_audit = self.examples_for(query, types, component)
        procedural = [{k: copy.deepcopy(rule[k]) for k in ['description', 'source_refs']}
                      for rule in self.procedural if not set(rule['source_components']).intersection(self.blocked | {component})]
        return {'discriminative_rules': rules, 'retrieved_examples': examples, 'procedural_rules': procedural,
                'reference_scope': 'Cards, rules and examples are reference knowledge; only the target audio is evidence. '
                                   'Cross-type distinctions do not impose exclusivity. Return only supported types from this Fold.'}, {
                'schema': 's1-consumption-receipt-v1', 'consumer_version': self.version,
                'policy_id': self.policy.get('policy_id'), 'knowledge_id': self.knowledge['knowledge_id'],
                'rules': rule_audit, 'retrieval': retrieval_audit}
