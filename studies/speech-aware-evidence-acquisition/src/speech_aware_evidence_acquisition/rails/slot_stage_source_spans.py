"""Reference-only positive spans, after the original example admission/budget.

Source identities and annotation pointers stay in the receipt. The model sees
only existing sentences, their already visible positive labels and span text.
"""
from __future__ import annotations

import copy
from collections import Counter


SPAN_SCOPE = ('Positive spans locate support inside the reference sentence only; '
              'unmarked types are unlabelled, not negative. Judge the target audio '
              'independently: implicit roles and unknown or unrecoverable values '
              'remain eligible; matching these reference words is not required.')


class PositiveSourceSpanLayout:
    def __init__(self, knowledge, config, consumer):
        if (not isinstance(config, dict) or set(config) != {'layout', 'annotation_source'}
                or config['layout'] != 'positive_source_spans_v1' or consumer is None):
            raise ValueError('positive source spans require pinned annotations and source consumer')
        self.knowledge, self.consumer = knowledge, consumer
        self.pin = copy.deepcopy(config['annotation_source'])
        if set(self.pin) != {'path', 'sha256'}:
            raise ValueError('annotation source requires path and byte SHA')
        rows = consumer._resolve_ref({**self.pin, 'pointer': ''})
        if not isinstance(rows, list):
            raise ValueError('annotation source must contain TRAIN rows')
        self.rows = {}
        for line, row in enumerate(rows):
            sid = str(row.get('slurp_id'))
            self.rows.setdefault(sid, []).append((line, row))
        self.static = {}
        for card in knowledge['cards'].values():
            for example in card.get('examples', []):
                self.static.setdefault((example['query'], card['type']), []).append(example)
        self.cache = {}

    def _annotation(self, sentence, supports, sources):
        """No query fields are accepted; every source must resolve unanimously."""
        key = (sentence, tuple(supports), tuple(sorted((str(s['source_id']), s['source_component']) for s in sources)))
        if key in self.cache:
            return copy.deepcopy(self.cache[key])
        signatures, refs = [], []
        for source in sources:
            sid, component = str(source['source_id']), source['source_component']
            if self.knowledge['source_components'].get(sid) != component:
                return None, [], 'source_component_mismatch'
            matches = self.rows.get(sid, [])
            if not matches:
                return None, [], 'missing_annotation'
            for line, row in matches:
                tokens = row.get('tokens', [])
                surfaces = [t.get('surface') for t in tokens]
                if (row.get('sentence') != sentence or not tokens
                        or any(not isinstance(s, str) or not s for s in surfaces)
                        or [t.get('id') for t in tokens] != list(range(len(tokens)))
                        or ' '.join(surfaces) != sentence):
                    return None, [], 'ambiguous_sentence_tokens'
                visible = {}
                for slot_type in supports:
                    spans = [e.get('span') for e in row.get('entities', []) if e.get('type') == slot_type]
                    if not spans:
                        return None, [], 'missing_positive_span'
                    for span in spans:
                        if (not isinstance(span, list) or not span
                                or any(type(i) is not int or i < 0 or i >= len(tokens) for i in span)
                                or span != list(range(span[0], span[-1] + 1))):
                            return None, [], 'invalid_positive_span'
                    visible[slot_type] = spans
                signature = (surfaces, visible)
                if signatures and signature != signatures[0]:
                    return None, [], 'ambiguous_positive_spans'
                signatures.append(signature)
                refs.append({**self.pin, 'pointer': '/' + str(line), 'source_id': sid,
                             'source_component': component,
                             'visible_token_spans': copy.deepcopy(visible)})
        if not signatures:
            return None, [], 'missing_source'
        surfaces, visible = signatures[0]
        result = ({t: [' '.join(surfaces[i] for i in span) for span in visible[t]] for t in supports}, refs, 'annotated')
        self.cache[key] = copy.deepcopy(result)
        return result

    def apply(self, content, receipt, component, sample_id):
        audit = {'layout': 'positive_source_spans_v1', 'examples': [], 'counts': Counter()}
        retrieval = receipt['retrieval']
        blocked = self.consumer.blocked | {component}
        cached = self.consumer.queries.get(sample_id)
        if cached:
            blocked |= set(cached['excluded_source_components'])
        dynamic_sources = retrieval.get('provenance', [])
        for location, example, supports, sources in self._examples(content, dynamic_sources):
            entry = {'location': location, 'visible_support_types': supports,
                     'original_source_components': sorted({s['source_component'] for s in sources}),
                     'original_source_refs': [r for s in sources for r in s.get('source_refs', [])]}
            if any(s['source_component'] in blocked for s in sources):
                spans, refs, status = None, [], 'annotation_source_excluded'
            else:
                spans, refs, status = self._annotation(example['query'], supports, sources)
            entry.update(status=status, annotation_refs=refs)
            if spans:
                example['positive_source_spans'] = spans
            audit['counts'][status] += 1
            audit['examples'].append(entry)
        if audit['counts']['annotated']:
            content['positive_source_span_scope'] = SPAN_SCOPE
        audit['counts'] = dict(audit['counts'])
        receipt['positive_source_spans'] = audit

    def _examples(self, content, dynamic_sources):
        for i, example in enumerate(content.get('retrieved_examples', [])):
            # Match the exact admitted sentence and original bank identity; a
            # merged static/dynamic display must never invent a weaker source.
            sources = [s for s in dynamic_sources
                       if self.consumer.bank[str(s['source_id'])]['query'] == example['query']]
            yield f'/retrieved_examples/{i}', example, list(example['observed_support_types']), sources
        for ci, card in enumerate(content['cards']):
            for ei, example in enumerate(card['examples']):
                supports = list(example.get('supports_types', [example['supports_type']]))
                sources = []
                for t in supports:
                    sources.extend({'source_id': s['source_id'], 'source_component': s['component']}
                                   for s in self.static.get((example['query'], t), []))
                # A dynamic duplicate may have contributed another visible type.
                sources.extend(s for s in dynamic_sources
                               if self.consumer.bank[str(s['source_id'])]['query'] == example['query'])
                yield f'/cards/{ci}/examples/{ei}', example, supports, sources


def strip_source_span_layout(wire):
    """Exact inverse used by prepare and independent archived-wire auditors."""
    import json
    from .slot_stages import canonical
    out = copy.deepcopy(wire)
    part = next(p for p in out['messages'][1]['content'] if p.get('type') == 'text')
    body = json.loads(part['text'])
    scope = body.pop('positive_source_span_scope', None)
    if scope is not None and scope != SPAN_SCOPE:
        raise ValueError('unrecognized source span scope')
    for e in body.get('retrieved_examples', []) + [e for c in body['cards'] for e in c['examples']]:
        spans = e.pop('positive_source_spans', None)
        if spans is not None:
            supports = e.get('observed_support_types', e.get('supports_types', [e.get('supports_type')]))
            if (set(spans) != set(supports) or not spans
                    or any(not isinstance(v, list) or not v or any(not isinstance(s, str) or not s for s in v)
                           for v in spans.values())):
                raise ValueError('source span labels differ from already visible support')
    part['text'] = canonical(body).decode()
    return out
