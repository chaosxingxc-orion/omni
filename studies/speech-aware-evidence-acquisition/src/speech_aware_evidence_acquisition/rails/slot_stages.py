"""Intent-independent S1 fan-out -> type union -> S2 per-type fan-out.

The caller owns transport, durable contact journal, lease and admission. This
module creates neither a provider nor an experiment queue. One shared semaphore
bounds actual model contacts across all samples handled by this instance.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from dataclasses import dataclass

from .slot_stage_knowledge import normalized


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def document(text):
    def unique(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate key: ' + k)
            out[k] = v
        return out
    return json.loads(text, object_pairs_hook=unique)


@dataclass(frozen=True)
class Query:
    sample_id: str
    audio: dict
    # No intent, query transcript, expected types or reference values allowed.
    source_component: str | None = None


class StageRails:
    """Four distinct rails with no intent-aware filtering or adjudication."""

    def __init__(self, knowledge, config):
        self.knowledge = copy.deepcopy(knowledge)
        self.config = copy.deepcopy(config)
        self.cards = self.knowledge['cards']
        assigned = [t for f in self.knowledge['folds'] for t in f['types']]
        if len(assigned) != len(set(assigned)) or set(assigned) != set(self.cards):
            raise ValueError('folds must cover each global slot type exactly once')
        if len({f['fold_id'] for f in self.knowledge['folds']}) != len(self.knowledge['folds']):
            raise ValueError('duplicate fold identity')
        for f in self.knowledge['folds']:
            if not f['types']:
                raise ValueError('empty fold')
        if config['audio_order'] not in ('first', 'last'):
            raise ValueError('audio_order must be first or last')
        for k in ('max_inflight_requests', 's1_max_tokens', 's2_max_tokens'):
            if type(config[k]) is not int or config[k] < 1:
                raise ValueError('positive configuration integer: ' + k)
        self.version = digest({'knowledge': self.knowledge, 'runtime': self.config})

    def _component(self, query):
        parts = query.sample_id.split(':')
        component = query.source_component
        if len(parts) >= 3 and parts[1] == 'train':
            actual = self.knowledge['source_components'].get(str(int(parts[2])))
            if actual is None or (component is not None and component != actual):
                raise ValueError('TRAIN query needs its actual source component for self-exclusion')
            component = actual
        return component

    def _examples(self, card, component, stage):
        limit = (self.config['s1_examples_per_type'] if stage == 'S1'
                 else self.config['s2_examples_by_kind'][card['kind']])
        examples, seen_values, seen_queries = [], set(), set()
        pool = [x for x in card['examples'] if x['component'] != component]
        for distinct_values in (True, False):
            for x in pool:
                q = normalized(x['query'])
                vals = x.get('values', [x.get('value')])
                v = tuple(normalized(y) for y in vals)
                if q in seen_queries or (distinct_values and v in seen_values):
                    continue
                if len(examples) >= limit:
                    return examples
                seen_queries.add(q)
                seen_values.add(v)
                # S1 examples demonstrate presence only, not incomplete negative labels.
                examples.append({'query': x['query'], 'supports_type': card['type']} if stage == 'S1'
                                else {'query': x['query'], 'values': vals})
        return examples

    def _card(self, slot_type, component, stage):
        c = self.cards[slot_type]
        view = {'type': slot_type, 'definition': (slot_type.replace('_', ' ') if component in c.get('definition_components', []) else c['definition']),
                'examples': self._examples(c, component, stage)}
        if stage == 'S2':
            view.update(kind=c['kind'], closed_world=c['closed_world'], value_rules=c.get('value_rules', []),
                        missing_value_policy=c['missing_value_policy'],
                        value_output=self.config['value_output'])
            if self.config['include_values_for_s2'] and c['kind'] == 'enumerated_observed':
                vocabulary = []
                for value in c['vocabulary']:
                    # Recompute variants from eligible sources; never leave a held-out
                    # surface in the prompt after merely removing its provenance.
                    surfaces = sorted({s['surface'] for s in value['sources'] if s['component'] != component})
                    if surfaces:
                        vocabulary.append(surfaces)
                view['observed_vocabulary'] = vocabulary
                view['vocabulary_scope'] = 'observed TRAIN variants, not exhaustive and not audio evidence'
        return view

    def before_model_call(self, stage, query, branch):
        if stage not in ('S1', 'S2'):
            raise ValueError('unknown S stage')
        if not isinstance(query.audio, dict) or query.audio.get('type') != 'input_audio':
            raise ValueError('original audio content part required')
        component = self._component(query)
        if stage == 'S1':
            fold = next(f for f in self.knowledge['folds'] if f['fold_id'] == branch)
            types = fold['types']
            schema = {'type': 'object', 'additionalProperties': False, 'required': ['slot_types'],
                      # Current xgrammar rejects uniqueItems. The after rail
                      # deduplicates types while retaining its repair audit.
                      'properties': {'slot_types': {'type': 'array',
                                      'items': {'type': 'string', 'enum': types}}}}
        else:
            if branch not in self.cards:
                raise ValueError('unknown selected slot type')
            types = [branch]
            schema = {'type': 'object', 'additionalProperties': False, 'required': ['values'],
                      'properties': {'values': {'type': 'array', 'items': {'type': ['string', 'null']}}}}
        content = {'stage': stage, 'cards': [self._card(t, component, stage) for t in types]}
        text = canonical(content).decode()
        parts = [{'type': 'text', 'text': text}]
        parts.insert(0 if self.config['audio_order'] == 'first' else len(parts), copy.deepcopy(query.audio))
        wire = {'model': self.config['model'], 'max_tokens': self.config[stage.lower() + '_max_tokens'],
                'temperature': self.config['temperature'], 'seed': self.config['seed'],
                'messages': [{'role': 'system', 'content': self.config[stage.lower() + '_instruction']},
                             {'role': 'user', 'content': parts}],
                'response_format': {'type': 'json_schema', 'json_schema': {'name': stage.lower() + '_slots',
                                                                         'strict': True, 'schema': schema}}}
        key = {'sample_id': query.sample_id, 'stage': stage, 'branch': branch,
               'version': self.version, 'wire_sha256': digest(wire)}
        return {**key, 'request_id': stage + '-' + digest(key), 'allowed_types': types, 'wire': wire}

    def after_model_call(self, packet, reply):
        """Preserve paid invalid outputs and partial good items; no semantic decisions."""
        base = {k: packet[k] for k in ('sample_id', 'stage', 'branch', 'request_id', 'wire_sha256')}
        base.update(raw_text=reply.get('text'), usage=reply.get('usage'), receipt=reply.get('receipt'),
                    finish_reason=reply.get('finish_reason'), repairs=[], rejected_items=[])
        if reply.get('finish_reason') != 'stop':
            return {**base, 'status': 'incomplete', 'selected_types': [], 'entities': []}
        text = reply.get('text')
        if isinstance(text, str) and text.strip().startswith('```') and text.strip().endswith('```'):
            lines = text.strip().splitlines()
            text = '\n'.join(lines[1:-1])
            base['repairs'].append('removed_outer_code_fence')
        try:
            obj = document(text)
            field = 'slot_types' if packet['stage'] == 'S1' else 'values'
            if not isinstance(obj, dict) or field not in obj or not isinstance(obj[field], list):
                raise ValueError('required array missing')
            if set(obj) != {field}:
                base['repairs'].append('ignored_extra_fields:' + ','.join(sorted(set(obj) - {field})))
            selected, entities = [], []
            for i, value in enumerate(obj[field]):
                if packet['stage'] == 'S1':
                    if not isinstance(value, str) or value not in packet['allowed_types']:
                        base['rejected_items'].append({'index': i, 'value': value, 'reason': 'outside_fold'})
                    elif value in selected:
                        base['repairs'].append('deduplicated_type:' + value)
                    else:
                        selected.append(value)
                elif value is None or (isinstance(value, str) and value.strip()):
                    # Preserve duplicates: they may represent two actual mentions.
                    entities.append({'type': packet['branch'], 'filler': value,
                                     'local_mention_index': i, 'source_request_id': packet['request_id']})
                else:
                    base['rejected_items'].append({'index': i, 'value': value, 'reason': 'invalid_value'})
            status = 'partial' if base['rejected_items'] else 'valid_nonempty' if selected or entities else 'valid_empty'
            return {**base, 'status': status, 'selected_types': selected, 'entities': entities}
        except (ValueError, TypeError) as error:
            return {**base, 'status': 'protocol_error', 'error': str(error), 'selected_types': [], 'entities': []}


class SlotStages:
    """Transport callback executes a bound packet and returns raw reply metadata.

    Use NativeStageCaller for real openJiuwen before/after hook invocation.
    Callback must persist raw responses before return and never retry unknowns.
    The scheduler has no retry loop. Failure is retained, never called empty.
    """

    def __init__(self, rails, caller):
        self.rails, self.caller = rails, caller
        self.limit = asyncio.Semaphore(rails.config['max_inflight_requests'])

    async def _call(self, packet):
        started = time.monotonic()
        async with self.limit:
            admitted = time.monotonic()
            result = await self.caller(packet, self.rails.after_model_call)
            result['timing'] = {'queue_seconds': admitted - started,
                                'call_and_rails_seconds': time.monotonic() - admitted}
            return result

    async def run(self, query):
        # Materialize every S1 request before acquiring a model-contact slot.
        prepared = [self.rails.before_model_call('S1', query, f['fold_id']) for f in self.rails.knowledge['folds']]
        s1 = await self._drain(prepared)
        selected = sorted({t for r in s1 for t in r['selected_types']})
        prepared_s2 = [self.rails.before_model_call('S2', query, t) for t in selected]
        s2 = await self._drain(prepared_s2)
        complete = all(r['status'] in ('valid_empty', 'valid_nonempty') for r in s1 + s2)
        return {'schema': 's1-s2-handoff-v1', 'sample_id': query.sample_id,
                'knowledge_id': self.rails.knowledge['knowledge_id'], 'complete': complete,
                's1': s1, 'selected_types': selected, 's2': s2,
                'entities': [e for r in s2 for e in r['entities']],
                'unfilled_selected_types': [r['branch'] for r in s2 if not r['entities']],
                'd_policy': 'Resolve cross-type alternatives and intent-slot assembly. Unknown/failed branches are not absence.'}

    async def run_s1(self, query):
        """Type-only experiment using the same rails, caller and instance limit."""
        prepared = [self.rails.before_model_call('S1', query, f['fold_id'])
                    for f in self.rails.knowledge['folds']]
        return await self.run_prepared_s1(query.sample_id, prepared)

    async def run_prepared_s1(self, sample_id, prepared):
        """Consume CPU-prepared packets without rebuilding prompts under a lease."""
        expected = [f['fold_id'] for f in self.rails.knowledge['folds']]
        if ([p['branch'] for p in prepared] != expected
                or any(p['stage'] != 'S1' or p['sample_id'] != sample_id
                       or p['version'] != self.rails.version
                       or digest(p['wire']) != p['wire_sha256'] for p in prepared)):
            raise ValueError('prepared S1 identity/fold/wire mismatch')
        s1 = await self._drain(prepared)
        return {'schema': 's1-handoff-v1', 'sample_id': sample_id,
                'knowledge_id': self.rails.knowledge['knowledge_id'],
                'complete': all(r['status'] in ('valid_empty', 'valid_nonempty') for r in s1),
                's1': s1, 'selected_types': sorted({t for r in s1 for t in r['selected_types']})}

    async def _drain(self, packets):
        results = await asyncio.gather(*(self._call(p) for p in packets), return_exceptions=True)
        # Programming/transport exceptions stop this sample after peers settle.
        # They are not converted into a fabricated valid-empty model answer.
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise errors[0]
        return results
