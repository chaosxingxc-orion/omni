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
    audio_sha256: str | None = None


class StageRails:
    """Four distinct rails with no intent-aware filtering or adjudication."""

    def __init__(self, knowledge, config, *, retrieval_cache=None, provenance_root=None):
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
        self.s1_none_label = config.get('s1_none_label')
        if self.s1_none_label is not None and (not isinstance(self.s1_none_label, str)
                or not self.s1_none_label.strip() or self.s1_none_label in self.cards):
            raise ValueError('S1 control label must be nonempty and outside business ontology')
        self.s1_candidate_presentation = config.get('s1_candidate_presentation', 'legacy')
        if self.s1_candidate_presentation not in ('legacy', 'unified_candidates_v1', 'type_marking_v1'):
            raise ValueError('unknown S1 candidate presentation')
        if self.s1_candidate_presentation == 'unified_candidates_v1' and self.s1_none_label is None:
            raise ValueError('unified S1 candidates require a control label')
        if self.s1_candidate_presentation == 'type_marking_v1' and self.s1_none_label is not None:
            raise ValueError('S1 type marking uses business booleans without a control label')
        for k in ('max_inflight_requests', 's1_max_tokens', 's2_max_tokens'):
            if type(config[k]) is not int or config[k] < 1:
                raise ValueError('positive configuration integer: ' + k)
        self.s1_consumer = None
        if config.get('s1_consumption') is not None:
            from .slot_stage_selector import S1KnowledgeConsumer
            self.s1_consumer = S1KnowledgeConsumer(self.knowledge, config['s1_consumption'], retrieval_cache,
                                                  provenance_root=provenance_root)
        elif retrieval_cache is not None:
            raise ValueError('retrieval cache requires an explicit S1 consumption policy')
        self.version = digest({'knowledge': self.knowledge, 'runtime': self.config})
        if self.s1_consumer is not None:
            self.version = digest({'base_version': self.version, 's1_consumer': self.s1_consumer.version})

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
        excluded = component in c.get('definition_components', [])
        definition = slot_type.replace('_', ' ') if excluded else c['definition']
        independent = c.get('independent_definition') if stage == 'S1' else None
        if excluded and self.s1_consumer is not None and independent is not None:
            if (independent.get('scope') != 'generic_type_meaning' or not independent.get('source_refs')
                    or 'source_components' not in independent or not independent.get('definition')):
                raise ValueError('independent definition requires compiled provenance')
            if not set(independent['source_components']).intersection(self.s1_consumer.blocked | {component}):
                definition = independent['definition']
        view = {'type': slot_type, 'definition': definition,
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
            output_labels = types + ([self.s1_none_label] if self.s1_none_label is not None else [])
            schema = {'type': 'object', 'additionalProperties': False, 'required': ['slot_types'],
                      # Current xgrammar rejects uniqueItems. The after rail
                      # deduplicates types while retaining its repair audit.
                      'properties': {'slot_types': {'type': 'array',
                                      'items': {'type': 'string', 'enum': output_labels}}}}
            if self.s1_candidate_presentation == 'type_marking_v1':
                schema = {'type': 'object', 'additionalProperties': False,
                          'required': ['slot_type_presence'], 'properties': {
                              'slot_type_presence': {'type': 'object', 'additionalProperties': False,
                                  'required': list(types),
                                  'properties': {t: {'type': 'boolean'} for t in types}}}}
        else:
            if branch not in self.cards:
                raise ValueError('unknown selected slot type')
            types = [branch]
            schema = {'type': 'object', 'additionalProperties': False, 'required': ['values'],
                      'properties': {'values': {'type': 'array', 'items': {'type': ['string', 'null']}}}}
        content = {'stage': stage, 'cards': [self._card(t, component, stage) for t in types]}
        if stage == 'S1' and self.s1_candidate_presentation == 'type_marking_v1':
            content['type_marking'] = {
                'types': list(types),
                'instruction': 'For every supplied type, return true if the target audio expresses '
                               'that type, otherwise false. Include every type exactly once. '
                               'All false means no current-Fold type is supported. '
                               'A type may be present even when its value is unknown.'}
        elif stage == 'S1' and self.s1_candidate_presentation == 'unified_candidates_v1':
            content['candidate_selection'] = {
                'candidates': [{'label': t, 'kind': 'business_type', 'reference': 'cards'} for t in types] + [{
                    'label': self.s1_none_label, 'display_name': 'Other slot type', 'kind': 'fold_complement',
                    'definition': 'A fallback for slot types outside the specific candidates in this Fold, '
                                  'selected when none of these specific candidates applies to the target audio. '
                                  'The audio may support types in other Folds or no slot types at all; '
                                  'proof of another Fold type is not required. This local fallback does not '
                                  'assert that any slot exists and does not extend the business slot ontology.'}],
                'instruction': 'Choose from this single candidate list: one or more business candidates '
                               'supported by target-audio evidence, or only the other-slot-type fallback. '
                               'This is multi-label selection, not forced single choice. A supported business '
                               'type remains eligible when its value is unknown, generic, or equals its type name.'}
        elif stage == 'S1' and self.s1_none_label is not None:
            content['empty_fold_control'] = {
                'label': self.s1_none_label,
                'instruction': 'When the target audio supports none of the business types in this Fold, '
                               'return only this control label. It is mutually exclusive with business types. '
                               'It is not a slot type and does not assert other Folds are empty. '
                               'An empty array remains accepted for compatibility.'}
        consumption_receipt = None
        if stage == 'S1' and self.s1_consumer is not None:
            reference, consumption_receipt = self.s1_consumer.before(query, types, component)
            consumption_receipt['definition_sources'] = [{
                'type': t, 'source_excluded': component in self.cards[t].get('definition_components', []),
                'independent_supplied': content['cards'][i]['definition'] == self.cards[t].get('independent_definition', {}).get('definition'),
                'source_refs': self.cards[t].get('independent_definition', {}).get('source_refs', [])
            } for i, t in enumerate(types)]
            if self.s1_consumer.policy.get('coverage_policy') == 'fold_supported_score_order':
                self.s1_consumer.retain_static_examples(content['cards'], reference, consumption_receipt)
            # Legacy score_order/type_round_robin retain their frozen wire.
            elif consumption_receipt['retrieval']['status'] == 'READY':
                for card in content['cards']:
                    card['examples'] = []
            content.update(reference)
            consumption_receipt['retrieval'].setdefault('static_retained_types',
                [card['type'] for card in content['cards'] if card['examples']])
        text = canonical(content).decode()
        parts = [{'type': 'text', 'text': text}]
        parts.insert(0 if self.config['audio_order'] == 'first' else len(parts), copy.deepcopy(query.audio))
        instruction = self.config[stage.lower() + '_instruction']
        if stage == 'S1' and self.s1_candidate_presentation == 'type_marking_v1':
            instruction += ('\nS1 per-type marking protocol: in this single Fold response, return only '
                            '{"slot_type_presence": {type_name: true_or_false, ...}}. '
                            'Use JSON booleans for every supplied business type, with no missing or extra keys. '
                            'All false is a valid empty Fold. Type presence does not require a recoverable value. '
                            'Do not return slot_types, control labels, transcripts, values, or explanations.')
        elif stage == 'S1' and self.s1_candidate_presentation == 'unified_candidates_v1':
            instruction += ('\nS1 unified candidate-selection protocol: the candidate_selection list includes '
                            'the business candidates described by cards and the other-slot-type fallback '
                            + self.s1_none_label + '. In the same selection, return one or more business '
                            'candidates with target-audio evidence, or only this control when no current-Fold '
                            'business candidate applies. Other-Fold evidence is not required. '
                            'Do not combine the fallback with specific type labels. Unknown, generic, or type-name-equal '
                            'values do not disqualify an evidenced business type. '
                            'Return only {"slot_types": [selected candidate labels]}. '
                            'The control is not a business type and is removed before S2 or final adjudication.')
        elif stage == 'S1' and self.s1_none_label is not None:
            instruction += ('\nS1 empty-Fold output protocol: ' + self.s1_none_label +
                            ' is a legal control label in slot_types, not a business slot type. '
                            'If no supplied business type has evidence in the target audio, return only '
                            '{"slot_types": [' + json.dumps(self.s1_none_label) + ']}. '
                            'This explicit empty-Fold protocol replaces any instruction above to return an empty array. '
                            'Do not combine the control label with business types; other Folds may still contain evidence.')
        wire = {'model': self.config['model'], 'max_tokens': self.config[stage.lower() + '_max_tokens'],
                'temperature': self.config['temperature'], 'seed': self.config['seed'],
                'messages': [{'role': 'system', 'content': instruction},
                             {'role': 'user', 'content': parts}],
                'response_format': {'type': 'json_schema', 'json_schema': {'name': stage.lower() + '_slots',
                                                                         'strict': True, 'schema': schema}}}
        key = {'sample_id': query.sample_id, 'stage': stage, 'branch': branch,
               'version': self.version, 'wire_sha256': digest(wire)}
        packet = {**key, 'request_id': stage + '-' + digest(key), 'allowed_types': types, 'wire': wire}
        if stage == 'S1' and self.s1_candidate_presentation == 'type_marking_v1':
            packet['s1_candidate_presentation'] = 'type_marking_v1'
        if consumption_receipt is not None:
            packet['s1_consumption'] = consumption_receipt
        if stage == 'S1' and self.s1_none_label is not None:
            packet['control_label'] = self.s1_none_label
        return packet

    def after_model_call(self, packet, reply):
        """Preserve paid invalid outputs and partial good items; no semantic decisions."""
        base = {k: packet[k] for k in ('sample_id', 'stage', 'branch', 'request_id', 'wire_sha256')}
        if 's1_consumption' in packet:
            base['s1_consumption'] = copy.deepcopy(packet['s1_consumption'])
        control = packet.get('control_label') if packet['stage'] == 'S1' else None
        if control is not None:
            base['s1_control'] = {'label': control, 'selected': False, 'occurrences': 0,
                                  'conflict': False, 'implicit_empty': False}
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
            if packet.get('s1_candidate_presentation') == 'type_marking_v1':
                if not isinstance(obj, dict) or set(obj) != {'slot_type_presence'}:
                    raise ValueError('required slot_type_presence object only')
                marks = obj['slot_type_presence']
                if not isinstance(marks, dict) or set(marks) != set(packet['allowed_types']):
                    raise ValueError('type marking requires every Fold type and no unknown types')
                if any(type(v) is not bool for v in marks.values()):
                    raise ValueError('type marking values must be JSON booleans')
                selected = [t for t in packet['allowed_types'] if marks[t]]
                return {**base, 'status': 'valid_nonempty' if selected else 'valid_empty',
                        'selected_types': selected, 'entities': [], 'slot_type_presence': marks}
            field = 'slot_types' if packet['stage'] == 'S1' else 'values'
            if not isinstance(obj, dict) or field not in obj or not isinstance(obj[field], list):
                raise ValueError('required array missing')
            if set(obj) != {field}:
                base['repairs'].append('ignored_extra_fields:' + ','.join(sorted(set(obj) - {field})))
            selected, entities = [], []
            for i, value in enumerate(obj[field]):
                if packet['stage'] == 'S1':
                    if control is not None and value == control:
                        base['s1_control']['occurrences'] += 1
                        base['s1_control']['selected'] = True
                        if base['s1_control']['occurrences'] > 1:
                            base['repairs'].append('deduplicated_control_label:' + control)
                    elif not isinstance(value, str) or value not in packet['allowed_types']:
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
            if control is not None:
                base['s1_control']['implicit_empty'] = not obj[field]
                if base['s1_control']['selected'] and selected:
                    base['s1_control']['conflict'] = True
                    base['repairs'].append('control_conflict_preserved_business_types:' + control)
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
