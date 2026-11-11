"""S1 reference/identity/control behavior; fixtures are not model evaluations."""
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest

from test_slot_stages import CONFIG, knowledge, query
from speech_aware_evidence_acquisition.rails.slot_stage_selector import digest, S1KnowledgeConsumer
from speech_aware_evidence_acquisition.rails.slot_stages import Query, StageRails, SlotStages, canonical


FIXTURES = Path(__file__).resolve().parent / 'fixtures'
SOURCE = FIXTURES / 's1-consumption-sources.json'


def source_ref(pointer):
    return {'path': SOURCE.name, 'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'pointer': pointer}


def fixture():
    k = knowledge()
    k['source_components'].update({'2': 'other', '3': 'third'})
    k['s1_discriminative_rules'] = [{'rule_id': 'r1', 'target_types': ['person'],
        'related_types': ['artist_name'], 'description': 'A sourced distinction, not exclusivity.',
        'source_refs': [source_ref('/distinction')], 'source_components': ['third'],
        'scope': {'stage':'S1','intent_independent':True,'multi_candidate':True,
                  'final_adjudication':False,'condition':'current_audio_evidence'}}]
    bank = []
    for sid, component, text, types in [(1, 'self', 'must not show', ['person']),
                                      (2, 'other', 'whole TRAIN sentence', ['person', 'artist_name']),
                                      (3, 'third', 'other complete sentence', [])]:
        bank.append({'source_id': sid, 'source_component': component, 'query': text,
                     'observed_types': types, 'source_refs': [source_ref('/train/'+str(sid-1))],
                     'sentence_sha256': hashlib.sha256(text.encode()).hexdigest()})
    cache = {'schema': 's1-audio-sentence-retrieval-cache-v1', 'knowledge_id': k['knowledge_id'],
             'encoder': {'identity_sha256': 'encoder-fixture', 'query_modality': 'audio', 'key_unit': 'full_train_sentence'},
             'input_pins': {'fixture': 'fixture'}, 'bank': bank,
             'queries': [{'sample_id': query().sample_id, 'source_component': 'self', 'audio_sha256': 'audio-file',
                          'audio_object_sha256': digest(query().audio), 'excluded_source_components': ['self'],
                          'neighbors': [{'source_id': 1, 'score': .9}, {'source_id': 2, 'score': -.1}, {'source_id': 3, 'score': -.2}]}]}
    cache['cache_id'] = digest(cache)
    c = copy.deepcopy(CONFIG)
    c['s1_consumption'] = {'schema': 's1-consumption-policy-v1', 'policy_id': 'fixture-v1',
                           'cache_id': cache['cache_id'], 'encoder_identity_sha256': 'encoder-fixture',
                           'max_examples': 2, 'max_example_bytes': 2000, 'max_rules': 4, 'max_rule_bytes': 2000,
                           'coverage_policy': 'score_order'}
    return k, c, cache


def content(packet):
    return json.loads(next(x['text'] for x in packet['wire']['messages'][1]['content'] if x['type'] == 'text'))


class ConsumptionTest(unittest.TestCase):
    def test_legacy_wire_and_request_identity_exactly_preserved(self):
        k, c, cache = fixture()
        a = StageRails(knowledge(), CONFIG)
        expected = json.loads((FIXTURES / 's1-parent-wire.json').read_text())
        self.assertEqual(expected['parent_source_sha256'], '39434a895891d250171c234cc797bc41512fcc1eaec999aac4e212eb982773e0')
        self.assertEqual(a.version, expected['version'])
        for stage, branch in [('S1','f1'),('S1','f2'),('S2','person')]:
            self.assertEqual(a.before_model_call(stage, query(), branch), expected['packets'][stage+':'+branch])
        self.assertNotIn('s1_consumption', a.before_model_call('S1', query(), 'f1'))

    def test_whole_sentence_signed_order_projected_to_fold_no_intent(self):
        k, c, cache = fixture()
        r = StageRails(k, c, retrieval_cache=cache, provenance_root=FIXTURES)
        p = r.before_model_call('S1', query(), 'f1')
        v = content(p)
        self.assertEqual(v['retrieved_examples'][0]['query'], 'whole TRAIN sentence')
        self.assertEqual(v['retrieved_examples'][0]['observed_support_types'], ['person'])
        self.assertEqual(v['retrieved_examples'][1]['observed_support_types'], [])
        self.assertNotIn('must not show', json.dumps(p))
        self.assertNotIn('intent', v)
        self.assertEqual(p['allowed_types'], ['person', 'time'])
        self.assertEqual(v['discriminative_rules'][0]['related_types'], ['artist_name'])
        self.assertTrue(all(not x['examples'] for x in v['cards']))
        self.assertEqual(p['s1_consumption']['retrieval']['provenance'][0]['score'], -.1)

    def test_missing_cache_or_query_degrades_preserving_cards(self):
        k, c, cache = fixture()
        r = StageRails(k, c, provenance_root=FIXTURES)
        p = r.before_model_call('S1', query(), 'f1')
        self.assertEqual(p['s1_consumption']['retrieval']['status'], 'DEGRADED_NO_CACHE')
        self.assertTrue(content(p)['cards'][0]['examples'])
        r = StageRails(k, c, retrieval_cache=cache, provenance_root=FIXTURES)
        p = r.before_model_call('S1', Query('external', query().audio, 'elsewhere'), 'f1')
        self.assertEqual(p['s1_consumption']['retrieval']['status'], 'DEGRADED_QUERY_NOT_CACHED')
        self.assertTrue(content(p)['cards'][0]['examples'])

    def test_query_audio_mismatch_and_encoder_digest_mismatch_fail(self):
        k, c, cache = fixture()
        r = StageRails(k, c, retrieval_cache=cache, provenance_root=FIXTURES)
        with self.assertRaisesRegex(ValueError, 'audio/sample/source'):
            r.before_model_call('S1', Query(query().sample_id, {'type':'input_audio','input_audio':{'data':'changed'}}, 'self'), 'f1')
        with self.assertRaisesRegex(ValueError, 'audio/sample/source'):
            r.before_model_call('S1', Query(query().sample_id, query().audio, 'self', 'bad-file-hash'), 'f1')
        cache['encoder']['identity_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            StageRails(k, c, retrieval_cache=cache, provenance_root=FIXTURES)

    def test_query_cache_gold_or_transcript_fields_rejected(self):
        for field in ['gold_types', 'transcript', 'intent']:
            k, c, cache = fixture()
            cache['queries'][0][field] = 'forbidden'
            cache.pop('cache_id'); cache['cache_id'] = digest(cache); c['s1_consumption']['cache_id'] = cache['cache_id']
            with self.assertRaisesRegex(ValueError, 'identity/audio/ranking'):
                StageRails(k, c, retrieval_cache=cache, provenance_root=FIXTURES)

    def test_tampered_bank_sentence_or_types_rehashed_still_rejected(self):
        for mode in ['sentence','types']:
            k,c,cache=fixture()
            row=cache['bank'][1]
            if mode=='sentence':
                row['query']='invented example'
                row['sentence_sha256']=hashlib.sha256(row['query'].encode()).hexdigest()
            else: row['observed_types']=['time']
            cache.pop('cache_id');cache['cache_id']=digest(cache)
            c['s1_consumption']['cache_id']=cache['cache_id']
            with self.assertRaisesRegex(ValueError,'original TRAIN row'):
                StageRails(k,c,retrieval_cache=cache,provenance_root=FIXTURES)

    def test_source_rule_exclusion_budget_and_dedup(self):
        k, c, cache = fixture()
        k['s1_discriminative_rules'][0]['source_components'] = ['self']
        c['s1_consumption']['max_example_bytes'] = 1
        r = StageRails(k, c, retrieval_cache=cache, provenance_root=FIXTURES)
        p = r.before_model_call('S1', query(), 'f1')
        self.assertFalse(content(p)['discriminative_rules'])
        self.assertFalse(content(p)['retrieved_examples'])
        self.assertEqual(p['s1_consumption']['rules']['source_excluded'], 1)
        self.assertEqual(p['s1_consumption']['retrieval']['status'], 'DEGRADED_NO_ELIGIBLE_EXAMPLES')
        self.assertTrue(content(p)['cards'][0]['examples'])

    def test_after_does_not_filter_business_candidates_by_retrieval(self):
        k, c, cache = fixture()
        r = StageRails(k, c, retrieval_cache=cache, provenance_root=FIXTURES)
        p = r.before_model_call('S1', query(), 'f1')
        result = r.after_model_call(p, {'text':'{"slot_types":["time","person"]}', 'finish_reason':'stop'})
        self.assertEqual(result['selected_types'], ['time', 'person'])
        self.assertEqual(result['s1_consumption'], p['s1_consumption'])
        self.assertNotIn('s1_consumption', r.before_model_call('S2', query(), 'person'))

    def test_provenance_bytes_pointer_and_scope_rejected(self):
        for mode in ['sha', 'pointer', 'scope', 'root']:
            k, c, cache = fixture()
            rule = k['s1_discriminative_rules'][0]
            if mode == 'sha': rule['source_refs'][0]['sha256'] = '0'*64
            if mode == 'pointer': rule['source_refs'][0]['pointer'] = '/definition'
            if mode == 'scope': rule['scope']['intent_independent'] = False
            with self.assertRaises(ValueError):
                StageRails(k, c, retrieval_cache=cache, provenance_root=None if mode=='root' else FIXTURES)

    def test_independent_definition_only_s1_and_source_exclusion(self):
        k, c, cache = fixture()
        k['cards']['person']['definition_components'] = ['self']
        k['cards']['person']['independent_definition'] = {
            'definition':'Independent person meaning.', 'source_refs':[source_ref('/definition')],
            'source_components':[], 'scope':'generic_type_meaning'}
        k['knowledge_supply'] = {'procedural_rules':[{
            'description':'A type can be present even if its value is unknown.',
            'source_refs':[source_ref('/procedure')], 'source_components':[]}]}
        r = StageRails(k, c, retrieval_cache=cache, provenance_root=FIXTURES)
        p = r.before_model_call('S1', query(), 'f1')
        self.assertEqual(content(p)['cards'][0]['definition'], 'Independent person meaning.')
        self.assertTrue(content(p)['procedural_rules'])
        self.assertEqual(content(r.before_model_call('S2',query(),'person'))['cards'][0]['definition'], 'person')
        k['cards']['person']['independent_definition']['source_components'] = ['self']
        r = StageRails(k,c,retrieval_cache=cache,provenance_root=FIXTURES)
        self.assertEqual(content(r.before_model_call('S1',query(),'f1'))['cards'][0]['definition'],'person')

    def test_audio_digest_once_for_ten_folds_and_mutation_rejected(self):
        from unittest.mock import patch
        from speech_aware_evidence_acquisition.rails import slot_stage_selector as selector
        k,c,cache=fixture()
        r=StageRails(k,c,retrieval_cache=cache,provenance_root=FIXTURES)
        q=query()
        with patch.object(selector,'digest',wraps=selector.digest) as audit:
            for _ in range(10): r.before_model_call('S1',q,'f1')
            self.assertEqual(audit.call_count,1)
            q.audio['input_audio']['data']='changed'
            with self.assertRaisesRegex(ValueError,'audio/sample/source'):
                r.before_model_call('S1',q,'f1')


class FoldSupportedSupplyTest(unittest.TestCase):
    def make(self, transform=None, **policy):
        k, c, cache = fixture()
        c['s1_consumption'].update(coverage_policy='fold_supported_score_order', **policy)
        if transform:
            transform(k, c, cache)
            cache.pop('cache_id')
            cache['cache_id'] = digest(cache)
            c['s1_consumption']['cache_id'] = cache['cache_id']
        return StageRails(k, c, retrieval_cache=cache, provenance_root=FIXTURES)

    def test_skip_top_empty_projection_then_take_later_signed_support(self):
        def ranking(k, c, cache):
            cache['queries'][0]['neighbors'] = [
                {'source_id':3,'score':.8}, {'source_id':1,'score':.7}, {'source_id':2,'score':-.1}]
        packet = self.make(ranking).before_model_call('S1', query(), 'f1')
        audit = packet['s1_consumption']['retrieval']
        self.assertEqual(audit['candidate_pool_count'], 2)
        self.assertEqual(audit['fold_supported_count'], 1)
        self.assertEqual(audit['source_excluded'], 1)
        self.assertEqual(audit['provenance'][0]['score'], -.1)
        self.assertEqual(content(packet)['retrieved_examples'][0]['observed_support_types'], ['person'])
        self.assertEqual(audit['covered_types'], ['person'])

    def test_no_fold_support_degrades_and_retains_all_static_labels(self):
        def ranking(k, c, cache):
            cache['queries'][0]['neighbors'] = [{'source_id':3,'score':-.2}]
        packet = self.make(ranking).before_model_call('S1', query(), 'f1')
        audit = packet['s1_consumption']['retrieval']
        self.assertEqual(audit['status'], 'DEGRADED_NO_FOLD_SUPPORT')
        self.assertEqual(audit['static_retained_types'], ['person','time'])
        self.assertFalse(content(packet)['retrieved_examples'])
        self.assertEqual(content(packet)['cards'][0]['examples'][0]['supports_types'], ['person','time'])
        self.assertNotIn('self evidence', json.dumps(packet))

    def test_partial_coverage_preserves_uncovered_static_and_full_schema(self):
        packet = self.make().before_model_call('S1', query(), 'f1')
        cards = content(packet)['cards']
        self.assertFalse(cards[0]['examples'])
        self.assertEqual(cards[1]['examples'], [{'query':'fixture example','supports_type':'time'}])
        self.assertEqual(packet['s1_consumption']['retrieval']['static_retained_types'], ['time'])
        self.assertEqual(packet['allowed_types'], ['person','time'])

    def test_budget_failure_retains_every_type_and_count_and_bytes_bounded(self):
        for limits in [dict(max_examples=0), dict(max_example_bytes=1), dict(max_examples=1)]:
            packet = self.make(**limits).before_model_call('S1', query(), 'f1')
            audit = packet['s1_consumption']['retrieval']
            self.assertLessEqual(audit['total_example_count'], audit['total_example_limit'])
            self.assertLessEqual(audit['total_example_bytes'], audit['total_example_byte_limit'])
            self.assertLessEqual(audit['selected'], limits.get('max_examples',2))
            if audit['selected'] == 0:
                self.assertEqual(audit['status'], 'DEGRADED_NO_ELIGIBLE_EXAMPLES')
                self.assertEqual(audit['static_retained_types'], ['person','time'])

    def test_static_dynamic_same_sentence_printed_once_keeps_both_supports(self):
        def same_sentence(k, c, cache):
            k['cards']['time']['examples'][0]['query'] = 'whole TRAIN sentence'
        packet = self.make(same_sentence).before_model_call('S1', query(), 'f1')
        view = content(packet)
        self.assertFalse(view['retrieved_examples'])
        self.assertEqual(view['cards'][1]['examples'][0]['supports_types'], ['person','time'])
        self.assertEqual(packet['s1_consumption']['retrieval']['covered_types'], ['person'])
        self.assertEqual(packet['s1_consumption']['retrieval']['dynamic_displayed_count'], 0)

    def test_excluded_support_cannot_replace_static(self):
        def exclusion(k, c, cache):
            cache['queries'][0]['excluded_source_components'].append('other')
        packet = self.make(exclusion).before_model_call('S1', query(), 'f1')
        self.assertEqual(packet['s1_consumption']['retrieval']['status'], 'DEGRADED_NO_FOLD_SUPPORT')
        self.assertEqual(packet['s1_consumption']['retrieval']['static_retained_types'], ['person','time'])

    def test_old_score_order_wire_matches_independent_frozen_packet(self):
        k,c,cache = fixture()
        packet = StageRails(k,c,retrieval_cache=cache,provenance_root=FIXTURES).before_model_call('S1',query(),'f1')
        frozen = json.loads((FIXTURES/'s1-score-order-wire.json').read_text())
        for key in ['wire','wire_sha256','request_id','version','allowed_types']:
            self.assertEqual(packet[key], frozen['packet'][key])

    def test_new_policy_preserves_s2_wire_and_none_control(self):
        oldk,oldc,oldcache = fixture()
        old = StageRails(oldk,oldc,retrieval_cache=oldcache,provenance_root=FIXTURES)
        new = self.make()
        self.assertEqual(old.before_model_call('S2',query(),'time')['wire'],new.before_model_call('S2',query(),'time')['wire'])
        new.s1_none_label = 'NONE_OF_FOLD'
        packet = new.before_model_call('S1',query(),'f1')
        self.assertEqual(packet['wire']['response_format']['json_schema']['schema']['properties']['slot_types']['items']['enum'],
                         ['person','time','NONE_OF_FOLD'])
        self.assertEqual(new.after_model_call(packet,{'text':'{"slot_types":["NONE_OF_FOLD"]}', 'finish_reason':'stop'})['status'],'valid_empty')


class NoneControlTest(unittest.TestCase):
    def setUp(self):
        c = copy.deepcopy(CONFIG); c['s1_none_label'] = 'NONE_OF_FOLD'
        self.r = StageRails(knowledge(), c)
        self.p = self.r.before_model_call('S1', query(), 'f1')

    def after(self, labels, reason='stop'):
        return self.r.after_model_call(self.p, {'text':json.dumps({'slot_types':labels}), 'finish_reason':reason})

    def test_control_only_repeated_and_implicit_empty(self):
        for labels in [['NONE_OF_FOLD'], ['NONE_OF_FOLD', 'NONE_OF_FOLD'], []]:
            result = self.after(labels)
            self.assertEqual(result['status'], 'valid_empty')
            self.assertEqual(result['selected_types'], [])
            self.assertEqual(result['s1_control']['occurrences'], len(labels))
        self.assertTrue(self.after([])['s1_control']['implicit_empty'])

    def test_control_mixed_preserves_types_and_records_conflict(self):
        result = self.after(['NONE_OF_FOLD','person','person'])
        self.assertEqual(result['status'], 'valid_nonempty')
        self.assertEqual(result['selected_types'], ['person'])
        self.assertTrue(result['s1_control']['conflict'])
        self.assertTrue(any(x.startswith('control_conflict') for x in result['repairs']))

    def test_unknown_partial_length_not_empty_or_control(self):
        self.assertEqual(self.after(['NONE_OF_FOLD','unknown'])['status'], 'partial')
        result = self.after(['NONE_OF_FOLD'], 'length')
        self.assertEqual(result['status'], 'incomplete')
        self.assertFalse(result['s1_control']['selected'])

    def test_control_outside_ontology_s2_and_backend_schema(self):
        self.assertEqual(self.p['allowed_types'], ['person','time'])
        schema = self.p['wire']['response_format']['json_schema']['schema']
        self.assertEqual(schema['properties']['slot_types']['items']['enum'], ['person','time','NONE_OF_FOLD'])
        self.assertNotIn('uniqueItems', json.dumps(schema))
        self.assertNotIn('oneOf', json.dumps(schema))
        self.assertIn('replaces any instruction above', self.p['wire']['messages'][0]['content'])
        self.assertIn('NONE_OF_FOLD', self.p['wire']['messages'][0]['content'])
        self.assertNotIn('control_label', self.r.before_model_call('S2', query(), 'person'))
        c = copy.deepcopy(CONFIG); c['s1_none_label'] = 'person'
        with self.assertRaises(ValueError):
            StageRails(knowledge(), c)


@unittest.skipUnless(os.environ.get('S_STAGES_NATIVE_TEST') == '1', 'requires fixed openJiuwen image')
class NativeControlTest(unittest.IsolatedAsyncioTestCase):
    async def test_ten_fold_native_before_after_preserves_control_and_receipt(self):
        from speech_aware_evidence_acquisition.rails.slot_stages_native import NativeStageCaller
        from speech_aware_evidence_acquisition.rails.core_client import CoreCompletion
        k = knowledge()
        # Ten distinct computational folds, each with one independent business type.
        template = copy.deepcopy(k['cards']['person'])
        k['cards'] = {f'type_{i}': {**copy.deepcopy(template), 'type':f'type_{i}'} for i in range(10)}
        k['folds'] = [{'fold_id':f'fold-{i:02}', 'types':[f'type_{i}']} for i in range(10)]
        c = copy.deepcopy(CONFIG); c['s1_none_label'] = 'NONE_OF_FOLD'
        c['s1_consumption'] = {'schema':'s1-consumption-policy-v1','policy_id':'native-fixture',
            'max_examples':2,'max_example_bytes':1000,'max_rules':2,'max_rule_bytes':1000}
        observed = []
        async def transport(packet, contact):
            observed.append(packet)
            labels = ['NONE_OF_FOLD'] + (packet['allowed_types'] if packet['branch']=='fold-00' else [])
            text = json.dumps({'slot_types':labels})
            raw = canonical({'choices':[{'message':{'role':'assistant','content':text},'finish_reason':'stop'}],
                             'usage':{'prompt_tokens':5,'completion_tokens':3}})
            return CoreCompletion(text=text, request_id=packet['request_id'], sample_id=packet['sample_id'],
                agent_contact_id=contact, request_sha256=packet['wire_sha256'], response_sha256=hashlib.sha256(raw).hexdigest(),
                response_text_sha256=hashlib.sha256(text.encode()).hexdigest(), tool_calls_json=b'null',
                wire_response_bytes=raw, tool_names=(), tool_schemas_sha256=hashlib.sha256(b'[]').hexdigest())
        result = await SlotStages(StageRails(k,c), NativeStageCaller(transport)).run_s1(query())
        self.assertEqual(len(observed),10)
        self.assertEqual(len(result['s1']),10)
        self.assertEqual(result['selected_types'],['type_0'])
        self.assertEqual(sum(r['s1_control']['conflict'] for r in result['s1']),1)
        self.assertEqual(sum(r['s1_control']['selected'] for r in result['s1']),10)
        self.assertTrue(all(r['s1_consumption']['retrieval']['status']=='DEGRADED_NO_CACHE' for r in result['s1']))


if __name__ == '__main__':
    unittest.main()
