"""S1/S2 behavior contracts; fixtures verify plumbing, never model accuracy."""
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest

from speech_aware_evidence_acquisition.rails.slot_stage_knowledge import build, split_folds
from speech_aware_evidence_acquisition.rails.slot_stages import Query, StageRails, SlotStages, canonical, digest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / 'configs/slots/s-stages-v1/runtime.json').read_text())


def knowledge():
    cards = {}
    for t in ['person', 'time', 'artist_name']:
        cards[t] = {'type': t, 'definition': 'Meaning of ' + t, 'definition_components': [],
                    'kind': 'enumerated_observed' if t == 'person' else 'open',
                    'closed_world': False, 'missing_value_policy': 'null if evidenced but unknown',
                    'examples': [{'query': 'fixture example', 'value': 'example', 'source_id': 2, 'component': 'other'},
                                 {'query': 'self evidence forbidden', 'value': 'secret', 'source_id': 1, 'component': 'self'}],
                    'vocabulary': [{'normalized': 'secret', 'sources': [{'surface': 'secret', 'component': 'self'}]},
                                   {'normalized': 'value', 'sources': [{'surface': 'value', 'component': 'other'}]}]}
    return {'knowledge_id': 'fixture', 'cards': cards, 'source_components': {'1': 'self'},
            'folds': [{'fold_id': 'f1', 'types': ['person', 'time']}, {'fold_id': 'f2', 'types': ['artist_name']}]}


def query():
    return Query('slurp:train:000001:audio-x', {'type': 'input_audio', 'input_audio': {'data': 'fixture', 'format': 'wav'}})


class RailsTest(unittest.TestCase):
    def setUp(self):
        self.rails = StageRails(knowledge(), CONFIG)

    def test_complete_disjoint_coloring(self):
        types = list('abcdef')
        edges = {('a', 'b'), ('a', 'c'), ('b', 'c'), ('d', 'e')}
        folds = split_folds(types, edges, 2)
        self.assertEqual(sorted(t for f in folds for t in f), types)
        self.assertTrue(all(len(f) <= 2 for f in folds))
        self.assertFalse(any(a in f and b in f for a, b in edges for f in folds))
        self.assertEqual(folds, split_folds(types, edges, 2))

    def test_s1_no_values_or_intent_or_self_example(self):
        p = self.rails.before_model_call('S1', query(), 'f1')
        wire = json.dumps(p['wire'])
        self.assertNotIn('secret', wire)
        self.assertNotIn('self evidence', wire)
        self.assertNotIn('observed_vocabulary', wire)
        self.assertNotIn('candidate_id', wire)
        self.assertEqual(p['allowed_types'], ['person', 'time'])
        self.assertEqual(p['wire']['messages'][1]['content'][-1]['type'], 'input_audio')

    def test_s2_only_selected_card_and_leave_self_out(self):
        p = self.rails.before_model_call('S2', query(), 'person')
        content = json.loads(p['wire']['messages'][1]['content'][0]['text'])
        self.assertEqual([c['type'] for c in content['cards']], ['person'])
        self.assertEqual(content['cards'][0]['observed_vocabulary'], [['value']])
        self.assertNotIn('secret', json.dumps(p))

    def test_fail_unknown_train_component(self):
        with self.assertRaises(ValueError):
            self.rails.before_model_call('S1', Query('slurp:train:999999:x', query().audio), 'f1')

    def test_type_dedupe_retains_partial(self):
        p = self.rails.before_model_call('S1', query(), 'f1')
        r = self.rails.after_model_call(p, {'text': '{"slot_types":["person","person","alien"]}', 'finish_reason': 'stop'})
        self.assertEqual(r['status'], 'partial')
        self.assertEqual(r['selected_types'], ['person'])
        self.assertEqual(len(r['rejected_items']), 1)

    def test_values_preserve_repeat_null_and_equal_label(self):
        p = self.rails.before_model_call('S2', query(), 'person')
        r = self.rails.after_model_call(p, {'text': '{"values":["person","person",null]}', 'finish_reason': 'stop'})
        self.assertEqual([e['filler'] for e in r['entities']], ['person', 'person', None])

    def test_empty_different_from_failure(self):
        p = self.rails.before_model_call('S1', query(), 'f1')
        empty = self.rails.after_model_call(p, {'text': '{"slot_types":[]}', 'finish_reason': 'stop'})
        failed = self.rails.after_model_call(p, {'text': '{"slot_types":[]}', 'finish_reason': 'length', 'usage': {'completion_tokens': 256}})
        self.assertEqual(empty['status'], 'valid_empty')
        self.assertEqual(failed['status'], 'incomplete')
        self.assertEqual(failed['usage']['completion_tokens'], 256)

    def test_duplicate_keys_not_silently_overwritten(self):
        p = self.rails.before_model_call('S1', query(), 'f1')
        r = self.rails.after_model_call(p, {'text': '{"slot_types":[],"slot_types":["person"]}', 'finish_reason': 'stop'})
        self.assertEqual(r['status'], 'protocol_error')

    def test_rails_do_not_mutate_knowledge(self):
        k = knowledge()
        original = copy.deepcopy(k)
        r = StageRails(k, CONFIG)
        r.before_model_call('S2', query(), 'person')
        self.assertEqual(k, original)

    def test_request_identity_changes_with_prompt(self):
        c = copy.deepcopy(CONFIG)
        c['s1_instruction'] += ' Another policy.'
        a = self.rails.before_model_call('S1', query(), 'f1')
        b = StageRails(knowledge(), c).before_model_call('S1', query(), 'f1')
        self.assertNotEqual(a['request_id'], b['request_id'])

    def test_build_preserves_equal_type_value_and_removes_bad(self):
        rows = [{'slurp_id': n, 'sentence': value, 'tokens': [{'id': 0, 'surface': value}],
                 'entities': [{'type': 'person', 'span': [0]}]} for n, value in [(1, 'person'), (2, 'bad')]]
        bundle = {'bad_components': ['bad'], 'dev_components': [], 'source_components': {'1': 'good', '2': 'bad'},
                  'snapshot': {'values': []}, 'original_sources': {}}
        policy = {'construction_split': 'train', 'kinds': {'person': 'enumerated_observed'}, 'closed_world_types': [],
                  'separation_groups': [], 'train_value_overlap_edges': True, 'max_types_per_fold': 8}
        k = build(rows, bundle, policy)
        self.assertEqual(k['report']['value_equals_type_mentions_retained'], 1)
        self.assertEqual([v['normalized'] for v in k['cards']['person']['vocabulary']], ['person'])

    def test_demo_keeps_all_same_type_mentions(self):
        rows = [{'slurp_id': 1, 'sentence': 'alice and alice',
                 'tokens': [{'id': 0, 'surface': 'alice'}, {'id': 1, 'surface': 'and'}, {'id': 2, 'surface': 'alice'}],
                 'entities': [{'type': 'person', 'span': [0]}, {'type': 'person', 'span': [2]}]}]
        b = {'bad_components': [], 'dev_components': [], 'source_components': {'1': 'a'},
             'snapshot': {'values': []}, 'original_sources': {}}
        policy = {'construction_split': 'train', 'kinds': {'person': 'open'}, 'closed_world_types': [],
                  'separation_groups': [], 'train_value_overlap_edges': True, 'max_types_per_fold': 8}
        k = build(rows, b, policy)
        self.assertEqual(k['cards']['person']['examples'][0]['values'], ['alice', 'alice'])


class AsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_folds_global_limit_and_stage_dependency(self):
        c = copy.deepcopy(CONFIG)
        c['max_inflight_requests'] = 2
        rails = StageRails(knowledge(), c)
        active, peak, done = 0, 0, []
        async def caller(p, after):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if p['stage'] == 'S2':
                self.assertEqual(len([x for x in done if x == 'S1']), 2)
            await asyncio.sleep(0.01)
            text = json.dumps({'slot_types': p['allowed_types']} if p['stage'] == 'S1' else {'values': ['value']})
            active -= 1
            done.append(p['stage'])
            return after(p, {'text': text, 'finish_reason': 'stop'})
        result = await SlotStages(rails, caller).run(query())
        self.assertEqual(peak, 2)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['s1']), 2)
        self.assertEqual(len(result['s2']), 3)

    async def test_empty_selection_no_s2(self):
        async def caller(p, after):
            self.assertEqual(p['stage'], 'S1')
            return after(p, {'text': '{"slot_types":[]}', 'finish_reason': 'stop'})
        result = await SlotStages(StageRails(knowledge(), CONFIG), caller).run(query())
        self.assertEqual(result['s2'], [])
        self.assertTrue(result['complete'])

    async def test_failed_fold_not_empty_and_good_fold_continues(self):
        async def caller(p, after):
            if p['branch'] == 'f1':
                return after(p, {'text': '', 'finish_reason': 'length'})
            return after(p, {'text': '{"slot_types":["artist_name"]}' if p['stage'] == 'S1' else '{"values":[null]}', 'finish_reason': 'stop'})
        result = await SlotStages(StageRails(knowledge(), CONFIG), caller).run(query())
        self.assertFalse(result['complete'])
        self.assertEqual(result['selected_types'], ['artist_name'])
        self.assertIsNone(result['entities'][0]['filler'])

    async def test_one_limit_shared_by_multiple_samples(self):
        c = copy.deepcopy(CONFIG)
        c['max_inflight_requests'] = 2
        active, peak = 0, 0
        async def caller(p, after):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.005)
            active -= 1
            return after(p, {'text': '{"slot_types":[]}', 'finish_reason': 'stop'})
        node = SlotStages(StageRails(knowledge(), c), caller)
        await asyncio.gather(node.run(query()), node.run(Query('other', query().audio)))
        self.assertEqual(peak, 2)

    @unittest.skipUnless(os.environ.get('S_STAGES_RUNNER_PATH'), 'requires existing journal runner')
    async def test_existing_journal_replays_and_never_resends_unknown(self):
        import importlib.util
        import sys
        import tempfile
        from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal
        from speech_aware_evidence_acquisition.rails.slot_stages_native import JournalStageTransport
        path = Path(os.environ['S_STAGES_RUNNER_PATH'])
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location('s_stages_original_runner', path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as temp:
            journal = object.__new__(ContactJournal)
            journal.root = Path(temp) / 'journal'
            journal.identity = {'run_id': 'S-STAGES-UNIT', 'attempt_id': 'offline', 'arm': 's-stages',
                                'protocol_hash': 'a' * 64, 'split_identity_hash': 'b' * 64}
            posted = []
            def post(wire):
                posted.append(wire)
                return canonical({'choices': [{'message': {'content': '{"slot_types":[]}'}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 4}})
            transport = JournalStageTransport(journal=journal, post=post, observed_response=runner.observed_response,
                admission=runner.Admission(Path(temp) / 'STOP'), check=lambda p: None)
            p = StageRails(knowledge(), CONFIG).before_model_call('S1', query(), 'f1')
            first = await transport(p, p['request_id'])
            second = await transport(p, p['request_id'])
            self.assertEqual(first.wire_response_bytes, second.wire_response_bytes)
            self.assertEqual(len(posted), 1)
            def unknown(wire):
                posted.append(wire)
                raise TimeoutError('ambiguous dispatch')
            transport.post = unknown
            p2 = StageRails(knowledge(), CONFIG).before_model_call('S1', query(), 'f2')
            with self.assertRaises(TimeoutError):
                await transport(p2, p2['request_id'])
            with self.assertRaisesRegex(ValueError, 'UNKNOWN_NO_RESEND'):
                await transport(p2, p2['request_id'])
            self.assertEqual(len(posted), 2)

    @unittest.skipUnless(os.environ.get('S_STAGES_NATIVE_TEST') == '1', 'requires fixed openJiuwen image')
    async def test_real_openjiuwen_hooks_parallel(self):
        from speech_aware_evidence_acquisition.rails.slot_stages_native import NativeStageCaller
        from speech_aware_evidence_acquisition.rails.core_client import CoreCompletion
        count = []
        async def transport(packet, contact):
            await asyncio.sleep(0.005)
            count.append(packet['stage'])
            text = json.dumps({'slot_types': packet['allowed_types'][:1]} if packet['stage'] == 'S1' else {'values': ['fixture', None]})
            raw = canonical({'choices': [{'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
                             'usage': {'prompt_tokens': 5, 'completion_tokens': 3}})
            return CoreCompletion(text=text, request_id=packet['request_id'], sample_id=packet['sample_id'],
                agent_contact_id=contact, request_sha256=packet['wire_sha256'], response_sha256=hashlib.sha256(raw).hexdigest(),
                response_text_sha256=hashlib.sha256(text.encode()).hexdigest(), tool_calls_json=b'null',
                wire_response_bytes=raw, tool_names=(), tool_schemas_sha256=hashlib.sha256(b'[]').hexdigest())
        result = await SlotStages(StageRails(knowledge(), CONFIG), NativeStageCaller(transport)).run(query())
        self.assertTrue(result['complete'])
        self.assertEqual(count.count('S1'), 2)
        self.assertEqual(count.count('S2'), 2)
        self.assertEqual(len(result['entities']), 4)


if __name__ == '__main__':
    unittest.main()
