import asyncio
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from speech_aware_evidence_acquisition.rails import configured_s1 as s


class ReadyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'relocated/deep/prepared/Z'
        self.out.mkdir(parents=True)
        self.task_path, self.lane_path, self.index_path = [self.root / n for n in ['task.json', 'lane.json', 'index.json']]
        for n, v in [('policy.json', dict(max_inflight_requests=64, s1_max_tokens=256, model='model')),
                     ('knowledge.json', {}), ('fit.json', {}), ('audio.json', {'audio': 1})]:
            s.save(self.root / n, v)
        self.rows = [dict(sample_id='sample-Z', lane='Z', audio_source=dict(path='audio.json', sha256=s.file_sha(self.root/'audio.json')),
                         inherited_fit=dict(budget=dict(path='fit.json',sha256=s.file_sha(self.root/'fit.json'))))]
        (self.root/'manifest.jsonl').write_bytes(s.canonical(self.rows[0])+b'\n')
        self.task=dict(schema='configured-s1-task-v1',stage='S1',image=s.IMAGE,context_length=65536,
                       concurrency=64,model='model',runtime_config=self.pin('policy.json'),manifest=self.pin('manifest.jsonl'))
        self.lane=dict(lane='Z',expected_samples=1,service=dict(context_length=65536,fixed_image_sha256=s.IMAGE,session_id='original'))
        self.write_context()
        s.save(self.out/'budget.json',dict(status='PASS'))
        self.ready()

    def pin(self,n):return dict(path=n,sha256=s.file_sha(self.root/n))
    def write_context(self):
        s.save(self.task_path,self.task);s.save(self.lane_path,self.lane)
        s.save(self.index_path,dict(lane='Z',task_sha256=s.file_sha(self.task_path),lane_sha256=s.file_sha(self.lane_path),
            dependencies=[self.pin(n) for n in ['policy.json','fit.json','knowledge.json']],audio=[self.pin('audio.json')]))
    def ready(self):
        ident=s.context(self.root,self.task_path,self.lane_path,self.index_path)[-1]
        s.save(self.out/'READY.json',dict(status='CPU_READY_FOR_REVIEW',identity=ident,
            artifacts=[dict(path='budget.json',sha256=s.file_sha(self.out/'budget.json'))],
            audio_stat_pins=[dict(path='audio.json',stat=s.stat_pin(self.root/'audio.json'))]))
    def validate(self):return s.validate_ready(self.root,self.out,self.task_path,self.lane_path,self.index_path)
    def ensure(self):return s.ensure_ready(self.root,self.out,self.task_path,self.lane_path,self.index_path,'/explicit/model')

    def test_relocated_paths_and_nonabc_lane(self):self.assertEqual(self.validate()['status'],'CPU_READY_FOR_REVIEW')
    def test_path_traversal_and_absolute_rejected(self):
        for p in ['../guess','/old/implicit']:
            with self.assertRaises(ValueError):s.resolve(self.root,p)
    def test_same_configuration_noop_without_prepare(self):
        with patch.object(s,'prepare',side_effect=AssertionError('must not prepare')):
            self.assertEqual(self.ensure()['status'],'NOOP_ALREADY_READY')
    def test_missing_dependency_revokes_old_ready(self):
        (self.root/'knowledge.json').unlink()
        with self.assertRaises(FileNotFoundError):self.ensure()
        self.assertFalse((self.out/'READY.json').exists())
        self.assertEqual(s.read(self.out/'PREPARE-FAILED.json')['status'],'FAILED_NO_READY')
    def test_config_change_cannot_use_old_ready(self):
        self.task['timeout_seconds']=1200;self.write_context()
        with self.assertRaises(ValueError):self.validate()
        with patch.object(s,'prepare',side_effect=RuntimeError('new FIT failed')):
            with self.assertRaises(RuntimeError):self.ensure()
        self.assertFalse((self.out/'READY.json').exists())
    def test_service_identity_change_invalidates(self):
        self.lane['service']['session_id']='other';self.write_context()
        with self.assertRaises(ValueError):self.validate()
    def test_input_index_binding_change_invalidates(self):
        self.task['expected_samples']=999;s.save(self.task_path,self.task)
        with self.assertRaisesRegex(ValueError,'task/lane'):self.validate()
    def test_artifact_change_invalidates(self):
        s.save(self.out/'budget.json',dict(status='OLD_PASS'))
        with self.assertRaisesRegex(ValueError,'artifact'):self.validate()
    def test_audio_changed_cannot_noop(self):
        s.save(self.root/'audio.json',dict(audio=2))
        with self.assertRaisesRegex(ValueError,'audio changed'):self.validate()
    def test_restore_exact_audio_and_immutable_template(self):
        audio=dict(type='input_audio',input_audio=dict(data='real inherited bytes'))
        template=dict(wire=dict(messages=[{},dict(content=[dict(type='text',text='ref'),{'__reuse_audio__':True}])]))
        original=copy.deepcopy(template)
        rebuilt=s.restore(template,audio)
        self.assertEqual(rebuilt['wire']['messages'][1]['content'][1],audio)
        self.assertEqual(template,original)
    def test_restore_requires_exact_one_placeholder(self):
        for content in [[],[{'__reuse_audio__':True},{'__reuse_audio__':True}]]:
            with self.assertRaises(ValueError):s.restore(dict(wire=dict(messages=[{},dict(content=content)])),{})


class MarkingSchemaTests(unittest.TestCase):
    def setUp(self):
        from speech_aware_evidence_acquisition.rails.slot_stages import StageRails
        from test_slot_stages import knowledge, query, CONFIG
        self.baseline = {**CONFIG, 's1_none_label': 'NONE_OF_FOLD'}
        self.policy = {**CONFIG, 's1_candidate_presentation': 'type_marking_v1',
                       's1_instruction': 'mark each type'}
        self.changed = ['s1_candidate_presentation', 's1_instruction', 's1_none_label']
        self.old = StageRails(knowledge(), self.baseline).before_model_call('S1', query(), 'f1')['wire']['response_format']
        self.new = StageRails(knowledge(), self.policy).before_model_call('S1', query(), 'f1')['wire']['response_format']

    def check(self, current, **kw):
        s.schema_change(self.old, current, kw.get('baseline', self.baseline),
                        kw.get('policy', self.policy), kw.get('changed', self.changed))

    def test_exact_marking_transform_does_not_mutate_parent(self):
        before = copy.deepcopy(self.old)
        s.scientific_changes({'allowed_changes': self.changed}, self.baseline, self.policy)
        self.check(self.new)
        self.assertEqual(self.old, before)

    def test_missing_key_wrong_value_and_extra_schema_edits_rejected(self):
        for mutation in ('missing', 'nonboolean', 'wrapper'):
            bad = copy.deepcopy(self.new)
            presence = bad['json_schema']['schema']['properties']['slot_type_presence']
            if mutation == 'missing':
                presence['required'].pop()
            elif mutation == 'nonboolean':
                presence['properties'][next(iter(presence['properties']))]['type'] = 'string'
            else:
                bad['json_schema']['name'] = 'unrelated'
            with self.assertRaises(ValueError):
                self.check(bad)

    def test_parent_and_declared_intervention_boundaries(self):
        for change in (dict(baseline={**self.baseline, 's1_none_label': 'OTHER_SLOT_TYPE'}),
                       dict(policy={**self.policy, 's1_none_label': 'NONE_OF_FOLD'}),
                       dict(changed=self.changed + ['s1_max_tokens'])):
            with self.assertRaises(ValueError):
                self.check(self.new, **change)
        with self.assertRaises(ValueError):
            s.scientific_changes({'allowed_changes': self.changed}, self.baseline,
                                 {**self.policy, 's1_max_tokens': 512})


class SupplyParentPreparationTests(unittest.TestCase):
    def check_prepare_branch(self, parent_consumes, edit):
        from types import SimpleNamespace
        from speech_aware_evidence_acquisition.rails.slot_stages import StageRails
        from test_slot_stages import CONFIG, knowledge

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            k = knowledge()
            text = 'A sourced distinction.'
            s.save(root / 'sources/rule.json', {'text': text})
            k['s1_discriminative_rules'] = [dict(
                rule_id='r1', target_types=['person'], related_types=['artist_name'],
                description=text, source_components=['other'],
                source_refs=[dict(path='rule.json', sha256=s.file_sha(root/'sources/rule.json'), pointer='/text')],
                scope=dict(stage='S1', intent_independent=True, multi_candidate=True,
                           final_adjudication=False, condition='current_audio_evidence'))]
            cache = dict(schema='s1-audio-sentence-retrieval-cache-v1', knowledge_id=k['knowledge_id'],
                         encoder=dict(identity_sha256='fixture', query_modality='audio', key_unit='full_train_sentence'),
                         input_pins={'fixture': 'fixture'}, bank=[], queries=[])
            cache['cache_id'] = s.sha(s.canonical(cache))
            consumption = dict(schema='s1-consumption-policy-v1', cache_id=cache['cache_id'],
                               encoder_identity_sha256='fixture', max_examples=2, max_example_bytes=2000,
                               max_rules=2, max_rule_bytes=2000)
            baseline = copy.deepcopy(CONFIG)
            if parent_consumes:
                baseline['s1_consumption'] = consumption
            policy = copy.deepcopy(baseline)
            if edit == 's1_consumption':
                policy['s1_consumption'] = {**consumption, 'max_examples': 1}
            else:
                policy['s1_instruction'] += ' Additional audio role instruction.'
            for name, value in [('knowledge', k), ('baseline', baseline), ('cache', cache)]:
                s.save(root / name, value)
            task = dict(knowledge=dict(path='knowledge', sha256=s.file_sha(root/'knowledge')),
                        baseline_runtime_config=dict(path='baseline', sha256=s.file_sha(root/'baseline')),
                        retrieval_cache=dict(path='cache', sha256=s.file_sha(root/'cache')),
                        provenance_root='sources', folds=2, types=3, model_root='/explicit/model',
                        model_assets=[], native_root='native', fit_helper='fit', grammar_evidence={},
                        allowed_changes=[edit])
            built = []

            def construct(*args, **kwargs):
                rails = StageRails(*args, **kwargs)
                built.append(rails)
                return rails

            class BeforeFit(Exception):
                pass

            # Execute the real prepare branch, builders, provenance validation,
            # and both Fold schema comparisons; stop before audio/FIT.
            with patch.object(s, 'context', return_value=(task, {}, {}, [], policy, {})), \
                 patch.object(s, 'build_rails', wraps=s.build_rails) as builder, \
                 patch('speech_aware_evidence_acquisition.rails.slot_stages.StageRails', side_effect=construct) as constructor, \
                 patch.object(s, 'load', return_value=SimpleNamespace(bind_local_chat_template=lambda *args: {})), \
                 patch('transformers.AutoTokenizer.from_pretrained', return_value=object()), \
                 patch.object(s, 'save', side_effect=BeforeFit) as saved:
                with self.assertRaises(BeforeFit):
                    s.prepare(root, root, None, None, None, '/explicit/model')
            self.assertEqual(builder.call_count, 2 if parent_consumes else 1)
            self.assertEqual(constructor.call_count, 2)
            self.assertIsNotNone(built[0].s1_consumer)
            self.assertEqual(constructor.call_args_list[0].kwargs,
                             dict(retrieval_cache=cache, provenance_root=root/'sources'))
            if parent_consumes:
                self.assertEqual(constructor.call_args_list[1].kwargs, constructor.call_args_list[0].kwargs)
                expected = StageRails(k, baseline, retrieval_cache=cache, provenance_root=root/'sources')
            else:
                self.assertEqual(constructor.call_args_list[1].kwargs, {})
                self.assertIsNone(built[1].s1_consumer)
                expected = StageRails(k, baseline)
            self.assertEqual(built[1].version, expected.version)
            self.assertEqual(saved.call_args.args[0].name, 'GRAMMAR-COMPATIBILITY.json')
            self.assertEqual(len(saved.call_args.args[1]['schemas']), 2)

    def test_prepare_grounding_parent_without_consumption_keeps_legacy_path(self):
        self.check_prepare_branch(False, 's1_consumption')

    def test_prepare_supply_parent_consumes_cache_and_provenance_for_u_and_e(self):
        for edit in ('s1_instruction', 's1_consumption'):
            with self.subTest(edit=edit):
                self.check_prepare_branch(True, edit)

    def test_candidate_without_consumption_still_rejects_supply_inputs(self):
        for task in ({'retrieval_cache': {}}, {'provenance_root': 'sources'},
                     {'retrieval_cache': {}, 'provenance_root': 'sources'}):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, 'explicit S1 consumption policy'):
                s.build_rails(Path('/root'), task, {}, {})

    def test_consuming_candidate_still_requires_both_supply_inputs(self):
        for task in ({}, {'retrieval_cache': {}}, {'provenance_root': 'sources'}):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, 'pinned cache and explicit provenance'):
                s.build_rails(Path('/root'), task, {}, {'s1_consumption': {}})

    def test_prepare_passes_explicit_supply_inputs_to_parent_builder(self):
        from types import SimpleNamespace
        baseline = {'s1_consumption': {'cache_id': 'fixed'}, 's1_instruction': 'old'}
        policy = {**baseline, 's1_instruction': 'new'}
        knowledge = {'folds': [], 'cards': {}}
        task = dict(knowledge={'path': 'knowledge'}, baseline_runtime_config={'path': 'baseline'},
                    retrieval_cache={'path': 'cache'}, provenance_root='sources',
                    folds=0, types=0, model_root='/explicit/model', model_assets=[],
                    native_root='native', fit_helper='fit', grammar_evidence={})
        helper = SimpleNamespace(bind_local_chat_template=lambda *args: {})
        # Stop at the grammar receipt, before audio or FIT. The real builder
        # contract remains explicit for both candidate and supply parent.
        class BeforeFit(Exception):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(s, 'context', return_value=(task, {}, {}, [], policy, {})), \
                 patch.object(s, 'pinned', side_effect=lambda root, pin: pin['path']), \
                 patch.object(s, 'read', side_effect=lambda path: knowledge if path == 'knowledge' else baseline), \
                 patch.object(s, 'build_rails', return_value=object()) as builder, \
                 patch.object(s, 'load', return_value=helper), \
                 patch('transformers.AutoTokenizer.from_pretrained', return_value=object()), \
                 patch.object(s, 'save', side_effect=BeforeFit):
                with self.assertRaises(BeforeFit):
                    s.prepare(Path(tmp), Path(tmp), None, None, None, '/explicit/model')
            self.assertEqual(builder.call_count, 2)
            self.assertEqual(builder.call_args_list[0].args, (Path(tmp), task, knowledge, policy))
            self.assertEqual(builder.call_args_list[1].args, (Path(tmp), task, knowledge, baseline))

    def test_builder_binds_cache_and_provenance_for_both_policies(self):
        task = dict(retrieval_cache={'path': 'cache'}, provenance_root='sources')
        cache = {'cache_id': 'fixed'}
        with patch.object(s, 'pinned', return_value='cache'), \
             patch.object(s, 'read', return_value=cache), \
             patch('speech_aware_evidence_acquisition.rails.slot_stages.StageRails') as rails:
            for instruction in ('old', 'new'):
                policy = dict(s1_consumption={'cache_id': 'fixed'}, s1_instruction=instruction)
                s.build_rails(Path('/root'), task, {}, policy)
                rails.assert_called_with({}, policy, retrieval_cache=cache,
                                         provenance_root=Path('/root/sources'))


if __name__ == '__main__':
    unittest.main()
