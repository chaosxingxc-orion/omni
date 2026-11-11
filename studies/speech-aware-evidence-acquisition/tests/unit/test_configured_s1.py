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


if __name__ == '__main__':
    unittest.main()
