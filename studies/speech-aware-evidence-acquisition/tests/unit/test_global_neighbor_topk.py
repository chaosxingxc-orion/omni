import unittest
from speech_aware_evidence_acquisition.analysis.global_neighbor_topk import Config, aggregate

class TopKTests(unittest.TestCase):
    def test_global_not_class_mean5(self):
        m = [dict(intent='a', source_group=i) for i in range(5)] + [dict(intent='b', source_group=5)]
        r = aggregate([.99,.1,.1,.1,.1,.9], m, Config(k=2))
        self.assertEqual([x['label'] for x in r['labels']], ['a','b'])
        self.assertLess((.99+.1*4)/5, .9)
    def test_duplicates_ties_and_variable_labels(self):
        m = [dict(intent=l, source_group=g) for l,g in [('x',1),('y',1),('x',2),('z',3)]]
        raw = aggregate([.9,.9,.8,.7],m,Config(k=3,source_distinct=False))
        dedup = aggregate([.9,.9,.8,.7],m,Config(k=3))
        self.assertEqual([n['member_index'] for n in raw['neighbors']], [0,1,2])
        self.assertEqual([n['member_index'] for n in dedup['neighbors']], [0,2,3])
        self.assertEqual([x['label'] for x in dedup['labels']], ['x','z'])
        self.assertEqual(dedup['labels'][0]['count'], 2)
        self.assertEqual(aggregate([.9,.9,.8,.7],m,Config(k=1))['labels'][0]['label'], 'x')
        self.assertEqual(aggregate([.9,.9,.8,.7],m,Config(k=1),{1})['labels'][0]['label'], 'x')
        with self.assertRaises(NotImplementedError): aggregate([.9]*4,m,Config(high_confidence_enabled=True))

    def test_query_source_exclusion_and_label_cardinality(self):
        members = [dict(intent=label, source_group=group) for label, group in
                   [('a', 1), ('b', 2), ('a', 3), ('c', 4)]]
        for distinct in (True, False):
            result = aggregate([.99, .9, .8, .7], members,
                               Config(k=3, source_distinct=distinct), {1})
            self.assertEqual([n['member_index'] for n in result['neighbors']], [1, 2, 3])
            self.assertEqual([v['label'] for v in result['labels']], ['b', 'a', 'c'])
        self.assertEqual(len(aggregate([.99, .9, .8, .7], members, Config(k=2))['labels']), 2)


class RunnerTests(unittest.TestCase):
    def test_external_pinned_inputs_and_exclusion(self):
        import hashlib
        import importlib.util
        import json
        import tempfile
        from pathlib import Path
        import numpy as np
        study = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location('topk_runner', study / 'scripts/analysis/global_neighbor_topk_runner.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            members = root / 'members.json'
            queries = root / 'queries.json'
            matrix = root / 'matrix.npy'
            members.write_text(json.dumps([{'intent': 'a', 'source_group': 1}, {'intent': 'b', 'source_group': 2}]))
            queries.write_text(json.dumps([{'sample_id': 'q', 'source_group': 1}]))
            np.save(matrix, np.array([[.9, .8]]))
            def pin(p):
                return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            config = {'members': pin(members), 'queries': pin(queries), 'matrices': {'x': pin(matrix)},
                      'k': [1], 'source_distinct': [True, False]}
            result = runner.run(config)
            self.assertEqual(result['routes']['x/raw/k1'][0]['labels'][0]['label'], 'b')
            members.write_text('[]')
            with self.assertRaises(ValueError):
                runner.run(config)

if __name__ == '__main__':
    unittest.main()
