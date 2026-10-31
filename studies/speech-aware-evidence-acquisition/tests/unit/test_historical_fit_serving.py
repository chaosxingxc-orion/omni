import importlib.util
import json
from pathlib import Path
import unittest
import tempfile

p=Path(__file__).parents[2]/'scripts/analysis/historical_fit_serving.py'
s=importlib.util.spec_from_file_location('historical',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def fixture(root):
    def pin(name, value):
        p=root/name;p.write_text(json.dumps(value));return {'relpath':name,'sha256':m.digest(p)}
    runtime=pin('runtime.json',{'deployment':{'instances':[{'instance_id':'old-i0','host':'localhost','port':8200}]}})
    identity={'model_fragment':'model','model_fragment_sha256':'model-sha','runtime_receipt_sha256':runtime['sha256'],
              'binary_sha256':'binary-sha','model_files':{'weights':'weights-sha'}}
    parent=pin('parent.json',{'session_id':'old-parent','identity':identity,'deployment_instance_ids':['old-i0']})
    session=pin('session.json',{'session_id':'old-i0','deployment_parent_session_id':'old-parent',
               'identity':{**identity,'host':'localhost','port':8200},'pid':123,'started_at':'old-time'})
    receipt=pin('receipt.json',{'session_id':'old-i0','host':'localhost','port':8200,'pid':123,'started_at':'old-time'})
    doc={'schema':m.SCHEMA,'coordinator':'old-parent','model_fragment':'model','parent':parent,'runtime':runtime,
         'members':{'old-i0':{'session':session,'receipt':receipt}}}
    path=root/'snapshot.json';path.write_text(json.dumps(doc))
    return path,doc,identity


def check_historical_fit_binds_own_deployment(tmp_path,fault):
    path,doc,identity=fixture(tmp_path);fit='old-i0'
    if fault=='new-runtime':identity={**identity,'runtime_receipt_sha256':'new-runtime'}
    if fault=='new-session':fit='new-i0'
    if fault in ('endpoint','pid','parent'):
        p=tmp_path/'session.json';value=json.loads(p.read_text())
        if fault=='endpoint':value['identity']['port']=9999
        if fault=='pid':value['pid']=456
        if fault=='parent':value['deployment_parent_session_id']='new-parent'
        p.write_text(json.dumps(value));doc['members']['old-i0']['session']['sha256']=m.digest(p)
    if fault=='missing-receipt':doc['members']['old-i0'].pop('receipt')
    if fault=='escape':doc['runtime']['relpath']='../runtime.json'
    path.write_text(json.dumps(doc));sha=m.digest(path)
    if fault=='snapshot':sha='wrong'
    if fault=='tamper':(tmp_path/'session.json').write_text('{}')
    if fault:
        with unittest.TestCase().assertRaises(ValueError):m.verify(path,sha,coordinator='old-parent',model_fragment='model',fit_session=fit,required_identity=identity)
    else:
        result=m.verify(path,sha,coordinator='old-parent',model_fragment='model',fit_session=fit,required_identity=identity)
        assert result['primary_fit_session']=='old-i0' and len(result['inputs'])==5
        assert result['new_model_calls']==0


class HistoricalFitTests(unittest.TestCase):
    def test_custody_cases(self):
        for fault in [None,'new-runtime','new-session','snapshot','tamper','endpoint','pid','parent','missing-receipt','escape']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                check_historical_fit_binds_own_deployment(Path(directory),fault)


class HistoricalBindingTests(unittest.TestCase):
    def setup_case(self, root):
        path, _, identity = fixture(root)
        source = {'cell': {'attempt_id': 'old', 'model_fragment': 'model'},
                  'doc': {'service_session': 'old-parent'},
                  'fit': {'run_id': 'fit-old', 'attempt_id': 'old-fit'},
                  'predecessor_commit': 'a' * 40}
        directory = root / 'prepared/fit-old/old-fit'
        directory.mkdir(parents=True)
        artifact = directory / 'contact.request-fit.json'
        artifact.write_text(json.dumps({'output': {'session_id': 'old-i0'},
            'inputs': {'sources': {'required_identity': identity}}}))
        current = {'primary_fit_session': 'new-i0', 'members': {
            'new-i0': {**identity, 'runtime_receipt_sha256': 'new-runtime'}}, 'inputs': {}}
        doc = {'schema': m.MAP_SCHEMA, 'read_id': 'read', 'run_id': 'generation',
               'attempt_id': 'new', 'predecessors': {'old': {
                   'predecessor_commit': 'a' * 40, 'fit_run_id': 'fit-old',
                   'fit_attempt_id': 'old-fit',
                   'snapshot': {'relpath': path.name, 'sha256': m.digest(path)}}}}
        mapping = root / 'mapping.json'
        mapping.write_text(json.dumps(doc))
        return source, artifact, current, mapping, doc

    def test_map_scope_and_hash(self):
        for fault in [None, 'hash', 'read', 'run', 'attempt', 'missing-ancestor',
                      'extra-ancestor', 'commit', 'fit-run', 'fit-attempt', 'snapshot', 'escape']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, _, _, path, doc = self.setup_case(root)
                if fault in ('read', 'run', 'attempt'):
                    doc[{'read': 'read_id', 'run': 'run_id', 'attempt': 'attempt_id'}[fault]] = 'foreign'
                if fault == 'missing-ancestor': doc['predecessors'] = {}
                if fault == 'extra-ancestor': doc['predecessors']['foreign'] = {}
                if fault in ('commit', 'fit-run', 'fit-attempt'):
                    doc['predecessors']['old'][{'commit': 'predecessor_commit',
                        'fit-run': 'fit_run_id', 'fit-attempt': 'fit_attempt_id'}[fault]] = 'foreign'
                if fault == 'snapshot': doc['predecessors']['old']['snapshot']['sha256'] = 'wrong'
                if fault == 'escape': doc['predecessors']['old']['snapshot']['relpath'] = '../snapshot.json'
                path.write_text(json.dumps(doc))
                digest = 'wrong' if fault == 'hash' else m.digest(path)
                def run():
                    return m.bindings(path, digest, read_id='read', run_id='generation',
                                      attempt_id='new', predecessors=[source])
                if fault:
                    with self.assertRaises(ValueError): run()
                else:
                    bindings, pins = run()
                    self.assertEqual(set(bindings), {'old'})
                    self.assertEqual(pins, {str(path): digest})

    def test_resolve_never_substitutes_new_runtime(self):
        for fault in [None, 'no-pin', 'model', 'binary', 'weights', 'empty-fit', 'foreign-fit']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, artifact, current, path, _ = self.setup_case(root)
                bindings, _ = m.bindings(path, m.digest(path), read_id='read',
                    run_id='generation', attempt_id='new', predecessors=[source])
                if fault in ('model', 'binary', 'weights'):
                    current['members']['new-i0'][{'model': 'model_fragment_sha256',
                        'binary': 'binary_sha256', 'weights': 'model_files'}[fault]] = 'changed'
                if fault == 'empty-fit': artifact.unlink()
                if fault == 'foreign-fit':
                    doc = json.loads(artifact.read_text())
                    doc['output']['session_id'] = 'new-i0'
                    artifact.write_text(json.dumps(doc))
                def run():
                    return m.resolve(root, source, {'service_session': 'new-parent'}, current,
                                     None if fault == 'no-pin' else bindings['old'])
                if fault:
                    with self.assertRaises(ValueError): run()
                else:
                    result = run()
                    self.assertEqual(result['primary_fit_session'], 'old-i0')
                    self.assertNotEqual(result['members']['old-i0']['runtime_receipt_sha256'],
                                        current['members']['new-i0']['runtime_receipt_sha256'])
                    self.assertIn('not generation attachment', result['custody'])
                self.assertIs(m.resolve(root, source, source['doc'], current), current)


if __name__ == '__main__':
    unittest.main()
