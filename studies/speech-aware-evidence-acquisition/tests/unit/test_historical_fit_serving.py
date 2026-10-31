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


if __name__ == '__main__':
    unittest.main()
