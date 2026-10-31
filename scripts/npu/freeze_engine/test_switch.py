"""Ownership and quiescence refusal before a recorded deployment mutation."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import copy

spec=importlib.util.spec_from_file_location('engine_switch',Path(__file__).with_name('switch.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class GuardTests(unittest.TestCase):
    def test_identity_refuses_foreign_or_changed_container(self):
        candidate={'name':'saea-omni-i1'}
        good={'Name':'/saea-omni-i1','Config':{'Labels':{m.LABEL:m.OLD_PROFILES['saea-omni-i1']}},'Image':m.IMAGE,'State':{'Running':True},'HostConfig':{'RestartPolicy':{'Name':'no'}}}
        m.validate_original(good,candidate)
        for field,value in [('Name','/foreign'),('Image','wrong'),('State',{'Running':False}),('Config',{'Labels':{m.LABEL:'foreign'}}),('HostConfig',{'RestartPolicy':{'Name':'always'}})]:
            changed=copy.deepcopy(good);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):m.validate_original(changed,candidate)
    def test_busy_and_missing_metrics_refuse(self):
        class Response:
            def __init__(self,text):self.text=text
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return self.text.encode()
        for text in ('vllm:num_requests_running 1\nvllm:num_requests_waiting 0','vllm:num_requests_running 0',''):
            with patch.object(m.urllib.request,'urlopen',return_value=Response(text)),self.assertRaises(ValueError):m.quiescent([{'port':8201,'name':'saea-omni-i1'}])
        with patch.object(m.urllib.request,'urlopen',return_value=Response('vllm:num_requests_running 0\nvllm:num_requests_waiting 0')):m.quiescent([{'port':8201,'name':'saea-omni-i1'}])

if __name__=='__main__':unittest.main()
