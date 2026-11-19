"""Scoped setting, one original call, ownership and exception restoration."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('audio_candidate',a.source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
state={'value':b'enable'};changes=[]
def put(values):
    state['value']=values['ALLOW_INTERNAL_FORMAT'].encode();changes.append(state['value'])
fake=SimpleNamespace(_C=SimpleNamespace(_npu_getOption=lambda key:state['value'],_npu_setOption=put))
tower=torch.nn.Module()
for name in ('conv2d1','conv2d2','conv2d3'):setattr(tower,name,torch.nn.Conv2d(1,1,1))
x=torch.ones(1,1,2,2);expected=tower.conv2d1(x);original=tower.conv2d1.forward
with patch.dict(sys.modules,{'torch_npu':fake}):
    audit,restore=module.attach(tower)
    assert torch.equal(tower.conv2d1(x),expected) and state['value']==b'enable'
    assert audit['calls']=={'conv2d1':1} and not hasattr(tower.conv2d2,'_omni_native_conv_context')
    assert changes==[b'disable',b'enable']
    try:module.attach(tower)
    except ValueError:pass
    else:raise AssertionError('duplicate owner accepted')
    try:
        with module.native_conv_context():raise RuntimeError('original failure')
    except RuntimeError:pass
    assert state['value']==b'enable'
    restore();restore();assert tower.conv2d1.forward==original and audit['restored']
    state['value']=None
    try:
        with module.native_conv_context():raise AssertionError('unknown config accepted')
    except ValueError:pass
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
result=dict(passed=True,checks=7,cpu_fixture=True,npu_acceptance=False)
(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
