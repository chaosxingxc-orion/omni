"""Offline test for bounded, explicitly scoped client deadline overrides."""
import ast
import copy
import json
from pathlib import Path

def main():
 source=Path(__file__).with_name('host_driver.py')
 tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_runtime_policy')
 ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns);apply=ns['apply_runtime_policy']
 policy={'tasks':{'new':{'timeout_seconds':900}}}
 before=dict(dataset_partition='eligible_remainder',timeout_seconds=600,concurrency=96,manifest_sha256='unchanged')
 cfg=copy.deepcopy(before);change=apply(cfg,{'task_id':'new'},policy)
 assert cfg==dict(before,timeout_seconds=900) and change['original_timeout_seconds']==600
 for task,partition in [('old','eligible_remainder'),('new','S_valid'),('new','fit')]:
  cfg=dict(before,dataset_partition=partition);old=copy.deepcopy(cfg)
  assert apply(cfg,{'task_id':task},policy) is None and cfg==old
 for value in [0,599,1201,True,'900']:
  try:apply(copy.deepcopy(before),{'task_id':'new'},{'tasks':{'new':{'timeout_seconds':value}}})
  except AssertionError:pass
  else:raise AssertionError('unbounded deadline accepted')
 try:apply(copy.deepcopy(before),{'task_id':'new'},{'tasks':{'new':{'concurrency':128}}})
 except AssertionError:pass
 else:raise AssertionError('model/resource change accepted')
 print(json.dumps(dict(status='PASS',tests=['only_targeted_new_TRAIN','evaluation_and_existing_settings_unchanged','bounded_integer_deadline','non_deadline_policy_rejected'],NPU_calls=0)))

if __name__=='__main__':main()
