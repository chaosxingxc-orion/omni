"""Actual prepared workflow, state bounds and relocated consumer regression."""
import argparse,copy,importlib.util,json,tempfile
from pathlib import Path
from run_package import read,pin,write
from finite_runtime import engineering_updates
from unified_config import bind
from checkpoint_budget import calculate
from dependencies import HERE,JEV
p=argparse.ArgumentParser()
for name in ('output','prepared','consumer'):p.add_argument('--'+name,required=True)
a=p.parse_args();root=Path(a.output)
checks=[]
base=read(a.prepared)
spec=dict(schema='engineering-update-sequence-v1',updates=3,fresh_actions_each_step=True,retain_optimizer_state=True,business_gain_claim=False)
for count in (1,3):
 value=copy.deepcopy(base);c=value['iteration'];package=value['training_package'];wf=c['unified_workflow']
 wf['max_optimizer_updates']=count
 if count==3:wf['engineering_update_sequence']=spec
 old=read(wf['checkpoint_budget_ref']);b=read(old['config_ref'])
 b.update(workflow_kind='finite_update',max_optimizer_updates=count,max_source_tasks=8*count+12,resident_max_commands=8*count+12)
 for role,row in old['roles'].items():
  if role in b['noncheckpoint_envelopes']:b['noncheckpoint_envelopes'][role]=copy.deepcopy(row['envelopes'])
 br=write(root/str(count)/'BUDGET-INPUT.json',b);b['_config_path']=br['path']
 wf['checkpoint_budget_ref']=write(root/str(count)/'BUDGET.json',calculate(b,HERE/'worker_dp8.py',JEV/'src/multimodal_jev/context_training_checkpoint.py'))
 bind(value,'finite_engineering',root/str(count)/'workflow')
 assert c['max_optimizer_updates']==package['budget']['max_optimizer_updates']==count
 assert c['resident']['max_commands']==8*count+12 and c['formal_rl_enabled'] is False
 assert engineering_updates(c)==count
 checks.append('actual workflow and budget '+str(count))
for field,bad in [('retain_optimizer_state',False),('fresh_actions_each_step',False),('business_gain_claim',True),('updates',4)]:
 altered=dict(spec,**{field:bad})
 try:engineering_updates(dict(execution_mode='finite_engineering',formal_rl_enabled=False,max_optimizer_updates=3,engineering_update_sequence=altered))
 except ValueError:checks.append('reject '+field)
 else:raise AssertionError(field)
path=HERE.parent.parent/'training-consumer-adapter/adapter.py'
sp=importlib.util.spec_from_file_location('relocated_adapter',path);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
c=read(a.consumer)
with mod.unified(c) as (learning,schedule,kernel):assert learning.__file__==str(HERE.parent.parent/'unified-learning/learning.py')
checks.append('physical predecessor pins import designated exact learning source')
with tempfile.TemporaryDirectory(dir=str(root)) as temp:
 bad=copy.deepcopy(c);f=Path(temp)/'learning.py';f.write_bytes(b'raise RuntimeError("different code")')
 bad['source_pins']['learning.py']=pin(f)
 try:
  with mod.unified(bad):raise AssertionError('drift accepted')
 except ValueError:checks.append('reject authentic but wrong designated source')
write(root/'RESULT.json',dict(passed=True,checks=checks,optimizer_updates=0,model_calls=0));print('PASS',len(checks))
