"""Closed numerical rejection settles cost, never re-forwards, ends resident."""
import argparse,contextlib,json,sys,time
from pathlib import Path
from unittest.mock import patch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
from dependencies import load
from run_package import read,write
iteration=load(Path(a.source),'rejection_iteration')
import finite_runtime,sampled_training,diagnostic_retention,storage,parity_gate,resident_transport
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
dummy=write(out/'dummy.json',dict(version='fixture',resident_identity={}))
cfg=dict(schema='omni-dp8-iteration-v1',output_root=str(out/'run'),run_id='rejection-fixture',
 monitor_seconds=900,max_optimizer_updates=1,deadline_epoch=time.time()+600,client_storage_bytes=0,
 execution_mode='finite_engineering',initial_input=dummy,trajectory=dummy,program_predecessor_output=None,
 resident={'enabled':True},budget={'prior_tokens_upper':0,'total_tokens':1000},stage_budgets={'diagnostic':{'tokens_upper':10}})
ref=write(out/'config.json',cfg)
state=dict(config_ref=ref,execution_mode='finite_engineering',phase='diagnostic',step=0,sequence=1,input_ref=dummy,policy_ref=dummy,
 learning_ref=dummy,resume_ref=None,trajectory_ref=dummy,witness_ref=dummy,charged_tokens_upper=10,
 reservations={'1-diagnostic':dict(stage='diagnostic',tokens_upper=10,status='intent_unknown')},
 pending_stage=dict(key='1-diagnostic',stage='diagnostic',sequence=1,purpose=None))
write(out/'run/CURSOR.json',state)
calls=[];ends=[];verifications=[]
def train(*args,**kwargs):
 calls.append(True);return dict(raw_ref=dummy,runtime_ref=dummy,charged_tokens_upper=37)
def verify(*args,**kwargs):
 verifications.append(True)
 current=read(out/'run/CURSOR.json')
 assert current['pending_stage'] is None and current['charged_tokens_upper']==37
 if len(verifications)==1:raise FileNotFoundError('fixture interruption after closure')
 raise ValueError('real DP8 R/C/P failed; original thresholds retained')
with contextlib.ExitStack() as stack:
 for obj,name,fn in [(finite_runtime,'validate_mode',lambda c:'finite_engineering'),
  (sampled_training,'specification',lambda c:None),(diagnostic_retention,'collect_ready',lambda *args:None),
  (storage,'configure_scope',lambda *args,**kw:None),(storage,'environment',lambda *args:None),
  (iteration,'train',train),(parity_gate,'verify',verify),(resident_transport,'end',lambda c:ends.append(True))]:
  stack.enter_context(patch.object(obj,name,fn))
 try:iteration.run(ref['path'])
 except FileNotFoundError:pass
 else:raise AssertionError('expected interruption')
 result=iteration.run(ref['path'])
 assert result['status']=='DIAGNOSTIC_REJECTED' and result['steps']==0 and result['achieved'] is False
 assert len(calls)==1 and ends==[True]
 state=read(out/'run/CURSOR.json')
 assert state['reservations']['1-diagnostic']['status']=='terminal_known' and state['charged_tokens_upper']==37
write(out/'RESULT.json',dict(passed=True,closed_command_calls=1,verification_attempts=2,
 optimizer_steps=0,resident_end_calls=1,cost_watermark_preserved=True,fixture=True))
print('PASS closed diagnostic rejection and interrupted verification recovery')
