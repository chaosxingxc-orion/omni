"""Report committed optimizer objectives from the original closed result graph."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

def closed_metadata(returned):
 """Read a projection of an already accepted local return; no shard traversal."""
 import hashlib,json
 marker=json.loads((returned/'CLOSED-COMMIT.json').read_bytes())
 def load(path,expected,size=None):
  raw=path.read_bytes()
  if (size is not None and len(raw)!=size) or hashlib.sha256(raw).hexdigest()!=expected:
   raise ValueError('closed objective metadata changed: '+str(path))
  return json.loads(raw)
 if marker.get('immutable') is not True:raise ValueError('uncommitted update return')
 manifest=load(returned/'CLOSED-MANIFEST.json',marker['manifest_sha256'])
 closure=marker['closure'];ret=manifest['RETURN.json']
 if ret['sha256']!=closure['return_sha256']:raise ValueError('closed return hash differs')
 def member(ref):
  name='files/'+ref['path'].lstrip('/')
  path=(returned/name).resolve()
  if not Path(ref['path']).is_absolute() or not path.is_relative_to(returned.resolve()) or manifest.get(name)!=ref:
   raise ValueError('objective reference absent from closed manifest')
  return load(path,ref['sha256'],ref['bytes'])
 closed=member(manifest['files/'+closure['closed_path'].lstrip('/')])
 if (manifest['files/'+closure['closed_path'].lstrip('/')]['sha256']!=closure['closed_sha256']
     or closed.get('immutable') is not True or closed['source_boot']!=closure['source_boot'] or closed['return_ref']!=ret):
  raise ValueError('source closure differs from local commit')
 return load(returned/'RETURN.json',ret['sha256'],ret['bytes']),member


def capture_receipts(root, capture):
 """Resolve capture metadata locally or in its immutable mailbox return."""
 from dependencies import local_path, require
 from run_package import read
 refs=(capture['input_ref'],capture['generation_inventory_ref'],capture['generation_closed_ref'])
 if all(local_path(ref['path']).is_file() for ref in refs):
  return tuple(read(ref) for ref in refs)
 for response in sorted((root/'mailbox').glob('*-capture/RESPONSE.json'),reverse=True):
  answer=read(response)
  if any(answer.get(key)!=ref for key,ref in zip(
      ('input_ref','generation_inventory_ref','generation_closed_ref'),refs,strict=True)):
   continue
  returned,member=closed_metadata(response.parent/'transport/returned')
  require(member(returned['response_ref'])==answer,'capture response differs from closed return')
  return tuple(member(ref) for ref in refs)
 raise ValueError('closed capture receipts unavailable')


def continued_capture(root, cursor, third_update_index):
 from run_package import read
 windows=cursor['sampled_windows']
 third=read(windows[third_update_index]['window_ref'])
 def is_successor(successor):
  if successor['start']!=third['end']:return False
  def membership(window):
   ref=window['dataset_ref']
   return read(ref).get('membership_dataset_ref',ref)
  # A phase may refresh upstream context while preserving the frozen draw order.
  return membership(third)==membership(successor)
 if len(windows)>third_update_index+1:
  entry=windows[third_update_index+1]
  ref=entry['window_ref']
  successor=read(ref)
  return dict(window_ref=ref,start=successor['start'],end=successor['end'],status='consumed') if (
   is_successor(successor) and entry.get('result_ref')) else None
 # The controller records capture only after accepting the closed owner return.
 # A queued request or a pending stage reservation is insufficient evidence.
 if cursor['phase'] in ('finalize','update'):
  ref=cursor['sampled_pending_window_ref'];successor=read(ref)
  if is_successor(successor):
   capture=cursor['capture']
   captured,inventory,_=capture_receipts(root,capture)
   if captured.get('sampled_window_ref')!=ref:return None
   closed_ref=capture['generation_closed_ref']
   if inventory.get('closed') is not True or inventory.get('generation_closed_ref')!=closed_ref:return None
   return dict(window_ref=ref,start=successor['start'],end=successor['end'],
               status='rollout_closed',generation_closed_ref=closed_ref)
 return None


def report(config_path):
 from run_package import read,pin
 from dependencies import require
 config_ref=pin(config_path);cfg=read(config_ref);root=Path(cfg['output_root'])
 # Read the running controller's atomic snapshot once, so the bytes and hash
 # refer to the same cursor even when the next window closes during reporting.
 cursor_path=root/'CURSOR.json'
 raw=cursor_path.read_bytes();cursor=json.loads(raw)
 cursor_ref=dict(path=str(cursor_path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
 require(cursor['config_ref']==config_ref,'cursor belongs to another run configuration')
 prior=read(cfg['initial_resume_ref']) if 'initial_resume_ref' in cfg else None
 initial_step=prior['step'] if prior else 0
 if prior:
  require(prior['clock']['successful_optimizer_steps']==initial_step,'initial checkpoint Clock differs')
 formal=(cfg.get('execution_mode')=='formal' and cfg.get('formal_rl_enabled') is True
         and cursor.get('execution_mode')=='formal' and bool(cfg.get('sampled_training')))
 cohorts=cursor.get('consumed_cohorts',[])
 windows=cursor.get('sampled_windows',[])
 if formal:
  require(len(windows)==len(cohorts),'formal updates must belong to the frozen sample windows')
 rows=[]
 consumed_through=0
 if cfg.get('branch_initial_ref'):
  require(cursor['branch_initial_ref']==cfg['branch_initial_ref'],'reported branch differs from controller')
  consumed_through=0 if cfg.get('dataset_start_mode')=='new_epoch' else read(cfg['branch_initial_ref'])['next_window_start']
 successful_window_indices=[]
 for index,cohort in enumerate(cohorts):
  result=cohort['result']
  if formal:
   window=windows[index]
   require(window['optimizer_steps']==result['optimizer_steps'] and
           window['result_ref']==result.get('result_ref'),'sampled window does not identify the update result')
   interval=read(window['window_ref'])
   require(interval['start']==consumed_through and interval['end']>interval['start'],
           'sampled window consumption is not contiguous')
   consumed_through=interval['end']
  if not formal and result['optimizer_steps']==0:continue
  require(type(result['optimizer_steps']) is int and result['optimizer_steps'] in (0,1),
          'at most one optimizer update per closed cohort')
  returned=root/('step-%06d'%cohort['sequence'])/'update/returned'
  closed,member=closed_metadata(returned)
  require(result==closed,'cursor result differs from sealed update return')
  optimizer_steps=result['optimizer_steps']
  if result.get('result_ref'):result=member(result['result_ref'])
  if optimizer_steps==0:
   require(result['optimizer_step'] is False and result['status']=='no_gradient'
           and result['clock']['successful_optimizer_steps']==initial_step+len(rows),
           'zero-signal window must retain the committed optimizer step')
   continue
  payload=result.get('result',{})
  obs=payload.get('objective_observation')
  require(obs and obs['schema']=='owned-global-true-objective-v1','global objective receipt missing')
  require(obs['optimizer_step'] is True and payload['optimizer_step'] is True,'optimizer did not update')
  step=obs['successful_step_after']
  require(obs['successful_step_before']==step-1 and result['clock']['successful_optimizer_steps']==step,
          'reported update does not advance the Clock by one')
  checkpoint=member(result['checkpoint_ref'])
  require(checkpoint['step']==step,'checkpoint step differs')
  if formal:
   require(result.get('formal_rl_enabled') is True and checkpoint['world_size']==8
           and checkpoint['clock']==result['clock'],'formal checkpoint identity differs')
  require(all(math.isfinite(float(v)) for v in obs['components'].values()),'nonfinite objective')
  require(obs['global_episode_denominator']>0 and not obs['padding_included'],'invalid objective denominator')
  successful_window_indices.append(index)
  rows.append(dict(step=step,loss=obs['components']['total'],components=obs['components'],
    denominator=obs['global_episode_denominator'],reward=obs['sampling_reward_diagnostics'],
    gradient_norm=obs.get('gradient_norm'),action_tokens=obs.get('action_tokens'),groups=obs.get('groups'),
    snapshot_sha256=obs['snapshot_sha256'],checkpoint_ref=result['checkpoint_ref'],
    trajectory_ref=cohort['trajectory_ref']))
 require([r['step'] for r in rows]==list(range(initial_step+1,cursor['step']+1)),
         'reported losses do not cover every committed optimizer step after recovery')
 if formal and windows:
  require(consumed_through==cursor['sampled_consumed']<=cfg['sampled_training']['sample_count'],
          'sampled cursor differs from closed windows')
 first_three=len(rows)>=3
 next_window=continued_capture(root,cursor,successful_window_indices[2]) if formal and first_three else None
 return dict(schema='committed-training-objectives-v1',config_ref=config_ref,
   cursor_ref=cursor_ref,phase=cursor['phase'],successful_updates=cursor['step'],
   initial_step=initial_step,run_updates=len(rows),formal_sampled_training=formal,
   sampled_consumed=cursor.get('sampled_consumed'),
   sampled_total=cfg['sampled_training']['sample_count'] if formal else None,
   observations=rows,first_three_complete=first_three,next_window=next_window,
   formal_startup_accepted=False,dev_gain_claim=False,
   status=('awaiting_controller_observation' if formal and first_three
           else 'three_updates_observed' if first_three else 'pending_real_optimizer_results'))


def controller_guard_owner(root, config_path):
 """Read the lock table so the observer never contends for the training guard."""
 try:
  guard=(Path(root)/'iteration.guard').stat()
 except FileNotFoundError:
  return None
 identity=(os.major(guard.st_dev),os.minor(guard.st_dev),guard.st_ino)
 for line in Path('/proc/locks').read_text().splitlines():
  fields=line.split()
  if len(fields)<8 or fields[1:4]!=['FLOCK','ADVISORY','WRITE']:
   continue
  major,minor,inode=fields[5].split(':')
  if (int(major,16),int(minor,16),int(inode))==identity:
   pid=int(fields[4])
   try:
    cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
   except (PermissionError,FileNotFoundError,ProcessLookupError):
    return None
   entry = Path(os.fsdecode(cmd[-3])).name if len(cmd) >= 4 else ''
   if len(cmd)>=4 and (entry=='iteration.py' or
      entry!='' and any(Path(os.fsdecode(arg)).name=='run_sampled_iteration.py' for arg in cmd[:-2])) and cmd[-2]==b'--config' and (
      cmd[-1]==os.fsencode(Path(config_path).resolve())):
    return pid
 return None


def observe(config_path):
 """Keep verified history visible even if the controller stops before handoff."""
 from run_package import read
 config=read(config_path)
 root=Path(config['output_root'])
 owner=controller_guard_owner(root,config_path)
 result=report(config_path)
 active=owner is not None and owner==controller_guard_owner(root,config_path)
 result['iteration_guard_held']=active
 if result['first_three_complete'] and result['formal_sampled_training']:
  complete=False
  if result['sampled_consumed']==result['sampled_total'] and (root/'SAMPLED-TRAIN-COMPLETE.json').is_file():
   seal=read(root/'SAMPLED-TRAIN-COMPLETE.json')
   complete=(seal['consumed']==result['sampled_total'] and seal['optimizer_steps']==result['successful_updates'])
  # These are the regular sampled controller phases after an optimizer update.
  # A fourth rollout can be reported when available, but is not a prerequisite.
  running=result['phase'] in ('endpoint','dev','phase_screen','screen','capture','finalize','update')
  accepted=complete or (active and running)
  status=('formal_sample_consumed' if complete else 'formal_training_continued' if accepted
          else 'formal_controller_not_running' if not active else 'formal_controller_not_continuing')
  result.update(formal_startup_accepted=accepted,status=status,controller_pid=owner if active else None)
 return result


def main():
 p=argparse.ArgumentParser()
 for name in ('release','config','output'):p.add_argument('--'+name,required=True)
 p.add_argument('--watch',action='store_true',help='Observe until verified handoff; never signal the trainer')
 a=p.parse_args();import sys,json
 sys.path.insert(0,str(Path(a.release)/'tree/parents/jev-system-rl/implementation'))
 from run_package import pin, read, write
 from storage import commit
 if not a.watch:
  result=observe(a.config);write(a.output,result);print(json.dumps(result));return
 config_ref=pin(a.config);config=read(config_ref)
 if (config.get('execution_mode')!='formal' or config.get('formal_rl_enabled') is not True
     or not config.get('sampled_training') or config.get('monitor_seconds')!=900):
  p.error('--watch requires the formal sampled TRAIN controller and its 900 second observation cadence')
 root=Path(config['output_root'])
 output=Path(a.output).resolve()
 if not output.parent.is_dir() or output.is_relative_to(root.resolve()):
  p.error('--output must be in an existing observer directory outside the training run')
 guard=output.with_name(output.name+'.observer.guard')
 with guard.open('a+b') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   if pin(a.config)!=config_ref:raise ValueError('observed training configuration changed')
   if (root/'CURSOR.json').is_file():
    result=observe(a.config)
   else:
    result=dict(config_ref=config_ref,status='awaiting_formal_cursor',formal_startup_accepted=False)
   if output.exists() and read(output).get('config_ref')!=config_ref:
    raise ValueError('observer output belongs to another run')
   commit(output,result,replace=output.exists())
   print(json.dumps(result),flush=True)
   if result['formal_startup_accepted']:
    return
   if result.get('phase') in ('done','diagnostic_rejected','phase_stopped'):
    raise SystemExit(2)
   time.sleep(config['monitor_seconds'])

if __name__=='__main__':main()
