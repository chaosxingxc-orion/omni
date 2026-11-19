"""Report committed optimizer objectives from the original closed result graph."""
import argparse,math
from pathlib import Path

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


def report(config_path):
 from run_package import read,pin
 from dependencies import require
 cfg=read(config_path);root=Path(cfg['output_root']);cursor_ref=pin(root/'CURSOR.json');cursor=read(cursor_ref)
 assert cursor['config_ref']==pin(config_path)
 rows=[]
 for cohort in cursor.get('consumed_cohorts',[]):
  result=cohort['result']
  if result['optimizer_steps']==0:continue
  assert result['optimizer_steps']==1
  returned=root/('step-%06d'%cohort['sequence'])/'update/returned'
  closed,member=closed_metadata(returned)
  require(result==closed,'cursor result differs from sealed update return')
  if result.get('result_ref'):result=member(result['result_ref'])
  payload=result.get('result',{})
  obs=payload.get('objective_observation')
  assert obs and obs['schema']=='owned-global-true-objective-v1'
  assert obs['optimizer_step'] and payload['optimizer_step']
  step=obs['successful_step_after'];assert obs['successful_step_before']==step-1
  assert result['clock']['successful_optimizer_steps']==step
  require(member(result['checkpoint_ref'])['step']==step,'checkpoint step differs')
  assert all(math.isfinite(float(v)) for v in obs['components'].values())
  assert obs['global_episode_denominator']>0 and not obs['padding_included']
  rows.append(dict(step=step,loss=obs['components']['total'],components=obs['components'],
    denominator=obs['global_episode_denominator'],reward=obs['sampling_reward_diagnostics'],
    snapshot_sha256=obs['snapshot_sha256'],checkpoint_ref=result['checkpoint_ref'],
    trajectory_ref=cohort['trajectory_ref']))
 require([r['step'] for r in rows]==list(range(1,cursor['step']+1)),
         'reported losses do not cover every committed optimizer step')
 return dict(schema='committed-training-objectives-v1',config_ref=pin(config_path),
   cursor_ref=cursor_ref,phase=cursor['phase'],successful_updates=cursor['step'],
   observations=rows,first_three_complete=len(rows)>=3,dev_gain_claim=False,
   status='three_updates_observed' if len(rows)>=3 else 'pending_real_optimizer_results')

def main():
 p=argparse.ArgumentParser()
 for name in ('release','config','output'):p.add_argument('--'+name,required=True)
 a=p.parse_args();import sys,json
 sys.path.insert(0,str(Path(a.release)/'tree/parents/jev-system-rl/implementation'))
 from run_package import write
 result=report(a.config);write(a.output,result);print(json.dumps(result))

if __name__=='__main__':main()
