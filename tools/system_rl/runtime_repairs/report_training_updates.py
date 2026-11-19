"""Report committed optimizer objectives from the original closed result graph."""
import argparse,math
from pathlib import Path

def report(config_path):
 from run_package import read,pin
 from transport import restore_locations
 cfg=read(config_path);root=Path(cfg['output_root']);cursor=read(root/'CURSOR.json')
 assert cursor['config_ref']==pin(config_path)
 rows=[]
 for cohort in cursor.get('consumed_cohorts',[]):
  result=cohort['result']
  if result['optimizer_steps']==0:continue
  assert result['optimizer_steps']==1
  returned=root/('step-%06d'%cohort['sequence'])/'update/returned'
  if (returned/'CLOSED-COMMIT.json').exists():restore_locations(returned)
  # Remote checkpoints and runtime refs are resolved by the closed manifest.
  if result.get('result_ref'):result=read(result['result_ref'])
  payload=result.get('result',{})
  obs=payload.get('objective_observation')
  assert obs and obs['schema']=='owned-global-true-objective-v1'
  assert obs['optimizer_step'] and payload['optimizer_step']
  step=obs['successful_step_after'];assert obs['successful_step_before']==step-1
  assert result['clock']['successful_optimizer_steps']==step
  assert read(result['checkpoint_ref'])['step']==step
  assert all(math.isfinite(float(v)) for v in obs['components'].values())
  assert obs['global_episode_denominator']>0 and not obs['padding_included']
  rows.append(dict(step=step,loss=obs['components']['total'],components=obs['components'],
    denominator=obs['global_episode_denominator'],reward=obs['sampling_reward_diagnostics'],
    snapshot_sha256=obs['snapshot_sha256'],checkpoint_ref=result['checkpoint_ref'],
    trajectory_ref=cohort['trajectory_ref']))
 assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
 return dict(schema='committed-training-objectives-v1',config_ref=pin(config_path),
   cursor_ref=pin(root/'CURSOR.json'),phase=cursor['phase'],successful_updates=cursor['step'],
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
