"""CPU fixture: exercise committed loss reporting via the real return wire format."""
import argparse, copy, io, json
from pathlib import Path
from checkpoint_inventory import unpack
from pull_closed import emit
from run_package import pin, write
from report_training_updates import report, closed_metadata

def run(root):
    root.mkdir(parents=True, exist_ok=False)
    cfg=write(root/'CONFIG.json', {'output_root':str(root)})
    cursor={'config_ref':cfg,'step':3,'phase':'fixture','consumed_cohorts':[]}
    for step in (1,2,3):
        command=root/'source'/str(step)
        ck=write(command/'COMMITTED.json',{'step':step})
        obs=dict(schema='owned-global-true-objective-v1',optimizer_step=True,
                 successful_step_before=step-1,successful_step_after=step,
                 components={'total':step/8},global_episode_denominator=8,
                 padding_included=False,sampling_reward_diagnostics={'fixture':True},snapshot_sha256='fixture')
        result=dict(optimizer_steps=1,clock={'successful_optimizer_steps':step},checkpoint_ref=ck,
                    result={'optimizer_step':True,'objective_observation':obs})
        ref=write(command/'RETURN.json',result)
        write(command/'CLOSED.json',dict(immutable=True,source_boot='fixture',return_ref=ref))
        stream=io.BytesIO();emit(command,stream,1024**2,expected_boot='fixture')
        returned=root/f'step-{step:06d}'/'update/returned';returned.parent.mkdir(parents=True)
        unpack(stream.getvalue(),returned,root/'cas',expected_boot='fixture')
        cursor['consumed_cohorts'].append(dict(sequence=step,trajectory_ref='fixture',result=result))
    path=root/'CURSOR.json'
    def run_report(value):
        path.write_text(json.dumps(value))
        return report(cfg['path'])
    got=run_report(cursor);assert got['first_three_complete'] and [r['loss'] for r in got['observations']]==[.125,.25,.375]
    checks=['three sealed returns projected in order']
    for key in ('omit','replace','duplicate'):
        bad=copy.deepcopy(cursor);cs=bad['consumed_cohorts']
        if key=='omit':cs.pop()
        if key=='replace':cs[1]['result']['result']['objective_observation']['components']['total']=-1
        if key=='duplicate':cs[1]=cs[0]
        try:run_report(bad)
        except ValueError:checks.append('reject '+key)
        else:raise AssertionError(key)
    path.write_text(json.dumps(cursor))
    ckpath=returned/('files/'+ck['path'].lstrip('/'))
    ckpath.write_text('{"step":999}')
    try:run_report(cursor)
    except ValueError:checks.append('reject modified checkpoint receipt')
    else:raise AssertionError('changed receipt')
    return checks

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    checks=run(a.output)
    write(a.output/'RESULT.json',dict(checks=checks,passed=True,fixture=True,actual_optimizer_steps=0))
    print('PASS',len(checks),'fixture checks; no training executed')
