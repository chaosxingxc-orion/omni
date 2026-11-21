"""CPU fixture: exercise committed loss reporting via the real return wire format."""
import argparse, copy, fcntl, io, json, subprocess, sys
from pathlib import Path
from checkpoint_inventory import unpack
from pull_closed import emit
from run_package import pin, write
from report_training_updates import observe, report, closed_metadata

def run(root):
    root.mkdir(parents=True, exist_ok=False)
    initial=write(root/'PREDECESSOR.json',dict(step=3,clock={'successful_optimizer_steps':3}))
    cfg=write(root/'CONFIG.json', {'output_root':str(root),'initial_resume_ref':initial,
        'execution_mode':'formal','formal_rl_enabled':True,'sampled_training':{'sample_count':10}})
    dataset=write(root/'DATASET.json',dict(count=10))
    cursor=dict(config_ref=cfg,step=6,phase='capture',execution_mode='formal',consumed_cohorts=[],
                sampled_windows=[],sampled_consumed=6)
    for index,step in enumerate((4,5,6)):
        command=root/'source'/str(step)
        clock={'successful_optimizer_steps':step}
        ck=write(command/'COMMITTED.json',dict(step=step,world_size=8,clock=clock))
        obs=dict(schema='owned-global-true-objective-v1',optimizer_step=True,
                 successful_step_before=step-1,successful_step_after=step,
                 components={'total':step/8},global_episode_denominator=8,
                 padding_included=False,sampling_reward_diagnostics={'fixture':True},snapshot_sha256='fixture')
        result=dict(formal_rl_enabled=True,optimizer_steps=1,clock=clock,checkpoint_ref=ck,
                    result={'optimizer_step':True,'objective_observation':obs})
        ref=write(command/'RETURN.json',result)
        write(command/'CLOSED.json',dict(immutable=True,source_boot='fixture',return_ref=ref))
        stream=io.BytesIO();emit(command,stream,1024**2,expected_boot='fixture')
        returned=root/f'step-{step:06d}'/'update/returned';returned.parent.mkdir(parents=True)
        unpack(stream.getvalue(),returned,root/'cas',expected_boot='fixture')
        cursor['consumed_cohorts'].append(dict(sequence=step,trajectory_ref='fixture',result=result))
        window=write(root/f'WINDOW-{index}.json',dict(dataset_ref=dataset,start=2*index,end=2*index+2))
        cursor['sampled_windows'].append(dict(window_ref=window,result_ref=None,optimizer_steps=1))
    path=root/'CURSOR.json'
    def run_report(value):
        path.write_text(json.dumps(value))
        return report(cfg['path'])
    got=run_report(cursor)
    assert got['first_three_complete'] and [r['loss'] for r in got['observations']]==[.5,.625,.75]
    assert got['initial_step']==3 and got['run_updates']==3
    assert not got['formal_startup_accepted'] and got['status']=='awaiting_controller_observation'
    checks=['exclude historical updates and verify committed losses independently of the controller']
    window=write(root/'WINDOW-3.json',dict(dataset_ref=dataset,start=6,end=8))
    pending=dict(cursor,sampled_pending_window_ref=window,pending_stage={'stage':'capture'})
    assert not run_report(pending)['formal_startup_accepted']
    assert not run_report(dict(pending,phase='done'))['formal_startup_accepted']
    closed=write(root/'GENERATION-CLOSED.json',dict(fixture=True))
    capture=dict(generation_closed_ref=closed,
        generation_inventory_ref=write(root/'GENERATION-INVENTORY.json',
                                       dict(closed=True,generation_closed_ref=closed)),
        input_ref=write(root/'CAPTURED-INPUT.json',dict(sampled_window_ref=window)))
    got=run_report(dict(pending,phase='finalize',capture=capture))
    assert got['next_window']['generation_closed_ref']==closed
    stopped=observe(cfg['path'])
    assert stopped['first_three_complete'] and not stopped['formal_startup_accepted']
    assert stopped['status']=='formal_controller_not_running'
    with (root/'iteration.guard').open('ab') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not observe(cfg['path'])['formal_startup_accepted']
    helper=root/'iteration.py'
    helper.write_text("import sys,fcntl,pathlib\n"
        "guard=(pathlib.Path(sys.argv[-1]).parent/'iteration.guard').open('ab')\n"
        "fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        "print('ready',flush=True)\nsys.stdin.read()\n")
    with subprocess.Popen([sys.executable,str(helper),'--config',cfg['path']],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) as driver:
        try:
            assert driver.stdout.readline().strip()=='ready'
            got=run_report(dict(cursor,phase='endpoint'))
            live=observe(cfg['path'])
            assert got['next_window'] is None and live['formal_startup_accepted'] and live['controller_pid']==driver.pid
            for phase in ('phase_stopped','done','diagnostic_rejected','freeze'):
                run_report(dict(cursor,phase=phase))
                assert not observe(cfg['path'])['formal_startup_accepted']
        finally:
            driver.stdin.close()
    assert not observe(cfg['path'])['formal_startup_accepted']
    checks.append('hand off at the third commit with the live controller; reject unrelated lock holders and stopped phases')
    stale=dict(capture,input_ref=write(root/'STALE-INPUT.json',
                                      dict(sampled_window_ref=cursor['sampled_windows'][2]['window_ref'])))
    assert run_report(dict(pending,phase='finalize',capture=stale))['next_window'] is None
    checks.append('closed capture must belong to the immediate successor window')
    context=write(root/'CONTEXT.json',dict(membership_dataset_ref=dataset))
    refreshed=write(root/'WINDOW-REFRESHED.json',dict(dataset_ref=context,start=6,end=8))
    capture['input_ref']=write(root/'REFRESHED-INPUT.json',dict(sampled_window_ref=refreshed))
    got=run_report(dict(pending,phase='finalize',capture=capture,sampled_pending_window_ref=refreshed))
    assert got['next_window']['window_ref']==refreshed
    checks.append('upstream context refresh retains frozen draw membership')
    command=root/'source/7'
    zero=write(command/'NO-GRADIENT.json',dict(status='no_gradient',optimizer_step=False,
               clock={'successful_optimizer_steps':6}))
    returned=dict(optimizer_steps=0,result_ref=zero)
    ref=write(command/'RETURN.json',returned)
    write(command/'CLOSED.json',dict(immutable=True,source_boot='fixture',return_ref=ref))
    stream=io.BytesIO();emit(command,stream,1024**2,expected_boot='fixture')
    destination=root/'step-000007/update/returned';destination.parent.mkdir(parents=True)
    unpack(stream.getvalue(),destination,root/'cas',expected_boot='fixture')
    advanced=copy.deepcopy(cursor)
    advanced['sampled_consumed']=8
    advanced['sampled_windows'].append(dict(window_ref=window,result_ref=zero,optimizer_steps=0))
    advanced['consumed_cohorts'].append(dict(sequence=7,result=returned))
    got=run_report(advanced)
    assert got['next_window']['status']=='consumed' and got['run_updates']==3 and got['sampled_consumed']==8
    assert not run_report(dict(advanced,phase='done'))['formal_startup_accepted']
    bad=copy.deepcopy(advanced);bad['sampled_consumed']-=1
    try:run_report(bad)
    except ValueError:pass
    else:raise AssertionError('inconsistent consumption')
    checks.append('already consumed zero-gradient successor proves continuation without a fourth loss')
    for key in ('omit','replace','duplicate'):
        bad=copy.deepcopy(cursor);cs=bad['consumed_cohorts']
        if key=='omit':cs.pop()
        if key=='replace':cs[1]['result']['result']['objective_observation']['components']['total']=-1
        if key=='duplicate':cs[1]=cs[0]
        try:run_report(bad)
        except ValueError:checks.append('reject '+key)
        else:raise AssertionError(key)
    path.write_text(json.dumps(cursor))
    ckpath=root/'step-000006/update/returned'/('files/'+ck['path'].lstrip('/'))
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
