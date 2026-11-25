"""Break measured dispatch-to-dispatch time into additive controller intervals.

The actor reads its own receipt mtimes. Only durations (never remote timestamps)
are compared with the controller's acknowledged-command envelope.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from report_step_latency import report


def durations(points):
    result = {}
    for (left, start), (right, stop) in zip(points, points[1:]):
        if stop < start:
            raise ValueError(f'out-of-order receipts: {left}, {right}')
        result[right] = round((stop - start) / 1e9, 3)
    return result


def dispatch_preparation(cohort, policy_ready_ns, first_dispatch_ns):
    """Control-host evidence; early runs did not produce all these markers."""
    supplies = list(cohort.glob('groups/*/episode-*/SUPPLY.json'))
    dispatches = list(cohort.glob('groups/*/episode-*/native-journal/*.dispatch.json'))
    if not supplies or {p.parent for p in supplies} != {p.parents[1] for p in dispatches}:
        return None
    times = [p.stat().st_mtime_ns for p in supplies]
    if min(times) > first_dispatch_ns:
        return None
    lanes = {}
    for supply in supplies:
        lane = json.loads((supply.parent/'LANE.json').read_bytes())['lane']
        lanes.setdefault(lane, []).append(supply)
    lane_rows = {}
    for lane, rows in lanes.items():
        first = min(next(p.parent.glob('native-journal/*.dispatch.json')).stat().st_mtime_ns for p in rows)
        last_supply = max(p.stat().st_mtime_ns for p in rows)
        lane_rows[lane] = dict(requests=len(rows), all_supplied_before_first_dispatch=last_supply <= first,
                               supply_span_s=(last_supply-min(p.stat().st_mtime_ns for p in rows))/1e9,
                               supplied_at_first_dispatch=sum(p.stat().st_mtime_ns<=first for p in rows),
                               last_supply_to_first_dispatch_s=(first-last_supply)/1e9)
    return dict(seconds=durations([('policy_ready', policy_ready_ns),
                ('first_supply', min(times)), ('first_supply_to_first_dispatch', first_dispatch_ns)]),
                lanes=lane_rows)


def actor_stages(update, step):
    intent = json.loads((update/'SUBMIT-INTENT.json').read_bytes())
    request = json.loads((update/'REQUEST.json').read_bytes())['run']
    command = json.loads((update/'COMMAND.json').read_bytes())
    owner = Path(intent['target'])
    root = owner/'hot/resident/results'/format(intent['sequence'], '06d')
    probe = """import json,sys
from pathlib import Path
root=Path(sys.argv[1]); step=int(sys.argv[2]); boot,run=sys.argv[3:5]
closed=json.loads((root/'CLOSED.json').read_bytes())
if not (Path('/proc/sys/kernel/random/boot_id').read_text().strip()==closed['source_boot']==boot
        and closed['immutable'] and closed['sequence']==int(root.name) and closed['run_id']==run
        and closed['return_ref']['sha256']==sys.argv[5]):raise ValueError('different resident result')
ref,=(root/'finalized-cohort').glob('REFERENCE-BATCH-*.json')
names=['COMMAND-INTENT.json',str(ref.relative_to(root)),'rank-0/UPDATE-INTENT.json',
       'rank-0/LOSS.json',f'step-{step}/COMMITTED.json','CLOSED.json']
print(json.dumps([(root/p).stat().st_mtime_ns for p in names]))
"""
    manifest = json.loads((update/'returned/CLOSED-MANIFEST.json').read_bytes())
    args = ['python3','-B','-c',probe,str(root),str(step),request['source_boot'],
            command['run_id'],manifest['RETURN.json']['sha256']]
    result = subprocess.run(['ssh','-oBatchMode=yes','-oConnectTimeout=15',request['host'],
                             shlex.join(args)], capture_output=True, text=True, check=True, timeout=30)
    names = ['command','reference_ready','assemble_training_input','update_to_loss',
             'commit_checkpoint','publish_actor_result']
    return durations(list(zip(names,json.loads(result.stdout),strict=True)))


def breakdown(run, config, step):
    windows = report(run,config,step-1)
    window = next((w for w in windows['steps'] if w['step']==step), None)
    if window is None:
        raise ValueError(f'step {step}: measured window and its closed successor are required')
    start, end = window['first_dispatch_ns'], window['successor_first_dispatch_ns']
    cohorts = sorted((run/'rollout').glob('*-capture/cohort'))
    cohort, = [c for c in cohorts if min((p.stat().st_mtime_ns for p in c.rglob('*.dispatch.json')),
                                          default=None)==start]
    update = run/f'step-{step:06d}'/'update'
    policy = run/f'policy-{step:06d}'
    responses = list(cohort.rglob('*.response.json'))
    if len(responses)!=window['episodes']:
        raise ValueError('response count differs')
    stages = durations([
        ('first_dispatch', start), ('receive_all', max(p.stat().st_mtime_ns for p in responses)),
        ('close_rollout',(cohort/'GENERATION-CLOSED.json').stat().st_mtime_ns),
        ('deliver_training_input',(update/'COMMAND-ACK.json').stat().st_mtime_ns),
        ('actor_and_return',(update/'returned/CLOSED-COMMIT.json').stat().st_mtime_ns),
        ('publish_and_switch_policy',(policy/'QUALIFIED-INPUT.json').stat().st_mtime_ns),
        ('prepare_next_dispatch',end)])
    actor = actor_stages(update,step)
    residual = round(stages['actor_and_return']-sum(actor.values()),3)
    focus = dict(before_update=round(actor['reference_ready']+actor['assemble_training_input'],3),
                 update_to_loss=actor['update_to_loss'],prepare_next_dispatch=stages['prepare_next_dispatch'])
    other = {k:stages[k] for k in ('receive_all','close_rollout','deliver_training_input')}
    other.update(commit_checkpoint=actor['commit_checkpoint'],publish_actor_result=actor['publish_actor_result'],
                 envelope_residual=residual,publish_and_switch_policy=stages['publish_and_switch_policy'])
    accounted = sum(focus.values())+sum(other.values())
    if abs(accounted-window['seconds'])>0.01:
        raise ValueError('time ledger does not reconcile to dispatch-to-dispatch interval')
    successor, = [c for c in cohorts if min((p.stat().st_mtime_ns for p in c.rglob('*.dispatch.json')),
                                             default=None)==end]
    return dict(step=step,seconds=window['seconds'],episodes=window['episodes'],
                next_dispatch_detail=dispatch_preparation(successor,(policy/'QUALIFIED-INPUT.json').stat().st_mtime_ns,end),
                checkpoint_sha256=window['checkpoint_sha256'],controller_seconds=stages,actor_seconds=actor,
                envelope_residual_seconds=residual,focus_seconds=focus,other_seconds=other,
                focus_total_seconds=round(sum(focus.values()),3),other_total_seconds=round(sum(other.values()),3),
                seconds_above_limit=round(max(0,window['seconds']-windows['maximum_seconds']),3),
                source=dict(run=str(run),cohort=str(cohort),command_ack=str(update/'COMMAND-ACK.json')))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--run',type=Path,required=True)
    cli.add_argument('--config',type=Path,required=True)
    cli.add_argument('--step',type=int,action='append',required=True)
    args=cli.parse_args()
    print(json.dumps([breakdown(args.run,args.config,step) for step in args.step],indent=2))
