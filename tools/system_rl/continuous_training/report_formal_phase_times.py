"""Bound historical wall time without subtracting clocks from different machines."""
import argparse,json,shlex,subprocess
from pathlib import Path

HOST_OBSERVE="""
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);names=json.loads(sys.argv[2])
launch=json.loads((r/'EXIT.json').read_bytes())
if launch['exit_code']!=0 or launch['complete_ranks']!=8:raise ValueError('not a completed actor')
def stamps(name):
    paths=[r/f'rank-{rank}'/(name+'.json') for rank in range(8)]
    return max(p.stat().st_mtime_ns for p in paths)
print(json.dumps({n:stamps(n) for n in names}))
"""

def controller_intervals(window, cohort, directory, policy):
    """All endpoints are mtimes written by the same live controller host."""
    events = [
        ('first_dispatch', window['first_dispatch_ns']),
        ('last_response', max(p.stat().st_mtime_ns for p in cohort.glob(
            'groups/*/episode-*/native-journal/*.response.json'))),
        ('inventory_closed', (cohort/'GENERATION-INVENTORY.json').stat().st_mtime_ns),
        ('submit_intent', (directory/'SUBMIT-INTENT.json').stat().st_mtime_ns),
        ('optimizer_return', (directory/'returned/CLOSED-COMMIT.json').stat().st_mtime_ns),
        ('promoted_policy', policy.stat().st_mtime_ns),
        ('successor_dispatch', window['successor_first_dispatch_ns']),
    ]
    intervals = {}
    for (start, a), (end, b) in zip(events, events[1:]):
        if b < a:
            raise ValueError(f'controller receipt time reversed: {start} -> {end}')
        intervals[f'{start}_to_{end}'] = (b - a) / 1e9
    return intervals


def report(run,policy,after,*,controller_only=False):
    from report_step_latency import report as measure
    measured=measure(run,policy,after);rows=[]
    for w in measured['steps']:
        # Find the receipt by actual successful step; cohort_sequence may
        # differ from Clock after a terminal zero-gradient group.
        matches=[]
        for p in run.glob('step-*/update/returned/RETURN.json'):
            ret=json.loads(p.read_bytes())
            if not ret.get('checkpoint_ref'):continue
            locations=json.loads((p.parent/'LOCATION-MAP.json').read_bytes())
            ref=ret['checkpoint_ref']
            if json.loads(Path(locations[ref['path']]).read_bytes())['step']==w['step']:matches.append((p,ret,locations))
        if len(matches)!=1:raise ValueError('ambiguous successful actor')
        p,ret,locations=matches[0];directory=p.parent.parent
        actual=json.loads(Path(locations[ret['result_ref']['path']]).read_bytes())
        if actual['result']['objective_observation']['global_episode_denominator']!=w['episodes']:raise ValueError('wrong actor scale')
        inv=json.loads(Path(actual['generation_inventory_ref']['path']).read_bytes());cohort=Path(inv['cohort_root'])
        if controller_only:
            policy_root=run/f"policy-{w['step']:06d}"
            promoted,=[path for path in (policy_root/'promoted/POLICY.json',policy_root/'POLICY.json') if path.is_file()]
            exported=json.loads(Path(locations[ret['policy_ref']['path']]).read_bytes())
            promotion=json.loads(promoted.read_bytes())
            if (promotion['assets']!=exported['assets'] or promotion['step']!=exported['step']
                    or promotion['step']!=w['step']):
                raise ValueError('promoted policy differs from the measured optimizer export')
            intervals=controller_intervals(w,cohort,directory,promoted)
            rows.append(dict(**w,control_host_seconds=intervals))
            continue
        end=max(x.stat().st_mtime_ns for x in cohort.glob('groups/*/episode-*/native-journal/*.response.json'))
        proof=Path(ret['checkpoint_ref']['path'])
        if proof.parent.parent.name!='hot':raise ValueError('historical one-container actor only')
        target=proof.parent.parent
        req=json.loads((directory/'REQUEST.json').read_bytes())['run']
        names=('COLLECTIVE-READY','FINALIZE-RESULT','ACCEPTED-COHORT','UPDATE-INTENT','BACKWARDS','LOSS','RESULT')
        cmd=shlex.join(['python3','-B','-c',HOST_OBSERVE,str(target),json.dumps(names)])
        stamps=json.loads(subprocess.check_output(['ssh','-oBatchMode=yes','-oConnectTimeout=10',req['host'],cmd],timeout=30))
        def delta(a,b):
            if stamps[b]<stamps[a]:raise ValueError('actor receipt time reversed')
            return (stamps[b]-stamps[a])/1e9
        intervals=dict(rollout=(end-w['first_dispatch_ns'])/1e9,
            construct_restore_finalize=delta('COLLECTIVE-READY','FINALIZE-RESULT'),
            pack_accept=delta('FINALIZE-RESULT','UPDATE-INTENT'),forward_backward=delta('UPDATE-INTENT','BACKWARDS'),
            update_loss=delta('BACKWARDS','LOSS'),checkpoint_result=delta('LOSS','RESULT'))
        remaining=w['seconds']-sum(intervals.values())
        if remaining<0:raise ValueError('overlap or invalid timing sources')
        rows.append(dict(step=w['step'],seconds=w['seconds'],phases_seconds=intervals,
                other_including_policy_switch_seconds=remaining,actor_receipt_root=str(target),
                generation_inventory_ref=actual['generation_inventory_ref'],actor_checkpoint_ref=ret['checkpoint_ref']))
    scope='control_host_observed_intervals' if controller_only else 'historical_observed_intervals_not_candidate_speed'
    return dict(scope=scope,rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('run','policy','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--after',type=int,required=True)
    p.add_argument('--controller-only', action='store_true', help='use only control-host receipts, without SSH')
    a=p.parse_args()
    from run_package import write
    data=report(a.run,a.policy,a.after,controller_only=a.controller_only);print(json.dumps(write(a.output,data)))
