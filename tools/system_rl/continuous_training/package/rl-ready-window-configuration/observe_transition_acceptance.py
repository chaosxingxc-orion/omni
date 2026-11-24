"""Report closed, policy-bound production windows following a controller handoff."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from report_policy_dispatch_gap import measure


def read(p): return json.loads(Path(p).read_bytes())


def report(spec,first,number,limit):
    run=Path(spec['iteration_ref']['path']).parent.parent/'run'
    entries=[]
    for cohort in sorted((run/'rollout').glob('*-capture/cohort')):
        if not (cohort.parent/'DRAINED.json').is_file():continue
        cfg=read(cohort/'INPUT.json');policy=read(cfg['policy']['current_ref']['path'])
        if policy['step']<first or len(entries)>=number:continue
        for k in ('static_supply_ref','prompt_length_index_ref'):
            if cfg['capture'].get(k)!=spec[k]:raise ValueError('rollout did not use configured '+k)
        closed=read(cohort.parent/'DRAINED.json')
        if not closed['terminal_verified'] or closed['unknown_requests']!=0:raise ValueError('unreconciled window')
        gap=measure(run,policy['step'],cohort)
        def journal(suffix):
            files=list(cohort.glob('groups/*/episode-*/native-journal/*.'+suffix+'.json'))
            rows={}
            for p in files:
                row=read(p);key=row['request_id']
                if key in rows:raise ValueError('duplicate request ID')
                rows[key]=(row['intent_sha256'],row['request_sha256'],row['identity'])
            return rows,[p.stat().st_mtime_ns for p in files]
        sent,d=journal('dispatch');received,r=journal('response')
        plan=read(cohort/'SELECTED.json')
        generated=[read(p) for p in cohort.glob('groups/*/GENERATED.json')]
        if (not d or sent!=received or len(d)!=sum(len(g['episodes']) for g in generated)
                or sorted(g['group_id'] for g in generated)!=sorted(plan['group_ids'])):
            raise ValueError('incomplete or mismatched request roster')
        entries.append(dict(step=policy['step'],recordings=len(plan['draws']),requests=len(d),
            gap_seconds=gap['elapsed_s'],gap_breakdown=gap['seconds'],generation_wall_seconds=(max(r)-min(d))/1e9,
            first_dispatch_ns=min(d),last_response_ns=max(r),unknown=0))
    for prev,next_ in zip(entries,entries[1:]):
        if next_['step']!=prev['step']+1:raise ValueError('non-consecutive optimizer steps')
        prev['dispatch_to_dispatch_s']=(next_['first_dispatch_ns']-prev['first_dispatch_ns'])/1e9
    cur=read(run/'CURSOR.json')
    return dict(target_seconds=limit,window_goal=number,windows=entries,within_target=bool(len(entries)==number and
        all(e['gap_seconds']<=limit for e in entries)),step=cur['step'],phase=cur['phase'],consumed=cur['sampled_consumed'])


def save(p,value):
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--first',required=True,type=int);p.add_argument('--windows',required=True,type=int)
    p.add_argument('--target',required=True,type=float);p.add_argument('--interval',type=float)
    a=p.parse_args()
    if a.windows<1 or (a.interval is not None and a.interval<=0):raise ValueError('invalid window settings')
    while True:
        value=report(read(a.config),a.first,a.windows,a.target);save(a.output,value)
        if not a.interval or len(value['windows'])==a.windows:break
        time.sleep(a.interval)
