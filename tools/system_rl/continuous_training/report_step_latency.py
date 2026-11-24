"""Measure actual contiguous sampled windows from the original control-host journal.

This report verifies timings. It is not a release or a numerical acceptance.
"""
import argparse,hashlib,json
from pathlib import Path


def read(p):
    return json.loads(p.read_bytes())


def report(root,policy,after):
    limits=read(policy)['acceptance']; windows=[]
    anchor=None
    for c in sorted((root/'rollout').glob('*-capture')):
        cohort=c/'cohort'; closed=cohort/'GENERATION-CLOSED.json'
        if not closed.is_file():continue
        selected=read(cohort/'SELECTED.json')
        step=read(Path(selected['policy']['current_ref']['path']))['step']+1
        if step<=after:continue
        n,g=limits['recordings_per_step'],limits['episodes_per_recording']
        if len(selected['draws'])!=n or len(read(closed)['groups'])!=n:raise ValueError('incomplete cohort')
        records=set(); rosters=set();dispatch=[];responses=[]
        for draw in selected['draws']:
            group=cohort/'groups'/str(draw['draw_index'])
            generated=read(group/'GENERATED.json');rec=draw['record']['sample_id']
            if rec in records or len(generated['episodes'])!=g:raise ValueError('recording/G differs')
            records.add(rec)
            for ep in generated['episodes']:
                call,=ep['calls']
                req=call['packet']['request_id']
                if req in rosters:raise ValueError('duplicate sampled call')
                rosters.add(req)
                folder=group/f"episode-{ep['replicate']}"/'native-journal'
                d,=folder.glob('*.dispatch.json');r,=folder.glob('*.response.json')
                sent,returned=read(d),read(r)
                if (sent['request_id']!=req or returned['request_id']!=req or
                        sent['request_sha256']!=returned['request_sha256'] or sent['intent_sha256']!=returned['intent_sha256'] or
                        d.name.split('.')[0]!=r.name.split('.')[0] or r.stat().st_mtime_ns<d.stat().st_mtime_ns):
                    raise ValueError('journal response timing/identity mismatch')
                dispatch.append(d.stat().st_mtime_ns);responses.append(r.stat().st_mtime_ns)
        first=min(dispatch)
        if anchor:
            previous,start,last=anchor
            if step!=previous+1 or first<=last:raise ValueError('non-contiguous sampled updates')
            commit=root/f'step-{previous:06d}'/'update/returned/CLOSED-COMMIT.json'
            if not read(commit)['immutable'] or not start<commit.stat().st_mtime_ns<first:
                raise ValueError('missing intervening optimizer return')
            seconds=(first-start)/1e9
            windows.append(dict(step=previous,recordings=n,episodes=n*g,seconds=seconds,
                within_limit=seconds<=limits['maximum_step_seconds'],first_dispatch_ns=start,
                successor_first_dispatch_ns=first,commit_sha256=hashlib.sha256(commit.read_bytes()).hexdigest()))
        anchor=(step,first,max(responses))
    k=limits['consecutive_steps']
    return dict(scope='timing_only',timing=limits['timing'],steps=windows,maximum_seconds=limits['maximum_step_seconds'],
        latest_consecutive_timing_pass=len(windows)>=k and all(w['within_limit'] for w in windows[-k:]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--after',type=int,required=True)
    a=p.parse_args();print(json.dumps(report(a.run,a.config,a.after),indent=2))
