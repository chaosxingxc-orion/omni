"""Exercise production scheduler on recorded request identities; transport has no network."""
import argparse,json,sys,threading
from pathlib import Path

def run(a):
    sys.path.insert(0,str(a.parents/'fresh-cohort-successor'))
    from cohort_io import load
    mod=load(a.capture,'capture_under_test')
    selected=json.loads((a.cohort/'SELECTED.json').read_bytes())
    history={}
    for d in selected['draws']:
        group=json.loads((a.cohort/'groups'/str(d['draw_index'])/'GENERATED.json').read_bytes())
        assert group['group_id']==d['group_id']
        history[d['group_id']]=[(e['calls'][0]['packet'],e['calls'][0]['reply']) for e in group['episodes']]
    spec=json.loads(a.policy.read_bytes())['rollout_dispatch'];width=spec['episode_concurrency']
    draws=selected['draws'];total=sum(len(items) for items in history.values())
    def exercise(mode,draws):
        submitted=[];saved={};known={};seen=0;peak=0;live=0
        release=threading.Event();gate=threading.Barrier(width,timeout=15);lock=threading.Lock()
        # The first request waits for a later request outside the first wave to start.
        class Transport:
            def __init__(self,i,reply):self.i=i;self.reply=reply
            def inspect(self,p):return {'state':'observed' if self.i in known else 'absent'}
            def reconcile(self,rid,packet):
                return dict(status='unknown' if known[self.i]=='unknown' else 'known',result=self.reply)
            def submit(self,rid,p):
                nonlocal live,peak
                with lock:
                    ordinal=len(submitted);submitted.append(rid);live+=1;peak=max(peak,live)
                try:
                    if mode=='normal':
                        if ordinal<width:gate.wait()
                        if ordinal==0:assert release.wait(15),'waited for slow request before refill'
                        if ordinal==width:release.set()
                    return self.reply
                finally:
                    with lock:live-=1
        def prepare(draw):
            nonlocal seen
            gid=draw['group_id']
            for k,(p,reply) in enumerate(history[gid]):
                i=seen;seen+=1
                if mode=='resume' and i%5==0:known[i]='known'
                if mode=='unknown' and i==total-1:known[i]='unknown'
                def finish(r,p=p):
                    assert r['request_id']==p['request_id'] and r['wire_sha256']==p['wire_sha256']
                    return r['request_id']
                def store(v,k=k):
                    assert (gid,k) not in saved
                    saved[gid,k]=v
                yield (Transport(i,reply),p,finish),store
        try:mod.dispatch_draws(draws,prepare,width,spec['dispatch_unit'])
        except ValueError as e:
            if mode!='unknown' or 'UNKNOWN' not in str(e) or submitted:raise
            return dict(unknown_blocked_before_contact=True)
        assert mode!='unknown' and len(saved)==seen and len(set(submitted))==len(submitted)==seen-len(known)
        assert all(saved[d['group_id'],k]==p['request_id'] for d in draws for k,(p,_) in enumerate(history[d['group_id']]))
        if mode=='normal':assert peak==width
        return dict(recordings=len(draws),requests=seen,submitted=len(submitted),reconciled=len(known),peak=peak)
    print(json.dumps(dict(scope='original dispatch function with offline transport',normal=exercise('normal',draws),
        resume=exercise('resume',draws[-3:]),unknown=exercise('unknown',draws),real_model_calls=0)))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('parents','capture','cohort','policy'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
