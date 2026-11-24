"""Compute client-side concurrency from complete original journal intervals, per lane."""
import argparse,collections,hashlib,json
from pathlib import Path

def read(p):return json.loads(p.read_bytes())

def report(root):
    selected=read(root/'SELECTED.json'); closed=read(root/'GENERATION-CLOSED.json')
    if closed['selected_cohort_ref']['sha256']!=hashlib.sha256((root/'SELECTED.json').read_bytes()).hexdigest():raise ValueError('wrong closure')
    seen=set(); lanes=collections.defaultdict(list)
    for d in selected['draws']:
        directory=root/'groups'/str(d['draw_index'])
        group=read(directory/'GENERATED.json')
        if group['group_id']!=d['group_id']:raise ValueError('wrong sampled group')
        for ep in group['episodes']:
            call,=ep['calls'];sub=directory/f"episode-{ep['replicate']}"
            lane=read(sub/'LANE.json')['lane']
            files=sub/'native-journal';a,=files.glob('*.dispatch.json');b,=files.glob('*.response.json')
            send,end=read(a),read(b);rid=call['packet']['request_id']
            if (rid in seen or send['request_id']!=rid or end['request_id']!=rid or
                send['intent_sha256']!=end['intent_sha256'] or send['request_sha256']!=end['request_sha256']
                or a.stat().st_mtime_ns>=b.stat().st_mtime_ns):raise ValueError('wrong request interval')
            seen.add(rid); lanes[lane].append((a.stat().st_mtime_ns,b.stat().st_mtime_ns,d['draw_index']))
    out={}
    for lane,rows in lanes.items():
        events=sorted((time,change) for a,b,_ in rows for time,change in ((a,1),(b,-1)))
        first=last=events[0][0];active=0;buckets=collections.Counter()
        for stamp,change in events:
            buckets[active]+=stamp-last;active+=change;last=stamp
        if active or last<=first:raise ValueError('invalid journal ordering')
        windows={}
        for a,b,idx in rows:
            x,y=windows.get(idx,(a,b));windows[idx]=(min(a,x),max(b,y))
        groups=sorted(windows.values())
        out[lane]=dict(requests=len(rows),recordings=len(groups),wall_s=(last-first)/1e9,
            peak=max(k for k,v in buckets.items() if v),mean_inflight=sum(k*v for k,v in buckets.items())/(last-first),
            client_idle_s=buckets[0]/1e9,adjacent_groups_overlapped=sum(next_a<prev_b for (_,prev_b),(next_a,_) in zip(groups,groups[1:])),
            seconds_by_inflight={k:v/1e9 for k,v in sorted(buckets.items())})
    return dict(metric='journal dispatch to response, not device utilization',lanes=out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cohort',nargs='+',type=Path)
    print(json.dumps([report(r) for r in p.parse_args().cohort],indent=2))
