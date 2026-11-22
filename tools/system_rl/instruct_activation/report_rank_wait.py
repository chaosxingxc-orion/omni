"""Report per-worker streamed and host durations for closed prefill steps only."""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from io_utils import doc, pin, require
from report_batch_stage import report
from report_prefill_timing import TILED, WHOLE


def collect(config, state, abi, bindings, lane, tile):
    rank_pids={}
    for line in bindings.read_text().splitlines():
        hit=re.search(r'Worker_TP(\d+) pid=(\d+).*?\[cpu_bind_mode\].*?rank=(\d+)',line)
        if hit:
            rank,pid,declared=map(int,hit.groups())
            require(rank==declared and rank_pids.setdefault(rank,pid)==pid,'worker binding collision')
    require(set(rank_pids)==set(range(4)) and len(set(rank_pids.values()))==4,'lost worker rank')
    steps=report(config,state,lane)['steps']
    active=[s for s in steps if len(s['decode'])!=len(s['alloc'])]
    layers=doc(pin(abi))['timed_layers'];rows=[]
    for rank,pid in sorted(rank_pids.items()):
        events=[json.loads(x) for x in (state/f'stages-{pid}.jsonl').read_bytes().splitlines()]
        frames=[e for e in events if e['kind']=='worker_forward']
        intervals=defaultdict(list)
        for e in events:
            if e['kind']=='prefill_interval': intervals[e['step']].append(e)
        sums=defaultdict(lambda:[0.,0.])
        for s in active:
            frame,=[f for f in frames if s['start_ns']<=f['start']<=f['t']<=s['end_ns']]
            group=intervals[frame['start']]
            tiles=(s['scheduled_tokens']+tile-1)//tile
            require(Counter((e['phase'],e['layer']) for e in group)==Counter({
                (p,i):tiles if p in TILED else 1 for p in (*WHOLE,*TILED) for i in layers}),
                'incomplete layer timing')
            for e in group:
                v=sums[e['phase']]
                v[0]+=e['device_ms'];v[1]+=e['host_ms']
        for p,(dev,host) in sums.items():
            rows.append(dict(rank=rank,steps=len(active),layers=len(layers),phase=p,
                             device_ms_per_step=dev/len(active),host_ms_per_step=host/len(active)))
    return rows


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('config','state','abi','bindings'):ap.add_argument(n,type=Path)
    ap.add_argument('lane');ap.add_argument('--tile',required=True,type=int)
    rows=collect(**vars(ap.parse_args()))
    w=csv.DictWriter(sys.stdout,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
