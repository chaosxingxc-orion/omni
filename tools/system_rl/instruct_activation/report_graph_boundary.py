"""Join device events outside graph replays to closed scheduler steps."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys

from io_utils import doc, pin, require
from report_batch_stage import report


PREFILL = ('before_attention','attention_kv','attention_gather','post_attention',
           'expert_cat','expert_reduce','weighted_sum')
DECODE = ('decode_before','decode_kv','decode_gather','decode_after','decode_reduce','decode_sum')


def summarize(config, state, abi, lane):
    ledger = report(config, state, lane)
    layers = doc(pin(abi))['timed_layers']
    values = defaultdict(list)
    for path in state.glob('stages-*.jsonl'):
        events = [json.loads(line) for line in path.read_bytes().splitlines()]
        frames = [e for e in events if e['kind']=='worker_forward']
        if not frames:
            continue
        timings = defaultdict(dict)
        for e in events:
            if e['kind']!='prefill_interval':
                continue
            key = e['layer'],e['phase']
            require(key not in timings[e['step']], 'duplicate segment')
            require(all(math.isfinite(e[k]) and e[k]>=0 for k in ('device_ms','host_ms')), 'invalid duration')
            timings[e['step']][key] = e
        for step in ledger['steps']:
            frame, = [f for f in frames if step['start_ns']<=f['start']<=f['t']<=step['end_ns']]
            values[step['step']].append(timings[frame['start']])
    out=[]
    for s in ledger['steps']:
        ranks=values[s['step']]
        require(len(ranks)==4, 'missing worker')
        pure=len(s['decode'])==len(s['alloc'])
        phases=DECODE if pure else PREFILL
        expected={(i,p) for i in layers for p in phases}
        require(all(set(x)==expected for x in ranks), f"incomplete step {s['step']}")
        for i,p in sorted(expected):
            device=[x[i,p]['device_ms'] for x in ranks]
            host=[x[i,p]['host_ms'] for x in ranks]
            out.append(dict(step=s['step'],kind='decode' if pure else 'mixed' if s['decode'] else 'prefill',
                            tokens=s['scheduled_tokens'],layer=i,phase=p,
                            device_median_ms=statistics.median(device),device_min_ms=min(device),
                            device_max_ms=max(device),host_median_ms=statistics.median(host)))
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','state','abi'):
        parser.add_argument(name,type=Path)
    parser.add_argument('lane')
    rows=summarize(**vars(parser.parse_args()))
    w=csv.DictWriter(sys.stdout,fieldnames=rows[0],lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
