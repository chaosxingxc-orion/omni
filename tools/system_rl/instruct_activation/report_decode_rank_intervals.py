"""Compare four TP workers at matching pure decode steps and layers."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from io_utils import doc, pin, require, save
from report_batch_stage import report


def summarize(config, state, abi, bindings, lane):
    ledger = report(config, state, lane)
    layers = doc(pin(abi))['timed_layers']
    steps = [s for s in ledger['steps'] if s['alloc'] and len(s['decode']) == len(s['alloc'])]
    ranks = {}
    for rank, pid, declared in re.findall(r'Worker_TP(\d+) pid=(\d+).*?\[cpu_bind_mode\].*?rank=(\d+)', bindings.read_text()):
        require(rank == declared and ranks.setdefault(rank, pid) == pid, 'worker identity differs')
    require(set(ranks) == set('0123') and len(set(ranks.values())) == 4, 'missing rank')
    phases = ['decode_before', 'decode_kv', 'decode_gather', 'decode_after', 'decode_reduce', 'decode_sum']
    samples = defaultdict(dict)
    for rank, pid in sorted(ranks.items()):
        events = [json.loads(line) for line in (state/f'stages-{pid}.jsonl').read_bytes().splitlines()]
        frames = [e for e in events if e['kind'] == 'worker_forward']
        intervals = defaultdict(dict)
        for e in events:
            if e['kind'] == 'prefill_interval' and e['phase'] in phases:
                key = (e['layer'], e['phase'])
                require(key not in intervals[e['step']], 'duplicate interval')
                intervals[e['step']][key] = e
        for step in steps:
            frame, = [f for f in frames if step['start_ns'] <= f['start'] <= f['t'] <= step['end_ns']]
            rows = intervals[frame['start']]
            require(set(rows) == {(i, p) for i in layers for p in phases}, 'missing intervals')
            for key, row in rows.items():
                samples[(step['step'], *key)][rank] = row
    stats = {}
    for phase in phases:
        observations = [r for (_, _, p), r in samples.items() if p == phase]
        def times(rows):
            return [e['device_ms'] for e in rows.values()]
        stats[phase] = dict(
            median_rank_min_ms=median(min(times(r)) for r in observations),
            median_rank_spread_ms=median(max(times(r))-min(times(r)) for r in observations),
            rank_mean_device_ms={k:mean(r[k]['device_ms'] for r in observations) for k in sorted(ranks)},
            rank_median_host_ms={k:median(r[k]['host_ms'] for r in observations) for k in sorted(ranks)})
    pairs = []
    for first, second in [('decode_kv','decode_gather'), ('decode_after','decode_reduce')]:
        differences = []
        for (s, i, p), earlier in samples.items():
            if p != first:
                continue
            later = samples[s, i, second]
            fastest, slowest = sorted(ranks, key=lambda k: earlier[k]['device_ms'])[::3]
            differences.append(later[fastest]['device_ms'] - later[slowest]['device_ms'])
        pairs.append(dict(earlier=first, later=second, count=len(differences),
                          fastest_earlier_had_longer_later=sum(d>0 for d in differences),
                          median_later_difference_ms=median(differences)))
    return dict(round=ledger['round_sha256'], steps=len(steps), sampled_layers=layers,
                layer_steps=len(layers)*len(steps), phases=stats, pairs=pairs,
                scope='Current-stream intervals include compute, dispatch and waiting. No shared device clock assumed.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('config','state','abi','bindings','output'):
        ap.add_argument(name,type=Path)
    ap.add_argument('lane')
    args = ap.parse_args()
    save(args.output,summarize(args.config,args.state,args.abi,args.bindings,args.lane))
