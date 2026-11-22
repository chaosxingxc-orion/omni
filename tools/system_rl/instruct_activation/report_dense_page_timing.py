"""Reconcile KV, packing and FIA intervals with each rank of a closed RL batch."""
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

PHASES = ('metadata_and_plan', 'query_pack', 'kv_pack', 'dense_fia', 'output_copy')


def summarize(cfg, state, abi, lane):
    steps = report(cfg, state, lane)['steps']
    layers = doc(pin(abi))['timed_layers']
    by_step = defaultdict(list)
    for path in state.glob('stages-*.jsonl'):
        with path.open() as f:
            events = [json.loads(line) for line in f]
        forwards = [e for e in events if e['kind'] == 'worker_forward']
        if not forwards:
            continue
        intervals = defaultdict(dict)
        for e in events:
            if e['kind'] != 'prefill_interval':
                continue
            key = (e['phase'], e['layer'])
            require(e['phase'] in PHASES and e['layer'] in layers
                    and key not in intervals[e['step']], 'duplicate or unexpected interval')
            require(all(math.isfinite(e[k]) and e[k] >= 0 for k in ('device_ms', 'host_ms')),
                    'invalid timing')
            intervals[e['step']][key] = e
        for s in steps:
            hits = [f for f in forwards if s['start_ns'] <= f['start'] <= f['t'] <= s['end_ns']]
            require(len(hits) == 1, 'missing or ambiguous worker forward')
            by_step[s['step']].append(intervals[hits[0]['start']])
    rows = []
    for s in steps:
        ranks = by_step[s['step']]
        require(len(ranks) == 4, 'expected four timing ranks')
        if not any(ranks):
            # The extension only handles ChunkedPrefill and PrefillCacheHit.
            rows.append(dict(step=s['step'], kind='decode' if len(s['decode']) == len(s['alloc']) else 'unobserved_prefill',
                             tokens=s['scheduled_tokens'], phase='outside_hook', layer=None,
                             tp_median_device_ms=None, tp_median_host_ms=None))
            continue
        require(len(s['decode']) != len(s['alloc']), 'unexpected pure decode in prefill hook')
        expected = {(p,i) for p in PHASES for i in layers}
        require(all(set(rank) == expected for rank in ranks), 'incomplete measured prefill step')
        for p,i in sorted(expected):
            rows.append(dict(step=s['step'], kind='mixed' if s['decode'] else 'prefill',
                             tokens=s['scheduled_tokens'], phase=p, layer=i,
                             tp_median_device_ms=statistics.median(r[p,i]['device_ms'] for r in ranks),
                             tp_median_host_ms=statistics.median(r[p,i]['host_ms'] for r in ranks)))
    require(any(r['phase'] == 'dense_fia' for r in rows), 'no attributable measurements')
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'state', 'abi'):
        p.add_argument(name, type=Path)
    p.add_argument('lane')
    a = p.parse_args()
    rows = summarize(a.config, a.state, a.abi, a.lane)
    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
