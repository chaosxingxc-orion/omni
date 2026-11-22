"""Distinguish HTTP occupancy, scheduler preemption, and mixed-step interference."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import math
from io_utils import doc, pin, require


def report(root, steps=None, lane=None):
    root = root.resolve()
    require(doc(pin(root/'DRAINED.json'))['guards_releasable'], 'round not drained')
    result = doc(pin(root/'RESULT.json'))
    require(result['passed'], 'unsuccessful round')
    ref, = result['rounds']
    r = doc(ref)
    require(r['admission'] == 'continuous', 'expected continuous load')
    out = {}
    for name, summary in r['per_service'].items():
        rows = [x for x in r['requests'] if x['service'] == name]
        require(len(rows) == summary['requests'], 'request denominator differs')
        events = defaultdict(int)
        for row in rows:
            a, b = row['ready_to_http_seconds'], row['finished_from_round_seconds']
            require(0 <= a < b and math.isclose(b-a, row['elapsed_seconds'], abs_tol=1e-6),
                    'inconsistent monotonic interval')
            events[a] += 1
            events[b] -= 1
        by_count = defaultdict(float)
        points = sorted(events)
        n = 0
        for i, t in enumerate(points):
            n += events[t]
            if i+1 < len(points):
                require(0 <= n <= r['concurrency_per_service'][name], 'invalid client concurrency')
                by_count[n] += points[i+1] - t
        require(n == 0, 'unbalanced HTTP intervals')
        peak = max(by_count)
        span = points[-1]-points[0]
        require(math.isclose(sum(k*v for k,v in by_count.items()), sum(x['elapsed_seconds'] for x in rows)),
                'client occupancy mismatch')
        def counter(phase):
            p = Path(ref['path']).parent/name/(phase+'.json')
            raw = doc(pin(p))['metrics'].splitlines()
            v, = [s for s in raw if s.startswith('vllm:num_preemptions_total{')]
            return float(v.rsplit(' ', 1)[1])
        before, after = counter('BEFORE'), counter('AFTER')
        require(before <= after, 'preemption counter reset')
        out[name] = dict(requests=len(rows), http_peak=peak,
                         http_peak_fraction=by_count[peak]/span,
                         mean_in_flight=sum(k*v for k,v in by_count.items())/span,
                         preemptions=after-before)
    if steps is not None:
        s = doc(pin(steps))
        require(lane in out and ref['sha256'] == s['round_sha256'], 'mismatched step window')
        rs = s['rows']
        gaps = r['metrics'][lane]['itl_intervals']
        require(gaps == s['intervals'] == sum(x['generation_tokens'] for x in rs), 'gaps differ')
        first = next(i for i,x in enumerate(rs) if x['generation_tokens'])
        out[lane]['step_counts'] = dict(
            multi_decode_steps=sum(x['generation_tokens'] > 1 for x in rs),
            with_decode=len(rs)-sum(not x['generation_tokens'] for x in rs),
            context_only_after_first_decode=sum(x['kind']=='prefill' for x in rs[first:]),
            mixed_over_1s=sum(x['kind']=='mixed' and x['elapsed_ms']>1000 for x in rs),
            mixed_steps=sum(x['kind']=='mixed' for x in rs))
    return dict(round_sha256=ref['sha256'], arms=out,
                scope='HTTP occupancy is not device utilization; step counts do not identify per-request scheduling gaps.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('rounds',type=Path)
    p.add_argument('--steps',type=Path)
    p.add_argument('--lane')
    a=p.parse_args()
    print(json.dumps(report(a.rounds,a.steps,a.lane),indent=2))
