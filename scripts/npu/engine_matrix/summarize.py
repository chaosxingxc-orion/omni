"""Read-only matrix analysis; no serving requests, deployment changes or promotion."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_bytes())


def quantile(values, fraction):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lo = int(index)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)


def trial(root, candidate, trajectories, now):
    total = trajectories * 3
    records = [read(p) for p in (root / 'requests').glob('*.result.json')]
    intents = len(list((root / 'requests').glob('*.intent.json')))
    completed = [r for r in records if r['result']['status'] == 'ok']
    groups = defaultdict(set)
    for r in completed:
        value = r['result']
        groups[value['trajectory']].add(value['contact'])
    done = sum(contacts == {0, 1, 2} for contacts in groups.values())
    final = read(root / 'complete.json') if (root / 'complete.json').exists() else None
    progress = read(root / 'progress.json') if (root / 'progress.json').exists() else {}
    segments = sorted((root / 'segments').glob('*'))
    generation_starts = []
    metrics = []
    for segment in segments:
        path = segment / 'metrics.jsonl'
        if not path.exists():
            continue
        lines = path.read_bytes().splitlines(keepends=True)
        samples = [json.loads(line) for line in lines if line.endswith(b'\n')]
        if samples:
            generation_starts.append(samples[0]['time'])
            metrics.extend(samples)
    terminal_paths = [segment / 'result.json' for segment in segments if (segment / 'result.json').exists()]
    terminal = read(terminal_paths[-1]) if terminal_paths and terminal_paths[-1].parent == segments[-1] else None
    elapsed = (final or terminal)['wall_s'] if (final or terminal) else (now - generation_starts[-1] if generation_starts else None)
    comparable = len(generation_starts) == 1 and bool(final) and not final['resumed'] and not final['interrupted']
    # Partial or resumed runs retain counts but never manufacture a new baseline.
    efficiency_valid = len(generation_starts) == 1 and elapsed is not None and elapsed > 0 and not (terminal and terminal.get('interrupted'))
    rate = done / elapsed if efficiency_valid and done else None
    output = sum(r['result'].get('completion_tokens') or 0 for r in completed)
    counters = {}
    if final:
        a, b = final['metrics_before'], final['metrics_after']
        for key in ('preemptions', 'prefix_cache_queries', 'prefix_cache_hits', 'generation_tokens'):
            x, y = (a or {}).get(key), (b or {}).get(key)
            counters[key] = y - x if x is not None and y is not None and y >= x else None
    hits, queries = counters.get('prefix_cache_hits'), counters.get('prefix_cache_queries')
    gauges = {}
    for key in ('kv_cache_usage_perc', 'num_requests_running', 'num_requests_waiting'):
        values = [s['metrics'][key] for s in metrics if s.get('metrics') and s['metrics'].get(key) is not None]
        gauges[key] = {'max': max(values), 'mean': sum(values) / len(values)} if values else None
    steady = (final or {}).get('steady', {})
    return {'candidate': candidate['id'], 'concurrency': candidate['client_concurrency'],
            'state': 'complete' if final else ('stopped' if terminal else ('generating' if generation_starts else ('fitting' if segments else 'waiting'))),
            'cards': 2, 'instances': 1, 'completed_requests': len(completed), 'total_requests': total,
            'completed_trajectories': done, 'total_trajectories': trajectories,
            'errors': len(records) - len(completed), 'inflight_or_unresolved_intents': intents - len(records),
            'fit_completed_trajectories': len(list((root / 'fit').glob('*.json'))),
            'generation_start': generation_starts[0] if len(generation_starts) == 1 else None,
            'generation_wall_s': elapsed, 'fit_wall_s': (final or terminal or progress).get('fit_wall_s'),
            'seconds_per_complete_trajectory': 1 / rate if rate else None,
            'complete_trajectories_h': rate * 3600 if rate else None,
            'ETA_s': (trajectories - done) / rate if rate else None,
            'output_tokens': output, 'output_tokens_s': output / elapsed if efficiency_valid else None,
            'mean_output_tokens': output / len(completed) if completed else None,
            'ttft_p95_s': quantile([r['result'].get('ttft_s') for r in completed], .95),
            'latency_p95_s': quantile([r['result'].get('latency_s') for r in completed], .95),
            'counter_deltas': counters, 'prefix_hit_ratio': hits / queries if hits is not None and queries else None,
            'gauges': gauges, 'steady': steady, 'uninterrupted_comparison_eligible': comparable,
            'steady_window_eligible': comparable and steady.get('qualified', False) and steady.get('wall_s', 0) >= 60}


def summarize(plan, out, round_index=0, now=None):
    now = time.time() if now is None else now
    candidates = plan['rounds'][round_index]
    trials = [trial(out / f'round-{round_index}' / c['id'], c, plan['trajectories'], now) for c in candidates]
    starts = [t['generation_start'] for t in trials]
    wall = None
    if all(v is not None for v in starts) and all(t['generation_wall_s'] is not None for t in trials):
        wall = max(t['generation_start'] + t['generation_wall_s'] for t in trials) - min(starts)
    done = sum(t['completed_trajectories'] for t in trials)
    return {'schema': 'saea-engine-matrix-read-v1', 'updated_unix': now,
            'dataset_rows': 0, 'cards': 2 * len(trials), 'instances': len(trials),
            'completed_requests': sum(t['completed_requests'] for t in trials),
            'total_requests': sum(t['total_requests'] for t in trials),
            'completed_trajectories': done, 'total_trajectories': plan['trajectories'] * len(trials),
            'host_generation_wall_s': wall, 'host_seconds_per_trajectory': wall / done if wall and done else None,
            'host_trajectories_h': done * 3600 / wall if wall else None,
            'ETA_s': max(t['ETA_s'] for t in trials) if all(t['ETA_s'] is not None for t in trials) else None,
            'all_complete': all(t['state'] == 'complete' for t in trials), 'trials': trials,
            'promotion': False, 'limitation': 'Synthetic complete trajectories are not A0 dataset samples. Single-round rankings do not prove optimum; confirm by crossover and real-task throughput.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--round', type=int, default=0); p.add_argument('--summary', type=Path)
    a = p.parse_args(); result = summarize(read(a.plan), a.out, a.round)
    value = json.dumps(result, indent=2) + '\n'
    if a.summary:
        a.summary.parent.mkdir(parents=True, exist_ok=True)
        temporary = a.summary.with_suffix('.tmp'); temporary.write_text(value); temporary.replace(a.summary)
    else:
        print(value, end='')


if __name__ == '__main__':
    main()
