"""Read-only live socket lookup comparison; no HTTP or model requests.

Run inside the fixed image with host PID/network namespaces and a pinned
source tree. The full-scan arm clears search hints, not live identity checks.
This measures socket lookup only, never inference throughput.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from speech_aware_evidence_acquisition.core.session import vllm_identity as identity


def benchmark(pid, host, port, repetitions=30, proc_root=Path('/proc')):
    if not 2 <= repetitions <= 200:
        raise ValueError('repetitions must be between 2 and 200')
    stat_path = proc_root / str(pid) / 'stat'
    def lifetime():
        # comm may contain spaces or parentheses; field 22 is starttime.
        return stat_path.read_text().rsplit(')', 1)[1].split()[19]
    initial = lifetime()
    results = {'full_scan': [], 'candidate_hint': []}
    with identity.http_probe_connection_scope():
        for index in range(repetitions):
            order = ('full_scan', 'candidate_hint') if index % 2 == 0 else ('candidate_hint', 'full_scan')
            for mode in order:
                if mode == 'full_scan':
                    identity._probe_local.cache.socket_candidates.clear()
                cpu = time.thread_time()
                wall = time.perf_counter()
                identity._verify_owns_listening_socket(proc_root, pid, host, port)
                results[mode].append({'wall_seconds': time.perf_counter()-wall,
                                      'cpu_seconds': time.thread_time()-cpu})
                if lifetime() != initial:
                    raise ValueError('serving process lifetime changed during comparison')
    def summarize(rows):
        return {key: {'mean': statistics.mean(r[key] for r in rows),
                      'median': statistics.median(r[key] for r in rows),
                      'max': max(r[key] for r in rows)}
                for key in ('wall_seconds', 'cpu_seconds')}
    return {'schema': 'saea-socket-lookup-comparison-v1',
            'scope': 'live socket lookup only; full-scan control clears search hints',
            'pid': pid, 'process_starttime': initial, 'host': host, 'port': port,
            'source_sha256': hashlib.sha256(Path(identity.__file__).read_bytes()).hexdigest(),
            'repetitions_per_arm': repetitions, 'new_model_calls': 0,
            'http_requests': 0, 'end_to_end_throughput_verified': False,
            'summary': {k: summarize(v) for k, v in results.items()}, 'observations': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--repetitions', type=int, default=30)
    parser.add_argument('--proc-root', type=Path, default=Path('/proc'))
    print(json.dumps(benchmark(**vars(parser.parse_args())), sort_keys=True))
