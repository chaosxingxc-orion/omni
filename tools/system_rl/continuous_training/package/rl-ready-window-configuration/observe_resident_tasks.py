"""Read-only stream of real engine gauges and device activity. No files on the NPU host."""
import concurrent.futures
import json
import re
import subprocess
import sys
import time
import urllib.request


def devices():
    raw = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=7)
    rows = {}
    card = None
    for line in raw.splitlines():
        fields = [part.strip() for part in line.split('|')[1:-1]]
        if len(fields) == 3 and re.fullmatch(r'\d+\s+\S+', fields[0]):
            card = int(fields[0].split()[0])
        elif len(fields) == 3 and card is not None and (m := re.fullmatch(r'(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)', fields[2])):
            used, capacity = map(int, m.groups()[1:])
            rows[card] = dict(aicore=int(m[1]), hbm_mib=used, capacity_mib=capacity)
            card = None
    if not rows:
        raise ValueError('no device records')
    return rows


def lane(url):
    names = ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc',
             'num_preemptions_total', 'prefix_cache_hits_total', 'prefix_cache_queries_total',
             'generation_tokens_total', 'prompt_tokens_total', 'request_success_total',
             'request_queue_time_seconds_sum', 'request_queue_time_seconds_count',
             'request_prefill_time_seconds_sum', 'request_decode_time_seconds_sum')
    with urllib.request.urlopen(url.rstrip('/') + '/metrics', timeout=3) as response:
        raw = response.read().decode()
    return {name: sum(values) if values else None for name in names
            for values in [[float(line.rsplit(' ', 1)[1]) for line in raw.splitlines()
                            if line.startswith('vllm:' + name + '{')]]}


def observe(config):
    interval, duration = config['interval_s'], config['duration_s']
    if not (interval > 0 and duration > 0):
        raise ValueError('positive window and interval required')
    end = time.monotonic() + duration
    with concurrent.futures.ThreadPoolExecutor() as pool:
        while time.monotonic() < end:
            before = time.monotonic()
            row = {'at': time.time()}
            jobs = {'devices': pool.submit(devices)}
            jobs.update((key, pool.submit(lane, url)) for key, url in config['services'].items())
            for key, job in jobs.items():
                try:
                    row[key] = job.result()
                except Exception as exc:
                    row[key] = {'error': repr(exc)}
            row['read_seconds'] = time.monotonic() - before
            print(json.dumps(row), flush=True)
            time.sleep(max(0, min(interval-row['read_seconds'], end-time.monotonic())))


if __name__ == '__main__':
    observe(json.loads(sys.argv[1]))
