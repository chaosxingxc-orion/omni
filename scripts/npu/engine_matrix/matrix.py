"""Durable synthetic engine trials. Deployment mutation is deliberately separate."""
from __future__ import annotations
import argparse, concurrent.futures, dataclasses, fcntl, hashlib, importlib.util, json, os
from pathlib import Path
import queue, random, re, signal, socket, http.client, sys, threading, time
from urllib.parse import urlparse, quote
D = Path(os.environ.get('SPEECHRL_DATA_DIR', '/mnt/cephfs/xingchao/omni'))
S = Path(os.environ.get('SAEA_STUDY_ROOT', str(Path(__file__).resolve().parents[3] / 'studies/speech-aware-evidence-acquisition')))
TOOL = S / 'scripts/serving/synthetic_sweep.py'
MODEL = 'qwen3-omni-30b-a3b-thinking'
IMAGE = 'sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026'


def raw(value): return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
def digest(value): return hashlib.sha256(raw(value)).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())


def durable(path, value, exclusive=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()) + '-' + str(threading.get_ident()))
    with tmp.open('wb') as f:
        f.write(raw(value)); f.flush(); os.fsync(f.fileno())
    if exclusive:
        try: os.link(tmp, path)
        finally: tmp.unlink()
    else: os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def load_tool():
    spec = importlib.util.spec_from_file_location('engine_synthetic_sweep', TOOL)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m


def plan_requests(count, contacts, seed, tool, *, content_seed=None):
    """Same lengths/audio durations across rounds; fresh content prevents inherited cache."""
    rng = random.Random(seed); mu = tool.lognormal_mu_for_mean(8500, .5, 3000, 20000)
    return [tool.Trajectory(i, f'{content_seed or seed}:{i}', round(rng.uniform(2, 8), 3),
            tuple(sorted(tool.sample_prompt_targets(rng, contacts, mu=mu, sigma=.5, low=3000, high=20000))))
            for i in range(count)]


def resolve_request(root, key, body_hash):
    root = Path(root); result = root / (key + '.result.json'); intent = root / (key + '.intent.json')
    if result.exists():
        d = read(result)
        if (not intent.exists() or d.get('request_id') != key or d.get('body_sha256') != body_hash
                or read(intent).get('body_sha256') != body_hash):
            raise ValueError('checkpoint request changed or lacks intent: ' + key)
        return d
    if intent.exists(): raise ValueError('unresolved sent request; reconcile before resume: ' + key)
    return None


def execute_request(root, key, body, send):
    root = Path(root); h = digest(body); done = resolve_request(root, key, h)
    if done is not None: return done, False
    durable(root / (key + '.intent.json'), {'body_sha256': h, 'request_id': key, 'started_at': time.time()})
    value = send()  # Exception intentionally leaves unresolved intent, never auto-replayed.
    out = {'body_sha256': h, 'request_id': key, 'finished_at': time.time(), 'result': value}
    durable(root / (key + '.result.json'), out)
    return out, True


def refuse_unresolved(root):
    for p in Path(root).rglob('*.intent.json'):
        if not p.with_name(p.name.replace('.intent.json', '.result.json')).exists():
            raise ValueError('unresolved request: ' + str(p))


def inspect_binding(candidate, doc, startup):
    args = doc['Args']; flags = {a: args[i+1] for i, a in enumerate(args[:-1]) if a.startswith('--')}
    if not doc['State']['Running']: raise ValueError('engine not running')
    if doc['Image'] != candidate['image']: raise ValueError('engine image differs')
    if flags.get('--served-model-name') != candidate['model']: raise ValueError('served model differs')
    if flags.get('--port') != str(urlparse(candidate['url']).port): raise ValueError('endpoint port differs')
    if flags.get('--tensor-parallel-size') != '2': raise ValueError('tensor parallel geometry differs')
    if flags.get('--max-model-len') != '24576': raise ValueError('context window differs')
    for key, flag in [('max_num_seqs', '--max-num-seqs'), ('max_num_batched_tokens', '--max-num-batched-tokens')]:
        wanted = candidate[key]
        if wanted is not None and flags.get(flag) != str(wanted): raise ValueError('engine differs: ' + flag)
        if wanted is None and flag in flags: raise ValueError('expected recorded server default: ' + flag)
    for key in ('enable_prefix_caching', 'enable_chunked_prefill'):
        value = candidate[key]
        if not re.search(r'\b' + key + '=' + str(value) + r'\b', startup):
            raise ValueError('startup engine flag not verified: ' + key)
    return {'container_id': doc['Id'], 'image': doc['Image'], 'args': args, 'pid': doc['State']['Pid'],
            'verified_engine_flags': {k: candidate[k] for k in ('enable_prefix_caching', 'enable_chunked_prefill')},
            'batched_tokens_default_not_inferred': candidate['max_num_batched_tokens'] is None}


def docker_get(path):
    """Read-only Docker API through a mounted Unix socket; no Docker CLI required."""
    class UnixConnection(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect('/var/run/docker.sock')
    conn = UnixConnection('localhost', timeout=30)
    try:
        conn.request('GET', path)
        response = conn.getresponse(); data = response.read()
        if response.status != 200: raise ValueError('Docker read failed: ' + str(response.status))
        return data
    finally: conn.close()


def validate_live(candidate):
    name = quote(candidate['container'], safe='')
    doc = json.loads(docker_get('/containers/' + name + '/json'))
    # Docker logs may include binary stream framing; ASCII config strings survive decoding.
    startup = docker_get('/containers/' + name + '/logs?stdout=1&stderr=1').decode(errors='replace')
    return inspect_binding(candidate, doc, startup)


def stable_binding(binding):
    d = json.loads(json.dumps(binding)); d['live'].pop('pid', None); d['live'].pop('container_id', None)
    return d


def fit_one(tool, config, trajectory, root, stop):
    """Cache every /tokenize contact, not only the final fitted trajectory."""
    fp = root / 'fit' / f'{trajectory.index}.json'
    if fp.exists(): return [tool.FittedContact(**v) for v in read(fp)]
    audio = trajectory.audio_b64(); words = trajectory.words(max(trajectory.targets) + 64); fits = []
    for contact, target in enumerate(trajectory.targets):
        def count(n_words, pad):
            if stop.is_set(): raise InterruptedError('stopped before tokenizer submission')
            if n_words > len(words): words.extend(trajectory.words(n_words * 2)[len(words):])
            messages = tool.contact_messages(audio, words, n_words, contact, len(trajectory.targets), pad)
            body = {'model': config.model, 'messages': messages,
                    'chat_template_kwargs': {'enable_thinking': True},
                    'add_generation_prompt': True, 'add_special_tokens': False}
            key = f't{trajectory.index:05d}-k{contact}-n{n_words}-p{pad}'
            done, _ = execute_request(root / 'tokenize', key, body,
                    lambda: {'count': tool.tokenize_count(config.base_urls[0], config.model, messages)})
            return done['result']['count']
        fits.append(tool.fit_contact(count, target))
    durable(fp, [dataclasses.asdict(v) for v in fits]); return fits


def steady_summary(samples, start, end):
    """Counter delta inside the observed continuously fed interval, excluding drain."""
    selected = [s for s in samples if start is not None and end is not None and start <= s['time'] <= end]
    result = {'start': start, 'end': end, 'samples': len(selected), 'qualified': False}
    if len(selected) < 3: return result
    first, last = selected[0], selected[-1]; elapsed = last['time'] - first['time']
    a, b = first['metrics'], last['metrics']
    if not a or not b or elapsed <= 0: return result
    for key in ('generation_tokens', 'request_success'):
        if a.get(key) is not None and b.get(key) is not None:
            delta = b[key] - a[key]
            if delta < 0: return {**result, 'reason': 'counter reset'}
            result[key + '_s'] = delta / elapsed
    result.update(qualified='generation_tokens_s' in result, wall_s=elapsed)
    return result


def run_trial(tool, candidate, root, trajectories, stop, barrier=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    config = tool.SweepConfig(base_urls=[candidate['url']], model=candidate['model'],
                             levels=[candidate['client_concurrency']], warmup=0)
    binding = {'candidate': candidate, 'live': validate_live(candidate),
               'workload_sha256': digest([dataclasses.asdict(t) for t in trajectories]),
               'tool_sha256': hashlib.sha256(TOOL.read_bytes()).hexdigest(),
               'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if (root / 'binding.json').exists():
        if stable_binding(read(root / 'binding.json')) != stable_binding(binding): raise ValueError('trial binding changed')
    else: durable(root / 'binding.json', binding)
    refuse_unresolved(root)
    total = sum(len(t.targets) for t in trajectories)
    if (root / 'complete.json').exists():
        result = read(root / 'complete.json')
        if (result['completed'] != total or len(list((root / 'requests').glob('*.result.json'))) != total
                or result.get('request_result_hashes') != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'requests').glob('*.result.json')}):
            raise ValueError('completed checkpoint coverage differs')
        return {**result, 'already_complete': True, 'new_requests': 0}
    tool.preflight(config)
    gauges = tool.read_metrics(candidate['url'])
    if gauges is None or any(gauges.get(k) is None or gauges[k] != 0 for k in ('num_requests_running', 'num_requests_waiting')):
        raise ValueError('endpoint not demonstrably idle')
    segment = str(time.time_ns()); segment_root = root / 'segments' / segment; segment_root.mkdir(parents=True)
    durable(segment_root / 'started.json', {'time': time.time(), 'live': binding['live'], 'metrics': gauges})
    fit_started = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            fits = dict(zip((t.index for t in trajectories), pool.map(lambda t: fit_one(tool, config, t, root, stop), trajectories)))
    except BaseException:
        stop.set(); raise
    fit_wall = time.time() - fit_started
    if stop.is_set(): return {'interrupted': True, 'phase': 'fit', 'new_requests': 0}
    if barrier is not None:
        try: barrier.wait(timeout=1800)
        except threading.BrokenBarrierError:
            stop.set(); raise RuntimeError('fit stage barrier failed; no blind dispatch')
    before = tool.read_metrics(candidate['url']); started = time.time(); lock = threading.Lock(); fresh = []
    pending = queue.Queue()
    for t in trajectories: pending.put(t)
    status = {'completed': len(list((root / 'requests').glob('*.result.json'))), 'steady_start': None, 'drain_start': None}
    resumed = status['completed'] > 0
    samples = []; sample_stop = threading.Event()
    def sample_loop():
        while not sample_stop.is_set():
            sample = {'time': time.time(), 'metrics': tool.read_metrics(candidate['url'])}; samples.append(sample)
            with (segment_root / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(sample) + '\n'); f.flush(); os.fsync(f.fileno())
            sample_stop.wait(5)
    sampler = threading.Thread(target=sample_loop, daemon=True); sampler.start()
    def worker(worker_id):
        try:
            while not stop.is_set():
                try: t = pending.get_nowait()
                except queue.Empty: return
                with lock:
                    if pending.empty() and status['drain_start'] is None: status['drain_start'] = time.time()
                audio = t.audio_b64(); fs = fits[t.index]; words = t.words(max(v.n_words for v in fs))
                for contact, fit in enumerate(fs):
                    if stop.is_set(): return
                    key = f't{t.index:05d}-k{contact}'
                    body = dict(tool._decoding(config), messages=tool.contact_messages(audio, words, fit.n_words, contact, len(fs), fit.pad))
                    def send():
                        r = tool.RequestResult(candidate['client_concurrency'], worker_id, 0, t.index, contact,
                                               root.name + '-' + key, fit.target, fit.tokenized)
                        r.started_s = time.time() - started
                        tool.send_chat(candidate['url'], body, r, timeout=config.timeout_s, stream=True)
                        return dataclasses.asdict(r)
                    done, new = execute_request(root / 'requests', key, body, send)
                    with lock:
                        if new:
                            fresh.append(done); status['completed'] += 1
                            if len(fresh) >= candidate['client_concurrency'] and status['steady_start'] is None:
                                status['steady_start'] = time.time()
                        elapsed = time.time() - started
                        durable(root / 'progress.json', {'completed_contacts': status['completed'], 'total_contacts': total,
                            'fresh_contacts': len(fresh), 'generation_elapsed_s': elapsed, 'fit_wall_s': fit_wall,
                            'seconds_per_fresh_contact': elapsed / len(fresh) if fresh else None,
                            'resources': {'cards': 2, 'instances': 1, 'tp': 2},
                            'ETA_s': (total - status['completed']) * elapsed / len(fresh) if fresh else None}, False)
                    if done['result']['status'] != 'ok': stop.set(); return
        except BaseException:
            stop.set(); raise
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=candidate['client_concurrency']) as pool:
            futures = [pool.submit(worker, i) for i in range(candidate['client_concurrency'])]
            for future in concurrent.futures.as_completed(futures): future.result()
    finally:
        sample_stop.set(); sampler.join(timeout=20)
    elapsed = time.time() - started; after = tool.read_metrics(candidate['url'])
    results = [read(p) for p in (root / 'requests').glob('*.result.json')]
    record = {'started_at': started, 'wall_s': elapsed, 'fit_wall_s': fit_wall, 'new_requests': len(fresh),
              'new_output_tokens': sum(d['result']['completion_tokens'] or 0 for d in fresh),
              'completed': len(results), 'total': total, 'errors': sum(d['result']['status'] != 'ok' for d in results),
              'metrics_before': before, 'metrics_after': after, 'interrupted': stop.is_set(),
              'resumed': resumed, 'steady': steady_summary(samples, status['steady_start'], status['drain_start']),
              'resume_comparison': 'resumed or interrupted trials require uninterrupted confirmation; all cost retained'}
    record['new_output_tokens_s'] = record['new_output_tokens'] / elapsed
    record['completed_requests_s'] = len(fresh) / elapsed
    durable(segment_root / 'result.json', record)
    if len(results) == total and not record['errors']:
        record['request_result_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'requests').glob('*.result.json')}
        durable(root / 'complete.json', record)
    return record


def validate_plan(plan, round_index):
    if plan.get('schema') != 'saea-engine-matrix-plan-v1': raise ValueError('plan schema')
    selected = plan['rounds'][round_index]
    if len(selected) != 4 or len({c['url'] for c in selected}) != 4 or len({c['id'] for c in selected}) != 4:
        raise ValueError('exactly four distinct endpoints/candidates required')
    if plan['trajectories'] < 3 * max(c['client_concurrency'] for c in selected):
        raise ValueError('at least three trajectories per worker required to measure fed interval')
    for c in selected:
        if not (0 < c['client_concurrency'] <= c['max_num_seqs']): raise ValueError('client exceeds recorded server sequences')
        if c['image'] != IMAGE or c['model'] != MODEL: raise ValueError('unreviewed image/model')
        if c['enable_prefix_caching'] is not True or c['enable_chunked_prefill'] is not True: raise ValueError('baseline flags must remain enabled')
    return selected


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--round', type=int, default=0); a = p.parse_args()
    plan = read(a.plan); selected = validate_plan(plan, a.round); tool = load_tool(); stop = threading.Event()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    barrier = None
    def request_stop(*_):
        stop.set()
        if barrier is not None: barrier.abort()
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, request_stop)
    with (D / 'locks/generation.guard').open('r+') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (D / 'locks/flight.lock').exists(): raise ValueError('scientific flight owns generation')
        a.out.mkdir(parents=True, exist_ok=True)
        with (a.out / 'run.lock').open('a+') as runlock:
            fcntl.flock(runlock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            trajectories = plan_requests(plan['trajectories'], 3, plan['seed'], tool, content_seed=f"{plan['seed']}:round-{a.round}")
            trial_roots = [a.out / f'round-{a.round}' / c['id'] for c in selected]
            barrier = threading.Barrier(4) if not any((root / 'complete.json').exists() for root in trial_roots) else None
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(run_trial, tool, c, root, trajectories, stop, barrier) for c, root in zip(selected, trial_roots)]
                try:
                    for f in concurrent.futures.as_completed(futures):
                        result = f.result(); print(json.dumps(result), flush=True)
                        if result.get('errors') or result.get('interrupted'):
                            raise RuntimeError('trial incomplete or failed; preserve checkpoints')
                except BaseException:
                    stop.set()
                    if barrier is not None: barrier.abort()
                    raise
if __name__ == '__main__': main()
