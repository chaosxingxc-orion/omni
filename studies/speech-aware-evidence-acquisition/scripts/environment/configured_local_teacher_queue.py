#!/usr/bin/env python3
"""Durable local orchestration of existing teacher CLIs; unknown work is held."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

SCHEMA = 'saea-local-teacher-queue-v1'
KINDS = {'external-exploration': ('external_exploration.py', 'saea-external-exploration-prepared-v1'),
         'tfgrpo-reference': ('tfgrpo_external_reference.py', 'saea-tfgrpo-external-reference-prepared-v1')}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''): h.update(part)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_bytes())


def private_dir(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and not path.is_symlink(), 'absolute owned directory required')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(path.resolve() == path and path.stat().st_mode & 0o077 == 0, 'private canonical directory required')
    return path


def persist(path, doc, replace=False):
    path = Path(path); temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(doc, stream, sort_keys=True, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    if replace: os.replace(temp, path)
    else: os.link(temp, path); temp.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def pinned(value):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'exact file pin required')
    path = Path(value['path'])
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink() and path.is_file(), 'canonical regular pinned file required')
    require(re.fullmatch('[a-f0-9]{64}', value['sha256']) and sha(path) == value['sha256'], 'pinned file changed')
    return path


def load(config):
    doc = read(config)
    require(set(doc) == {'schema', 'queue_id', 'state_dir', 'max_parallel', 'jobs'} and doc['schema'] == SCHEMA, 'exact local queue schema required')
    require(re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}', doc['queue_id']), 'queue identity required')
    require(type(doc['max_parallel']) is int and 1 <= doc['max_parallel'] <= 8, 'bounded parallelism required')
    require(isinstance(doc['jobs'], list) and 0 < len(doc['jobs']) <= 1000, 'finite job list required')
    names, roots = set(), set()
    for job in doc['jobs']:
        require(set(job) == {'id', 'kind', 'cli', 'prepared', 'depends_on'}, 'exact job schema required')
        require(isinstance(job['id'], str) and re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', job['id']) and job['id'] not in names, 'unique job ID required')
        require(job['kind'] in KINDS, 'supported teacher adapter required')
        cli, prepared = pinned(job['cli']), pinned(job['prepared'])
        require(cli.name == KINDS[job['kind']][0] and prepared.name == 'prepared.json', 'existing CLI and prepared manifest required')
        require(read(prepared)['schema'] == KINDS[job['kind']][1], 'prepared algorithm mismatch')
        require(prepared.parent not in roots, 'prepared request cannot be queued twice')
        roots.add(prepared.parent); names.add(job['id'])
        require(isinstance(job['depends_on'], list) and len(set(job['depends_on'])) == len(job['depends_on']), 'unique dependencies required')
    edges = {j['id']: j['depends_on'] for j in doc['jobs']}
    def visit(name, stack):
        require(name in names and name not in stack, 'unknown dependency or cycle')
        for parent in edges[name]: visit(parent, stack | {name})
    for name in names: visit(name, set())
    root = private_dir(doc['state_dir'])
    identity = {'config_sha256': sha(config), 'runner_sha256': sha(__file__), 'queue_id': doc['queue_id']}
    if (root/'identity.json').exists(): require(read(root/'identity.json') == identity, 'queue config or implementation changed')
    else: persist(root/'identity.json', identity)
    return doc, root


def process_identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z': return None
        return {'pid': pid, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'start_ticks': stat[19]}
    except (OSError, IndexError): return None


def alive(identity): return bool(identity and process_identity(identity['pid']) == identity)


def worker(config, job_id, action='run'):
    doc, root = load(config); job = next(j for j in doc['jobs'] if j['id'] == job_id)
    d = private_dir(root/job_id)
    with (d/'worker.guard').open('a+b') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_name = 'result.json' if action == 'run' else 'reconciled.json'
        if (d/result_name).exists(): return read(d/result_name)
        if action == 'run' and (d/'intent.json').exists(): raise ValueError('existing dispatch intent: reconcile only')
        if action == 'run': persist(d/'intent.json', {'job_id': job_id, 'prepared_sha256': job['prepared']['sha256'], 'created_at': time.time()})
        persist(d/'worker.json', process_identity(os.getpid()), replace=True)
        cli, prepared = pinned(job['cli']), pinned(job['prepared'])
        # No environment values, provider output or exception text enter queue logs.
        proc = subprocess.Popen([sys.executable, str(cli), action, str(prepared.parent)], stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        persist(d/'child.json', process_identity(proc.pid), replace=True)
        code = proc.wait(); collected = prepared.parent/'collected.json'
        result = {'job_id': job_id, 'action': action, 'exit_code': code, 'completed_at': time.time(),
                  'status': 'complete' if code == 0 and collected.is_file() else 'failed-held',
                  'collected': {'path': str(collected), 'sha256': sha(collected)} if code == 0 and collected.is_file() else None,
                  'automatic_retry': False}
        persist(d/result_name, result); return result


def job_status(root, job):
    d = root/job['id']
    for name in ('reconciled.json', 'result.json'):
        if (d/name).exists():
            result = read(d/name)
            if result['status'] == 'complete': pinned(result['collected'])
            return result['status']
    for name in ('worker.json', 'child.json'):
        if (d/name).exists() and alive(read(d/name)): return 'running'
    if any((d/n).exists() for n in ('launch.json', 'intent.json')): return 'unknown-held'
    return 'pending'


def launch_worker(config, root, job):
    d = private_dir(root/job['id'])
    persist(d/'launch.json', {'job_id': job['id'], 'requested_at': time.time(), 'automatic_retry': False})
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--config', str(Path(config).resolve()), '--job', job['id']],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    persist(d/'worker.json', process_identity(proc.pid), replace=True); return proc


def run(config):
    doc, root = load(config)
    with (root/'queue.guard').open('a+b') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        persist(root/'supervisor.json', process_identity(os.getpid()), replace=True)
        children = {}
        while True:
            for pid, proc in list(children.items()):
                if proc.poll() is not None: del children[pid]
            states = {j['id']: job_status(root, j) for j in doc['jobs']}
            count = sum(v == 'running' for v in states.values())
            for job in doc['jobs']:
                if count >= doc['max_parallel']: break
                if states[job['id']] == 'pending' and all(states[p] == 'complete' for p in job['depends_on']):
                    proc = launch_worker(config, root, job); children[proc.pid] = proc
                    states[job['id']] = 'running'; count += 1
            persist(root/'state.json', {'queue_id': doc['queue_id'], 'jobs': states, 'observed_at': time.time(), 'automatic_retry': False}, replace=True)
            if not count: return states
            if children:
                pid, status = os.waitpid(-1, 0)
                if pid in children:
                    children.pop(pid).returncode = os.waitstatus_to_exitcode(status)
            else: time.sleep(600)


def start(config):
    _, root = load(config)
    with (root/'launch.guard').open('a+b') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'supervisor.json').exists() and alive(read(root/'supervisor.json')): return {'state': 'running', **read(root/'supervisor.json')}
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'run', '--config', str(Path(config).resolve())],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        identity = process_identity(proc.pid); persist(root/'supervisor.json', identity, replace=True)
        return {'state': 'started', **(identity or {}), 'automatic_retry': False}


def reconcile(config, job_id):
    doc, root = load(config); job = next(j for j in doc['jobs'] if j['id'] == job_id)
    require(job_status(root, job) != 'running', 'live worker or child cannot be reconciled')
    return worker(config, job_id, action='collect')


def export(config, job_id, output):
    """Build a private, source-pinned handoff; never deploy or select a candidate."""
    doc, root = load(config); job = next(j for j in doc['jobs'] if j['id'] == job_id)
    require(job_status(root, job) == 'complete', 'only completed collected work can be exported')
    d = root/job_id
    result = read(d/('reconciled.json' if (d/'reconciled.json').exists() else 'result.json'))
    collected = pinned(result['collected']); record = read(collected)
    files = {'collected.json': collected}
    if job['kind'] == 'tfgrpo-reference':
        candidate = pinned(record['candidate'])
        require(candidate.resolve().is_relative_to(collected.parent), 'candidate must belong to prepared task')
        files['candidate.json'] = candidate
    output = Path(output)
    require(not output.exists(), 'fresh handoff directory required')
    output = private_dir(output)
    entries = []
    for name, path in files.items():
        raw = path.read_bytes()
        # Immutable publication and fsync; source bytes are not reserialized.
        fd = os.open(output/name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        entries.append({'path': name, 'sha256': sha(output/name), 'source_sha256': sha(path)})
    manifest = {'schema': 'saea-local-teacher-handoff-v1', 'queue_id': doc['queue_id'],
                'job_id': job_id, 'prepared_sha256': job['prepared']['sha256'], 'files': entries,
                'candidate_accepted': False, 'deployment_ready': False,
                'meaning': 'collected candidate only; full source custody and semantic release remain required'}
    persist(output/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'run', 'worker', 'reconcile', 'status', 'export'])
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--job'); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action in ('worker', 'reconcile', 'export'): require(args.job, 'job ID required')
    if args.action == 'export':
        require(args.output, 'private output directory required'); result = export(args.config, args.job, args.output)
    elif args.action == 'status':
        doc, root = load(args.config); result = {j['id']: job_status(root, j) for j in doc['jobs']}
    else: result = {'start': start, 'run': run, 'worker': worker, 'reconcile': reconcile}[args.action](args.config, *([args.job] if args.job else []))
    print(json.dumps(result, sort_keys=True))
