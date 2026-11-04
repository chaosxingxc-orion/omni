#!/usr/bin/env python3
"""Durable local orchestration of existing teacher CLIs; unknown work is held."""
import argparse
from contextlib import ExitStack
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


def implementation_inventory(cli):
    """Pin the entire first-party source tree, including dynamic import targets."""
    cli = Path(cli).resolve(); study = cli.parents[2]
    source = study/'src'
    require(source.is_dir(), 'adapter source tree required')
    paths = {cli, *source.rglob('*.py'), * (study/'scripts/environment').glob('*.py')}
    require(all(p.is_file() and not p.is_symlink() and p.resolve() == p for p in paths),
            'implementation files must be canonical regular files')
    return {'schema': 'saea-local-teacher-implementation-v1', 'cli': str(cli),
            'files': [{'path': str(p), 'sha256': sha(p)} for p in sorted(paths)]}


def verify_implementation(job):
    manifest = read(pinned(job['implementation']))
    require(manifest == implementation_inventory(pinned(job['cli'])),
            'adapter implementation inventory changed')
    return manifest


def load(config, *, export_runner=None):
    doc = read(config)
    require(set(doc) == {'schema', 'queue_id', 'state_dir', 'max_parallel', 'jobs'} and doc['schema'] == SCHEMA, 'exact local queue schema required')
    require(re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}', doc['queue_id']), 'queue identity required')
    require(type(doc['max_parallel']) is int and 1 <= doc['max_parallel'] <= 8, 'bounded parallelism required')
    require(isinstance(doc['jobs'], list) and 0 < len(doc['jobs']) <= 1000, 'finite job list required')
    names, roots = set(), set()
    for job in doc['jobs']:
        require(set(job) == {'id', 'kind', 'cli', 'prepared', 'implementation', 'depends_on'}, 'exact job schema required')
        require(isinstance(job['id'], str) and re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', job['id']) and job['id'] not in names, 'unique job ID required')
        require(job['kind'] in KINDS, 'supported teacher adapter required')
        cli, prepared = pinned(job['cli']), pinned(job['prepared'])
        verify_implementation(job)
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
    runner = pinned(export_runner) if export_runner is not None else Path(__file__)
    if export_runner is not None:
        require((root/'identity.json').is_file(), 'existing export source identity required')
    identity = {'config_sha256': sha(config), 'runner_sha256': sha(runner), 'queue_id': doc['queue_id']}
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


def live_job(d):
    return any((d/name).exists() and alive(read(d/name)) for name in ('worker.json', 'child.json'))


def last_result(d):
    for name in ('reconciled.json', 'result.json'):
        if (d/name).exists(): return read(d/name)
    return None


def worker(config, job_id, action='run'):
    doc, root = load(config); job = next(j for j in doc['jobs'] if j['id'] == job_id)
    d = private_dir(root/job_id)
    with ExitStack() as stack:
        guard = stack.enter_context((d/'worker.guard').open('a+b'))
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Shared across queue identities, independent of a queue's state directory.
        prepared = pinned(job['prepared'])
        source_guard = stack.enter_context((prepared.parent/'local-queue.guard').open('a+b'))
        fcntl.flock(source_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == 'collect': require(not live_job(d), 'live worker or child cannot be reconciled')
        previous = last_result(d)
        if previous and previous['status'] == 'complete':
            pinned(previous['collected']); return previous
        if action == 'run' and (d/'intent.json').exists(): raise ValueError('existing dispatch intent: reconcile only')
        if action == 'run': persist(d/'intent.json', {'job_id': job_id, 'prepared_sha256': job['prepared']['sha256'], 'created_at': time.time()})
        attempt_id = uuid.uuid4().hex
        if action == 'collect':
            attempts = private_dir(d/'reconciliations')
            persist(attempts/(attempt_id+'.intent.json'), {'job_id': job_id, 'created_at': time.time()})
            # Do not collect while an original provider process still owns its guard.
            # The reference CLI also acquires its existing learner/runner.lock.
            for path in sorted(prepared.parent.rglob('request.guard')):
                provider_guard = stack.enter_context(path.open('a+b'))
                fcntl.flock(provider_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_implementation(job)
        persist(d/'worker.json', process_identity(os.getpid()), replace=True)
        cli = pinned(job['cli'])
        proc = subprocess.Popen([sys.executable, str(cli), action, str(prepared.parent)], stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        persist(d/'child.json', process_identity(proc.pid), replace=True)
        code = proc.wait(); collected = prepared.parent/'collected.json'
        result = {'job_id': job_id, 'action': action, 'attempt_id': attempt_id, 'exit_code': code, 'completed_at': time.time(),
                  'status': 'complete' if code == 0 and collected.is_file() else 'failed-held',
                  'collected': {'path': str(collected), 'sha256': sha(collected)} if code == 0 and collected.is_file() else None,
                  'automatic_retry': False}
        if action == 'collect':
            persist(attempts/(attempt_id+'.result.json'), result)
            persist(d/'reconciled.json', result, replace=True)
        else: persist(d/'result.json', result)
        persist(d/'child.json', None, replace=True)
        persist(d/'worker.json', None, replace=True)
        return result


def job_status(root, job):
    d = root/job['id']
    if live_job(d): return 'running'
    result = last_result(d)
    if result:
        if result['status'] == 'complete': pinned(result['collected'])
        return result['status']
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


def source_references(document, source_path, profile_bindings):
    """Resolve pins using their document semantics, without consulting the filesystem.

    Preparation copies a profile verbatim, but separately pins the template it
    actually used. Its old relative template spelling is not relative to the
    copied profile. All other relative references are confined to their document.
    """
    source_path = Path(source_path)
    def walk(value, keys=()):
        if isinstance(value, dict):
            if {'path', 'sha256'} <= value.keys():
                raw = value['path']
                require(isinstance(raw, str) and raw and isinstance(value['sha256'], str),
                        'source reference path and digest required')
                path = Path(raw)
                if (isinstance(document, dict) and document.get('schema') == 'saea-external-exploration-profile-v1'
                        and keys == ('template',) and str(source_path) in profile_bindings):
                    binding = profile_bindings[str(source_path)]
                    require(value['sha256'] == binding['sha256'], 'copied profile template binding differs')
                    path = Path(binding['path'])
                elif not path.is_absolute():
                    require('..' not in path.parts, 'relative source reference escapes document directory')
                    path = source_path.parent/path
                require(path.is_absolute() and '..' not in path.parts, 'canonical source reference required')
                yield {'path': str(path), 'sha256': value['sha256']}
            for key, child in value.items(): yield from walk(child, keys+(key,))
        elif isinstance(value, list):
            for i, child in enumerate(value): yield from walk(child, keys+(i,))
    yield from walk(document)


def profile_binding(prepared):
    if prepared.get('schema') != 'saea-external-exploration-prepared-v1': return {}
    files = prepared.get('files', {})
    if 'profile.json' not in files or 'template.md' not in files: return {}
    return {files['profile.json']['path']: files['template.md']}


def source_inventory(job, collected):
    """Resolve every explicit pin recursively; retain the original absolute path map."""
    todo = list(Path(job['prepared']['path']).parent.rglob('*'))
    todo += [collected, pinned(job['implementation'])]
    files = {}
    bindings = profile_binding(read(job['prepared']['path']))
    forbidden = {'authorization', 'api_key', 'apikey', 'access_token', 'refresh_token',
                 'password', 'secret', 'proxy_authorization', 'environment', 'environ'}
    def inspect(value):
        if isinstance(value, dict):
            require(not forbidden.intersection(str(k).lower().replace('-', '_') for k in value),
                    'credential-bearing source cannot be exported')
            require(not value.get('headers'), 'header-bearing source cannot be exported')
            for child in value.values(): inspect(child)
        elif isinstance(value, list):
            for child in value: inspect(child)
    while todo:
        path = Path(todo.pop())
        require(not path.is_symlink(), 'symlink source cannot be exported')
        if path.is_dir(): continue
        require(path.is_file() and path.resolve() == path, 'canonical source file required')
        if str(path) in files: continue
        require(path.name not in ('.env', 'credentials'), 'credential file cannot be exported')
        raw = path.read_bytes()
        if path.name == 'headers.json':
            metadata = json.loads(raw)
            require(isinstance(metadata, dict) and set(metadata) <= {'content_type', 'http_status', 'request_id'},
                    'credential or arbitrary headers cannot be exported')
        require(not re.search(rb'(?i)(authorization\s*[:=]\s*["\x27]?bearer\s+|"(?:api_key|access_token|password)"\s*:)', raw),
                'credential-bearing source cannot be exported')
        if path.suffix == '.json':
            document = json.loads(raw)
            inspect(document)
            for reference in source_references(document, path, bindings):
                todo.append(pinned(reference))
        files[str(path)] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    return sorted(files.values(), key=lambda item: item['path'])


def verify_handoff(output):
    output = Path(output); manifest = read(output/'manifest.json')
    require(manifest['schema'] == 'saea-local-teacher-handoff-v2', 'handoff schema required')
    expected = {'manifest.json'}
    originals = set()
    for item in manifest['files']:
        path = output/item['path']
        require(path.resolve().is_relative_to(output.resolve()) and not path.is_symlink(), 'handoff path escapes')
        require(sha(path) == item['sha256'] == item['source_sha256'], 'handoff bytes changed')
        require(item['source_path'] not in originals, 'duplicate source')
        originals.add(item['source_path']); expected.add(item['path'])
    require({str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} == expected, 'handoff inventory differs')
    # Check every source pin resolves within this package, not back to sender files.
    index = {i['source_path']: i for i in manifest['files']}
    documents = {entry['source_path']: read(output/entry['path'])
                 for entry in manifest['files'] if entry['source_path'].endswith('.json')}
    bindings = {}
    for document in documents.values():
        if isinstance(document, dict):
            for key, value in profile_binding(document).items():
                require(key not in bindings or bindings[key] == value, 'conflicting profile bindings')
                bindings[key] = value
    for source, document in documents.items():
        for reference in source_references(document, source, bindings):
            require(reference['path'] in index
                    and index[reference['path']]['sha256'] == reference['sha256'],
                    'handoff source closure incomplete')
    return manifest


def export(config, job_id, output, *, source_runner=None):
    """Private replay inventory, no credentials and no automatic candidate promotion."""
    doc, root = load(config, export_runner=source_runner); job = next(j for j in doc['jobs'] if j['id'] == job_id)
    d = root/job_id
    with ExitStack() as stack:
        for guard_path in (d/'worker.guard', Path(job['prepared']['path']).parent/'local-queue.guard'):
            guard = stack.enter_context(guard_path.open('a+b'))
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(job_status(root, job) == 'complete', 'only completed collected work can be exported')
        collected = pinned(last_result(d)['collected'])
        sources = source_inventory(job, collected)
        output = Path(output); require(not output.exists(), 'fresh handoff directory required')
        output = private_dir(output); private_dir(output/'files')
        entries = []
        for i, entry in enumerate(sources):
            path = pinned(entry); name = 'files/'+str(i)
            fd = os.open(output/name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(path.read_bytes()); stream.flush(); os.fsync(stream.fileno())
            require(sha(output/name) == entry['sha256'] == sha(path), 'source changed during export')
            entries.append({'path': name, 'sha256': entry['sha256'], 'source_sha256': entry['sha256'], 'source_path': str(path)})
        manifest = {'schema': 'saea-local-teacher-handoff-v2', 'queue_id': doc['queue_id'],
                    'job_id': job_id, 'prepared_sha256': job['prepared']['sha256'], 'files': entries,
                    'exporter': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
                    'source_runner': source_runner,
                    'candidate_accepted': False, 'deployment_ready': False,
                    'meaning': 'complete prepared tree and recursive pinned source inventory; original-path map retained for replay'}
        persist(output/'manifest.json', manifest)
        return verify_handoff(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'run', 'worker', 'reconcile', 'status', 'export', 'pin-implementation', 'verify-handoff'])
    parser.add_argument('--config', type=Path); parser.add_argument('--cli', type=Path); parser.add_argument('--job'); parser.add_argument('--output', type=Path)
    parser.add_argument('--source-runner-pin', type=Path, help='export-only original runner file pin; never changes execution identity')
    args = parser.parse_args()
    require(args.source_runner_pin is None or args.action == 'export', 'source runner override is export-only')
    if args.action not in ('pin-implementation', 'verify-handoff'): require(args.config, 'queue config required')
    if args.action in ('worker', 'reconcile', 'export'): require(args.job, 'job ID required')
    if args.action == 'pin-implementation':
        require(args.cli and args.output, 'CLI and fresh output required'); result = implementation_inventory(args.cli); persist(args.output, result)
    elif args.action == 'verify-handoff':
        require(args.output, 'handoff directory required'); result = verify_handoff(args.output)
    elif args.action == 'export':
        require(args.output, 'private output directory required'); result = export(args.config, args.job, args.output, source_runner=read(args.source_runner_pin) if args.source_runner_pin else None)
    elif args.action == 'status':
        doc, root = load(args.config); result = {j['id']: job_status(root, j) for j in doc['jobs']}
    else: result = {'start': start, 'run': run, 'worker': worker, 'reconcile': reconcile}[args.action](args.config, *([args.job] if args.job else []))
    print(json.dumps(result, sort_keys=True))
