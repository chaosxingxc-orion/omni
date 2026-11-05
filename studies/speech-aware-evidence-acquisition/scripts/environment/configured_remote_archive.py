#!/usr/bin/env python3
"""Private developer-side snapshots of configured remote state; never restore."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import time
import uuid

SCHEMA = 'saea-configured-remote-archive-v1'
REMOTE_BASE = '/var/lib/omni-local-state'
PREFIX = 'SAEA_ARCHIVE_INVENTORY='


class TransportError(Exception):
    """An SSH transport failure, not a remote validation failure."""


REMOTE = '''import json, os, subprocess, sys
from pathlib import Path
spec=json.loads(sys.argv[1]); base=Path(spec['base']); owned=base/spec['owned_root']
if owned.is_symlink(): raise ValueError('owned root must not be a symlink')
present=[]; absent=[]
for name in spec['paths']:
    p=base/name
    if p.is_symlink(): raise ValueError('archive root must not be a symlink: '+name)
    if not p.exists(): absent.append(name); continue
    if not p.resolve().is_relative_to(base.resolve()): raise ValueError('archive path escapes local state')
    if name not in spec['attempts'] and not p.resolve().is_relative_to(owned.resolve()):
        raise ValueError('archive path escapes its owned root')
    if name in spec['attempts'] and p.resolve() != base.resolve()/name:
        raise ValueError('attempt path must not contain symlink aliases')
    if not (p.is_dir() or p.is_file()): raise ValueError('archive root is not file or directory')
    if name in spec['attempts'] and not p.is_file(): raise ValueError('attempt must be a regular file')
    present.append(name)
meta={'included':present,'absent':absent,
      'attempts_included':[p for p in spec['attempts'] if p in present],
      'attempts_absent':[p for p in spec['attempts'] if p in absent]}
print('SAEA_ARCHIVE_INVENTORY='+json.dumps(meta,sort_keys=True),file=sys.stderr,flush=True)
args=['tar','-C',str(base),'-cf','-']
args += ['--',*present] if present else ['--files-from=/dev/null']
sys.exit(subprocess.call(args))
'''


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value, absolute=False):
    require(isinstance(value, str) and value and not any(c in value for c in ('\x00', '\n', '\r')),
            'clean path required')
    path = PurePosixPath(value)
    require(path.is_absolute() == absolute and str(path) == value and '..' not in path.parts
            and value not in ('.', '/') and not value.startswith('//'), 'canonical scoped path required')
    return path


def load_config(path):
    doc = json.loads(Path(path).read_bytes())
    fields = {'schema', 'host', 'container_name', 'remote_root', 'relative_paths',
              'shared_attempt_paths', 'local_output_root', 'interval_seconds'}
    require(isinstance(doc, dict) and fields <= set(doc)
            and not set(doc) - fields - {'expected_container_id'}, 'exact archive schema required')
    if 'expected_container_id' in doc:
        require(isinstance(doc['expected_container_id'], str)
                and re.fullmatch('[a-f0-9]{64}', doc['expected_container_id']), 'full container ID required')
    require(doc['schema'] == SCHEMA and doc['host'] == 'npu2', 'only npu2 archive host is permitted')
    require(isinstance(doc['container_name'], str) and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}', doc['container_name']),
            'exact container name required')
    root = canonical(doc['remote_root'], True)
    require(root.parent == PurePosixPath(REMOTE_BASE), 'remote root must be one owned local-state directory')
    canonical(doc['local_output_root'], True)
    require(type(doc['interval_seconds']) is int and doc['interval_seconds'] == 600, 'archive interval must be 600 seconds')
    require(isinstance(doc['relative_paths'], list) and doc['relative_paths'], 'explicit archive paths required')
    paths = [canonical(p) for p in doc['relative_paths']]
    require(len(set(paths)) == len(paths) and not any(a != b and a.is_relative_to(b) for a in paths for b in paths),
            'archive paths must be unique and non-overlapping')
    require(isinstance(doc['shared_attempt_paths'], list), 'explicit attempt inventory required')
    attempts = [canonical(p, True) for p in doc['shared_attempt_paths']]
    require(len(set(attempts)) == len(attempts) and all(p.is_relative_to(REMOTE_BASE) and 'attempts' in p.parts
            and p.suffix == '.json' for p in attempts), 'shared attempts must be unique JSON paths under local state')
    all_paths = [root / p for p in paths] + attempts
    require(len(set(all_paths)) == len(all_paths) and not any(a != b and a.is_relative_to(b) for a in all_paths for b in all_paths),
            'shared attempts must not overlap archive roots')
    return doc


def digest(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def specification(doc):
    base = PurePosixPath(REMOTE_BASE)
    attempts = [str(PurePosixPath(p).relative_to(base)) for p in doc['shared_attempt_paths']]
    return {'base': REMOTE_BASE, 'owned_root': str(PurePosixPath(doc['remote_root']).relative_to(base)),
            'paths': [str((PurePosixPath(doc['remote_root']) / p).relative_to(base))
            for p in doc['relative_paths']] + attempts, 'attempts': attempts}


def ssh(doc, argv):
    return ['ssh', '-oBatchMode=yes', '-oConnectTimeout=30', doc['host'], shlex.join(argv)]


def archive_command(doc):
    return ssh(doc, ['sudo', '-n', 'python3', '-c', REMOTE, json.dumps(specification(doc), sort_keys=True)])


def persist(path, doc):
    with Path(path).open('x') as handle:
        json.dump(doc, handle, sort_keys=True, indent=2)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    descriptor = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def inspect(doc):
    try:
        result = subprocess.run(ssh(doc, ['sudo', '-n', 'docker', 'inspect', '--type', 'container', doc['container_name']]),
                                capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as error:
        raise TransportError('container inspection SSH timeout') from error
    if result.returncode == 255:
        raise TransportError('container inspection SSH exit 255: ' + result.stderr[-1000:])
    require(result.returncode == 0, 'container inspection failed: ' + result.stderr[-1000:])
    container, = json.loads(result.stdout)
    require(container['Name'] == '/' + doc['container_name'], 'container name differs')
    state = container['State']
    return {'container_id': container['Id'], 'state': state['Status'], 'running': state['Running'],
            'exit_code': state['ExitCode'], 'finished_at': state['FinishedAt']}


def verify_inventory(path, stderr, spec):
    rows = [line[len(PREFIX):] for line in stderr.splitlines() if line.startswith(PREFIX)]
    require(len(rows) == 1, 'missing or ambiguous inventory; partial archive retained')
    meta = json.loads(rows[0])
    require(set(meta) == {'included', 'absent', 'attempts_included', 'attempts_absent'}, 'exact inventory required')
    for key in meta:
        require(isinstance(meta[key], list) and all(isinstance(p, str) for p in meta[key])
                and len(meta[key]) == len(set(meta[key])), 'invalid inventory list')
    require(set(meta['included']).isdisjoint(meta['absent'])
            and set(meta['included'] + meta['absent']) == set(spec['paths']), 'inventory does not cover configured paths')
    require(meta['attempts_included'] == [p for p in spec['attempts'] if p in meta['included']]
            and meta['attempts_absent'] == [p for p in spec['attempts'] if p in meta['absent']], 'attempt inventory differs')
    seen = set(); attempts = []; count = 0
    with tarfile.open(path, mode='r|') as archive:
        for member in archive:
            name = str(canonical(member.name.rstrip('/')))
            require(name not in seen and any(PurePosixPath(name).is_relative_to(p) for p in meta['included']),
                    'duplicate or unconfigured archive member')
            require(not any(PurePosixPath(name).is_relative_to(p) for p in meta['absent']), 'absent path appears in archive')
            seen.add(name); count += 1
            if name in meta['attempts_included']:
                require(member.isfile() and member.size <= 16 * 1024 * 1024, 'attempt must be bounded regular JSON')
                value = json.load(archive.extractfile(member))
                require(isinstance(value, dict), 'attempt record must be a JSON object')
                attempts.append(name)
    require(set(meta['included']) <= seen and set(attempts) == set(meta['attempts_included']), 'declared archive roots or attempts missing')
    return {**meta, 'members': count, 'attempt_records_verified': sorted(attempts), 'inventory_verified': True}


def snapshot(doc, observed):
    output = Path(doc['local_output_root'])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory = output / (stamp + '-' + uuid.uuid4().hex[:12])
    directory.mkdir(mode=0o700)
    archive_path = directory / 'state.tar'
    record = {'schema': SCHEMA, 'config_sha256': digest(doc),
              'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'container': observed, 'archive': 'state.tar', 'inventory_verified': False,
              'automatic_restore_authority': False, 'new_model_calls': 0}
    persist(directory / 'intent.json', {**record, 'config': doc})
    checksum = hashlib.sha256(); size = 0; process = None
    try:
        with (directory / 'stderr.log').open('xb') as errors, archive_path.open('xb') as target:
            process = subprocess.Popen(archive_command(doc), stdout=subprocess.PIPE, stderr=errors)
            try:
                while True:
                    chunk = process.stdout.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk); checksum.update(chunk); size += len(chunk)
                record['tar_exit_code'] = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate(); process.wait()
                process.stdout.close()
                target.flush(); os.fsync(target.fileno())
                errors.flush(); os.fsync(errors.fileno())
        stderr = (directory / 'stderr.log').read_text(errors='replace')
        if record['tar_exit_code'] == 255:
            raise TransportError('archive SSH exit 255: ' + stderr[-1000:])
        record.update(verify_inventory(archive_path, stderr, specification(doc)))
        record['status'] = {0: 'complete', 1: 'changed-during-read'}.get(record['tar_exit_code'], 'failed')
        record['consistent_snapshot'] = False  # tar is not a transactional snapshot, even at exit 0.
    except (KeyboardInterrupt, SystemExit) as error:
        record.update(status='interrupted', error_type=type(error).__name__)
        raise
    except (TransportError, subprocess.TimeoutExpired) as error:
        record.update(status='failed', failure_kind='transport', error_type=type(error).__name__, error=str(error))
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        record.update(bytes=size, sha256=checksum.hexdigest())
        persist(directory / 'result.json', record)
    return {**record, 'directory': str(directory)}


def wait_event(doc):
    # Only a read-only wait is retried. A timeout never proves generation ended.
    started = time.monotonic()
    outcome = 'terminal-event'
    try:
        result = subprocess.run(ssh(doc, ['sudo', '-n', 'timeout', '--signal=TERM',
                                '--kill-after=5s', '600', 'docker', 'wait', doc['container_name']]),
                                capture_output=True, text=True, timeout=630)
        require(result.returncode in (0, 124, 137, 255),
                'container wait failed: ' + result.stderr[-1000:])
        if result.returncode != 0:
            outcome = 'transport-unavailable' if result.returncode == 255 else 'interval-elapsed'
    except subprocess.TimeoutExpired:
        outcome = 'local-wait-timeout'
    if outcome != 'terminal-event':
        print(json.dumps({'event': 'archive-wait-observation', 'outcome': outcome,
                          'terminal_verified': False}), flush=True)
        # An early transport failure must not turn the 600-second watch into a spin.
        remaining = doc['interval_seconds'] - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
    # Caller re-inspects and checks the original container ID before any snapshot.


def bind_container(doc, observed):
    """Called under the existing archive guard, before reading any remote bytes.

    Legacy configurations bind on first observation. Operators should supply an
    expected ID when adopting an existing watcher; a name is not its history.
    """
    identity = observed.get('container_id')
    require(isinstance(identity, str) and re.fullmatch('[a-f0-9]{64}', identity),
            'full observed container ID required')
    require(doc.get('expected_container_id', identity) == identity,
            'container replaced: expected ID differs')
    record = {'schema': 'saea-archive-container-identity-v1', 'host': doc['host'],
              'remote_root': doc['remote_root'], 'container_name': doc['container_name'],
              'container_id': identity, 'config_sha256': digest(doc)}
    path = Path(doc['local_output_root']) / (doc['container_name'] + '.identity.json')
    require(not path.is_symlink(), 'archive identity cannot be a symlink')
    if path.exists():
        require(path.is_file() and json.loads(path.read_bytes()) == record,
                'container replaced or persistent archive identity/config differs')
        return
    # File and directory durability: a crash before publication leaves a private
    # partial file; it never replaces or repairs an existing identity.
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(record, handle, sort_keys=True); handle.write('\n')
        handle.flush(); os.fsync(handle.fileno())
    os.link(temp, path)  # no replacement even if a non-cooperating writer races
    temp.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def execute(config, mode):
    require(mode in ('once', 'watch'), 'archive mode must be once or watch')
    doc = load_config(config)
    output = Path(doc['local_output_root'])
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(output.is_dir() and not output.is_symlink() and output.stat().st_mode & 0o077 == 0,
            'archive output root must be private (0700) and not a symlink')
    guard = output / (doc['container_name'] + '.archive.guard')
    with guard.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = None
        while True:
            try:
                observed = inspect(doc)
            except TransportError as error:
                if mode == 'once':
                    raise
                event = {'schema': SCHEMA, 'status': 'failed', 'failure_kind': 'transport',
                         'operation': 'inspect', 'error': str(error), 'config_sha256': digest(doc),
                         'container_id': identity, 'retry_seconds': doc['interval_seconds']}
                persist(output / ('transport-' + uuid.uuid4().hex + '.json'), event)
                print(json.dumps(event, sort_keys=True), flush=True)
                time.sleep(doc['interval_seconds'])
                continue
            bind_container(doc, observed)
            require(identity is None or identity == observed['container_id'], 'container replaced during archive watch')
            identity = observed['container_id']
            record = snapshot(doc, observed)
            print(json.dumps(record, sort_keys=True), flush=True)
            if mode == 'once':
                return record
            if record['status'] == 'failed':
                if record.get('failure_kind') != 'transport':
                    return record
                time.sleep(doc['interval_seconds'])
                continue
            if observed['state'] in ('exited', 'dead') and not observed['running']:
                return record
            require(observed['state'] in ('running', 'paused', 'restarting'), 'container is not running or terminal')
            wait_event(doc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('once', 'watch'))
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.config, args.mode)
    raise SystemExit(1 if result['status'] == 'failed' else 0)
