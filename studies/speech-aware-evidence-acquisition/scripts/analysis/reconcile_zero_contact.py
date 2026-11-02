"""Preserve an exact stale lease after terminal, zero-contact reconciliation."""
import argparse
from dataclasses import fields, replace
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

SCHEMA = 'saea-zero-contact-reconciliation-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def pinned(pin):
    require(isinstance(pin, dict) and set(pin) == {'path', 'sha256'}, 'exact file pin required')
    path = Path(pin['path'])
    require(path.is_absolute() and path.is_file(), 'absolute existing pinned file required')
    blob = path.read_bytes()
    require(digest(blob) == pin['sha256'], 'pinned evidence changed')
    return json.loads(blob)


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    sync(path.parent)


def sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def replay_source(repo, data, registered, target, arm):
    sys.path[:0] = [str(repo / 'src'), str(repo / 'scripts/analysis')]
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
    allowed = {f.name for f in fields(ExecutionPlan)}
    source = ExecutionPlan(**{k: v for k, v in registered['execution_plan'].items() if k in allowed},
                           owner_allowance=registered.get('owner_allowance'))
    source.validate(source_root=repo)
    replay = NativeContactReplay(data, source, replace(source, attempt_id=target), arm)
    require(not replay.request_ids and replay.source_actual_usage == {'calls': 0, 'audio_seconds': 0.0},
            'native replay is not zero-contact')
    return replay.source_binding()


def quiet(ports):
    from tfgrpo_recovery_queue import quiescence
    return quiescence(ports)


def empty_tree(path):
    require(not path.is_symlink(), 'custody path may not be a symlink')
    if path.exists():
        require(path.is_dir(), 'custody directory required')
        require(not any(p.is_symlink() or not p.is_dir() for p in path.rglob('*')),
                'native or FIT custody exists; zero-contact reconciliation refused')


def reconcile(config, *, observe=quiet, replay=replay_source, pid_exists=None):
    require(set(config) == {'schema', 'repo', 'data', 'out', 'source_plan', 'baseline', 'started', 'container_inspect',
        'container_id', 'container_name', 'lease_sha256', 'target_attempt', 'arm', 'fit', 'ports'}
        and config['schema'] == SCHEMA, 'exact zero-contact configuration required')
    repo, data, out = (Path(config[k]).resolve() for k in ('repo', 'data', 'out'))
    require(out.is_relative_to(data / 'work/codex') and out != data / 'work/codex' and not out.exists(),
            'fresh owned reconciliation output required')
    require(config['ports'] == [8200, 8201], 'explicit two-instance topology required')
    require(re.fullmatch('[0-9a-f]{64}', config['container_id']) is not None, 'exact container ID required')
    require(isinstance(config['fit'], dict) and set(config['fit']) == {'run_id', 'attempt_id'}, 'exact FIT identity required')
    registered = pinned(config['source_plan']); started = pinned(config['started'])
    baseline = pinned(config['baseline'])
    fit = baseline.get('live_phase_fit', baseline.get('native_contact_fit'))
    require(isinstance(fit, dict) and all(fit.get(k) == v for k, v in config['fit'].items())
            and baseline.get('arm') == config['arm'], 'registered baseline FIT or arm differs')
    plan = registered['execution_plan']; run = plan['run_id']; attempt = plan['attempt_id']
    for value in (run, attempt, config['target_attempt'], *config['fit'].values()):
        require(isinstance(value, str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', value), 'confined identity required')
    require(attempt != config['target_attempt'], 'fresh target attempt required')
    require(type(started['pid']) is int and started['pid'] > 0, 'exact started PID required')
    pid_exists = pid_exists or (lambda pid: Path('/proc', str(pid)).exists())
    guard = data / 'locks/generation.guard'; metadata = data / 'locks/flight.lock'
    require(guard.is_file() and not guard.is_symlink(), 'existing stable generation guard required')
    with guard.open('r+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not out.exists(), 'existing reconciliation output')
        original = metadata.read_bytes()
        require(digest(original) == config['lease_sha256'], 'lease bytes changed')
        lease = json.loads(original)
        require(lease.get('schema') == 'saea-generation-lease-v1' and all(lease.get(k) == v for k, v in {
            'run_id': run, 'attempt_id': attempt, 'pid': started['pid'], 'study_head': started['study_commit']}.items()),
            'lease does not belong to exact source worker')
        guard_identity = os.fstat(handle.fileno())

        def audit():
            container, = pinned(config['container_inspect'])
            state = container['State']
            require(container['Id'] == config['container_id'] and container['Name'] == '/' + config['container_name']
                    and state['Status'] in ('exited', 'dead') and state['Running'] is False
                    and state['Pid'] == 0 and state.get('FinishedAt') and not state.get('Restarting', False),
                    'exact container is not terminal')
            begin = datetime.fromisoformat(state['StartedAt'].replace('Z', '+00:00')).timestamp()
            end = datetime.fromisoformat(state['FinishedAt'].replace('Z', '+00:00')).timestamp()
            require(begin <= started['time'] <= end, 'started evidence is outside exact container lifetime')
            env = dict(row.split('=', 1) for row in container['Config']['Env'])
            require(env.get('SAEA_OBSERVABILITY_RUN_ID') == run
                    and env.get('SAEA_OBSERVABILITY_ATTEMPT_ID') == attempt,
                    'container does not bind exact source run and attempt')
            require(not pid_exists(started['pid']), 'source PID exists; reuse requires explicit identity audit')
            source = json.loads((data / 'attempts' / run / (attempt + '.json')).read_bytes())
            require(source.get('state') == 'ABORTED' and source.get('run_id') == run
                    and source.get('attempt_id') == attempt and source.get('actual') == {'calls': 0, 'audio_seconds': 0.0},
                    'source must be ABORTED with exact zero usage')
            empty_tree(data / 'contact-journals' / run / attempt)
            empty_tree(data / 'prepared' / run / attempt)
            fit = config['fit']
            require(not (data / 'attempts' / fit['run_id'] / (fit['attempt_id'] + '.json')).exists(), 'FIT attempt exists')
            for base in ('prepared', 'contact-journals', 'runs'):
                empty_tree(data / base / fit['run_id'] / fit['attempt_id'])
            target = config['target_attempt']
            require(not (data / 'attempts' / run / (target + '.json')).exists()
                    and not (data / 'runs' / run / target).exists()
                    and not (data / 'contact-journals' / run / target).exists(), 'target attempt already exists')
            require(pinned(config['source_plan']) == registered and pinned(config['started']) == started
                    and pinned(config['baseline']) == baseline,
                    'source evidence changed')
            return container, replay(repo, data, registered, target, config['arm'])

        container, binding = audit()
        quiescence = observe(config['ports'])
        _, rechecked = audit()
        require(rechecked == binding and metadata.read_bytes() == original, 'source or lease changed during audit')
        current_guard = guard.stat()
        require((current_guard.st_dev, current_guard.st_ino) == (guard_identity.st_dev, guard_identity.st_ino),
                'stable generation guard was replaced')
        out.mkdir(parents=True, exist_ok=False)
        destination = metadata.with_name('preserved-flight-' + uuid.uuid4().hex + '.json')
        evidence = {'schema': SCHEMA, 'source': binding, 'container': container,
            'started': started, 'quiescence': quiescence, 'config': config,
            'preserved_lease_path': str(destination),
            'lease_sha256': digest(original), 'new_model_calls': 0, 'source_state_changed': False,
            'target_attempt_created': False, 'guard_preserved': True, 'observed_at': time.time()}
        save(out / 'reconciliation.json', evidence)
        require(not destination.exists() and metadata.read_bytes() == original, 'lease changed before preservation')
        os.rename(metadata, destination)
        sync(out); sync(metadata.parent)
        save(out / 'complete.json', {'schema': SCHEMA, 'lease_sha256': digest(original), 'lease_preserved': True,
                                    'preserved_lease_path': str(destination),
                                    'new_model_calls': 0, 'recovery_dispatch_authorized': False})
        return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    sys.path[:0] = [str(Path(config['repo']) / 'src'), str(Path(config['repo']) / 'scripts/analysis')]
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    print(json.dumps(reconcile(config), sort_keys=True))
