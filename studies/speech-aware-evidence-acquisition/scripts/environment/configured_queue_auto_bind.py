#!/usr/bin/env python3
"""Bind a successor to a real predecessor queue container; honor a dispatch hold."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time


def load_module(path):
    spec = importlib.util.spec_from_file_location('pinned_queue', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_pin(pin):
    if set(pin) != {'path', 'sha256'}:
        raise ValueError('exact pin required')
    path = Path(pin['path'])
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != pin['sha256']:
        raise ValueError('pinned input changed')
    return json.loads(blob)


def write_new(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if Path(path).read_bytes() != data:
            raise ValueError('existing output differs')
        return
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def step(config, queue):
    fragment = read_pin(config['dependency'])
    generation = fragment['after_without_container_id'].get('condition') == 'generation-released'
    if config['source_binding'] is not None:
        upstream = read_pin(config['source_binding'])
        status_path = Path(upstream['state_dir']) / 'status.json'
        if not status_path.exists(): return 'waiting-source'
        status = json.loads(status_path.read_bytes())
        if status.get('status') in ('unknown', 'held') or (status.get('status') == 'failed' and not generation):
            raise ValueError('upstream binder failed, held or unknown')
        if not status.get('bound_queue_sha256'): return 'waiting-source'
        if status.get('config_sha256') != config['source_binding']['sha256']:
            raise ValueError('upstream binder identity differs')
        if config['source_queue']['path'] != upstream['bound_queue']:
            raise ValueError('upstream bound path differs')
        source = read_pin({'path':upstream['bound_queue'], 'sha256':status['bound_queue_sha256']})
    else:
        source = read_pin(config['source_queue'])
    ready = read_pin(config['successor_queue'])
    source_path = Path(source['state_dir']) / 'state.json'
    if not source_path.exists(): return 'waiting-source'
    state = json.loads(source_path.read_bytes())
    if state.get('identity') != queue.identities(Path(config['source_queue']['path'])) or state.get('queue_id') != source['queue_id']:
        raise ValueError('source queue identity differs')
    jobs = [j for j in source['jobs'] if j['id'] == config['source_job']]
    records = [j for j in state['jobs'] if j['id'] == config['source_job']]
    if len(jobs) != 1 or len(records) != 1 or jobs[0]['launch'] != fragment['after_without_container_id']['launch']:
        raise ValueError('source job or launch differs')
    record = records[0]
    if (record.get('status') in ('unknown', 'held') or state.get('status') in ('unknown', 'held')
            or (not generation and (record.get('status') == 'failed' or state.get('status') == 'failed'))):
        raise ValueError('source failed or unknown')
    cid = record.get('container_id')
    if cid is None:
        if record.get('launch_requested') and record.get('status') not in ('launch-intent',):
            raise ValueError('source launch identity unresolved')
        return 'waiting-container'
    if not isinstance(cid, str) or not re.fullmatch('[a-f0-9]{64}', cid):
        raise ValueError('source container id invalid')
    dep = dict(fragment['after_without_container_id'], container_id=cid)
    _, observed = queue.observe(dep['launch'], dep.get('launcher_artifact'))
    if observed.get('container_id') != cid or observed.get('state') not in ('running', 'exited'):
        raise ValueError('source container or launch receipt differs')
    if len(ready['jobs']) != 1 or ready['jobs'][0]['after'] is not None:
        raise ValueError('fresh single-job successor required')
    ready['handoff_verifier'] = fragment['handoff_verifier']
    ready['jobs'][0]['after'] = dep
    output = Path(config['bound_queue'])
    write_new(output, ready)
    queue.execute(output, 'plan')
    if Path(config['hold_path']).exists(): return 'held'
    result = queue.execute(output, 'once')
    return result['status']


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    args = parser.parse_args(); path = Path(args.config)
    config = json.loads(path.read_bytes())
    expected = {'schema','queue_tool','source_queue','source_binding','source_job','dependency','successor_queue','bound_queue','hold_path','state_dir','interval_seconds'}
    if set(config) != expected or config['schema'] != 'saea-configured-queue-auto-bind-v1' or config['interval_seconds'] != 600:
        raise ValueError('exact auto-bind configuration required')
    tool = Path(config['queue_tool']['path'])
    if hashlib.sha256(tool.read_bytes()).hexdigest() != config['queue_tool']['sha256']:
        raise ValueError('queue tool changed')
    queue = load_module(tool); root = Path(config['state_dir']); root.mkdir(parents=True, exist_ok=True)
    with (root / 'guard').open('a+b') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                status = step(config, queue)
                bound = Path(config['bound_queue'])
                queue.atomic(root / 'status.json', {'status':status, 'observed_at':time.time(), 'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'bound_queue_sha256': hashlib.sha256(bound.read_bytes()).hexdigest() if bound.is_file() else None})
                if status in ('complete','failed','unknown'): return 0 if status == 'complete' else 1
            except Exception as error:
                # Preserve only an identity recorded by a previous successful step.
                previous = {}
                try:
                    previous = json.loads((root / 'status.json').read_bytes())
                except (OSError, ValueError):
                    pass
                config_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                bound = Path(config['bound_queue'])
                bound_hash = previous.get('bound_queue_sha256')
                if (previous.get('config_sha256') != config_hash or not bound_hash
                        or not bound.is_file() or hashlib.sha256(bound.read_bytes()).hexdigest() != bound_hash):
                    bound_hash = None
                queue.atomic(root / 'status.json', {'status':'failed','error_type':type(error).__name__,'message':str(error),'observed_at':time.time(),
                    'config_sha256':config_hash,'bound_queue_sha256':bound_hash})
                return 1
            time.sleep(600)


if __name__ == '__main__': raise SystemExit(main())
