#!/usr/bin/env python3
"""Finite, durable queue over the existing pinned configured job launcher."""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

SCHEMA = 'saea-configured-job-queue-v1'


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


launcher = sibling('configured_job_launch')
binding = sibling('configured_preflight_binding')
require = launcher.require


def canonical_hash(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def pin_path(pin):
    require(isinstance(pin, dict) and set(pin) == {'path', 'sha256'}
            and isinstance(pin['sha256'], str) and re.fullmatch('[a-f0-9]{64}', pin['sha256']),
            'exact immutable file pin required')
    path = launcher.absolute(pin['path'])
    require(path.is_file() and launcher.digest(path) == pin['sha256'], 'queue input absent or changed')
    return path


def load(path):
    doc = json.loads(Path(path).read_bytes())
    required = {'schema', 'queue_id', 'state_dir', 'monitor_interval_seconds', 'jobs'}
    require(isinstance(doc, dict) and required <= set(doc) and not set(doc) - required - {'handoff_verifier'}
            and doc['schema'] == SCHEMA, 'exact queue configuration required')
    if 'handoff_verifier' in doc:pin_path(doc['handoff_verifier'])
    require(isinstance(doc['queue_id'], str) and re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}', doc['queue_id']), 'queue ID required')
    root = launcher.absolute(doc['state_dir']).resolve()
    require(len(root.parts) >= 4, 'owned scoped queue state directory required')
    require(doc['monitor_interval_seconds'] == 600, '600 second or terminal-event monitoring required')
    require(isinstance(doc['jobs'], list) and 0 < len(doc['jobs']) <= 1000, 'finite nonempty queue required')
    names, containers, receipts = set(), set(), set()
    for job in doc['jobs']:
        require(isinstance(job, dict) and set(job) == {'id', 'launch', 'preflight', 'after'}, 'exact queue job required')
        require(isinstance(job['id'], str) and re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}', job['id'])
                and job['id'] not in names, 'unique confined job ID required')
        names.add(job['id']); launch = launcher.load_config(pin_path(job['launch']))
        receipt = Path(launch['receipt_dir']).resolve()
        require(launch['container_name'] not in containers and receipt not in receipts
                and not root.is_relative_to(receipt) and not receipt.is_relative_to(root), 'distinct job and queue custody required')
        containers.add(launch['container_name']); receipts.add(receipt)
        if job['preflight'] is not None:pin_path(job['preflight'])
        if job['after'] is not None:
            dep = job['after']
            release = isinstance(dep, dict) and dep.get('condition') == 'generation-released'
            require(isinstance(dep, dict) and set(dep) == ({'condition', 'launch', 'container_id', 'release'} if release
                    else {'condition', 'launch', 'container_id'})
                    and dep['condition'] in ('container-exit-zero', 'generation-released')
                    and isinstance(dep['container_id'], str) and re.fullmatch('[a-f0-9]{64}', dep['container_id']),
                    'exact known terminal-container dependency required')
            producer = launcher.load_config(pin_path(dep['launch']))
            require(producer['container_name'] != launch['container_name'], 'self dependency refused')
            if release:
                require('handoff_verifier' in doc, 'pinned shared generation handoff verifier required')
                release_identity(producer, dep['release'])
    return doc


def atomic(path, doc):
    """Replace one state record atomically, with file and directory fsync."""
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temp.open('x') as handle:
        handle.write(json.dumps(doc, sort_keys=True, indent=2) + '\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def identities(path):
    result = {'config_sha256':launcher.digest(path), 'queue_sha256':launcher.digest(__file__),
        'launcher_sha256':launcher.digest(launcher.__file__), 'binding_sha256':launcher.digest(binding.__file__),
        'probe_sha256':launcher.digest(binding.probe.__file__)}
    doc = json.loads(Path(path).read_bytes())
    if 'handoff_verifier' in doc:result['handoff_sha256'] = launcher.digest(pin_path(doc['handoff_verifier']))
    return result


def release_helper(doc):
    path = pin_path(doc['handoff_verifier'])
    require(path.name == 'generation_release_handoff.py', 'shared generation handoff implementation required')
    spec = importlib.util.spec_from_file_location('queue_generation_handoff', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def release_identity(producer, release):
    require(isinstance(release, dict) and set(release) == {'path', 'logical_data', 'expected'}, 'exact generation release mapping required')
    expected = release['expected']
    require(isinstance(expected, dict) and set(expected) == {'candidate', 'run_id', 'attempt_id', 'study_commit',
        'profile', 'service_session', 'rows'}, 'complete producer generation identity required')
    for key in ('candidate', 'run_id', 'attempt_id', 'profile', 'service_session'):
        require(isinstance(expected[key], str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', expected[key]), 'confined generation identity required')
    require(type(expected['rows']) is int and expected['rows'] > 0 and expected['study_commit'] == producer['expected_commit'],
            'producer commit or row count differs')
    env = producer['environment']
    require(env.get('SAEA_OBSERVABILITY_RUN_ID') == expected['run_id']
            and env.get('SAEA_OBSERVABILITY_ATTEMPT_ID') == expected['attempt_id']
            and env.get('SAEA_DEPLOYMENT_PROFILE') == expected['profile'], 'producer launch run, attempt or deployment differs')
    logical = launcher.absolute(release['logical_data'])
    args = producer['argv']
    require(args.count('--out') == 1 and args.index('--out') + 1 < len(args), 'producer must declare exact release output directory')
    output = launcher.absolute(args[args.index('--out') + 1])
    actual, mount = binding.host_path(producer, str(output / 'generation-released.json'))
    require(not mount['readonly'] and actual.absolute() == launcher.absolute(release['path'])
            and actual.resolve() == actual.absolute(), 'producer release path or writable mapping differs')
    require(env.get('SPEECHRL_DATA_DIR', str(logical)) == str(logical), 'producer logical data mapping differs')
    if '--data' in args:
        require(args.count('--data') == 1 and args[args.index('--data') + 1] == str(logical), 'producer data argument differs')
    return actual, logical, expected


def verify_generation_release(doc, producer, release):
    """Reuse the shared seal verifier through explicit host mount translation."""
    path, logical, expected = release_identity(producer, release)
    require(path.is_file() and not path.is_symlink(), 'immutable generation release file required')
    blob = path.read_bytes(); value = json.loads(blob)
    require(set(value) == set(expected) | {'schema', 'generation_sealed', 'generation_lease_released',
        'scoring_complete', 'consumer_requires_own_lease_and_readiness', 'released_at_unix',
        'campaign_seconds_to_release', 'artifacts', 'seal_receipt'}, 'exact generation release schema required')
    started_path = path.with_name('started.json')
    require(started_path.is_file() and not started_path.is_symlink(), 'producer started receipt required')
    started_blob = started_path.read_bytes(); started = json.loads(started_blob)
    require(started.get('study_commit') == expected['study_commit'], 'producer startup commit differs')
    for key in ('candidate', 'job', 'attempt_id', 'rows'):
        if key in started:
            require(started[key] == expected['candidate' if key == 'job' else key], 'producer startup identity differs')
    mapped = copy.deepcopy(value); root = None
    files = {'generation_manifest':expected['run_id']+'.generation-manifest.json',
        'score_ledger_cells':'score-ledger-cells.json', 'score_jobs':'score-jobs.jsonl'}
    require(isinstance(value.get('artifacts'), dict) and set(value['artifacts']) == set(files), 'all generation sealed inputs required')
    for key, name in files.items():
        logical_path = logical / 'runs' / expected['run_id'] / expected['attempt_id'] / name
        pin = value['artifacts'][key]
        require(isinstance(pin, dict) and set(pin) == {'path', 'sha256'} and pin['path'] == str(logical_path)
                and value['seal_receipt'].get(key+'_path') == str(logical_path), 'generation seal logical path differs')
        physical, mount = binding.host_path(producer, str(logical_path))
        require(not mount['readonly'] and physical.absolute() == physical.resolve(), 'generation artifact host mapping redirected')
        candidate_root = physical.parents[3]
        if root is None:root = candidate_root
        require(root == candidate_root and physical == root / 'runs' / expected['run_id'] / expected['attempt_id'] / name,
                'generation artifact mapping must preserve runs layout')
        mapped['artifacts'][key]['path'] = str(physical)
        mapped['seal_receipt'][key+'_path'] = str(physical)
    class MappedDocument:
        def read_bytes(self):return json.dumps(mapped).encode()
    release_helper(doc).verify_release(MappedDocument(), data=root, expected=expected)
    require(path.read_bytes() == blob and started_path.read_bytes() == started_blob, 'generation release changed during verification')
    return {'path':str(path), 'sha256':hashlib.sha256(blob).hexdigest(),
        'started_sha256':hashlib.sha256(started_blob).hexdigest(), 'cpu_scoring_may_continue':True,
        'scoring_complete':False, 'consumer_requires_own_lease_and_readiness':True}


def wait_generation_event(doc, path):
    """Arm inotify before rechecking existence, closing publication races."""
    path = Path(path); directory = path.parent
    while not directory.exists():directory = directory.parent
    events = release_helper(doc).DirectoryEvents(directory)
    try:
        if path.exists():return
        events.wait(doc['monitor_interval_seconds'])
    finally:events.close()


def observe(pin):
    config = launcher.load_config(pin_path(pin))
    observed = launcher.status(config, canonical_hash(config), launcher.digest(launcher.__file__))
    # An inspect-only foreign/adopted container is not an attributable launch.
    intent = Path(config['receipt_dir']) / 'launch-intent.json'
    if observed['state'] != 'not-launched':
        require(intent.is_file(), 'configured container lacks its durable launch intent')
    if observed.get('container_id'):
        require(re.fullmatch('[a-f0-9]{64}', observed['container_id']), 'full immutable container ID required')
    return config, observed


def outcome(observed):
    if observed['state'] == 'exited' and observed.get('running') is False:
        return 'complete' if type(observed.get('exit_code')) is int and observed['exit_code'] == 0 else 'failed'
    if observed['state'] == 'running' and observed.get('running') is True:return 'running'
    return 'unknown'


def step(config_path, doc, state, save):
    require(identities(config_path) == state['identity'], 'queue configuration or implementation changed')
    load(config_path)  # Recheck every immutable input before any dispatch.
    if state['status'] in ('failed', 'complete'):return state
    index = len([j for j in state['jobs'] if j.get('status') == 'complete'])
    if index == len(doc['jobs']):
        state['status'] = 'complete'; save(); return state
    job, record = doc['jobs'][index], state['jobs'][index]
    if state['status'] == 'unknown' and not record.get('launch_requested'):return state
    if index:
        prior = doc['jobs'][index - 1]; receipt = state['jobs'][index - 1]
        _, observed = observe(prior['launch'])
        require(observed.get('container_id') == receipt.get('container_id') and outcome(observed) == 'complete',
                'prior queued container no longer matches successful terminal receipt')
    if job['after'] is not None:
        dep = job['after']; producer, observed = observe(dep['launch'])
        require(observed.get('container_id') == dep['container_id'], 'predecessor container identity differs')
        record['dependency'] = observed
        result = outcome(observed)
        if dep['condition'] == 'generation-released':
            release_path, _, _ = release_identity(producer, dep['release'])
            if release_path.exists():
                require(observed['state'] in ('running', 'exited'), 'producer is not a known running or terminal container')
                event = verify_generation_release(doc, producer, dep['release'])
                require(record.get('generation_release', event) == event, 'previous generation release bytes changed')
                record['generation_release'] = event; result = 'complete'
            elif result == 'complete':result = 'unknown'
            state['wait_release_path'] = str(release_path) if result == 'running' else None
        else:state['wait_release_path'] = None
        if result != 'complete':
            state['status'] = 'waiting' if result == 'running' else result
            state['wait_container_id'] = dep['container_id'] if result == 'running' else None
            save(); return state
    if record['status'] == 'failed':
        state['status'] = 'failed'; save(); return state
    launch_path = pin_path(job['launch']); launch_doc, observed = observe(job['launch'])
    previous_id = record.get('container_id')
    if previous_id:require(observed.get('container_id') == previous_id, 'queued container identity changed')
    if observed['state'] == 'not-launched':
        # If a launch was attempted, only the launcher intent can establish a
        # safe reconciliation; an unknown result must never start a new one.
        if record.get('launch_requested'):
            state['status'] = record['status'] = 'unknown'; save(); return state
        if job['preflight'] is not None:binding.verify(pin_path(job['preflight']), launch_path)
        launcher.execute(launch_path, 'layout')
        # Read-only preflight precedes our durable dispatch boundary.
        launcher.execute(launch_path, 'plan')
        record.update(status='launch-intent', launch_requested=True, requested_at=time.time())
        state['status'] = 'dispatching'; save()
        if job['preflight'] is None:observed = launcher.execute(launch_path, 'launch')
        else:observed = binding.execute(pin_path(job['preflight']), launch_path, 'launch')
    result = outcome(observed)
    if observed.get('container_id'):
        require(not previous_id or observed['container_id'] == previous_id, 'queued launch identity changed')
        record['container_id'] = observed['container_id']
    if observed.get('container_id'):
        require(re.fullmatch('[a-f0-9]{64}', observed['container_id']), 'full immutable launched container ID required')
    record.update(status=result, observed=observed)
    state['wait_container_id'] = record.get('container_id') if result == 'running' else None
    state['status'] = 'ready' if result == 'complete' else result
    if result == 'complete':
        record['result'] = {'container_exit_success':True, 'scientific_success':False,
            'verification':'verify shared runtime artifacts separately', 'completed_at':time.time()}
        if index + 1 == len(doc['jobs']):state['status'] = 'complete'
    save(); return state


def execute(path, mode):
    require(mode in ('plan', 'once', 'watch'), 'queue mode required')
    path = Path(path).resolve(); doc = load(path); identity = identities(path)
    if mode == 'plan':
        plans = []
        for job in doc['jobs']:
            launch = pin_path(job['launch'])
            if job['preflight'] is not None:binding.verify(pin_path(job['preflight']), launch)
            if job['after'] is not None:
                _, producer = observe(job['after']['launch'])
                require(producer.get('container_id') == job['after']['container_id'], 'planned predecessor identity differs')
            plans.append({'id':job['id'], 'launch':launcher.execute(launch, 'plan'), 'after':job['after']})
        return {'schema':SCHEMA, 'identity':identity, 'jobs':plans, 'new_model_calls':0,
            'dependency_policy':'serial-container-exit-zero-or-verified-generation-release', 'generation_release_supported':True}
    root = Path(doc['state_dir']); root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.guard').open('a+b') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / 'state.json'
        if state_path.exists():
            state = json.loads(state_path.read_bytes())
            require(state.get('schema') == SCHEMA and state.get('identity') == identity
                    and [j['id'] for j in state['jobs']] == [j['id'] for j in doc['jobs']], 'queue custody identity changed')
        else:
            state = {'schema':SCHEMA, 'identity':identity, 'queue_id':doc['queue_id'], 'status':'ready',
                'jobs':[{'id':j['id'], 'status':'pending'} for j in doc['jobs']], 'automatic_retry':False}
        def save():
            state['observed_at'] = time.time(); atomic(state_path, state)
        save()
        while True:
            try:step(path, doc, state, save)
            except Exception as error:
                state.update(status='unknown', error={'type':type(error).__name__, 'message':str(error)})
                save(); raise
            if mode == 'once' or state['status'] in ('failed', 'unknown', 'complete'):return state
            if state['status'] == 'ready':continue
            cid = state.get('wait_container_id')
            require(cid, 'running queue requires exact container identity')
            try:
                if state.get('wait_release_path'):
                    wait_generation_event(doc, state['wait_release_path']); continue
                waited = launcher.run(['docker', 'wait', cid], timeout=doc['monitor_interval_seconds'])
                require(waited.returncode == 0, 'Docker wait failed; queue held')
            except subprocess.TimeoutExpired:pass
            except Exception as error:
                state.update(status='unknown', error={'type':type(error).__name__, 'message':str(error)})
                save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'once', 'watch'))
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.config, args.mode), sort_keys=True))
