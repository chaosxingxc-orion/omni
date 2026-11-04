#!/usr/bin/env python3
"""Finite, durable queue over the existing pinned configured job launcher."""
import argparse
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


def predecessor_runner(pin):
    """Bind historical launch identity without importing or executing old code.

    The reviewed queue config, not the receipt under observation, supplies this
    immutable artifact pin. Current read-only Docker/custody checks remain in use.
    """
    path = pin_path(pin)
    require(path.name == 'configured_job_launch.py' and not path.is_symlink()
            and path.absolute() == path.resolve(), 'regular historical launcher artifact required')
    return pin['sha256']


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
        fields = {'id', 'launch', 'preflight', 'after'}
        require(isinstance(job, dict) and fields <= set(job) <= fields | {
            'depends_on', 'after_terminal', 'on_fail', 'generation_release',
            'after_handoff', 'unless_generation_released'}, 'exact queue job required')
        require(job.get('on_fail', 'stop') in ('stop', 'continue'), 'explicit queue failure policy required')
        require(isinstance(job['id'], str) and re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}', job['id'])
                and job['id'] not in names, 'unique confined job ID required')
        names.add(job['id']); launch = launcher.load_config(pin_path(job['launch']))
        if 'generation_release' in job:
            require('handoff_verifier' in doc, 'pinned shared generation handoff verifier required')
            release_identity(launch, job['generation_release'])
        receipt = Path(launch['receipt_dir']).resolve()
        require(launch['container_name'] not in containers and receipt not in receipts
                and not root.is_relative_to(receipt) and not receipt.is_relative_to(root), 'distinct job and queue custody required')
        containers.add(launch['container_name']); receipts.add(receipt)
        if job['preflight'] is not None:pin_path(job['preflight'])
        if job['after'] is not None:
            dep = job['after']
            release = isinstance(dep, dict) and dep.get('condition') == 'generation-released'
            require(isinstance(dep, dict) and set(dep) - {'launcher_artifact'} == ({'condition', 'launch', 'container_id', 'release'} if release
                    else {'condition', 'launch', 'container_id'})
                    and dep['condition'] in ('container-exit-zero', 'generation-released')
                    and isinstance(dep['container_id'], str) and re.fullmatch('[a-f0-9]{64}', dep['container_id']),
                    'exact known terminal-container dependency required')
            if 'launcher_artifact' in dep:predecessor_runner(dep['launcher_artifact'])
            producer = launcher.load_config(pin_path(dep['launch']))
            require(producer['container_name'] != launch['container_name'], 'self dependency refused')
            if release:
                require('handoff_verifier' in doc, 'pinned shared generation handoff verifier required')
                release_identity(producer, dep['release'])
    if any('depends_on' in job for job in doc['jobs']):
        require(all('depends_on' in job for job in doc['jobs']), 'graph jobs must all declare dependencies')
        graph = {}; by_id = {j['id']: j for j in doc['jobs']}
        for job in doc['jobs']:
            deps, terminal = job['depends_on'], job.get('after_terminal', [])
            for edges in (deps, terminal):
                require(isinstance(edges, list) and all(isinstance(dep, str) for dep in edges)
                    and len(set(edges)) == len(edges) and set(edges) <= names, 'unique known dependencies required')
            require(not set(deps) & set(terminal), 'dependency conditions must not overlap')
            extra = set()
            if 'unless_generation_released' in job:
                producer = job['unless_generation_released']
                require(isinstance(producer, str) and producer in by_id
                    and 'generation_release' in by_id[producer]
                    and terminal == [producer] and deps == []
                    and 'generation_release' not in job and 'after_handoff' not in job,
                    'conditional cleanup requires its exact terminal producer')
                extra.add(producer)
            if 'after_handoff' in job:
                handoff = job['after_handoff']
                require(isinstance(handoff, dict) and set(handoff) == {'producer', 'settlement'}
                    and all(isinstance(v, str) and v in by_id for v in handoff.values()),
                    'exact known handoff producer and settlement required')
                producer, cleanup = by_id[handoff['producer']], by_id[handoff['settlement']]
                require('generation_release' in producer
                    and cleanup.get('unless_generation_released') == producer['id']
                    and cleanup.get('after_terminal') == [producer['id']]
                    and not set(handoff.values()) & (set(deps) | set(terminal)),
                    'handoff must select corresponding conditional settlement or release')
                extra.update(handoff.values())
            graph[job['id']] = set(deps) | set(terminal) | extra
        done = set()
        while graph:
            ready = {key for key, deps in graph.items() if deps <= done}
            require(ready, 'queue dependency cycle')
            done.update(ready)
            graph = {key: deps for key, deps in graph.items() if key not in ready}
    else:
        require(all(job.get('on_fail', 'stop') == 'stop' for job in doc['jobs']),
                'continue requires an explicit dependency graph')
        require(not any('after_terminal' in job for job in doc['jobs']), 'terminal dependencies require graph mode')
        require(not any(set(job) & {'generation_release', 'after_handoff', 'unless_generation_released'}
                        for job in doc['jobs']), 'generation handoff requires graph mode')
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


def producer_output(producer, logical, expected):
    """Resolve one supported command without evaluating or rewriting its inputs."""
    args = producer['argv']
    if '--config' not in args:
        require(args.count('--out') == 1 and args.index('--out') + 1 < len(args),
                'producer must declare exact release output directory')
        if '--data' in args:
            require(args.count('--data') == 1 and args.index('--data') + 1 < len(args)
                and args[args.index('--data') + 1] == str(logical), 'producer data argument differs')
        return launcher.absolute(args[args.index('--out') + 1]), None
    require(len(args) == 4 and args[2] == '--config'
        and Path(args[1]).name == 'learning_strategy_job.py', 'exact supported producer config entry required')
    config_path, mount = binding.host_path(producer, args[3])
    require(mount['readonly'] and config_path.absolute() == config_path.resolve()
        and not config_path.is_symlink(), 'producer command requires immutable read-only mapping')
    pins = [p for p in producer['required_inputs'] if p['path'] == str(config_path)]
    require(len(pins) == 1 and pin_path(pins[0]) == config_path, 'producer command must be pinned exactly once')
    config = json.loads(config_path.read_bytes())
    require(isinstance(config, dict) and set(config) == {'schema','mode','repo','data','arguments'}
        and config['schema'] == 'saea-learning-strategy-command-v1' and config['mode'] == 'run',
        'exact learning producer run command required')
    values = config['arguments']
    require(isinstance(values, dict) and set(values) == {'job','out','after','predecessor_run'},
        'exact learning producer arguments required')
    for value in [config['repo'],config['data'],values['job'],values['out'],values['after']]:
        require(str(launcher.absolute(value)) == value, 'canonical learning producer path required')
    require(config['data'] == str(logical)
        and args[1] == str(Path(config['repo'])/'scripts/analysis/learning_strategy_job.py'),
        'learning producer repository or data differs')
    repo, repo_mount = binding.host_path(producer, config['repo'])
    require(repo_mount['readonly'] and repo == Path(producer['repo_path']), 'learning producer repository mapping differs')
    require(isinstance(values['predecessor_run'],str)
        and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',values['predecessor_run']),
        'learning predecessor run required')
    job_path, job_mount = binding.host_path(producer, str(Path(values['job'])/'complete.json'))
    require(job_mount['readonly'] and job_path.is_file() and not job_path.is_symlink()
        and job_path.absolute() == job_path.resolve(), 'learning producer requires read-only job manifest')
    job_blob = job_path.read_bytes(); job = json.loads(job_blob)
    require(isinstance(job,dict) and job.get('schema') == 'saea-learning-strategy-job-v1'
        and job.get('job') == expected['candidate'] and job.get('run_id') == expected['run_id']
        and job.get('service_session') == expected['service_session']
        and type(job.get('runtime_rows')) is int and job['runtime_rows'] == expected['rows']
        and expected['attempt_id'] == 'learning-v1', 'learning producer job identity differs')
    require(pin_path(pins[0]) == config_path, 'learning command changed during verification')
    return launcher.absolute(values['out']), {'job_manifest_path':str(job_path),
        'job_manifest_sha256':hashlib.sha256(job_blob).hexdigest(), 'command':pins[0]}


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
    output, _ = producer_output(producer, logical, expected)
    actual, mount = binding.host_path(producer, str(output / 'generation-released.json'))
    require(not mount['readonly'] and actual.absolute() == launcher.absolute(release['path'])
            and actual.resolve() == actual.absolute(), 'producer release path or writable mapping differs')
    require(env.get('SPEECHRL_DATA_DIR', str(logical)) == str(logical), 'producer logical data mapping differs')
    return actual, logical, expected


def verify_generation_release(doc, producer, release):
    """Reuse the shared seal verifier through explicit host mount translation."""
    path, logical, expected = release_identity(producer, release)
    _, command_receipt = producer_output(producer, logical, expected)
    require(path.is_file() and not path.is_symlink(), 'immutable generation release file required')
    blob = path.read_bytes(); value = json.loads(blob)
    require(set(value) == set(expected) | {'schema', 'generation_sealed', 'generation_lease_released',
        'scoring_complete', 'consumer_requires_own_lease_and_readiness', 'released_at_unix',
        'campaign_seconds_to_release', 'artifacts', 'seal_receipt'}, 'exact generation release schema required')
    started_path = path.with_name('started.json')
    require(started_path.is_file() and not started_path.is_symlink(), 'producer started receipt required')
    started_blob = started_path.read_bytes(); started = json.loads(started_blob)
    require(started.get('study_commit') == expected['study_commit'], 'producer startup commit differs')
    if command_receipt is not None:
        require(started.get('job') == expected['candidate'], 'learning producer started job differs')
    for key in ('candidate', 'job', 'attempt_id', 'rows'):
        if key in started:
            require(started[key] == expected['candidate' if key == 'job' else key], 'producer startup identity differs')
    mappings = {}
    helper = release_helper(doc)
    files = helper.release_paths(value, data=logical, expected=expected)
    require(isinstance(value.get('artifacts'), dict) and set(value['artifacts']) == set(files), 'all generation sealed inputs required')
    for key, logical_path in files.items():
        pin = value['artifacts'][key]
        require(isinstance(pin, dict) and set(pin) == {'path', 'sha256'} and pin['path'] == str(logical_path),
                'generation seal logical path differs')
        physical, mount = binding.host_path(producer, str(logical_path))
        require(not mount['readonly'] and physical.absolute() == physical.resolve(), 'generation artifact host mapping redirected')
        require(physical not in mappings.values(), 'generation artifact mappings alias')
        mappings[logical_path] = physical
    helper.verify_release(path, data=logical, expected=expected,
        resolve_artifact=lambda target:mappings[target])
    require(path.read_bytes() == blob and started_path.read_bytes() == started_blob, 'generation release changed during verification')
    require(producer_output(producer, logical, expected)[1] == command_receipt,
        'learning producer command or job changed during verification')
    return {'path':str(path), 'sha256':hashlib.sha256(blob).hexdigest(),
        'started_sha256':hashlib.sha256(started_blob).hexdigest(), 'cpu_scoring_may_continue':True,
        'scoring_complete':False, 'consumer_requires_own_lease_and_readiness':True,
        **({'producer_command':command_receipt} if command_receipt is not None else {})}


def wait_generation_event(doc, path):
    """Arm inotify before rechecking existence, closing publication races."""
    path = Path(path); directory = path.parent
    while not directory.exists():directory = directory.parent
    events = release_helper(doc).DirectoryEvents(directory)
    try:
        if path.exists():return
        events.wait(doc['monitor_interval_seconds'])
    finally:events.close()


def observe(pin, launcher_artifact=None):
    config = launcher.load_config(pin_path(pin))
    runner_sha = predecessor_runner(launcher_artifact) if launcher_artifact is not None else launcher.digest(launcher.__file__)
    observed = launcher.status(config, canonical_hash(config), runner_sha)
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


def dependency_status(doc, job, record):
    if job['after'] is None:return 'complete', None, None
    dep = job['after']; producer, observed = observe(dep['launch'], dep.get('launcher_artifact'))
    require(observed.get('container_id') == dep['container_id'], 'predecessor container identity differs')
    record['dependency'] = observed
    result = outcome(observed); release_path = None
    if dep['condition'] == 'generation-released':
        release_path, _, _ = release_identity(producer, dep['release'])
        if release_path.exists():
            require(observed['state'] in ('running', 'exited'), 'producer is not a known running or terminal container')
            event = verify_generation_release(doc, producer, dep['release'])
            require(record.get('generation_release', event) == event, 'previous generation release bytes changed')
            record['generation_release'] = event; result = 'complete'
        elif result == 'complete':result = 'unknown'
    return (result, dep['container_id'] if result == 'running' else None,
            str(release_path) if release_path and result == 'running' else None)


def refresh_generation(doc, job, record):
    """Release the lane only after verifying this launched producer's seal."""
    producer, observed = observe(job['launch'])
    require(record.get('container_id') == observed.get('container_id')
            and record.get('container_id'), 'generation producer container identity changed')
    result = outcome(observed)
    require(result in ('running', 'complete', 'failed'), 'generation producer state unknown')
    path, _, _ = release_identity(producer, job['generation_release'])
    if path.exists():
        proof = verify_generation_release(doc, producer, job['generation_release'])
        require(record.get('own_generation_release', proof) == proof, 'producer release custody changed')
        record['own_generation_release'] = proof
    else:
        require('own_generation_release' not in record, 'previous producer release disappeared')
        if result == 'complete':
            raise ValueError('successful producer exited without generation release')
    record['observed'] = observed
    record['status'] = 'scoring' if result == 'running' and 'own_generation_release' in record else result
    if result in ('complete', 'failed'):
        record['result'] = {'container_exit_success': result == 'complete',
            'generation_released': 'own_generation_release' in record, 'scientific_success': False}
    return path


def handoff_ready(job, records):
    handoff = job.get('after_handoff')
    return handoff is None or ('own_generation_release' in records[handoff['producer']]
        or records[handoff['settlement']]['status'] == 'complete')


def step(config_path, doc, state, save):
    require(identities(config_path) == state['identity'], 'queue configuration or implementation changed')
    load(config_path)  # Recheck every immutable input before any dispatch.
    if state['status'] in ('failed', 'complete'):return state
    graph = all('depends_on' in job for job in doc['jobs'])
    if graph:
        records = {record['id']:record for record in state['jobs']}
        for job in doc['jobs']:
            record = records[job['id']]
            if 'generation_release' in job and record.get('container_id'):
                refresh_generation(doc, job, record)
                if (record['status'] == 'failed' and 'own_generation_release' not in record
                        and job.get('on_fail', 'stop') == 'stop'):
                    state['status'] = 'failed'; save(); return state
        for job in doc['jobs']:
            producer = job.get('unless_generation_released')
            record = records[job['id']]
            if producer and record['status'] == 'pending' and 'own_generation_release' in records[producer]:
                record.update(status='skipped', reason='verified generation release', producer=producer)
        active = [i for i, record in enumerate(state['jobs'])
                  if record['status'] in ('running', 'launch-intent', 'unknown')]
        require(len(active) <= 1, 'multiple active containers in a serial resource lane')
        ready = [i for i, job in enumerate(doc['jobs']) if records[job['id']]['status'] == 'pending'
                 and all(records[dep]['status'] == 'complete' for dep in job['depends_on'])
                 and all(records[dep]['status'] in ('complete', 'failed')
                         for dep in job.get('after_terminal', []))
                 and handoff_ready(job, records)]
        eligible, waiting = [], []
        if not active:
            for i in ready:
                job = doc['jobs'][i]; record = records[job['id']]
                result, wait_id, release_path = dependency_status(doc, job, record)
                if result == 'complete':
                    eligible.append(i); break
                if result == 'running':
                    waiting.append((wait_id, release_path)); continue
                if result == 'failed' and job.get('on_fail', 'stop') == 'continue':
                    record.update(status='blocked', reason='external dependency failed'); continue
                state.update(status=result, wait_container_id=None, wait_release_path=None)
                save(); return state
        if not active and not eligible:
            state['blocked_jobs'] = [r['id'] for r in state['jobs'] if r['status'] in ('pending', 'blocked')]
            scoring = [r for r in state['jobs'] if r['status'] == 'scoring']
            if waiting:
                state.update(status='waiting', wait_container_id=waiting[0][0], wait_release_path=waiting[0][1])
            elif scoring:
                state.update(status='waiting', wait_container_id=scoring[0]['container_id'], wait_release_path=None)
            else:
                state['status'] = 'complete' if all(r['status'] in ('complete', 'skipped') for r in state['jobs']) else 'failed'
            save(); return state
        index = (active or eligible)[0]
    else:
        index = len([j for j in state['jobs'] if j.get('status') == 'complete'])
        if index == len(doc['jobs']):
            state['status'] = 'complete'; save(); return state
    job, record = doc['jobs'][index], state['jobs'][index]
    if graph and job.get('after_handoff'):
        handoff = job['after_handoff']; producer = next(j for j in doc['jobs'] if j['id'] == handoff['producer'])
        previous = records[producer['id']]
        refresh_generation(doc, producer, previous)
        if 'own_generation_release' not in previous:
            cleanup = next(j for j in doc['jobs'] if j['id'] == handoff['settlement'])
            _, observed = observe(cleanup['launch'])
            require(records[cleanup['id']]['status'] == 'complete'
                and observed.get('container_id') == records[cleanup['id']].get('container_id')
                and outcome(observed) == 'complete', 'handoff settlement is not the exact successful container')
    if state['status'] == 'unknown' and not record.get('launch_requested'):return state
    predecessors = ([i for i, j in enumerate(doc['jobs'])
                     if j['id'] in job['depends_on'] + job.get('after_terminal', [])]
                    if graph else ([index - 1] if index else []))
    for prior_index in predecessors:
        prior = doc['jobs'][prior_index]; receipt = state['jobs'][prior_index]
        _, observed = observe(prior['launch'])
        expected = receipt['status'] if graph and prior['id'] in job.get('after_terminal', []) else 'complete'
        require(observed.get('container_id') == receipt.get('container_id')
                and outcome(observed) == expected and expected in ('complete', 'failed'),
                'prior queued container no longer matches required terminal receipt')
    result, wait_id, release_path = dependency_status(doc, job, record)
    if result != 'complete':
        state.update(status='waiting' if result == 'running' else result,
                     wait_container_id=wait_id, wait_release_path=release_path)
        save(); return state
    state['wait_release_path'] = None
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
        planned = launcher.execute(launch_path, 'plan')
        readiness = planned['generation_readiness']
        if readiness['state'] not in ('clear', 'not-configured'):
            state.update(status='held', generation_readiness=readiness)
            save(); return state
        state.pop('generation_readiness', None)
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
    if result == 'running' and 'generation_release' in job:
        producer_path, _, _ = release_identity(launch_doc, job['generation_release'])
        state['wait_release_path'] = str(producer_path)
    state['status'] = 'ready' if result == 'complete' or (
        result == 'failed' and graph and job.get('on_fail', 'stop') == 'continue') else result
    if result == 'complete':
        record['result'] = {'container_exit_success':True, 'scientific_success':False,
            'verification':'verify shared runtime artifacts separately', 'completed_at':time.time()}
        if all(r['status'] == 'complete' for r in state['jobs']):state['status'] = 'complete'
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
            plans.append({'id':job['id'], 'launch':launcher.execute(launch, 'plan'), 'after':job['after'],
                **({'depends_on':job['depends_on'], 'after_terminal':job.get('after_terminal', []),
                    'on_fail':job.get('on_fail', 'stop'),
                    **{k:job[k] for k in ('generation_release','after_handoff','unless_generation_released') if k in job}}
                   if 'depends_on' in job else {})})
        return {'schema':SCHEMA, 'identity':identity, 'jobs':plans, 'new_model_calls':0,
            'dependency_policy':('dependency-ready-container-exit-zero' if all('depends_on' in j for j in doc['jobs'])
                else 'serial-container-exit-zero-or-verified-generation-release'), 'generation_release_supported':True}
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
            if mode == 'once' or state['status'] in ('failed', 'unknown', 'complete', 'held'):return state
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'once', 'watch'))
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    result = execute(args.config, args.mode)
    print(json.dumps(result, sort_keys=True))
    # A completed polling command is not necessarily a successful queue.
    # Preserve the full terminal result even when an independent branch failed.
    if result.get('status') in ('failed', 'unknown', 'held'):
        return 1
    if args.mode == 'watch' and result.get('status') != 'complete':
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
