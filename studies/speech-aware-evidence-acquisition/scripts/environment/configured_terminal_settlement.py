#!/usr/bin/env python3
"""Settle one configured producer without restarting or replaying any contact."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time


SCHEMA = 'saea-configured-terminal-settlement-v1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/environment')]


def sibling(name):
    spec = importlib.util.spec_from_file_location('settlement_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


queue = sibling('configured_job_queue')
launcher, binding, require = queue.launcher, queue.binding, queue.require
transition = sibling('transition_serving_budget')


def read(path):
    require(path.is_file() and not path.is_symlink() and path.absolute() == path.resolve(),
            'canonical regular custody file required')
    return json.loads(path.read_bytes())


def pin(pin):
    path = queue.pin_path(pin)
    read(path)
    return path


def mapped(producer, path, *, writable=None):
    target, mount = binding.host_path(producer, str(path))
    require(target.is_absolute() and target.absolute() == target.resolve() and not target.is_symlink(),
            'canonical producer host mapping required')
    if writable is not None:
        require(mount['readonly'] is not writable, 'producer mount access differs')
    return target


def stamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.utcoffset() is not None, 'timezone-aware producer timing required')
    return result.timestamp()


def producer_job(producer, logical, expected):
    """Resolve only established learning CLI or exact GEPA run arguments."""
    args = producer['argv']
    if len(args) == 4 and Path(args[1]).name == 'learning_strategy_job.py':
        require(args[2] == '--config', 'exact learning command required')
        command = read(mapped(producer, args[3], writable=False))
        return Path(command['arguments']['job'])
    require(len(args) == 15 and Path(args[1]).name == 'gepa_mutation_job.py'
        and args[2] == 'run', 'only configured learning or exact GEPA run producers are supported')
    flags = args[3::2]
    require(len(set(flags)) == 6 and set(flags) == {
        '--repo', '--data', '--job', '--out', '--after', '--predecessor-run'},
        'exact unique GEPA run arguments required')
    values = dict(zip(flags, args[4::2], strict=True))
    for key in ('--repo', '--data', '--job', '--out', '--after'):
        require(str(launcher.absolute(values[key])) == values[key], 'canonical GEPA path required')
    require(values['--data'] == str(logical)
        and args[1] == str(Path(values['--repo'])/'scripts/analysis/gepa_mutation_job.py')
        and mapped(producer, values['--repo'], writable=False) == Path(producer['repo_path']),
        'GEPA logical repository or data differs')
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', values['--predecessor-run']),
        'GEPA predecessor identity required')
    job = Path(values['--job'])
    require(job.is_relative_to(logical/'work/codex'), 'owned GEPA job directory required')
    document = read(mapped(producer, job/'complete.json', writable=False))
    require(document.get('schema') == 'saea-gepa-mutation-job-v1'
        and all(document.get(k) == expected[v] for k,v in (
            ('job','candidate'),('run_id','run_id'),('service_session','service_session'),
            ('runtime_rows','rows')))
        and type(document.get('runtime_rows')) is int and document['runtime_rows'] == 1
        and document.get('generation_calls') == 1 and document.get('fit_calls') == 1
        and expected['attempt_id'] == 'learning-v1', 'GEPA job identity or one-contact budget differs')
    return job


def load(path):
    doc = read(Path(path))
    require(set(doc) == {'schema', 'producer_launch', 'producer_launcher', 'handoff_verifier',
        'generation_release', 'source_plan', 'runtime_receipt', 'arm', 'state_dir'}
        and doc['schema'] == SCHEMA, 'exact terminal settlement configuration required')
    producer = launcher.load_config(pin(doc['producer_launch']))
    helper = queue.pin_path(doc['producer_launcher'])
    require(helper.name == 'configured_job_launch.py' and helper.resolve() == helper.absolute(),
            'pinned producer launcher required')
    queue.release_helper(doc)
    release, logical, expected = queue.release_identity(producer, doc['generation_release'])
    require(producer['pid_host'] is True, 'producer must expose host PID identity')
    require(doc['arm'] == 'prepared-context-text', 'only prepared learning contacts are supported')
    producer_job(producer, logical, expected)
    source = pin(doc['source_plan']); pin(doc['runtime_receipt'])
    require(source == Path(producer['repo_path']) / 'docs/readiness' / expected['candidate'] / 'generation-plan.json',
            'source must be the registered producer generation plan')
    state = launcher.absolute(doc['state_dir'])
    require(state.absolute() == state.resolve() and len(state.parts) >= 4, 'scoped canonical settlement state required')
    for item in producer['mounts']:
        target = Path(item['source']).resolve()
        require(not target.is_relative_to(state), 'settlement state must not contain producer inputs')
    for name in ('locks', 'attempts', 'contact-journals', 'runs', 'serving', 'prepared'):
        base = logical / name
        protected = ([mapped(producer, base)] if any(base.is_relative_to(m['target'])
            for m in producer['mounts']) else [Path(m['source']).resolve() for m in producer['mounts']
            if Path(m['target']).is_relative_to(base)])
        require(all(not state.is_relative_to(p) for p in protected), 'settlement state overlaps runtime custody')
    return doc, producer, release, logical, expected, state


def producer_terminal(doc, producer):
    """Bind a future container ID only through durable launch intent AND result."""
    root = Path(producer['receipt_dir'])
    intent, result = read(root / 'launch-intent.json'), read(root / 'launch-result.json')
    config_sha = queue.canonical_hash(producer)
    runner_sha = doc['producer_launcher']['sha256']
    require(intent == {'schema': launcher.SCHEMA, 'version': launcher.VERSION,
        'config_sha256': config_sha, 'runner_sha256': runner_sha,
        'expected_commit': producer['expected_commit'], 'automatic_retry': False}, 'producer launch intent differs')
    require(set(result) == {'returncode', 'stdout', 'stderr', 'automatic_retry'}
        and type(result['returncode']) is int and result['returncode'] == 0
        and result['automatic_retry'] is False and isinstance(result['stdout'], str)
        and re.fullmatch('[a-f0-9]{64}', result['stdout'].strip()), 'unresolved producer launch result')
    container_id = result['stdout'].strip()
    inspected = launcher.run(['docker', 'container', 'inspect', container_id])
    require(inspected.returncode == 0, 'exact producer container inspection failed')
    container, = json.loads(inspected.stdout)
    labels = container['Config'].get('Labels') or {}
    require(container['Id'] == container_id and container['Name'] == '/' + producer['container_name']
        and container['Image'] == producer['image'] and labels.get('saea.launch.config') == config_sha
        and labels.get('saea.launch.runner') == runner_sha, 'producer container custody differs')
    env = dict(value.split('=', 1) for value in container['Config'].get('Env', []))
    require(all(env.get(k) == v for k, v in producer['environment'].items())
        and container['HostConfig'].get('PidMode') == 'host', 'producer runtime environment differs')
    state = container['State']
    require(state['Status'] == 'exited' and state['Running'] is False and type(state['Pid']) is int
        and state['Pid'] == 0 and not state.get('Restarting', False)
        and type(state['ExitCode']) is int, 'producer is not terminal')
    require(stamp(state['FinishedAt']) >= stamp(state['StartedAt']), 'producer lifetime differs')
    return {'container_id': container_id, 'state': state, 'launch_intent_sha256': launcher.digest(root / 'launch-intent.json'),
        'launch_result_sha256': launcher.digest(root / 'launch-result.json')}


def source_custody(doc, producer, logical, expected, release):
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal
    registered = read(pin(doc['source_plan']))
    values = dict(registered['execution_plan'])
    if 'owner_allowance' in registered:
        require('owner_allowance' not in values or values['owner_allowance'] == registered['owner_allowance'],
                'plan owner allowance differs')
        values['owner_allowance'] = registered['owner_allowance']
    plan = ExecutionPlan(**values).validate(source_root=Path(producer['repo_path']))
    require((plan.run_id, plan.attempt_id) == (expected['run_id'], expected['attempt_id']), 'source plan identity differs')
    logical_job = producer_job(producer, logical, expected)
    job_path = mapped(producer, logical_job / 'complete.json', writable=False)
    job_pin = registered.get('job_manifest', {})
    require(job_pin == {'root': 'data', 'relpath': str(logical_job.relative_to(logical) / 'complete.json'),
                       'sha256': launcher.digest(job_path)}, 'registered plan job manifest differs')
    attempt_path = mapped(producer, logical / 'attempts' / plan.run_id / (plan.attempt_id + '.json'), writable=True)
    attempt = read(attempt_path)
    require(attempt.get('state') in ('ABORTED', 'FINALIZED') and attempt.get('run_id') == plan.run_id
        and attempt.get('attempt_id') == plan.attempt_id and attempt.get('carrier_lock_key') == plan.carrier_lock_key,
        'exact terminal attempt required')
    calls = attempt.get('actual', {}).get('calls')
    require(type(calls) is int and calls >= 0, 'exact recorded contact count required')
    journal = ContactJournal(logical, plan, doc['arm'])
    journal.root = mapped(producer, logical / 'contact-journals' / plan.run_id / plan.attempt_id, writable=True)
    files = sorted(journal.root.glob('*')) if journal.root.exists() else []
    require(all(p.is_file() and not p.is_symlink() and re.fullmatch(r'[a-f0-9]{64}\.(intent|response)\.json', p.name)
        for p in files), 'unexpected journal custody')
    intents = [p for p in files if p.name.endswith('.intent.json')]
    require(len(intents) == calls and len(files) == 2 * calls, 'unknown or orphan contact; settlement blocked')
    pins = []
    for intent in intents:
        request_id = read(intent)['request_id']
        require(intent == journal._path(request_id, 'intent'), 'contact filename differs')
        observed = journal.inspect(request_id)
        require(observed['state'] == 'observed', 'unknown request; settlement blocked')
        response = journal._path(request_id, 'response')
        pins.append({'request_id': request_id, 'intent_sha256': launcher.digest(intent), 'response_sha256': launcher.digest(response)})
    started = read(release.with_name('started.json'))
    require(started.get('study_commit') == expected['study_commit']
        and started.get('candidate', started.get('job')) == expected['candidate'], 'producer startup identity differs')
    return {'attempt_sha256': launcher.digest(attempt_path), 'attempt_state': attempt['state'],
        'actual': attempt['actual'], 'contacts': pins, 'started': started,
        'started_sha256': launcher.digest(release.with_name('started.json'))}


def deployment(doc, producer, logical, expected):
    receipt = read(pin(doc['runtime_receipt']))
    session_path = mapped(producer, logical / 'serving' / expected['service_session'] / 'session.json')
    session = read(session_path)
    instances = receipt['deployment']['instances']
    require(session.get('session_id') == expected['service_session']
        and session.get('identity', {}).get('runtime_receipt_sha256') == doc['runtime_receipt']['sha256']
        and session.get('deployment_instance_ids') == [i['instance_id'] for i in instances], 'producer deployment session differs')
    return instances


def finalized_artifact(doc, producer, logical, expected, source):
    """Reconcile a synchronous artifact seal, never invent a release event."""
    from speech_aware_evidence_acquisition.core.flight.receipts import read_flight_receipts
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha as learner_sha
    require(source['attempt_state'] == 'FINALIZED', 'artifact reconciliation needs FINALIZED attempt')
    root = logical / 'runs' / expected['run_id'] / expected['attempt_id']
    pins = {}
    def artifact(key, path, sha=None):
        target = Path(path)
        require(target.is_relative_to(root) and '..' not in target.parts, 'artifact escapes exact run')
        physical = mapped(producer, target, writable=True)
        require(physical.is_file(), 'sealed artifact missing')
        actual = launcher.digest(physical)
        require(sha is None or actual == sha, 'sealed artifact hash changed')
        pins[key] = {'path': str(target), 'sha256': actual}
        return physical
    receipt_path = artifact('flight_receipts', root / 'flight-receipts.json')
    receipts = read_flight_receipts(receipt_path)
    require(receipts['run_id'] == expected['run_id'] and receipts['attempt_id'] == expected['attempt_id'],
            'artifact flight receipt identity differs')
    stages = receipts['stages']; seal = stages['seal']; phase = stages['run_phases']
    finalize = seal.get('finalize', {})
    require(finalize.get('attempt_state') == 'FINALIZED' and finalize.get('result_kind') == 'artifact'
        and finalize.get('status') == 'not-applicable' and finalize.get('scores_path') is None
        and phase.get('result_kind') == 'artifact', 'not a synchronous artifact seal')
    require(stages['preflight']['study_commit'] == expected['study_commit'], 'artifact source head differs')
    manifest_target = root / (expected['run_id'] + '.run-manifest.json')
    require(seal['manifest_path'] == str(manifest_target), 'artifact manifest path differs')
    manifest = read(artifact('manifest', manifest_target, seal['manifest_sha256']))
    registered = read(pin(doc['source_plan']))['execution_plan']
    require(manifest.get('schema') == 'saea-run-manifest-v1' and manifest.get('result_kind') == 'artifact'
        and manifest.get('status') == 'COMPLETED' and manifest.get('failure_reason') is None
        and manifest.get('study_commit') == expected['study_commit']
        and manifest.get('run_id') == expected['run_id']
        and manifest.get('attempt', {}).get('attempt_id') == expected['attempt_id']
        and manifest['attempt'].get('run_id') == expected['run_id']
        and manifest.get('arm') == doc['arm'] and manifest.get('scores') is None
        and manifest.get('scorers') == []
        and all(registered.get(k) == v for k, v in manifest['plan'].items())
        and manifest['plan']['run_id'] == expected['run_id']
        and manifest['plan']['attempt_id'] == expected['attempt_id'], 'artifact manifest source differs')
    count = expected['rows']; split_ids = manifest['split']['ids']
    require(type(count) is int and count > 0 and len(split_ids) == len(set(split_ids)) == count
        and manifest['split']['count'] == count
        and manifest['split']['identity_hash'] == registered['split_identity_hash']
        and manifest['config']['config_hash'] == stages['preflight']['config_hash']
        and manifest['protocol_hash'] == registered['protocol_hash'], 'artifact split/protocol differs')
    for key in ('processed_dataset_rows', 'total_dataset_rows', 'outputs_row_count'):
        require(type(phase[key]) is int and phase[key] == count, 'artifact generation incomplete')
    require(phase['remaining_dataset_rows'] == 0 and seal['outputs_row_count'] == count
        and phase['contact_count'] == source['actual']['calls'] == len(source['contacts'])
        and seal['cost']['calls_used'] == source['actual']['calls']
        and seal['cost']['audio_seconds_used'] == source['actual']['audio_seconds']
        and manifest['cost'] == seal['cost'], 'artifact cost/terminal count differs')
    outputs_target = root / (expected['run_id'] + '.outputs.jsonl')
    require(seal['outputs_path'] == manifest['outputs']['path'] == str(outputs_target), 'artifact outputs path differs')
    outputs_path = artifact('outputs', outputs_target, manifest['outputs']['sha256'])
    rows = [json.loads(line) for line in outputs_path.read_text().splitlines()]
    require(len(rows) == count and [r['sample_id'] for r in rows] == split_ids
        and len({r['request_id'] for r in rows}) == count
        and all(r.get('result_kind') == 'artifact' and r.get('arm') == doc['arm'] and r.get('admitted') is True for r in rows),
        'artifact rows incomplete or foreign')
    phases_target = root / (expected['run_id'] + '.phases.json')
    require(seal['phases_path'] == str(phases_target), 'artifact phases path differs')
    phases = read(artifact('phases', phases_target, seal['phases_sha256']))
    require(phases['run_id'] == expected['run_id'] and phases['attempt_id'] == expected['attempt_id']
        and phases['result_kind'] == 'artifact', 'artifact phase identity differs')
    contacts = [c for p in phases['phases'] for c in p['contact_receipts']]
    require(len(contacts) == len(source['contacts']) and len({c['request_id'] for c in contacts}) == len(contacts)
        and {c['request_id'] for c in contacts} == {c['request_id'] for c in source['contacts']},
        'artifact phase/native contact coverage differs')
    by_id = {c['request_id']: c for c in contacts}
    require(all(all(row[k] == by_id[row['request_id']][k] for k in ('sample_id','request_sha256','response_sha256','text'))
        for row in rows), 'artifact row/phase contact differs')
    # Learner envelopes use stage/identity keys; native journals use request IDs.
    # Their file hashes are intentionally different and must be joined explicitly.
    response_pins = phase['learning']['contact_response_sha256']
    learner_root = root/'tfgrpo/learner/contacts'
    actual_files = mapped(producer, learner_root, writable=True)
    require(isinstance(response_pins, dict) and len(response_pins) == len(contacts)
        and set(response_pins) == {p.name for p in actual_files.glob('*.response.json')},
        'artifact learner response coverage differs')
    observed_ids = set()
    for name, sha in response_pins.items():
        require(re.fullmatch(r'[a-f0-9]{64}\.response\.json', name), 'artifact learner filename differs')
        envelope = read(artifact('learner_response:'+name, learner_root/name, sha))
        receipt = envelope['response']; rid = receipt['receipt_id'].rsplit('/',1)[-1]
        require(rid in by_id and rid not in observed_ids
            and receipt['receipt_id'] == expected['run_id']+'/'+expected['attempt_id']+'/'+rid
            and envelope['response_sha256'] == learner_sha(receipt)
            and all(receipt[k] == by_id[rid][k] for k in ('request_sha256','response_sha256','text','usage')),
            'artifact learner/phase binding differs')
        native = logical/'contact-journals'/expected['run_id']/expected['attempt_id']
        key = hashlib.sha256(rid.encode()).hexdigest()
        native_intent = read(mapped(producer,native/(key+'.intent.json'),writable=True))
        native_response = read(mapped(producer,native/(key+'.response.json'),writable=True))
        require(receipt['request_sha256'] == native_intent['request_sha256']
            and receipt['response_sha256'] == native_response['response_sha256'],
            'artifact learner/native response custody differs')
        observed_ids.add(rid)
    for key, cost_key in (('calls','calls_used'),('prompt_tokens','prompt_tokens_total'),('completion_tokens','completion_tokens_total')):
        require(sum(c['usage'][key] for c in contacts) == phase['learning']['usage'][key] == seal['cost'][cost_key],
                'artifact token accounting differs')
    candidate = phase['learning']['candidate']
    target = logical / candidate['relpath']
    require(candidate['root'] == 'data' and target == root/'tfgrpo/learner/candidate.json', 'artifact candidate path differs')
    candidate_doc = read(artifact('candidate', target, candidate['sha256']))
    require(candidate_doc['contact_response_sha256'] == response_pins, 'artifact candidate contacts differ')
    for key in ('raw_trace','trace_manifest'):
        target = root / (expected['run_id'] + ('.trace.jsonl' if key == 'raw_trace' else '.trace-manifest.json'))
        require(manifest[key]['path'] == str(target), 'artifact trace path differs')
        artifact(key, target, manifest[key]['sha256'])
    return {'kind':'synchronous-artifact-finalized','artifacts':pins,'rows':count,
        'observed_contacts':len(contacts),'cost':seal['cost'],'generation_complete':True,
        'scientific_success':False,'historical_release_created':False}


def lease_custody(lease, source, terminal, expected, pid_exists):
    require(lease.get('schema') == 'saea-generation-lease-v1' and all(lease.get(k) == v for k, v in {
        'run_id': expected['run_id'], 'attempt_id': expected['attempt_id'],
        'session_id': expected['service_session'], 'study_head': expected['study_commit'],
        'run_ids': [expected['run_id']]}.items()), 'foreign or unsupported generation lease')
    pid = lease.get('pid')
    require(type(pid) is int and pid > 0 and not pid_exists(pid), 'source PID exists or is unknown')
    started = source['started']
    if 'pid' in started:
        require(type(started['pid']) is int and pid == started['pid'], 'lease PID differs from producer startup')
    begin, end = (stamp(terminal['state'][key]) for key in ('StartedAt', 'FinishedAt'))
    require(begin <= stamp(lease['started_at']) <= end, 'lease outside producer lifetime')
    if 'time' in started:
        require(type(started['time']) in (int, float) and begin <= started['time'] <= end,
                'startup outside producer lifetime')


def sync(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


@contextmanager
def guard(path):
    require(path.is_file() and not path.is_symlink(), 'existing stable generation guard required')
    with path.open('r+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = os.fstat(handle.fileno())
        def check():
            now = path.stat()
            require(not path.is_symlink() and (now.st_dev, now.st_ino) == (identity.st_dev, identity.st_ino),
                    'generation guard replaced')
        check()
        yield check, {'device': identity.st_dev, 'inode': identity.st_ino}
        check()


def execute(path, *, pid_exists=None):
    doc, producer, release, logical, expected, out = load(path)
    identity = {'config_sha256': launcher.digest(path), 'tool_sha256': launcher.digest(__file__)}
    pid_exists = pid_exists or (lambda pid: Path('/proc', str(pid)).exists())
    locks = mapped(producer, logical / 'locks', writable=True)
    require(not out.is_relative_to(locks), 'settlement state must be separate from shared locks')
    out.mkdir(parents=True, exist_ok=True); sync(out.parent)
    with (out / 'settlement.guard').open('a+b') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = out / 'complete.json'; audit_path = out / 'audit.json'
        if complete.exists():
            done = read(complete); audit = read(audit_path)
            require(done.get('identity') == identity and done.get('audit_sha256') == launcher.digest(audit_path)
                and audit.get('identity') == identity, 'completed settlement custody changed')
            if audit['mode'] == 'stale-lease':
                require(launcher.digest(audit['preserved_lease']) == audit['lease_sha256'], 'preserved lease changed')
            return done
        with guard(locks / 'generation.guard') as (check_guard, guard_identity):
            terminal = producer_terminal(doc, producer)
            source = source_custody(doc, producer, logical, expected, release)
            instances = deployment(doc, producer, logical, expected)
            lease_path = locks / 'flight.lock'
            saved = locks / ('settled-flight-' + identity['config_sha256'] + '.json')
            pending = read(audit_path) if audit_path.exists() else None
            if pending:
                require(pending.get('identity') == identity and pending['guard'] == guard_identity
                    and pending['terminal'] == terminal and pending['source'] == source, 'pending settlement evidence changed')
            lease = None
            if lease_path.exists():
                require(not saved.exists(), 'both live and preserved lease exist')
                lease_blob = lease_path.read_bytes(); lease = read(lease_path)
                lease_custody(lease, source, terminal, expected, pid_exists)
                mode = 'stale-lease'; release_proof = None
            elif pending and pending['mode'] == 'stale-lease':
                require(saved.is_file() and launcher.digest(saved) == pending['lease_sha256'], 'missing durable preserved lease')
                lease_blob = saved.read_bytes(); lease = read(saved)
                lease_custody(lease, source, terminal, expected, pid_exists)
                mode = 'stale-lease'; release_proof = None
            else:
                require(not saved.exists(), 'unattributed preserved lease')
                if release.exists():
                    release_proof = queue.verify_generation_release(doc, producer, doc['generation_release'])
                    mode = 'sealed-release'
                else:
                    release_proof = finalized_artifact(doc, producer, logical, expected, source)
                    mode = 'finalized-artifact'
            quiet = transition.quiescence(instances)
            require(producer_terminal(doc, producer) == terminal
                and source_custody(doc, producer, logical, expected, release) == source, 'producer evidence changed during settlement')
            check_guard()
            if mode == 'sealed-release':
                require(queue.verify_generation_release(doc, producer, doc['generation_release']) == release_proof,
                        'release evidence changed during settlement')
            elif mode == 'finalized-artifact':
                require(finalized_artifact(doc, producer, logical, expected, source) == release_proof,
                        'artifact evidence changed during settlement')
            if pending:
                require(pending['mode'] == mode and (mode != 'stale-lease' or pending['lease_sha256'] == hashlib.sha256(lease_blob).hexdigest()),
                        'pending settlement mode or lease differs')
                if mode != 'stale-lease':require(pending['release'] == release_proof, 'release proof changed')
            else:
                audit = {'schema': SCHEMA, 'identity': identity, 'mode': mode, 'terminal': terminal, 'source': source,
                    'guard': guard_identity, 'quiescence': quiet, 'release': release_proof, 'observed_at_unix': time.time(),
                    'preserved_lease': str(saved) if lease else None,
                    'lease_sha256': hashlib.sha256(lease_blob).hexdigest() if lease else None, 'new_model_calls': 0}
                queue.atomic(audit_path, audit)
            if mode == 'stale-lease' and lease_path.exists():
                require(lease_path.read_bytes() == lease_blob and not saved.exists(), 'lease changed before preservation')
                check_guard(); os.rename(lease_path, saved); sync(locks)
            require(not lease_path.exists(), 'new or foreign lease blocks completion')
            check_guard()
            done = {'schema': SCHEMA, 'identity': identity, 'audit_sha256': launcher.digest(audit_path),
                'settled': True, 'mode': mode, 'container_id': terminal['container_id'],
                'producer_exit_code': terminal['state']['ExitCode'], 'observed_contacts': len(source['contacts']),
                'new_model_calls': 0, 'scientific_success': False,
                'successor_requires_own_lease_and_readiness': True, 'completed_at_unix': time.time()}
            queue.atomic(complete, done)
            return done


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    print(json.dumps(execute(parser.parse_args().config), sort_keys=True))
