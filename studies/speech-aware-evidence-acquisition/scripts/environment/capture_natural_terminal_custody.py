#!/usr/bin/env python3
"""Capture a bounded natural-drain observation; never release or modify a lease."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from checkpoint_terminal_custody import inspect_generation, _pinned
from transition_serving_budget import quiescence


def launch_config_digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def observe_running(*, config_path, config_sha256, output, repo, proc_root=Path('/proc')):
    """Pin stable live producer identity without locking or changing its lease."""
    config_pin = {'path': config_path, 'sha256': config_sha256}
    config = json.loads(_pinned(config_pin))
    if config.get('schema') != 'saea-producer-observation-config-v1':
        raise ValueError('producer observation config required')
    pins = {k: config[k] for k in ('plan', 'runtime_receipt', 'session', 'launch_config', 'launch_intent', 'launch_result')}
    blobs = {k: _pinned(v) for k, v in pins.items()}
    launch = json.loads(blobs['launch_config'])
    intent, result = json.loads(blobs['launch_intent']), json.loads(blobs['launch_result'])
    registered = json.loads(blobs['plan'])
    raw = dict(registered['execution_plan'])
    if 'owner_allowance' in registered:
        if 'owner_allowance' in raw and raw['owner_allowance'] != registered['owner_allowance']:
            raise ValueError('owner allowance conflict')
        raw['owner_allowance'] = registered['owner_allowance']
    plan = ExecutionPlan(**raw).validate(source_root=repo)
    receipt, session = json.loads(blobs['runtime_receipt']), json.loads(blobs['session'])
    container = result.get('stdout', '').strip()
    if (intent.get('schema') != 'saea-configured-job-launch-v1'
            or not isinstance(intent.get('runner_sha256'), str) or len(intent['runner_sha256']) != 64
            or intent.get('config_sha256') != launch_config_digest(launch) or result.get('returncode') != 0
            or len(container) != 64 or session.get('session_id') != config.get('session_id')
            or session.get('identity', {}).get('runtime_receipt_sha256') != pins['runtime_receipt']['sha256']
            or session.get('deployment_instance_ids') != [i['instance_id'] for i in receipt['deployment']['instances']]
            or launch.get('environment', {}).get('SAEA_OBSERVABILITY_RUN_ID') != plan.run_id
            or launch.get('environment', {}).get('SAEA_OBSERVABILITY_ATTEMPT_ID') != plan.attempt_id):
        raise ValueError('producer inputs differ')
    output = Path(output)
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise ValueError('fresh absolute output required')
    logical_locks = str(Path(launch['argv'][launch['argv'].index('--data') + 1]) / 'locks')
    def snapshot():
        rows = json.loads(subprocess.check_output(['docker', 'inspect', container]))
        if len(rows) != 1:
            raise ValueError('exact container required')
        row = rows[0]; state = row['State']; pid = state.get('Pid')
        actual_mounts = {(m['Source'], m['Destination'], m['RW']) for m in row['Mounts']}
        declared = {(m['source'], m['target'], not m['readonly']) for m in launch['mounts']}
        labels = row.get('Config', {}).get('Labels') or {}
        if (row['Id'] != container or state.get('Running') is not True or state.get('Status') != 'running'
                or type(pid) is not int or pid < 1 or [row['Path'], *row['Args']] != launch['argv']
                or row['Image'] != launch['image'] or declared != actual_mounts
                or labels.get('saea.launch.config') != intent.get('config_sha256')
                or labels.get('saea.launch.runner') != intent.get('runner_sha256')):
            raise ValueError('running container identity differs')
        matches = [m for m in row['Mounts'] if m['Destination'] == logical_locks and m['RW']]
        if len(matches) != 1 or matches[0]['Source'] != launch['generation_lease']['lock_directory']:
            raise ValueError('producer guard mount differs')
        lock_dir = Path(matches[0]['Source']); guard = lock_dir/'generation.guard'; lease = lock_dir/'flight.lock'
        if guard.is_symlink() or not guard.is_file() or lease.is_symlink():
            raise ValueError('existing guard and lease required')
        st = guard.stat(); lease_blob = lease.read_bytes(); lock = json.loads(lease_blob)
        if any(lock.get(k) != v for k, v in {'run_id':plan.run_id,'attempt_id':plan.attempt_id,
                                               'pid':pid,'session_id':config['session_id']}.items()):
            raise ValueError('live lease identity differs')
        proc = proc_root/str(pid)
        stat = (proc/'stat').read_text(); fields = stat[stat.rfind(')')+2:].split()
        start = int(fields[19]); cmdline = (proc/'cmdline').read_bytes().split(b'\0')
        command = [x.decode() for x in cmdline if x]
        if start < 1 or command != launch['argv']:
            raise ValueError('live process command or start differs')
        boot = (proc_root/'sys/kernel/random/boot_id').read_text().strip()
        return {'container_id':container,'producer_pid':pid,'producer_start_ticks':start,'boot_id':boot,
                'guard':{'device':st.st_dev,'inode':st.st_ino},'lock_directory':str(lock_dir),
                'lease_sha256':hashlib.sha256(lease_blob).hexdigest(), 'lease_identity':{k:lock[k] for k in ('run_id','attempt_id','pid','session_id')},
                'process_cmdline':command}
    before = snapshot(); after = snapshot()
    if before != after:
        raise ValueError('live producer identity changed during observation')
    _pinned(config_pin)
    for pin in pins.values(): _pinned(pin)
    observation = {'schema':'saea-producer-process-observation-v1', **pins, **before,
                   'session_id':config['session_id'], 'run_id':plan.run_id, 'attempt_id':plan.attempt_id,
                   'observed_before_exit':time.time(), 'config':config_pin}
    publish_new(output, (json.dumps(observation, sort_keys=True, indent=2)+'\n').encode())
    return observation


def capture(*, data, identity_path, identity_sha256, arm, output, repo):
    data, output = Path(data).resolve(), Path(output)
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise ValueError('fresh absolute output required')
    if output.is_relative_to(data):
        raise ValueError('output must be outside the data root')
    identity_pin = {'path': identity_path, 'sha256': identity_sha256}
    identity_blob = _pinned(identity_pin)
    identity = json.loads(identity_blob)
    if identity.get('schema') != 'saea-producer-process-observation-v1':
        raise ValueError('producer identity pin required')
    if (type(identity.get('producer_pid')) is not int or identity['producer_pid'] < 1
            or type(identity.get('producer_start_ticks')) is not int or identity['producer_start_ticks'] < 1
            or not isinstance(identity.get('container_id'), str) or len(identity['container_id']) != 64):
        raise ValueError('producer process identity invalid')
    plan_blob = _pinned(identity['plan']); registered = json.loads(plan_blob)
    runtime_blob = _pinned(identity['runtime_receipt'])
    launch_blob = _pinned(identity['launch_config'])
    intent = json.loads(_pinned(identity['launch_intent']))
    result = json.loads(_pinned(identity['launch_result']))
    launch = json.loads(launch_blob)
    if (intent.get('schema') != 'saea-configured-job-launch-v1'
            or not isinstance(intent.get('runner_sha256'), str) or len(intent['runner_sha256']) != 64
            or intent.get('config_sha256') != launch_config_digest(launch)
            or result.get('returncode') != 0 or result.get('stdout', '').strip() != identity['container_id']):
        raise ValueError('launcher evidence differs from producer pin')
    raw = dict(registered['execution_plan'])
    if 'owner_allowance' in registered:
        if 'owner_allowance' in raw and raw['owner_allowance'] != registered['owner_allowance']:
            raise ValueError('owner allowance conflict')
        raw['owner_allowance'] = registered['owner_allowance']
    plan = ExecutionPlan(**raw).validate(source_root=repo)
    if (identity.get('run_id') != plan.run_id or identity.get('attempt_id') != plan.attempt_id
            or identity.get('process_cmdline') != launch['argv']
            or identity.get('lease_identity') != {'run_id':plan.run_id,'attempt_id':plan.attempt_id,
                'pid':identity['producer_pid'],'session_id':identity['session_id']}
            or type(identity.get('observed_before_exit')) not in (int, float)
            or identity['observed_before_exit'] <= 0):
        raise ValueError('live producer observation differs from plan')
    receipt_blob = runtime_blob; receipt = json.loads(receipt_blob)
    session_id, expected_pid = identity['session_id'], identity['producer_pid']
    container = identity['container_id']
    lock_dir = Path(identity['lock_directory'])
    guard = lock_dir/'generation.guard'; lease = lock_dir/'flight.lock'
    logical_locks = str(Path(launch['argv'][launch['argv'].index('--data') + 1])/'locks')
    if (launch.get('generation_lease', {}).get('lock_directory') != str(lock_dir)
            or not any(m.get('source') == str(lock_dir) and m.get('target') == logical_locks
                       and m.get('readonly') is False for m in launch['mounts'])
            or launch.get('environment', {}).get('SAEA_OBSERVABILITY_RUN_ID') != plan.run_id
            or launch.get('environment', {}).get('SAEA_OBSERVABILITY_ATTEMPT_ID') != plan.attempt_id):
        raise ValueError('producer plan or guard mapping differs from launch')
    if guard.is_symlink() or not guard.is_file() or expected_pid < 1:
        raise ValueError('existing stable guard and producer PID required')
    with guard.open('rb') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stat = os.fstat(handle.fileno())
        if identity.get('guard') != {'device':stat.st_dev, 'inode':stat.st_ino}:
            raise ValueError('guard inode differs from producer pin')
        def verify_guard():
            cur = guard.stat()
            if guard.is_symlink() or (cur.st_dev, cur.st_ino) != (stat.st_dev, stat.st_ino):
                raise ValueError('guard replaced')
            if os.path.lexists(lease):
                raise ValueError('lease exists; natural drain not established')
        def inspect_container():
            rows = json.loads(subprocess.check_output(['docker', 'inspect', container]))
            if len(rows) != 1:
                raise ValueError('exact container required')
            row = rows[0]; state = row['State']
            if state.get('Running') is not False or state.get('Status') != 'exited' or state.get('Pid') != 0:
                raise ValueError('producer container has not exited')
            mounts = {(m['Source'], m['Destination'], m['RW']) for m in row['Mounts']}
            declared = {(m['source'], m['target'], not m['readonly']) for m in launch['mounts']}
            if (row['Id'] != container or [row['Path'], *row['Args']] != launch['argv']
                    or row['Image'] != launch['image']
                    or (row.get('Config', {}).get('Labels') or {}).get('saea.launch.config') != intent.get('config_sha256')
                    or (row.get('Config', {}).get('Labels') or {}).get('saea.launch.runner') != intent.get('runner_sha256')
                    or mounts != declared or (str(lock_dir), logical_locks, True) not in mounts):
                raise ValueError('container command or guard mount differs from producer')
            return {'id': row['Id'], 'state': state}
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if boot != identity['boot_id'] or type(identity.get('producer_start_ticks')) is not int or identity['producer_start_ticks'] < 1:
            raise ValueError('producer boot or start identity differs')
        verify_guard()
        before = inspect_container()
        if Path('/proc', str(expected_pid)).exists():
            raise ValueError('producer PID still exists or has been reused')
        observation = inspect_generation(data_root=data, plan=plan, arm=arm, source_root=repo)
        attempt_path = data/'attempts'/plan.run_id/(plan.attempt_id+'.json')
        attempt_blob = attempt_path.read_bytes()
        session_blob = _pinned(identity['session'])
        session = json.loads(session_blob)
        if (session.get('session_id') != session_id or session.get('identity', {}).get('runtime_receipt_sha256') != hashlib.sha256(receipt_blob).hexdigest()
                or session.get('deployment_instance_ids') != [i['instance_id'] for i in receipt['deployment']['instances']]):
            raise ValueError('resident session differs from runtime receipt')
        quiet = quiescence(receipt['deployment']['instances'])
        verify_guard()
        if inspect_container() != before or Path('/proc', str(expected_pid)).exists() or Path('/proc/sys/kernel/random/boot_id').read_text().strip() != boot:
            raise ValueError('producer or boot changed during observation')
        if (attempt_path.read_bytes() != attempt_blob
                or inspect_generation(data_root=data, plan=plan, arm=arm, source_root=repo) != observation):
            raise ValueError('attempt or journal changed during observation')
        for pin in (identity_pin, identity['plan'], identity['runtime_receipt'], identity['launch_config'],
                    identity['launch_intent'], identity['launch_result'], identity['session']):
            _pinned(pin)
        result = {'schema':'saea-natural-terminal-observation-v1', 'mode':'natural-drain',
            'container':before, 'boot_id':boot, 'producer_pid':expected_pid, 'session_id':session_id,
            'producer_start_ticks':identity['producer_start_ticks'], 'producer_identity_sha256':identity_sha256,
            'guard':{'device':stat.st_dev,'inode':stat.st_ino}, 'lease_absent_under_guard':True,
            'run_id':plan.run_id,'attempt_id':plan.attempt_id,
            'plan_sha256':hashlib.sha256(plan_blob).hexdigest(),
            'attempt_sha256':hashlib.sha256(attempt_blob).hexdigest(),
            'runtime_receipt_sha256':hashlib.sha256(receipt_blob).hexdigest(),
            'session_sha256':hashlib.sha256(session_blob).hexdigest(),
            'observed_requests':observation['observed_requests'],
            **({'not_sent_requests':observation['not_sent_requests']} if observation.get('not_sent_requests') else {}),'quiescence':quiet,'captured_at':time.time()}
        verify_guard()
        publish_new(output, (json.dumps(result, sort_keys=True, indent=2)+'\n').encode())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observe-running', action='store_true')
    parser.add_argument('--config-path')
    parser.add_argument('--config-sha256')
    for name in ('data','identity-path','identity-sha256','arm','output','repo'):
        parser.add_argument('--'+name)
    args = vars(parser.parse_args())
    if args.pop('observe_running'):
        result = observe_running(config_path=args['config_path'], config_sha256=args['config_sha256'], output=args['output'], repo=args['repo'])
    else:
        args.pop('config_path'); args.pop('config_sha256')
        result = capture(**args)
    print(json.dumps(result, sort_keys=True))
