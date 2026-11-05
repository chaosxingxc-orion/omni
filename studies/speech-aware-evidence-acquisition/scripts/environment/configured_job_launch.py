#!/usr/bin/env python3
"""Pinned JSON-to-Docker launch. Scientific admission remains in shared jobs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

SCHEMA = 'saea-configured-job-launch-v1'
VERSION = '1.0.0'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv, **kwargs):
    return subprocess.run(argv, text=True, capture_output=True, **kwargs)


def absolute(value):
    require(isinstance(value, str) and value.startswith('/') and '..' not in Path(value).parts
            and not any(c in value for c in ('\x00', '\n', ',')), 'clean absolute path required')
    return Path(value)


def persist(path, value, replace=False):
    path = Path(path)
    target = path.with_name(path.name + '.tmp-' + str(os.getpid())) if replace else path
    with target.open('x') as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    if replace:
        os.replace(target, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_config(path):
    doc = json.loads(Path(path).read_bytes())
    keys = {'schema', 'container_name', 'image', 'repo_path', 'expected_commit', 'receipt_dir',
            'argv', 'working_directory', 'network', 'pid_host', 'cpuset_cpus', 'environment',
            'mounts', 'create_directories', 'allowed_create_roots', 'required_inputs', 'monitor_interval_seconds'}
    require(isinstance(doc, dict) and set(doc) - {'generation_lease', 'fresh_outputs', 'source_identity', 'process_access'} == keys and doc['schema'] == SCHEMA, 'exact launcher schema required')
    for key, pattern in [('container_name', r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}'),
                         ('image', r'sha256:[a-f0-9]{64}'), ('expected_commit', r'[a-f0-9]{40}'),
                         ('cpuset_cpus', r'[0-9]+(?:[-,][0-9]+)*')]:
        require(isinstance(doc[key], str) and re.fullmatch(pattern, doc[key]), 'invalid ' + key)
    for key in ('repo_path', 'receipt_dir', 'working_directory'):
        absolute(doc[key])
    require(isinstance(doc['argv'], list) and doc['argv'] and all(isinstance(a, str) and a and '\x00' not in a for a in doc['argv']), 'argv array required')
    absolute(doc['argv'][0])
    require(doc['network'] in ('host', 'none') and type(doc['pid_host']) is bool, 'explicit network/process mode required')
    require(isinstance(doc['environment'], dict) and all(isinstance(k, str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', k)
            and isinstance(v, str) and '\x00' not in v for k, v in doc['environment'].items()), 'environment mapping required')
    require(doc['monitor_interval_seconds'] == 600, 'monitor every 600 seconds or terminal event')
    require(isinstance(doc['allowed_create_roots'], list) and isinstance(doc['create_directories'], list), 'explicit owned directories required')
    roots = [absolute(p).resolve() for p in doc['allowed_create_roots']]
    require(all(len(p.parts) >= 4 for p in roots), 'creation roots must be scoped to owned project')
    for p in [doc['receipt_dir'], *doc['create_directories']]:
        require(any(absolute(p).resolve().is_relative_to(r) for r in roots), 'creation escapes owned roots')
    require(isinstance(doc['mounts'], list) and doc['mounts'], 'explicit mounts required')
    targets = set()
    for mount in doc['mounts']:
        require(set(mount) == {'source', 'target', 'readonly'} and type(mount['readonly']) is bool, 'exact mount required')
        absolute(mount['source']); target = absolute(mount['target'])
        require(str(target) not in targets and target != Path('/'), 'unique non-root mount targets required')
        targets.add(str(target))
    require(isinstance(doc['required_inputs'], list), 'pinned input list required')
    for pin in doc['required_inputs']:
        require(set(pin) == {'path', 'sha256'} and isinstance(pin['sha256'], str) and re.fullmatch(r'[a-f0-9]{64}', pin['sha256']), 'exact input pin required')
        absolute(pin['path'])
    if 'fresh_outputs' in doc:
        outputs = doc['fresh_outputs']
        require(isinstance(outputs, list) and outputs and all(isinstance(p, str) for p in outputs)
                and len(outputs) == len(set(outputs)), 'distinct fresh output paths required')
        for output in outputs:
            mapped = output_path(doc, output)
            require(any(mapped.resolve().is_relative_to(r) for r in roots),
                    'fresh output escapes owned roots')
    if 'generation_lease' in doc:
        lease = doc['generation_lease']
        require(isinstance(lease, dict) and set(lease) == {'lock_directory'}, 'exact generation lease check required')
        locks = absolute(lease['lock_directory'])
        data = absolute(doc['environment'].get('SPEECHRL_DATA_DIR'))
        logical = data / 'locks'
        mounts = [m for m in doc['mounts'] if logical.is_relative_to(Path(m['target']))]
        require(mounts, 'generation lease must map the runtime data locks')
        mount = max(mounts, key=lambda m: len(Path(m['target']).parts))
        mapped = Path(mount['source']) / logical.relative_to(mount['target'])
        require(locks == mapped, 'generation lease directory differs from runtime mount')
    identity = doc.get('source_identity')
    if identity is not None:
        require(isinstance(identity, dict) and set(identity) == {'mode', 'manifest_sha256'}
                and identity['mode'] == 'image-release-v1'
                and isinstance(identity['manifest_sha256'], str)
                and re.fullmatch('[a-f0-9]{64}', identity['manifest_sha256']), 'exact release source identity required')
        require(doc['repo_path'] == '/opt/saea-release/studies/speech-aware-evidence-acquisition', 'release repository path differs')
        require(doc['environment'].get('SAEA_RELEASE_MANIFEST') == '/opt/saea-release/source-manifest.json'
                and doc['environment'].get('SAEA_RELEASE_MANIFEST_SHA256') == identity['manifest_sha256'], 'release manifest environment differs')
        require(doc['environment'].get('PYTHONPATH') == '/opt/saea-release/studies/speech-aware-evidence-acquisition/src'
                and doc['environment'].get('PYTHONDONTWRITEBYTECODE') == '1', 'release Python source environment differs')
        require(doc['argv'][0] in ('/opt/saea-venv/bin/python', '/usr/local/python3.12.13/bin/python3')
                and len(doc['argv']) >= 2 and (doc['argv'][1].startswith('/opt/saea-release/studies/speech-aware-evidence-acquisition/scripts/') or doc['argv'][1].startswith('/opt/saea-release/scripts/npu/'))
                and '..' not in Path(doc['argv'][1]).parts, 'release entrypoint must be a sealed study script')
        require(not any(Path('/opt/saea-release').is_relative_to(Path(m['target'])) or Path(m['target']).is_relative_to('/opt/saea-release') for m in doc['mounts']), 'release source cannot be mounted over')
    if identity is not None:
        require(not any(k.startswith(('PYTHON', 'LD_', 'DYLD_')) and k not in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'LD_LIBRARY_PATH') for k in doc['environment']), 'release interpreter environment override forbidden')
        if 'SAEA_EXPERIMENT_MANIFEST' in doc['environment']:
            asset=doc['environment']['SAEA_EXPERIMENT_MANIFEST'];absolute(asset)
            require(re.fullmatch('[a-f0-9]{64}',doc['environment'].get('SAEA_EXPERIMENT_MANIFEST_SHA256','')),'external experiment manifest pin required')
            matches=[m for m in doc['mounts'] if Path(asset).is_relative_to(Path(m['target']))]
            require(matches and max(matches,key=lambda m:len(Path(m['target']).parts))['readonly'],'external experiment manifest requires readonly mount')
        require(doc['working_directory'] == doc['repo_path'], 'release working directory must be sealed')
        for mount in doc['mounts']:
            target=Path(mount['target'])
            require(not any(target.is_relative_to(p) or Path(p).is_relative_to(target) for p in ('/opt', '/usr', '/bin', '/lib', '/lib64', '/etc', '/root')), 'release dependency mount override forbidden')
    if 'process_access' in doc:
        require(doc['process_access'] == 'resident-identity-read-v1' and identity is not None
                and doc['pid_host'] and doc['network'] == 'host',
                'resident process inspection requires release, host PID and network')
    return doc


def output_path(doc, value):
    """Map a container output through its most specific writable bind mount."""
    target = absolute(value)
    mounts = [m for m in doc['mounts'] if target.is_relative_to(Path(m['target']))]
    require(mounts, 'fresh output requires a bind mount')
    mount = max(mounts, key=lambda m: len(Path(m['target']).parts))
    require(not mount['readonly'] and target != Path(mount['target']),
            'fresh output must be below a writable bind mount')
    source = Path(mount['source'])
    mapped = source / target.relative_to(mount['target'])
    require(mapped.resolve().is_relative_to(source.resolve()), 'fresh output escapes mount source')
    return mapped


def fresh_output_conflicts(doc):
    """Read-only advisory checks; runtime O_EXCL/attempt guards remain authoritative."""
    conflicts = []
    for target in doc.get('fresh_outputs', []):
        mapped = output_path(doc, target)
        if os.path.lexists(mapped):
            conflicts.append({'container_path': target, 'host_path': str(mapped)})
    return conflicts


def generation_readiness(doc):
    """Read-only advisory check; runtime still owns the atomic generation lease.

    Plans may be made while a predecessor holds the lane. Only an actual new
    launch requires clear state, after queue dependencies have been satisfied.
    No metadata is parsed, cleared, or treated as proof of request quiescence.
    """
    if 'generation_lease' not in doc:
        return {'state': 'not-configured', 'automatic_retry': False}
    locks = Path(doc['generation_lease']['lock_directory'])
    result = {'lock_directory': str(locks), 'automatic_retry': False}
    try:
        require(locks.is_dir(), 'generation lock directory absent')
        guard = locks / 'generation.guard'
        if os.path.lexists(guard):
            require(not guard.is_symlink() and guard.is_file(), 'regular generation guard required')
            # O_RDONLY and no O_CREAT: never replace an existing guard inode.
            fd = os.open(guard, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return {**result, 'state': 'lease-held'}
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        if os.path.lexists(locks / 'engine-transition.pending.json'):
            return {**result, 'state': 'unfinished-engine-transition'}
        if os.path.lexists(locks / 'flight.lock'):
            return {**result, 'state': 'unreconciled-metadata',
                    'required_action': 'record PID/socket/contact quiescence using audited reconciliation'}
        return {**result, 'state': 'clear'}
    except (OSError, ValueError) as exc:
        return {**result, 'state': 'unreadable', 'error_type': type(exc).__name__}


def mount_checks(doc):
    for mount in doc['mounts']:
        source, target = Path(mount['source']), Path(mount['target'])
        require(source.exists(), 'missing mount source: ' + str(source))
        ancestors = [m for m in doc['mounts'] if target != Path(m['target']) and target.is_relative_to(Path(m['target']))]
        if ancestors:
            parent = max(ancestors, key=lambda m: len(Path(m['target']).parts))
            underneath = Path(parent['source']) / target.relative_to(parent['target'])
            if parent['readonly']:
                require(underneath.exists(), 'missing nested mountpoint under read-only parent: ' + str(underneath))
            if underneath.exists():
                require(underneath.is_dir() == source.is_dir(), 'mountpoint type differs')


def preflight(doc):
    identity = doc.get('source_identity')
    if identity is None:
        head = run(['git', '-C', doc['repo_path'], 'rev-parse', 'HEAD'])
        clean = run(['git', '-C', doc['repo_path'], 'status', '--porcelain'])
        require(head.returncode == clean.returncode == 0 and head.stdout.strip() == doc['expected_commit'] and not clean.stdout.strip(), 'clean pinned execution repository required')
    else:
        check = ('import hashlib,json,subprocess,sys; p="/opt/saea-release/source-manifest.json"; '
                 'b=open(p,"rb").read(); assert hashlib.sha256(b).hexdigest()==sys.argv[1]; '
                 'assert json.loads(b)["commit"]==sys.argv[2]; '
                 'subprocess.run([sys.executable,"/opt/saea-release/verify-release.py",p,"/opt/saea-release"],check=True)')
        checked = run(['docker','run','--rm','--network','none','--entrypoint','/opt/saea-venv/bin/python',doc['image'],'-c',check,identity['manifest_sha256'],doc['expected_commit']])
        require(checked.returncode == 0, 'release image/source manifest verification failed')
    for pin in doc['required_inputs']:
        require(Path(pin['path']).is_file() and digest(pin['path']) == pin['sha256'], 'required input absent or changed: ' + pin['path'])
    mount_checks(doc)
    conflicts = fresh_output_conflicts(doc)
    require(not conflicts, 'fresh output conflicts; inspect and preserve prior attempts: '
            + json.dumps(conflicts, sort_keys=True))


def command(doc, config_sha, runner_sha):
    args = ['docker', 'run', '-d', '--name', doc['container_name'], '--network', doc['network'], '--cpuset-cpus', doc['cpuset_cpus'],
            '--label', 'saea.launch.config=' + config_sha, '--label', 'saea.launch.runner=' + runner_sha]
    if doc.get('source_identity') is not None:
        args += ['--read-only', '--tmpfs', '/tmp:rw,nosuid,nodev,size=1g', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges']
    if doc.get('process_access') == 'resident-identity-read-v1':
        args += ['--cap-add', 'SYS_PTRACE']
    if doc['pid_host']:
        args += ['--pid', 'host']
    for m in sorted(doc['mounts'], key=lambda m: len(Path(m['target']).parts)):
        args += ['--mount', 'type=bind,src=' + m['source'] + ',dst=' + m['target'] + (',readonly' if m['readonly'] else '')]
    for key, value in sorted(doc['environment'].items()):
        args += ['-e', key + '=' + value]
    return args + ['-w', doc['working_directory'], '--entrypoint', doc['argv'][0], doc['image'], *doc['argv'][1:]]


def status(doc, config_sha, runner_sha):
    intent = Path(doc['receipt_dir']) / 'launch-intent.json'
    if intent.exists():
        old = json.loads(intent.read_bytes())
        require(old['config_sha256'] == config_sha and old['runner_sha256'] == runner_sha, 'launch receipt identity changed')
    result = run(['docker', 'container', 'inspect', doc['container_name']])
    if result.returncode:
        require('No such container' in result.stderr or 'No such object' in result.stderr, 'Docker inspection failed: ' + result.stderr[-500:])
        return {'state': 'unresolved-launch-intent' if intent.exists() else 'not-launched', 'automatic_retry': False}
    container, = json.loads(result.stdout)
    labels = container['Config'].get('Labels') or {}
    require(container['Image'] == doc['image'] and labels.get('saea.launch.config') == config_sha
            and labels.get('saea.launch.runner') == runner_sha, 'existing container identity differs')
    state = container['State']
    return {'state': state['Status'], 'container_id': container['Id'], 'running': state['Running'],
            'exit_code': state['ExitCode'], 'started_at': state['StartedAt'], 'finished_at': state['FinishedAt'],
            'scientific_success': 'verify shared runtime artifacts separately', 'automatic_retry': False}


def execute(path, mode):
    doc = load_config(path)
    config_sha = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    runner_sha = digest(__file__)
    root = Path(doc['receipt_dir'])
    if mode == 'plan':
        preflight(doc)
        return {'version': VERSION, 'config_sha256': config_sha, 'runner_sha256': runner_sha,
                'command': command(doc, config_sha, runner_sha), 'new_model_calls': 0,
                'generation_readiness': generation_readiness(doc)}
    if mode == 'layout':
        for p in sorted(doc['create_directories'], key=lambda p: len(Path(p).parts)):
            Path(p).mkdir(parents=True, exist_ok=True)
        mount_checks(doc)
        require(root.is_dir(), 'receipt directory must be explicitly created')
        return {'layout_ready': True, 'new_model_calls': 0}
    if mode in ('status', 'watch'):
        while True:
            observed = status(doc, config_sha, runner_sha)
            if mode == 'status':
                return observed
            require(root.is_dir(), 'receipt directory required')
            persist(root / 'status.json', {**observed, 'observed_at_unix': time.time()}, replace=True)
            if observed['state'] not in ('running', 'created', 'restarting', 'paused'):
                return observed
            try:
                run(['docker', 'wait', doc['container_name']], timeout=doc['monitor_interval_seconds'])
            except subprocess.TimeoutExpired:
                pass
            # A created container may have failed OCI init; avoid a fast wait loop.
            if observed['state'] == 'created':
                return observed
    require(mode == 'launch' and root.is_dir(), 'layout required before launch')
    with (root / 'launch.guard').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = status(doc, config_sha, runner_sha)
        if observed['state'] != 'not-launched':
            return {**observed, 'reused_launch': True}
        preflight(doc)
        readiness = generation_readiness(doc)
        require(readiness['state'] in ('clear', 'not-configured'),
                'generation launch not ready: ' + readiness['state'] + '; inspect lane custody, never auto-clear or retry')
        persist(root / 'launch-intent.json', {'schema': SCHEMA, 'version': VERSION, 'config_sha256': config_sha,
                'runner_sha256': runner_sha, 'expected_commit': doc['expected_commit'], 'automatic_retry': False})
        result = run(command(doc, config_sha, runner_sha))
        persist(root / 'launch-result.json', {'returncode': result.returncode, 'stdout': result.stdout,
                'stderr': result.stderr, 'automatic_retry': False})
        require(result.returncode == 0, 'Docker launch failed; intent retained: ' + result.stderr[-500:])
        return status(doc, config_sha, runner_sha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'layout', 'launch', 'status', 'watch'))
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.config, args.mode), sort_keys=True))
