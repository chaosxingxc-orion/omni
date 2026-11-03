#!/usr/bin/env python3
"""Bind a completed CPU preflight to one compatible successor configuration."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

SCHEMA = 'saea-preflight-binding-v1'


def sibling(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = sibling('configured_job_launch')
probe = sibling('configured_environment_preflight')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read_pin(value):
    require(set(value) == {'path', 'sha256'}, 'exact file pin required')
    path = launcher.absolute(value['path'])
    require(path.is_file() and sha(path) == value['sha256'], 'binding input absent or changed: ' + str(path))
    return path


def overlap(a, b):
    a, b = Path(a), Path(b)
    return a.is_relative_to(b) or b.is_relative_to(a)


def host_path(doc, target):
    target = launcher.absolute(target)
    matches = [m for m in doc['mounts'] if target.is_relative_to(m['target'])]
    require(matches, 'preflight artifact must be explicitly mounted')
    mount = max(matches, key=lambda m: len(Path(m['target']).parts))
    return Path(mount['source']) / target.relative_to(mount['target']), mount


def source_artifacts(source, ready):
    args = source['argv']
    require(len(args) == 6 and args[2] == '--config' and args[4] == '--output',
            'direct configured environment preflight entry required')
    script, script_mount = host_path(source, args[1])
    config, config_mount = host_path(source, args[3])
    output, output_mount = host_path(source, args[5])
    require(script.name == 'configured_environment_preflight.py' and script_mount['readonly']
            and config_mount['readonly'] and not output_mount['readonly'], 'preflight mount modes differ')
    require(Path(ready).resolve() == (output / 'ready.json').resolve(), 'ready does not belong to source output')
    require({'path': str(config), 'sha256': sha(config)} in source['required_inputs'], 'source must pin probe configuration')
    return script, config, output_mount


def compatibility(source, successor, config, output_mount):
    require(source['network'] == 'none' and not source['pid_host'], 'CPU preflight must isolate network and processes')
    require(source['image'] == successor['image'], 'successor image drift')
    require(source['environment'] == successor['environment'], 'successor environment drift')
    require(source['working_directory'] == successor['working_directory'], 'successor working directory drift')
    require(source['repo_path'] == successor['repo_path'] and source['expected_commit'] == successor['expected_commit'],
            'successor execution repository drift')
    require(all(source['environment'].get(k) == v for k, v in config['required_environment'].items()),
            'source launch environment does not match probe')
    interpreters = [item['python'] for item in config['interpreters']]
    require(successor['argv'][0] in interpreters, 'successor interpreter was not probed')
    dependencies = [m for m in source['mounts'] if m != output_mount]
    require(all(m['readonly'] for m in dependencies), 'preflight dependencies must be read-only mounts')
    declarations = probe.runtime_paths(config)
    require(not (set(source['environment']) & set(probe.RUNTIME_PATH_KINDS)) - set(declarations),
            'runtime environment paths must be explicitly declared')
    protected = ['/usr', '/lib', '/lib64', '/bin', '/sbin', '/opt',
                 source['working_directory'], *interpreters, *(m['target'] for m in dependencies)]
    for key, value in source['environment'].items():
        if key not in declarations:
            protected.extend(part for part in value.split(':') if part.startswith('/'))
    require(not any(overlap(output_mount['target'], path) for path in protected),
            'preflight output overlaps dependency paths')
    for mount in dependencies:
        require(mount in successor['mounts'], 'successor dependency mount drift: ' + mount['target'])
    for mount in successor['mounts']:
        if mount not in dependencies:
            require(not any(overlap(mount['target'], path) for path in protected),
                    'successor mount shadows dependency: ' + mount['target'])
    data = declarations.get('SPEECHRL_DATA_DIR')
    if data:
        root = Path(data['path'])
        require(not any(root.is_relative_to(path) for path in protected),
                'data root cannot be a dependency path')
        require(any(Path(m['target']).is_relative_to(root) for m in dependencies),
                'data root requires read-only dependency leaves')
    timing = declarations.get('SAEA_STAGE_TIMING_PATH')
    if timing:
        sink, mount = host_path(successor, timing['path'])
        require(mount['target'] == timing['mount'] and not mount['readonly'],
                'timing file requires its declared writable parent mount')
        require(not Path(mount['source']).exists() or Path(mount['source']).is_dir(),
                'timing mount source must be a directory')
        require(not sink.exists() or sink.is_file(), 'timing output must be a file')
        require(not any(overlap(timing['path'], p) for p in protected)
                and not overlap(timing['path'], output_mount['target']),
                'timing output overlaps dependency or preflight evidence')
        require(sink.resolve().is_relative_to(Path(mount['source']).resolve()),
                'timing output escapes its declared host mount')
        require(not any(overlap(Path(mount['source']).resolve(), Path(m['source']).resolve())
                        for m in [*dependencies, output_mount]),
                'timing output aliases dependency or preflight evidence')
    footprint = {'image': source['image'], 'environment': source['environment'],
            'working_directory': source['working_directory'], 'interpreters': interpreters,
            'dependency_mounts': dependencies, 'expected_commit': source['expected_commit']}
    if declarations:
        footprint['runtime_paths'] = declarations
    return footprint


def validate_ready(source, ready):
    ready = Path(ready)
    require(ready.is_file(), 'successful readiness is absent')
    script, config_path, output_mount = source_artifacts(source, ready)
    config = probe.load_config(config_path)
    readiness = json.loads(ready.read_bytes())
    result_path = ready.with_name('result.json')
    require(result_path.is_file() and readiness.get('result_sha256') == sha(result_path),
            'preflight result absent or changed')
    result = json.loads(result_path.read_bytes())
    require(readiness.get('schema') == probe.SCHEMA and result.get('schema') == probe.SCHEMA
            and readiness.get('ok') is True and result.get('ok') is True
            and readiness.get('config_sha256') == result.get('config_sha256') == probe.digest(config)
            and result.get('script_sha256') == sha(script)
            and readiness.get('new_model_calls') == result.get('new_model_calls') == 0,
            'successful preflight provenance differs')
    checks = result.get('checks', [])
    require(checks and all(c.get('ok') is True for c in checks), 'preflight contains failed checks')
    actual = [(c['interpreter'], c.get('configured_python')) for c in checks if 'interpreter' in c]
    expected = [(i['name'], i['python']) for i in config['interpreters']]
    require(actual == expected, 'preflight interpreter evidence differs')
    file_inputs = []
    for required in config.get('required_files', []):
        host, mount = host_path(source, required['path'])
        require(mount != output_mount and mount['readonly'],
                'required file must belong to a read-only dependency mount')
        require(host.resolve().is_relative_to(Path(mount['source']).resolve()),
                'required file escapes dependency mount')
        require(host.is_file() and sha(host) == required['sha256'],
                'required dependency file absent or changed: ' + required['path'])
        file_inputs.append(pin(host))
    return config, output_mount, [pin(ready), pin(result_path), pin(config_path), pin(script), *file_inputs]


def completed_source(source):
    root = Path(source['receipt_dir'])
    intent_path = root / 'launch-intent.json'
    require(intent_path.is_file(), 'preflight launch intent absent')
    intent = json.loads(intent_path.read_bytes())
    require(intent.get('config_sha256') == probe.digest(source)
            and intent.get('runner_sha256') == launcher.digest(launcher.__file__), 'preflight launch identity differs')
    observed = launcher.status(source, probe.digest(source), intent['runner_sha256'])
    require(observed['state'] == 'exited' and observed.get('running') is False
            and observed.get('exit_code') == 0, 'preflight container is not successfully terminal')
    return pin(intent_path), observed['container_id']


def bind(source_path, ready, successor_path, output):
    source = launcher.load_config(source_path)
    successor = launcher.load_config(successor_path)
    config, output_mount, inputs = validate_ready(source, ready)
    footprint = compatibility(source, successor, config, output_mount)
    require({'path': str(Path(ready).resolve()), 'sha256': sha(ready)} in successor['required_inputs'],
            'successor must pin successful readiness')
    intent, container_id = completed_source(source)
    binding = {'schema': SCHEMA, 'source': pin(source_path), 'successor': pin(successor_path),
               'inputs': [*inputs, intent], 'source_container_id': container_id,
               'compatibility': footprint, 'helper_sha256': sha(__file__),
               'launcher_sha256': sha(launcher.__file__), 'probe_module_sha256': sha(probe.__file__),
               'automatic_retry': False}
    launcher.persist(Path(output), binding)
    return binding


def verify(binding_path, successor_path):
    binding = json.loads(Path(binding_path).read_bytes())
    require(binding.get('schema') == SCHEMA and binding.get('helper_sha256') == sha(__file__)
            and binding.get('launcher_sha256') == sha(launcher.__file__)
            and binding.get('probe_module_sha256') == sha(probe.__file__), 'binding implementation identity differs')
    source_path = read_pin(binding['source'])
    pinned_successor = read_pin(binding['successor'])
    require(Path(successor_path).resolve() == pinned_successor.resolve(), 'successor configuration path differs')
    inputs = [read_pin(item) for item in binding['inputs']]
    source = launcher.load_config(source_path)
    successor = launcher.load_config(pinned_successor)
    intent = json.loads(inputs[-1].read_bytes())
    require(intent.get('config_sha256') == probe.digest(source)
            and intent.get('runner_sha256') == sha(launcher.__file__),
            'bound source launch identity differs')
    config, output_mount, _ = validate_ready(source, inputs[0])
    require(compatibility(source, successor, config, output_mount) == binding['compatibility'],
            'environment compatibility binding differs')
    return successor


def execute(binding_path, successor_path, mode):
    require(mode in ('plan', 'launch'), 'binding supports plan or launch only')
    verify(binding_path, successor_path)
    # Original launcher owns guard, intent, container reconciliation and unknown-start refusal.
    return launcher.execute(successor_path, mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('bind', 'plan', 'launch'))
    parser.add_argument('--successor', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--ready', type=Path)
    args = parser.parse_args()
    if args.mode == 'bind':
        if not args.source or not args.ready:
            parser.error('--source and --ready required')
        value = bind(args.source, args.ready, args.successor, args.binding)
    else:
        value = execute(args.binding, args.successor, args.mode)
    print(json.dumps(value, sort_keys=True))
