#!/usr/bin/env python3
"""CPU dependency checks; a successful receipt can be pinned by the launcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

SCHEMA = 'saea-environment-preflight-v1'
RUNTIME_PATH_KINDS = {'SPEECHRL_DATA_DIR': 'data-root', 'SAEA_STAGE_TIMING_PATH': 'timing-file'}
PROBE = '''import importlib, importlib.metadata, json, sys
spec = json.loads(sys.argv[1])
checks = []
for module in spec['imports']:
    try:
        importlib.import_module(module)
        checks.append({'module': module, 'ok': True})
    except Exception as exc:
        checks.append({'module': module, 'ok': False, 'error_type': type(exc).__name__})
for package, expected in spec['distributions'].items():
    try:
        actual = importlib.metadata.version(package)
        checks.append({'distribution': package, 'expected': expected, 'actual': actual, 'ok': actual == expected})
    except Exception as exc:
        checks.append({'distribution': package, 'ok': False, 'error_type': type(exc).__name__})
print('SAEA_PREFLIGHT=' + json.dumps({'checks': checks, 'ok': all(c['ok'] for c in checks),
                                    'executable': sys.executable, 'python_version': sys.version.split()[0]}))
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def canonical_runtime_path(value):
    require(isinstance(value, str) and value.startswith('/') and not value.startswith('//') and value != '/'
            and str(Path(value)) == value and '..' not in Path(value).parts
            and not any(c in value for c in ('\x00', '\n', '\r', ':', ',')),
            'runtime path must be canonical, absolute and non-root')
    return Path(value)


def runtime_paths(doc):
    declarations = doc.get('runtime_paths', {})
    require(isinstance(declarations, dict) and set(declarations) <= set(RUNTIME_PATH_KINDS),
            'only known runtime environment paths may be declared')
    for key, item in declarations.items():
        fields = {'kind', 'path', 'mount'} if key == 'SAEA_STAGE_TIMING_PATH' else {'kind', 'path'}
        require(isinstance(item, dict) and set(item) == fields
                and item.get('kind') == RUNTIME_PATH_KINDS[key], 'exact runtime path semantics required: ' + key)
        value = canonical_runtime_path(item['path'])
        require(doc['required_environment'].get(key) == str(value),
                'runtime path must match required environment: ' + key)
        if key == 'SAEA_STAGE_TIMING_PATH':
            parent = canonical_runtime_path(item['mount'])
            require(value != parent and value.is_relative_to(parent),
                    'timing file must be strictly below its declared mount')
    require(not (set(doc['required_environment']) & set(RUNTIME_PATH_KINDS)) - set(declarations),
            'runtime environment paths must be explicitly declared')
    return declarations


def load_config(path):
    doc = json.loads(Path(path).read_bytes())
    require(isinstance(doc, dict) and set(doc) - {'runtime_paths'} == {'schema', 'required_environment', 'interpreters', 'timeout_seconds'}, 'exact preflight schema required')
    require(doc['schema'] == SCHEMA, 'unsupported preflight schema')
    require(type(doc['timeout_seconds']) is int and 1 <= doc['timeout_seconds'] <= 300, 'bounded timeout required')
    require(isinstance(doc['required_environment'], dict) and all(
        isinstance(k, str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', k) and isinstance(v, str) and v
        for k, v in doc['required_environment'].items()), 'explicit environment values required')
    runtime_paths(doc)
    require(isinstance(doc['interpreters'], list) and doc['interpreters'], 'interpreter checks required')
    names = set()
    for item in doc['interpreters']:
        require(isinstance(item, dict) and set(item) == {'name', 'python', 'environment_key', 'imports', 'distributions'}, 'exact interpreter schema required')
        require(isinstance(item['name'], str) and re.fullmatch(r'[a-z][a-z0-9_-]*', item['name']) and item['name'] not in names, 'unique interpreter name required')
        names.add(item['name'])
        require(isinstance(item['python'], str) and Path(item['python']).is_absolute() and '\x00' not in item['python'], 'absolute interpreter required')
        require(item['environment_key'] is None or (isinstance(item['environment_key'], str)
                and doc['required_environment'].get(item['environment_key']) == item['python']),
                'interpreter selection must match required environment')
        require(isinstance(item['imports'], list) and item['imports'] and all(isinstance(x, str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*', x) for x in item['imports']), 'module names required')
        require(isinstance(item['distributions'], dict) and all(isinstance(k, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', k) and isinstance(v, str) and v for k, v in item['distributions'].items()), 'exact distribution versions required')
    return doc


def persist(path, doc):
    with path.open('x') as handle:
        handle.write(json.dumps(doc, sort_keys=True, indent=2) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def execute(config, output):
    doc = load_config(config)
    started = time.monotonic()
    output = Path(output)
    # A unique directory preserves old failure receipts and concurrent invocations.
    output.mkdir(exist_ok=False)
    checks = [{'environment': key, 'ok': os.environ.get(key) == value}
              for key, value in sorted(doc['required_environment'].items())]
    if all(check['ok'] for check in checks):
        for spec in doc['interpreters']:
            try:
                result = subprocess.run([spec['python'], '-B', '-c', PROBE, json.dumps(spec)],
                                        text=True, capture_output=True, timeout=doc['timeout_seconds'],
                                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
                lines = [line[len('SAEA_PREFLIGHT='):] for line in result.stdout.splitlines() if line.startswith('SAEA_PREFLIGHT=')]
                require(result.returncode == 0 and len(lines) == 1, 'interpreter probe failed')
                probe = json.loads(lines[0])
                require(type(probe['ok']) is bool and isinstance(probe['checks'], list) and probe['checks'], 'invalid probe response')
                checks.append({'interpreter': spec['name'], 'configured_python': spec['python'], **probe})
            except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as exc:
                checks.append({'interpreter': spec['name'], 'configured_python': spec['python'],
                               'ok': False, 'error_type': type(exc).__name__})
    ok = all(check['ok'] for check in checks)
    result = {'schema': SCHEMA, 'config_sha256': digest(doc), 'ok': ok,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'checks': checks, 'elapsed_seconds': time.monotonic() - started,
              'new_model_calls': 0, 'automatic_retry': False}
    persist(output / 'result.json', result)
    if ok:
        persist(output / 'ready.json', {'schema': SCHEMA, 'config_sha256': digest(doc),
                                      'result_sha256': hashlib.sha256((output / 'result.json').read_bytes()).hexdigest(),
                                      'ok': True, 'new_model_calls': 0})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.config, args.output)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result['ok'] else 1)
