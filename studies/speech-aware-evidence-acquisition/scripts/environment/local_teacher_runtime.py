"""Create/verify a pinned local Python environment, then run the durable queue.

The configured interpreter is local. No SSH, model call, or image download occurs
here. Bootstrap is explicit; normal execution verifies the receipt and never pip
installs. API credentials remain inherited environment values.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def config(path):
    value = json.loads(Path(path).read_bytes())
    require(set(value) == {'schema', 'python', 'python_sha256', 'venv', 'requirements',
                           'requirements_sha256', 'python_version', 'machine'}, 'exact runtime configuration required')
    require(value['schema'] == 'saea-local-teacher-runtime-v1', 'runtime schema differs')
    for key in ('python', 'venv', 'requirements'):
        require(Path(value[key]).is_absolute(), 'absolute runtime paths required')
    require(sha(value['python']) == value['python_sha256'], 'base interpreter changed')
    require(sha(value['requirements']) == value['requirements_sha256'], 'requirements changed')
    actual = json.loads(subprocess.check_output([value['python'], '-c',
        'import json,platform;print(json.dumps([platform.python_version(),platform.machine()]))']))
    require(actual == [value['python_version'], value['machine']], 'Python version or architecture differs')
    return value


def requirements(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, rest = line.split('==', 1)
        version, hasharg = rest.split()
        require(hasharg.startswith('--hash=sha256:') and len(hasharg.split(':')[-1]) == 64,
                'exact package versions and wheel hashes required')
        result[name] = version
    require(bool(result), 'nonempty dependency lock required')
    return result


def inspect_runtime(value):
    python = Path(value['venv'])/'bin/python'
    code = '''import json,importlib.metadata,platform,sys
names=json.loads(sys.argv[1]);print(json.dumps({'python_version':platform.python_version(),
'machine':platform.machine(),'packages':{n:importlib.metadata.version(n) for n in names}}))'''
    want = requirements(value['requirements'])
    got = json.loads(subprocess.check_output([str(python), '-c', code, json.dumps(list(want))]))
    require(got == {'python_version': value['python_version'], 'machine': value['machine'], 'packages': want},
            'runtime dependencies differ')
    return got


def receipt(value):
    return {'schema': 'saea-local-teacher-runtime-receipt-v1', 'config': value,
            'runtime': inspect_runtime(value), 'launcher_sha256': sha(__file__)}


def bootstrap(value, wheelhouse=None):
    target = Path(value['venv'])
    require(not target.exists(), 'fresh venv required; existing environments are never modified')
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([value['python'], '-m', 'venv', str(target)], check=True)
    args = [str(target/'bin/python'), '-m', 'pip', 'install', '--require-hashes', '--no-deps',
            '--only-binary=:all:', '-r', value['requirements']]
    if wheelhouse:
        args.extend(['--no-index', '--find-links', str(Path(wheelhouse).resolve())])
    subprocess.run(args, check=True)
    with (target/'teacher-runtime.json').open('x') as stream:
        json.dump(receipt(value), stream, sort_keys=True, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def verify(value):
    require(json.loads((Path(value['venv'])/'teacher-runtime.json').read_bytes()) == receipt(value),
            'runtime receipt differs; rebuild a fresh versioned environment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('bootstrap', 'check', 'queue'))
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--wheelhouse', type=Path)
    args, remaining = parser.parse_known_args(); value = config(args.runtime)
    require(args.action == 'queue' or not remaining, 'unexpected runtime arguments')
    if args.action == 'bootstrap':
        bootstrap(value, args.wheelhouse)
    else:
        verify(value)
    if args.action == 'queue':
        command = remaining[1:] if remaining[:1] == ['--'] else remaining
        require(bool(command), 'queue action required')
        python = str(Path(value['venv'])/'bin/python')
        queue = Path(__file__).with_name('configured_local_teacher_queue.py')
        os.execve(python, [python, str(queue), *command], {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    print(json.dumps({'status': 'verified', 'runtime': inspect_runtime(value)}))


if __name__ == '__main__':
    main()
