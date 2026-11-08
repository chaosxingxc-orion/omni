#!/usr/bin/env python3
"""Run the frozen TFG operator in a container with phase-specific file mounts."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

IMAGE = 'sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026'
INPUTS = ('matrix', 'members', 'metadata', 'parent', 'keys', 'gold')


def mount(source, dest, readonly=True):
    # Docker's --mount syntax has comma-separated fields.
    if ',' in str(source):
        raise ValueError('mount source cannot contain a comma')
    return ['--mount', f'type=bind,src={source},dst={dest}' + (',readonly' if readonly else '')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('rank', 'score'))
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path, help='existing writable directory, shared by both phases')
    p.add_argument('--name', required=True, help='unique Docker container name')
    p.add_argument('--image', default=IMAGE)
    p.add_argument('--cpus', default='4')
    p.add_argument('--memory', default='4g')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--user', default=f'{os.getuid()}:{os.getgid()}')
    args = p.parse_args()
    if args.threads < 1:
        p.error('threads must be positive')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]+', args.name):
        p.error('invalid Docker container name')
    cfg = json.loads(args.config.read_text())
    if set(cfg['inputs']) != set(INPUTS):
        p.error('config inputs must contain exactly: ' + ', '.join(INPUTS))
    output = args.output.resolve(strict=True)
    if not output.is_dir():
        p.error('output must be a directory')
    source_paths = {}
    for key in INPUTS:
        source = Path(cfg['inputs'][key]['path']).expanduser()
        if not source.is_absolute():
            source = args.config.resolve().parent / source
        # Gold need not exist for rank, but its resolved location must still be outside output.
        source = source.resolve(strict=(key != 'gold' or args.phase == 'score'))
        if source == output or output in source.parents:
            p.error(f'{key} input must be outside output')
        if key != 'gold' or args.phase == 'score':
            if not source.is_file():
                p.error(f'{key} input must be a file')
        source_paths[key] = source
        cfg['inputs'][key]['path'] = '/inputs/' + key
    if source_paths['gold'] in (source_paths[k] for k in INPUTS if k != 'gold'):
        p.error('gold must be a separate file')
    if args.phase == 'rank' and source_paths['gold'].exists():
        for key in INPUTS[:-1]:
            if os.path.samefile(source_paths['gold'], source_paths[key]):
                p.error('gold must not alias a ranking input')
    if args.phase == 'rank' and source_paths['gold'].exists():
        for directory, _, files in os.walk(output, followlinks=False):
            for filename in files:
                item = Path(directory) / filename
                if item.exists() and os.path.samefile(item, source_paths['gold']):
                    p.error('output must not contain an alias of gold')
    cfg['output'] = '/output'
    operator = Path(__file__).resolve().with_name('teacher_tfg_v2_cpu.py')
    cmd = ['docker', 'run', '--rm', '--name', args.name, '--network', 'none', '--user', args.user,
           '--cpus', args.cpus, '--memory', args.memory, '--pids-limit', '64', '--cap-drop', 'ALL',
           '--security-opt', 'no-new-privileges', '--read-only', '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m',
           '-e', 'OPENBLAS_NUM_THREADS=' + str(args.threads), '-e', 'OMP_NUM_THREADS=' + str(args.threads),
           '-e', 'PYTHONDONTWRITEBYTECODE=1']
    cmd += mount(output, '/output', False) + mount(operator, '/operator.py')
    for key in INPUTS:
        if key != 'gold' or args.phase == 'score':
            cmd += mount(source_paths[key], '/inputs/' + key)
    # Same canonical bytes in both phases, including the unmounted gold pin.
    with tempfile.TemporaryDirectory(prefix='teacher-tfg-config-') as tmp:
        path = Path(tmp) / 'config.json'
        path.write_text(json.dumps(cfg, sort_keys=True, indent=2) + '\n')
        os.chmod(tmp, 0o755)
        os.chmod(path, 0o644)
        cmd += mount(path, '/config.json')
        cmd += ['--entrypoint', '/opt/saea-venv/bin/python', args.image, '/operator.py', args.phase, '--config', '/config.json']
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
