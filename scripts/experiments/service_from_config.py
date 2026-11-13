#!/usr/bin/env python3
"""Start a configured Docker service once, with local durable launch receipts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.experiments.local_runtime_storage import check_paths


def inspect(cid):
    return json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]


def local(path):
    path = Path(path)
    assert path.is_absolute() and path.resolve() == path, 'noncanonical local path'
    parent = path
    while not parent.exists():
        parent = parent.parent
    fs = subprocess.check_output(['findmnt', '-T', str(parent), '-no', 'FSTYPE'], text=True).strip()
    assert fs in ('ext4', 'xfs', 'btrfs'), 'not a local persistent filesystem: ' + fs
    return path


def save(path, data):
    local(path.parent)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate(config):
    argv = config['create_argv']
    assert argv[:2] == ['docker', 'create']
    assert argv[argv.index('--name') + 1] == config['name']
    assert argv[argv.index('--log-driver') + 1] == 'none'
    assert argv[argv.index('--restart') + 1] == 'no'
    assert '--read-only' in argv
    assert argv[argv.index('--entrypoint') + 2] == config['image']
    roots = [local(p) for p in config['local_directories']]
    assert roots and local(config['receipt_root']) in roots
    records = check_paths({str(i): p for i, p in enumerate(roots)})
    budget = config['hot_workload_budget_bytes']
    assert isinstance(budget, int) and budget > 0, 'explicit workload budget required'
    for row in records:
        stat = os.statvfs(next(p for p in [Path(row['path']), *Path(row['path']).parents] if p.exists()))
        reserve = max(20 * 1024**3, stat.f_blocks * stat.f_frsize * 0.10)
        assert row['free_bytes'] >= reserve + budget, 'workload exceeds free space after reserve'
    environment = dict(argv[i + 1].split('=', 1) for i, item in enumerate(argv) if item == '-e')
    mandatory = {'HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TORCHINDUCTOR_CACHE_DIR',
                 'TRITON_CACHE_DIR', 'ASCEND_WORK_PATH', 'ASCEND_CACHE_PATH'}
    assert mandatory <= environment.keys(), 'missing required runtime paths'
    for key in config.get('required_directory_env', []):
        assert key in environment, 'missing directory environment: ' + key
        assert local(environment[key]) in roots, 'directory must be explicitly created: ' + key
    assert local(str(Path(environment['HOME']) / 'ascend/log')) in roots, 'default Ascend log directory missing'
    for i, arg in enumerate(argv):
        if arg == '-v':
            parts = argv[i + 1].split(':')
            if len(parts) == 3 and parts[2] == 'ro':
                continue
            source = local(parts[0])
            assert any(source.is_relative_to(root) for root in roots)
        elif arg == '--mount':
            mount = dict(pair.split('=', 1) for pair in argv[i + 1].split(',') if '=' in pair)
            source = local(mount['src'])
            assert any(source.is_relative_to(root) for root in roots)
    assert all(isinstance(arg, str) for arg in argv)
    return roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate', 'start', 'status'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    raw = Path(args.config).read_bytes()
    config = json.loads(raw)
    roots = validate(config)
    receipt = Path(config['receipt_root'])
    if args.command == 'validate':
        print(json.dumps({'valid': True, 'name': config['name']}))
        return
    if args.command == 'status':
        created = json.loads((receipt / 'CREATED.json').read_bytes())
        state = inspect(created['Id'])
        result = {'id': state['Id'], 'state': state['State']}
        try:
            with urllib.request.urlopen(config['endpoint'] + '/v1/models', timeout=8) as response:
                result['models'] = json.load(response)
        except Exception as error:
            result['health_error'] = str(error)
        print(json.dumps(result))
        return
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    with (receipt / 'launch.guard').open('a+b') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (receipt / 'START-INTENT.json').exists(), 'already attempted; inspect receipts, never blindly replay'
        old = inspect(config['predecessor_cid'])
        assert not old['State']['Running'] and old['State']['Pid'] == 0
        assert old['Image'] == config['image']
        old_cmd = old['Config']['Cmd']
        argv = config['create_argv']
        assert argv[argv.index('serve'):] == old_cmd[old_cmd.index('serve'):], 'model arguments changed'
        expected_devices = sorted(d['PathOnHost'] for d in old['HostConfig']['Devices'])
        actual_devices = sorted(argv[i + 1] for i, item in enumerate(argv) if item == '--device')
        assert expected_devices == actual_devices
        save(receipt / 'START-INTENT.json', {'config_sha256': hashlib.sha256(raw).hexdigest(), 'boot': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'predecessor': old['Id']})
        created = subprocess.run(argv, capture_output=True, text=True)
        save(receipt / 'CREATE-RESULT.json', {'returncode': created.returncode, 'stdout': created.stdout, 'stderr': created.stderr})
        assert created.returncode == 0, created.stderr
        cid = created.stdout.strip()
        state = inspect(cid)
        save(receipt / 'CREATED.json', state)
        assert state['Image'] == config['image'] and state['HostConfig']['ReadonlyRootfs']
        for mount in state['Mounts']:
            if mount['RW']:
                source = local(mount['Source'])
                assert any(source.is_relative_to(root) for root in roots)
        started = subprocess.run(['docker', 'start', cid], capture_output=True, text=True)
        save(receipt / 'START-RESULT.json', {'returncode': started.returncode, 'stdout': started.stdout, 'stderr': started.stderr})
        save(receipt / 'STARTED.json', inspect(cid))
        assert started.returncode == 0, started.stderr
        print(json.dumps({'started': True, 'cid': cid, 'name': config['name']}))


if __name__ == '__main__':
    main()
