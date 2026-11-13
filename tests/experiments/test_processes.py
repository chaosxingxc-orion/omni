"""Real detached controllers/workers; NPU-marked nodes use CPU-only fixtures."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_contract import REPO, config

CLI = REPO / 'scripts/experiments/experiment.py'


def invoke(path, mode, *args):
    return subprocess.run([sys.executable, str(CLI), mode, '--config', str(path), *args],
                          capture_output=True, text=True, timeout=10)


def save(path, cfg):
    path.write_text(json.dumps(cfg))


def wait_for(path, predicate=lambda x: True, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            value = json.loads(path.read_text())
            if predicate(value): return value
        time.sleep(0.05)
    raise AssertionError('timed out waiting for ' + str(path))


def start(path):
    for command in ('prepare', 'run'):
        result = invoke(path, command)
        assert result.returncode == 0, result.stdout + result.stderr


def root_for(path, cfg):
    return path.parent / 'runs' / 'experiments' / cfg['experiment']['id']


def node(tmp_path, name, resource='cpu', deps=(), seconds=0.3, code=0):
    return {'id': name, 'resource': resource, 'depends_on': list(deps),
            'argv': [sys.executable, '-c', f'import time; time.sleep({seconds}); raise SystemExit({code})'],
            'cwd': str(tmp_path)}


def test_real_parallel_dag_and_failure_isolation(tmp_path):
    path, cfg = config(tmp_path)
    cfg['tasks'] = [node(tmp_path, 'cpu'), node(tmp_path, 'npu', 'npu'),
                    node(tmp_path, 'bad', deps=['cpu'], code=3),
                    node(tmp_path, 'blocked', deps=['bad']), node(tmp_path, 'deep', deps=['blocked'])]
    save(path, cfg)
    start(path)
    root = root_for(path, cfg)
    wait_for(root / 'STATUS.json', lambda x: x['state'] == 'FAILED')
    cpu = json.loads((root / 'tasks/cpu/START.json').read_text())
    npu = json.loads((root / 'tasks/npu/START.json').read_text())
    cpu_exit = json.loads((root / 'tasks/cpu/EXIT.json').read_text())
    npu_exit = json.loads((root / 'tasks/npu/EXIT.json').read_text())
    assert max(cpu['at'], npu['at']) < min(cpu_exit['at'], npu_exit['at'])
    assert not (root / 'tasks/blocked/DISPATCH.json').exists()
    states = {t['id']: t['state'] for t in json.loads(invoke(path, 'status').stdout)['tasks']}
    assert states == {'cpu': 'SUCCEEDED', 'npu': 'SUCCEEDED', 'bad': 'FAILED', 'blocked': 'BLOCKED', 'deep': 'BLOCKED'}
    before = (root / 'tasks/npu/DISPATCH.json').read_bytes()
    assert invoke(path, 'run').returncode == 0
    wait_for(root / 'STATUS.json', lambda x: x['state'] == 'FAILED')
    assert (root / 'tasks/npu/DISPATCH.json').read_bytes() == before


def test_two_experiments_share_one_lane(tmp_path):
    p1, c1 = config(tmp_path, 'first')
    p2, c2 = config(tmp_path, 'second')
    for p, c in [(p1, c1), (p2, c2)]:
        c['tasks'] = [node(tmp_path, 'npu', 'npu', seconds=0.6)]
        save(p, c)
    start(p1); start(p2)
    r1, r2 = root_for(p1, c1), root_for(p2, c2)
    end1 = wait_for(r1 / 'tasks/npu/EXIT.json')
    end2 = wait_for(r2 / 'tasks/npu/EXIT.json')
    start1 = json.loads((r1 / 'tasks/npu/START.json').read_text())
    start2 = json.loads((r2 / 'tasks/npu/START.json').read_text())
    assert end1['at'] <= start2['at'] or end2['at'] <= start1['at']


def test_unknown_timeout_is_not_resubmitted(tmp_path):
    path, cfg = config(tmp_path)
    task = node(tmp_path, 'uncertain', 'npu', seconds=5)
    task['timeout_seconds'] = 0.15
    cfg['tasks'] = [task, node(tmp_path, 'independent')]
    save(path, cfg); start(path)
    root = root_for(path, cfg)
    wait_for(root / 'STATUS.json', lambda x: x['state'] == 'UNKNOWN')
    receipt = (root / 'tasks/uncertain/DISPATCH.json').read_bytes()
    assert json.loads((root / 'tasks/independent/EXIT.json').read_text())['state'] == 'SUCCEEDED'
    result = invoke(path, 'reconcile')
    assert json.loads(result.stdout)['redispatched'] == []
    assert invoke(path, 'run').returncode == 0
    wait_for(root / 'STATUS.json', lambda x: x['state'] == 'UNKNOWN')
    assert receipt == (root / 'tasks/uncertain/DISPATCH.json').read_bytes()
    assert (tmp_path / 'runs/lanes/fixture.guard.LEASE.json').exists()


def test_prepare_pins_config_and_manifest(tmp_path):
    path, cfg = config(tmp_path)
    assert invoke(path, 'plan').returncode == 0
    assert not (tmp_path / 'runs').exists()
    assert invoke(path, 'prepare').returncode == 0
    cfg['experiment']['thinking'] = True
    save(path, cfg)
    assert invoke(path, 'prepare').returncode != 0
    assert invoke(path, 'run').returncode != 0


def test_controller_death_preserves_unknown_and_does_not_relaunch(tmp_path):
    path, cfg = config(tmp_path)
    cfg['tasks'] = [node(tmp_path, 'one', seconds=1.0)]
    save(path, cfg); start(path)
    root = root_for(path, cfg)
    child = wait_for(root / 'tasks/one/CHILD.json')
    observed = json.loads(invoke(path, 'status').stdout)
    controller = observed['controller']['start']
    os.kill(controller['pid'], signal.SIGKILL)
    wait_for(root / 'tasks/one/EXIT.json')
    result = invoke(path, 'run')
    assert result.returncode != 0
    assert 'no EXIT' in result.stderr + result.stdout
    assert json.loads(invoke(path, 'reconcile').stdout)['redispatched'] == []


def test_npu_nonzero_never_releases_persistent_lane(tmp_path):
    path, cfg = config(tmp_path)
    cfg['tasks'] = [node(tmp_path, 'npu', 'npu', code=3)]
    save(path, cfg); start(path)
    root = root_for(path, cfg)
    terminal = wait_for(root / 'tasks/npu/EXIT.json')
    assert terminal['state'] == 'UNKNOWN'
    wait_for(root / 'STATUS.json', lambda x: x['state'] == 'UNKNOWN')
    from tools.experiments.config import load_config
    from tools.experiments.runtime import acquire_lane
    doc = load_config(path)
    assert acquire_lane(doc, doc['config']['tasks'][0]) is None
