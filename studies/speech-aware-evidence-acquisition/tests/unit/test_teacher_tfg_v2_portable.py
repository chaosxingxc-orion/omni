import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/analysis/run_teacher_tfg_v2_portable.py'
KEYS = ('matrix', 'members', 'metadata', 'parent', 'keys', 'gold')


def setup(tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    for key in KEYS:
        (inputs / key).write_text(key)
    output = tmp_path / 'output'
    output.mkdir()
    cfg = {'inputs': {k: {'path': str(inputs / k), 'sha256': 'placeholder'} for k in KEYS}}
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(cfg))
    docker = tmp_path / 'docker'
    docker.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
mounts = [args[i+1] for i, a in enumerate(args) if a == '--mount']
config_mount = next(x for x in mounts if ',dst=/config.json,' in x)
config_path = config_mount.split(',')[1][4:]
pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps({'argv': args, 'config': pathlib.Path(config_path).read_text()}))
''')
    docker.chmod(0o755)
    return config, output


def run(tmp_path, config, output, phase):
    capture = tmp_path / (phase + '.json')
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'], CAPTURE=str(capture))
    result = subprocess.run([sys.executable, str(SCRIPT), phase, '--config', str(config), '--output', str(output), '--name', 'test-' + phase, '--image', 'test-image', '--cpus', '2.5', '--threads', '2', '--user', '123:456'], env=env, capture_output=True, text=True)
    return result, capture


def test_phase_mounts_and_stable_config(tmp_path):
    config, output = setup(tmp_path)
    for phase in ('rank', 'score'):
        result, capture = run(tmp_path, config, output, phase)
        assert result.returncode == 0, result.stderr
    rank = json.loads((tmp_path / 'rank.json').read_text())
    score = json.loads((tmp_path / 'score.json').read_text())
    assert rank['config'] == score['config']
    for phase, record in [('rank', rank), ('score', score)]:
        argv = record['argv']
        mounts = [argv[i+1] for i, arg in enumerate(argv) if arg == '--mount']
        assert sum('dst=/inputs/' in m for m in mounts) == (5 if phase == 'rank' else 6)
        assert any('dst=/inputs/gold,readonly' in m for m in mounts) == (phase == 'score')
        assert not any(str(tmp_path / 'inputs' / 'gold') in m for m in mounts) == (phase == 'rank')
        assert argv[-4:] == ['/operator.py', phase, '--config', '/config.json']
        assert argv[argv.index('--user') + 1] == '123:456'
        assert argv[argv.index('--cpus') + 1] == '2.5'
        assert 'OPENBLAS_NUM_THREADS=2' in argv and 'test-image' in argv


def test_reject_gold_under_output_and_alias(tmp_path):
    config, output = setup(tmp_path)
    cfg = json.loads(config.read_text())
    cfg['inputs']['gold']['path'] = str(output / 'gold')
    config.write_text(json.dumps(cfg))
    result, capture = run(tmp_path, config, output, 'rank')
    assert result.returncode != 0 and not capture.exists()
    cfg['inputs']['gold']['path'] = cfg['inputs']['matrix']['path']
    config.write_text(json.dumps(cfg))
    result, capture = run(tmp_path, config, output, 'rank')
    assert result.returncode != 0 and not capture.exists()
