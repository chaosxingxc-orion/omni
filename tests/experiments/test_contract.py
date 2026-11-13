"""Experiment contracts: no live model, daemon or remote host required."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.experiments.config import ConfigError, load_config, task_environment
from tools.experiments.state import Store, identity, observe, verified_file


def config(tmp_path, name='fixture'):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{"samples": []}')
    cfg = {
        'schema_version': 1,
        'storage': {'shared_root': str(tmp_path), 'run_root': '${shared_root}/runs'},
        'release': {'image_digest': 'sha256:' + 'a' * 64},
        'experiment': {'id': name, 'model': 'fixture-no-model', 'thinking': False,
                       'dataset': {'manifest': str(manifest), 'version': 'fixture'},
                       'science': {'purpose': 'process regression only'}},
        'execution': {'network': 'none', 'poll_seconds': 0.05,
                      'parallelism': {'cpu': 2, 'npu': 1},
                      'lanes': {'npu': [{'id': 'fixture', 'lock_path': '${run_root}/lanes/fixture.guard'}]}},
        'tasks': [{'id': 'one', 'resource': 'cpu', 'argv': ['/bin/true'], 'cwd': str(tmp_path)}],
    }
    path = tmp_path / (name + '.json')
    path.write_text(json.dumps(cfg))
    return path, cfg


def test_config_read_only_and_explicit_environment(tmp_path):
    path, cfg = config(tmp_path)
    doc = load_config(path)
    assert not (tmp_path / 'runs').exists()
    env = task_environment(doc, doc['config']['tasks'][0])
    assert env['TMPDIR'].startswith(str(tmp_path / 'runs'))
    assert env['PYTHONDONTWRITEBYTECODE'] == '1'
    assert 'SSH_AUTH_SOCK' not in env


@pytest.mark.parametrize('fault', ['unknown_field', 'unknown_var', 'cycle', 'unknown_dep', 'duplicate', 'escape', 'cache_escape'])
def test_invalid_config_rejected_before_write(tmp_path, fault):
    path, cfg = config(tmp_path)
    task = cfg['tasks'][0]
    if fault == 'unknown_field': cfg['execution']['paralellism'] = 2
    if fault == 'unknown_var': task['argv'] = ['${typo}/python']
    if fault == 'cycle': task['depends_on'] = ['one']
    if fault == 'unknown_dep': task['depends_on'] = ['missing']
    if fault == 'duplicate': cfg['tasks'].append(copy.deepcopy(task))
    if fault == 'escape': task['outputs'] = [str(tmp_path / 'outside')]
    if fault == 'cache_escape': task['env'] = {'TMPDIR': '/tmp'}
    path.write_text(json.dumps(cfg))
    with pytest.raises(ConfigError): load_config(path)
    assert not (tmp_path / 'runs').exists()


def test_pid_reuse_cross_host_and_missing_history_are_distinct():
    current = identity()
    assert observe(current)['state'] == 'alive'
    assert observe({**current, 'start_ticks': current['start_ticks'] + 1})['state'] == 'absent'
    assert observe({**current, 'host': 'other-host'})['state'] == 'unknown'
    assert observe({'pid': os.getpid(), 'at': 123})['state'] == 'unknown'


def test_stat_receipt_reuse_then_changed_input_rejected(tmp_path):
    pin = tmp_path / 'pin'
    pin.write_bytes(b'original')
    expected = hashlib.sha256(pin.read_bytes()).hexdigest()
    first = verified_file(pin, expected)
    assert not first['cache_reused']
    assert not verified_file(pin, expected, first)['cache_reused']
    pin.write_bytes(b'changed!')
    with pytest.raises(ConfigError): verified_file(pin, expected, first)


def test_store_never_follows_write_symlink_outside_root(tmp_path):
    root = tmp_path / 'runs'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (root / 'redirect').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ConfigError): Store(root).write(root / 'redirect' / 'state.json', {})
    assert list(outside.iterdir()) == []


def test_large_cache_requires_explicit_immutable_asset(tmp_path):
    pin = tmp_path / 'large'
    pin.write_bytes(b'x' * (1024 * 1024 + 1))
    expected = hashlib.sha256(pin.read_bytes()).hexdigest()
    first = verified_file(pin, expected, immutable=True)
    assert verified_file(pin, expected, first, immutable=True)['cache_reused']
    assert not verified_file(pin, expected, first)['cache_reused']
