"""Adversarial cases found after the initial integration acceptance."""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import pytest
from test_contract import config
from test_processes import invoke, node, root_for, save, start, wait_for
from tools.experiments.config import ConfigError, load_config
from tools.experiments.runtime import prepare, task_status


def test_terminal_from_other_dispatch_cannot_satisfy_dependency(tmp_path):
    path, cfg = config(tmp_path)
    doc = load_config(path)
    prepare(doc)
    task_dir = Path(doc['run_dir']) / 'tasks/one'
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / 'DISPATCH.json').write_text(json.dumps({'dispatch_id': 'current', 'config_sha256': doc['sha256']}))
    (task_dir / 'EXIT.json').write_text(json.dumps({'dispatch_id': 'other', 'state': 'SUCCEEDED', 'returncode': 0}))
    assert task_status(doc, doc['config']['tasks'][0])['state'] == 'UNKNOWN'


def test_duplicate_output_producers_rejected(tmp_path):
    path, cfg = config(tmp_path)
    output = '${run_root}/experiments/fixture/outputs/shared.txt'
    cfg['tasks'][0]['outputs'] = [output]
    cfg['tasks'].append({**copy.deepcopy(cfg['tasks'][0]), 'id': 'two'})
    save(path, cfg)
    with pytest.raises(ConfigError): load_config(path)


def test_status_does_not_relabel_old_run_with_modified_config(tmp_path):
    path, cfg = config(tmp_path)
    assert invoke(path, 'prepare').returncode == 0
    cfg['experiment']['model'] = 'different-model'
    save(path, cfg)
    result = invoke(path, 'status')
    assert result.returncode != 0


def test_pinned_script_changed_while_predecessor_runs_is_not_executed(tmp_path):
    path, cfg = config(tmp_path)
    script = tmp_path / 'consumer.py'
    script.write_text('print("original")\n')
    cfg['tasks'] = [node(tmp_path, 'slow', seconds=0.8),
                    {'id': 'consumer', 'resource': 'cpu', 'depends_on': ['slow'],
                     'argv': [sys.executable, str(script)], 'cwd': str(tmp_path),
                     'artifacts': [{'path': str(script), 'sha256': hashlib.sha256(script.read_bytes()).hexdigest()}]}]
    save(path, cfg); start(path)
    root = root_for(path, cfg)
    wait_for(root / 'tasks/slow/CHILD.json')
    script.write_text('print("MUTATED")\n')
    end = wait_for(root / 'STATUS.json', lambda x: x['state'] != 'RUNNING')
    out = root / 'tasks/consumer/stdout.log'
    assert not out.exists() or 'MUTATED' not in out.read_text()
    assert end['state'] != 'SUCCEEDED'


def test_prepare_receipt_cannot_repin_locked_input(tmp_path):
    path, cfg = config(tmp_path)
    doc = load_config(path)
    prepare(doc)
    manifest = Path(cfg['experiment']['dataset']['manifest'])
    manifest.write_text('{"other": []}')
    root = Path(doc['run_dir'])
    receipt_path = root / 'PREPARE.json'
    receipt = json.loads(receipt_path.read_text())
    from tools.experiments.state import verified_file, stat_key
    receipt['artifacts'][0] = verified_file(manifest, None)
    receipt['manifest_sha256'] = receipt['artifacts'][0]['sha256']
    (root / 'inputs/dataset-manifest').write_bytes(manifest.read_bytes())
    receipt['manifest_copy_stat'] = stat_key(root / 'inputs/dataset-manifest')
    receipt_path.write_text(json.dumps(receipt))
    from tools.experiments.runtime import assert_frozen
    with pytest.raises(ConfigError): assert_frozen(doc)


def test_output_symlink_cannot_certify_external_file(tmp_path):
    path, cfg = config(tmp_path)
    outside = tmp_path / 'external'
    outside.write_text('not generated')
    output = '${run_root}/experiments/fixture/outputs/result'
    cfg['tasks'] = [{'id': 'one', 'resource': 'cpu', 'cwd': str(tmp_path),
                     'argv': [sys.executable, '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).symlink_to(sys.argv[2])', output, str(outside)],
                     'outputs': [output]}]
    save(path, cfg); start(path)
    end = wait_for(root_for(path, cfg) / 'tasks/one/EXIT.json')
    assert end['state'] != 'SUCCEEDED'


def test_bad_output_does_not_stop_independent_task(tmp_path):
    path, cfg = config(tmp_path)
    output = '${run_root}/experiments/fixture/outputs/preexisting'
    cfg['tasks'] = [dict(node(tmp_path, 'bad'), outputs=[output]), node(tmp_path, 'independent')]
    save(path, cfg)
    assert invoke(path, 'prepare').returncode == 0
    root = root_for(path, cfg)
    (root / 'outputs/preexisting').write_text('preserve me')
    assert invoke(path, 'run').returncode == 0
    good = wait_for(root / 'tasks/independent/EXIT.json', timeout=4)
    assert good['state'] == 'SUCCEEDED'
    assert (root / 'outputs/preexisting').read_text() == 'preserve me'


@pytest.mark.parametrize('field,value', [('image', 'sha256:' + 'b' * 64), ('model', 'wrong-model'), ('thinking', True)])
def test_downstream_launch_and_science_mismatch_rejected(tmp_path, field, value):
    path, cfg = config(tmp_path)
    downstream = {'image': cfg['release']['image_digest'], 'model': cfg['experiment']['model'], 'thinking': False}
    downstream[field] = value
    launch = tmp_path / 'launch.json'
    launch.write_text(json.dumps(downstream))
    cfg['execution']['bindings'] = [{'path': str(launch), 'sha256': hashlib.sha256(launch.read_bytes()).hexdigest(),
        'matches': {'/image': '/release/image_digest', '/model': '/experiment/model', '/thinking': '/experiment/thinking'}}]
    save(path, cfg)
    assert invoke(path, 'prepare').returncode != 0
    assert not (tmp_path / 'runs').exists()


def test_observed_worker_counts_against_restarted_controller_capacity(tmp_path):
    import subprocess
    from tools.experiments.state import identity
    path, cfg = config(tmp_path)
    cfg['execution']['parallelism']['cpu'] = 1
    cfg['tasks'] = [node(tmp_path, 'existing'), node(tmp_path, 'next', seconds=.1)]
    save(path, cfg)
    assert invoke(path, 'prepare').returncode == 0
    doc = load_config(path)
    root = Path(doc['run_dir'])
    td = root / 'tasks/existing'
    td.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1.2)'])
    try:
        dispatch = {'dispatch_id': 'prior-worker', 'config_sha256': doc['sha256'], 'at': time.time()}
        (td / 'DISPATCH.json').write_text(json.dumps(dispatch))
        (td / 'START.json').write_text(json.dumps({**dispatch, **identity(process.pid)}))
        assert invoke(path, 'run').returncode == 0
        time.sleep(.3)
        assert not (root / 'tasks/next/DISPATCH.json').exists()
        process.wait(timeout=3)
        end = wait_for(root / 'tasks/next/EXIT.json')
        assert end['state'] == 'SUCCEEDED'
        wait_for(root / 'STATUS.json', lambda x: x['state'] == 'UNKNOWN')
    finally:
        if process.poll() is None: process.kill()
        process.wait()
