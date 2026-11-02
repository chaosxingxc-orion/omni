import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

path = Path(__file__).parents[2] / 'scripts/environment/configured_job_launch.py'
spec = importlib.util.spec_from_file_location('configured_launch', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def configured(tmp_path):
    source = tmp_path / 'data'
    source.mkdir()
    receipt = tmp_path / 'receipts'
    pin = source / 'input.json'
    pin.write_text('{}')
    doc = {'schema': m.SCHEMA, 'container_name': 'owned-test', 'image': 'sha256:' + 'a' * 64,
           'repo_path': str(tmp_path / 'repo'), 'expected_commit': 'b' * 40,
           'receipt_dir': str(receipt), 'argv': ['/bin/python3', '/job.py', 'run'],
           'working_directory': '/tmp', 'network': 'none', 'pid_host': False, 'cpuset_cpus': '1-2',
           'environment': {'LITERAL': '$(do-not-execute)`literal`'},
           'mounts': [{'source': str(source), 'target': '/data', 'readonly': True}],
           'create_directories': [str(receipt)], 'allowed_create_roots': [str(tmp_path)],
           'required_inputs': [{'path': str(pin), 'sha256': m.digest(pin)}], 'monitor_interval_seconds': 600}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(doc))
    return path, doc


class Docker:
    def __init__(self, config, failure=None):
        self.config, self.failure, self.container = config, failure, None
        self.launches = 0

    def __call__(self, argv, **kwargs):
        if argv[0] == 'git':
            return SimpleNamespace(returncode=0, stdout=self.config['expected_commit'] if argv[-1] == 'HEAD' else '', stderr='')
        if argv[:3] == ['docker', 'container', 'inspect']:
            return SimpleNamespace(returncode=0 if self.container else 1,
                                   stdout=json.dumps([self.container]) if self.container else '', stderr='' if self.container else 'No such container')
        assert argv[:3] == ['docker', 'run', '-d']
        self.launches += 1
        assert (Path(self.config['receipt_dir']) / 'launch-intent.json').exists()
        if self.failure == 'crash':
            raise OSError('response lost')
        if self.failure == 'error':
            return SimpleNamespace(returncode=125, stdout='', stderr='OCI mount error')
        labels = {argv[i + 1].split('=', 1)[0]: argv[i + 1].split('=', 1)[1]
                  for i, a in enumerate(argv) if a == '--label'}
        self.container = {'Image': self.config['image'], 'Id': 'actual-container', 'Config': {'Labels': labels},
                          'State': {'Status': 'running', 'Running': True, 'ExitCode': 0, 'StartedAt': 'now', 'FinishedAt': ''}}
        return SimpleNamespace(returncode=0, stdout='actual-container', stderr='')


def test_launch_once_with_literal_argv_and_reconcile_container(tmp_path, monkeypatch):
    path, doc = configured(tmp_path)
    docker = Docker(doc)
    monkeypatch.setattr(m, 'run', docker)
    m.execute(path, 'layout')
    plan = m.execute(path, 'plan')
    assert 'LITERAL=$(do-not-execute)`literal`' in plan['command']
    assert m.execute(path, 'launch')['state'] == 'running'
    assert m.execute(path, 'launch')['reused_launch']
    assert docker.launches == 1
    docker.container['State'].update(Status='exited', Running=False, ExitCode=1)
    observed = m.execute(path, 'watch')
    assert observed['exit_code'] == 1
    assert observed['scientific_success'] != True


@pytest.mark.parametrize('failure', ['crash', 'error'])
def test_launch_uncertainty_never_repeats_request(tmp_path, monkeypatch, failure):
    path, doc = configured(tmp_path)
    docker = Docker(doc, failure)
    monkeypatch.setattr(m, 'run', docker)
    m.execute(path, 'layout')
    with pytest.raises((ValueError, OSError)):
        m.execute(path, 'launch')
    assert m.execute(path, 'launch')['state'] == 'unresolved-launch-intent'
    assert docker.launches == 1


def test_missing_nested_mountpoint_is_caught_before_docker_launch(tmp_path, monkeypatch):
    path, doc = configured(tmp_path)
    nested = tmp_path / 'nested'
    nested.mkdir()
    doc['mounts'].append({'source': str(nested), 'target': '/data/new/path', 'readonly': False})
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='nested mountpoint'):
        m.execute(path, 'layout')
    (tmp_path / 'data/new/path').mkdir(parents=True)
    assert m.execute(path, 'layout')['layout_ready']


@pytest.mark.parametrize('fault', ['pin', 'commit', 'dirty', 'config', 'foreign-container', 'escape', 'shell'])
def test_identity_drift_and_invalid_layout_refused(tmp_path, monkeypatch, fault):
    path, doc = configured(tmp_path)
    docker = Docker(doc)
    monkeypatch.setattr(m, 'run', docker)
    m.execute(path, 'layout')
    if fault in ('config', 'foreign-container'):
        m.execute(path, 'launch')
        if fault == 'config':
            doc['argv'].append('changed'); path.write_text(json.dumps(doc))
        else:
            docker.container['Config']['Labels'] = {}
    elif fault == 'pin':
        (tmp_path / 'data/input.json').write_text('changed')
    elif fault == 'commit':
        doc['expected_commit'] = 'c' * 40; path.write_text(json.dumps(doc)); docker.config = {**doc, 'expected_commit': 'b' * 40}
    elif fault == 'dirty':
        def dirty(argv, **kwargs):
            if argv[0] == 'git' and argv[-1] == '--porcelain':
                return SimpleNamespace(returncode=0, stdout=' M changed', stderr='')
            return docker(argv, **kwargs)
        monkeypatch.setattr(m, 'run', dirty)
    elif fault == 'escape':
        doc['create_directories'] = ['/etc/new']; path.write_text(json.dumps(doc))
    else:
        doc['argv'] = 'python /job.py'; path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        m.execute(path, 'launch')
