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


def lease_configured(tmp_path):
    path, doc = configured(tmp_path)
    locks = tmp_path / 'data/locks'; locks.mkdir()
    doc['environment']['SPEECHRL_DATA_DIR'] = '/data'
    doc['generation_lease'] = {'lock_directory': str(locks)}
    path.write_text(json.dumps(doc))
    return path, doc, locks


def test_fresh_outputs_report_all_conflicts_before_dispatch(tmp_path, monkeypatch):
    path, doc = configured(tmp_path)
    doc['mounts'][0]['readonly'] = False
    doc['fresh_outputs'] = ['/data/generation', '/data/timing.jsonl', '/data/runs/run/attempt']
    path.write_text(json.dumps(doc))
    (tmp_path/'data/generation').mkdir()
    (tmp_path/'data/timing.jsonl').write_text('historical timing')
    run_dir = tmp_path/'data/runs/run/attempt'; run_dir.mkdir(parents=True)
    (run_dir/'flight-receipts.json').write_text('preflight only')
    docker = Docker(doc); monkeypatch.setattr(m, 'run', docker)
    m.execute(path, 'layout')
    with pytest.raises(ValueError, match='fresh output conflicts') as error:
        m.execute(path, 'launch')
    assert all(p in str(error.value) for p in doc['fresh_outputs'])
    assert docker.launches == 0
    assert not (Path(doc['receipt_dir'])/'launch-intent.json').exists()
    assert (tmp_path/'data/timing.jsonl').read_text() == 'historical timing'
    assert (run_dir/'flight-receipts.json').read_text() == 'preflight only'


def test_fresh_outputs_nested_mount_and_existing_launch_are_preserved(tmp_path, monkeypatch):
    path, doc = configured(tmp_path)
    nested = tmp_path/'telemetry'; nested.mkdir()
    (tmp_path/'data/telemetry').mkdir()
    doc['mounts'].append({'source': str(nested), 'target': '/data/telemetry', 'readonly': False})
    doc['fresh_outputs'] = ['/data/telemetry/timing.jsonl']
    path.write_text(json.dumps(doc))
    docker = Docker(doc); monkeypatch.setattr(m, 'run', docker)
    m.execute(path, 'layout')
    assert m.execute(path, 'launch')['state'] == 'running'
    (nested/'timing.jsonl').write_text('now produced by current run')
    assert m.execute(path, 'launch')['reused_launch']
    assert docker.launches == 1
    assert m.fresh_output_conflicts(doc)[0]['host_path'] == str(nested/'timing.jsonl')


@pytest.mark.parametrize('fault', ['readonly', 'unmapped', 'mount-root', 'escape', 'dangling'])
def test_fresh_output_paths_reject_invalid_or_existing_targets(tmp_path, monkeypatch, fault):
    path, doc = configured(tmp_path)
    doc['mounts'][0]['readonly'] = False
    doc['fresh_outputs'] = ['/data/output']
    if fault == 'readonly': doc['mounts'][0]['readonly'] = True
    elif fault == 'unmapped': doc['fresh_outputs'] = ['/unmapped/output']
    elif fault == 'mount-root': doc['fresh_outputs'] = ['/data']
    elif fault == 'escape': (tmp_path/'data/output').symlink_to('/outside-owned-root')
    else: (tmp_path/'data/output').symlink_to(tmp_path/'data/missing')
    path.write_text(json.dumps(doc))
    docker = Docker(doc); monkeypatch.setattr(m, 'run', docker)
    with pytest.raises(ValueError):
        m.execute(path, 'layout')
        m.execute(path, 'launch')
    assert docker.launches == 0


def test_stale_metadata_blocks_before_intent_without_mutating_guard(tmp_path, monkeypatch):
    path, doc, locks = lease_configured(tmp_path)
    docker = Docker(doc); monkeypatch.setattr(m, 'run', docker)
    guard = locks/'generation.guard'; guard.touch(); inode = guard.stat().st_ino
    metadata = locks/'flight.lock'
    # The real failure shape: prior generation died and intentionally retained custody.
    metadata.write_text(json.dumps({'schema': 'saea-generation-lease-v1', 'pid': 999999999,
                                   'run_id': 'failed-v1', 'attempt_id': 'generation-v1'}))
    before = metadata.read_bytes()
    m.execute(path, 'layout')
    assert m.execute(path, 'plan')['generation_readiness']['state'] == 'unreconciled-metadata'
    with pytest.raises(ValueError, match='unreconciled-metadata'):
        m.execute(path, 'launch')
    assert docker.launches == 0
    assert not (Path(doc['receipt_dir'])/'launch-intent.json').exists()
    assert metadata.read_bytes() == before and guard.stat().st_ino == inode


def test_plan_while_predecessor_holds_lease_then_launch_after_release(tmp_path, monkeypatch):
    import fcntl
    path, doc, locks = lease_configured(tmp_path)
    docker = Docker(doc); monkeypatch.setattr(m, 'run', docker)
    m.execute(path, 'layout')
    with (locks/'generation.guard').open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert m.execute(path, 'plan')['generation_readiness']['state'] == 'lease-held'
        with pytest.raises(ValueError, match='lease-held'):m.execute(path, 'launch')
    # Existing guard alone must not block the successor.
    assert m.execute(path, 'launch')['state'] == 'running'
    (locks/'flight.lock').write_text('now owned by the running successor')
    assert m.execute(path, 'launch')['reused_launch']
    assert docker.launches == 1


@pytest.mark.parametrize('name', ['flight.lock', 'engine-transition.pending.json'])
def test_pending_or_dangling_metadata_is_never_treated_as_absent(tmp_path, monkeypatch, name):
    path, doc, locks = lease_configured(tmp_path)
    docker = Docker(doc); monkeypatch.setattr(m, 'run', docker)
    (locks/name).symlink_to(locks/'missing-target')
    m.execute(path, 'layout')
    with pytest.raises(ValueError, match='generation launch not ready'):m.execute(path, 'launch')
    assert docker.launches == 0 and (locks/name).is_symlink()


def test_lease_directory_must_match_actual_runtime_mapping(tmp_path):
    path, doc, locks = lease_configured(tmp_path)
    doc['generation_lease']['lock_directory'] = str(tmp_path/'unrelated')
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='runtime mount'):m.load_config(path)


def test_missing_guard_is_not_created_and_unreadable_root_blocks(tmp_path):
    path, doc, locks = lease_configured(tmp_path)
    assert m.generation_readiness(m.load_config(path))['state'] == 'clear'
    assert not (locks/'generation.guard').exists()
    locks.rmdir()
    assert m.generation_readiness(m.load_config(path))['state'] == 'unreadable'


def test_release_source_is_explicit_and_rejects_source_overlay(tmp_path):
    path, doc = configured(tmp_path)
    doc['repo_path']='/opt/saea-release/studies/speech-aware-evidence-acquisition';doc['working_directory']=doc['repo_path']
    doc['source_identity']={'mode':'image-release-v1','manifest_sha256':'c'*64}
    doc['environment'].update(SAEA_RELEASE_MANIFEST='/opt/saea-release/source-manifest.json', SAEA_RELEASE_MANIFEST_SHA256='c'*64, PYTHONPATH='/opt/saea-release/studies/speech-aware-evidence-acquisition/src', PYTHONDONTWRITEBYTECODE='1');doc['argv']=['/opt/saea-venv/bin/python','/opt/saea-release/studies/speech-aware-evidence-acquisition/scripts/analysis/intent_slot_workload.py']
    path.write_text(json.dumps(doc));assert m.load_config(path)['source_identity']['mode']=='image-release-v1'
    doc['mounts'].append({'source':str(tmp_path/'data'),'target':'/opt/saea-release/studies','readonly':True})
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='cannot be mounted'):m.load_config(path)


def mounted_release(tmp_path):
    path, doc = configured(tmp_path)
    release = tmp_path / 'release'
    release.mkdir()
    doc['repo_path'] = '/opt/saea-release/studies/speech-aware-evidence-acquisition'
    doc['working_directory'] = doc['repo_path']
    doc['source_identity'] = {'mode': 'mounted-release-v1', 'manifest_sha256': 'c' * 64}
    doc['environment'].update(SAEA_RELEASE_MANIFEST='/opt/saea-release/source-manifest.json',
        SAEA_RELEASE_MANIFEST_SHA256='c' * 64, PYTHONPATH=doc['repo_path'] + '/src',
        PYTHONDONTWRITEBYTECODE='1')
    doc['argv'] = ['/opt/saea-venv/bin/python', doc['repo_path'] + '/scripts/analysis/intent_slot_workload.py']
    doc['mounts'].append({'source': str(release), 'target': '/opt/saea-release', 'readonly': True})
    path.write_text(json.dumps(doc))
    return path, doc


def test_framework_runtime_uses_external_logs_and_same_bootstrap(tmp_path):
    path, doc = mounted_release(tmp_path)
    doc['mounts'][0]['readonly'] = False
    doc['framework_runtime'] = {'log_directory': '/data/framework/logs'}
    path.write_text(json.dumps(doc))
    loaded = m.load_config(path)
    live = m.command(loaded, 'config', 'runner')
    smoke = m.command(loaded, 'config', 'runner', runtime_check=True)
    assert m.FRAMEWORK_BOOTSTRAP in live and m.FRAMEWORK_BOOTSTRAP in smoke
    assert live[-1] == doc['argv'][1]
    assert '--runtime-check' in smoke and smoke[-1] == doc['argv'][1]
    assert smoke[smoke.index('--network') + 1] == 'none'
    assert '/data/framework/logs' in live
    assert not any('dst=/opt/saea-release/studies/' in a for a in live)


@pytest.mark.parametrize('target', ['/opt/saea-release/logs', '/tmp/logs', '/data'])
def test_framework_runtime_rejects_unmanaged_or_source_logs(tmp_path, target):
    path, doc = mounted_release(tmp_path)
    doc['framework_runtime'] = {'log_directory': target}
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        m.load_config(path)


def test_framework_smoke_cannot_dispatch_campaign_or_repeat(tmp_path, monkeypatch):
    path, doc = mounted_release(tmp_path)
    doc['mounts'][0]['readonly'] = False
    doc['framework_runtime'] = {'log_directory': '/data/framework/logs'}
    path.write_text(json.dumps(doc))
    m.execute(path, 'layout')
    monkeypatch.setattr(m, 'preflight', lambda doc: None)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='check-container', stderr='')
    monkeypatch.setattr(m, 'run', run)
    assert m.execute(path, 'runtime-check')['container_id'] == 'check-container'
    assert '--runtime-check' in calls[0]
    assert not (Path(doc['receipt_dir']) / 'launch-intent.json').exists()
    with pytest.raises(FileExistsError):
        m.execute(path, 'runtime-check')
    assert len(calls) == 1


@pytest.mark.parametrize('check_only', [False, True])
def test_framework_bootstrap_preserves_script_imports_and_arguments(tmp_path, check_only):
    import subprocess
    import sys
    # Framework stubs isolate bootstrap semantics, without importing an installed
    # framework or executing model code. Real framework import has its own smoke.
    modules = {
        'openjiuwen/core/common/logging/log_config.py':
            'class Config:\n def get_snapshot(self): return {}\n'
            ' def load_from_dict(self, cfg): assert cfg["log_path"]\nlog_config = Config()\n',
        'openjiuwen/core/common/logging/__init__.py':
            'class Logger:\n def info(self, msg): pass\nlogger = Logger()\n',
        'speech_aware_evidence_acquisition/reproduction/ojw/runner.py': '',
        'job_helpers.py': 'VALUE = "sibling imported"\n',
        'job.py': 'import sys\nfrom job_helpers import VALUE\n'
            'assert sys.argv[1:] == ["--literal", "$(untouched)"]\nprint(VALUE)\n',
    }
    for name, content in modules.items():
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
    result = subprocess.run([sys.executable, '-I', '-c', m.FRAMEWORK_BOOTSTRAP,
        str(tmp_path / 'logs'), '--runtime-check' if check_only else '--run-script',
        str(tmp_path / 'job.py'), '--literal', '$(untouched)'],
        cwd='/', capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert ('sibling imported' in result.stdout) is not check_only
    assert ('framework_runtime_ready' in result.stdout) is check_only


@pytest.mark.parametrize('fault', ['writable', 'nested', 'parent', 'dependency', 'missing'])
def test_mounted_release_rejects_unsealed_mounts(tmp_path, fault):
    path, doc = mounted_release(tmp_path)
    if fault == 'writable': doc['mounts'][-1]['readonly'] = False
    elif fault == 'nested': doc['mounts'].append({'source': str(tmp_path), 'target': doc['repo_path'], 'readonly': True})
    elif fault == 'parent': doc['mounts'][-1]['target'] = '/opt'
    elif fault == 'dependency': doc['mounts'].append({'source': str(tmp_path), 'target': '/usr/local', 'readonly': True})
    else: doc['mounts'].pop()
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='release'): m.load_config(path)


@pytest.mark.parametrize('valid', [False, True])
def test_mounted_release_checks_exact_tree_before_launch(tmp_path, monkeypatch, valid):
    path, doc = mounted_release(tmp_path)
    docker = Docker(doc)
    checks = []
    def run(argv, **kwargs):
        if argv[:3] == ['docker', 'run', '--rm']:
            checks.append(argv)
            assert 'type=bind,src=' + str(tmp_path / 'release') + ',dst=/opt/saea-release,readonly' in argv
            assert '-I' in argv
            assert 'docker/saea-base/release/verify-release.py' in argv[argv.index('-c') + 1]
            return SimpleNamespace(returncode=0 if valid else 1, stdout='', stderr='')
        return docker(argv, **kwargs)
    monkeypatch.setattr(m, 'run', run)
    m.execute(path, 'layout')
    if valid:
        assert m.execute(path, 'launch')['state'] == 'running'
        assert docker.launches == 1
    else:
        with pytest.raises(ValueError, match='manifest verification'): m.execute(path, 'launch')
        assert docker.launches == 0
        assert not (Path(doc['receipt_dir']) / 'launch-intent.json').exists()
    assert len(checks) == 1


def test_release_manifest_verification_fails_closed_before_launch(tmp_path,monkeypatch):
    path,doc=configured(tmp_path);doc['repo_path']='/opt/saea-release/studies/speech-aware-evidence-acquisition';doc['working_directory']=doc['repo_path']
    doc['source_identity']={'mode':'image-release-v1','manifest_sha256':'c'*64}
    doc['environment'].update(SAEA_RELEASE_MANIFEST='/opt/saea-release/source-manifest.json',SAEA_RELEASE_MANIFEST_SHA256='c'*64,PYTHONPATH='/opt/saea-release/studies/speech-aware-evidence-acquisition/src',PYTHONDONTWRITEBYTECODE='1');doc['argv']=['/opt/saea-venv/bin/python','/opt/saea-release/studies/speech-aware-evidence-acquisition/scripts/analysis/intent_slot_workload.py']
    path.write_text(json.dumps(doc))
    docker=Docker(doc)
    def run(args,**kwargs):
        if args[:3]==['docker','run','--rm']:return SimpleNamespace(returncode=1,stdout='',stderr='bad manifest')
        return docker(args,**kwargs)
    monkeypatch.setattr(m,'run',run)
    m.execute(path,'layout')
    with pytest.raises(ValueError,match='manifest verification'):m.execute(path,'launch')
    assert not (Path(doc['receipt_dir'])/'launch-intent.json').exists()

@pytest.mark.parametrize('fault',['pythonpath','entrypoint','bytecode'])
def test_release_refuses_unsealed_code_execution(tmp_path,fault):
    path,doc=configured(tmp_path);doc['repo_path']='/opt/saea-release/studies/speech-aware-evidence-acquisition';doc['working_directory']=doc['repo_path']
    doc['source_identity']={'mode':'image-release-v1','manifest_sha256':'c'*64}
    doc['environment'].update(SAEA_RELEASE_MANIFEST='/opt/saea-release/source-manifest.json',SAEA_RELEASE_MANIFEST_SHA256='c'*64,PYTHONPATH='/opt/saea-release/studies/speech-aware-evidence-acquisition/src',PYTHONDONTWRITEBYTECODE='1')
    doc['argv']=['/opt/saea-venv/bin/python','/opt/saea-release/studies/speech-aware-evidence-acquisition/scripts/analysis/intent_slot_workload.py']
    if fault=='pythonpath':doc['environment']['PYTHONPATH']='/other'
    if fault=='entrypoint':doc['argv'][1]='/data/other.py'
    if fault=='bytecode':doc['environment'].pop('PYTHONDONTWRITEBYTECODE')
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='release'):m.load_config(path)


def test_release_rejects_unknown_identity_key(tmp_path):
    path,doc=configured(tmp_path);doc['repo_path']='/opt/saea-release/studies/speech-aware-evidence-acquisition';doc['working_directory']=doc['repo_path'];doc['source_identity']={'mode':'image-release-v1','manifest_sha256':'c'*64,'extra':1};path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='identity'):m.load_config(path)


@pytest.mark.parametrize('fault',['PYTHONHOME','PYTHONSTARTUP','LD_PRELOAD','mount-dependency','workdir','argv-empty','mount-malformed'])
def test_release_closed_execution_environment(tmp_path,fault):
    path,doc=configured(tmp_path)
    doc['repo_path']='/opt/saea-release/studies/speech-aware-evidence-acquisition';doc['working_directory']=doc['repo_path']
    doc['source_identity']={'mode':'image-release-v1','manifest_sha256':'c'*64}
    doc['environment'].update(SAEA_RELEASE_MANIFEST='/opt/saea-release/source-manifest.json',SAEA_RELEASE_MANIFEST_SHA256='c'*64,PYTHONPATH=doc['repo_path']+'/src',PYTHONDONTWRITEBYTECODE='1')
    doc['argv']=['/opt/saea-venv/bin/python',doc['repo_path']+'/scripts/analysis/intent_slot_workload.py']
    if fault=='mount-dependency':doc['mounts'][0]['target']='/usr/local'
    elif fault=='mount-malformed':doc['mounts']=[{}]
    elif fault=='workdir':doc['working_directory']='/data'
    elif fault=='argv-empty':doc['argv']=[]
    else:doc['environment'][fault]='/data/inject'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):m.load_config(path)


def test_release_resident_process_access_is_explicit(tmp_path):
    path,doc=configured(tmp_path)
    doc['repo_path']='/opt/saea-release/studies/speech-aware-evidence-acquisition'; doc['working_directory']=doc['repo_path']
    doc['source_identity']={'mode':'image-release-v1','manifest_sha256':'c'*64}
    doc['environment'].update(SAEA_RELEASE_MANIFEST='/opt/saea-release/source-manifest.json',SAEA_RELEASE_MANIFEST_SHA256='c'*64,PYTHONPATH=doc['repo_path']+'/src',PYTHONDONTWRITEBYTECODE='1')
    doc['argv']=['/opt/saea-venv/bin/python',doc['repo_path']+'/scripts/environment/configured_environment_preflight.py']
    doc.update(process_access='resident-identity-read-v1',network='host',pid_host=True)
    path.write_text(json.dumps(doc)); loaded=m.load_config(path); cmd=m.command(loaded,'a'*64,'b'*64)
    assert cmd.count('--cap-add')==1 and 'SYS_PTRACE' in cmd and 'DAC_OVERRIDE' not in cmd
    doc['pid_host']=False;path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='resident process'):m.load_config(path)


def test_explicit_numeric_asset_owner(tmp_path):
    path,doc=configured(tmp_path);doc['user']='1008:1008';path.write_text(json.dumps(doc))
    loaded=m.load_config(path);cmd=m.command(loaded,'a'*64,'b'*64)
    assert cmd[cmd.index('--user')+1]=='1008:1008'
    doc['user']='root';path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='uid:gid'):m.load_config(path)
