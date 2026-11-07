"""Real contact/seal custody; Docker inspection is the only external double."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_frozen_core_adapter import _plan
from test_contact_journal import payload
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal, canonical, digest

spec = importlib.util.spec_from_file_location('configured_settlement',
    Path(__file__).parents[2] / 'scripts/environment/configured_terminal_settlement.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def pin(path):return {'path': str(path), 'sha256': m.launcher.digest(path)}


def test_docker_nanosecond_timestamp_is_accepted():
    assert m.stamp('2026-11-07T10:23:37.05639452Z') == m.stamp('2026-11-07T10:23:37.056394+00:00')


@pytest.fixture
def configured(tmp_path, monkeypatch):
    data, repo = tmp_path / 'data', tmp_path / 'repo'
    logical = Path('/logical-data'); logical_repo = Path('/logical-repo')
    candidate, run, attempt, session = 'candidate-v1', 'SAEA-SETTLE-V1', 'learning-v1', 'resident'
    expected = {'candidate': candidate, 'run_id': run, 'attempt_id': attempt,
        'study_commit': 'b' * 40, 'profile': 'profile-v1', 'service_session': session, 'rows': 1}
    output = data / 'work/output'; output.mkdir(parents=True)
    job = data / 'work/job'; job.mkdir(parents=True)
    complete = write(job / 'complete.json', {'schema': 'saea-learning-strategy-job-v1',
        'job': candidate, 'run_id': run, 'service_session': session, 'runtime_rows': 1})
    plan = _plan(run_id=run, attempt_id=attempt)
    source_plan = write(repo / 'docs/readiness' / candidate / 'generation-plan.json',
        {'execution_plan': asdict(plan), 'job_manifest': {'root': 'data', 'relpath': 'work/job/complete.json',
         'sha256': m.launcher.digest(complete)}})
    command = write(tmp_path / 'config/run.json', {'schema': 'saea-learning-strategy-command-v1',
        'mode': 'run', 'repo': str(logical_repo), 'data': str(logical), 'arguments': {
            'job': str(logical / 'work/job'), 'out': str(logical / 'work/output'),
            'after': str(logical / 'prior-release.json'), 'predecessor_run': 'SAEA-PRIOR'}})
    receipt = write(tmp_path / 'runtime.json', {'deployment': {'instances': [
        {'instance_id': 'tp2', 'host': '127.0.0.1', 'port': 8203}]}})
    write(data / 'serving' / session / 'session.json', {'session_id': session,
        'identity': {'runtime_receipt_sha256': m.launcher.digest(receipt)}, 'deployment_instance_ids': ['tp2']})
    launch_doc = {'schema': m.launcher.SCHEMA, 'container_name': 'configured-producer',
        'image': 'sha256:' + 'a' * 64, 'repo_path': str(repo), 'expected_commit': 'b' * 40,
        'receipt_dir': str(tmp_path / 'launch'), 'argv': ['/bin/python',
            '/logical-repo/scripts/analysis/learning_strategy_job.py', '--config', '/config/run.json'],
        'working_directory': str(logical_repo), 'network': 'host', 'pid_host': True, 'cpuset_cpus': '240-247',
        'environment': {'SAEA_OBSERVABILITY_RUN_ID': run, 'SAEA_OBSERVABILITY_ATTEMPT_ID': attempt,
                        'SAEA_DEPLOYMENT_PROFILE': expected['profile']},
        'mounts': [{'source': str(repo), 'target': str(logical_repo), 'readonly': True},
                   {'source': str(data), 'target': str(logical), 'readonly': False},
                   {'source': str(job), 'target': str(logical / 'work/job'), 'readonly': True},
                   {'source': str(command.parent), 'target': '/config', 'readonly': True}],
        'create_directories': [str(tmp_path / 'launch')], 'allowed_create_roots': [str(tmp_path)],
        'required_inputs': [pin(command)], 'monitor_interval_seconds': 600}
    launch = write(tmp_path / 'producer.json', launch_doc)
    helper = Path(m.launcher.__file__)
    config_sha = m.queue.canonical_hash(launch_doc)
    intent = write(tmp_path / 'launch/launch-intent.json', {'schema': m.launcher.SCHEMA,
        'version': m.launcher.VERSION, 'config_sha256': config_sha, 'runner_sha256': m.launcher.digest(helper),
        'expected_commit': 'b' * 40, 'automatic_retry': False})
    result = write(tmp_path / 'launch/launch-result.json', {'returncode': 0,
        'stdout': 'c' * 64, 'stderr': '', 'automatic_retry': False})
    container = {'Id': 'c' * 64, 'Name': '/configured-producer', 'Image': launch_doc['image'],
        'Config': {'Labels': {'saea.launch.config': config_sha, 'saea.launch.runner': m.launcher.digest(helper)},
                   'Env': [k+'='+v for k,v in launch_doc['environment'].items()]},
        'HostConfig': {'PidMode': 'host'},
        'State': {'Status': 'exited', 'Running': False, 'Pid': 0, 'ExitCode': 1,
            'StartedAt': '2026-11-03T10:00:00Z', 'FinishedAt': '2026-11-03T10:10:00Z'}}
    inspected = []
    def docker(args, **kwargs):
        assert args == ['docker', 'container', 'inspect', 'c' * 64]
        inspected.append(args)
        return SimpleNamespace(returncode=0, stdout=json.dumps([container]), stderr='')
    monkeypatch.setattr(m.launcher, 'run', docker)
    monkeypatch.setattr(m.transition, 'quiescence', lambda instances: {'quiet': True, 'ports': [i['port'] for i in instances]})
    locks = data / 'locks'; locks.mkdir(); (locks / 'generation.guard').touch()
    lease = write(locks / 'flight.lock', {'schema': 'saea-generation-lease-v1',
        'run_id': run, 'attempt_id': attempt, 'run_ids': [run], 'study_head': 'b'*40,
        'session_id': session, 'pid': 999999999, 'started_at': '2026-11-03T10:01:00Z'})
    attempt_path = write(data / 'attempts' / run / (attempt + '.json'), {'state': 'ABORTED',
        'run_id': run, 'attempt_id': attempt, 'carrier_lock_key': plan.carrier_lock_key,
        'actual': {'calls': 1, 'audio_seconds': 0.0}})
    journal = ContactJournal(data, plan, 'prepared-context-text')
    p = payload(); journal.intent(p); journal.response(p['request_id'], digest(canonical(p)), b'raw observed', 2.5)
    write(output / 'started.json', {'study_commit': 'b' * 40, 'job': candidate})
    handoff = Path(__file__).parents[4] / 'scripts/npu/generation_release_handoff.py'
    config = {'schema': m.SCHEMA, 'producer_launch': pin(launch), 'producer_launcher': pin(helper),
        'handoff_verifier': pin(handoff), 'generation_release': {'path': str(output / 'generation-released.json'),
            'logical_data': str(logical), 'expected': expected},
        'source_plan': pin(source_plan), 'runtime_receipt': pin(receipt),
        'arm': 'prepared-context-text', 'state_dir': str(tmp_path / 'state/settlement')}
    path = write(tmp_path / 'settlement.json', config)
    return SimpleNamespace(**locals())


def seal(f):
    document = {**f.expected, 'schema': 'saea-generation-released-v1', 'generation_sealed': True,
        'generation_lease_released': True, 'scoring_complete': False, 'consumer_requires_own_lease_and_readiness': True,
        'released_at_unix': 1, 'campaign_seconds_to_release': 1, 'artifacts': {}, 'seal_receipt': {}}
    for key, filename in {'generation_manifest': f.run+'.generation-manifest.json',
        'score_ledger_cells': 'score-ledger-cells.json', 'score_jobs': 'score-jobs.jsonl'}.items():
        actual = write(f.data / 'runs' / f.run / f.attempt / filename, {})
        target = str(f.logical / 'runs' / f.run / f.attempt / filename)
        document['artifacts'][key] = {'path': target, 'sha256': m.launcher.digest(actual)}
        document['seal_receipt'].update({key+'_path': target, key+'_sha256': m.launcher.digest(actual)})
    write(f.output / 'generation-released.json', document)
    f.lease.unlink(); f.container['State']['ExitCode'] = 0


def test_stale_lease_dynamic_container_and_pid_are_preserved(configured):
    f = configured; original = f.lease.read_bytes(); guard_inode = (f.locks/'generation.guard').stat().st_ino
    result = m.execute(f.path)
    assert result['settled'] and result['mode'] == 'stale-lease' and result['observed_contacts'] == 1
    assert result['producer_exit_code'] == 1 and result['new_model_calls'] == 0
    assert not f.lease.exists() and list(f.locks.glob('settled-flight-*'))[0].read_bytes() == original
    assert (f.locks/'generation.guard').stat().st_ino == guard_inode
    # Repeating completed settlement must not touch a legitimate successor lease.
    write(f.lease, {'run_id': 'successor'}); f.inspected.clear()
    assert m.execute(f.path) == result and not f.inspected and f.lease.exists()


def test_success_requires_real_sealed_release(configured):
    f = configured; seal(f)
    assert m.execute(f.path)['mode'] == 'sealed-release'
    assert not list(f.locks.glob('settled-flight-*'))


@pytest.mark.parametrize('fault', ['no-release', 'changed-artifact', 'wrong-seal', 'wrong-release-run'])
def test_no_lease_never_means_success(configured, fault):
    f = configured; seal(f)
    path = f.output / 'generation-released.json'
    if fault == 'no-release':path.unlink()
    elif fault == 'changed-artifact':write(f.data / 'runs' / f.run / f.attempt / 'score-ledger-cells.json', {'changed': True})
    else:
        doc = json.loads(path.read_bytes())
        if fault == 'wrong-seal':doc['seal_receipt']['score_jobs_sha256'] = '0'*64
        else:doc['run_id'] = 'foreign'
        write(path, doc)
    with pytest.raises((ValueError, FileNotFoundError)):m.execute(f.path)
    assert not (Path(f.config['state_dir'])/'complete.json').exists()


@pytest.mark.parametrize('fault', ['unknown', 'orphan', 'wrong-filename', 'response-tamper', 'call-count',
    'active-attempt', 'wrong-attempt', 'foreign-lease', 'live-pid', 'bad-time', 'changed-plan',
    'intent', 'result', 'missing-result', 'container-label', 'container-id', 'running',
    'runtime-environment', 'session', 'busy', 'guard-replaced'])
def test_uncertainty_never_releases_lease(configured, monkeypatch, fault):
    f = configured; original = f.lease.read_bytes()
    if fault == 'unknown':f.journal._path('req-1', 'response').unlink()
    elif fault == 'orphan':f.journal._path('req-1', 'intent').unlink()
    elif fault == 'wrong-filename':f.journal._path('req-1', 'intent').rename(f.journal.root / ('0'*64+'.intent.json'))
    elif fault == 'response-tamper':write(f.journal._path('req-1', 'response'), {})
    elif fault in ('call-count', 'active-attempt', 'wrong-attempt'):
        doc = json.loads(f.attempt_path.read_bytes())
        if fault == 'call-count':doc['actual']['calls'] = 2
        elif fault == 'active-attempt':doc['state'] = 'RUNNING'
        else:doc['run_id'] = 'foreign'
        write(f.attempt_path, doc)
    elif fault in ('foreign-lease', 'bad-time'):
        doc = json.loads(f.lease.read_bytes())
        if fault == 'foreign-lease':doc['run_id'] = 'foreign'
        else:doc['started_at'] = '2026-11-02T10:00:00Z'
        write(f.lease, doc); original = f.lease.read_bytes()
    elif fault == 'live-pid':monkeypatch.setattr(m, 'lease_custody', lambda *a: (_ for _ in ()).throw(ValueError('live PID')))
    elif fault == 'changed-plan':write(f.source_plan, {})
    elif fault == 'intent':write(f.intent, {})
    elif fault == 'result':write(f.result, {'returncode': 0, 'stdout': 'd'*64, 'stderr': '', 'automatic_retry': False})
    elif fault == 'missing-result':f.result.unlink()
    elif fault == 'container-label':f.container['Config']['Labels']['saea.launch.config'] = '0'*64
    elif fault == 'container-id':f.container['Id'] = 'd'*64
    elif fault == 'running':f.container['State']['Running'] = True
    elif fault == 'runtime-environment':f.container['Config']['Env'] = []
    elif fault == 'session':write(f.data/'serving/resident/session.json', {})
    elif fault == 'busy':monkeypatch.setattr(m.transition, 'quiescence', lambda *a: (_ for _ in ()).throw(ValueError('busy')))
    elif fault == 'guard-replaced':
        def replace(instances):
            path = f.locks/'generation.guard'; path.rename(f.locks/'old.guard'); path.touch(); return {}
        monkeypatch.setattr(m.transition, 'quiescence', replace)
    with pytest.raises((ValueError, AssertionError, KeyError)):m.execute(f.path)
    assert f.lease.read_bytes() == original
    assert not (Path(f.config['state_dir'])/'complete.json').exists()


@pytest.mark.parametrize('crash', ['before-rename', 'after-rename', 'after-complete'])
def test_interrupted_settlement_resumes_without_replay(configured, monkeypatch, crash):
    f = configured; atomic, rename = m.queue.atomic, m.os.rename
    def broken_atomic(path, doc):
        if path.name == 'complete.json' and crash == 'after-rename':raise OSError('crash')
        atomic(path, doc)
        if path.name == 'complete.json' and crash == 'after-complete':raise OSError('crash')
    def broken_rename(*args):raise OSError('crash')
    monkeypatch.setattr(m.queue, 'atomic', broken_atomic)
    if crash == 'before-rename':monkeypatch.setattr(m.os, 'rename', broken_rename)
    with pytest.raises(OSError):m.execute(f.path)
    monkeypatch.setattr(m.queue, 'atomic', atomic); monkeypatch.setattr(m.os, 'rename', rename)
    assert m.execute(f.path)['settled']
    assert len(list(f.locks.glob('settled-flight-*'))) == 1 and not f.lease.exists()


def test_live_pid_is_directly_rejected(configured):
    f = configured
    with pytest.raises(ValueError, match='source PID'):m.execute(f.path, pid_exists=lambda pid: True)
    assert f.lease.exists()


def test_failed_settlement_does_not_delete_unknown_contacts(configured):
    f = configured; f.journal._path('req-1', 'response').unlink()
    intent = f.journal._path('req-1', 'intent').read_bytes()
    with pytest.raises(ValueError, match='unknown'):m.execute(f.path)
    assert f.journal._path('req-1', 'intent').read_bytes() == intent


def test_docker_nanosecond_timestamps_are_supported(configured):
    f = configured
    f.container['State'].update(StartedAt='2026-11-03T10:00:00.531999650Z',
                                FinishedAt='2026-11-03T10:10:00.264380710Z')
    assert m.execute(f.path)['settled']


@pytest.mark.parametrize('fault', ['missing-guard', 'pid-namespace', 'job-manifest', 'protected-state', 'changed-complete'])
def test_additional_custody_boundaries(configured, fault):
    f = configured
    if fault == 'missing-guard':(f.locks/'generation.guard').unlink()
    elif fault == 'pid-namespace':f.container['HostConfig']['PidMode'] = ''
    elif fault == 'job-manifest':
        doc = json.loads(f.complete.read_bytes());doc['extra'] = True;write(f.complete, doc)
    elif fault == 'protected-state':
        f.config['state_dir'] = str(f.data/'contact-journals/settlement');write(f.path, f.config)
    else:
        m.execute(f.path); path = Path(f.config['state_dir'])/'complete.json'
        doc = json.loads(path.read_bytes());doc['identity']['config_sha256'] = '0'*64;write(path, doc)
    with pytest.raises(ValueError):m.execute(f.path)
    if fault != 'changed-complete':assert f.lease.exists()


def test_sealed_release_change_during_quiescence_is_blocked(configured, monkeypatch):
    f = configured; seal(f)
    def changed(instances):
        write(f.data/'runs'/f.run/f.attempt/'score-ledger-cells.json', {'changed': True})
        return {}
    monkeypatch.setattr(m.transition, 'quiescence', changed)
    with pytest.raises(ValueError):m.execute(f.path)
    assert not (Path(f.config['state_dir'])/'complete.json').exists()


def test_narrow_per_run_mounts_do_not_require_parent_data_mount(configured):
    f = configured
    d = f.launch_doc
    d['mounts'] = [mt for mt in d['mounts'] if mt['target'] != str(f.logical)]
    for relative in ['locks', 'attempts', 'serving', 'work/output',
                     'contact-journals/'+f.run, 'runs/'+f.run, 'prepared/'+f.run]:
        d['mounts'].append({'source': str(f.data/relative), 'target': str(f.logical/relative), 'readonly': False})
    write(f.launch, d); f.config['producer_launch'] = pin(f.launch); write(f.path, f.config)
    sha = m.queue.canonical_hash(d)
    intent = json.loads(f.intent.read_bytes()); intent['config_sha256'] = sha; write(f.intent, intent)
    f.container['Config']['Labels']['saea.launch.config'] = sha
    assert m.execute(f.path)['settled']


def as_gepa(f):
    import copy
    job = f.data / 'work/codex/gepa-job'
    job.mkdir(parents=True)
    document = {'schema':'saea-gepa-mutation-job-v1','job':f.candidate,'run_id':f.run,
        'service_session':f.session,'runtime_rows':1,'generation_calls':1,'fit_calls':1}
    complete = write(job/'complete.json',document)
    plan=json.loads(f.source_plan.read_bytes())
    plan['job_manifest']={'root':'data','relpath':'work/codex/gepa-job/complete.json','sha256':m.launcher.digest(complete)}
    write(f.source_plan,plan)
    launch=copy.deepcopy(f.launch_doc)
    launch['argv']=['/bin/python','/logical-repo/scripts/analysis/gepa_mutation_job.py','run',
        '--repo','/logical-repo','--data',str(f.logical),'--job',str(f.logical/'work/codex/gepa-job'),
        '--out',str(f.logical/'work/output'),'--after',str(f.logical/'prior-release.json'),
        '--predecessor-run','SAEA-PRIOR']
    launch['mounts'].append({'source':str(job),'target':str(f.logical/'work/codex/gepa-job'),'readonly':True})
    launch['required_inputs']=[pin(complete)]
    write(f.launch,launch)
    config_sha=m.queue.canonical_hash(launch)
    intent=json.loads(f.intent.read_bytes());intent['config_sha256']=config_sha;write(f.intent,intent)
    f.container['Config']['Labels']['saea.launch.config']=config_sha
    config={**f.config,'producer_launch':pin(f.launch),'source_plan':pin(f.source_plan)}
    write(f.path,config)
    return job,launch,config


def test_gepa_failed_known_contact_uses_same_strict_settlement(configured):
    f=configured;as_gepa(f)
    result=m.execute(f.path)
    assert result['settled'] and result['observed_contacts']==1 and result['new_model_calls']==0
    assert not f.lease.exists()


@pytest.mark.parametrize('fault',['unknown','job-schema','calls','data','entry','flag','writable','manifest'])
def test_gepa_custody_or_cli_drift_never_releases(configured,fault):
    f=configured;job,launch,config=as_gepa(f)
    if fault=='unknown':f.journal._path('req-1','response').unlink()
    elif fault in ('job-schema','calls'):
        doc=json.loads((job/'complete.json').read_bytes())
        doc['schema' if fault=='job-schema' else 'generation_calls']='other' if fault=='job-schema' else 2
        write(job/'complete.json',doc)
    elif fault=='manifest':
        plan=json.loads(f.source_plan.read_bytes());plan['job_manifest']['sha256']='0'*64
        write(f.source_plan,plan);config['source_plan']=pin(f.source_plan)
    else:
        if fault=='data':launch['argv'][6]='/wrong-data'
        elif fault=='entry':launch['argv'][1]='/other/gepa_mutation_job.py'
        elif fault=='flag':launch['argv'][3]='--extra'
        else:launch['mounts'][-1]['readonly']=False
        write(f.launch,launch);config['producer_launch']=pin(f.launch)
    write(f.path,config)
    with pytest.raises((ValueError,FileNotFoundError)):m.execute(f.path)
    assert f.lease.exists()
