"""Durable queue drives the real launcher against an explicit Docker double."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_configured_job_launch import configured

path = Path(__file__).parents[2] / 'scripts/environment/configured_job_queue.py'
spec = importlib.util.spec_from_file_location('configured_queue', path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class Docker:
    def __init__(self):self.containers = {}; self.launches = []; self.fail = None; self.waits = []
    def __call__(self, argv, **kwargs):
        if argv[0] == 'git':return SimpleNamespace(returncode=0, stdout='b' * 40 if argv[-1] == 'HEAD' else '', stderr='')
        if argv[:3] == ['docker','container','inspect']:
            c = self.containers.get(argv[-1])
            return SimpleNamespace(returncode=0 if c else 1, stdout=json.dumps([c]) if c else '', stderr='' if c else 'No such container')
        if argv[:2] == ['docker','wait']:
            self.waits.append((argv[-1],kwargs['timeout']))
            c = next(c for c in self.containers.values() if c['Id'] == argv[-1])
            c['State'].update(Status='exited',Running=False,ExitCode=0,FinishedAt='end')
            return SimpleNamespace(returncode=0,stdout='0',stderr='')
        assert argv[:3] == ['docker','run','-d']
        name = argv[argv.index('--name')+1]; self.launches.append(name)
        if self.fail == 'unknown':raise OSError('launch response lost')
        labels = dict(argv[i+1].split('=',1) for i,a in enumerate(argv) if a == '--label')
        self.containers[name] = {'Image':'sha256:'+'a'*64, 'Id':f'{len(self.launches):064x}',
            'Config':{'Labels':labels}, 'State':{'Status':'running','Running':True,'ExitCode':0,'StartedAt':'start','FinishedAt':''}}
        if self.fail == 'after-create':raise OSError('created but response lost')
        return SimpleNamespace(returncode=0,stdout=self.containers[name]['Id'],stderr='')


def pin(path):return {'path':str(path),'sha256':m.launcher.digest(path)}


def test_registered_real_engine_release_identity_without_observability(tmp_path):
    root = tmp_path/'admission'; root.mkdir()
    candidate = 'slurp-psd-rails-fold-a-slot-off-snapshot-v5'
    expected = {'candidate':candidate,'run_id':'SAEA-OFF-V5','attempt_id':'generation-v1',
                'study_commit':'b'*40,'profile':'profile','service_session':'session','rows':1080}
    prefix = f'docs/readiness/{candidate}/'
    rels = {'candidate':prefix+'candidate.json', 'generation':prefix+'generation-plan-registered-v1.json'}
    documents = {'candidate':{**expected,'cell':candidate},
                 'generation':{'schema':'saea-registered-generation-plan-v1','cell':candidate,
                     'plan_kind':'generation','expected_session_id':'session',
                     'execution_plan':{'run_id':'SAEA-OFF-V5','attempt_id':'generation-v1'}}}
    for name, rel in rels.items():
        file = root/rel; file.parent.mkdir(parents=True,exist_ok=True); file.write_text(json.dumps(documents[name]))
    manifest = root/'manifest.json'
    manifest.write_text(json.dumps({'schema':'saea-experiment-workspace-v1','source_commit':'b'*40,
        'source_manifest_sha256':'a'*64,'files':{rel:m.launcher.digest(root/rel) for rel in rels.values()}}))
    marker=root/'completion.json';marker.write_text(json.dumps({'schema':'saea-external-admission-completion-v1',
        'candidate':candidate,'workspace_manifest':{'sha256':m.launcher.digest(manifest)},'binding':rels}))
    output=tmp_path/'control';output.mkdir()
    doc={'repo_path':'/opt/saea-release/studies/speech-aware-evidence-acquisition','expected_commit':'b'*40,
         'argv':['/opt/saea-venv/bin/python','/opt/saea-release/studies/speech-aware-evidence-acquisition/scripts/analysis/run_real_engine_candidate.py',
                 '--repo','/opt/saea-release/studies/speech-aware-evidence-acquisition','--data','/data',
                 '--candidate',candidate,'--out','/data/work/codex/generation-v1','--runtime-config','/config/runtime.json'],
         'environment':{'SAEA_DEPLOYMENT_PROFILE':'profile','SPEECHRL_DATA_DIR':'/data',
             'SAEA_EXPERIMENT_WORKSPACE':'/experiment-workspace','SAEA_EXPERIMENT_WORKSPACE_MANIFEST':'/experiment-workspace/manifest.json',
             'SAEA_EXPERIMENT_WORKSPACE_SHA256':m.launcher.digest(manifest),'SAEA_RELEASE_MANIFEST_SHA256':'a'*64},
         'external_admission':{'marker':str(marker),'sha256':m.launcher.digest(marker),'workspace_manifest':str(manifest)},
         'mounts':[{'source':str(root),'target':'/experiment-workspace','readonly':True},
                   {'source':str(output),'target':'/data/work/codex','readonly':False}]}
    release={'path':str(output/'generation-v1/generation-released.json'),'logical_data':'/data','expected':expected}
    assert m.release_identity(doc,release)[0] == Path(release['path'])
    wrong=copy.deepcopy(release);wrong['expected']['run_id']='OTHER'
    with pytest.raises(ValueError,match='candidate or plan identity'):m.release_identity(doc,wrong)
    wrong=copy.deepcopy(release);wrong['path'] += '.other'
    with pytest.raises(ValueError,match='release path'):m.release_identity(doc,wrong)
    wrong=copy.deepcopy(doc);wrong['environment']['SAEA_OBSERVABILITY_RUN_ID']='SAEA-OFF-V5'
    with pytest.raises(ValueError,match='launch run or attempt'):m.release_identity(wrong,release)
    (root/rels['generation']).write_text('{}')
    with pytest.raises(ValueError,match='asset differs'):m.release_identity(doc,release)


@pytest.fixture
def queue(tmp_path,monkeypatch):
    jobs = []
    for index in range(2):
        root=tmp_path/f'job{index}';root.mkdir();launch,doc=configured(root)
        doc['container_name']=f'job{index}';launch.write_text(json.dumps(doc))
        jobs.append({'id':f'job{index}','launch':pin(launch),'preflight':None,'after':None})
    config={'schema':m.SCHEMA,'queue_id':'test','state_dir':str(tmp_path/'state/queue'),
        'monitor_interval_seconds':600,'jobs':jobs}
    path=tmp_path/'queue.json';path.write_text(json.dumps(config))
    docker=Docker();monkeypatch.setattr(m.launcher,'run',docker)
    return path,config,docker


def test_serial_watch_uses_terminal_event_and_never_repeats(queue):
    path,config,docker=queue
    result=m.execute(path,'watch')
    assert result['status']=='complete' and docker.launches==['job0','job1']
    assert docker.waits==[('0'*63+'1',600),('0'*63+'2',600)]
    assert all(j['result']['scientific_success'] is False for j in result['jobs'])
    assert m.execute(path,'watch')['status']=='complete' and len(docker.launches)==2


def test_graph_plan_preserves_dependencies_and_failure_policy(queue):
    path, config, docker = queue
    # Independent branches remain explicitly independent in the plan.
    jobs = config['jobs']
    jobs[0]['id'] = 'a'; jobs[0]['depends_on'] = []; jobs[0]['on_fail'] = 'continue'
    jobs[1]['id'] = 'b'; jobs[1]['depends_on'] = []; jobs[1]['on_fail'] = 'continue'
    config['jobs'] = [jobs[0], jobs[1]]
    path.write_text(json.dumps(config))
    for job in jobs:m.launcher.execute(job['launch']['path'], 'layout')
    planned = m.execute(path, 'plan')
    assert planned['dependency_policy'] == 'dependency-ready-container-exit-zero'
    assert all('depends_on' in j for j in planned['jobs'])


def test_plan_uses_pinned_predecessor_launcher(queue, monkeypatch):
    path, config, _ = queue
    historic = {'path':'/old/configured_job_launch.py','sha256':'a'*64}
    config['jobs'][0]['after'] = {'condition':'container-exit-zero',
        'launch':config['jobs'][1]['launch'],'container_id':'1'*64,'launcher_artifact':historic}
    observed=[]
    monkeypatch.setattr(m,'load',lambda path:config)
    monkeypatch.setattr(m,'pin_path',lambda pin:Path(pin['path']))
    monkeypatch.setattr(m,'identities',lambda path:{})
    monkeypatch.setattr(m,'observe',lambda pin, artifact=None:(observed.append(artifact) or ({},{'container_id':'1'*64})))
    monkeypatch.setattr(m.launcher,'execute',lambda path, mode:{})
    m.execute(path,'plan')
    assert observed == [historic]


def test_graph_cycle_and_partial_dependency_declarations_rejected(queue):
    path, config, docker = queue
    config['jobs'][0].update(id='a', depends_on=['b'], on_fail='continue')
    config['jobs'][1].update(id='b', depends_on=['a'], on_fail='continue')
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='cycle'):
        m.execute(path, 'plan')
    config['jobs'][0].update(depends_on=[])
    config['jobs'][1].pop('depends_on')
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='graph jobs'):
        m.execute(path, 'plan')


def test_graph_runs_dependency_before_earlier_child_and_resumes_once(queue):
    path, config, docker = queue
    config['jobs'][0]['depends_on'] = ['job1']
    config['jobs'][1]['depends_on'] = []
    path.write_text(json.dumps(config))
    assert m.execute(path, 'once')['status'] == 'running'
    assert docker.launches == ['job1']
    assert m.execute(path, 'watch')['status'] == 'complete'
    assert docker.launches == ['job1', 'job0']
    assert m.execute(path, 'watch')['status'] == 'complete'
    assert docker.launches == ['job1', 'job0']


def test_graph_failure_blocks_descendant_but_runs_independent(queue, tmp_path):
    path, config, docker = queue
    root = tmp_path / 'independent'; root.mkdir()
    launch, doc = configured(root); doc['container_name'] = 'independent'
    launch.write_text(json.dumps(doc))
    config['jobs'][0].update(depends_on=[], on_fail='continue')
    config['jobs'][1].update(depends_on=['job0'], on_fail='continue')
    config['jobs'].append({'id':'independent', 'launch':pin(launch), 'preflight':None,
                          'after':None, 'depends_on':[], 'on_fail':'continue'})
    path.write_text(json.dumps(config)); m.execute(path, 'once')
    docker.containers['job0']['State'].update(Status='exited', Running=False, ExitCode=1)
    result = m.execute(path, 'watch')
    assert result['status'] == 'failed' and result['blocked_jobs'] == ['job1']
    assert docker.launches == ['job0', 'independent']
    assert result['jobs'][0]['status'] == 'failed' and result['jobs'][1]['status'] == 'pending'


def test_graph_unknown_launch_stops_even_independent_branch(queue):
    path, config, docker = queue
    for job in config['jobs']:job.update(depends_on=[], on_fail='continue')
    path.write_text(json.dumps(config)); docker.fail = 'unknown'
    with pytest.raises(OSError):m.execute(path, 'once')
    docker.fail = None
    assert m.execute(path, 'watch')['status'] == 'unknown'
    assert docker.launches == ['job0']


def test_graph_terminal_cleanup_runs_after_failure_before_successor(queue, tmp_path):
    path, config, docker = queue
    config['jobs'][0].update(depends_on=[], on_fail='continue')
    config['jobs'][1].update(depends_on=[], after_terminal=['job0'])
    root = tmp_path / 'successor'; root.mkdir(); launch, doc = configured(root)
    doc['container_name'] = 'successor'; launch.write_text(json.dumps(doc))
    config['jobs'].append({'id':'successor', 'launch':pin(launch), 'preflight':None,
                          'after':None, 'depends_on':['job1']})
    path.write_text(json.dumps(config)); m.execute(path, 'once')
    docker.containers['job0']['State'].update(Status='exited', Running=False, ExitCode=1)
    result = m.execute(path, 'watch')
    assert result['status'] == 'failed'  # Preserve original failure despite successful cleanup.
    assert docker.launches == ['job0', 'job1', 'successor']
    assert [j['status'] for j in result['jobs']] == ['failed', 'complete', 'complete']


def test_graph_unknown_launch_never_triggers_terminal_cleanup(queue):
    path, config, docker = queue
    config['jobs'][0].update(depends_on=[], on_fail='continue')
    config['jobs'][1].update(depends_on=[], after_terminal=['job0'])
    path.write_text(json.dumps(config)); docker.fail = 'unknown'
    with pytest.raises(OSError):m.execute(path, 'once')
    docker.fail = None
    assert m.execute(path, 'watch')['status'] == 'unknown'
    assert docker.launches == ['job0']


def test_terminal_dependencies_cannot_create_hidden_cycle(queue):
    path, config, docker = queue
    config['jobs'][0].update(depends_on=['job1'])
    config['jobs'][1].update(depends_on=[], after_terminal=['job0'])
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='cycle'):m.execute(path, 'plan')


def test_graph_waiting_external_producer_does_not_block_independent_job(queue, tmp_path):
    path, config, docker = queue
    root = tmp_path / 'producer'; root.mkdir(); producer, doc = configured(root)
    doc['container_name'] = 'producer'; producer.write_text(json.dumps(doc))
    m.launcher.execute(producer, 'layout'); observed = m.launcher.execute(producer, 'launch')
    for job in config['jobs']:job.update(depends_on=[], on_fail='continue')
    config['jobs'][0]['after'] = {'condition':'container-exit-zero', 'launch':pin(producer),
                                 'container_id':observed['container_id']}
    path.write_text(json.dumps(config))
    assert m.execute(path, 'once')['status'] == 'running'
    assert docker.launches == ['producer', 'job1']
    assert m.execute(path, 'watch')['status'] == 'complete'
    assert docker.launches == ['producer', 'job1', 'job0']


def test_nonzero_blocks_successor_without_retry(queue):
    path,config,docker=queue;m.execute(path,'once')
    docker.containers['job0']['State'].update(Status='exited',Running=False,ExitCode=2)
    assert m.execute(path,'once')['status']=='failed'
    assert m.execute(path,'watch')['status']=='failed' and docker.launches==['job0']


@pytest.mark.parametrize('failure',['unknown','after-create'])
def test_crash_intent_reconciles_only_existing_exact_container(queue,failure):
    path,config,docker=queue;docker.fail=failure
    with pytest.raises(OSError):m.execute(path,'once')
    docker.fail=None
    observed=m.execute(path,'once')
    assert observed['status']==('unknown' if failure=='unknown' else 'running')
    assert docker.launches==['job0']


def test_queue_crash_before_launcher_boundary_holds_without_fabricating_outcome(queue,monkeypatch):
    path,config,docker=queue;original=m.launcher.execute
    def fail(path,mode):
        if mode=='launch':raise OSError('crash before launcher')
        return original(path,mode)
    monkeypatch.setattr(m.launcher,'execute',fail)
    with pytest.raises(OSError):m.execute(path,'once')
    monkeypatch.setattr(m.launcher,'execute',original)
    assert m.execute(path,'once')['status']=='unknown' and not docker.launches


@pytest.mark.parametrize('fault',['config','launch_pin','foreign_container','removed_container'])
def test_identity_drift_refuses_new_dispatch(queue,fault):
    path,config,docker=queue;m.execute(path,'once')
    if fault=='config':config['jobs'].reverse();path.write_text(json.dumps(config))
    elif fault=='launch_pin':Path(config['jobs'][0]['launch']['path']).write_text('{}')
    elif fault=='foreign_container':docker.containers['job0']['Id']='f'*64
    else:docker.containers.clear()
    with pytest.raises(ValueError):m.execute(path,'once')
    assert docker.launches==['job0']


def test_known_external_predecessor_must_exit_zero(queue,tmp_path):
    path,config,docker=queue
    root=tmp_path/'producer';root.mkdir();producer,doc=configured(root)
    doc['container_name']='producer';producer.write_text(json.dumps(doc))
    m.launcher.execute(producer,'layout');observed=m.launcher.execute(producer,'launch')
    config['jobs'][0]['after']={'condition':'container-exit-zero','launch':pin(producer),'container_id':observed['container_id']}
    path.write_text(json.dumps(config))
    assert m.execute(path,'once')['status']=='waiting' and docker.launches==['producer']
    docker.containers['producer']['State'].update(Status='exited',Running=False,ExitCode=0)
    assert m.execute(path,'once')['status']=='running' and docker.launches==['producer','job0']


def test_preflight_binding_is_verified_before_any_layout_or_dispatch(queue,tmp_path,monkeypatch):
    path,config,docker=queue;p=tmp_path/'binding.json';p.write_text('{}')
    config['jobs'][0]['preflight']=pin(p);path.write_text(json.dumps(config))
    calls=[]
    def reject(*args):calls.append(args);raise ValueError('invalid frozen preflight')
    monkeypatch.setattr(m.binding,'verify',reject)
    with pytest.raises(ValueError,match='preflight'):m.execute(path,'once')
    assert len(calls)==1 and not docker.launches
    assert m.execute(path,'once')['status']=='unknown' and len(calls)==1


def test_queue_singleton_never_launches_under_held_guard(queue):
    path,config,docker=queue;root=Path(config['state_dir']);root.mkdir(parents=True)
    with (root/'queue.guard').open('a+b') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):m.execute(path,'once')
    assert not docker.launches


def test_plan_does_not_create_queue_or_launch(queue):
    path,config,docker=queue
    for job in config['jobs']:m.launcher.execute(job['launch']['path'],'layout')
    result=m.execute(path,'plan')
    assert len(result['jobs'])==2 and result['generation_release_supported'] is True
    assert not Path(config['state_dir']).exists() and not docker.launches


def generation_fixture(queue, tmp_path, *, learning=False):
    import os
    path, config, docker = queue
    root=tmp_path/'producer';root.mkdir();producer,doc=configured(root)
    output=root/'output';output.mkdir();data=root/'state';data.mkdir()
    expected={'candidate':'candidate','run_id':'SAEA-PRODUCER','attempt_id':'generation-v1',
        'study_commit':'b'*40,'profile':'profile','service_session':'session','rows':312}
    doc.update(container_name='producer',argv=['/bin/python3','/job.py','--data','/logical','--out','/output'])
    doc['environment'].update(SAEA_OBSERVABILITY_RUN_ID=expected['run_id'],
        SAEA_OBSERVABILITY_ATTEMPT_ID=expected['attempt_id'],SAEA_DEPLOYMENT_PROFILE='profile',SPEECHRL_DATA_DIR='/logical')
    doc['mounts'] += [{'source':str(output),'target':'/output','readonly':False},
        {'source':str(data),'target':'/logical','readonly':False}]
    if learning:
        expected.update(attempt_id='learning-v1',rows=51)
        Path(doc['repo_path']).mkdir()
        doc['environment']['SAEA_OBSERVABILITY_ATTEMPT_ID']='learning-v1'
        commands=root/'config';commands.mkdir();job=root/'job';job.mkdir()
        run=data/'runs';run.mkdir()
        command={'schema':'saea-learning-strategy-command-v1','mode':'run','repo':'/study',
            'data':'/logical','arguments':{'job':'/control/job-v1','out':'/output',
                'after':'/previous/generation-released.json','predecessor_run':'PREVIOUS'}}
        command_path=commands/'generation.json';command_path.write_text(json.dumps(command))
        (job/'complete.json').write_text(json.dumps({'schema':'saea-learning-strategy-job-v1',
            'job':expected['candidate'],'run_id':expected['run_id'],
            'service_session':expected['service_session'],'runtime_rows':51}))
        doc['argv']=['/bin/python3','/study/scripts/analysis/learning_strategy_job.py','--config','/config/generation.json']
        doc['required_inputs'].append(pin(command_path))
        doc['mounts']=[r for r in doc['mounts'] if r['target']!='/logical']+[
            {'source':doc['repo_path'],'target':'/study','readonly':True},
            {'source':str(commands),'target':'/config','readonly':True},
            {'source':str(job),'target':'/control/job-v1','readonly':True},
            {'source':str(run),'target':'/logical/runs/SAEA-PRODUCER','readonly':False}]
    producer.write_text(json.dumps(doc));m.launcher.execute(producer,'layout')
    observed=m.launcher.execute(producer,'launch')
    helper=Path(os.environ.get('SAEA_TEST_HANDOFF_HELPER') or str(Path(__file__).resolve().parents[4]/'scripts/npu/generation_release_handoff.py'))
    assert helper.is_file(), 'formal test must mount the existing shared handoff helper'
    config['handoff_verifier']=pin(helper)
    config['jobs'][0]['after']={'condition':'generation-released','launch':pin(producer),
        'container_id':observed['container_id'],'release':{'path':str(output/'generation-released.json'),
        'logical_data':'/logical','expected':expected}}
    seal={};artifacts={}
    for key,name in [('generation_manifest','SAEA-PRODUCER.generation-manifest.json'),
            ('score_ledger_cells','score-ledger-cells.json'),('score_jobs','score-jobs.jsonl')]:
        p=(data/'runs/learning-v1' if learning else data/'runs/SAEA-PRODUCER/generation-v1')/name
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        logical=str(Path('/logical/runs/SAEA-PRODUCER')/expected['attempt_id']/name)
        artifacts[key]={'path':logical,'sha256':m.launcher.digest(p)}
        seal[key+'_path']=logical;seal[key+'_sha256']=m.launcher.digest(p)
    release={'schema':'saea-generation-released-v1',**expected,'generation_sealed':True,
        'generation_lease_released':True,'scoring_complete':False,'consumer_requires_own_lease_and_readiness':True,
        'released_at_unix':123.,'campaign_seconds_to_release':12.,'artifacts':artifacts,'seal_receipt':seal}
    (output/'started.json').write_text(json.dumps({'study_commit':expected['study_commit'],'candidate':'candidate','rows':312}))
    if learning:
        (output/'started.json').write_text(json.dumps({'study_commit':expected['study_commit'],'job':'candidate'}))
    path.write_text(json.dumps(config))
    return output/'generation-released.json',release,data


def test_learning_config_single_run_mount_releases_original_bytes_while_scoring(queue,tmp_path,monkeypatch):
    release,value,data=generation_fixture(queue,tmp_path,learning=True);path,config,docker=queue
    blob=json.dumps(value,indent=3).encode();release.write_bytes(blob)
    real=m.release_helper;seen=[]
    def helper(doc):
        module=real(doc);original=module.verify_release
        def verify(receipt,**kwargs):
            assert receipt==release and receipt.read_bytes()==blob
            assert kwargs['data']==Path('/logical')
            seen.append(kwargs['resolve_artifact'](Path(value['artifacts']['score_jobs']['path'])))
            return original(receipt,**kwargs)
        module.verify_release=verify;return module
    monkeypatch.setattr(m,'release_helper',helper)
    result=m.execute(path,'once')
    assert result['status']=='running' and docker.launches==['producer','job0']
    assert docker.containers['producer']['State']['Running'] and release.read_bytes()==blob
    assert seen==[data/'runs/learning-v1/score-jobs.jsonl']
    assert result['jobs'][0]['generation_release']['producer_command']['job_manifest_sha256']


@pytest.mark.parametrize('fault',['missing_pin','config_changed','writable_config','wrong_mode','extra_field',
    'repo','data','out','writable_job','job_run','job_rows','job_session','started_job','artifact','symlink'])
def test_learning_producer_drift_blocks_dispatch(queue,tmp_path,fault):
    release,value,data=generation_fixture(queue,tmp_path,learning=True);path,config,docker=queue
    producer_path=Path(config['jobs'][0]['after']['launch']['path']);producer=json.loads(producer_path.read_bytes())
    command_path=tmp_path/'producer/config/generation.json';command=json.loads(command_path.read_bytes())
    job_path=tmp_path/'producer/job/complete.json';job=json.loads(job_path.read_bytes())
    if fault=='missing_pin':producer['required_inputs']=[]
    elif fault=='config_changed':command_path.write_text('{}')
    elif fault=='writable_config':next(r for r in producer['mounts'] if r['target']=='/config')['readonly']=False
    elif fault in ('wrong_mode','extra_field','repo','data','out'):
        if fault=='wrong_mode':command['mode']='prepare'
        elif fault=='extra_field':command['arguments']['extra']=True
        elif fault=='out':command['arguments']['out']='/output/foreign'
        else:command[fault]='/foreign'
        command_path.write_text(json.dumps(command));producer['required_inputs']=[pin(command_path)]
    elif fault=='writable_job':next(r for r in producer['mounts'] if r['target']=='/control/job-v1')['readonly']=False
    elif fault.startswith('job_'):
        job[{'job_run':'run_id','job_rows':'runtime_rows','job_session':'service_session'}[fault]]='foreign'
        job_path.write_text(json.dumps(job))
    elif fault=='started_job':release.with_name('started.json').write_text(json.dumps({'study_commit':'b'*40}))
    elif fault=='artifact':(data/'runs/learning-v1/score-jobs.jsonl').write_text('changed')
    else:
        target=data/'runs/learning-v1/score-jobs.jsonl';target.unlink();outside=tmp_path/'outside';outside.write_text('{}');target.symlink_to(outside)
    producer_path.write_text(json.dumps(producer));config['jobs'][0]['after']['launch']=pin(producer_path)
    docker.containers['producer']['Config']['Labels']['saea.launch.config']=m.canonical_hash(producer)
    path.write_text(json.dumps(config));release.write_text(json.dumps(value))
    with pytest.raises(ValueError):m.execute(path,'once')
    assert docker.launches==['producer']


def test_verified_generation_release_launches_while_producer_scores(queue,tmp_path):
    release,value,data=generation_fixture(queue,tmp_path);path,config,docker=queue
    assert m.execute(path,'once')['status']=='waiting' and docker.launches==['producer']
    release.write_text(json.dumps(value))
    result=m.execute(path,'once')
    assert result['status']=='running' and docker.launches==['producer','job0']
    assert docker.containers['producer']['State']['Running'] is True
    assert result['jobs'][0]['generation_release']['cpu_scoring_may_continue'] is True
    assert not docker.waits


@pytest.mark.parametrize('fault',['run','commit','lease_flag','score_flag','artifact','logical_path','started','container','event_change'])
def test_generation_release_drift_cannot_authorize_dispatch(queue,tmp_path,fault):
    release,value,data=generation_fixture(queue,tmp_path);path,config,docker=queue
    if fault=='run':value['run_id']='FOREIGN'
    elif fault=='commit':value['study_commit']='c'*40
    elif fault=='lease_flag':value['generation_lease_released']=False
    elif fault=='score_flag':value['scoring_complete']=True
    elif fault=='artifact':(data/'runs/SAEA-PRODUCER/generation-v1/score-jobs.jsonl').write_text('changed')
    elif fault=='logical_path':value['artifacts']['score_jobs']['path']='/other'
    elif fault=='started':release.with_name('started.json').write_text('{}')
    elif fault=='container':docker.containers['producer']['Id']='f'*64
    release.write_text(json.dumps(value))
    if fault=='event_change':
        m.execute(path,'once');value['released_at_unix']+=1;release.write_text(json.dumps(value))
    with pytest.raises(ValueError):m.execute(path,'once')
    assert docker.launches==(['producer','job0'] if fault=='event_change' else ['producer'])


@pytest.mark.parametrize('code',[0,1])
def test_terminal_without_sealed_release_holds(queue,tmp_path,code):
    generation_fixture(queue,tmp_path);path,config,docker=queue
    docker.containers['producer']['State'].update(Status='exited',Running=False,ExitCode=code)
    assert m.execute(path,'once')['status']==('unknown' if code==0 else 'failed')
    assert docker.launches==['producer']


def test_release_wait_uses_file_event_not_scoring_container_wait(queue,tmp_path,monkeypatch):
    release,value,data=generation_fixture(queue,tmp_path);path,config,docker=queue
    real=m.release_helper;waits=[]
    class Events:
        def __init__(self,directory):assert directory==release.parent
        def wait(self,seconds):
            waits.append(seconds);release.write_text(json.dumps(value))
        def close(self):pass
    monkeypatch.setattr(m,'release_helper',lambda doc:SimpleNamespace(verify_release=real(doc).verify_release,release_paths=real(doc).release_paths,DirectoryEvents=Events))
    result=m.execute(path,'watch')
    assert result['status']=='complete' and waits==[600]
    assert all(cid!=docker.containers['producer']['Id'] for cid,_ in docker.waits)


def internal_generation_fixture(queue, tmp_path, *, learning=True):
    release, value, data = generation_fixture(queue, tmp_path, learning=learning)
    path, config, docker = queue
    dependency = config['jobs'][0]['after']
    producer = {'id':'producer', 'launch':dependency['launch'], 'preflight':None, 'after':None,
        'depends_on':[], 'on_fail':'continue', 'generation_release':dependency['release']}
    cleanup, successor = config['jobs']
    cleanup.update(after=None, depends_on=[], after_terminal=['producer'],
        unless_generation_released='producer')
    successor.update(depends_on=[], after_handoff={'producer':'producer','settlement':cleanup['id']})
    config['jobs'] = [producer, cleanup, successor]
    path.write_text(json.dumps(config))
    return release, value, data


def test_internal_release_dispatches_successor_while_cpu_scoring(queue,tmp_path):
    release,value,_=internal_generation_fixture(queue,tmp_path);path,config,docker=queue
    assert m.execute(path,'once')['status']=='running'
    release.write_text(json.dumps(value))
    result=m.execute(path,'once')
    assert [j['status'] for j in result['jobs']]==['scoring','skipped','running']
    assert docker.launches==['producer','job1'] and docker.containers['producer']['State']['Running']
    assert not docker.waits
    assert m.execute(path,'watch')['status']=='complete'
    assert docker.launches==['producer','job1']


@pytest.mark.parametrize('corrupt',[False,True])
def test_finalized_artifact_releases_lane_without_task_scoring_files(queue,tmp_path,corrupt):
    from test_artifact_generation_release import artifact_fixture
    release, value, data = internal_generation_fixture(queue,tmp_path,learning=False)
    path,config,docker=queue
    expected = config['jobs'][0]['generation_release']['expected']
    document,mapping=artifact_fixture(data,expected,logical=Path('/logical'))
    assert m.execute(path,'once')['status']=='running'
    release.write_text(json.dumps(document))
    if corrupt:
        mapping[Path(document['artifacts']['attempt']['path'])].write_text('{}')
        with pytest.raises(ValueError,match='sealed input changed'):m.execute(path,'once')
        assert docker.launches==['producer']
    else:
        result=m.execute(path,'once')
        assert [j['status'] for j in result['jobs']]==['scoring','skipped','running']
        assert docker.launches==['producer','job1']


def test_internal_failed_generation_uses_cleanup_before_independent_successor(queue,tmp_path):
    internal_generation_fixture(queue,tmp_path);path,config,docker=queue
    m.execute(path,'once')
    docker.containers['producer']['State'].update(Status='exited',Running=False,ExitCode=1)
    result=m.execute(path,'watch')
    assert result['status']=='failed'
    assert docker.launches==['producer','job0','job1']
    assert [j['status'] for j in result['jobs']]==['failed','complete','complete']


def test_internal_cpu_failure_after_release_does_not_block_valid_generation(queue,tmp_path):
    release,value,_=internal_generation_fixture(queue,tmp_path);path,config,docker=queue
    m.execute(path,'once');release.write_text(json.dumps(value))
    docker.containers['producer']['State'].update(Status='exited',Running=False,ExitCode=1)
    result=m.execute(path,'watch')
    assert result['status']=='failed' and docker.launches==['producer','job1']
    assert result['jobs'][1]['status']=='skipped'


@pytest.mark.parametrize('fault',['missing','changed','container','artifact'])
def test_internal_release_custody_changes_hold_dispatch(queue,tmp_path,fault):
    release,value,data=internal_generation_fixture(queue,tmp_path);path,config,docker=queue
    m.execute(path,'once');release.write_text(json.dumps(value));m.execute(path,'once')
    if fault=='missing':release.unlink()
    elif fault=='changed':value['released_at_unix']+=1;release.write_text(json.dumps(value))
    elif fault=='container':docker.containers['producer']['Id']='e'*64
    else:(data/'runs/learning-v1/score-jobs.jsonl').write_text('changed')
    with pytest.raises(ValueError):m.execute(path,'once')
    assert docker.launches==['producer','job1']


@pytest.mark.parametrize('fault',['cycle','foreign-cleanup','unknown','no-release','mixed-edge'])
def test_internal_handoff_rejects_ambiguous_graph(queue,tmp_path,fault):
    internal_generation_fixture(queue,tmp_path);path,config,_=queue
    producer,cleanup,successor=config['jobs']
    if fault=='cycle':producer['depends_on']=['job1']
    elif fault=='foreign-cleanup':cleanup['unless_generation_released']='job1'
    elif fault=='unknown':successor['after_handoff']['producer']='missing'
    elif fault=='no-release':producer.pop('generation_release')
    else:successor['depends_on']=['producer']
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):m.load(path)


def test_internal_success_without_release_cannot_skip_cleanup(queue,tmp_path):
    internal_generation_fixture(queue,tmp_path);path,config,docker=queue
    m.execute(path,'once')
    docker.containers['producer']['State'].update(Status='exited',Running=False,ExitCode=0)
    with pytest.raises(ValueError,match='without generation release'):m.execute(path,'once')
    assert docker.launches==['producer']


def test_failed_independent_branch_retains_results_and_nonzero_cli(queue,tmp_path,capsys):
    internal_generation_fixture(queue,tmp_path);path,config,docker=queue
    m.execute(path,'once')
    docker.containers['producer']['State'].update(Status='exited',Running=False,ExitCode=1)
    assert m.main(['watch','--config',str(path)])==1
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='failed' and docker.launches==['producer','job0','job1']
    assert [j['status'] for j in result['jobs']]==['failed','complete','complete']


def test_successful_queue_cli_and_live_once_keep_zero_exit(queue,capsys):
    path,_,docker=queue
    assert m.main(['once','--config',str(path)])==0
    assert json.loads(capsys.readouterr().out)['status']=='running'
    assert m.main(['watch','--config',str(path)])==0
    assert json.loads(capsys.readouterr().out)['status']=='complete'
    assert docker.launches==['job0','job1']


def test_queued_generation_checks_lease_after_dependency_before_dispatch(queue):
    path, config, docker = queue
    first, second = config['jobs']
    launch_path = Path(second['launch']['path']); doc = json.loads(launch_path.read_text())
    locks = Path(doc['mounts'][0]['source'])/'locks'; locks.mkdir()
    doc['environment']['SPEECHRL_DATA_DIR'] = '/data'
    doc['generation_lease'] = {'lock_directory':str(locks)}
    launch_path.write_text(json.dumps(doc)); second['launch'] = pin(launch_path)
    path.write_text(json.dumps(config))
    for job in config['jobs']:m.launcher.execute(job['launch']['path'], 'layout')
    metadata = locks/'flight.lock'; metadata.write_text('{"run_id":"predecessor"}')
    # Planning future jobs is allowed while the predecessor has metadata.
    assert m.execute(path,'plan')['jobs'][1]['launch']['generation_readiness']['state'] == 'unreconciled-metadata'
    assert m.execute(path,'once')['status'] == 'running'
    assert docker.launches == ['job0']
    docker.containers['job0']['State'].update(Status='exited',Running=False,ExitCode=0)
    assert m.execute(path,'once')['status'] == 'ready'
    held = m.execute(path,'once')
    assert held['status'] == 'held'
    assert held['generation_readiness']['state'] == 'unreconciled-metadata'
    state = json.loads((Path(config['state_dir'])/'state.json').read_text())
    assert not state['jobs'][1].get('launch_requested')
    assert not (Path(doc['receipt_dir'])/'launch-intent.json').exists()
    assert docker.launches == ['job0'] and metadata.exists()
    # Model audited predecessor cleanup; queue itself never clears metadata.
    metadata.unlink()
    assert m.execute(path,'once')['status'] == 'running'
    assert docker.launches == ['job0','job1']


def historical_predecessor(queue, tmp_path):
    path, config, docker = queue
    root = tmp_path / 'old'; root.mkdir()
    launch, doc = configured(root); doc['container_name'] = 'previous-version'
    launch.write_text(json.dumps(doc)); m.launcher.execute(launch, 'layout')
    old = root / 'configured_job_launch.py'
    # A historical version is an immutable identity artifact, never executed.
    old.write_text(Path(m.launcher.__file__).read_text() + '\nraise RuntimeError("must not execute historical code")\n')
    runner_sha = m.launcher.digest(old); config_sha = m.canonical_hash(doc)
    m.launcher.persist(Path(doc['receipt_dir']) / 'launch-intent.json',
        {'config_sha256': config_sha, 'runner_sha256': runner_sha})
    docker(m.launcher.command(doc, config_sha, runner_sha))
    container = docker.containers[doc['container_name']]
    config['jobs'][0]['after'] = {'condition': 'container-exit-zero',
        'launch': pin(launch), 'container_id': container['Id'], 'launcher_artifact': pin(old)}
    path.write_text(json.dumps(config))
    return launch, old, container


def test_cross_version_requires_independent_artifact_pin(queue, tmp_path):
    path, config, docker = queue
    launch, old, container = historical_predecessor(queue, tmp_path)
    with pytest.raises(ValueError, match='receipt identity changed'):
        m.observe(pin(launch))
    assert m.execute(path, 'once')['status'] == 'waiting'
    assert docker.launches == ['previous-version']
    container['State'].update(Status='exited', Running=False, ExitCode=0, FinishedAt='end')
    assert m.execute(path, 'watch')['status'] == 'complete'
    assert docker.launches == ['previous-version', 'job0', 'job1']
    assert docker.containers['job0']['Config']['Labels']['saea.launch.runner'] == m.launcher.digest(m.launcher.__file__)
    assert m.execute(path, 'watch')['status'] == 'complete'
    assert len(docker.launches) == 3


@pytest.mark.parametrize('fault', ['changed-artifact', 'wrong-version', 'receipt', 'container-label', 'container-id', 'symlink', 'foreign-field'])
def test_cross_version_identity_faults_never_dispatch(queue, tmp_path, fault):
    path, config, docker = queue
    launch, old, container = historical_predecessor(queue, tmp_path)
    dep = config['jobs'][0]['after']
    if fault == 'changed-artifact':old.write_text('changed')
    elif fault == 'wrong-version':dep['launcher_artifact'] = pin(Path(m.launcher.__file__))
    elif fault == 'receipt':
        receipt = Path(json.loads(launch.read_text())['receipt_dir']) / 'launch-intent.json'
        value = json.loads(receipt.read_text()); value['runner_sha256'] = 'f'*64; receipt.write_text(json.dumps(value))
    elif fault == 'container-label':container['Config']['Labels']['saea.launch.runner'] = 'f'*64
    elif fault == 'container-id':dep['container_id'] = 'f'*64
    elif fault == 'symlink':
        link = tmp_path / 'link'; link.mkdir(); target = link / 'configured_job_launch.py'; target.symlink_to(old)
        dep['launcher_artifact'] = pin(target)
    else:dep['runner_sha256'] = m.launcher.digest(old)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):m.execute(path, 'once')
    assert docker.launches == ['previous-version']


def test_cross_version_missing_container_retains_unknown_without_retry(queue, tmp_path):
    path, config, docker = queue
    launch, old, container = historical_predecessor(queue, tmp_path)
    docker.containers.clear()
    _, observed = m.observe(pin(launch), pin(old))
    assert observed['state'] == 'unresolved-launch-intent' and observed['automatic_retry'] is False
    with pytest.raises(ValueError, match='predecessor container identity differs'):
        m.execute(path, 'once')
    assert docker.launches == ['previous-version']


def test_cross_version_generation_release_preserves_seal_checks(queue, tmp_path):
    release, value, _ = generation_fixture(queue, tmp_path)
    path, config, docker = queue
    dep = config['jobs'][0]['after']
    launch = m.launcher.load_config(dep['launch']['path'])
    old = tmp_path/'configured_job_launch.py'; old.write_text('# immutable historical launcher version\n')
    dep['launcher_artifact'] = pin(old); runner_sha = m.launcher.digest(old)
    receipt = Path(launch['receipt_dir'])/'launch-intent.json'
    intent = json.loads(receipt.read_text()); intent['runner_sha256'] = runner_sha; receipt.write_text(json.dumps(intent))
    docker.containers['producer']['Config']['Labels']['saea.launch.runner'] = runner_sha
    path.write_text(json.dumps(config)); release.write_text(json.dumps(value))
    result = m.execute(path, 'once')
    assert result['status'] == 'running' and docker.launches == ['producer','job0']
    assert result['jobs'][0]['generation_release']['cpu_scoring_may_continue'] is True


def test_cross_version_foreign_container_without_intent_rejected(queue, tmp_path):
    path, config, docker = queue
    launch, old, _ = historical_predecessor(queue, tmp_path)
    (Path(json.loads(launch.read_text())['receipt_dir'])/'launch-intent.json').unlink()
    with pytest.raises(ValueError, match='durable launch intent'):
        m.execute(path, 'once')
    assert docker.launches == ['previous-version']
