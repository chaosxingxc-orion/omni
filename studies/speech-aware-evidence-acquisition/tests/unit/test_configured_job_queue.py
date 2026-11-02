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


def generation_fixture(queue, tmp_path):
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
        p=data/'runs/SAEA-PRODUCER/generation-v1'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        logical=str(Path('/logical/runs/SAEA-PRODUCER/generation-v1')/name)
        artifacts[key]={'path':logical,'sha256':m.launcher.digest(p)}
        seal[key+'_path']=logical;seal[key+'_sha256']=m.launcher.digest(p)
    release={'schema':'saea-generation-released-v1',**expected,'generation_sealed':True,
        'generation_lease_released':True,'scoring_complete':False,'consumer_requires_own_lease_and_readiness':True,
        'released_at_unix':123.,'campaign_seconds_to_release':12.,'artifacts':artifacts,'seal_receipt':seal}
    (output/'started.json').write_text(json.dumps({'study_commit':expected['study_commit'],'candidate':'candidate','rows':312}))
    path.write_text(json.dumps(config))
    return output/'generation-released.json',release,data


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
    monkeypatch.setattr(m,'release_helper',lambda doc:SimpleNamespace(verify_release=real(doc).verify_release,DirectoryEvents=Events))
    result=m.execute(path,'watch')
    assert result['status']=='complete' and waits==[600]
    assert all(cid!=docker.containers['producer']['Id'] for cid,_ in docker.waits)
