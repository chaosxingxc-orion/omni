import hashlib
import json
import threading
import pytest
from generation_release_handoff import verify_release, wait_release, DirectoryEvents
from tfrl_current_queue import run_lanes


def fixture(tmp_path):
    expected = dict(candidate='candidate', run_id='run', attempt_id='attempt',
                    study_commit='a'*40, profile='profile', service_session='session', rows=313)
    root = tmp_path/'runs/run/attempt';root.mkdir(parents=True)
    seal = {};artifacts = {}
    for key, name in [('generation_manifest','run.generation-manifest.json'),
                      ('score_ledger_cells','score-ledger-cells.json'),('score_jobs','score-jobs.jsonl')]:
        p=root/name;p.write_text('{}')
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        artifacts[key]={'path':str(p),'sha256':digest}
        seal[key+'_path']=str(p);seal[key+'_sha256']=digest
    doc=dict(schema='saea-generation-released-v1',**expected,generation_sealed=True,
             generation_lease_released=True,scoring_complete=False,
             released_at_unix=123.,campaign_seconds_to_release=12.,
             consumer_requires_own_lease_and_readiness=True,artifacts=artifacts,seal_receipt=seal)
    path=tmp_path/'work/codex/queue/h1-run/generation-released.json'
    path.parent.mkdir(parents=True);path.write_text(json.dumps(doc))
    return path,expected,doc


@pytest.mark.parametrize('fault',[None,'run','rows','commit','lease','score','artifact','path','seal','missing'])
def test_verified_release_requires_exact_source_and_sealed_bytes(tmp_path,fault):
    path,expected,doc=fixture(tmp_path)
    if fault=='run':doc['run_id']='different'
    if fault=='rows':doc['rows']=312
    if fault=='commit':doc['study_commit']='b'*40
    if fault=='lease':doc['generation_lease_released']=False
    if fault=='score':doc['scoring_complete']=True
    if fault=='artifact':Path(doc['artifacts']['score_jobs']['path']).write_text('changed')
    if fault=='path':doc['artifacts']['score_jobs']['path']=str(tmp_path/'outside')
    if fault=='seal':doc['seal_receipt']['score_jobs_sha256']='b'*64
    if fault=='missing':doc['artifacts'].pop('score_jobs')
    path.write_text(json.dumps(doc))
    if fault:
        with pytest.raises(ValueError):verify_release(path,data=tmp_path,expected=expected)
    else:assert verify_release(path,data=tmp_path,expected=expected)['rows']==313


from pathlib import Path


def test_same_live_child_waits_600_seconds_without_replay(tmp_path):
    path,expected,doc=fixture(tmp_path);path.unlink();events=[]
    class Child:
        def poll(self):events.append('same-child');return None
    class Watch:
        def __init__(self,p):events.append('watch')
        def add(self,p):events.append('add')
        def wait(self,seconds):
            assert seconds==600
            events.append('wait');path.write_text(json.dumps(doc))
        def close(self):events.append('close')
    wait_release(Child(),path,data=tmp_path,expected=expected,events_factory=Watch)
    assert events==['watch','add','same-child','wait','close']


@pytest.mark.parametrize('exit_code',[0,1])
def test_terminal_without_release_never_advances(tmp_path,exit_code):
    path,expected,doc=fixture(tmp_path);path.unlink()
    class Child:
        def poll(self):return exit_code
    with pytest.raises(RuntimeError,match='without release'):
        wait_release(Child(),path,data=tmp_path,expected=expected)


def test_kernel_directory_event_wakes_after_atomic_publish(tmp_path):
    watch=DirectoryEvents(tmp_path)
    try:
        tmp_path.joinpath('tmp').write_text('released')
        tmp_path.joinpath('tmp').rename(tmp_path/'generation-released.json')
        assert watch.wait(.1)
        assert not watch.wait(0)
    finally:watch.close()


@pytest.mark.parametrize('failure',[None,'score1','read1','release1'])
def test_next_generation_starts_before_first_score_finishes(tmp_path,failure):
    events=[];second_started=threading.Event()
    class Child:
        def __init__(self,role,job):self.role,self.job=role,job
        def wait(self):
            if self.role=='score' and self.job=='1':
                assert second_started.wait(5),'next generation blocked on previous score'
            events.append(self.role+self.job)
            return int(failure==self.role+self.job)
    def generation(job):
        events.append('generation'+job)
        if job=='2':second_started.set()
        return Child('score',job)
    def read(job):events.append('read-start'+job);return Child('read',job)
    def release(job,child):
        events.append('release'+job)
        if failure=='release'+job:raise ValueError('unverified release')
    call=lambda:run_lanes(['1','2'],generation,read,lambda *a,**k:None,release)
    if failure:
        with pytest.raises((ValueError,RuntimeError)):call()
    else:call()
    if failure=='release1':assert events==['generation1','release1']
    else:
        assert events.index('generation2')<events.index('score1')
        assert ('read-start1' in events)==(failure!='score1')
        assert events.count('generation1')==events.count('generation2')==1
        assert 'read2' in events


@pytest.mark.parametrize('iteration',[1,2,3])
def test_real_queue_wires_verified_release_and_detached_children(tmp_path,monkeypatch,iteration):
    import tfrl_current_queue as m
    monkeypatch.setattr(m,'D',tmp_path)
    tree=tmp_path/'tree';study=tree/'studies/speech-aware-evidence-acquisition'
    prefix='seed' if iteration==1 else 'bank'
    def put(path,doc):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(doc))
    for half,n in [('h1',313),('h2',312)]:
        name=f'slurp-tfrl-current-{prefix}-i{iteration}-{half}-v1'
        spec=dict(run_id=f'SAEA-SLURP-TFRL-CURRENT-{prefix.upper()}-I{iteration}-{half.upper()}-V1',
                  attempt_id='generation-v1',fit_run_id='fit-'+half,rows=n,iteration=iteration,
                  profile='qwen3-omni-real500-2xtp4-s128-b2048-v1',service_session='session',
                  workload_kind='tfrl-real-seed-half' if iteration==1 else 'tfrl-real-banked-half')
        put(study/'docs/readiness'/name/'candidate.json',spec)
        put(study/'docs/readiness'/name/'analysis-recipe-v1.json',spec)
        put(study/'configs/experiment'/f'{name}.json',{'cells':[{'readiness':{'state':'ready'}}]})
    monkeypatch.setattr(m.subprocess,'check_output',lambda cmd,**kw:'' if cmd[-1]=='--porcelain' else 'a'*40)
    launches=[];releases=[]
    class Child:
        pid=999
        def wait(self):return 0
    def launch(cmd,**kw):
        launches.append((cmd,kw));return Child()
    monkeypatch.setattr(m.subprocess,'Popen',launch)
    def released(child,path,*,data,expected):
        assert isinstance(child,Child)
        assert path.name=='generation-released.json'
        assert expected['study_commit']=='a'*40 and expected['rows'] in (312,313)
        assert expected['candidate'].startswith(f'slurp-tfrl-current-{prefix}-i{iteration}')
        releases.append(expected)
    monkeypatch.setattr(m,'wait_release',released)
    out=tmp_path/'work/codex/newqueue'
    m.run(tree,out,iteration)
    assert len(launches)==4 and len(releases)==2
    assert all(kw['start_new_session'] for cmd,kw in launches)
    assert json.loads((out/'status.json').read_bytes())['state']=='complete'
    with pytest.raises(ValueError,match='fresh owned'):m.run(tree,out,iteration)
    assert len(launches)==4
