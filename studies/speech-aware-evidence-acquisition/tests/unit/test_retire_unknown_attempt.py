import copy
import fcntl
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
import pytest
from test_frozen_core_adapter import _plan,_payload
from speech_aware_evidence_acquisition.core.attempts import AttemptStore
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal,NativeContactReplay,ContactJournalError,canonical,digest
from scripts.environment import retire_unknown_attempt as m


def fixture(tmp_path, monkeypatch):
    plan=_plan();data=tmp_path/'data'; data.mkdir()
    attempt=AttemptStore(data/'attempts').open_attempt(plan.run_id,plan.attempt_id,carrier_lock_key=plan.carrier_lock_key,reserved_calls=2,reserved_audio_seconds=250)
    attempt.record_usage(calls=2,audio_seconds=200)
    j=ContactJournal(data,plan,'bare-core')
    for rid in ('req-1','req-2'):
        p=_payload(request_id=rid,history=[{'role':'control','contact_id':'solver:'+rid}]);j.intent(p)
        if rid=='req-1':j.response(rid,digest(canonical(p)),b'{"text":"ok"}',1)
    locks=data/'locks';locks.mkdir();(locks/'generation.guard').touch()
    lease={'run_id':plan.run_id,'attempt_id':plan.attempt_id,'pid':999999999,'session_id':'resident','study_head':'a'*40,'run_ids':[plan.run_id]}
    (locks/'flight.lock').write_bytes(canonical(lease))
    rp=tmp_path/'runtime.json';rp.write_text(json.dumps({'deployment':{'instances':[{'instance_id':'one','host':'127.0.0.1','port':8200}]}}))
    session=data/'serving/resident/session.json';session.parent.mkdir(parents=True);session.write_text(json.dumps({'session_id':'resident','identity':{'runtime_receipt_sha256':m.sha(rp)},'deployment_instance_ids':['one']}))
    pp=tmp_path/'plan.json';pp.write_text(json.dumps({'execution_plan':asdict(plan),'expected_session_id':'resident'}))
    st=tmp_path/'started.json';st.write_text(json.dumps({'pid':999999999,'study_commit':'a'*40}))
    monkeypatch.setattr(m,'docker_inspect',lambda _: {'Id':'c'*64,'State':{'Status':'exited','Running':False,'Pid':0,'ExitCode':143,'StartedAt':'2999-01-01T00:00:00Z'},'Config':{'Labels':{'saea.launch.config':'b'*64}}})
    monkeypatch.setattr(m,'quiescence',lambda _: {'quiet':True})
    monkeypatch.setattr(m,'verify_service',lambda *a: None)
    fit=tmp_path/'fit';fit.mkdir();checkpoint=tmp_path/'checkpoint';checkpoint.write_text('checkpoint')
    config={'schema':m.SCHEMA,'mode':'retire-unknown-no-retry','data':str(data),'out':str(data/'work/codex/retire'),
      'plan':str(pp),'plan_sha256':m.sha(pp),'run_id':plan.run_id,'attempt_id':plan.attempt_id,'arm':'bare-core',
      'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'pid':999999999,'container':'c'*64,'exit_code':143,
      'launch_config_identity':'b'*64,'study_commit':'a'*40,'started':str(st),'started_sha256':m.sha(st),
      'lease_sha256':m.sha(locks/'flight.lock'),'attempt_sha256':m.sha(attempt.path),'actual':{'calls':2,'audio_seconds':200.0},
      'runtime_receipt':str(rp),'runtime_receipt_sha256':m.sha(rp),'session_id':'resident','unknown_count':1,
      'fit_receipts_root':str(fit),'fit_count':0,'checkpoint':str(checkpoint),'checkpoint_sha256':m.sha(checkpoint),
      'new_baseline_run_id':'new-baseline-run','config_sha256':'d'*64,'inventory_sha256':digest(canonical(m.inventory(j)))}
    return config,plan,j


def test_retire_unknown_and_idempotent(tmp_path,monkeypatch):
    c,plan,j=fixture(tmp_path,monkeypatch);result=m.execute(c)
    assert result['status']=='RETIRED_UNKNOWN_RESOURCE_RELEASED' and result['unknown_count']==1
    assert not (Path(c['data'])/'locks/flight.lock').exists()
    assert (Path(c['data'])/'locks/generation.guard').exists()
    assert AttemptStore(Path(c['data'])/'attempts').handle(plan.run_id,plan.attempt_id).state()=='ABORTED'
    assert m.execute(c)==result
    with pytest.raises(ContactJournalError,match='retired'):
        NativeContactReplay(c['data'],plan,replace(plan,attempt_id='other'),'bare-core')


def test_aborted_dispatch_is_archived_without_erasing_cost(tmp_path,monkeypatch):
    c,plan,j=fixture(tmp_path,monkeypatch)
    request=j.inspect('req-2')['intent']['request_sha256']
    j.delivery_event('req-2',request,'dispatch')
    store=AttemptStore(Path(c['data'])/'attempts')
    assert store.abort_if_opened(plan.run_id,plan.attempt_id,reason='HTTP 500')
    before=Path(c['data'])/'attempts'/plan.run_id/(plan.attempt_id+'.json')
    c['attempt_sha256']=m.sha(before);c['inventory_sha256']=digest(canonical(m.inventory(j)))
    result=m.execute(c)
    assert result['actual']=={'calls':2,'audio_seconds':200.0} and result['unknown_count']==1
    assert list(Path(c['out']).glob('*.dispatch.json'))
    assert json.loads((Path(c['out'])/'attempt-before.json').read_bytes())['state']=='ABORTED'
    assert not (Path(c['data'])/'locks/flight.lock').exists()
    assert (Path(c['data'])/'locks/generation.guard').exists()


def test_aborted_changed_evidence_never_releases(tmp_path,monkeypatch):
    c,plan,j=fixture(tmp_path,monkeypatch)
    AttemptStore(Path(c['data'])/'attempts').abort_if_opened(plan.run_id,plan.attempt_id,reason='failed')
    with pytest.raises(ValueError,match='attempt changed'):m.execute(c)
    assert (Path(c['data'])/'locks/flight.lock').exists()


def test_orphan_dispatch_is_rejected(tmp_path,monkeypatch):
    c,plan,j=fixture(tmp_path,monkeypatch)
    (j.root/('f'*64+'.dispatch.json')).write_text('{}')
    with pytest.raises(ValueError,match='orphan delivery'):m.inventory(j)


@pytest.mark.parametrize('fault',['boot','pid','container','lease','plan','attempt','cost','inventory','busy','guard','same_run','unknown'])
def test_no_release_on_bad_evidence(tmp_path,monkeypatch,fault):
    c,plan,j=fixture(tmp_path,monkeypatch)
    if fault=='boot':c['boot_id']='bad'
    elif fault=='pid':c['pid']=os.getpid()
    elif fault=='container':c['container']='f'*64
    elif fault=='lease':c['lease_sha256']='0'*64
    elif fault=='plan':c['plan_sha256']='0'*64
    elif fault=='attempt':c['attempt_sha256']='0'*64
    elif fault=='cost':c['actual']['calls']=1
    elif fault=='inventory':c['inventory_sha256']='0'*64
    elif fault=='same_run':c['new_baseline_run_id']=plan.run_id
    elif fault=='unknown':c['unknown_count']=0
    elif fault=='busy':monkeypatch.setattr(m,'quiescence',lambda _: (_ for _ in ()).throw(ValueError('busy')))
    guard=None
    if fault=='guard':
        guard=(Path(c['data'])/'locks/generation.guard').open('r+b');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        with pytest.raises((ValueError,BlockingIOError)):m.execute(c)
    finally:
        if guard:guard.close()
    assert (Path(c['data'])/'locks/flight.lock').exists()
    assert AttemptStore(Path(c['data'])/'attempts').handle(plan.run_id,plan.attempt_id).state()=='OPENED'


def test_changed_unknown_after_snapshot_refuses(tmp_path,monkeypatch):
    c,plan,j=fixture(tmp_path,monkeypatch)
    j.response('req-2',j.inspect('req-2')['intent']['request_sha256'],b'{"text":"late"}',1)
    with pytest.raises(ValueError):m.execute(c)
    assert (Path(c['data'])/'locks/flight.lock').exists()


@pytest.mark.parametrize('phase',['marker','abort','lease'])
def test_interrupted_retirement_resumes(tmp_path,monkeypatch,phase):
    c,plan,j=fixture(tmp_path,monkeypatch)
    if phase=='marker':
        original=m.AttemptStore.abort_if_opened
        monkeypatch.setattr(m.AttemptStore,'abort_if_opened',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('crash')))
    elif phase=='abort':
        original=m.publish_new
        def publish(path,blob):
            if path.name=='attempt-aborted.json':raise RuntimeError('crash')
            return original(path,blob)
        monkeypatch.setattr(m,'publish_new',publish)
    else:
        original=m.publish_new
        def publish(path,blob):
            if path.name=='complete.json':raise RuntimeError('crash')
            return original(path,blob)
        monkeypatch.setattr(m,'publish_new',publish)
    with pytest.raises(RuntimeError):m.execute(c)
    if phase=='marker':monkeypatch.setattr(m.AttemptStore,'abort_if_opened',original)
    else:monkeypatch.setattr(m,'publish_new',original)
    assert m.execute(c)['unknown_count']==1
    assert not (Path(c['data'])/'locks/flight.lock').exists()
    new=Path(c['data'])/'locks/flight.lock';new.write_text('new tenant')
    assert m.execute(c)['unknown_count']==1 and new.read_text()=='new tenant'


def test_corrupt_archived_copy_prevents_resume(tmp_path,monkeypatch):
    c,plan,j=fixture(tmp_path,monkeypatch)
    original=m.AttemptStore.abort_if_opened
    monkeypatch.setattr(m.AttemptStore,'abort_if_opened',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('crash')))
    with pytest.raises(RuntimeError):m.execute(c)
    monkeypatch.setattr(m.AttemptStore,'abort_if_opened',original)
    archived=next(Path(c['out']).glob('*.intent.json'));archived.write_text('bad')
    with pytest.raises(ValueError,match='archive journal copy'):m.execute(c)
    assert (Path(c['data'])/'locks/flight.lock').exists()


def test_all_retired_recovery_paths_refuse(tmp_path):
    from speech_aware_evidence_acquisition.core.flight.resume import read_checkpoint_predecessor, read_predecessor_attempt, ResumeRecoveryError
    root=tmp_path;run='retired-run';attempt='old';p=root/'contact-journals'/run/attempt;p.mkdir(parents=True);(p/'retirement.json').write_text('{}')
    directory=root/'runs'/run/attempt
    with pytest.raises(ResumeRecoveryError,match='retired'):read_checkpoint_predecessor(run_id=run,attempt_id=attempt,attempt_dir=directory,expected_checkpoint_sha256='0'*64,materialize_dir=root)
    with pytest.raises(ResumeRecoveryError,match='retired'):read_predecessor_attempt(run_id=run,attempt_id=attempt,attempt_dir=directory)


def test_wrong_live_service_refuses(tmp_path,monkeypatch):
    c,plan,j=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'verify_service',lambda *a: (_ for _ in ()).throw(ValueError('wrong service')))
    with pytest.raises(ValueError,match='wrong service'):m.execute(c)
    assert (Path(c['data'])/'locks/flight.lock').exists()

@pytest.mark.parametrize('phase',['copy','manifest'])
def test_pre_marker_crash_resumes_without_deleting(tmp_path,monkeypatch,phase):
    c,plan,j=fixture(tmp_path,monkeypatch);original=m.publish_new;calls=[]
    def publish(path,blob):
        if (phase=='copy' and path.name.endswith('.response.json')) or (phase=='manifest' and path.name=='retirement.json'):
            raise RuntimeError('crash')
        calls.append(str(path));return original(path,blob)
    monkeypatch.setattr(m,'publish_new',publish)
    with pytest.raises(RuntimeError):m.execute(c)
    monkeypatch.setattr(m,'publish_new',original)
    assert m.execute(c)['unknown_count']==1
