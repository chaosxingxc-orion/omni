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
