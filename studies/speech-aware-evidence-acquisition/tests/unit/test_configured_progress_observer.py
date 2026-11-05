"""Progress counts verified samples/groups, with real journals and runtime output."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_contact_journal import payload
from test_frozen_core_adapter import _plan
from test_real_engine_progress import fixture as row_fixture
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal, canonical, digest

spec=importlib.util.spec_from_file_location('progress_observer_test',Path(__file__).parents[2]/'scripts/environment/configured_progress_observer.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def pin(p):return {'path':str(p),'sha256':m.queue.launcher.digest(p)}

def test_pinned_plan_source_is_preserved_in_journal_validation(tmp_path):
    calls=[]
    class Plan:
        run_id='SAEA-OBSERVER-V1';attempt_id='generation-v1';protocol_hash='a'*64;split_identity_hash='b'*64
        def validate(self,*,source_root=None):
            calls.append(source_root)
            assert source_root==tmp_path/'original-source'
    journal=ContactJournal(tmp_path,Plan(),'bare-core',source_root=tmp_path/'original-source')
    assert calls==[tmp_path/'original-source'] and journal.identity['run_id']=='SAEA-OBSERVER-V1'

def test_observed_and_unknown_costs_stay_distinct(tmp_path):
    plan=_plan();producer={'mounts':[{'source':str(tmp_path),'target':str(tmp_path),'readonly':True}]}
    journal=ContactJournal(tmp_path,plan,'bare-core')
    for rid in ('req-1','req-2'):journal.intent(payload(rid))
    journal.response('req-1',digest(canonical(payload())),canonical({'text':'ok','usage':{'prompt_tokens':8,'completion_tokens':3}}),2)
    result,records=m.native_contacts(producer,tmp_path,plan,'bare-core')
    assert result['observed_contacts']==1 and result['attempted_contacts']==2
    assert result['inflight_or_unknown_contacts']==1 and result['unknown_cost'] is None
    assert result['known_usage']['prompt_tokens']==8 and result['new_model_calls']==0
    assert result['observer_model_calls']==0 and 'observer invocation only' in result['new_model_calls_scope']
    calls=result['workload_model_calls']
    assert (calls['attempted'],calls['observed'],calls['inflight_or_unknown'])==(2,1,1)
    assert (calls['run_id'],calls['attempt_id'])==(plan.run_id,plan.attempt_id)
    assert 'excludes FIT and prior attempts' in calls['scope']
    assert 'not a polling-interval delta' in calls['scope']
    fit={'schema':'saea-native-fit-custody-v1','identity':journal.identity,'payload':payload()}
    fp=journal.root/(digest(canonical(m.comparable_request(payload())))+'.fit.json');fp.write_bytes(canonical(fit))
    result,_=m.native_contacts(producer,tmp_path,plan,'bare-core')
    assert result['fit_custody_records']==1 and result['known_usage']['calls']==1
    fit['identity']={};fp.write_bytes(canonical(fit))
    with pytest.raises(ValueError,match='FIT custody'):m.native_contacts(producer,tmp_path,plan,'bare-core')
    fp.unlink()
    journal._path('req-2','intent').unlink()
    journal._path('req-2','response').write_text('{}')
    with pytest.raises(ValueError,match='orphan'):m.native_contacts(producer,tmp_path,plan,'bare-core')

def test_checkpoint_snapshot_validates_roster_and_removes_scratch(tmp_path,monkeypatch):
    ck,roster=row_fixture(tmp_path);scratch=tmp_path/'scratch';scratch.mkdir()
    monkeypatch.setattr(m.settlement,'mapped',lambda *a,**k:ck)
    plan=SimpleNamespace(run_id='RUN',attempt_id='attempt')
    job={'total_rows':2,'roster':pin(roster)};lane={'cards':4,'instances':1}
    with ck.open('ab') as f:f.write(b'{"partial"')
    result=m.row_progress(job,{},tmp_path,plan,20,'running',scratch,lane)
    assert result['completed_rows']==1 and result['torn_tail_bytes']>0
    assert result['cards']==4 and result['wall_seconds_per_complete_row']==20
    assert not list(scratch.iterdir()) and not result['full_roster_comparison_eligible']

@pytest.mark.parametrize('strategy',['ace','reflexion'])
def test_real_runtime_learning_updates_are_replayed_not_call_divided(tmp_path,monkeypatch,strategy):
    if strategy=='ace':from test_ace_runtime import setup
    else:from test_reflexion_runtime import setup
    runtime,transport,captured,source,spec,fits=setup(tmp_path,monkeypatch)
    runtime.run_phases();plan=runtime.plan;root=runtime._run_dir()/'tfgrpo/learner'
    job=tmp_path/'observer-job.json';job.write_text(json.dumps({'run_id':plan.run_id,'runtime_rows':1,'batch_sha256':m.sha(json.loads((root/'batch.json').read_bytes()))}))
    producer={'mounts':[{'source':str(tmp_path),'target':str(tmp_path),'readonly':True},
        {'source':str(tmp_path/'data/contact-journals'),'target':str(tmp_path/'contact-journals'),'readonly':True},
        {'source':str(root),'target':str(tmp_path/'runs'/plan.run_id/plan.attempt_id/'tfgrpo/learner'),'readonly':True}]}
    contact,native=m.native_contacts(producer,tmp_path,plan,'prepared-context-text')
    j={'learner_job':pin(job),'total_rows':1}
    result=m.learning_progress(j,producer,tmp_path,plan,native)
    assert result['completed_rows']==1 and contact['observed_contacts']==(2 if strategy=='ace' else 1)
    update=root/'updates/0000.json';value=json.loads(update.read_bytes());value['query_id']='foreign';update.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='group identity'):m.learning_progress(j,producer,tmp_path,plan,native)
    runtime._client.close()

@pytest.mark.parametrize('state',['running','failed','complete'])
def test_process_success_never_claims_full_roster_or_unknown_cost_zero(tmp_path,monkeypatch,state):
    observed={'state':'running' if state=='running' else 'exited','running':state=='running',
        'exit_code':1 if state=='failed' else 0,'started_at':'2026-11-03T10:00:00Z','finished_at':'2026-11-03T10:10:00Z'}
    monkeypatch.setattr(m,'source',lambda j:({},observed,None))
    monkeypatch.setattr(m,'native_contacts',lambda *a:({'known_usage':{'calls':7},'unknown_cost':None,'workload_model_calls':{'attempted':8,'observed':7,'inflight_or_unknown':1}},{}))
    monkeypatch.setattr(m,'learning_progress',lambda *a:{'completed_rows':2,'total_rows':5})
    j={'id':'sample','kind':'learning-updates','total_rows':5,'logical_data':str(tmp_path),'arm':'test'}
    result=m.observe_job(j,{'cards':2,'instances':1},tmp_path,m.timestamp('2026-11-03T10:10:00Z'))
    assert result['completed_rows']==2 and result['wall_seconds_per_complete_row']==300
    assert result['state']==('stopped' if state=='complete' else state)
    assert result['estimated_remaining_seconds']==(900 if state=='running' else None)
    assert not result['scientific_gain_claimed'] and result['new_model_calls']==0
    assert result['observer_model_calls']==0 and result['workload_model_calls']['observed']==7

def test_not_launched_never_reads_native_or_reports_rate(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'source',lambda j:({}, {'state':'not-launched'},None))
    monkeypatch.setattr(m,'native_contacts',lambda *a:pytest.fail('not launched'))
    j={'id':'future','kind':'learning-updates','total_rows':51,'logical_data':str(tmp_path)}
    result=m.observe_job(j,{'cards':2,'instances':1},tmp_path,0)
    assert result['state']=='not-started' and result['wall_seconds_per_complete_row'] is None
    assert result['observer_model_calls']==0 and result['workload_model_calls'] is None

def test_source_exact_launch_result_is_required(tmp_path,monkeypatch):
    from test_configured_job_launch import configured
    launch,doc=configured(tmp_path);root=Path(doc['receipt_dir']);root.mkdir(parents=True)
    helper=Path(m.queue.launcher.__file__);runner=m.queue.launcher.digest(helper)
    (root/'launch-intent.json').write_text(json.dumps({'schema':m.queue.launcher.SCHEMA,'version':m.queue.launcher.VERSION,
        'config_sha256':m.queue.canonical_hash(doc),'runner_sha256':runner,'expected_commit':doc['expected_commit'],'automatic_retry':False}))
    (root/'launch-result.json').write_text(json.dumps({'returncode':0,'stdout':'d'*64,'stderr':'','automatic_retry':False}))
    monkeypatch.setattr(m.queue.launcher,'status',lambda *a:{'state':'running','container_id':'c'*64})
    with pytest.raises(ValueError,match='exact launch result'):m.source({'launch':pin(launch),'launcher':pin(helper)})


def test_recovery_rows_never_become_fresh_inference_throughput(tmp_path,monkeypatch):
    plan=_plan();producer={'mounts':[{'source':str(tmp_path),'target':str(tmp_path),'readonly':True}]}
    root=tmp_path/'runs'/plan.run_id/plan.attempt_id;root.mkdir(parents=True)
    receipt=tmp_path/'recovery.json'
    receipt.write_text(json.dumps({'schema':'saea-native-recovery-v1',
        'target_plan':{k:getattr(plan,k) for k in ('run_id','attempt_id','protocol_hash','split_identity_hash')},
        'source':{'plan':{'attempt_id':'previous-attempt'}}}))
    declaration={'schema':'saea-resume-declaration-v1','run_id':plan.run_id,'attempt_id':plan.attempt_id,
        'native_recovery_receipt':pin(receipt)}
    (root/'resume-declaration.json').write_text(json.dumps(declaration))
    observed={'state':'running','running':True,'exit_code':0,'started_at':'2026-11-03T10:00:00Z'}
    monkeypatch.setattr(m,'source',lambda j:(producer,observed,plan))
    monkeypatch.setattr(m,'native_contacts',lambda *a:({'known_usage':{'calls':245},'workload_model_calls':{'attempted':339,'observed':245,'inflight_or_unknown':94}},{}))
    monkeypatch.setattr(m,'row_progress',lambda *a:{'completed_rows':100,'total_rows':200,'rows_per_second':10})
    job={'id':'recovery','kind':'row-checkpoint','total_rows':200,'logical_data':str(tmp_path),'arm':'test'}
    result=m.observe_job(job,{'cards':4,'instances':1},tmp_path,m.timestamp('2026-11-03T10:00:10Z'))
    assert result['completed_rows']==100 and result['contacts']['known_usage']['calls']==245
    assert result['observer_model_calls']==0 and result['new_model_calls']==0
    assert result['workload_model_calls']['observed']==245
    assert result['wall_seconds_per_complete_row'] is None and result['rows_per_second'] is None
    assert result['estimated_remaining_seconds'] is None
    assert result['reconstruction_wall_seconds_per_row']==.1
    assert result['recovery']['source_attempt_id']=='previous-attempt'
    receipt.write_text('{}')
    with pytest.raises(ValueError,match='receipt hash'):m.recovery_context(producer,tmp_path,plan)
    declaration['attempt_id']='wrong';(root/'resume-declaration.json').write_text(json.dumps(declaration))
    with pytest.raises(ValueError,match='declaration identity'):m.recovery_context(producer,tmp_path,plan)



def test_real_atomic_publication_temporary_is_not_a_contact(tmp_path,monkeypatch):
    import os, threading
    from speech_aware_evidence_acquisition.core import atomic_file
    plan=_plan();producer={'mounts':[{'source':str(tmp_path),'target':str(tmp_path),'readonly':True}]}
    journal=ContactJournal(tmp_path,plan,'bare-core');journal.root.mkdir(parents=True,exist_ok=True)
    reached=threading.Event();release=threading.Event();original=os.fsync;errors=[]
    def fsync(fd):
        if threading.current_thread().name=='publisher':
            reached.set();assert release.wait(10)
        return original(fd)
    monkeypatch.setattr(atomic_file.os,'fsync',fsync)
    def publish():
        try:atomic_file.publish_new(journal._path('req-1','intent'),b'{}')
        except Exception as e:errors.append(e)
    thread=threading.Thread(target=publish,name='publisher');thread.start()
    try:
        assert reached.wait(10)
        result,records=m.native_contacts(producer,tmp_path,plan,'bare-core')
        assert result['attempted_contacts']==0 and len(result['uncommitted_publish_files'])==1 and records=={}
    finally:release.set();thread.join(10)
    assert not errors and not thread.is_alive()

@pytest.mark.parametrize('name',['illegal.json','.'+'a'*64+'.intent.json.bad'])
def test_observer_rejects_invalid_visible_files_and_temporary_symlinks(tmp_path,name):
    plan=_plan();producer={'mounts':[{'source':str(tmp_path),'target':str(tmp_path),'readonly':True}]}
    journal=ContactJournal(tmp_path,plan,'bare-core');journal.root.mkdir(parents=True,exist_ok=True)
    path=journal.root/name
    if name.startswith('.'):path.symlink_to(tmp_path/'absent')
    else:path.write_text('{}')
    with pytest.raises(ValueError,match='unexpected contact journal file'):m.native_contacts(producer,tmp_path,plan,'bare-core')


def test_temporary_disappearing_after_lstat_is_snapshot_transient(tmp_path,monkeypatch):
    plan=_plan();producer={'mounts':[{'source':str(tmp_path),'target':str(tmp_path),'readonly':True}]}
    journal=ContactJournal(tmp_path,plan,'bare-core');journal.root.mkdir(parents=True,exist_ok=True)
    path=journal.root/('.'+'a'*64+'.intent.json.inflight');path.write_bytes(b'{}')
    original=Path.lstat
    def lstat(p,*args,**kwargs):
        result=original(p,*args,**kwargs)
        if p==path:p.unlink()
        return result
    monkeypatch.setattr(Path,'lstat',lstat)
    result,_=m.native_contacts(producer,tmp_path,plan,'bare-core')
    assert result['uncommitted_publish_files']==[path.name] and result['attempted_contacts']==0
