"""Queue custody uses actual graph journals, without model/network calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import a0_recovery_queue as q
from test_a0_native_recovery import stopped
from test_llama_server_transport import _spawn_live_matching_process
from speech_aware_evidence_acquisition.core.flight.native_recovery import load_native_recovery
from dataclasses import replace


def dead_process():
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    identity=q.process_identity(child.pid)
    child.terminate();child.wait()
    return identity


def source(tmp_path,unknown=False):
    old,gate,calls=stopped(tmp_path,unknown=unknown)
    return old


def test_live_worker_never_aborted_or_replayed(tmp_path):
    old=source(tmp_path)
    with pytest.raises(ValueError,match='still live'):
        q.reconcile(old.world.data_root,old.plan,'recovery',old.client.arm,tmp_path/'queue',q.process_identity(os.getpid()))


def test_dead_known_response_reconciles_exact_lease_and_publishes_real_receipt(tmp_path):
    old=source(tmp_path);data=old.world.data_root;started=dead_process()
    lock=data/'locks/flight.lock';q.save(lock,{'run_id':old.plan.run_id,'attempt_id':old.plan.attempt_id,'pid':started['pid']})
    attempt=old.client._adapter._attempt_handle.path
    document=q.read(attempt);document['state']='OPENED';attempt.write_bytes(q.canonical(document))
    root=data/'work/queue';declaration=q.reconcile(data,old.plan,'recovery',old.client.arm,root,started)
    assert q.read(attempt)['state']=='ABORTED' and not lock.exists()
    assert q.read(root/'preserved-flight.lock.json')['pid']==started['pid']
    replay=load_native_recovery(declaration,data_root=data,plan=replace(old.plan,attempt_id='recovery'),arm=old.client.arm)
    assert len(replay.request_ids)==1
    assert q.read(root/'reconciliation.json')['contacts'][0]['state']=='observed'


def test_unknown_contact_preserves_lease_and_refuses_replay(tmp_path):
    old=source(tmp_path,unknown=True);data=old.world.data_root;started=dead_process()
    lock=data/'locks/flight.lock';q.save(lock,{'run_id':old.plan.run_id,'attempt_id':old.plan.attempt_id,'pid':started['pid']})
    with pytest.raises(ValueError,match='unknown generation'):
        q.reconcile(data,old.plan,'recovery',old.client.arm,data/'work/queue',started)
    assert lock.exists() and not (data/'work/queue/recovery.json').exists()


def test_other_lease_never_removed(tmp_path):
    old=source(tmp_path);data=old.world.data_root;started=dead_process()
    lock=data/'locks/flight.lock';q.save(lock,{'run_id':'other','attempt_id':old.plan.attempt_id,'pid':started['pid']})
    with pytest.raises(ValueError,match='another worker'):
        q.reconcile(data,old.plan,'recovery',old.client.arm,data/'work/queue',started)
    assert lock.exists()


def test_repeated_chain_is_explicitly_held_before_losing_ancestor(tmp_path):
    old=source(tmp_path);data=old.world.data_root;started=dead_process()
    pin=data/'runs'/old.plan.run_id/old.plan.attempt_id/'resume-declaration.json'
    q.save(pin,{'predecessors':[{'attempt_id':'original'}]})
    with pytest.raises(ValueError,match='multi-predecessor'):
        q.reconcile(data,old.plan,'recovery',old.client.arm,data/'work/queue',started)
    assert q.read(pin)['predecessors'][0]['attempt_id']=='original'


def test_pid_reuse_does_not_mistake_different_process_for_worker():
    identity=q.process_identity(os.getpid());identity['start_ticks']-=1
    assert q.require_terminal(identity)['terminal']


def test_queue_existing_launch_without_worker_identity_never_respawns(tmp_path,monkeypatch):
    data=tmp_path;out=data/'work/codex/queue';own=out/'attempt-0'
    q.save(own/'launch-intent.json',{'index':0,'recovery':None})
    monkeypatch.setattr(q.subprocess,'Popen',lambda *a,**k:pytest.fail('must not spawn ambiguous previous launch'))
    with pytest.raises(ValueError,match='lacks worker identity'):
        q.supervise(tmp_path,data,'candidate',out)


def test_kernel_guard_owner_blocks_terminal_reconciliation(tmp_path):
    old=source(tmp_path);data=old.world.data_root;started=dead_process()
    with q.exclusive(data/'locks/generation.guard'):
        with pytest.raises(BlockingIOError):
            q.reconcile(data,old.plan,'recovery',old.client.arm,data/'work/queue',started)
    assert not (data/'work/queue/recovery.json').exists()


def test_queue_restart_observes_live_worker_without_second_launch(tmp_path,monkeypatch):
    out=tmp_path/'work/codex/queue';own=out/'attempt-0'
    q.save(own/'launch-intent.json',{'index':0,'recovery':None})
    q.save(own/'started.json',q.process_identity(os.getpid()))
    monkeypatch.setattr(q.subprocess,'Popen',lambda *a,**k:pytest.fail('live worker must never duplicate'))
    def stop_wait(seconds):
        assert seconds==600
        raise InterruptedError('verified wait')
    monkeypatch.setattr(q.time,'sleep',stop_wait)
    with pytest.raises(InterruptedError,match='verified wait'):
        q.supervise(tmp_path,tmp_path,'candidate',out)


def test_corrupt_spend_never_releases_stale_lease(tmp_path):
    old=source(tmp_path);data=old.world.data_root;started=dead_process()
    lock=data/'locks/flight.lock';q.save(lock,{'run_id':old.plan.run_id,'attempt_id':old.plan.attempt_id,'pid':started['pid']})
    attempt=old.client._adapter._attempt_handle.path
    document=q.read(attempt);document['actual']['calls']+=1;attempt.write_bytes(q.canonical(document))
    with pytest.raises(ValueError,match='usage and contact journal differ'):
        q.reconcile(data,old.plan,'recovery',old.client.arm,data/'work/queue',started)
    assert lock.exists() and not (data/'work/queue/recovery.json').exists()


def test_unresolved_fit_never_releases_lease(tmp_path):
    old=source(tmp_path);data=old.world.data_root;started=dead_process()
    fit={'run_id':'fit-run','attempt_id':'fit-attempt'}
    endpoint=data/'prepared'/fit['run_id']/fit['attempt_id']/'orphan.request-fit.json.contacts.jsonl'
    endpoint.parent.mkdir(parents=True,exist_ok=True);endpoint.write_text('{}\n')
    lock=data/'locks/flight.lock';q.save(lock,{'run_id':old.plan.run_id,'attempt_id':old.plan.attempt_id,'pid':started['pid']})
    with pytest.raises(ValueError,match='unresolved FIT'):
        q.reconcile(data,old.plan,'recovery',old.client.arm,data/'work/queue',started,fit)
    assert lock.exists()


def adoption_fixture(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    old=source(tmp_path/'original');data=old.world.data_root
    repo=data/'repo';run=data/'work/codex/legacy/run';out=data/'work/codex/adopted'
    name='fixed439';worker=q.process_identity(os.getpid());wrapper={**worker,'pid':worker['pid']+1000}
    spec={'run_id':old.plan.run_id,'attempt_id':old.plan.attempt_id,'profile':'tp4',
          'workload_kind':'tfrl-hardset-a0-observation','baseline':name}
    q.save(repo/'docs/readiness'/name/'candidate.json',spec)
    q.save(repo/'docs/readiness'/name/'generation-plan-registered-v1.json',{'execution_plan':q.asdict(old.plan)})
    q.save(repo/'configs/baseline'/f'{name}.json',{'native_contact_fit':{'run_id':'fit','attempt_id':'fit-v1'}})
    q.save(run/'started.json',{'pid':worker['pid'],'time':q.time.time(),'study_commit':'a'*40,'candidate':name,'rows':439,'profile':'tp4'})
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda _: 'a'*40)
    monkeypatch.setattr(q,'process_identity',lambda pid: dict(worker if pid==worker['pid'] else wrapper))
    commands={worker['pid']:[b'python',b'scripts/analysis/run_real_engine_candidate.py',b'--repo',str(repo).encode(),b'--data',str(data).encode(),b'--candidate',name.encode(),b'--out',str(run).encode()],
              wrapper['pid']:[b'bash',str(repo/'scripts/flights/nx/hardset_a0_launch.sh').encode(),b'dispatch']}
    monkeypatch.setattr(q,'process_command',lambda pid:commands[pid])
    monkeypatch.setattr(q,'container_identity',lambda pid,cid:cid)
    return repo,data,name,out,run,worker,wrapper,commands


def test_adopt_live_legacy_never_spawns_or_mutates_source(tmp_path,monkeypatch):
    repo,data,name,out,run,worker,wrapper,commands=adoption_fixture(tmp_path,monkeypatch)
    before={p:p.read_bytes() for p in repo.rglob('*') if p.is_file()}
    monkeypatch.setattr(q.subprocess,'Popen',lambda *a,**k:pytest.fail('adoption must not launch'))
    q.adopt(repo,data,name,out,run=run,wrapper_pid=wrapper['pid'],container_id='c'*64,container_pid=worker['pid'])
    assert q.read(out/'attempt-0/launch-intent.json')['adopted']
    assert all(p.read_bytes()==v for p,v in before.items())
    def waiting(n):assert n==600;raise InterruptedError('same worker observed')
    monkeypatch.setattr(q.time,'sleep',waiting)
    with pytest.raises(InterruptedError,match='same worker'):
        q.supervise(repo,data,name,out)


@pytest.mark.parametrize('fault',['commit','command','docker_pid','cgroup'])
def test_adopt_rejects_wrong_live_identity(tmp_path,monkeypatch,fault):
    repo,data,name,out,run,worker,wrapper,commands=adoption_fixture(tmp_path,monkeypatch)
    if fault=='commit':
        doc=q.read(run/'started.json');doc['study_commit']='b'*40;(run/'started.json').write_bytes(q.canonical(doc))
    if fault=='command':commands[worker['pid']][-1]=b'/another/run'
    if fault=='cgroup':monkeypatch.setattr(q,'container_identity',lambda *a:(_ for _ in ()).throw(ValueError('wrong cgroup')))
    with pytest.raises(ValueError):q.adopt(repo,data,name,out,run=run,wrapper_pid=wrapper['pid'],container_id='c'*64,container_pid=worker['pid']+(fault=='docker_pid'))
    assert not (out/'registration.json').exists()


def test_adopt_race_or_existing_custody_never_launches(tmp_path,monkeypatch):
    repo,data,name,out,run,worker,wrapper,commands=adoption_fixture(tmp_path,monkeypatch)
    q.save(out/'partial.json',{'partial':True})
    with pytest.raises(ValueError,match='already has custody'):
        q.adopt(repo,data,name,out,run=run,wrapper_pid=wrapper['pid'],container_id='c'*64,container_pid=worker['pid'])


def test_adopted_completion_requires_wrapper_terminal(tmp_path,monkeypatch):
    repo,data,name,out,run,worker,wrapper,commands=adoption_fixture(tmp_path,monkeypatch)
    q.adopt(repo,data,name,out,run=run,wrapper_pid=wrapper['pid'],container_id='c'*64,container_pid=worker['pid'])
    plan=q.ExecutionPlan(**q.read(out/'attempt-0/plan.json'))
    monkeypatch.setattr(q,'process_identity',lambda pid:wrapper if pid==wrapper['pid'] else None)
    with pytest.raises(ValueError,match='still live'):q.adopted_completion(out/'attempt-0',data,plan)


def test_partial_adoption_never_dispatches_recovery(tmp_path,monkeypatch):
    own=tmp_path/'work/codex/queue/attempt-0'
    q.save(own/'launch-intent.json',{'index':0,'recovery':None,'adopted':True})
    q.save(own/'started.json',dead_process())
    monkeypatch.setattr(q.subprocess,'Popen',lambda *a,**k:pytest.fail('partial adoption must not launch'))
    with pytest.raises(ValueError,match='partial adoption'):
        q.supervise(tmp_path,tmp_path,'candidate',own.parent)


def test_adoption_identity_race_refuses_publication(tmp_path,monkeypatch):
    repo,data,name,out,run,worker,wrapper,commands=adoption_fixture(tmp_path,monkeypatch)
    count=0
    def identity(pid):
        nonlocal count
        count+=1
        answer=dict(worker if pid==worker['pid'] else wrapper)
        if count>2:answer['start_ticks']+=1
        return answer
    monkeypatch.setattr(q,'process_identity',identity)
    with pytest.raises(ValueError,match='changed while adopting'):
        q.adopt(repo,data,name,out,run=run,wrapper_pid=wrapper['pid'],container_id='c'*64,container_pid=worker['pid'])
    assert not (out/'registration.json').exists()


def test_adopted_complete_requires_finalized_source(tmp_path,monkeypatch):
    repo,data,name,out,run,worker,wrapper,commands=adoption_fixture(tmp_path,monkeypatch)
    q.adopt(repo,data,name,out,run=run,wrapper_pid=wrapper['pid'],container_id='c'*64,container_pid=worker['pid'])
    plan=q.ExecutionPlan(**q.read(out/'attempt-0/plan.json'))
    result={'completed':[plan.run_id],'states':{plan.run_id:{'state':'scored'}}}
    q.save(run/'complete.json',{'result':result})
    monkeypatch.setattr(q,'process_identity',lambda pid:None)
    with pytest.raises(ValueError,match='finalized'):q.adopted_completion(out/'attempt-0',data,plan)
    path=data/'attempts'/plan.run_id/(plan.attempt_id+'.json')
    doc=q.read(path);doc['state']='FINALIZED';path.write_bytes(q.canonical(doc))
    assert q.adopted_completion(out/'attempt-0',data,plan)['adopted']
