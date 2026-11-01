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
