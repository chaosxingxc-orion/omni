import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).parents[2]
import sys
sys.path.insert(0,str(ROOT/'scripts/environment'))
spec=importlib.util.spec_from_file_location('pre_attempt_settlement',ROOT/'scripts/environment/configured_pre_attempt_settlement.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_absence_rejects_generation_and_fit_custody(tmp_path,monkeypatch):
    monkeypatch.setattr(m.s,'mapped',lambda producer,path,**kwargs:tmp_path/path.relative_to('/data'))
    exp={'run_id':'RUN','attempt_id':'gen','fit_run_id':'RUN-FIT','fit_attempt_id':'fit'}
    assert len(m.absent({},Path('/data'),exp)) == 6
    for relative in ('attempts/RUN/gen.json','runs/RUN/gen','contact-journals/RUN/gen','attempts/RUN-FIT/fit.json','runs/RUN-FIT/fit','contact-journals/RUN-FIT/fit'):
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
        with pytest.raises(ValueError,match='custody exists'):m.absent({},Path('/data'),exp)
        p.unlink()


def test_guard_refuses_live_owner(tmp_path):
    import fcntl
    p=tmp_path/'generation.guard';p.touch()
    with p.open('r+b') as active:
        fcntl.flock(active,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            with m.s.guard(p):pass


def setup_case(tmp_path, monkeypatch):
    root=tmp_path/'state';root.mkdir();locks=tmp_path/'locks';locks.mkdir();(locks/'generation.guard').touch()
    started=tmp_path/'output';started.mkdir();(started/'started.json').write_text(json.dumps({'candidate':'candidate','study_commit':'a'*40,'pid':99999999,'time':2e10+1}))
    exp={'candidate':'candidate','run_id':'RUN','attempt_id':'gen','fit_run_id':'RUN-FIT','fit_attempt_id':'fit','service_session':'session','study_commit':'a'*40}
    lease={'schema':'saea-generation-lease-v1','run_id':'RUN','attempt_id':'gen','run_ids':['RUN'],'session_id':'session','study_head':'a'*40,'pid':99999999,'started_at':'2026-01-01T00:00:10+00:00','token':'private'}
    (locks/'flight.lock').write_text(json.dumps(lease));plan=tmp_path/'plan.json';plan.write_text(json.dumps({'execution_plan':{},'cell':'candidate','expected_session_id':'session'}));receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps({'deployment':{'instances':[{'instance_id':'instance'}]}}));session=tmp_path/'session.json';session.write_text(json.dumps({'session_id':'session','identity':{'runtime_receipt_sha256':'b'*64},'deployment_instance_ids':['instance']}));launch=tmp_path/'launch.json';launch.write_text('{}');tool=tmp_path/'configured_job_launch.py';tool.touch();manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'files':{'generation':'e'*64}}));candidate=tmp_path/'candidate.json';candidate.write_text(json.dumps({**exp}))
    producer={'argv':['/python','/repo/scripts/analysis/run_real_engine_candidate.py','--repo','/repo','--data','/data','--candidate','candidate','--out','/data/work/codex/out'],'repo_path':'/repo','expected_commit':'a'*40,'pid_host':True,'mounts':[],'external_admission':{'workspace_manifest':str(manifest)},'environment':{'SAEA_EXPERIMENT_WORKSPACE':'/workspace'}}
    cfg={'schema':m.SCHEMA,'producer_launch':{'path':str(launch),'sha256':'c'*64},'producer_launcher':{'path':str(tool),'sha256':'d'*64},'source_plan':{'path':str(plan),'sha256':'e'*64},'runtime_receipt':{'path':str(receipt),'sha256':'b'*64},'expected':exp,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'state_dir':str(root/'output')};cp=tmp_path/'cfg.json';cp.write_text(json.dumps(cfg))
    monkeypatch.setattr(m.s.launcher,'load_config',lambda p:producer);monkeypatch.setattr(m.s,'pin',lambda p:Path(p['path']));monkeypatch.setattr(m.s.queue,'pin_path',lambda p:Path(p['path']))
    def mapped(producer,path,**kw):
        path=str(path)
        if path.endswith('/locks'):return locks
        if path.endswith('/session.json'):return session
        if path.endswith('/started.json'):return started/'started.json'
        if path=='/workspace/generation':return plan
        if path=='/workspace/candidate':return candidate
        return tmp_path/'empty'/path.removeprefix('/data/')
    monkeypatch.setattr(m.s,'mapped',mapped)
    monkeypatch.setattr(m.s.launcher,'validate_external_admission',lambda *args:{'binding':{'generation':'generation','candidate':'candidate'}})
    import speech_aware_evidence_acquisition.contracts as contracts
    monkeypatch.setattr(contracts,'ExecutionPlan',lambda **kwargs:SimpleNamespace(validate=lambda **kwargs:SimpleNamespace(run_id='RUN',attempt_id='gen')))
    terminal={'container_id':'f'*64,'state':{'ExitCode':1,'StartedAt':'2026-01-01T00:00:00+00:00','FinishedAt':'2026-01-01T00:01:00+00:00'}}
    monkeypatch.setattr(m.s,'producer_terminal',lambda *a:terminal);monkeypatch.setattr(m.s.transition,'quiescence',lambda instances:{'queues':{'0':'clear'}})
    monkeypatch.setattr(m.s,'stamp',lambda value: {'2026-01-01T00:00:00+00:00':2e10,'2026-01-01T00:00:10+00:00':2e10+10,'2026-01-01T00:01:00+00:00':2e10+60}[value])
    return cp,locks,lease,terminal


def test_complete_and_original_preserved(tmp_path,monkeypatch):
    cp,locks,lease,_=setup_case(tmp_path,monkeypatch);raw=(locks/'flight.lock').read_bytes();result=m.execute(cp)
    assert result['settled'] and not (locks/'flight.lock').exists()
    assert (tmp_path/'state/output/preserved-flight.lock.json').read_bytes()==raw
    assert m.execute(cp)==result


@pytest.mark.parametrize('fault',['owner','service','attempt','changed','terminal'])
def test_refuses_unsafe_state_without_removing_lease(tmp_path,monkeypatch,fault):
    cp,locks,lease,terminal=setup_case(tmp_path,monkeypatch);p=locks/'flight.lock'
    if fault=='owner':lease['pid']=os.getpid();p.write_text(json.dumps(lease))
    if fault=='service':monkeypatch.setattr(m.s.transition,'quiescence',lambda x: (_ for _ in ()).throw(ValueError('service unknown')))
    if fault=='attempt':
        f=tmp_path/'empty/attempts/RUN/gen.json';f.parent.mkdir(parents=True);f.touch()
    if fault=='changed':lease['run_id']='OTHER';p.write_text(json.dumps(lease))
    if fault=='terminal':terminal['state']['ExitCode']=0
    raw=p.read_bytes()
    with pytest.raises((ValueError, OSError)):m.execute(cp)
    assert p.read_bytes()==raw


def test_audit_write_failure_preserves_lease(tmp_path,monkeypatch):
    cp,locks,_,_=setup_case(tmp_path,monkeypatch);raw=(locks/'flight.lock').read_bytes()
    monkeypatch.setattr(m.s.queue,'atomic',lambda *args: (_ for _ in ()).throw(OSError('injected')))
    with pytest.raises(OSError,match='injected'):m.execute(cp)
    assert (locks/'flight.lock').read_bytes()==raw
    assert (tmp_path/'state/output/preserved-flight.lock.json').read_bytes()==raw
