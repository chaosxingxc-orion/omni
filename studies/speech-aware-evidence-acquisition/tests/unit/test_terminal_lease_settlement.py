import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
ROOT=Path(__file__).parents[2]
sys.path[:0]=[str(ROOT/'scripts/analysis')]
spec=importlib.util.spec_from_file_location('settle_terminal',ROOT/'scripts/environment/settle_terminal_lease.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def prepared(tmp_path,monkeypatch):
    import tfgrpo_recovery_queue as recovery
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core import contact_journal
    locks=tmp_path/'locks';locks.mkdir();(locks/'generation.guard').touch()
    lease={'run_id':'r','attempt_id':'a','pid':999999999}
    blob=json.dumps(lease).encode();(locks/'flight.lock').write_bytes(blob)
    ap=tmp_path/'attempts/r/a.json';ap.parent.mkdir(parents=True)
    ap.write_text(json.dumps({'state':'ABORTED','actual':{'calls':1}}))
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'execution_plan':{'run_id':'r','attempt_id':'a'}}))
    native=tmp_path/'native';native.mkdir();(native/'r.intent.json').write_text('{"request_id":"request"}')
    monkeypatch.setattr(contracts,'ExecutionPlan',lambda **k:SimpleNamespace(**k))
    monkeypatch.setattr(contact_journal,'ContactJournal',lambda *a:SimpleNamespace(root=native,inspect=lambda rid:{'state':'observed'}))
    monkeypatch.setattr(recovery,'quiescence',lambda ports:{'quiet':True})
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a:json.dumps([{'Id':'exact','State':{'Running':False,'Status':'exited','Pid':0}}]).encode())
    args={'repo':ROOT,'data':tmp_path,'out':tmp_path/'work/codex/reconcile','container':'owned',
          'expected_run':'r','expected_attempt':'a','expected_pid':999999999,'plan':plan,'plan_sha256':m.hashlib.sha256(plan.read_bytes()).hexdigest()}
    return args,blob


def test_observed_terminal_lease_preserved_before_release(tmp_path,monkeypatch):
    args,blob=prepared(tmp_path,monkeypatch)
    assert m.settle(**args)['observed_contacts']==1
    assert not (tmp_path/'locks/flight.lock').exists()
    assert (args['out']/'preserved-flight.lock.json').read_bytes()==blob
    with pytest.raises(ValueError):m.settle(**args)


@pytest.mark.parametrize('fault',['foreign','unknown','active','pin','busy'])
def test_never_releases_unproven_lease(tmp_path,monkeypatch,fault):
    args,blob=prepared(tmp_path,monkeypatch)
    from speech_aware_evidence_acquisition.core import contact_journal
    import tfgrpo_recovery_queue as recovery
    if fault=='foreign':args['expected_run']='other'
    elif fault=='unknown':
        monkeypatch.setattr(contact_journal,'ContactJournal',lambda *a:SimpleNamespace(root=tmp_path/'native',inspect=lambda rid:{'state':'unknown'}))
    elif fault=='active':
        monkeypatch.setattr(m.subprocess,'check_output',lambda *a:json.dumps([{'Id':'exact','State':{'Running':True,'Status':'running','Pid':10}}]).encode())
    elif fault=='pin':args['plan_sha256']='0'*64
    else:
        def busy(ports):raise ValueError('requests in progress')
        monkeypatch.setattr(recovery,'quiescence',busy)
    with pytest.raises(ValueError):m.settle(**args)
    assert (tmp_path/'locks/flight.lock').read_bytes()==blob
