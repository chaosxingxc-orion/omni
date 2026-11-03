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


def test_final_report_allowance_preserved_for_plan_validation(tmp_path,monkeypatch):
    args,blob=prepared(tmp_path,monkeypatch)
    from speech_aware_evidence_acquisition import contracts
    allowance={'path':'docs/receipts/owner/test.json','sha256':'a'*64}
    doc=json.loads(args['plan'].read_bytes());doc['owner_allowance']=allowance
    args['plan'].write_text(json.dumps(doc));args['plan_sha256']=m.hashlib.sha256(args['plan'].read_bytes()).hexdigest()
    def plan(**kwargs):
        assert kwargs['owner_allowance']==allowance
        return SimpleNamespace(**kwargs)
    monkeypatch.setattr(contracts,'ExecutionPlan',plan)
    assert m.settle(**args)['observed_contacts']==1


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


def configured_deployment(tmp_path, monkeypatch):
    args, blob = prepared(tmp_path, monkeypatch)
    record = json.loads(blob); record['session_id'] = 'resident'
    blob = json.dumps(record).encode(); (tmp_path/'locks/flight.lock').write_bytes(blob)
    receipt = tmp_path/'runtime.json'
    receipt.write_text(json.dumps({'deployment': {'instances': [{'instance_id': 'tp2', 'host': '127.0.0.1', 'port': 8203}]}}))
    digest = m.hashlib.sha256(receipt.read_bytes()).hexdigest()
    session = tmp_path/'serving/resident/session.json'; session.parent.mkdir(parents=True)
    session.write_text(json.dumps({'session_id': 'resident', 'identity': {'runtime_receipt_sha256': digest},
                                   'deployment_instance_ids': ['tp2']}))
    args.update(runtime_receipt=receipt, runtime_receipt_sha256=digest)
    return args, blob, session


def test_configured_settlement_uses_only_lease_bound_deployment(tmp_path, monkeypatch):
    args, blob, session = configured_deployment(tmp_path, monkeypatch)
    sys.path.insert(0, str(ROOT/'scripts/environment'))
    import transition_serving_budget as transition
    seen = []
    monkeypatch.setattr(transition, 'quiescence', lambda instances: seen.extend(instances) or {'quiet': True})
    assert m.settle(**args)['observed_contacts'] == 1
    assert [i['port'] for i in seen] == [8203]
    assert (args['out']/'preserved-flight.lock.json').read_bytes() == blob


@pytest.mark.parametrize('fault', ['receipt', 'session', 'membership', 'busy'])
def test_configured_settlement_preserves_unproven_partition(tmp_path, monkeypatch, fault):
    args, blob, session = configured_deployment(tmp_path, monkeypatch)
    sys.path.insert(0, str(ROOT/'scripts/environment'))
    import transition_serving_budget as transition
    def busy(instances):raise ValueError('busy deployment')
    monkeypatch.setattr(transition, 'quiescence', busy)
    if fault == 'receipt':args['runtime_receipt_sha256'] = '0'*64
    elif fault in ('session', 'membership'):
        doc = json.loads(session.read_bytes())
        if fault == 'session':doc['identity']['runtime_receipt_sha256'] = '0'*64
        else:doc['deployment_instance_ids'] = ['different']
        session.write_text(json.dumps(doc))
    with pytest.raises(ValueError):m.settle(**args)
    assert (tmp_path/'locks/flight.lock').read_bytes() == blob
