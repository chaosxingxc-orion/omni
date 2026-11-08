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


def test_not_sent_settlement_preserves_classification_and_original_charge(tmp_path,monkeypatch):
    args,blob=prepared(tmp_path,monkeypatch)
    from speech_aware_evidence_acquisition.core import contact_journal
    monkeypatch.setattr(contact_journal,'ContactJournal',lambda *a:SimpleNamespace(
        root=tmp_path/'native',inspect=lambda rid:{'state':'not_sent'}))
    result=m.settle(**args)
    assert result['observed_contacts']==0 and result['not_sent_contacts']==1
    rec=json.loads((args['out']/'reconciliation.json').read_bytes())
    assert rec['observed_requests']==[] and rec['not_sent_requests']==['request']
    assert rec['attempt']['actual']['calls']==1
    assert (args['out']/'preserved-flight.lock.json').read_bytes()==blob


def pre_attempt_world(tmp_path, monkeypatch):
    args, blob, session = configured_deployment(tmp_path, monkeypatch)
    (tmp_path/'attempts/r/a.json').unlink()
    (tmp_path/'native/r.intent.json').unlink()
    record=json.loads(blob); record.update(study_head='a'*40, started_at='2026-01-01T00:00:01+00:00')
    blob=json.dumps(record).encode(); (tmp_path/'locks/flight.lock').write_bytes(blob)
    plan=json.loads(args['plan'].read_bytes());plan['expected_session_id']='resident';args['plan'].write_text(json.dumps(plan))
    args['plan_sha256']=m.hashlib.sha256(args['plan'].read_bytes()).hexdigest()
    doc={'Id':'f'*64,'Config':{'Env':['SAEA_OBSERVABILITY_RUN_ID=r','SAEA_OBSERVABILITY_ATTEMPT_ID=a']},
         'State':{'Running':False,'Status':'exited','Pid':0,'StartedAt':'2026-01-01T00:00:00Z','FinishedAt':'2026-01-01T00:00:02Z'}}
    monkeypatch.setattr(m.subprocess,'check_output',lambda argv,**kw: b'error log' if argv[1]=='logs' else json.dumps([doc]).encode())
    import transition_serving_budget as transition
    monkeypatch.setattr(transition,'quiescence',lambda instances:{'quiet':True})
    args.update(container='f'*64, pre_attempt=True, lease_sha256=m.hashlib.sha256(blob).hexdigest(), expected_study_head='a'*40)
    return args,blob,doc


def test_pre_attempt_preserves_original_evidence(tmp_path, monkeypatch):
    args,blob,_=pre_attempt_world(tmp_path,monkeypatch)
    assert m.settle(**args)['new_model_calls']==0
    assert (args['out']/'preserved-flight.lock.json').read_bytes()==blob
    assert (args['out']/'container.log').read_bytes()==b'error log'
    assert not (tmp_path/'locks/flight.lock').exists()
    assert (tmp_path/'locks/generation.guard').exists()


@pytest.mark.parametrize('fault',['attempt','intent','run','lease','container','environment','time','multiple','busy'])
def test_pre_attempt_refuses_uncertain_custody(tmp_path,monkeypatch,fault):
    args,blob,doc=pre_attempt_world(tmp_path,monkeypatch)
    if fault=='attempt':(tmp_path/'attempts/r/a.json').write_text('{}')
    elif fault in ('intent','run'):
        path=tmp_path/('contact-journals/r/a' if fault=='intent' else 'runs/r/a');path.mkdir(parents=True)
        (path/'unknown').write_text('x')
    elif fault=='lease':args['lease_sha256']='0'*64
    elif fault=='container':doc['Id']='0'*64
    elif fault=='environment':doc['Config']['Env']=[]
    elif fault=='time':doc['State']['FinishedAt']='2025-01-01T00:00:00Z'
    elif fault=='multiple':
        receipt=json.loads(args['runtime_receipt'].read_bytes()); receipt['deployment']['instances'].append(receipt['deployment']['instances'][0]);args['runtime_receipt'].write_text(json.dumps(receipt))
        args['runtime_receipt_sha256']=m.hashlib.sha256(args['runtime_receipt'].read_bytes()).hexdigest()
        s=tmp_path/'serving/resident/session.json';session=json.loads(s.read_bytes());session['identity']['runtime_receipt_sha256']=args['runtime_receipt_sha256'];s.write_text(json.dumps(session))
    else:
        import transition_serving_budget as transition
        def busy(_):raise ValueError('busy')
        monkeypatch.setattr(transition,'quiescence',busy)
    with pytest.raises(ValueError):m.settle(**args)
    assert (tmp_path/'locks/flight.lock').read_bytes()==blob
