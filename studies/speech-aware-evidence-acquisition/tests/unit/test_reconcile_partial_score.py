import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/analysis/reconcile_partial_score.py'
spec = importlib.util.spec_from_file_location('score_reconcile', SCRIPT)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def fixture(tmp_path, monkeypatch):
    paths = {}
    def write(name, value):
        p = tmp_path / name
        p.write_text(json.dumps(value) if not isinstance(value, str) else value)
        paths[name] = p
        return p
    run, attempt = 'RUN', 'generation-v1'
    lock = write('attempt_lock', '')
    ids = ['a', 'b']
    gen = {'run_id':run, 'study_commit':'abc', 'attempt':{'run_id':run,'attempt_id':attempt,'state_path':str(tmp_path/'attempt')},'scores':None}
    write('generation_manifest', gen)
    scores = write('scores', {'slurp-slu-official-train':{'per_sample':[{'sample_id':x} for x in ids]}})
    manifest = dict(gen, scores={'path':str(scores), 'sha256':r.sha(scores)})
    write('run_manifest', manifest)
    write('attempt', {'state':'OPENED','run_id':run,'attempt_id':attempt,'actual':{'calls':2},'reserved':{'calls':5}})
    write('ledger', {'human':'ok'})
    write('ledger_json', {'human':'ok','derived':'yes'})
    write('ledger_md', 'row\n')
    for name in ('jobs','failed_summary','failed_stderr','flight_receipts'):
        write(name, '{}')
    stderr = write('recovery_stderr', 'handle.mark_finalized() Read-only file system ' + str(lock))
    write('recovery_batch', {'operation':'finalize','job_count':1,'failed':1,'succeeded':0,'jobs':[{'status':'failed','returncode':1,'result_file':None,'complete_receipt_file':None,'stderr_sha256':r.sha(stderr),'manifest':str(paths['run_manifest']),'ledger_cells':str(paths['ledger'])}]})
    for name in ('producer','recovery'):
        launch=write(name+'_launch', {'name':name})
        digest=hashlib.sha256(json.dumps(json.loads(launch.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
        write(name+'_launch_intent', {'config_sha256':digest})
        write(name+'_launch_result', {'returncode':0,'stdout':('a' if name=='producer' else 'b')*64,'automatic_retry':False})
    source = ('generation_manifest','jobs','ledger','failed_summary','failed_stderr','flight_receipts')
    pins = lambda: {k:{'path':str(v),'sha256':r.sha(v)} for k,v in paths.items()}
    inputs={k:pins()[k] for k in source}; inputs['manifest']=pins()['generation_manifest']
    write('recovery_intent', {'schema':'saea-sealed-overlay-score-recovery-intent-v1','rows':2,'new_model_calls':0,'inputs':inputs,'source_identities':{'generation_commit':'abc'}})
    config={'schema':'saea-partial-score-reconciliation-v1','producer_container':'a'*64,'recovery_container':'b'*64,'rows':2,'output_dir':str(tmp_path/'out'),'pins':pins(),'actual':{'calls':2},'reserved':{'calls':5}}
    cp=write('config',config)
    monkeypatch.setattr(r,'terminal',lambda cid,socket:{'id':cid,'exit_code':1,'finished_at':'time'})
    def verify(path):
        return SimpleNamespace(attempt_state=json.loads(paths['attempt'].read_text())['state'], outputs_rows=[1,2],split_ids=set(ids))
    monkeypatch.setattr(r,'verify_bundle',verify)
    monkeypatch.setattr(r,'ledger_row_for_bundle',lambda b,c:('row',dict(c,derived='yes')))
    class Handle:
        def __init__(self, **kwargs): pass
        def mark_finalized(self):
            d=json.loads(paths['attempt'].read_text());d['state']='FINALIZED';paths['attempt'].write_text(json.dumps(d))
    monkeypatch.setattr(r,'AttemptHandle',Handle)
    return cp, paths


def test_reconcile_and_reenter(tmp_path, monkeypatch):
    cp,p=fixture(tmp_path,monkeypatch)
    a=r.reconcile(cp); b=r.reconcile(cp)
    assert a==b and json.loads(p['attempt'].read_text())['state']=='FINALIZED'


def test_interrupted_after_finalization_can_reenter(tmp_path, monkeypatch):
    cp,p=fixture(tmp_path,monkeypatch)
    result=r.reconcile(cp)
    (tmp_path/'out/receipt.json').unlink()
    assert r.reconcile(cp)==result


@pytest.mark.parametrize('name,value', [('scores',{}),('ledger_json',{}),('recovery_stderr','other'),('attempt',{'state':'OPENED','actual':{'calls':3}})])
def test_changed_inputs_refuse(tmp_path, monkeypatch, name, value):
    cp,p=fixture(tmp_path,monkeypatch)
    p[name].write_text(json.dumps(value) if isinstance(value,dict) else value)
    with pytest.raises((ValueError,KeyError)):
        r.reconcile(cp)
    assert not (tmp_path/'out/intent.json').exists()


def test_running_container_refused(tmp_path,monkeypatch):
    cp,p=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(r,'terminal',lambda *args: (_ for _ in ()).throw(ValueError('not terminal')))
    with pytest.raises(ValueError,match='not terminal'):r.reconcile(cp)
    assert not (tmp_path/'out/intent.json').exists()
