"""Synchronous artifact seal recovery never fabricates historical release events."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_configured_terminal_settlement import configured, m, write
from speech_aware_evidence_acquisition.core.flight.receipts import FlightReceipts, _canonical_hash


def reseal(path, mutate):
    doc = json.loads(path.read_bytes()); mutate(doc)
    for value in doc['stages'].values():
        value.pop('stage_hash', None); value['stage_hash'] = _canonical_hash(value)
    doc.pop('receipts_hash', None); doc['receipts_hash'] = _canonical_hash(doc); write(path, doc)


@pytest.fixture
def finalized(configured):
    f = configured; f.lease.unlink()
    attempt = json.loads(f.attempt_path.read_bytes()); attempt['state'] = 'FINALIZED'; write(f.attempt_path, attempt)
    root = f.data / 'runs' / f.run / f.attempt
    logical = f.logical / 'runs' / f.run / f.attempt
    def saved(name, doc):
        path = write(root / name, doc)
        return {'path': str(logical / name), 'sha256': m.launcher.digest(path)}
    cost = {'calls_used':1,'audio_seconds_used':0.0,'prompt_tokens_total':5,'completion_tokens_total':3}
    contact = {'request_id':'req-1','sample_id':'sample-1','request_sha256':'a'*64,
               'response_sha256':'b'*64,'text':'reflection',
               'usage':{'calls':1,'prompt_tokens':5,'completion_tokens':3}}
    row = {k:v for k,v in contact.items() if k != 'usage'}
    row.update(result_kind='artifact',arm='prepared-context-text',admitted=True)
    output = root / (f.run+'.outputs.jsonl'); write(output, row)
    output_pin = {'path':str(logical/output.name),'sha256':m.launcher.digest(output)}
    candidate = saved('tfgrpo/learner/candidate.json', {'candidate':'unaccepted'})
    candidate = {'root':'data','relpath':str(Path(candidate['path']).relative_to(f.logical)),'sha256':candidate['sha256']}
    phases = saved(f.run+'.phases.json', {'run_id':f.run,'attempt_id':f.attempt,'result_kind':'artifact',
        'phases':[{'contact_receipts':[contact]}]})
    registered = json.loads(f.source_plan.read_bytes())['execution_plan']
    manifest = saved(f.run+'.run-manifest.json', {'schema':'saea-run-manifest-v1','result_kind':'artifact',
        'status':'COMPLETED','failure_reason':None,'study_commit':f.expected['study_commit'],
        'run_id':f.run,'attempt':{'run_id':f.run,'attempt_id':f.attempt},'arm':f.config['arm'],
        'scores':None,'scorers':[],'plan':registered,'split':{'count':1,'ids':['sample-1'],
        'identity_hash':registered['split_identity_hash']},'config':{'config_hash':'c'*64},
        'protocol_hash':registered['protocol_hash'],'cost':cost,'outputs':output_pin,
        'raw_trace':saved(f.run+'.trace.jsonl',{}),'trace_manifest':saved(f.run+'.trace-manifest.json',{})})
    response = f.journal._path('req-1','response')
    receipts = FlightReceipts(f.run, f.attempt)
    receipts.record('preflight', {'study_commit':f.expected['study_commit'],'config_hash':'c'*64})
    receipts.record('preprocess', {}); receipts.record('serve', {})
    receipts.record('run_phases', {'result_kind':'artifact','processed_dataset_rows':1,'total_dataset_rows':1,
        'outputs_row_count':1,'remaining_dataset_rows':0,'contact_count':1,'learning':{
        'contact_response_sha256':{response.name:m.launcher.digest(response)},'candidate':candidate,
        'usage':contact['usage']}})
    receipts.record('seal', {'finalize':{'attempt_state':'FINALIZED','result_kind':'artifact',
        'status':'not-applicable','scores_path':None},'manifest_path':manifest['path'],
        'manifest_sha256':manifest['sha256'],'outputs_path':output_pin['path'],'outputs_row_count':1,
        'cost':cost,'phases_path':phases['path'],'phases_sha256':phases['sha256']})
    receipts_path = write(root/'flight-receipts.json', receipts.as_json())
    return SimpleNamespace(**locals())


def test_finalized_artifact_is_reconciled_without_history_rewrite(finalized):
    s=finalized; f=s.f; original=f.attempt_path.read_bytes(); inode=(f.locks/'generation.guard').stat().st_ino
    result=m.execute(f.path)
    assert result['mode']=='finalized-artifact' and result['observed_contacts']==1
    assert result['scientific_success'] is False and result['new_model_calls']==0
    assert not (f.output/'generation-released.json').exists() and f.attempt_path.read_bytes()==original
    assert (f.locks/'generation.guard').stat().st_ino==inode
    assert m.execute(f.path)==result


@pytest.mark.parametrize('fault',['unfinalized','receipt-hash','receipt-head','manifest-hash',
    'incomplete','native-response','candidate-changed','phase-changed','output-changed','cost','wrong-seal-kind'])
def test_incomplete_or_changed_artifact_never_settles(finalized,fault):
    s=finalized;f=s.f
    if fault=='unfinalized':
        doc=json.loads(f.attempt_path.read_bytes());doc['state']='ABORTED';write(f.attempt_path,doc)
    elif fault=='receipt-hash':
        doc=json.loads(s.receipts_path.read_bytes());doc['receipts_hash']='0'*64;write(s.receipts_path,doc)
    elif fault=='receipt-head':reseal(s.receipts_path,lambda d:d['stages']['preflight'].update(study_commit='x'*40))
    elif fault=='manifest-hash':write(s.root/(f.run+'.run-manifest.json'),{})
    elif fault=='incomplete':reseal(s.receipts_path,lambda d:d['stages']['run_phases'].update(remaining_dataset_rows=1))
    elif fault=='native-response':reseal(s.receipts_path,lambda d:d['stages']['run_phases']['learning'].update(contact_response_sha256={}))
    elif fault=='candidate-changed':write(s.root/'tfgrpo/learner/candidate.json',{})
    elif fault=='phase-changed':write(s.root/(f.run+'.phases.json'),{})
    elif fault=='output-changed':write(s.output,{})
    elif fault=='cost':reseal(s.receipts_path,lambda d:d['stages']['seal']['cost'].update(calls_used=2))
    else:reseal(s.receipts_path,lambda d:d['stages']['seal']['finalize'].update(result_kind='task'))
    with pytest.raises((ValueError,RuntimeError)):m.execute(f.path)
    assert not (Path(f.config['state_dir'])/'complete.json').exists()
    assert not (f.output/'generation-released.json').exists()


def test_artifact_changes_during_quiescence_block_completion(finalized,monkeypatch):
    s=finalized;f=s.f
    def changed(instances):
        write(s.root/'tfgrpo/learner/candidate.json',{});return {}
    monkeypatch.setattr(m.transition,'quiescence',changed)
    with pytest.raises(ValueError,match='hash changed'):m.execute(f.path)
    assert not (Path(f.config['state_dir'])/'complete.json').exists()
