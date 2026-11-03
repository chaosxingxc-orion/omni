"""Finalized proposal releases must not invent task-scoring artifacts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


REPO = Path(__file__).resolve().parents[4]
helper = module('artifact_release_helper_test', REPO/'scripts/npu/generation_release_handoff.py')
producer = module('artifact_release_producer_test', Path(__file__).resolve().parents[2]/'scripts/analysis/run_real_engine_candidate.py')


def artifact_fixture(data, expected, *, logical=None):
    logical = data if logical is None else logical
    run, aid = expected['run_id'], expected['attempt_id']
    root = data/'runs'/run/aid; root.mkdir(parents=True, exist_ok=True)
    target = logical/'runs'/run/aid
    artifacts = {}; mapping = {}
    def write(key, path, where, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)+'\n')
        artifacts[key] = {'path':str(where), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        mapping[where] = path
    outputs = root/(run+'.outputs.jsonl'); outputs.write_text(
        ''.join(json.dumps({'sample_id':str(i), 'result_kind':'artifact'})+'\n' for i in range(expected['rows'])))
    artifacts['outputs'] = {'path':str(target/outputs.name), 'sha256':hashlib.sha256(outputs.read_bytes()).hexdigest()}
    mapping[target/outputs.name] = outputs
    attempt = logical/'attempts'/run/(aid+'.json')
    write('attempt', data/'attempts'/run/(aid+'.json'), attempt,
        {'state':'FINALIZED','run_id':run,'attempt_id':aid})
    write('manifest', root/(run+'.run-manifest.json'), target/(run+'.run-manifest.json'),
        {'schema':'saea-run-manifest-v1','result_kind':'artifact','status':'COMPLETED','run_id':run,
         'study_commit':expected['study_commit'],'plan':{'run_id':run,'attempt_id':aid},
         'attempt':{'state_path':str(attempt),'run_id':run,'attempt_id':aid},
         'scorers':[], 'scores':None,'outputs':artifacts['outputs']})
    write('phases', root/(run+'.phases.json'), target/(run+'.phases.json'), {'phases':[]})
    seal = {'manifest_path':artifacts['manifest']['path'],'manifest_sha256':artifacts['manifest']['sha256'],
        'outputs_path':artifacts['outputs']['path'],'outputs_row_count':expected['rows'],
        'phases_path':artifacts['phases']['path'],'phases_sha256':artifacts['phases']['sha256'],
        'finalize':{'status':'not-applicable','result_kind':'artifact','attempt_state':'FINALIZED','scores_path':None}}
    document = {'schema':helper.ARTIFACT_RELEASE_SCHEMA, **expected, 'generation_sealed':True,
        'generation_lease_released':True,'scoring_complete':False,'consumer_requires_own_lease_and_readiness':True,
        'released_at_unix':123.,'campaign_seconds_to_release':12.,'artifacts':artifacts,'seal_receipt':seal}
    return document, mapping


def expected():
    return dict(candidate='candidate',run_id='RUN',attempt_id='learning-v1',study_commit='a'*40,
        profile='profile',service_session='session',rows=2)


@pytest.mark.parametrize('fault',[None,'manifest','outputs','attempt','phases','missing','finalized',
    'score','rows','path','seal','task_schema','source_kind','attempt_state'])
def test_artifact_release_requires_complete_finalized_custody(tmp_path,fault):
    exp=expected();doc,mapping=artifact_fixture(tmp_path,exp)
    if fault in ('manifest','outputs','attempt','phases'):
        mapping[Path(doc['artifacts'][fault]['path'])].write_text('{}')
    elif fault=='missing':doc['artifacts'].pop('attempt')
    elif fault=='finalized':doc['seal_receipt']['finalize']['attempt_state']='OPENED'
    elif fault=='score':doc['scoring_complete']=True
    elif fault=='rows':doc['seal_receipt']['outputs_row_count']=1
    elif fault=='path':doc['artifacts']['manifest']['path']=str(tmp_path/'elsewhere.json')
    elif fault=='seal':doc['seal_receipt']['manifest_sha256']='b'*64
    elif fault=='task_schema':doc['schema']='saea-generation-released-v1'
    elif fault in ('source_kind','attempt_state'):
        key='manifest' if fault=='source_kind' else 'attempt'
        p=mapping[Path(doc['artifacts'][key]['path'])];value=json.loads(p.read_text())
        value['result_kind' if key=='manifest' else 'state']='task-answer' if key=='manifest' else 'OPENED'
        p.write_text(json.dumps(value));digest=hashlib.sha256(p.read_bytes()).hexdigest()
        doc['artifacts'][key]['sha256']=digest
        if key=='manifest':doc['seal_receipt']['manifest_sha256']=digest
    if fault:
        with pytest.raises(ValueError):helper.verify_release_document(doc,data=tmp_path,expected=exp)
    else:
        before=copy.deepcopy(doc)
        assert helper.verify_release_document(doc,data=tmp_path,expected=exp)==before
        assert not list(tmp_path.rglob('score-jobs.jsonl'))


def test_native_artifact_producer_uses_verified_scoreless_bundle(tmp_path,monkeypatch):
    from speech_aware_evidence_acquisition.core.flight.receipts import FlightReceipts
    from speech_aware_evidence_acquisition.core import bundle
    exp=expected();doc,_=artifact_fixture(tmp_path,exp)
    receipts=FlightReceipts(exp['run_id'],exp['attempt_id']);receipts.record('seal',doc['seal_receipt'])
    runtime=SimpleNamespace(run_id=exp['run_id'],attempt_id=exp['attempt_id'],receipts=receipts,_data_root=tmp_path)
    checked=[]
    def verify(path):
        checked.append(path);return SimpleNamespace(attempt_state='FINALIZED')
    monkeypatch.setattr(bundle,'verify_bundle',verify)
    result=producer.generation_release_document(runtime,candidate=exp['candidate'],
        spec={k:exp[k] for k in ('run_id','attempt_id','profile','service_session')},
        commit=exp['study_commit'],rows=exp['rows'],elapsed=12.)
    assert checked==[Path(doc['seal_receipt']['manifest_path'])]
    assert result['schema']==helper.ARTIFACT_RELEASE_SCHEMA
    assert result['seal_receipt']==receipts.require('seal')
    assert all(result['seal_receipt'][key]==value for key,value in doc['seal_receipt'].items())
    assert result['artifacts']==doc['artifacts']
    assert helper.verify_release_document(result,data=tmp_path,expected=exp)==result


def test_finalized_artifact_maps_disjoint_host_mounts_without_rewriting(tmp_path):
    exp=expected();doc,mapping=artifact_fixture(tmp_path,exp,logical=Path('/logical'))
    before=copy.deepcopy(doc)
    assert helper.verify_release_document(doc,data=Path('/logical'),expected=exp,resolve_artifact=mapping.__getitem__)==before
    assert doc==before


@pytest.mark.parametrize('fault',['commit','plan_run','plan_attempt','manifest_attempt_run',
    'manifest_attempt_id','attempt_run','attempt_id','duplicate_sample','empty_sample','missing_sample'])
def test_artifact_rejects_rehashed_cross_run_or_duplicate_rows(tmp_path,fault):
    exp=expected();doc,mapping=artifact_fixture(tmp_path,exp)
    key='outputs' if fault in ('duplicate_sample','empty_sample','missing_sample') else (
        'attempt' if fault in ('attempt_run','attempt_id') else 'manifest')
    path=mapping[Path(doc['artifacts'][key]['path'])]
    if key=='outputs':
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if fault=='duplicate_sample':rows[1]['sample_id']=rows[0]['sample_id']
        elif fault=='empty_sample':rows[0]['sample_id']=''
        else:rows[0].pop('sample_id')
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    else:
        value=json.loads(path.read_text())
        if fault=='commit':value['study_commit']='b'*40
        elif fault=='plan_run':value['plan']['run_id']='OTHER'
        elif fault=='plan_attempt':value['plan']['attempt_id']='other'
        elif fault=='manifest_attempt_run':value['attempt']['run_id']='OTHER'
        elif fault=='manifest_attempt_id':value['attempt']['attempt_id']='other'
        elif fault=='attempt_run':value['run_id']='OTHER'
        else:value['attempt_id']='other'
        path.write_text(json.dumps(value))
    digest=hashlib.sha256(path.read_bytes()).hexdigest();doc['artifacts'][key]['sha256']=digest
    if key=='manifest':doc['seal_receipt']['manifest_sha256']=digest
    elif key=='outputs':
        manifest_path=mapping[Path(doc['artifacts']['manifest']['path'])]
        manifest=json.loads(manifest_path.read_text());manifest['outputs']=doc['artifacts']['outputs']
        manifest_path.write_text(json.dumps(manifest));digest=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        doc['artifacts']['manifest']['sha256']=digest;doc['seal_receipt']['manifest_sha256']=digest
    with pytest.raises(ValueError):helper.verify_release_document(doc,data=tmp_path,expected=exp)
