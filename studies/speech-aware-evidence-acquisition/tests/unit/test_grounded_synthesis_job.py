import copy
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/analysis'))
import grounded_synthesis_job as job
from test_grounded_synthesis_runtime import setup
from speech_aware_evidence_acquisition.rails import grounded_synthesis_runtime as runtime_module
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical


def prepared(tmp_path,monkeypatch):
    runtime,_,_,source=setup(tmp_path,monkeypatch)
    repo=tmp_path/'repo';data=tmp_path
    model=repo/'configs/model/frozen-model.json';model.parent.mkdir(parents=True)
    model.write_text(json.dumps({'decoding_params':{'max_tokens':100,'temperature':0}}))
    sid=runtime._samples[0].sample_id
    documents={'plan.json':source['plan'],'spec.json':source['source'],
        'roster.json':{'items':[{'sample_id':sid,'stratum':'fixed'}]},
        'identity.json':runtime.session_required_identity()}
    for name,doc in documents.items():(data/name).write_bytes(canonical(doc))
    monkeypatch.setattr(job,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(job,'load_source',runtime_module.load_source)
    monkeypatch.setattr(job,'load_token_counter',runtime_module.load_token_counter)
    monkeypatch.setattr(job.base,'deployment_identity',lambda *a:runtime.session_required_identity())
    out=data/'work/codex/grounded-job'
    result=job.prepare(repo,data,shard_plan=data/'plan.json',source_spec=data/'spec.json',
        source_roster=data/'roster.json',tokenizer_relpath='tokenizer',
        runtime_identity=data/'identity.json',out=out,job_name='grounded-test',
        run_id='GROUNDED-TEST',service_session='resident')
    return repo,data,out,result


def test_prepare_derives_finite_registered_artifact(tmp_path,monkeypatch):
    repo,data,out,result=prepared(tmp_path,monkeypatch)
    verified,reg,roster=job.verify_job(repo,data,out)
    assert result==verified and verified['generation_calls']==1
    assert verified['candidate_publishable'] is False
    assert reg['contacts'][0]['sample_id']==roster['items'][0]['sample_id']
    assert job.final.verify_job(repo,data,out)==(verified,reg,roster)


@pytest.mark.parametrize('fault',['cost','groups','roster','manifest','input'])
def test_rejects_self_consistent_job_drift(tmp_path,monkeypatch,fault):
    repo,data,out,_=prepared(tmp_path,monkeypatch)
    complete=job.read(out/'complete.json')
    if fault=='cost':complete['historical_custody_sha256']='a'*64
    elif fault=='groups':complete['groups']=2
    elif fault=='roster':
        roster=job.read(out/'construction-roster.json');roster['items'][0]['stratum']='changed'
        (out/'construction-roster.json').write_bytes(job.raw(roster))
        complete['artifacts']['construction-roster.json']=job.file_sha(out/'construction-roster.json')
    elif fault=='manifest':complete['generation_calls']=2
    else:(data/'roster.json').write_text('{}')
    (out/'complete.json').write_bytes(job.raw(complete))
    with pytest.raises(ValueError):job.verify_job(repo,data,out)


def test_run_publishes_release_before_final_result(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    from speech_aware_evidence_acquisition.core.flight import campaign, factory
    from speech_aware_evidence_acquisition.core.flight.receipts import FlightReceipts
    import hashlib
    repo, data = tmp_path / 'tree/study', tmp_path
    out = data / 'work/codex/current'
    receipts = FlightReceipts('R', 'learning-v1')
    seal = {}
    for key in ('generation_manifest', 'score_ledger_cells', 'score_jobs'):
        p = tmp_path / key
        p.write_bytes(b'sealed')
        seal[key + '_path'] = str(p)
        seal[key + '_sha256'] = hashlib.sha256(b'sealed').hexdigest()
    receipts.record('seal', seal)
    flight = SimpleNamespace(run_id='R', attempt_id='learning-v1', receipts=receipts)
    cell = SimpleNamespace(document={'dataset_fragment': 'fixed'})
    monkeypatch.setattr(job, 'assert_execution_environment', lambda: None)
    monkeypatch.setattr(git_commit, 'require_clean_study_repo', lambda p: 'commit')
    monkeypatch.setattr(job, 'verify_job', lambda *a: ({'job': 'job', 'run_id': 'R', 'runtime_rows': 3,
                        'service_session': 'resident'}, {'model_id': 'model'}, {}))
    monkeypatch.setattr(job.base, 'require_learning_deployment', lambda *a: None)
    monkeypatch.setattr(job.base, 'verify_release', lambda *a, **k: None)
    monkeypatch.setattr(campaign, 'load_campaign_cells', lambda *a: [cell])
    monkeypatch.setattr(campaign, 'declared_readiness', lambda *a: SimpleNamespace(state='ready'))
    monkeypatch.setattr(factory, 'build_flight_runtime', lambda *a, **k: flight)
    monkeypatch.setenv('SAEA_DEPLOYMENT_PROFILE', 'profile')
    def execute(runtimes, **kwargs):
        kwargs['on_generation_released'](flight)
        event = job.read(out / 'generation-released.json')
        assert event['generation_lease_released'] and not event['scoring_complete']
        assert event['rows'] == 3 and event['run_id'] == 'R'
        return {'pending': [], 'completed': ['R'], 'states': {'R': {'state': 'scored'}}}
    monkeypatch.setattr(campaign, 'run_campaign', execute)
    predecessor = tmp_path / 'predecessor.json'
    predecessor.write_text('{}')
    assert job.run(repo, data, tmp_path / 'job', out, predecessor, 'PREVIOUS')['completed']
    assert (out / 'complete.json').exists()
