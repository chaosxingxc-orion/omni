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
