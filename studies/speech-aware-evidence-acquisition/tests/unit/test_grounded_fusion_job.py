import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/analysis'))
import grounded_fusion_job as job
from test_grounded_fusion_runtime import setup
from speech_aware_evidence_acquisition.rails import grounded_fusion_runtime as runtime_module
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical


def prepared(tmp_path,monkeypatch):
    runtime,_,_,source=setup(tmp_path,monkeypatch)
    repo=tmp_path/'repo';model=repo/'configs/model/frozen-model.json';model.parent.mkdir(parents=True)
    model.write_text(json.dumps({'decoding_params':{'max_tokens':100,'temperature':0}}))
    sid=runtime._samples[0].sample_id
    for name,doc in {'spec.json':source['source'],'roster.json':{'items':[{'sample_id':sid}]},
                     'identity.json':runtime.session_required_identity()}.items():
        (tmp_path/name).write_bytes(canonical(doc))
    monkeypatch.setattr(job,'assert_execution_environment',lambda:None)
    for name in ('load_finalized_shards','load_token_counter','build_fusion_input','fusion_messages'):
        monkeypatch.setattr(job,name,getattr(runtime_module,name))
    monkeypatch.setattr(job.base,'deployment_identity',lambda *a:runtime.session_required_identity())
    out=tmp_path/'work/codex/fusion-job'
    result=job.prepare(repo,tmp_path,source_spec=tmp_path/'spec.json',source_roster=tmp_path/'roster.json',
        tokenizer_relpath='tokenizer',runtime_identity=tmp_path/'identity.json',out=out,
        job_name='fusion-test',run_id='FUSION-TEST',service_session='resident')
    return repo,tmp_path,out,result


def test_finite_job_derives_registration_and_dispatches_verify(tmp_path,monkeypatch):
    repo,data,out,result=prepared(tmp_path,monkeypatch)
    verified,reg,roster=job.verify_job(repo,data,out)
    assert result==verified and verified['generation_calls']==1
    assert verified['semantic_review_required'] and not verified['candidate_deployment_accepted']
    assert reg['contacts'][0]['stage']=='grounded_final_update'
    assert job.final.verify_job(repo,data,out)==(verified,reg,roster)
    assert job.shards.verify_job(repo,data,out)==(verified,reg,roster)


@pytest.mark.parametrize('fault',['cost','calls','groups','roster','source','identity'])
def test_rejects_changed_job_derivation(tmp_path,monkeypatch,fault):
    repo,data,out,_=prepared(tmp_path,monkeypatch)
    complete=job.read(out/'complete.json')
    if fault=='cost':complete['historical_custody_sha256']='a'*64
    elif fault=='calls':complete['generation_calls']=2
    elif fault=='groups':complete['groups']=2
    elif fault=='identity':complete['runtime_identity_sha256']='a'*64
    elif fault=='source':(data/'roster.json').write_text('{}')
    else:
        roster=job.read(out/'construction-roster.json');roster['count']=2
        (out/'construction-roster.json').write_bytes(job.raw(roster))
        complete['artifacts']['construction-roster.json']=job.file_sha(out/'construction-roster.json')
    (out/'complete.json').write_bytes(job.raw(complete))
    with pytest.raises(ValueError):job.verify_job(repo,data,out)
