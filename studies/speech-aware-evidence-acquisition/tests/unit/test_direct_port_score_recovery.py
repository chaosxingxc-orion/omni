"""Recovery must preserve failed evidence and bind exactly the old generation."""
import json
from pathlib import Path
import sys
import pytest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/analysis'))
import direct_port_score_recovery as recovery
import direct_port_score_evidence as reader
from test_direct_port_score_evidence import fixture


def put(path,doc):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(doc))


def setup(tmp_path,monkeypatch):
    bundle,success,ids,result,summary=fixture(tmp_path)
    out=tmp_path/recovery.REL
    out.mkdir()
    (tmp_path/'score-batch').rename(out/'batch')
    score=success['stages']['score']
    score['output_dir']=str(out/'batch')
    original={**bundle.manifest,'scores':None}
    gen=tmp_path/'RUN.generation-manifest.json';put(gen,original)
    failure=tmp_path/'score-batch';failure.mkdir()
    stderr=failure/'attempt.stderr.log'
    stderr.write_text("ModuleNotFoundError: No module named 'progress'")
    failed={'job_count':1,'succeeded':0,'failed':1,'jobs':[{
        'operation':'finalize','status':'failed','returncode':1,'result_file':None,
        'complete_receipt_file':None,'manifest':str(bundle.manifest_path),
        'stderr_log':stderr.name,'stderr_sha256':recovery.sha(stderr)}]}
    put(failure/'batch-summary.json',failed)
    receipts={'run_id':'RUN','attempt_id':'attempt','stages':{
        'seal':{'manifest_path':str(bundle.manifest_path),'generation_manifest_path':str(gen)},
        'score':{'operation':'finalize','output_dir':str(failure),'result':failed}}}
    rp=tmp_path/'flight-receipts.json';put(rp,receipts)
    entry={'run_id':'RUN','attempt_id':'attempt','manifest':str(bundle.manifest_path),
        'manifest_sha256':recovery.sha(gen),
        'preserved_files':{str(p):recovery.sha(p) for p in (gen,rp,stderr,failure/'batch-summary.json')}}
    registration=tmp_path/'registration.json'
    put(registration,{'runs':{'lock':entry},'source_commit':'source'})
    monkeypatch.setattr(recovery,'require_registration',recovery.read)
    intent={'registration_sha256':recovery.sha(registration),'key':'lock','source_commit':'source'}
    put(out/'intent.json',intent)
    score['source_generation_manifest_sha256']=entry['manifest_sha256']
    receipt={'schema':'saea-direct-score-recovery-receipt-v1','recovery_id':recovery.RECOVERY_ID,
        'registration':str(registration),'registration_sha256':recovery.sha(registration),'key':'lock',
        'intent_sha256':recovery.sha(out/'intent.json'),'score':score}
    put(out/'recovery-receipt.json',receipt)
    return bundle,receipts,ids,receipt,registration


def test_saved_recovery_reuses_scores_without_replacing_failed_flight(tmp_path,monkeypatch):
    b,receipts,ids,_,_=setup(tmp_path,monkeypatch)
    before=(tmp_path/'flight-receipts.json').read_bytes()
    result=reader.verify_saved_score(b,receipts,tmp_path,ids)
    assert result['rows']==2 and result['new_scoring_jobs']==0
    assert str(tmp_path/recovery.RECEIPT) in result['inputs']
    assert before==(tmp_path/'flight-receipts.json').read_bytes()
    assert receipts['stages']['score']['result']['failed']==1


@pytest.mark.parametrize('case',['failed-evidence','registration','generation','intent','run-identity',
    'manifest-fields','score-directory','score-source','wrong-final-hash','score-result','successful-original'])
def test_recovery_refuses_source_or_receipt_drift(tmp_path,monkeypatch,case):
    b,receipts,ids,r,registration=setup(tmp_path,monkeypatch)
    if case=='failed-evidence':
        (tmp_path/'score-batch/attempt.stderr.log').write_text('changed')
    elif case=='registration': registration.write_text(registration.read_text()+' ')
    elif case=='generation':
        (tmp_path/'RUN.generation-manifest.json').write_text('{}')
    elif case=='intent': (tmp_path/recovery.REL/'intent.json').write_text('{}')
    elif case=='run-identity': receipts['run_id']='OTHER'
    elif case=='manifest-fields': b.manifest['status']='OTHER'
    elif case=='score-directory': r['score']['output_dir']=str(tmp_path/'score-batch')
    elif case=='score-source': r['score']['source_generation_manifest_sha256']='0'*64
    elif case=='wrong-final-hash': r['score']['finalized_manifest_sha256']='0'*64
    elif case=='score-result':
        result=tmp_path/recovery.REL/'batch/attempt.score.json'
        result.write_text(result.read_text()+' ')
    else:
        receipts['stages']['score']['result'].update(succeeded=1,failed=0)
    put(tmp_path/recovery.RECEIPT,r)
    with pytest.raises(ValueError):
        reader.verify_saved_score(b,receipts,tmp_path,ids)


def test_fail_without_recovery_does_not_hide_original_failure(tmp_path):
    b,receipts,ids,_,_=fixture(tmp_path)
    receipts['stages']['score']['result']['failed']=1
    with pytest.raises(ValueError,match='no separately registered recovery'):
        reader.verify_saved_score(b,receipts,tmp_path,ids)


def test_failed_subprocess_repr_preserves_exact_missing_dependency(tmp_path,monkeypatch):
    _,receipts,_,_,_=setup(tmp_path,monkeypatch)
    stderr=tmp_path/'score-batch/attempt.stderr.log'
    stderr.write_text("stderr='ModuleNotFoundError: No module named \\'progress\\'\\n'")
    receipts['stages']['score']['result']['jobs'][0]['stderr_sha256']=recovery.sha(stderr)
    put(tmp_path/'score-batch/batch-summary.json',receipts['stages']['score']['result'])
    assert recovery.require_failed_score(receipts,tmp_path)['operation']=='finalize'


@pytest.mark.parametrize('failed',[False,True])
def test_execute_intent_precedes_finalize_and_existing_intent_prevents_replay(tmp_path,monkeypatch,failed):
    tool=tmp_path/'tool';source=tmp_path/'source';data=tmp_path/'data'
    directory=data/'runs/RUN/attempt';directory.mkdir(parents=True)
    manifest=directory/'RUN.run-manifest.json';put(manifest,{'status':'COMPLETED'})
    put(directory/'flight-receipts.json',{'stages':{'seal':{}}})
    entry={'manifest':str(manifest),'preserved_files':{}}
    env=tmp_path/'smoke.json';put(env,{'official_command':['isolated-python']})
    doc={'implementation_sha256':recovery.sha(Path(recovery.__file__)),'source_repo':str(source),
         'environment_receipt':str(env),'environment_sha256':recovery.sha(env),'runs':{'lock':entry}}
    put(tool/recovery.REGISTRATION,doc)
    monkeypatch.setattr(recovery,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(recovery,'clean_head',lambda p:'head')
    monkeypatch.setattr(recovery,'require_registration',lambda p:doc)
    monkeypatch.setattr(recovery,'inspect_run',lambda *a:entry)
    monkeypatch.setattr(recovery,'read_flight_receipts',recovery.read)
    monkeypatch.setattr(recovery,'load_sealed_score_jobs',lambda seal:('job',))
    import speech_aware_evidence_acquisition.reproduction.slurp_official_eval as official
    monkeypatch.setattr(official,'official_evaluator_python',lambda:'isolated-python')
    calls=[]
    def batch(jobs,**kwargs):
        assert (directory/recovery.REL/'intent.json').is_file()
        assert kwargs['operation']=='finalize' and kwargs['include_per_sample'] is True
        assert kwargs['context'].repo_root==source
        calls.append(jobs)
        return {'failed':int(failed),'succeeded':int(not failed)}
    monkeypatch.setattr(recovery,'run_score_batch',batch)
    entry['manifest_sha256']=recovery.sha(manifest)
    if failed:
        with pytest.raises(ValueError,match='recovery scoring failed'):
            recovery.execute(tool,source,data,'lock')
        assert not (directory/recovery.RECEIPT).exists()
    else:
        assert recovery.execute(tool,source,data,'lock')['status']=='FINALIZED'
        assert (directory/recovery.RECEIPT).is_file()
    with pytest.raises(FileExistsError):
        recovery.execute(tool,source,data,'lock')
    assert len(calls)==1
