import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

p = Path(__file__).resolve().parents[2] / 'scripts/analysis/slurp_paired_roster_read.py'
spec = importlib.util.spec_from_file_location('paired_receipt_test_module', p)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)

@pytest.mark.parametrize('fault', [None, 'receipt_manifest', 'intent', 'job', 'result', 'payload'])
def test_synchronous_receipt_validates_real_finalizer_chain(tmp_path, fault):
    root=tmp_path/'score-finalize-recovery-v1';(root/'batch').mkdir(parents=True)
    manifest=tmp_path/'r.run-manifest.json';manifest.write_text('{}')
    score=tmp_path/'r.scores.json';score.write_text('{}')
    generation=tmp_path/'r.generation-manifest.json';generation.write_text('{}')
    def write(p,d):p.write_text(json.dumps(d))
    intent={'schema':'saea-seal-only-finalize-recovery-intent-v1','inputs':{'generation':{'sha256':module.digest(generation)}}}
    write(root/'intent.json',intent)
    payload={'verified':True,'attempt_state':'FINALIZED','manifest':str(manifest),'scores':str(score)}
    if fault=='payload': payload['verified']=False
    file=root/'batch'/'result.json';write(file,payload)
    job={'job_id':'a','operation':'finalize','status':'succeeded','returncode':0,'manifest':str(manifest),'result_file':'result.json','result_sha256':module.digest(file)}
    if fault=='job':job['job_id']='other'
    if fault=='result':job['result_sha256']='0'*64
    receipt={'schema':'saea-seal-only-finalize-recovery-v1','finalized_manifest_sha256':module.digest(manifest),'intent_sha256':module.digest(root/'intent.json'),'result':{'schema':'saea-parallel-score-batch-v1','operation':'finalize','job_count':1,'succeeded':1,'failed':0,'jobs':[job]}}
    if fault=='receipt_manifest':receipt['finalized_manifest_sha256']='0'*64
    if fault=='intent':receipt['intent_sha256']='0'*64
    write(root/'recovery-receipt.json',receipt)
    arm=SimpleNamespace(run_dir=tmp_path,manifest_sha256=module.digest(manifest),generation_manifest_sha256=module.digest(generation),manifest_path=manifest,scores_path=score,spec=SimpleNamespace(label='r',attempt_id='a'),score_jobs=[])
    if fault:
        with pytest.raises(module.StopStageError):module._check_score_job(arm)
    else:module._check_score_job(arm)

def test_successor_cost_scope_preserves_totals_and_unknown_rates():
    arm=SimpleNamespace(rows=1080,generation_manifest={'cost_provenance':{'kind':'predecessor-attempts'},
        'cost':{'calls_used':3196,'prompt_tokens_total':42568396,'completion_tokens_total':4445366}},
        flight_receipts={'stages':{'run_phases':{ 'wall_seconds':None}}})
    v=module._arm_efficiency(arm)
    assert v['cost_scope']=='predecessor-attempts' and v['contacts']==3196
    assert v['prompt_tokens_total']==42568396 and v['completion_tokens_total']==4445366
    assert v['contacts_per_minute'] is None
    arm.generation_manifest['cost']['calls_used']=None
    assert module._arm_efficiency(arm)['contacts_per_row'] is None
