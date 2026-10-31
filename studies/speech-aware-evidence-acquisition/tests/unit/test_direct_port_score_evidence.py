"""Consume the complete shared campaign score without another scorer contact."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/analysis'))
import direct_port_score_evidence as read


def fixture(tmp_path):
    ids = ['s1', 's2']
    scores = {read.SCORER: {'aggregate': {'identity_check': {'passed': True}, 'slu_f1': 0.5},
                           'per_sample': [{'sample_id': i} for i in ids]}}
    score_file = tmp_path / 'RUN.scores.json'; score_file.write_text(json.dumps(scores))
    manifest = {'status': 'COMPLETED', 'scorers': [read.SCORER],
                'scores': {'path': str(score_file), 'sha256': read.sha(score_file)}}
    mp = tmp_path / 'RUN.run-manifest.json'; mp.write_text(json.dumps(manifest))
    bundle = SimpleNamespace(manifest=manifest, manifest_path=mp, split_ids=set(ids), attempt_state='FINALIZED')
    out = tmp_path / 'score-batch'; out.mkdir()
    payload = {'manifest': str(mp), 'verified': True, 'scores': str(score_file),
               'score_details': scores, 'attempt_state': 'FINALIZED'}
    result = out / 'attempt.score.json'; result.write_text(json.dumps(payload))
    job = {'operation': 'finalize', 'manifest': str(mp), 'status': 'succeeded', 'returncode': 0,
           'job_id': 'attempt', 'result_file': result.name, 'result_sha256': read.sha(result)}
    summary = {'job_count': 1, 'succeeded': 1, 'failed': 0, 'jobs': [job]}
    summary_file = out / 'batch-summary.json'; summary_file.write_text(json.dumps(summary))
    receipt = {'stages': {'score': {'job_count': 1, 'include_per_sample': True,
               'output_dir': str(out), 'result': summary, 'operation': 'finalize',
               'finalized_manifest_sha256': read.sha(mp)}}}
    return bundle, receipt, ids, result, summary_file


def test_shared_finalize_result_is_consumed_without_scorer(tmp_path):
    bundle, receipt, ids, _, _ = fixture(tmp_path)
    result = read.verify_saved_score(bundle, receipt, tmp_path, ids)
    assert result['rows'] == 2 and result['aggregate']['slu_f1'] == 0.5
    assert result['new_scoring_jobs'] == 0 and result['operation'] == 'finalize'


@pytest.mark.parametrize('case', ['incomplete', 'wrong-roster', 'tampered-result', 'tampered-manifest',
                                  'failed-batch', 'missing-stage', 'missing-details', 'duplicate-rows', 'escaped-result'])
def test_refuses_invalid_complete_score_evidence(tmp_path, case):
    bundle, receipt, ids, result, summary_file = fixture(tmp_path)
    if case == 'incomplete': bundle.attempt_state = 'OPENED'
    elif case == 'wrong-roster': ids = ['s1', 'foreign']
    elif case == 'tampered-result': result.write_text(result.read_text() + ' ')
    elif case == 'tampered-manifest': bundle.manifest_path.write_text(bundle.manifest_path.read_text() + ' ')
    elif case == 'failed-batch': receipt['stages']['score']['result']['failed'] = 1
    elif case == 'missing-stage': receipt['stages'] = {}
    elif case in ('missing-details', 'duplicate-rows'):
        payload = json.loads(result.read_bytes())
        if case == 'missing-details': payload.pop('score_details')
        else: payload['score_details'][read.SCORER]['per_sample'][1]['sample_id'] = 's1'
        result.write_text(json.dumps(payload))
        receipt['stages']['score']['result']['jobs'][0]['result_sha256'] = read.sha(result)
        summary_file.write_text(json.dumps(receipt['stages']['score']['result']))
    else:
        receipt['stages']['score']['result']['jobs'][0]['result_file'] = '../../foreign'
        summary_file.write_text(json.dumps(receipt['stages']['score']['result']))
    with pytest.raises(ValueError): read.verify_saved_score(bundle, receipt, tmp_path, ids)


@pytest.mark.parametrize('ids', [['s1'], ['s1', 's1'], ['s1', 'foreign']])
def test_score_coverage_is_whole_roster(ids):
    scores = {read.SCORER: {'aggregate': {'identity_check': {'passed': True}},
                           'per_sample': [{'sample_id': i} for i in ids]}}
    with pytest.raises(ValueError, match='entire roster'):
        read.complete_scores(scores, ['s1', 's2'], [read.SCORER])


@pytest.mark.parametrize('case', ['valid','structural-only','wrong-experiment','wrong-pin','duplicate-read'])
def test_read_scope_is_bound_to_quality_recipe(tmp_path, case):
    recipe = {'schema':'saea-direct-port-complete-score-read-v1', 'read_run_id':'READ',
              'quality_read':case != 'structural-only', 'new_scoring_jobs':0,
              'experiments':['EXP' if case != 'wrong-experiment' else 'OTHER']}
    path = tmp_path / 'docs/recipe.json';path.parent.mkdir();path.write_text(json.dumps(recipe))
    digest = read.sha(path) if case != 'wrong-pin' else '0'*64
    line = f'| today | READ | docs/recipe.json | root countersigned; Plan SHA256 {digest} |\n'
    (tmp_path/'docs/exposure-ledger.md').write_text('| date | run/experiment id | artifacts | notes |\n| --- | --- | --- | --- |\n' + line * (2 if case == 'duplicate-read' else 1))
    if case == 'valid':
        assert read.require_quality_read(tmp_path,'READ',path,'EXP')['quality_read']
    else:
        with pytest.raises(ValueError):read.require_quality_read(tmp_path,'READ',path,'EXP')


def test_saved_score_operation_uses_independent_complete_receipt(tmp_path):
    from speech_aware_evidence_acquisition.core.bundle.artifacts import VerifiedBundle
    from speech_aware_evidence_acquisition.core.scoring_receipt import capture_score_source, write_complete_score_receipt
    simple, receipt, ids, result, summary_file = fixture(tmp_path)
    outputs = tmp_path / 'RUN.outputs.jsonl'; outputs.write_text(''.join(json.dumps({'sample_id':i})+'\n' for i in ids))
    simple.manifest.update(run_id='RUN', config={'config_hash':'a'*64},
                          split={'identity_hash':'b'*64,'count':2,'ids':ids},
                          outputs={'path':str(outputs),'sha256':read.sha(outputs)})
    simple.manifest_path.write_text(json.dumps(simple.manifest))
    bundle = VerifiedBundle(simple.manifest_path, simple.manifest, outputs,
                            tuple({'sample_id':i} for i in ids), {}, frozenset(ids), 'FINALIZED')
    old = json.loads(result.read_bytes())
    payload = {'manifest':str(bundle.manifest_path),'verified':True,'scores':old['score_details']}
    result.write_text(json.dumps(payload))
    rp=result.with_name('attempt.complete-score-receipt.json')
    saved=write_complete_score_receipt(rp,job_id='attempt',result_path=result,
                                       source=capture_score_source(bundle),verified_bundle=bundle)
    score=receipt['stages']['score'];score.pop('operation');score.pop('finalized_manifest_sha256')
    job=score['result']['jobs'][0]
    job.update(operation='score',result_sha256=read.sha(result),complete_receipt_file=rp.name,complete_receipt_sha256=saved.sha256)
    summary_file.write_text(json.dumps(score['result']))
    assert read.verify_saved_score(bundle,receipt,tmp_path,ids)['operation']=='score'
