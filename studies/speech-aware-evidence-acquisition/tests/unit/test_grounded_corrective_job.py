"""The optional strategy stays in the existing prepare/verify/register route."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]/'scripts/analysis'))
import grounded_fusion_job as job
from speech_aware_evidence_acquisition.rails import grounded_fusion_runtime as runtime
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical
from test_grounded_corrective_runtime import fixture


def prepared(tmp_path, monkeypatch):
    source, document, failed, _, _, write = fixture(tmp_path, monkeypatch)
    for name in ('load_finalized_shards', 'load_token_counter', 'build_fusion_input'):
        monkeypatch.setattr(job, name, getattr(runtime, name))
    monkeypatch.setattr(job, 'assert_execution_environment', lambda: None)
    monkeypatch.setattr(job.base, 'deployment_identity', lambda *a: {'model_fragment': 'model'})
    repo = tmp_path/'repo'; model = repo/'configs/model/model.json'; model.parent.mkdir(parents=True)
    model.write_bytes(canonical({'decoding_params': {'max_tokens': 16384}}))
    write(tmp_path/'spec.json', {})
    write(tmp_path/'failed-spec.json', failed)
    write(tmp_path/'identity.json', {'model_fragment': 'model'})
    write(tmp_path/'roster.json', {'items': [{'sample_id': g['query_id']} for g in source['batch']['groups']]})
    out = tmp_path/'work/codex/corrective-job'
    args = dict(source_spec=tmp_path/'spec.json', source_roster=tmp_path/'roster.json',
        tokenizer_relpath='tokenizer', runtime_identity=tmp_path/'identity.json', out=out,
        job_name='corrective-test', run_id='CORRECTIVE-TEST', service_session='resident',
        strategy=runtime.CORRECTIVE_STRATEGY, failed_fusion_spec=tmp_path/'failed-spec.json')
    result = job.prepare(repo, tmp_path, **args)
    return repo, out, result, args


def test_same_job_entry_prepares_and_reverifies_corrective_strategy(tmp_path, monkeypatch):
    repo, out, result, _ = prepared(tmp_path, monkeypatch)
    checked, reg, roster = job.verify_job(repo, tmp_path, out)
    assert result == checked and checked['generation_calls'] == 1 and checked['fit_calls'] == 1
    assert checked['historical_usage']['calls'] == 313 and checked['correction_round'] == 1
    assert reg['schema'] == runtime.FUSION_RUNTIME_SCHEMA
    assert reg['contacts'][0]['stage'] == runtime.STAGE
    assert job.final.verify_job(repo, tmp_path, out) == (checked, reg, roster)
    assert job.shards.verify_job(repo, tmp_path, out) == (checked, reg, roster)
    assert not checked['candidate_deployment_accepted']


@pytest.mark.parametrize('fault', ['cost', 'round', 'strategy', 'request', 'failed_attempt'])
def test_corrective_job_mutations_fail_existing_verifier(tmp_path, monkeypatch, fault):
    repo, out, _, _ = prepared(tmp_path, monkeypatch)
    complete = job.read(out/'complete.json')
    if fault == 'cost': complete['historical_usage']['calls'] -= 1
    elif fault == 'round': complete['correction_round'] = 2
    elif fault == 'strategy': complete['strategy'] = 'fusion-v1'
    elif fault == 'request':
        source = job.read(out/'source.json'); source['measurement']['input_tokens'] += 1
        (out/'source.json').write_bytes(job.raw(source))
        complete['artifacts']['source.json'] = job.file_sha(out/'source.json')
    else:
        source = job.read(out/'source.json')
        (tmp_path/source['failed_fusion']['attempt']['relpath']).write_text('{}')
    (out/'complete.json').write_bytes(job.raw(complete))
    with pytest.raises(ValueError): job.verify_job(repo, tmp_path, out)
