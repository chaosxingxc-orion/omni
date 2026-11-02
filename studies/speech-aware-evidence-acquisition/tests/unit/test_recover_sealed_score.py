"""A failed scorer can be recovered without regenerating or hiding failure."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/analysis'))
import recover_sealed_score as recovery


def setup(tmp_path, monkeypatch):
    repo = tmp_path / 'tree/studies/study'; repo.mkdir(parents=True)
    run = tmp_path / 'data/runs/R/attempt'; run.mkdir(parents=True)
    manifest = run / 'R.run-manifest.json'; manifest.write_text('{}')
    generation = run / 'R.generation-manifest.json'; generation.write_text('{}')
    original = run / 'flight-receipts.json'; original.write_text('failed receipt')
    env = tmp_path / 'environment.json'; python = tmp_path / 'python'; python.write_text('binary')
    env.write_text(json.dumps({'status': 'PASS', 'dataset_rows': 0, 'official_command': [str(python)]}))
    entry = {'preserved_files': {str(original): recovery.sha(original)}}
    monkeypatch.setattr(recovery, 'assert_execution_environment', lambda: None)
    monkeypatch.setattr(recovery, 'official_evaluator_python', lambda: str(python))
    monkeypatch.setattr(recovery.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(recovery, 'inspect_source', lambda *a: (('job',), entry))
    monkeypatch.setattr(recovery, 'verify_bundle', lambda *a: SimpleNamespace(attempt_state='FINALIZED', manifest={}))
    monkeypatch.setattr(recovery, 'read_flight_receipts', lambda *a: {'stages': {'seal': {'generation_manifest_path': str(generation)}}})
    return dict(repo=repo, data=tmp_path/'data', manifest=manifest, out=run/'score-recovery', expected_rows=313, environment_receipt=env), original


def test_intent_before_scoring_preserves_failure_and_prevents_replay(tmp_path, monkeypatch):
    args, original = setup(tmp_path, monkeypatch); calls = []
    def score(jobs, **kwargs):
        assert (args['out']/'intent.json').is_file()
        assert kwargs['operation'] == 'finalize' and kwargs['include_per_sample'] is True
        assert kwargs['context'].repo_root == args['repo']
        calls.append(jobs)
        return {'failed': 0, 'succeeded': 1}
    monkeypatch.setattr(recovery, 'run_score_batch', score)
    assert recovery.recover(**args)['new_model_calls'] == 0
    assert original.read_text() == 'failed receipt'
    with pytest.raises(ValueError, match='fresh recovery'):
        recovery.recover(**args)
    assert len(calls) == 1


@pytest.mark.parametrize('case', ['wrong-python', 'smoke-failed', 'dependency-missing', 'failed-batch', 'source-drift', 'not-finalized', 'generation-drift'])
def test_bad_environment_or_mutation_cannot_publish_success(tmp_path, monkeypatch, case):
    args, original = setup(tmp_path, monkeypatch); calls = []
    def score(*a, **k):
        calls.append(1)
        if case == 'source-drift': original.write_text('changed')
        return {'failed': int(case == 'failed-batch'), 'succeeded': int(case != 'failed-batch')}
    monkeypatch.setattr(recovery, 'run_score_batch', score)
    if case in ['wrong-python', 'smoke-failed']:
        env = json.loads(args['environment_receipt'].read_text())
        if case == 'wrong-python': env['official_command'] = ['other']
        else: env['status'] = 'FAIL'
        args['environment_receipt'].write_text(json.dumps(env))
    if case == 'dependency-missing':
        monkeypatch.setattr(recovery.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stderr='missing'))
    if case in ['not-finalized', 'generation-drift']:
        monkeypatch.setattr(recovery, 'verify_bundle', lambda *a: SimpleNamespace(attempt_state='OPENED' if case == 'not-finalized' else 'FINALIZED', manifest={'changed': True} if case == 'generation-drift' else {}))
    with pytest.raises(ValueError): recovery.recover(**args)
    assert not (args['out']/'recovery-receipt.json').exists()
    if case in ['wrong-python', 'smoke-failed', 'dependency-missing']: assert not calls


def test_inspect_rejects_nonpositive_row_count(tmp_path):
    for n in [0, -1, True]:
        with pytest.raises(ValueError, match='positive'):
            recovery.inspect_source(tmp_path, tmp_path/'missing', n)
