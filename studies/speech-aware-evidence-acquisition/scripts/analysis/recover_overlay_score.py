"""Finalize one sealed overlay run with independently verified scoring source."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.bundle import verify_bundle
from speech_aware_evidence_acquisition.core.flight.scoring import load_sealed_score_jobs
from speech_aware_evidence_acquisition.core.flight.receipts import read_flight_receipts
from speech_aware_evidence_acquisition.core.scoring_batch import ScoringContext, run_score_batch
from speech_aware_evidence_acquisition.core.scoring_source import verify_separate_scoring_tool
from speech_aware_evidence_acquisition.reproduction.slurp_official_eval import official_evaluator_python
from speech_aware_evidence_acquisition.contracts import assert_execution_environment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, obj):
    publish_new(path, (json.dumps(obj, indent=2, sort_keys=True) + '\n').encode())


def recover(args):
    assert_execution_environment()
    source, data, manifest, generation, out = (Path(x).resolve() for x in
        (args.source_repo, args.data, args.manifest, args.generation_manifest, args.out))
    if out.parent != manifest.parent or out.exists():
        raise ValueError('fresh recovery directory beside original manifest required')
    identity = verify_separate_scoring_tool(source, REPO)
    if sha(manifest) != args.manifest_sha256 or sha(generation) != args.manifest_sha256:
        raise ValueError('original generation and run manifest pins differ')
    if sha(args.jobs) != args.jobs_sha256 or sha(args.ledger) != args.ledger_sha256:
        raise ValueError('original score job or ledger pin differs')
    if sha(args.failed_summary) != args.failed_summary_sha256 or sha(args.failed_stderr) != args.failed_stderr_sha256:
        raise ValueError('original failed scoring evidence pin differs')
    failure = json.loads(Path(args.failed_summary).read_bytes())
    stderr = Path(args.failed_stderr).read_text()
    if (failure.get('failed') != 1 or failure.get('succeeded') != 0
            or failure.get('job_count') != 1 or failure.get('operation') != 'finalize'
            or 'delivery config differs from recorded run' not in stderr):
        raise ValueError('expected a terminal delivery config scoring failure')
    evaluator = official_evaluator_python()
    if sha(evaluator) != args.official_python_sha256:
        raise ValueError('official interpreter pin differs')
    bundle = verify_bundle(manifest)
    receipts = read_flight_receipts(manifest.parent / 'flight-receipts.json')
    jobs = load_sealed_score_jobs(receipts['stages']['seal'])
    score_stage = receipts['stages']['score']
    if (score_stage.get('result') != failure or score_stage.get('operation') != 'finalize'
            or Path(score_stage['output_dir']).resolve() != Path(args.failed_summary).resolve().parent):
        raise ValueError('failed score evidence differs from original flight receipt')
    failed_job, = failure['jobs']
    if (failed_job.get('status') != 'failed' or failed_job.get('returncode') == 0
            or failed_job.get('result_file') is not None or failed_job.get('complete_receipt_file') is not None
            or failed_job.get('stderr_sha256') != args.failed_stderr_sha256
            or Path(failed_job['manifest']).resolve() != manifest
            or Path(args.failed_stderr).resolve() != Path(args.failed_summary).resolve().parent / failed_job['stderr_log']):
        raise ValueError('failed score job evidence differs')
    if (bundle.attempt_state != 'OPENED' or bundle.manifest.get('status') != 'COMPLETED'
            or bundle.manifest.get('scores') is not None or len(jobs) != 1
            or jobs[0].manifest.resolve() != manifest or jobs[0].ledger_cells.resolve() != Path(args.ledger).resolve()
            or len(bundle.outputs_rows) != args.rows or len(bundle.split_ids) != args.rows
            or bundle.manifest['study_commit'] != identity['generation_commit']
            or list(manifest.parent.glob('*.scores.json'))):
        raise ValueError('complete original unscored sealed bundle required')
    # Refuse a different job table even if the receipt resolves to a valid job.
    from speech_aware_evidence_acquisition.core.scoring_batch import load_score_jobs
    if load_score_jobs(Path(args.jobs), require_ledger_cells=True) != jobs:
        raise ValueError('sealed score job table differs from receipt')
    out.mkdir()
    pins = {name: {'path': str(Path(getattr(args, name)).resolve()),
                   'sha256': sha(getattr(args, name))}
            for name in ('manifest', 'generation_manifest', 'jobs', 'ledger', 'failed_summary', 'failed_stderr')}
    pins['flight_receipts'] = {'path':str(manifest.parent / 'flight-receipts.json'), 'sha256':sha(manifest.parent / 'flight-receipts.json')}
    intent = {'schema': 'saea-sealed-overlay-score-recovery-intent-v1',
              'source_identities': identity, 'inputs': pins, 'rows': args.rows,
              'runtime_config': {'path': os.environ['SAEA_RUNTIME_CONFIG'],
                                 'sha256': os.environ['SAEA_RUNTIME_CONFIG_SHA256']},
              'workspace_manifest_sha256': os.environ['SAEA_EXPERIMENT_WORKSPACE_SHA256'],
              'assets_manifest_sha256': os.environ['SAEA_EXPERIMENT_MANIFEST_SHA256'],
              'official_python': {'path': evaluator, 'sha256': sha(evaluator)}, 'new_model_calls': 0}
    publish(out / 'intent.json', intent)
    start = time.monotonic()
    summary = run_score_batch(jobs, output_dir=out / 'batch', max_workers=1,
        operation='finalize', include_per_sample=True,
        context=ScoringContext(source, data, Path(args.lock), tool_repo_root=REPO))
    if summary.get('failed') or summary.get('succeeded') != 1:
        raise ValueError('scoring failed; preserve intent and batch for reconciliation')
    after = verify_bundle(manifest)
    if after.attempt_state != 'FINALIZED' or len(after.outputs_rows) != args.rows:
        raise ValueError('finalized complete bundle required')
    old = json.loads(generation.read_bytes())
    if {k:v for k,v in after.manifest.items() if k != 'scores'} != {k:v for k,v in old.items() if k != 'scores'}:
        raise ValueError('generation fields changed during scoring')
    for name in ('generation_manifest', 'jobs', 'ledger', 'failed_summary', 'failed_stderr', 'flight_receipts'):
        if sha(pins[name]['path']) != pins[name]['sha256']:
            raise ValueError('preserved source changed: ' + name)
    receipt = {'schema':'saea-sealed-overlay-score-recovery-v1', 'intent_sha256':sha(out/'intent.json'),
               'finalized_manifest_sha256':sha(manifest), 'result':summary,
               'duration_seconds':time.monotonic()-start, 'new_model_calls':0}
    publish(out / 'recovery-receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-repo','data','manifest','generation-manifest','manifest-sha256',
                 'jobs','jobs-sha256','ledger','ledger-sha256','failed-summary','failed-summary-sha256',
                 'failed-stderr','failed-stderr-sha256','official-python-sha256','lock','out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--rows', type=int, required=True)
    print(json.dumps(recover(parser.parse_args()), sort_keys=True))
