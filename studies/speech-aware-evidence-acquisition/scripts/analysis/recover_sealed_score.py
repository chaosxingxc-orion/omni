"""Recover an environment-failed sealed score without any inference entrypoint.

The failed flight receipt and generation files remain immutable. A new durable
intent binds the corrected evaluator and a separate successful score receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from direct_port_score_recovery import clean_head, preserved, publish, read, require_failed_score, sha
from speech_aware_evidence_acquisition.contracts import assert_execution_environment
from speech_aware_evidence_acquisition.core.bundle import verify_bundle
from speech_aware_evidence_acquisition.core.flight.receipts import read_flight_receipts
from speech_aware_evidence_acquisition.core.flight.scoring import load_sealed_score_jobs
from speech_aware_evidence_acquisition.core.scoring_batch import ScoringContext, run_score_batch
from speech_aware_evidence_acquisition.reproduction.slurp_official_eval import official_evaluator_python


def inspect_source(repo, manifest, expected_rows):
    if type(expected_rows) is not int or expected_rows <= 0:
        raise ValueError('positive expected row count required')
    bundle = verify_bundle(manifest)
    directory = manifest.parent
    receipts = read_flight_receipts(directory / 'flight-receipts.json')
    require_failed_score(receipts, directory)
    jobs = load_sealed_score_jobs(receipts['stages']['seal'])
    if (bundle.attempt_state != 'OPENED' or bundle.manifest.get('status') != 'COMPLETED'
            or bundle.manifest.get('scores') is not None or len(jobs) != 1
            or jobs[0].manifest.resolve() != manifest.resolve()
            or len(bundle.outputs_rows) != expected_rows
            or len(bundle.split_ids) != expected_rows
            or len(set(bundle.split_ids)) != expected_rows
            or bundle.manifest['study_commit'] != clean_head(repo)
            or receipts['run_id'] != bundle.manifest['run_id']
            or receipts['attempt_id'] != jobs[0].job_id):
        raise ValueError('complete original unscored sealed bundle required')
    if list(directory.glob('*.scores.json')):
        raise ValueError('partial scoring artifacts require reconciliation')
    files = [p for p in directory.iterdir()
             if p.is_file() and p != manifest and not p.name.startswith('.')]
    files += [p for p in (directory / 'score-batch').iterdir() if p.is_file()]
    return jobs, {'manifest': str(manifest.resolve()), 'manifest_sha256': sha(manifest),
                  'source_commit': bundle.manifest['study_commit'], 'rows': expected_rows,
                  'run_id': receipts['run_id'], 'attempt_id': receipts['attempt_id'],
                  'preserved_files': {str(p): sha(p) for p in sorted(files)}}


def recover(*, repo, data, manifest, out, expected_rows, environment_receipt):
    assert_execution_environment()
    repo, data, manifest, out = (Path(p).resolve() for p in (repo, data, manifest, out))
    if out.parent != manifest.parent or out.exists():
        raise ValueError('fresh recovery directory beside original manifest required')
    smoke = read(environment_receipt)
    python = official_evaluator_python()
    if (smoke.get('status') != 'PASS' or smoke.get('dataset_rows') != 0
            or smoke.get('official_command', [None])[0] != python):
        raise ValueError('corrected evaluator must match successful synthetic smoke')
    probe = subprocess.run([python, '-c', 'import progress, jiwer, tabulate, textdistance, numpy'],
                           capture_output=True, text=True, timeout=60)
    if probe.returncode:
        raise ValueError('corrected evaluator dependencies unavailable: ' + probe.stderr[-2000:])
    jobs, entry = inspect_source(repo, manifest, expected_rows)
    out.mkdir()
    intent = {'schema': 'saea-sealed-score-recovery-intent-v1', 'source': entry,
              'implementation_sha256': sha(Path(__file__)), 'evaluator_python': python,
              'evaluator_python_sha256': sha(Path(python).resolve()),
              'environment_receipt': str(environment_receipt),
              'environment_sha256': sha(environment_receipt), 'new_model_calls': 0}
    publish(out / 'intent.json', intent)
    start = time.monotonic()
    summary = run_score_batch(jobs, output_dir=out / 'batch', max_workers=1,
                              operation='finalize', include_per_sample=True,
                              context=ScoringContext(repo, data, repo.parents[1] / 'docs/datasets.lock.json'))
    if summary.get('failed') or summary.get('succeeded') != 1:
        raise ValueError('recovery scoring failed; preserve intent and reconcile, never replay')
    preserved(entry)
    bundle = verify_bundle(manifest)
    if bundle.attempt_state != 'FINALIZED':
        raise ValueError('score worker did not finalize original attempt')
    original = read(Path(read_flight_receipts(manifest.parent / 'flight-receipts.json')
                         ['stages']['seal']['generation_manifest_path']))
    if {k: v for k, v in bundle.manifest.items() if k != 'scores'} != {k: v for k, v in original.items() if k != 'scores'}:
        raise ValueError('finalization changed generation fields')
    receipt = {'schema': 'saea-sealed-score-recovery-v1', 'source': entry,
               'intent_sha256': sha(out / 'intent.json'),
               'finalized_manifest_sha256': sha(manifest), 'result': summary,
               'duration_seconds': time.monotonic() - start, 'new_model_calls': 0}
    publish(out / 'recovery-receipt.json', receipt)
    return {'status': 'FINALIZED', 'rows': expected_rows, 'new_model_calls': 0,
            'receipt': str(out / 'recovery-receipt.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'data', 'manifest', 'out', 'environment-receipt'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--expected-rows', type=int, required=True)
    print(json.dumps(recover(**vars(parser.parse_args()))))
