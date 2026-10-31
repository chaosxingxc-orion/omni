"""Verify a Direct campaign's complete saved official score without rescoring."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts/checks'))
from speech_aware_evidence_acquisition.core.bundle.verify import verify_bundle
from speech_aware_evidence_acquisition.core.flight.receipts import read_flight_receipts
from speech_aware_evidence_acquisition.core.scoring_receipt import SavedScoreReceiptRef, verify_complete_score_receipt
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
import flight_registration_check as registration

SCORER = 'slurp-slu-official-train'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def within(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('score evidence escapes its registered run')
    return path


def complete_scores(scores, ids, scorers):
    if not isinstance(scores, dict) or set(scores) != set(scorers) or SCORER not in scores:
        raise ValueError('score set differs from the complete official run')
    for name, result in scores.items():
        if not isinstance(result, dict) or not isinstance(result.get('aggregate'), dict) or not isinstance(result.get('per_sample'), list):
            raise ValueError('saved score lacks aggregate/per-sample evidence')
        if name == 'cost':
            continue
        rows = result['per_sample']
        observed = [row.get('sample_id') for row in rows]
        if len(observed) != len(ids) or set(observed) != set(ids):
            raise ValueError('saved score does not cover the entire roster exactly once')
    if scores[SCORER]['aggregate'].get('identity_check', {}).get('passed') is not True:
        raise ValueError('official scorer identity check did not pass')
    return scores


def verify_saved_score(bundle, receipts, run_dir, expected_ids):
    """Handle the shared campaign's finalize path and the existing score path."""
    manifest = bundle.manifest
    if bundle.attempt_state != 'FINALIZED' or manifest.get('status') != 'COMPLETED':
        raise ValueError('official result requires a finalized complete bundle')
    if len(expected_ids) != len(set(expected_ids)) or set(bundle.split_ids) != set(expected_ids):
        raise ValueError('executed roster differs from the registered complete roster')
    score = receipts['stages'].get('score')
    recovery_inputs = {}
    if score and score.get('result', {}).get('failed'):
        from direct_port_score_recovery import resolve
        score, recovery_inputs = resolve(bundle, receipts, run_dir)
    if not score or score.get('job_count') != 1 or score.get('include_per_sample') is not True:
        raise ValueError('exactly one complete scoring job is required')
    output = within(score['output_dir'], run_dir)
    summary_path = output / 'batch-summary.json'
    summary = json.loads(summary_path.read_bytes())
    if summary != score['result'] or (summary['job_count'], summary['succeeded'], summary['failed']) != (1, 1, 0):
        raise ValueError('score batch and flight receipt disagree or scoring failed')
    job, = summary['jobs']
    if job['status'] != 'succeeded' or job['returncode'] != 0 or Path(job['manifest']).resolve() != bundle.manifest_path.resolve():
        raise ValueError('score job does not match the complete run')
    result_path = within(output / job['result_file'], output)
    if sha(result_path) != job['result_sha256']:
        raise ValueError('score result hash mismatch')
    payload = json.loads(result_path.read_bytes())
    if payload.get('verified') is not True or Path(payload['manifest']).resolve() != bundle.manifest_path.resolve():
        raise ValueError('score payload has a different source')
    if job['operation'] == 'finalize':
        if score.get('operation') != 'finalize' or payload.get('attempt_state') != 'FINALIZED':
            raise ValueError('finalize operation identity differs')
        if score['finalized_manifest_sha256'] != sha(bundle.manifest_path):
            raise ValueError('finalized manifest differs from scoring receipt')
        scores_path = within(payload['scores'], run_dir)
        if (Path(manifest['scores']['path']).resolve() != scores_path or
                sha(scores_path) != manifest['scores']['sha256']):
            raise ValueError('sealed score artifact differs from finalizer result')
        scores = json.loads(scores_path.read_bytes())
        if payload.get('score_details') != scores:
            raise ValueError('finalizer per-sample details differ from saved scores')
    elif job['operation'] == 'score':
        reference = SavedScoreReceiptRef(job['job_id'], within(output / job['complete_receipt_file'], output), job['complete_receipt_sha256'])
        verify_complete_score_receipt(reference, bundle)
        scores = payload['scores']
    else:
        raise ValueError('unrecognized scoring operation')
    complete_scores(scores, expected_ids, manifest['scorers'])
    return {'aggregate': scores[SCORER]['aggregate'], 'rows': len(expected_ids),
            'scorer': SCORER, 'official_scoring_jobs': 1, 'new_scoring_jobs': 0,
            'inputs': {**recovery_inputs, **{str(p): sha(p) for p in (bundle.manifest_path, summary_path, result_path)}},
            'operation': job['operation']}


def require_quality_read(repo, read_id, recipe_path, experiment):
    recipe_path = within(recipe_path, repo)
    ledger = registration.parse_ledger(repo, Path('docs/exposure-ledger.md'))
    rows = ledger.get(read_id, [])
    if len(rows) != 1 or not registration.is_countersigned(rows[0]):
        raise ValueError('quality READ must be uniquely registered and countersigned')
    row = rows[0]
    if (row.get('artifacts') != str(recipe_path.relative_to(repo)) or
            sha(recipe_path) not in row.get('notes', '')):
        raise ValueError('quality READ must pin this exact recipe')
    recipe = json.loads(recipe_path.read_bytes())
    if (recipe.get('schema') != 'saea-direct-port-complete-score-read-v1' or
            recipe.get('read_run_id') != read_id or recipe.get('quality_read') is not True or
            recipe.get('new_scoring_jobs') != 0 or
            experiment not in recipe.get('experiments', [])):
        raise ValueError('registered recipe does not permit this complete quality read')
    return recipe


def build(repo, data, experiment, read_id, recipe_path):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', experiment):
        raise ValueError('invalid experiment identity')
    recipe = require_quality_read(repo, read_id, recipe_path, experiment)
    document = json.loads((repo / 'configs/experiment' / (experiment + '.json')).read_bytes())
    cell, = document['cells']
    base = json.loads((repo / 'configs/baseline' / (cell['baseline_fragment'] + '.json')).read_bytes())
    if base.get('policy') != 'phase-direct-only-v1' or document['family_budget']['official_complete_scoring_jobs'] != 1:
        raise ValueError('not a complete Direct scored campaign')
    run_dir = data / 'runs' / cell['run_id'] / cell['attempt_id']
    bundle = verify_bundle(run_dir / (cell['run_id'] + '.run-manifest.json'))
    receipts_path = run_dir / 'flight-receipts.json'
    receipts = read_flight_receipts(receipts_path)
    if (receipts['run_id'], receipts['attempt_id']) != (cell['run_id'], cell['attempt_id']):
        raise ValueError('flight receipt identity mismatch')
    preflight = receipts['stages']['preflight']
    if (bundle.manifest['run_id'] != cell['run_id'] or
            bundle.manifest['study_commit'] != preflight['study_commit'] or
            bundle.manifest['config']['config_hash'] != preflight['config_hash'] or
            bundle.manifest['protocol_hash'] != preflight['protocol_hash']):
        raise ValueError('executed bundle and preflight identities disagree')
    declared = recipe['identities'][experiment]
    if declared != {
        'run_id': cell['run_id'], 'attempt_id': cell['attempt_id'],
        'rows': document['population']['count'],
        'split_identity_hash': document['population']['split_identity_hash'],
    }:
        raise ValueError('quality recipe identity differs from registered campaign')
    ids = document['populations'][0]['ids']
    if len(ids) != document['population']['count'] or len(ids) < 400:
        raise ValueError('full development roster required; no smoke quality read')
    result = verify_saved_score(bundle, receipts, run_dir, ids)
    result.update(schema='saea-direct-port-complete-score-evidence-v1', run_id=cell['run_id'],
                  attempt_id=cell['attempt_id'], read_run_id=read_id, claims_benefit=False)
    result['inputs'][str(receipts_path)] = sha(receipts_path)
    result['inputs'][str(recipe_path)] = sha(recipe_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=REPO)
    parser.add_argument('--data-root', type=Path, required=True)
    for name in ('experiment', 'read-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.repo_root, args.data_root, args.experiment, args.read_id, args.recipe)
    publish_new(args.out, (json.dumps(result, indent=2, sort_keys=True) + '\n').encode())
    print(json.dumps({'written': str(args.out), 'rows': result['rows'], 'new_scoring_jobs': 0}))


if __name__ == '__main__':
    main()
