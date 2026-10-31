"""Build a fresh, source-bound Fold A agreement using the standard vote builder.

This is model-free preparation. It does not admit a flight or modify its source.
Partial outputs remain evidence and are never overwritten by this wrapper.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
import fold_a_current_comparator as comparator
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.rails.phase_supply import SEALED_AGREEMENT_SCHEMA
from speech_aware_evidence_acquisition.rails.kb_policy import EXEMPLAR_INTENT_AGREEMENT_DIMENSION

read, sha, raw = comparator.read, comparator.sha, comparator.raw
ARTIFACT = 'SAEA-SLURP-CURRENT-A0-FOLD-A-AGREEMENT-NPU-V1'
OUTPUT = 'prepared/' + ARTIFACT


def verify_artifact(data, receipt, ids):
    """Require a complete probability table and the standard builder's custody."""
    info = receipt['artifact']
    artifact = comparator.inputs.confined(data, info['relpath'])
    if (receipt['schema'] != 'saea-phase-agreement-receipt-v1'
            or receipt['artifact_id'] != ARTIFACT
            or receipt['contact_class'] != 'model-free'
            or set(receipt['contacts']) != {'model', 'server', 'encoder', 'embedder',
                                           'scorer', 'network', 'gpu', 'paid_api'}
            or any(value != 0 for value in receipt['contacts'].values())
            or info['relpath'] != OUTPUT + '/agreement.jsonl'
            or info['rows'] != len(ids) or sha(artifact) != info['sha256']):
        raise ValueError('agreement receipt or artifact identity differs')
    rows = [json.loads(line) for line in artifact.read_text().splitlines()]
    if len(rows) != len(ids) or {row['id'] for row in rows} != set(ids):
        raise ValueError('complete unique Fold A agreement rows required')
    for row in rows:
        scores, votes, excluded = row['scores'], row['vote_total'], row['self_excluded']
        if (row['schema'] != SEALED_AGREEMENT_SCHEMA
                or row['dimension'] != EXEMPLAR_INTENT_AGREEMENT_DIMENSION
                or type(votes) is not int or not 0 <= votes <= 20
                or type(excluded) is not int or excluded < 0 or votes + excluded > 20
                or not isinstance(scores, dict)):
            raise ValueError('invalid agreement row shape or vote count')
        for key, value in scores.items():
            if (not key.startswith('intent-boundary:') or type(value) not in (int, float)
                    or not math.isfinite(value) or not 0 < value <= 1
                    or abs(value * votes - round(value * votes)) > 1e-9):
                raise ValueError('invalid agreement probability')
        if ((votes == 0 and scores) or (votes > 0 and abs(sum(scores.values()) - 1) > 1e-9)):
            raise ValueError('agreement shares do not match vote total')
    origin = receipt['derived_from']
    if (origin['experiment'] != comparator.CELL or origin['cell'] != comparator.CELL
            or origin['vote_store'] != 'train-exemplars' or origin['vote_phase'] != 'solver'
            or origin['target_store'] != 'intent-boundaries'
            or origin['self_exclusion'] != 'sample-id utterance identity (first three fields)'):
        raise ValueError('agreement source recall or self-exclusion differs')
    return rows


def prepare(repo, data, lock, comparator_complete, out, source_commit, workers=8):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    repo, data = repo.resolve(), data.resolve()
    if (subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip()
            or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != source_commit):
        raise ValueError('committed clean comparator source required')
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('workers must be between 1 and 16')
    completion = comparator.inputs.confined(data, str(comparator_complete))
    derivation = read(completion)
    if (derivation['allowed_execution_stages'] != ['check'] or derivation['model_contacts'] != 0
            or derivation['claims_system_lock'] is not False):
        raise ValueError('never-flown comparator derivation required')
    for name, digest in derivation['output_hashes'].items():
        if sha(comparator.inputs.confined(repo, name)) != digest:
            raise ValueError('derived comparator configuration changed')
    required = {f'configs/{kind}/{comparator.CELL}.json' for kind in ('baseline', 'experiment')}
    if set(derivation['output_hashes']) != required:
        raise ValueError('exact comparator configuration pair required')
    cell, = read(repo / f'configs/experiment/{comparator.CELL}.json')['cells']
    if cell['allowed_execution_stages'] != ['check'] or cell['readiness']['state'] != 'held':
        raise ValueError('comparator must remain held and check-only')
    ids = [row['sample_id'] for row in read(repo / comparator.inputs.ROSTER)['items']]
    if len(ids) != 1080 or len(set(ids)) != 1080 or any(not x.startswith('slurp:train:') for x in ids):
        raise ValueError('complete unique TRAIN Fold A roster required')
    out = comparator.inputs.confined(data, str(out))
    target = comparator.inputs.confined(data, OUTPUT)
    if out.exists() or target.exists() or out.is_relative_to(repo) or target.is_relative_to(out) or out.is_relative_to(target):
        raise ValueError('fresh separate preparation and artifact directories required')
    builder = repo / 'scripts/build_slurp_intent_exemplar_agreement.py'
    pins = {'schema': 'saea-fold-a-agreement-preparation-intent-v1',
            'contact_class': 'model-free', 'registered_before_result': True,
            'rows': len(ids), 'source_commit': source_commit, 'comparator_completion_sha256': sha(completion),
            'builder_sha256': sha(builder), 'dataset_lock_sha256': sha(lock),
            'artifact_id': ARTIFACT, 'output': OUTPUT, 'workers': workers,
            'model_contacts': 0, 'claims_system_lock': False}
    out.mkdir(parents=True, exist_ok=False)
    publish_new(out / 'intent.json', raw(pins))
    try:
        # mkdir is the exclusive claim; the standard builder writes only here.
        target.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, str(builder), '--repo-root', str(repo), '--data-root', str(data),
                   '--runtime-root', str(data), '--lock-path', str(lock), '--experiment', comparator.CELL,
                   '--cell', comparator.CELL, '--vote-store', 'train-exemplars', '--vote-phase', 'solver',
                   '--target-store', 'intent-boundaries', '--target-facet-template', 'intent-boundary:{key}',
                   '--output', OUTPUT, '--artifact-id', ARTIFACT, '--workers', str(workers)]
        with (out / 'builder.log').open('xb') as log:
            subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
        if sha(completion) != pins['comparator_completion_sha256'] or sha(lock) != pins['dataset_lock_sha256']:
            raise ValueError('parent evidence changed during agreement preparation')
        for name, digest in derivation['output_hashes'].items():
            if sha(comparator.inputs.confined(repo, name)) != digest:
                raise ValueError('comparator changed during agreement preparation')
        receipt = read(target / 'receipt.json')
        verify_artifact(data, receipt, ids)
        if receipt['builder_sha256'] != pins['builder_sha256']:
            raise ValueError('agreement builder identity changed')
        result = {**pins, 'rows': len(ids), 'receipt_sha256': sha(target / 'receipt.json'),
                  'artifact_sha256': sha(target / 'agreement.jsonl'), 'status': 'complete',
                  'next_required': 'Register the current Fold A evaluation and prepare its admitted phase supply.'}
        publish_new(out / 'complete.json', raw(result))
        return result
    except BaseException as error:
        publish_new(out / 'failed.json', raw({'reason': str(error), 'automatic_retry': False}))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root', 'data-root', 'lock-path', 'comparator-complete', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    print(json.dumps(prepare(a.repo_root, a.data_root, a.lock_path, a.comparator_complete,
                             a.out, a.source_commit, a.workers), sort_keys=True))


if __name__ == '__main__': main()
