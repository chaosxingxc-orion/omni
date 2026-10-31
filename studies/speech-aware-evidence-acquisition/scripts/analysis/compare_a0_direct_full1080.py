"""Verify paired full-roster Direct/A0 evidence without new inference or scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/analysis')]
from direct_port_score_evidence import verify_saved_score, SCORER

DIRECT_PINS = {
    'lock': '919d602f104f0052b3e84a49372bf324b424263e456e61d2a2128b0e83f4270e',
    'aa': 'fb1f59ed3e22f56fb803d7465b2969e8deeee64cf3e5430ad5ffd18cb56a5ae9',
    'ab': 'a6e39d6b37c60d56b3d0b65f66771f212a159240448023dfd59dc699739ba298',
}
RECIPE_SHA = 'efb43a80731ccc01dfb0fa038664687b188f416d56063e23628ae29937e0db38'
PARENT_SHA = '1d30c501612686004936b5dd295885d9452175e6449b4a66187c08227a322cea'


def read(p): return json.loads(p.read_bytes())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def confined(data, path):
    path = (data / path).resolve()
    if not path.is_relative_to(data.resolve()): raise ValueError('source escapes data root')
    return path


def score_matrix(result, ids):
    import numpy as np
    rows = result['per_sample']
    keys = [r['sample_id'] for r in rows]
    aggregate = result['aggregate']
    if (len(keys) != len(ids) or len(set(keys)) != len(keys) or set(keys) != set(ids)
            or aggregate['samples'] != len(ids) or not aggregate['identity_check']['passed']):
        raise ValueError('complete paired roster and official identity required')
    by_id = {r['sample_id']: r for r in rows}
    matrix = np.array([[sum(by_id[s][f'slu_f1_{d}_{c}'] for d in ('word', 'char'))
                        for c in ('tp', 'fp', 'fn')] for s in ids], dtype=float)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError('invalid sufficient statistics')
    tp, fp, fn = matrix.sum(axis=0)
    micro = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    if abs(micro - aggregate['official']['slu_f1']) > 1e-6:
        raise ValueError('micro counts differ from official score')
    return matrix


def score_identity(result):
    a = result['aggregate']
    return {k: a[k] for k in ('evaluator_commit', 'scorer_version', 'split_id',
                              'projection', 'filler_normalization')}


def paired(results, ids, replicates=10000):
    import numpy as np
    if set(results) != {'direct_lock', 'direct_aa', 'direct_ab', 'a0'}:
        raise ValueError('retain all three Direct references and A0')
    if any(score_identity(r) != score_identity(results['a0']) for r in results.values()):
        raise ValueError('official scorer or normalization differs')
    matrices = {n: score_matrix(r, ids) for n, r in results.items()}
    groups = sorted({s.split(':audio-')[0] for s in ids})
    lookup = {s: i for i, s in enumerate(groups)}
    indices = np.array([lookup[s.split(':audio-')[0]] for s in ids])
    weights = np.random.default_rng(20261031).multinomial(len(groups),
        np.full(len(groups), 1 / len(groups)), size=replicates)
    draws = {}
    for name, matrix in matrices.items():
        cluster = np.zeros((len(groups), 3)); np.add.at(cluster, indices, matrix)
        totals = weights @ cluster
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        draws[name] = np.divide(2 * totals[:, 0], denominator,
            out=np.zeros(replicates), where=denominator != 0)
    comparisons = {}
    for name in ('direct_lock', 'direct_aa', 'direct_ab'):
        comparisons['a0_minus_' + name] = {
            'slu_f1_delta_pp': 100 * (results['a0']['aggregate']['official']['slu_f1']
                - results[name]['aggregate']['official']['slu_f1']),
            'paired_percentile_95_interval_pp': np.quantile(100 * (draws['a0'] - draws[name]), [.025, .975]).tolist(),
            'intent_delta_pp': 100 * (results['a0']['aggregate']['official']['intent_accuracy']
                - results[name]['aggregate']['official']['intent_accuracy'])}
    return {'comparisons': comparisons, 'utterance_clusters': len(groups),
            'bootstrap_replicates': replicates, 'seed': 20261031}


def load_direct(data, ids):
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.flight.receipts import read_flight_receipts
    results, pins = {}, {}
    for name, digest in DIRECT_PINS.items():
        ep = data / 'work/codex/agent-after-recovered-direct-v1/direct-completion' / (name + '-score-evidence.json')
        if sha(ep) != digest: raise ValueError('Direct read evidence changed')
        evidence = read(ep)
        for path, expected in evidence['inputs'].items():
            if sha(confined(data, path)) != expected: raise ValueError('Direct source pin changed')
        mp, = [confined(data, p) for p in evidence['inputs'] if p.endswith('.run-manifest.json')]
        bundle = verify_bundle(mp)
        verified = verify_saved_score(bundle, read_flight_receipts(mp.parent / 'flight-receipts.json'), mp.parent, ids)
        if verified['aggregate'] != evidence['aggregate'] or verified['rows'] != 1080:
            raise ValueError('Direct official evidence differs')
        sp = confined(data, bundle.manifest['scores']['path'])
        if sha(sp) != bundle.manifest['scores']['sha256']: raise ValueError('Direct score pin differs')
        results['direct_' + name] = read(sp)[SCORER]
        score_matrix(results['direct_' + name], ids)
        pins[name] = {'evidence_sha256': digest, 'manifest_sha256': sha(mp), 'scores_sha256': sha(sp)}
    return results, pins


def run(data, parent, out, a0=None):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    import flight_registration_check as registration
    recipe_path = ROOT / 'docs/readiness/2026-10-31-a0-direct-full1080-comparison/analysis-recipe-v1.json'
    recipe = read(recipe_path)
    if sha(recipe_path) != '20daf9aff885ea62b1c74b173c3270dc05d62223ef65ea9679f75a1caf138995':
        raise ValueError('paired read recipe changed')
    entries = registration.parse_ledger(ROOT, Path('docs/exposure-ledger.md')).get(recipe['read_run_id'], [])
    if (len(entries) != 1 or not registration.is_countersigned(entries[0])
            or entries[0]['artifacts'] != str(recipe_path.relative_to(ROOT))
            or sha(recipe_path) not in entries[0].get('notes', '')):
        raise ValueError('paired read must be uniquely registered and pinned')
    if sha(parent) != PARENT_SHA: raise ValueError('full1080 parent changed')
    ids = sorted(r['sample_id'] for r in read(parent)['items'])
    if len(ids) != 1080 or len(set(ids)) != 1080: raise ValueError('full1080 unique roster required')
    out = confined(data, out)
    if out.exists() or not out.is_relative_to(data.resolve() / 'analysis'):
        raise ValueError('fresh analysis output required')
    results, pins = load_direct(data, ids)
    result = {'schema': 'saea-a0-direct-full1080-comparison-v1', 'rows': 1080,
        'recipe_sha256': sha(recipe_path),
        'direct_pins': pins, 'parent_sha256': PARENT_SHA, 'new_model_calls': 0, 'new_scoring_jobs': 0,
        'complete_comparison': False, 'claims_full_test_benefit': False}
    if a0 is not None:
        a0 = confined(data, a0)
        complete = read(a0 / 'complete.json')
        if (complete['schema'] != 'saea-a0-full1080-source-bound-result-v1' or complete['rows'] != 1080
                or complete['recipe_sha256'] != RECIPE_SHA or complete['official_scoring_jobs'] != 1):
            raise ValueError('source-bound full1080 A0 required')
        for filename, key in (('scores.json','scores_sha256'), ('full1080.outputs.jsonl','outputs_sha256'), ('provenance.json','provenance_sha256')):
            if sha(a0 / filename) != complete[key]: raise ValueError('A0 sealed artifact changed')
        provenance = read(a0 / 'provenance.json')
        for key in ('source_custody', 'continuation_manifest'):
            p = provenance[key]
            if sha(confined(data, p['relpath'])) != p['sha256']: raise ValueError('A0 source changed')
        if provenance['continuation_manifest']['sha256'] != complete['continuation_manifest_sha256']:
            raise ValueError('A0 continuation identity differs')
        outputs = [json.loads(line)['sample_id'] for line in (a0 / 'full1080.outputs.jsonl').read_text().splitlines()]
        if sorted(outputs) != ids: raise ValueError('A0 outputs are not the full paired roster')
        results['a0'] = read(a0 / 'scores.json')
        result.update(paired(results, ids), complete_comparison=True, a0_complete_sha256=sha(a0 / 'complete.json'))
    result['quality'] = {name: {k: r['aggregate']['official'][k] for k in ('slu_f1', 'intent_accuracy', 'span_f1')}
                         for name, r in results.items()}
    result['limitations'] = ['Full train Fold B only; no test conclusion.',
        'A0 sources use recorded deployment transitions; no repeated full A0 variance estimate.',
        'Intervals measure paired sample uncertainty, not run-to-run uncertainty; no multiplicity correction.',
        'Direct lock is the primary reference; aa and ab are retained as sensitivity references.']
    with out.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'parent', 'out'): p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--a0', type=Path)
    result = run(**vars(p.parse_args()))
    print(json.dumps({k: result[k] for k in ('rows', 'complete_comparison', 'quality')}))
