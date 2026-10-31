"""Paired, utterance-cluster bootstrap of an existing official checkpoint READ.

This is exploratory completed-subset evidence, never a full-roster lock decision.
No model contacts or additional gold reads are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def compare(root, replicates=10000):
    import numpy as np
    complete = json.loads((root / 'complete.json').read_bytes())
    columns = {}
    ids = None
    for name, record in complete['columns'].items():
        raw = (root / (name + '.scores.json')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['scores_sha256']:
            raise ValueError('score hash differs')
        result = json.loads(raw)
        rows = result['per_sample']
        keys = [r['sample_id'] for r in rows]
        if len(set(keys)) != len(keys) or len(keys) != complete['rows']:
            raise ValueError('invalid sample coverage')
        if ids is None: ids = sorted(keys)
        if sorted(keys) != ids: raise ValueError('unpaired sample sets')
        by_id = {r['sample_id']: r for r in rows}
        matrix = np.array([[sum(by_id[k][f'slu_f1_{d}_{c}'] for d in ('word', 'char'))
                            for c in ('tp', 'fp', 'fn')] for k in ids])
        total = matrix.sum(axis=0)
        f1 = 2 * total[0] / (2 * total[0] + total[1] + total[2])
        if abs(f1 - result['aggregate']['official']['slu_f1']) > 1e-6:
            raise ValueError('micro aggregation does not match official score')
        columns[name] = (matrix, result['aggregate']['official'])
    # Audio recordings of the same utterance must stay together when resampled.
    groups = sorted({k.split(':audio-')[0] for k in ids})
    group_index = {k: i for i, k in enumerate(groups)}
    indices = np.array([group_index[k.split(':audio-')[0]] for k in ids])
    rng = np.random.default_rng(20261031)
    weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=replicates)
    distributions = {}
    for name, (matrix, _) in columns.items():
        cluster = np.zeros((len(groups), 3))
        np.add.at(cluster, indices, matrix)
        totals = weights @ cluster
        distributions[name] = 2 * totals[:, 0] / (2 * totals[:, 0] + totals[:, 1] + totals[:, 2])
    pairs = {}
    for name, reference in [(n, 'lock') for n in columns if n != 'lock'] + [('l1-support', 'l1-support-sample0')]:
        delta = 100 * (distributions[name] - distributions[reference])
        pairs[name + '_minus_' + reference] = {
            'slu_f1_delta_pp': 100 * (columns[name][1]['slu_f1'] - columns[reference][1]['slu_f1']),
            'exploratory_percentile_95_interval_pp': np.quantile(delta, [.025, .975]).tolist()}
    return {'schema': 'saea-checkpoint-branch-comparison-v1', 'rows': len(ids),
            'utterance_clusters': len(groups), 'bootstrap_replicates': replicates, 'seed': 20261031,
            'checkpoint_sha256': complete['checkpoint_sha256'], 'new_model_calls': 0,
            'source_complete_sha256': hashlib.sha256((root / 'complete.json').read_bytes()).hexdigest(),
            'limitation': 'Completed nonrandom train subset; exploratory intervals, no multiplicity correction; not full-roster or test results.',
            'columns': {n: {**{k: v[k] for k in ('slu_f1', 'intent_accuracy', 'span_f1')},
                           'scores_sha256': complete['columns'][n]['scores_sha256']} for n, (_, v) in columns.items()},
            'paired_comparisons': pairs}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.input)
    with args.output.open('x') as f: json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
