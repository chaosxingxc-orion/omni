"""TRAIN-only calibrated TFG experiments; rank is sealed before separate gold scoring."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
def source_groups(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        keys = [('id', str(row['slurp_id'])), ('text', ' '.join(row['sentence'].casefold().split()))]
        keys += [('recording', Path(r['file']).stem.removesuffix('-headset')) for r in row.get('recordings', [])]
        for key in keys:
            if key in seen:
                a, b = find(i), find(seen[key])
                if a != b: parent[a] = b
            else: seen[key] = i
    return np.array([find(i) for i in range(len(rows))])


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def zscore(values):
    std = np.std(values, ddof=0)
    return np.clip((values - np.mean(values)) / std, -2, 2) if std else np.zeros_like(values, dtype=float)


def aggregate(raw, columns, bias=None):
    """Eligibility is raw > 0, even when calibrated evidence or label score is negative."""
    n = raw.shape[0]; score = np.zeros((n, len(columns))); median = np.zeros_like(score)
    valid = np.zeros_like(score, dtype=bool)
    for j, (cols, count) in enumerate(columns):
        if not len(cols): continue
        values = raw[:, cols]; eligible = values > 0
        adjusted = values if bias is None else values - bias[cols]
        selected = np.sort(np.where(eligible, adjusted, -np.inf), axis=1)[:, ::-1][:, :5]
        depth = np.minimum(eligible.sum(axis=1), 5); valid[:, j] = depth > 0
        mean = np.divide(np.where(np.isfinite(selected), selected, 0).sum(axis=1), depth, out=np.zeros(n), where=depth>0)
        top = np.where(depth > 0, selected[:, 0], 0)
        score[:, j] = mean + .5 * (top - mean) * 32 / (32 + count)
        for k in range(1, 6):
            ix = depth == k
            if np.any(ix): median[ix, j] = np.median(selected[ix, :k], axis=1)
    return score, valid, median


def orders(scores, valid):
    return [[int(j) for j in np.argsort(-row, kind='stable') if mask[j]] for row, mask in zip(scores, valid)]


def swaps(parent, scores, medians, tau):
    output = [row.copy() for row in parent]
    for i, row in enumerate(output):
        if len(row) >= 2 and scores[i, row[0]] - scores[i, row[1]] < tau and medians[i, row[1]] > medians[i, row[0]]:
            row[0], row[1] = row[1], row[0]
    return output


def main():
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['rank', 'score']); p.add_argument('--config', required=True); a = p.parse_args()
    cfg = read(a.config); root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    guard = (root/'runtime.guard').open('a+b'); fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start, cpu = time.monotonic(), time.process_time()
    for name, pin in cfg['inputs'].items():
        if name == 'gold' and a.phase == 'rank': continue
        assert digest(pin['path']) == pin['sha256'], name
    path = lambda name: cfg['inputs'][name]['path']
    members, parent = read(path('members')), read(path('parent'))
    labels, ids = parent['intents'], parent['sample_ids']; n = len(members)
    assert n == 8554 and len(ids) == len(set(ids)) == 1080 and len(labels) == 60
    columns = []
    for label in labels:
        indices = np.array([i for i, m in enumerate(members) if m['intent'] == label], dtype=int)
        columns.append((indices, len({members[i]['sentence_sha256'] for i in indices})))
    if a.phase == 'rank':
        assert not (root/'rank-freeze.json').exists()
        rows = {str(r['slurp_id']): r for r in map(json.loads, Path(path('metadata')).read_text().splitlines())}
        groups = source_groups([rows[m['utterance_id']] for m in members])
        keys = np.load(path('keys'), mmap_mode='r'); assert keys.shape[0] == n
        assert np.allclose(np.linalg.norm(keys, axis=1), 1, atol=1e-6)
        incoming = np.zeros(n, dtype=np.int64); r10 = np.empty(n); train_scores = np.empty((n, len(labels))); train_valid = np.empty((n, len(labels)), dtype=bool)
        for start_row in range(0, n, cfg['block_rows']):
            end = min(n, start_row + cfg['block_rows']); sim = keys[start_row:end] @ keys.T
            sim[groups[start_row:end, None] == groups[None, :]] = -np.inf
            # Same source excluded; fixed k=20 all legal questions, deterministic stable ties.
            nn = np.argsort(-sim, axis=1, kind='stable')[:, :20]
            near = np.take_along_axis(sim, nn, axis=1); assert np.isfinite(near).all()
            incoming += np.bincount(nn.ravel(), minlength=n); r10[start_row:end] = near[:, :10].mean(axis=1)
            train_scores[start_row:end], train_valid[start_row:end], _ = aggregate(sim, columns)
        train_order = orders(train_scores, train_valid)
        assert all(len(row) >= 4 for row in train_order)
        gap12 = np.array([train_scores[i, row[0]]-train_scores[i, row[1]] for i,row in enumerate(train_order)])
        gap14 = np.array([train_scores[i, row[0]]-train_scores[i, row[3]] for i,row in enumerate(train_order)])
        delta = float(np.median(gap14)); taus = {str(q): float(np.quantile(gap12, q)) for q in cfg['tau_quantiles']}
        assert delta >= 0 and np.isfinite(delta)
        matrix = np.load(path('matrix'), mmap_mode='r'); assert matrix.shape == (1080, n)
        scores, valid, medians = aggregate(matrix, columns); baseline = orders(scores, valid)
        assert baseline == parent['orders']['question-gepa-coverage-gated-depth']
        result = {'parent': baseline}; bias_info = {}
        for feature, values in [('hubness', incoming), ('r10', r10)]:
            z = zscore(values)
            for scale in cfg['bias_multipliers']:
                bias = scale * delta * z
                assert np.max(np.abs(bias)) <= 3 * delta + 1e-15
                s, v, _ = aggregate(matrix, columns, bias)
                assert np.array_equal(v, valid)
                name = f'n1_{feature}_{scale:g}'; result[name] = orders(s, v)
                bias_info[name] = dict(min=float(bias.min()), max=float(bias.max()), negative_eligible_label_scores=int(((s<0)&v).sum()))
        for q in cfg['tau_quantiles']:
            name = f'n2_p{int(q*100):02d}'; result[name] = swaps(baseline, scores, medians, taus[str(q)])
            assert all(set(x[:3]) == set(y[:3]) and x[2:] == y[2:] for x,y in zip(baseline,result[name]))
        save(root/'calibration.json', dict(train_n=n, source_groups=int(len(set(groups))), delta=delta, delta_definition='TRAIN source-excluded median parent score top1 minus top4', taus=taus, tau_definition='TRAIN source-excluded parent top1 minus top2 quantile', hubness_mean=float(incoming.mean()), hubness_std=float(incoming.std()), r10_mean=float(r10.mean()), r10_std=float(r10.std()), bias=bias_info, text_audio_scale_warning='TRAIN text cosine margins are not audio cosine margins'))
        save(root/'rankings.json', dict(sample_ids=ids, intents=labels, orders=result, gold_read=False))
        save(root/'rank-freeze.json', dict(gold_read=False, parent_exact=True, config_sha256=digest(a.config), files=[dict(name=x,sha256=digest(root/x)) for x in ('calibration.json','rankings.json')], wall_seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    else:
        assert not (root/'score.json').exists(); freeze=read(root/'rank-freeze.json')
        assert not freeze['gold_read'] and freeze['parent_exact'] and freeze['config_sha256']==digest(a.config)
        for pin in freeze['files']: assert digest(root/pin['name']) == pin['sha256']
        data=read(root/'rankings.json'); assert data['sample_ids']==ids and data['intents']==labels
        goldmap={}
        for row in read(path('gold')):
            sid=row['sample_id']; g=row['gold_intent']; assert sid not in goldmap or goldmap[sid]==g; goldmap[sid]=g
        assert set(goldmap)==set(ids); gold=[goldmap[s] for s in ids]; sizes=Counter(m['intent'] for m in members)
        ranks={name:np.array([row.index(labels.index(g))+1 if labels.index(g) in row else np.inf for row,g in zip(orders_,gold)]) for name,orders_ in data['orders'].items()}
        def metric(v): return dict(n=len(v), R1=int((v<=1).sum()), R3=int((v<=3).sum()), R5=int((v<=5).sum()), MRR=float(np.mean(np.where(np.isfinite(v),1/v,0))))
        indexes={l:np.array([i for i,g in enumerate(gold) if g==l],dtype=int) for l in labels}; assert all(len(x) for x in indexes.values())
        rare=np.array([i for i,g in enumerate(gold) if sizes[g]<=32]); other=np.array([i for i,g in enumerate(gold) if g.endswith('_other')]); base=ranks['parent']; out={}
        for name,v in ranks.items():
            per={l:metric(v[ix]) for l,ix in indexes.items()}; rows_=data['orders'][name]
            out[name]=dict(overall=metric(v),macro60={k:float(np.mean([x[k]/x['n'] if k!='MRR' else x[k] for x in per.values()])) for k in ('R1','R3','R5','MRR')}, rare=metric(v[rare]),others=metric(v[other]),paired={f'R{k}':dict(repaired=int(((base>k)&(v<=k)).sum()),regressed=int(((base<=k)&(v>k)).sum())) for k in (1,3,5)},changed_ordered_top3=sum(x[:3]!=y[:3] for x,y in zip(data['orders']['parent'],rows_)),changed_set_top3=sum(set(x[:3])!=set(y[:3]) for x,y in zip(data['orders']['parent'],rows_)),per_class=per)
        save(root/'score.json',dict(n=1080,results=out,rank_freeze_sha256=digest(root/'rank-freeze.json'),wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(a.phase+' complete',flush=True)

if __name__ == '__main__': main()
