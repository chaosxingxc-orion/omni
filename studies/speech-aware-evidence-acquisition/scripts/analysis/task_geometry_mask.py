"""TF C2 inspired retrieval-bank adaptation, sealed before gold scoring."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import resource
import time

import numpy as np
from question_bank_selection_cli import rank, read
from question_bank_selection import digest


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def tokens(text):
    return set(re.findall(r'[^\W_]+', text.casefold(), flags=re.UNICODE))


def masks(members, rows, cards, topidx, dfs, peers, rare_max):
    labels = [m['intent'] for m in members]
    frequency = Counter(t for c in cards.values() for t in tokens(c['facet']['representation']))
    sizes = Counter(labels)
    assert topidx.shape[0] == len(labels) and topidx.shape[1] >= 5
    assert np.all((topidx[:, :5] >= 0) & (topidx[:, :5] < len(labels)))
    y = np.array(labels)
    same = (y[topidx[:, :5]] == y[:, None]).sum(axis=1)
    lexical = {d: np.array([bool(tokens(rows[m['utterance_id']]['sentence']) &
                {t for t in tokens(cards[m['intent']]['facet']['representation']) if frequency[t] < d})
                for m in members]) for d in dfs}
    geometric = {p: same >= p for p in peers}
    rare = np.array([sizes[x] <= rare_max for x in labels])
    result = {}
    for d in dfs:
        for p in peers:
            result[f'joint_df{d}_peer{p}'] = lexical[d] & geometric[p]
        result[f'lexical_df{d}'] = lexical[d].copy()
    for p in peers:
        result[f'geometric_peer{p}'] = geometric[p].copy()
    fallbacks = {}
    for name, mask in result.items():
        mask[rare] = True
        fallback = []
        for label in sorted(sizes):
            indices = y == label
            if not mask[indices].any():
                mask[indices] = True
                fallback.append(label)
        fallbacks[name] = fallback
    return result, fallbacks, frequency, same


def save_array(path, array):
    with path.open('xb') as f:
        np.save(f, array)
        f.flush(); os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['rank', 'score'])
    parser.add_argument('--config', required=True, type=Path)
    a = parser.parse_args(); cfg = read(a.config); root = Path(cfg['output'])
    root.mkdir(parents=True, exist_ok=True)
    guard = (root / 'runtime.guard').open('a+b'); fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start, cpu = time.monotonic(), time.process_time()
    for key, pin in cfg['inputs'].items():
        if key == 'gold' and a.phase == 'rank':
            continue
        assert digest(pin['path']) == pin['sha256'], key
    pin = lambda key: cfg['inputs'][key]['path']
    members, parent, cards = read(pin('members')), read(pin('parent')), read(pin('cards'))
    labels = parent['intents']; ids = parent['sample_ids']
    assert len(members) == 8554 and len(ids) == len(set(ids)) == 1080
    assert len(labels) == len(set(labels)) == len(cards) == 60 and set(labels) == set(cards)
    assert set(m['intent'] for m in members) <= set(labels)
    if a.phase == 'rank':
        assert not (root / 'rank-freeze.json').exists()
        rows = {str(r['slurp_id']): r for r in map(json.loads, Path(pin('metadata')).read_text().splitlines())}
        geom = np.load(pin('geometry')); matrix = np.load(pin('matrix'), mmap_mode='r')
        assert matrix.shape == (1080, 8554)
        all_masks, fallbacks, freq, same = masks(members, rows, cards, geom['topidx'], cfg['df'], cfg['peer'], cfg['rare_max'])
        reference = rank(matrix, members, labels, np.ones(len(members), dtype=bool))['coverage_frozen']
        assert reference == parent['orders']['question-gepa-coverage-gated-depth']
        orders = {'parent': reference}; details = {}
        for name, mask in all_masks.items():
            save_array(root / (name + '.npy'), mask)
            orders[name] = rank(matrix, members, labels, mask)['coverage_frozen']
            details[name] = dict(kept=int(mask.sum()), excluded=int((~mask).sum()), fallback_labels=fallbacks[name],
                                 changed_top3=sum(x[:3] != y[:3] for x, y in zip(reference, orders[name])))
        save(root/'rankings.json', dict(sample_ids=ids, intents=labels, orders=orders, gold_read=False))
        save(root/'mask-summary.json', dict(config=cfg['policy'], counts=details,
            empty_labels=sorted(set(labels)-set(m['intent'] for m in members)),
            df_documents=len(cards), same_peer_histogram=dict(Counter(map(int, same))), token_df=dict(freq)))
        files = [root/'rankings.json', root/'mask-summary.json'] + [root/(n+'.npy') for n in all_masks]
        save(root/'rank-freeze.json', dict(gold_read=False, parent_exact=True, n=1080,
            config_sha256=digest(a.config), files=[dict(name=p.name, sha256=digest(p)) for p in files],
            wall_seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        print('FROZEN', flush=True)
    else:
        assert not (root/'score.json').exists()
        frozen = read(root/'rank-freeze.json'); assert not frozen['gold_read'] and frozen['parent_exact']
        assert frozen['config_sha256'] == digest(a.config)
        for item in frozen['files']:
            assert digest(root/item['name']) == item['sha256']
        data = read(root/'rankings.json'); assert data['sample_ids'] == ids and data['intents'] == labels
        goldmap = {}
        for row in read(pin('gold')):
            sid = row['sample_id']; g = row['gold_intent']
            assert sid not in goldmap or goldmap[sid] == g
            goldmap[sid] = g
        assert set(goldmap) == set(ids)
        gold = [goldmap[s] for s in ids]; sizes = Counter(m['intent'] for m in members)
        indexes = {l: np.array([i for i, g in enumerate(gold) if g == l], dtype=int) for l in labels}
        assert all(len(x) for x in indexes.values())
        def metric(v):
            return dict(n=len(v), R1=int((v<=1).sum()), R3=int((v<=3).sum()), R5=int((v<=5).sum()), MRR=float(np.mean(np.where(np.isfinite(v),1/v,0))))
        ranks = {name: np.array([order.index(labels.index(g))+1 if labels.index(g) in order else np.inf for order,g in zip(orders,gold)]) for name,orders in data['orders'].items()}
        base = ranks['parent']; out = {}
        for name, v in ranks.items():
            per = {l: metric(v[ix]) for l, ix in indexes.items()}
            rare = np.array([i for i,g in enumerate(gold) if sizes[g] <= cfg['rare_max']]); other = np.array([i for i, g in enumerate(gold) if g.endswith('_other')], dtype=int)
            paired = {f'R{k}': dict(repaired=int(((base>k)&(v<=k)).sum()), regressed=int(((base<=k)&(v>k)).sum())) for k in (1,3,5)}
            out[name] = dict(overall=metric(v), macro60={k:float(np.mean([p[k]/p['n'] if k!='MRR' else p[k] for p in per.values()])) for k in ('R1','R3','R5','MRR')}, rare=metric(v[rare]), others=metric(v[other]) if len(other) else None, paired=paired,
                same_ordered_top3=sum(x[:3] == y[:3] for x,y in zip(data['orders']['parent'],data['orders'][name])),
                same_set_top3=sum(set(x[:3]) == set(y[:3]) for x,y in zip(data['orders']['parent'],data['orders'][name])), per_class=per)
        save(root/'score.json', dict(n=1080, rare_definition='legal TRAIN class size <=32 including empty', others_definition='registry labels ending _other', results=out,
            rank_freeze_sha256=digest(root/'rank-freeze.json'), wall_seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        print('SCORED', flush=True)

if __name__ == '__main__': main()
