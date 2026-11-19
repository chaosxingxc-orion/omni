"""Explain a closed attention output discrepancy using its observed Q/K/V."""
import argparse
import json
from pathlib import Path

import numpy as np
from compare_retained_layer import compare, read_at
from collect_retained_boundaries import store


def analyze(root, layer, head_dim, max_rows):
    if head_dim < 1 or max_rows < 1:
        raise ValueError('positive head dimension and example budget required')
    report = compare(root, layer)
    load = lambda p: json.loads(p.read_bytes())
    _, hf = read_at(root/'hf', load(root/'hf/RESULT.json')['layer_sha256'], 'hf', 0)
    q, k, v = (hf[n+'.out'][0].reshape(hf[n+'.out'][0].shape[0], -1, head_dim)
               for n in ('q','k','v'))
    cuts = hf['cu_seqlens'][0]
    if cuts[0] != 0 or cuts[-1] != q.shape[0] or np.any(np.diff(cuts) <= 0):
        raise ValueError('invalid packed sequence boundaries')
    vl = load(root/'vl/RESULT.json')
    refs = {r['rank']: r for r in vl['ranks']}
    rows, total = [], 0
    for rank in report['ranks']:
        r = rank['rank']
        if (rank['weight_shard_mismatches'] or not rank['norm_eps_match'] or
            any(rank['measurements'][key]['unequal'] for key in ('cu_seqlens','q.out','k.out','v.out'))):
            raise ValueError('inputs are not aligned at attention boundary')
        _, actual = read_at(root/f'vl/rank-{r}', refs[r]['layer_sha256'], 'vl', r)
        left = np.split(hf['out.in'][0],len(refs),axis=1)[r]
        right = actual['out.in'][0]
        for i, j in np.argwhere(left != right):
            total += 1
            if len(rows) >= max_rows: continue
            head, feature = divmod(int(j)+r*left.shape[1],head_dim)
            seg=int(np.searchsorted(cuts,i,side='right'))-1
            lo,hi=cuts[seg:seg+2]
            logits=k[lo:hi,head].astype(np.float64) @ q[i,head].astype(np.float64) / np.sqrt(head_dim)
            weights=np.exp(logits-logits.max())
            value=float(weights @ v[lo:hi,head,feature].astype(np.float64) / weights.sum())
            rows.append(dict(rank=r,token=int(i),head=head,feature=feature,
                             hf=float(left[i,j]),vl=float(right[i,j]),cpu_math_reference=value))
    return dict(layer=layer,checked_attention_values=int(hf['out.in'][0].size),
                unequal=total,examples=rows,reference_is_actual_npu_execution=False,
                token_effect_established=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True)
    c=json.loads(p.parse_args().config.read_bytes())
    store(Path(c['output']),json.dumps(analyze(Path(c['root']),c['layer'],c['head_dim'],c['max_examples']),indent=2).encode())
