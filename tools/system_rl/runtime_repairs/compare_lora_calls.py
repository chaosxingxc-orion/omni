"""Compare actual O LoRA A operands, channel order and BF16 results by absolute token."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(file):
    return json.loads(file.read_bytes())


def checkpoint_weight(path, layer, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError('adapter does not match HTTP activation')
    start = int.from_bytes(raw[:8], 'little') + 8
    meta = json.loads(raw[8:start])[f'base_model.model.model.layers.{layer}.self_attn.o_proj.lora_A.weight']
    if meta['dtype'] != 'BF16' or meta['shape'] != [8, 4096]:
        raise ValueError('unexpected published weight')
    a, b = meta['data_offsets']
    return digest(raw[start+a:start+b])


def unpack(root, item, lane, expected, key):
    folder = root / ('state-'+lane)
    complete = [f for f in (folder/'full-logits').glob('*/COMPLETE.json') if read(f)==item]
    path, = complete
    workers = sorted(item['workers'],key=lambda w:w['rank'])
    if [w['rank'] for w in workers]!=list(range(4)):
        raise ValueError('missing TP ranks')
    calls = workers[0]['prefix_trace']['gemm']
    for w in workers:
        t = w['prefix_trace']
        if t['gemm'] != calls or any(c['weight'] != expected or not c['input_local_shard_exact'] for c in calls):
            raise ValueError('inconsistent live O weights or shard ordering')
    rows=[]
    refs=workers[0]['prefix_trace']['operand_refs']
    end=workers[0]['prefix_trace']['first_computed']
    for i,c in enumerate(calls):
        ref=refs[f'{i}-{key}']; meta_path=path.parent/'workers/rank-0'/Path(ref['path']).name
        m=read(meta_path);raw=meta_path.with_suffix('.bin').read_bytes()
        if (digest(meta_path.read_bytes())!=ref['sha256'] or digest(raw)!=m['sha256'] or
                m['identity']['engine_request_id']!=item['engine_request_id'] or m['dtype']!='torch.bfloat16'):
            raise ValueError('operand custody mismatch')
        data=np.frombuffer(raw,'<u2').reshape(m['shape'])
        if c['prior']!=end or len(data)!=c['rows']:
            raise ValueError('missing scheduler rows')
        rows.append(data[c['begin']:c['end']]);end+=len(rows[-1])
    if end!=len(item['prompt_token_ids']):
        raise ValueError('incomplete prompt')
    return np.concatenate(rows), calls


def compare(traces, root, adapter, layer):
    c,w=[read(traces/(k+'.json')) for k in ('cold','warm')]
    if c['prompt_token_ids']!=w['prompt_token_ids']:
        raise ValueError('different target')
    cut=w['workers'][0]['prefix_trace']['first_computed']
    if not 0<cut<len(c['prompt_token_ids']) or c['workers'][0]['prefix_trace']['first_computed']!=0:
        raise ValueError('expected cold and partial windows')
    results={};calls=[]
    seen_weight=None
    for label,record,lane in [('cold',c,'b'),('warm',w,'a')]:
        expected=checkpoint_weight(adapter,layer,read(root/('state-'+lane)/'ACTIVATION.json')['weights_ref']['sha256'])
        results[label]={}
        for k in ('x','u'):
            y, roster=unpack(root,record,lane,expected,k)
            results[label][k]=y[cut:] if label=='cold' else y
        ref=record['workers'][0]['prefix_trace']['operand_refs']['0-a']
        weight_path,=(root/('state-'+lane)/'full-logits').glob('*/workers/rank-0/operand-0-a.json') if lane=='b' else [
            f for f in (root/'state-a/full-logits').glob('*/workers/rank-0/operand-0-a.json')
            if digest(f.read_bytes())==ref['sha256']]
        weights=weight_path.with_suffix('.bin').read_bytes()
        if digest(weights)!=expected or (seen_weight is not None and seen_weight!=expected):
            raise ValueError('BF16 checkpoint identity changed across paths')
        seen_weight=expected
        if any(g['site']!=f'op{layer}.' for g in roster):
            raise ValueError('different layer')
        calls.append([dict(prior=g['prior'],rows=g['rows'],stride=g['x_stride']) for g in roster])
    lhs,rhs=results.values()
    if lhs['x'].shape!=rhs['x'].shape or not np.array_equal(lhs['x'],rhs['x']):
        raise ValueError('logical inputs differ')
    different=lhs['u']!=rhs['u']
    indices=np.argwhere(different)
    report=dict(cached=cut,input_shape=list(lhs['x'].shape),output_elements=int(different.size),
                weight_matches_checkpoint=True,local_channel_order_matches=True,same_logical_input=True,
                output_differences=len(indices),distinct_rows=len(np.unique(indices[:,0])),
                first_difference=None,scheduler=dict(zip(('cold','warm'),calls)))
    if len(indices):
        i,j=map(int,indices[0]);report['first_difference']=dict(position=i+cut,channel=j,
                        cold_bits=int(lhs['u'][i,j]),warm_bits=int(rhs['u'][i,j]))
    # Diagnose the small set of disagreements by a separate high-precision dot.
    # Both captures stay authoritative; this is an arithmetic reference only.
    if len(indices):
        to64=lambda a:(a.astype('<u4')<<16).view('<f4').astype(np.float64)
        a=to64(np.frombuffer(weights,'<u2').reshape(8,4096))
        xc=to64(lhs['x'][indices[:,0]])
        hi=np.sum(xc*a[indices[:,1]],axis=1,dtype=np.float64)
        cv=to64(lhs['u'][different]);wv=to64(rhs['u'][different])
        cd=np.abs(cv-hi);wd=np.abs(wv-hi)
        report['reference_on_differences']=dict(cold_closer=int((cd<wd).sum()),warm_closer=int((wd<cd).sum()),
            tied=int((cd==wd).sum()),max_absolute_output_delta=float(np.max(np.abs(cv-wv))))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('traces','root','adapter'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--layer',type=int,required=True)
    print(json.dumps(compare(**vars(p.parse_args())),indent=2))
