"""Compare actual layer operands in their recorded dtype and TP layout."""
import hashlib
import json
import numpy as np


def read_at(root, expected, backend, rank):
    doc=json.loads((root/'LAYER.json').read_bytes())
    b=(root/'LAYER.safetensors').read_bytes()
    assert hashlib.sha256(b).hexdigest()==doc['sha256']==expected
    assert doc['rank']==rank and doc['backend']==backend
    start=8+int.from_bytes(b[:8],'little');out={}
    for key,v in json.loads(b[8:start]).items():
        dtype={'BF16':'<u2','F32':'<f4','I32':'<i4','I64':'<i8'}[v['dtype']]
        lo,hi=v['data_offsets'];x=np.frombuffer(b[start+lo:start+hi],dtype=dtype).reshape(v['shape'])
        if v['dtype']=='BF16':x=(x.astype('<u4')<<16).view('<f4')
        assert np.isfinite(x).all()
        out[key]=(x,v['dtype'])
    return doc,out


def compare(root,layer):
    hf=json.loads((root/'hf/RESULT.json').read_bytes());vl=json.loads((root/'vl/RESULT.json').read_bytes())
    hdoc,h=read_at(root/'hf',hf['layer_sha256'],'hf',0)
    rows=[]
    for spec in sorted(vl['ranks'],key=lambda x:x['rank']):
        r=spec['rank'];vd,v=read_at(root/f'vl/rank-{r}',spec['layer_sha256'],'vl',r)
        assert vd['layer']==hdoc['layer']==layer
        exp={k:x for k,x in hdoc['weights'].items() if k.endswith(f'.rank{r}')}
        assert set(exp)==set(vd['weights'])
        weight_diff=[k for k,x in exp.items() if x!=vd['weights'][k]]
        checks={}
        def check(name,left,right):
            a,ad=left;b,bd=right
            assert a.shape==b.shape,(name,a.shape,b.shape)
            # A mixed-precision norm may agree after conversion into the actual
            # BF16 linear kernel. Preserve both the raw and rounded comparisons.
            rec=dict(shape=list(a.shape),hf_dtype=ad,vl_dtype=bd,unequal=int(np.count_nonzero(a!=b)),
                     max_abs=float(np.max(np.abs(a.astype('float64')-b))))
            if {ad,bd}=={'F32','BF16'}:
                def bf(x):
                    u=x.astype('<f4').view('<u4')
                    return (u+0x7fff+((u>>16)&1))>>16
                rec['unequal_after_bf16']=int(np.count_nonzero(bf(a)!=bf(b)))
            checks[name]=rec
        def part(val):return np.split(val[0],4,axis=-1)[r],val[1]
        for name in ['cu_seqlens','norm.in','norm.out']:check(name,h[name],v[name])
        for i,n in enumerate(('q','k','v')):
            check(n+'.in',h[n+'.in'],v['qkv.in'])
            check(n+'.out',part(h[n+'.out']),(np.split(v['qkv.out'][0],3,axis=-1)[i],v['qkv.out'][1]))
        for name in ['out.in','out.out','norm2.in','norm2.out','fc1.in','fc1.out','fc2.in','fc2.out']:
            check(name,part(h[name]) if name in ('out.in','fc1.out','fc2.in') else h[name],v[name])
        rows.append(dict(rank=r,weight_shard_mismatches=weight_diff,norm_eps_match=hdoc['norm_eps']==vd['norm_eps'],
            first_observed_value_difference=next((k for k,x in checks.items() if x['unequal']),None),
            measurements=checks))
    assert [r['rank'] for r in rows]==list(range(4))
    return dict(layer=layer,ranks=rows,causal_token_effect_verified=False)
