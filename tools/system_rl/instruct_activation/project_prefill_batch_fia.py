"""Pack all requests into a single causal FIA for mixed and prefill language steps."""
import inspect, sys
from pathlib import Path
from project_prefill_tile import build as tile_parent
from project_screen_lanes import read,save
from decode_lineage import snapshot,changes,digest


def batches(q_lengths,kv_lengths,tile):
    if not q_lengths or len(q_lengths)!=len(kv_lengths) or any(type(q)!=int or type(k)!=int or not 0<q<=k<=24576 for q,k in zip(q_lengths,kv_lengths)) or tile!=128:
        raise ValueError('invalid query / KV lengths')
    out=[]
    for q,k in zip(q_lengths,kv_lengths):
        first=k-q
        end=(k+tile-1)//tile*tile
        lead=first%tile
        out.append((lead, end-first+lead, end))
    return out


def install_span():
    import torch,torch_npu
    from itertools import accumulate
    import attention_contract_compute as ac
    prev=ac.compute; masks={}
    def run(q,k,v,ql,kl,**opts):
        # The language layer's decode wrapper handles pure decode before this hook.
        # Delegate unrelated families and scopes to their actual installed handler.
        if max(ql)<=1:
            return prev(q,k,v,ql,kl,**opts)
        if torch.is_grad_enabled() or q.shape!=(sum(ql),8,128) or k.shape!=(sum(kl),1,128) or v.shape!=k.shape or not q.device==k.device==v.device or q.dtype!=torch.bfloat16 or not q.dtype==k.dtype==v.dtype or opts.get('scale')!=128**-.5:
            raise ValueError('unexpected mixed-prefill causal operands')
        spans=batches(ql,kl,128); qp=[];kp=[];vp=[];a=b=0
        for nq,nk,(lead,nrow,end) in zip(ql,kl,spans,strict=True):
            x=q.new_zeros((nrow,8,128));x[lead:lead+nq].copy_(q[a:a+nq]);qp.append(x)
            def pad(t):
                z=t[b:b+nk]
                return torch.cat((z,t.new_zeros((end-nk,1,128)))) if end!=nk else z
            kp.append(pad(k));vp.append(pad(v));a+=nq;b+=nk
        if q.device not in masks:
            masks[q.device]=torch.ones((2048,2048),device=q.device,dtype=torch.bool).triu_(1)
        ends=list(accumulate(x[1] for x in spans))
        result=torch_npu.npu_fused_infer_attention_score(query=torch.cat(qp),key=torch.cat(kp),
            value=torch.cat(vp),atten_mask=masks[q.device],input_layout='TND',sparse_mode=3,
            actual_seq_lengths=ends,actual_seq_lengths_kv=list(accumulate(x[2] for x in spans)),
            num_heads=8,num_key_value_heads=1,scale=opts['scale'])[0].reshape(-1,8,128)
        return torch.cat([result[e-nrow+lead:e-nrow+lead+nq] for e,nq,(lead,nrow,_) in zip(ends,ql,spans,strict=True)])
    ac.compute=run


def build(spec):
    if spec['query_layout']!='one_packed_batch_span':raise ValueError('unknown layout')
    tile_parent(spec)
    root=Path(spec['projection']['target']);parent=Path(spec['measured_parent'])
    before=snapshot(parent)
    if snapshot(root)!=before:raise ValueError('parent differs')
    rel='parents/jev-system-rl/implementation/decode_tiled_attention.py'
    path=root/'bundle'/rel;rp=root/'bundle/ROWS.json';rows=read(rp)
    if list(range(48))!=rows['tiled_layers'] or rows['decode_attention_sha256']!=digest(path):
        raise ValueError('tiled attention identity differs')
    src=path.read_text();anchor='    cached = {}\n'
    if src.count(anchor)!=1:raise ValueError('unexpected attention install')
    src=src.replace(anchor,'    install_span()\n'+anchor)+'\n\n'+inspect.getsource(batches)+'\n\n'+inspect.getsource(install_span)
    compile(src,rel,'exec');path.write_text(src);rows['decode_attention_sha256']=digest(path);save(rp,rows)
    for name in spec['projection']['lanes']:
        p=root/'bundle'/('SERVER-'+name+'.json');r=read(p)
        r['local_activation']['rows_ref'].update(sha256=digest(rp),bytes=rp.stat().st_size);save(p,r)
    d=changes(before,snapshot(root))
    if d['engine'] or set(d['files'])!={rel} or set(d['rows'])!={'decode_attention_sha256'}:raise ValueError('unexpected changes')
    save(root/'PACKED-SPAN.json',dict(parent_sha256=digest(parent/'CANDIDATE.json'),changes=d,layout=spec['query_layout']))

if __name__=='__main__':build(read(Path(sys.argv[1])))
