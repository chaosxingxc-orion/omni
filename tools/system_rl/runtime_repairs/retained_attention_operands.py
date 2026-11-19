"""Record actual inputs to the installed precision attention, before its SDPA call."""
from functools import wraps


def observe(impl, shared, mode, rows, *, shape_shadow=False):
    import torch
    import attention_contract_compute as contract
    # corrected() imports contract.compute at call time. Scope the observation
    # to this layer's original method, and restore it even on a backend failure.
    original=impl.forward_fused_infer_attention
    prior=vars(impl).get('forward_fused_infer_attention')
    @wraps(original)
    def traced(*args,**kwargs):
        target=shared.get('at')
        if target is None:return original(*args,**kwargs)
        compute=contract.compute
        @wraps(compute)
        def record(q,k,v,ql,kl,**kw):
            sched=shared.get('scheduler')
            if sched:
                idx=sched['slot']
                if len(ql)!=sched['batch'] or len(ql)!=len(kl) or sum(ql[:idx])!=sched['lo'] or sum(ql[:idx+1])!=sched['hi']:
                    raise ValueError('scheduler and actual attention endpoints disagree')
            else:
                if len(ql)!=1 or len(kl)!=1:raise ValueError('unbound packed attention')
                idx=0
            if sum(kl)!=len(k) or len(k)!=len(v) or sum(ql)!=len(q):
                raise ValueError('logical attention stream lengths differ')
            base=sum(kl[:idx]); end=kl[idx]-ql[idx]+target-sum(ql[:idx])+1
            if end!=shared['absolute']+1:raise ValueError('cache positions disagree with original prefix')
            for name,tensor in [('query',q[target]),('key_prefix',k[base:base+end]),('value_prefix',v[base:base+end])]:
                if name in rows:raise ValueError('repeated target attention')
                rows[name]=tensor.detach().cpu().contiguous()
            result=compute(q,k,v,ql,kl,**kw)
            rows['attention_out']=result[target].detach().cpu().contiguous()
            if shape_shadow and mode=='prefill':
                if len(q)!=len(k) or target!=end-1 or end<=1:
                    raise ValueError('shape shadow requires full causal prefill')
                # Shadow only: all calls below use the very same target Q and
                # logical K/V. The original result still flows to the model.
                full=compute(q[:end],k[:end],v[:end],[end],[end],**kw)[-1]
                single=compute(q[target:target+1],k[:end],v[:end],[1],[end],**kw)[0]
                rows['shape_shadow_single_out']=single.detach().cpu().contiguous()
                rows['shape_shadow']=dict(
                    prefill_q_rows=len(q),prefix_kv_rows=end,
                    prefix_full_vs_cached_shape=delta(full,single),
                    actual_full_vs_prefix_full=delta(result[target],full),
                    actual_full_vs_cached_shape=delta(result[target],single))
                if shape_shadow=='variants':
                    rows['shape_shadow']['variants']=variants(q,k,v,target,kw,result[target],single)
            return result
        contract.compute=record
        try:return original(*args,**kwargs)
        finally:contract.compute=compute
    impl.forward_fused_infer_attention=traced
    def restore():
        if vars(impl).get('forward_fused_infer_attention') is not traced:raise ValueError('trace ownership changed')
        if prior is None:del impl.forward_fused_infer_attention
        else:impl.forward_fused_infer_attention=prior
    return restore


def variants(q,k,v,target,opts,actual,single):
    """Same actual BF16 operands, change mask form or arithmetic independently."""
    import torch
    import torch_npu
    from torch.nn import functional as F

    end=target+1
    scale=opts.get('scale') or q.shape[-1]**-.5
    causal=torch.arange(len(q),device=q.device)
    def sdpa(a,b,c,mask):
        return F.scaled_dot_product_attention(a.transpose(0,1)[None],
            b.transpose(0,1)[None],c.transpose(0,1)[None],
            attn_mask=mask,is_causal=False,enable_gqa=True,dropout_p=0.,scale=scale
        )[0].transpose(0,1)
    def math(a,b,c,mask):
        # Bounded shadow only. Keep float accumulation and immediately round
        # the attention output back to the model's BF16 boundary.
        aa=a.transpose(0,1).float()
        bb=b.transpose(0,1).float()
        cc=c.transpose(0,1).float()
        score=(aa@bb.transpose(-1,-2))*scale
        return (score.masked_fill(~mask,float('-inf')).softmax(-1)@cc).transpose(0,1).bfloat16()
    def fia(a,b,c):
        mask=torch.triu(torch.ones((2048,2048),device=q.device,dtype=torch.bool),1)
        return torch_npu.npu_fused_infer_attention_score(
            query=a,key=b,value=c,atten_mask=mask,input_layout='TND',
            actual_seq_lengths=[len(a)],actual_seq_lengths_kv=[len(b)],
            num_heads=q.shape[1],num_key_value_heads=k.shape[1],scale=scale,
            sparse_mode=3)[0].reshape_as(a)

    one=q[target:end]
    single_mask=causal[None,:]<=target
    # Choose actual neighboring Q rows. Full shape is tested by fused kernels;
    # the float math shadow stays small without using synthetic padding.
    start=max(0,end-64)
    tile_mask=causal[None,:]<=causal[start:][...,None]
    explicit=sdpa(q,k,v,causal[None,:]<=causal[:,None])[target]
    single_masked=sdpa(one,k,v,single_mask)[0]
    math_tile=math(q[start:],k,v,tile_mask)[target-start]
    math_single=math(one,k[:end],v[:end],single_mask[:,:end])[0]
    fia_full=fia(q,k,v)[target]
    fia_single=fia(one,k[:end],v[:end])[0]
    # Intervene on causally invisible V while preserving actual Q, K, shape
    # and mask. This distinguishes real future leakage from a rounding change.
    altered=v.clone()
    altered[end:]=torch.flip(altered[end:],dims=(-1,))*4
    masked_future=sdpa(q,k,altered,causal[None,:]<=causal[:,None])[target]

    def control(a,b):
        return delta(a.detach().cpu(),b.detach().cpu())
    return dict(explicit_full_vs_actual=control(explicit,actual),
                explicit_full_vs_single_masked=control(explicit,single_masked),
                single_masked_vs_cached=control(single_masked,single),
                explicit_reference_future_intervention=control(explicit,masked_future),
                fp32_tile_vs_single=control(math_tile,math_single),
                fp32_single_vs_current_cached=control(math_single,single),
                fia_full_vs_single=control(fia_full,fia_single),
                fia_single_vs_current_cached=control(fia_single,single))


def delta(a,b):
    import torch
    if a.shape!=b.shape or a.dtype!=b.dtype or not (a.isfinite().all() and b.isfinite().all()):
        raise ValueError('invalid paired attention operand')
    different=a!=b
    result=dict(shape=list(a.shape),unequal=int(different.sum()),max_abs=float((a.float()-b.float()).abs().max()))
    if result['unequal'] and a.ndim<3:
        at=tuple(int(i) for i in different.nonzero()[0])
        result['first_difference']=dict(index=list(at),left=float(a[at]),right=float(b[at]))
    return result


def compare(old,new):
    import torch
    expected={'query','key_prefix','value_prefix','attention_out'}
    if set(old)!=expected or set(new)-{'shape_shadow','shape_shadow_single_out'}!=expected:
        raise ValueError('missing live attention operands')
    result={}
    for k in sorted(expected):
        x,y=new[k],old[k]
        row=delta(x,y)
        positions=(x!=y).flatten(1).any(-1).nonzero().flatten() if k.endswith('prefix') else None
        result[k]=dict(**row,
                       differing_positions=len(positions) if positions is not None else None,
                       first_position=int(positions[0]) if positions is not None and len(positions) else None)
    return result
