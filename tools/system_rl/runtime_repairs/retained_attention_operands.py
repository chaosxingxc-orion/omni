"""Record actual inputs to the installed precision attention, before its SDPA call."""
from functools import wraps


def observe(impl, shared, mode, rows):
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
            if len(ql)!=1 or len(kl)!=1 or len(k)!=len(v) or len(k)!=kl[0]:
                raise ValueError('expected single original query and logical KV stream')
            # Query row in a chunk; keys are logical prefix positions.
            end=kl[0]-ql[0]+target+1
            if end!=shared['absolute']+1:raise ValueError('cache positions disagree with original prefix')
            for name,tensor in [('query',q[target]),('key_prefix',k[:end]),('value_prefix',v[:end])]:
                if name in rows:raise ValueError('repeated target attention')
                rows[name]=tensor.detach().cpu().contiguous()
            result=compute(q,k,v,ql,kl,**kw)
            rows['attention_out']=result[target].detach().cpu().contiguous()
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


def compare(old,new):
    import torch
    expected={'query','key_prefix','value_prefix','attention_out'}
    if set(old)!=expected or set(new)!=expected:raise ValueError('missing live attention operands')
    result={}
    for k,x in new.items():
        y=old[k]
        if x.shape!=y.shape or x.dtype!=y.dtype or not (x.isfinite().all() and y.isfinite().all()):
            raise ValueError('invalid paired attention operand')
        unequal=(x!=y);n=int(unequal.sum());positions=unequal.flatten(1).any(-1).nonzero().flatten() if k.endswith('prefix') else None
        result[k]=dict(shape=list(x.shape),unequal=n,max_abs=float((x.float()-y.float()).abs().max()),
                       differing_positions=len(positions) if positions is not None else None,
                       first_position=int(positions[0]) if positions is not None and len(positions) else None)
    return result
