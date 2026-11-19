"""Observe the first mismatching history position before and after caching."""
from functools import wraps


def attach(text, position, predictor, layers=2):
    import torch
    import attention_contract_compute as contract
    state = {}
    rows = {}
    hooks = []
    restores = []

    def keep(key, value):
        if key in rows:
            raise ValueError('duplicate observed prefix boundary '+key)
        rows[key] = value.detach().contiguous().cpu().clone()

    def on_entry(module, args):
        positions, hidden, _ = args
        axes = positions if positions.ndim == 2 else positions.unsqueeze(0)
        state.clear()
        for key, pos in [('origin', position), ('predictor', predictor)]:
            hit = torch.where(axes.eq(pos).all(0))[0].tolist()
            if len(hit) > 1:
                raise ValueError('ambiguous history position')
            if hit:
                state[key]=hit[0]
        if 'origin' in state:
            keep('decoder.input',hidden[state['origin']])

    hooks.append(text.layers[0].register_forward_pre_hook(on_entry))
    if type(layers) is not int or not 1 <= layers <= 8:
        raise ValueError('bounded origin layers required')
    for i in range(layers):
        layer=text.layers[i]
        attn=layer.self_attn
        for name,module in [('input_norm',layer.input_layernorm),('qkv_proj',attn.qkv_proj),
                            ('o_proj',attn.o_proj),('post_norm',layer.post_attention_layernorm),
                            ('mlp',layer.mlp)]:
            def capture(m,args,result,key=f'{i}.{name}'):
                if 'origin' in state:
                    out=result[0] if isinstance(result,tuple) else result
                    keep(key,out[state['origin']])
            hooks.append(module.register_forward_hook(capture))
        impl=attn.attn.impl
        before=vars(impl).get('forward_fused_infer_attention')
        original=impl.forward_fused_infer_attention

        def bind(original, i):
            @wraps(original)
            def step(*a,**kw):
                compute=contract.compute
                @wraps(compute)
                def observe(q,k,v,ql,kl,**opts):
                    if len(ql)!=1 or len(kl)!=1:
                        raise ValueError('isolated request required for prefix capture')
                    base=kl[0]-ql[0]
                    if 'origin' in state:
                        at=state['origin']
                        if base+at!=position:
                            raise ValueError('incorrect absolute origin position')
                        for name,x,index in [('q',q,at),('k',k,position),('v',v,position)]:
                            keep(f'{i}.attention.{name}',x[index])
                    if 'predictor' in state:
                        if base+state['predictor']!=predictor:
                            raise ValueError('incorrect absolute predictor position')
                        # Compare stored K/V at the earlier position in the same
                        # original request, without altering cache reads or writes.
                        keep(f'{i}.later.k',k[position])
                        keep(f'{i}.later.v',v[position])
                    result=compute(q,k,v,ql,kl,**opts)
                    if 'origin' in state:
                        keep(f'{i}.attention.out',result[state['origin']])
                    return result
                contract.compute=observe
                try:
                    return original(*a,**kw)
                finally:
                    contract.compute=compute
            return step
        wrapped=bind(original,i)
        impl.forward_fused_infer_attention=wrapped
        restores.append((impl,before,wrapped))

    def close():
        for h in hooks:
            h.remove()
        for impl,before,wrapped in reversed(restores):
            if vars(impl).get('forward_fused_infer_attention') is not wrapped:
                raise ValueError('prefix trace ownership changed')
            if before is None:
                del impl.forward_fused_infer_attention
            else:
                impl.forward_fused_infer_attention=before
        if len(rows)!=1+11*layers:
            raise ValueError(f'incomplete prefix trace: {list(rows)}')
        return rows
    return close


def compare(decode, prefill):
    import torch
    if decode.keys()!=prefill.keys():
        raise ValueError('different observed prefix boundaries')
    def diff(a,b):
        if a.shape!=b.shape or a.dtype!=b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('invalid pair at history position')
        return dict(unequal=int(torch.count_nonzero(a!=b)),max_abs=float((a.float()-b.float()).abs().max()))
    prefix={k:diff(decode[k],prefill[k]) for k in prefill if '.later.' not in k}
    stored={key.replace('.later.', '.'):diff(decode[key.replace('.later.', '.attention.')], value)
            for key, value in decode.items() if '.later.' in key}
    ordered=['decoder.input']
    for i in range(len(stored)//2):
        ordered.extend(f'{i}.{n}' for n in ('input_norm','qkv_proj','attention.q','attention.k',
                      'attention.v','attention.out','o_proj','post_norm','mlp'))
    if set(ordered)!=set(prefix):
        raise ValueError('incomplete computational order')
    return dict(first_prefill_difference=next((k for k in ordered if prefix[k]['unequal']),None),
                initial_prefill_vs_forced_prefill=prefix,initial_vs_cached_positions=stored)
