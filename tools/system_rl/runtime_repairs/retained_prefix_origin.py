"""Observe the first mismatching history position before and after caching."""
from functools import wraps


def attach(text, position, predictor, layers=2, runner=None, seed=None, shadow_v=False):
    import torch
    import attention_contract_compute as contract
    state = {}
    rows = {}
    hooks = []
    restores = []
    shadow = {}

    def keep(key, value):
        if key in rows:
            raise ValueError('duplicate observed prefix boundary '+key)
        rows[key] = value.detach().contiguous().cpu().clone()

    def on_entry(module, args):
        positions, hidden, _ = args
        axes = positions if positions.ndim == 2 else positions.unsqueeze(0)
        state.clear()
        if runner is not None:
            slots=[j for j,rid in enumerate(runner.input_batch.req_ids) if runner.requests[rid].sampling_params.seed==seed]
            if not slots:return
            slot,=slots; lo,hi=map(int,runner.query_start_loc.np[slot:slot+2])
            state.update(slot=slot,lo=lo,hi=hi)
        for key, pos in [('origin', position), ('predictor', predictor)]:
            hit = torch.where(axes.eq(pos).all(0))[0].tolist()
            if runner is not None:hit=[i for i in hit if lo<=i<hi]
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
                    if 'origin' not in state and 'predictor' not in state:return compute(q,k,v,ql,kl,**opts)
                    if runner is None and (len(ql)!=1 or len(kl)!=1):
                        raise ValueError('isolated request required for prefix capture')
                    j=state.get('slot',0); start=sum(kl[:j])
                    if runner is not None and (sum(ql[:j])!=state['lo'] or sum(ql[:j+1])!=state['hi']):
                        raise ValueError('unpaired history scheduler row')
                    base=kl[j]-ql[j]-sum(ql[:j])
                    if 'origin' in state:
                        at=state['origin']
                        if base+at!=position:
                            raise ValueError('incorrect absolute origin position')
                        for name,x,index in [('q',q,at),('k',k,start+position),('v',v,start+position)]:
                            keep(f'{i}.attention.{name}',x[index])
                    if 'predictor' in state:
                        if base+state['predictor']!=predictor:
                            raise ValueError('incorrect absolute predictor position')
                        # Compare stored K/V at the earlier position in the same
                        # original request, without altering cache reads or writes.
                        keep(f'{i}.later.k',k[start+position])
                        keep(f'{i}.later.v',v[start+position])
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

    if shadow_v:
        from torch.overrides import TorchFunctionMode
        import hashlib
        import torch_npu
        proj=text.layers[0].self_attn.qkv_proj
        active=[]
        def digest(t):
            return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        class Probe(TorchFunctionMode):
            def __torch_function__(self,func,types,args=(),kwargs=None):
                out=func(*args,**(kwargs or {}))
                if func is not torch.nn.functional.linear or len(args)<2:return out
                x,w=args[:2]
                if x.ndim!=2 or tuple(w.shape)!=(512,2048) or x.dtype!=w.dtype or w.dtype!=torch.bfloat16:return out
                row=state['origin']
                branch=len(shadow)
                if branch>=2:raise ValueError('unexpected number of full K/V branches')
                def grouped(a):
                    y,=torch_npu.npu_grouped_matmul(x=[a.contiguous()],weight=[w.T.contiguous().unsqueeze(0)],
                        split_item=2,group_list_type=1,group_type=0,group_list=torch.tensor([len(a)],device=a.device,dtype=torch.int64))
                    return y
                g=grouped(x)[row];one=grouped(x[row:row+1])[0]
                shadow[str(branch)]=dict(input=digest(x[row]),weight=digest(w),native=digest(out[row]),
                    native_single=digest(func(x[row:row+1],w)[0]),grouped=digest(g),grouped_single=digest(one),
                    native_shard=digest(out[row,runner.rank*128:(runner.rank+1)*128]) if hasattr(runner,'rank') else None,
                    rows=len(x),row=row)
                return out
        def start(module,args):
            if 'origin' in state:
                mode=Probe();mode.__enter__();active.append(mode)
        def end(module,args,out):
            if active:
                active.pop().__exit__(None,None,None)
                if len(shadow)!=2:raise ValueError('expected live full-width K,V base projections, got '+str(len(shadow)))
        hooks.extend((proj.register_forward_pre_hook(start),proj.register_forward_hook(end,always_call=True)))
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
        if shadow_v:rows["projection_shadow"]=shadow
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
    sa=decode.pop("projection_shadow",None);sb=prefill.pop("projection_shadow",None)
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
                projection_shadow=dict(decode=sa,prefill=sb) if sa else None,initial_prefill_vs_forced_prefill=prefix,initial_vs_cached_positions=stored)
