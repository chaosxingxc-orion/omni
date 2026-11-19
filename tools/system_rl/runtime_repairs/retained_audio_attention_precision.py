"""Replay HF's frozen noncausal audio SDPA with live TP4 head operands."""

from functools import wraps


def full_head_sdpa(query, key, value, bounds, scale):
    """Apply the reference SDPA independently to each original audio window."""
    import torch
    from torch.nn.functional import scaled_dot_product_attention

    if (query.ndim != 4 or query.shape[0] != 1 or query.shape[2:] != (20, 64)
            or any(x.shape != query.shape or x.dtype != torch.bfloat16 for x in (query, key, value))
            or scale != 64**-.5 or len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != query.shape[1]
            or any(a >= b for a, b in zip(bounds, bounds[1:]))):
        raise ValueError('expected BF16 full-head audio windows and HF attention scale')
    outputs = []
    for a, b in zip(bounds, bounds[1:]):
        q, k, v = (x[:, a:b].transpose(1, 2) for x in (query, key, value))
        attended = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.,
                                                is_causal=False, scale=scale)
        outputs.append(attended.transpose(1, 2).contiguous())
    return torch.cat(outputs, dim=1)


def attach(tower, layers, rank, *, compare_native=False):
    import torch
    from audio_conv_precision import native_conv_context
    from vllm.distributed import tensor_model_parallel_all_gather as gather

    if (len(tower.layers) != 32 or not isinstance(layers, list) or not layers
            or any(type(i) is not int or i not in range(32) for i in layers)
            or layers != sorted(set(layers)) or rank not in range(4)):
        raise ValueError('explicit frozen audio attention layers required')
    mods = [(i, tower.layers[i].self_attn) for i in layers]
    if any(m.training or (m.num_heads,m.num_local_heads,m.head_dim) != (20,5,64)
           for _, m in mods):
        raise ValueError('unexpected audio TP4 attention layout')
    audit = {str(i): dict(calls=0, comparison=[]) for i in layers}
    installed = []

    def bind(orig, scale, row):
        @wraps(orig)
        def forward(query, key, value, cu_seqlens, max_seqlen):
            if torch.is_grad_enabled() or query.ndim != 4 or query.shape[2:] != (5,64):
                raise ValueError('diagnostic frozen audio attention requires 5 local heads')
            if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32,torch.int64):
                raise ValueError('original cumulative audio window lengths required')
            local = torch.stack((query,key,value)).contiguous()
            joined = gather(local, dim=3).contiguous()
            if not torch.equal(joined[:,:,:,rank*5:(rank+1)*5], local):
                raise ValueError('audio attention head order changed')
            # Construct the same T,H,D-contiguous operands as HF projection,
            # then transpose to B,H,T,D in the reference call.
            bounds = cu_seqlens.cpu().tolist()
            with native_conv_context(), torch.autocast('npu',dtype=torch.bfloat16):
                output = full_head_sdpa(*joined.unbind(0), bounds, scale)
            out = output[:,:,rank*5:(rank+1)*5].contiguous()
            if out.shape != query.shape or out.dtype != query.dtype:
                raise ValueError('reference attention changed shape or dtype')
            if compare_native:
                actual = orig(query=query,key=key,value=value,cu_seqlens=cu_seqlens,max_seqlen=max_seqlen)
                if actual.shape != out.shape or actual.dtype != out.dtype:
                    raise ValueError('native audio attention ABI differs')
                row['comparison'].append(dict(unequal=int((actual!=out).sum()),
                                max_abs=float((actual.float()-out.float()).abs().max())))
            row['calls'] += 1
            return out
        return forward

    for i, owner in mods:
        mod = owner.attn
        prior = vars(mod).get('forward')
        fn = bind(mod.forward, owner.scaling, audit[str(i)])
        mod.forward = fn
        installed.append((mod, prior, fn))

    def restore():
        for mod, prior, fn in reversed(installed):
            if vars(mod).get('forward') is not fn:
                raise ValueError('audio attention implementation owner changed')
            if prior is None:
                del mod.forward
            else:
                mod.forward = prior
        if any(not row['calls'] for row in audit.values()):
            raise ValueError('candidate audio attention did not execute')
        return dict(arithmetic='hf_full_head_windowed_bf16_sdpa', layers=audit)

    return restore
