"""Diagnostic-only common BF16 FIA for the original causal text attention."""
from collections import Counter
from functools import wraps


def causal_attention():
    """One implementation for the live TP4 and full-head FSDP scopes."""
    import torch
    import torch_npu

    masks = {}

    def compute(q, k, v, ql, kl, *, scale):
        if (torch.is_grad_enabled() or q.device.type != 'npu'
                or not (q.device == k.device == v.device)
                or any(x.dtype != torch.bfloat16 for x in (q, k, v))
                or q.shape[1:] != (8, 128) or k.shape[1:] != (1, 128)
                or k.shape != v.shape or scale != 128**-.5
                or not ql or len(ql) != len(kl)
                or any(type(a) is not int or type(b) is not int or not 0 < a <= b <= 24576
                       for a, b in zip(ql, kl))
                or sum(ql) != len(q) or sum(kl) != len(k)):
            raise ValueError('expected no-gradient BF16 causal Omni attention operands')
        if q.device not in masks:
            masks[q.device] = torch.ones((2048, 2048), device=q.device, dtype=torch.bool).triu_(1)
        output, qi, ki = [], 0, 0
        for nq, nk in zip(ql, kl):
            query = q[qi:qi+nq].contiguous()
            result = torch_npu.npu_fused_infer_attention_score(
                query=query, key=k[ki:ki+nk].contiguous(), value=v[ki:ki+nk].contiguous(),
                atten_mask=masks[q.device], input_layout='TND', sparse_mode=3,
                actual_seq_lengths=[nq], actual_seq_lengths_kv=[nk], num_heads=8,
                num_key_value_heads=1, scale=scale)[0].reshape_as(query)
            if result.dtype != q.dtype:
                raise ValueError('FIA changed the model attention dtype')
            output.append(result)
            qi += nq
            ki += nk
        return torch.cat(output)

    return compute


def _validate(layers, model_layers):
    if (not isinstance(layers, list) or not layers
            or any(type(i) is not int or i not in range(len(model_layers)) for i in layers)
            or layers != sorted(set(layers))):
        raise ValueError('expected explicitly ordered, unique candidate layers')


def attach_vllm(model, layers):
    import attention_contract_compute as contract
    text = model.language_model.model
    _validate(layers, text.layers)
    compute = causal_attention()
    restores, counts, active = [], Counter(), []
    baseline = contract.compute

    def consume(q, k, v, ql, kl, **opts):
        if not active:
            return baseline(q, k, v, ql, kl, **opts)
        out = compute(q, k, v, ql, kl, **opts)
        counts[active[-1]] += 1
        return out

    def wrap(orig, index):
        @wraps(orig)
        def forward(*a, **kw):
            if active:
                raise ValueError('overlapping causal attention scopes')
            active.append(index)
            try:
                return orig(*a, **kw)
            finally:
                active.pop()
        return forward

    for i in layers:
        impl = text.layers[i].self_attn.attn.impl
        prior = vars(impl).get('forward_fused_infer_attention')
        fn = wrap(impl.forward_fused_infer_attention, i)
        impl.forward_fused_infer_attention = fn
        restores.append((impl, prior, fn))
    contract.compute = consume

    def close():
        for impl, prior, fn in reversed(restores):
            if vars(impl).get('forward_fused_infer_attention') is not fn:
                raise ValueError('attention candidate ownership changed')
            if prior is None:
                del impl.forward_fused_infer_attention
            else:
                impl.forward_fused_infer_attention = prior
        if contract.compute is not consume or active:
            raise ValueError('attention leaf ownership changed')
        contract.compute = baseline
        return _receipt(layers, counts)
    return close


def attach_hf(text, layers, unwrap):
    """Intercept the real HF SDPA call inside the selected decoder attention only."""
    import torch
    from torch.nn import functional as F
    from torch.overrides import TorchFunctionMode

    _validate(layers, text.layers)
    compute = causal_attention()
    counts, handles, active = Counter(), [], []

    class AttentionMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kw = kwargs or {}
            if func is not F.scaled_dot_product_attention:
                return func(*args, **kw)
            if len(args) != 3 or set(kw)-{'attn_mask','is_causal','scale','enable_gqa','dropout_p'}:
                raise ValueError('unexpected HF SDPA call ABI')
            q, k, v = args
            if (q.ndim != 4 or q.shape[:2] != (1, 32) or k.shape != (1, 4, q.shape[2], 128)
                    or kw.get('attn_mask') is not None or not kw.get('is_causal')
                    or kw.get('dropout_p', 0.) != 0. or not kw.get('enable_gqa')):
                raise ValueError('expected causal unpadded HF single-row batch, 32Q/4KV')
            n = q.shape[2]
            groups = [compute(q[0, i*8:(i+1)*8].transpose(0, 1),
                              k[0, i:i+1].transpose(0, 1), v[0, i:i+1].transpose(0, 1),
                              [n], [n], scale=kw.get('scale') or 128**-.5)
                      for i in range(4)]
            counts[active[-1][0]] += 1
            return torch.cat(groups, dim=1).transpose(0, 1).unsqueeze(0)

    def enter(i):
        def hook(*_):
            mode = AttentionMode()
            mode.__enter__()
            active.append((i, mode))
        return hook

    def leave(i):
        def hook(*_):
            if not active or active[-1][0] != i:
                raise ValueError('HF attention scope changed')
            _, mode = active.pop()
            mode.__exit__(None, None, None)
        return hook

    for i in layers:
        module = unwrap(text.layers[i]).self_attn
        handles.extend((module.register_forward_pre_hook(enter(i)),
                        module.register_forward_hook(leave(i), always_call=True)))

    def close():
        for h in handles:
            h.remove()
        if active:
            raise ValueError('unfinished HF attention scope')
        return _receipt(layers, counts)
    return close


def _receipt(layers, counts):
    if set(counts) != set(layers):
        raise ValueError('candidate layers were not consumed by the forward')
    return dict(arithmetic='causal_bf16_tnd_fia', layers=dict(counts),
                training_backward_validated=False)
