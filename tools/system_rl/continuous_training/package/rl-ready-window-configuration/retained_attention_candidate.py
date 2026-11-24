"""Diagnostic-only common BF16 FIA for the original causal text attention."""
from collections import Counter
from functools import wraps


def tiled(fia, q, k, v, tile):
    """Use the same absolute-position query tile and KV extent in prefill and decode."""
    import torch
    n,m=len(q),len(k);start=m-n
    if n<=0 or start<0 or len(v)!=m or tile!=128:raise ValueError('invalid absolute query tile')
    pieces=[]
    for lo in range(start//tile*tile,m,tile):
        stop=lo+tile;left=max(lo,start);right=min(stop,m)
        # Unrequested query rows do not interact. Future padded keys are
        # masked by the same bottom-right causal mask as real future keys.
        qq=q.new_zeros((tile,*q.shape[1:]))
        qq[left-lo:right-lo].copy_(q[left-start:right-start])
        def prefix(x):
            if stop<=m:return x[:stop].contiguous()
            return torch.cat((x,x.new_zeros((stop-m,*x.shape[1:]))))
        pieces.append(fia(qq,prefix(k),prefix(v))[left-lo:right-lo])
    return torch.cat(pieces)


def causal_attention(*, training_vjp=False, absolute_tile=None):
    """One implementation for the live TP4 and full-head FSDP scopes."""
    import torch
    import torch_npu

    masks = {}

    class CausalPrefill(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, n, scale):
            ctx.save_for_backward(q,k,v)
            ctx.n, ctx.scale = n, scale
            return compute(q,k,v,[n],[n],scale=scale)

        @staticmethod
        def backward(ctx, dy):
            # Compute the derivative of causal grouped-query attention using
            # the training operator's own softmax state. Never feed it FIA
            # scratch buffers, which have a different ABI.
            with torch.enable_grad():
                q,k,v = (x.detach().contiguous().requires_grad_() for x in ctx.saved_tensors)
                n=ctx.n
                output = torch_npu.npu_fusion_attention(q,k,v,8,'TND',atten_mask=masks[q.device],
                         scale=ctx.scale,actual_seq_qlen=[n],actual_seq_kvlen=[n],
                         sparse_mode=3,keep_prob=1.)[0]
                grads = torch.autograd.grad(output,(q,k,v),dy.contiguous())
            return (*grads, None, None)

    def compute(q, k, v, ql, kl, *, scale):
        if ((torch.is_grad_enabled() and not training_vjp) or q.device.type != 'npu'
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
        if torch.is_grad_enabled():
            if ql != kl or len(ql) != 1:
                raise ValueError('training VJP requires a single full causal prefill')
            return CausalPrefill.apply(q,k,v,ql[0],scale)
        output, qi, ki = [], 0, 0
        for nq, nk in zip(ql, kl):
            query = q[qi:qi+nq].contiguous()
            def fia(qq,kk,vv):
                return torch_npu.npu_fused_infer_attention_score(
                    query=qq,key=kk,value=vv,atten_mask=masks[q.device],input_layout='TND',sparse_mode=3,
                    actual_seq_lengths=[len(qq)],actual_seq_lengths_kv=[len(kk)],num_heads=8,
                    num_key_value_heads=1,scale=scale)[0].reshape_as(qq)
            keys=k[ki:ki+nk].contiguous();vals=v[ki:ki+nk].contiguous()
            result=fia(query,keys,vals) if absolute_tile is None else tiled(fia,query,keys,vals,absolute_tile)
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


def attach_vllm(model, layers, *, tiled_layers=()):
    import attention_contract_compute as contract
    text = model.language_model.model
    _validate(layers, text.layers)
    compute = causal_attention()
    fixed=causal_attention(absolute_tile=128) if tiled_layers else None
    restores, counts, active = [], Counter(), []
    baseline = contract.compute

    def consume(q, k, v, ql, kl, **opts):
        if not active:
            return baseline(q, k, v, ql, kl, **opts)
        out = (fixed if active[-1] in tiled_layers else compute)(q, k, v, ql, kl, **opts)
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
        return _receipt(layers, counts, tiled_layers)
    return close


def hf_prefill_rows(q,k,v,kw,use):
    """Run each real sequence on its original absolute tile coordinates."""
    import torch
    batch,heads,n,dim=q.shape
    # For an explicit padding mask Transformers repeats each of the 4 KV
    # heads eight times before SDPA. Recover one identical view per group;
    # its summed VJP flows back to the original shared KV operand.
    if not kw.get('enable_gqa') and kw.get('attn_mask') is not None and k.shape==v.shape==(batch,32,n,dim):
        def shared(x):
            grouped=x.reshape(batch,4,8,n,dim)
            if not torch.equal(grouped,grouped[:,:,:1].expand_as(grouped)):
                raise ValueError('HF explicit KV heads are not repeated groups')
            return grouped[:,:,0]
        k,v=shared(k),shared(v)
        kw=dict(kw,enable_gqa=True)
    if heads!=32 or dim!=128 or k.shape!=(batch,4,n,dim) or v.shape!=k.shape or kw.get('dropout_p',0.)!=0. or not kw.get('enable_gqa'):
        raise ValueError('unexpected HF grouped-query prefill geometry')
    mask=kw.get('attn_mask')
    if mask is None:
        if not kw.get('is_causal'):raise ValueError('unmasked prefill must be causal')
        spans=[(0,n)]*batch
    else:
        # HF SDPA supplies a causal mask including padding. Accept only full
        # causal spans, not arbitrary sparse/sliding/document attention.
        if kw.get('is_causal') or mask.shape!=(batch,1,n,n):raise ValueError('unexpected HF padded causal mask')
        if mask.dtype==torch.bool:allowed=mask
        else:
            if not mask.is_floating_point() or not ((mask==0)|(mask==float('-inf'))|(mask==torch.finfo(mask.dtype).min)).all():
                raise ValueError('unsupported additive attention bias')
            allowed=mask==0
        spans=[]
        for row in allowed[:,0]:
            # True self-edges identify real tokens for the explicit padded causal mask.
            ids=row.diagonal().nonzero().flatten()
            if not len(ids):raise ValueError('empty actor sequence')
            lo,hi=int(ids[0]),int(ids[-1])+1
            if hi-lo!=len(ids):raise ValueError('noncontiguous actor tokens')
            sub=row[lo:hi]
            if (sub[:,:lo].any() or sub[:,hi:].any() or
                not torch.equal(sub[:,lo:hi],torch.ones_like(sub[:,lo:hi]).tril())):
                raise ValueError('not full causal attention on a single recording')
            spans.append((lo,hi))
    rows=[]
    for b,(lo,hi) in enumerate(spans):
        length=hi-lo
        groups=[use(q[b,i*8:(i+1)*8,lo:hi].transpose(0,1),
                    k[b,i:i+1,lo:hi].transpose(0,1),v[b,i:i+1,lo:hi].transpose(0,1),
                    [length],[length],scale=kw.get('scale') or 128**-.5) for i in range(4)]
        rows.append(torch.nn.functional.pad(torch.cat(groups,dim=1).transpose(0,1),(0,0,lo,n-hi)))
    return torch.stack(rows)


def attach_hf(text, layers, unwrap, *, training_vjp=False, tiled_layers=()):
    """Intercept the real HF SDPA call inside the selected decoder attention only."""
    import torch
    from torch.nn import functional as F
    from torch.overrides import TorchFunctionMode

    _validate(layers, text.layers)
    compute = causal_attention(training_vjp=training_vjp)
    fixed=causal_attention(absolute_tile=128,training_vjp=training_vjp) if tiled_layers else None
    counts, handles, active = Counter(), [], []

    class AttentionMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kw = kwargs or {}
            if func is not F.scaled_dot_product_attention:
                return func(*args, **kw)
            if len(args) != 3 or set(kw)-{'attn_mask','is_causal','scale','enable_gqa','dropout_p'}:
                raise ValueError('unexpected HF SDPA call ABI')
            q, k, v = args
            use=fixed if active[-1][0] in tiled_layers else compute
            result=hf_prefill_rows(q,k,v,kw,use)
            counts[active[-1][0]] += 1
            return result

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
        return _receipt(layers, counts, tiled_layers)
    return close


def _receipt(layers, counts, tiled_layers):
    if set(counts) != set(layers):
        raise ValueError('candidate layers were not consumed by the forward')
    return dict(arithmetic='causal_bf16_tnd_fia', layers=dict(counts),absolute_tile_layers=list(tiled_layers),
                training_backward_validated=False)
