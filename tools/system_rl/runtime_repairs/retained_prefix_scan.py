"""Locate the earliest changed operation at any original prefill row."""

ORDER = ("input_norm", "qkv_proj", "query", "key", "value", "attention",
         "o_proj", "post_norm", "mlp")


def observe(text, prefix_rows, layers):
    import attention_contract_compute as contract
    if not (0 < prefix_rows <= 8192 and 0 < layers <= 8):
        raise ValueError("bounded prefix scan required")
    captured, handles, restores = {}, [], []
    state = {}

    def save(name, x):
        if not state.get("capture"):
            return
        if name in captured or x.shape[0] < prefix_rows or x.dtype != torch.bfloat16:
            raise ValueError("invalid scan boundary " + name)
        captured[name] = x[:prefix_rows].detach().cpu().contiguous().clone()

    def begin(module, args):
        pos, hidden, _ = args
        axes = pos if pos.ndim == 2 else pos.unsqueeze(0)
        state["capture"] = len(hidden) >= prefix_rows and bool(axes[:, 0].eq(0).all())
        if state["capture"]:
            state["forward_rows"] = len(hidden)
            captured["positions"] = axes[:, :prefix_rows].detach().cpu().contiguous().clone()
            save("decoder.input", hidden)

    handles.append(text.layers[0].register_forward_pre_hook(begin))
    for i in range(layers):
        layer = text.layers[i]
        attn = layer.self_attn
        modules = (("input_norm", layer.input_layernorm), ("qkv_proj", attn.qkv_proj),
                   ("o_proj", attn.o_proj), ("post_norm", layer.post_attention_layernorm),
                   ("mlp", layer.mlp))
        for name, mod in modules:
            def called(m, args, out, key=f"{i}.{name}"):
                save(key, out[0] if isinstance(out, tuple) else out)
            handles.append(mod.register_forward_hook(called))
        # Installed precision adapter calls contract.compute with actual logical
        # KV. This hook leaves the original arithmetic and buffer ownership intact.
        impl = attn.attn.impl
        name = "forward_fused_infer_attention"
        prior = vars(impl).get(name)
        orig = getattr(impl, name)

        def wrapper(orig, i):
            def call(*args, **kwargs):
                if not state.get("capture"):
                    return orig(*args, **kwargs)
                compute = contract.compute

                def hooked(q, k, v, nq, nk, **opts):
                    if nq != nk or len(nq) != 1 or nq[0] != state["forward_rows"]:
                        raise ValueError("prefix scan expects a single uncached prefill")
                    save(f"{i}.query", q)
                    save(f"{i}.key", k)
                    save(f"{i}.value", v)
                    out = compute(q, k, v, nq, nk, **opts)
                    save(f"{i}.attention", out)
                    return out
                contract.compute = hooked
                try:
                    return orig(*args, **kwargs)
                finally:
                    contract.compute = compute
            return call

        fn = wrapper(orig, i)
        setattr(impl, name, fn)
        restores.append((impl, name, prior, fn))

    def close():
        for handle in handles:
            handle.remove()
        for owner, name, prior, wrapper_fn in reversed(restores):
            if vars(owner).get(name) is not wrapper_fn:
                raise ValueError("scan hook owner changed")
            if prior is None:
                delattr(owner, name)
            else:
                setattr(owner, name, prior)
        keys = ["positions", "decoder.input"] + [f"{i}.{name}" for i in range(layers) for name in ORDER]
        if set(keys) != captured.keys():
            raise ValueError("incomplete prefix scan")
        return state["forward_rows"], {k: captured[k] for k in keys}
    return close


def compare(a, b):
    import torch
    na, a = a
    nb, b = b
    if tuple(a) != tuple(b):
        raise ValueError("prefix scan boundaries do not match")
    out = {}
    for name, x in a.items():
        y = b[name]
        if name == "positions":
            if not torch.equal(x, y):
                raise ValueError("different original mRoPE positions")
            continue
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError("prefix scan shapes do not match")
        changed = (x != y).reshape(len(x), -1)
        per_row = changed.sum(-1)
        rows = per_row.nonzero().flatten().tolist()
        out[name] = dict(changed_rows=len(rows), first_rows=rows[:32], unequal=int(per_row.sum()))
    return dict(rows=[na, nb], first=next((k for k,v in out.items() if v["unequal"]), None), boundaries=out)
