"""Small same-device controls for derivatives needed by the frozen-expert LoRA actor."""


def measure(device, rank, lengths, seed):
    import torch
    import torch_npu
    from retained_expert_candidate import project, frozen_project
    from retained_attention_candidate import causal_attention

    rng = torch.Generator().manual_seed(seed + rank)
    def draw(shape):
        return torch.randn(shape, generator=rng).bfloat16()
    def stats(actual, expected):
        x, y = actual.detach().cpu().double(), expected.detach().cpu().double()
        if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('invalid gradient comparison')
        return dict(n=x.numel(), unequal=int((x!=y).sum()), max_abs=float((x-y).abs().max()),
                    relative_l2=float((x-y).norm()/y.norm().clamp_min(1e-30)))
    def backward(fn, xs, upstream, ref):
        try:
            with torch.enable_grad():
                y = fn(*xs)
                if not y.requires_grad:
                    return dict(available=False,reason='no autograd output')
                grads = torch.autograd.grad(y,xs,upstream.to(device))
            return dict(available=True,gradients=[stats(x,y) for x,y in zip(grads,ref,strict=True)])
        except (RuntimeError, NotImplementedError) as exc:
            return dict(available=False,reason=str(exc)[:1200])
    rows=[]
    for n in lengths:
        if not 1 < n <= 256:
            raise ValueError('bounded actual-operator VJP required')
        for left,right in ((2048,1536),(768,2048)):
            x,w,dy = draw((n,left)),draw((left,right)),draw((n,right))
            arg=x.to(device).requires_grad_(); weight=w.to(device)
            # Expert weights are frozen; hidden input is the path to earlier LoRA.
            ref=dy.double()@w.double().T
            rows.append(dict(operator='grouped',shape=[n,left,right],
                    direct=backward(lambda v:project(v,weight),(arg,),dy,(ref,)),
                    repaired=backward(lambda v:frozen_project(v,weight),(arg,),dy,(ref,)),
                    forward_delta=stats(frozen_project(arg,weight),project(arg.detach(),weight)),
                    reference_rounding=stats(ref.bfloat16(),ref)))
        q,k,v=(draw((n,h,128)) for h in (8,1,1))
        dy=draw(q.shape)
        cq,ck,cv=(x.double().requires_grad_() for x in (q,k,v))
        scores=torch.einsum('thd,shd->hts',cq,ck.expand(-1,8,-1))*128**-.5
        mask=torch.ones(n,n,dtype=torch.bool).triu_(1)
        out=torch.einsum('hts,shd->thd',scores.masked_fill(mask,-torch.inf).softmax(-1),cv.expand(-1,8,-1))
        ref=torch.autograd.grad(out,(cq,ck,cv),dy.double())
        xs=tuple(x.to(device).requires_grad_() for x in (q,k,v))
        m=torch.ones((2048,2048),dtype=torch.bool,device=device).triu_(1)
        def train(q,k,v):
            return torch_npu.npu_fusion_attention(q,k,v,8,'TND',atten_mask=m,scale=128**-.5,
                       actual_seq_qlen=[n],actual_seq_kvlen=[n],sparse_mode=3,keep_prob=1.)[0]
        with torch.no_grad():
            inference=causal_attention()(*xs,[n],[n],scale=128**-.5)
            training=train(*xs)
        reference_forward = causal_attention(training_vjp=True)
        def bridge(q,k,v):
            return reference_forward(q,k,v,[n],[n],scale=128**-.5)
        rows.append(dict(operator='attention',length=n,train_vs_inference=stats(training,inference),
                         bridge_forward=stats(bridge(*xs),inference),
                         bridge_vjp=backward(bridge,xs,dy,ref),
                         train_vjp=backward(train,xs,dy,ref),
                         reference_rounding=[stats(t.bfloat16(),t) for t in ref]))
    return dict(rank=rank,seed=seed,rows=rows,optimizer_steps=0)
