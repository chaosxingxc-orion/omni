"""Observe live route operands for one history row using the existing MoE window."""


class Window:
    def __init__(self):
        self.audit, self.rows, self.undo = {}, {}, []

    def pre(self, module, fn):
        self.undo.append(module.register_forward_pre_hook(fn, with_kwargs=True).remove)

    def post(self, module, fn):
        self.undo.append(module.register_forward_hook(fn, with_kwargs=True).remove)

    def method(self, owner, name, factory):
        had = name in vars(owner)
        previous = vars(owner).get(name)
        wrapped = factory(getattr(owner, name))
        setattr(owner, name, wrapped)
        def undo():
            if vars(owner).get(name) is not wrapped:
                raise ValueError('observer ownership changed: '+name)
            if had:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self.undo.append(undo)

    def tensor(self, key, value):
        if key in self.rows:
            raise ValueError('duplicate live route observation '+key)
        self.rows[key] = value.detach().cpu().contiguous().clone()


def hook(text, position, expert_operands=False, layer_index=0):
    import torch
    from expert_window_vllm_capture import attach, validate
    if type(layer_index) is not int or not 0 <= layer_index < len(text.layers):
        raise ValueError('invalid MoE observation layer')
    mlp = text.layers[layer_index].mlp
    experts = [m for m in mlp.modules() if type(m).__name__=='AscendRoutedExperts']
    if len(experts)!=1:
        raise ValueError('expected a unique routed expert module')
    router = experts[0].router
    snapshot = {}
    def on_entry(module,args):
        pos, hidden, _ = args
        axes=pos if pos.ndim==2 else pos.unsqueeze(0)
        match=torch.where(axes.eq(position).all(0))[0].tolist()
        if not match:
            return
        if len(match)!=1 or snapshot:
            raise ValueError('ambiguous MoE observation')
        row = match[0]
        window = Window()
        operands = None
        if expert_operands:
            from retained_expert_operands import ExpertOperands
            operands = ExpertOperands(window, ('down',) if expert_operands=='down' else ('gate','down'))
        attach(window,text,layer=layer_index,rows=[row],tokens=len(hidden),stages=True,
               dispatcher_class=True,operands=operands)
        def route(old):
            def run(hidden_states,router_logits,indices_type,**kw):
                logits=router_logits[row].detach().clone()
                weights,ids=old(hidden_states,router_logits,indices_type,**kw)
                window.tensor('router.logits',logits)
                window.tensor('router.ids',ids[row])
                window.tensor('router.weights',weights[row])
                return weights,ids
            return run
        window.method(router,'_compute_routing',route)
        snapshot.update(window=window,rows=len(hidden),device=hidden.device,probe=operands)
    def after(module,args,output):
        window=snapshot.get('window')
        if not window:
            return
        validate(window)
        for restore in reversed(window.undo):
            restore()
        snapshot.pop('window')
        snapshot['capture']=window.rows
        probe = snapshot.pop('probe')
        if probe:
            snapshot['operands'] = probe.close()
    # Layer prehook is before any MoE forward hooks. Close after the whole block
    # so the previously registered module hooks have all completed.
    layer=text.layers[layer_index]
    pre=layer.register_forward_pre_hook(on_entry)
    post=layer.register_forward_hook(after)
    def close():
        pre.remove()
        post.remove()
        if 'capture' not in snapshot:
            raise ValueError('MoE window did not execute')
        return snapshot
    return close


def compare(a,b,rank):
    import torch
    order=('router.logits','router.ids','router.weights','gate_up.input','gate_up.output',
           'activation.input','activation.output','down.input','down.output',
           'combine.weights','combine.input','combine.output')
    left=a['capture']
    right=b['capture']
    report={}
    for name in order:
        key = name if name.startswith('router.') else 'expert_window.'+name
        x,y=left[key],right[key]
        if x.shape!=y.shape or x.dtype!=y.dtype or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('incompatible expert stage '+name)
        report[name]=dict(unequal=int((x!=y).sum()),max_abs=float((x.float()-y.float()).abs().max()))
    # The installed overlay sends full-width down/combine values only from
    # rank zero. Other ranks' zero tensors are not evidence of those values.
    result = dict(rows=[a['rows'],b['rows']],rank=rank,stages=report,
                  first_observation=next((k for k,v in report.items() if v['unequal']),None))
    if 'operands' in a:
        from retained_expert_operands import compare as compare_operands
        result['expert_operands'] = compare_operands(a, b, a['device'], controls=(rank==0))
    return result
