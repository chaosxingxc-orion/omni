"""Paired frozen base projection; keep real TP gathers and LoRA math."""
from collections import Counter
from contextlib import ExitStack
from functools import wraps


def attach(text,layers,*,side,unwrap=lambda m:m,training_vjp=False, projections=('k_proj','v_proj')):
    import torch
    from torch.overrides import TorchFunctionMode
    from retained_expert_candidate import project, frozen_project
    if side not in ('vl','hf') or not layers or layers!=sorted(set(layers)) or any(type(i)!=int or not 0<=i<48 for i in layers):
        raise ValueError('expected explicit KV projection layers')
    shapes={'q_proj':(4096,2048),'k_proj':(512,2048),'v_proj':(512,2048),'o_proj':(2048,4096)}
    if projections not in (('q_proj',),('k_proj','v_proj'),('q_proj','k_proj','v_proj'),('o_proj',)):
        raise ValueError('expected one Q or O, or the complete K/V or Q/K/V projection scope')
    selected={shapes[p] for p in projections}
    row=projections==('o_proj',)
    scopes=[];counts=Counter();stack=ExitStack()

    class BaseKV(TorchFunctionMode):
        def __torch_function__(self,func,types,args=(),kwargs=None):
            if func is not torch.nn.functional.linear or len(args)<2 or tuple(args[1].shape) not in selected:
                return func(*args,**(kwargs or {}))
            x,w=args[:2]
            if (kwargs or (len(args)!=2 and (len(args)!=3 or args[2] is not None))
                    or (torch.is_grad_enabled() and not (side=="hf" and training_vjp)) or x.dtype!=torch.bfloat16 or w.dtype!=x.dtype or x.shape[-1]!=w.shape[1]):
                raise ValueError('invalid frozen full-width base projection')
            n=scopes[-1]
            y=(frozen_project if training_vjp else project)(x.reshape(-1,w.shape[1]),w.T)
            counts[n]+=1
            return y.reshape(*x.shape[:-1],w.shape[0])
    def replace(owner,attr,fn):
        before=vars(owner).get(attr)
        if attr in vars(owner):stack.callback(setattr,owner,attr,before)
        else:stack.callback(delattr,owner,attr)
        setattr(owner,attr,fn)
    try:
        if side=='vl':
            # The legacy backend still makes a shadow call. Replace only the
            # adapter's consumed full-width projection, not the shadow.
            import projection_family_runtime as runtime
            name='row' if row else 'qkv'
            original=getattr(runtime,name)
            @wraps(original)
            def paired(*args,**kwargs):
                if not scopes:return original(*args,**kwargs)
                before=counts[scopes[-1]]
                with BaseKV():out=original(*args,**kwargs)
                if counts[scopes[-1]]!=before+len(projections):raise ValueError('incomplete consumed Q/K/V projections')
                return out
            replace(runtime,name,paired)
        for i in layers:
            attn=unwrap(unwrap(text.layers[i]).self_attn)
            if side=='vl':
                mod=attn.o_proj if row else attn.qkv_proj
                stack.callback(mod.register_forward_pre_hook(lambda m,a,n=i:scopes.append(n)).remove)
                stack.callback(mod.register_forward_hook(lambda m,a,v:leave(),always_call=True).remove)
            else:
                for name in projections:
                    mod=unwrap(getattr(attn,name))
                    base=unwrap(mod.get_base_layer() if hasattr(mod,'get_base_layer') else mod)
                    def wrap(old,n):
                        @wraps(old)
                        def call(*args,**kwargs):
                            scopes.append(n);before=counts[n]
                            try:
                                with BaseKV():out=old(*args,**kwargs)
                                if counts[n]!=before+1:raise ValueError('expected one consumed HF base projection')
                                return out
                            finally:scopes.pop()
                        return call
                    replace(base,'forward',wrap(base.forward,i))
    except BaseException:
        stack.close();raise
    def leave():
        scopes.pop()
    def close():
        stack.close()
        if scopes or set(counts)!=set(layers) or any(n%len(projections) for n in counts.values()):
            raise ValueError('incomplete paired K/V execution')
        return dict(arithmetic='full_width_bf16_base_kv_grouped' if projections==('k_proj','v_proj') else ('full_width_bf16_base_o_grouped' if row else 'full_width_bf16_base_qkv_grouped'),layers=dict(counts))
    return close
