"""Use actual frozen router operands. Keep the existing softmax, top-k and weighting."""
from contextlib import ExitStack
from functools import wraps


def attach(text,layers,*,side,unwrap=lambda m:m,training_vjp=False):
    import torch
    from retained_expert_candidate import frozen_project
    from torch.overrides import TorchFunctionMode
    if side not in ('hf','vl') or not layers or layers!=sorted(set(layers)) or any(type(i)!=int or i not in range(48) for i in layers):
        raise ValueError('explicit Omni router layers required')
    stack=ExitStack();calls={}
    def wrap(orig,index):
        class RouterLinear(TorchFunctionMode):
            def __torch_function__(self,func,types,args=(),kwargs=None):
                if func is not torch.nn.functional.linear:return func(*args,**(kwargs or {}))
                if len(args)!=2 or kwargs:raise ValueError('unexpected live router linear ABI')
                x,w=args
                # The locked vLLM wrapper also executes the original FP32
                # projection as a shadow before its one consumed BF16 result.
                if side=='vl' and x.dtype==w.dtype==torch.float32:
                    return func(*args,**(kwargs or {}))
                if (x.ndim!=2 or tuple(w.shape)!=(128,2048) or x.shape[1]!=2048 or x.dtype!=torch.bfloat16
                        or w.dtype!=x.dtype or x.device!=w.device or x.device.type!='npu'
                        or (torch.is_grad_enabled() and not (side=='hf' and training_vjp))):
                    raise ValueError('expected no-grad full-width BF16 router')
                y=frozen_project(x,w.T)
                calls[index]+=1
                return y
        @wraps(orig)
        def apply(*a,**kw):
            before=calls[index]
            with RouterLinear():out=orig(*a,**kw)
            if calls[index]!=before+1:raise ValueError('router must replace exactly one projection')
            return out
        return apply
    try:
        for i in layers:
            mlp=unwrap(unwrap(text.layers[i]).mlp)
            if side=='hf':
                module,key=mlp.gate,'forward'
            else:
                module,=[m for m in mlp.modules() if type(m).__name__=='AscendMoERunner']
                key='_apply_router_linear'
            orig=getattr(module,key);present=key in vars(module)
            if present:stack.callback(setattr,module,key,vars(module)[key])
            else:stack.callback(delattr,module,key)
            calls[i]=0;setattr(module,key,wrap(orig,i))
    except BaseException:
        stack.close();raise
    def close():
        stack.close()
        if not all(calls.values()):raise ValueError('missing router execution')
        return dict(arithmetic='full_width_bf16_router_grouped_matmul',layers=calls)
    return close
