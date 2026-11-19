"""Use actual frozen router operands. Keep the existing softmax, top-k and weighting."""
from contextlib import ExitStack
from functools import wraps


def attach(text,layers,*,side,unwrap=lambda m:m):
    import torch
    import torch_npu
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
                if (x.ndim!=2 or tuple(w.shape)!=(128,2048) or x.shape[1]!=2048 or x.dtype!=torch.bfloat16
                        or w.dtype!=x.dtype or x.device!=w.device or x.device.type!='npu' or torch.is_grad_enabled()):
                    raise ValueError('expected no-grad full-width BF16 router')
                y,=torch_npu.npu_grouped_matmul(x=[x.contiguous()],weight=[w.T.contiguous().unsqueeze(0)],
                    split_item=2,group_list_type=1,group_type=0,
                    group_list=torch.tensor([len(x)],device=x.device,dtype=torch.int64))
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
