"""Paired row arithmetic at configured, full-width LoRA A projection sites."""
from functools import wraps


def validate(spec):
    grouped = spec.get('algorithm') == 'grouped_bf16'
    if type(spec.get('adapter_rank',8)) is not int or spec.get('adapter_rank',8)<=0:
        raise ValueError('invalid adapter rank')
    if set(spec)-{'adapter_rank'} != {'layers','algorithm' if grouped else 'tile_rows','site'} or spec['site'] not in (('o_lora_a','qkv_lora_a','qkvo_lora_a') if grouped else ('o_lora_a',)):
        raise ValueError('unknown LoRA A configuration')
    layers=spec['layers']
    if (not isinstance(layers,list) or not layers or layers!=sorted(set(layers))
            or any(type(i)!=int or not 0<=i<48 for i in layers)
            or (not grouped and (type(spec['tile_rows'])!=int or not 1<=spec['tile_rows']<=4096))):
        raise ValueError('invalid configured O LoRA A row geometry')


def fixed_rows(linear,x,w,tile):
    import torch
    pieces=[]
    for rows in x.split(tile,dim=0):
        n=len(rows)
        if n<tile:
            padded=rows.new_zeros((tile,rows.shape[1]))
            padded[:n].copy_(rows)
        else:padded=rows.contiguous()
        pieces.append(linear(padded,w)[:n])
    return torch.cat(pieces) if len(pieces)>1 else pieces[0]


def attach(text,spec):
    import torch
    from torch.overrides import TorchFunctionMode
    import projection_family_runtime as runtime
    validate(spec)
    from contextlib import ExitStack
    stack=ExitStack()
    tile=spec.get('tile_rows')
    def install(kind,module_name,width,branches):
        original=getattr(runtime,kind)
        active=[]
        class Mode(TorchFunctionMode):
            def __torch_function__(self,func,types,args=(),kwargs=None):
                if func is torch.nn.functional.linear and len(args)>=2 and args[1].shape==(spec.get('adapter_rank',8),width):
                    x,w=args[:2]
                    if (torch.is_grad_enabled() or kwargs or len(args)!=2 or x.ndim!=2
                            or x.dtype!=w.dtype or x.dtype!=torch.bfloat16 or x.device.type!='npu'):
                        raise ValueError('unexpected LoRA A ABI: '+kind)
                    active[-1][1]+=1
                    return grouped_linear(x,w) if tile is None else fixed_rows(func,x,w,tile)
                return func(*args,**(kwargs or {}))
        @wraps(original)
        def bounded(*a,**kw):
            if not active:return original(*a,**kw)
            with Mode():return original(*a,**kw)
        def restore():
            if getattr(runtime,kind) is not bounded or active:raise ValueError('LoRA geometry ownership changed')
            setattr(runtime,kind,original)
        for i in spec['layers']:
            mod=getattr(text.layers[i].self_attn,module_name)
            def enter(m,a,n=i):
                if active:raise ValueError('nested projection: '+kind)
                active.append([n,0])
            def leave(m,a,result):
                _,calls=active.pop()
                if result is not None and calls != branches*m.lora_a_stacked[0].shape[0]:
                    raise ValueError(f'incomplete {kind} LoRA A consumption: {calls}')
            stack.callback(mod.register_forward_pre_hook(enter).remove)
            stack.callback(mod.register_forward_hook(leave,always_call=True).remove)
        setattr(runtime,kind,bounded)
        stack.callback(restore)
    try:
        if spec['site']!='qkv_lora_a':install('row','o_proj',4096,1)
        if spec['site']!='o_lora_a':install('qkv','qkv_proj',2048,3)
    except BaseException:
        stack.close()
        raise
    close=stack.close
    return close


def grouped_linear(x,w):
    import torch
    import torch_npu
    if (x.ndim<2 or w.ndim!=2 or w.shape[0]<=0 or w.shape[1] not in (2048,4096) or x.shape[-1]!=w.shape[1] or x.dtype!=torch.bfloat16
            or w.dtype!=x.dtype or w.device!=x.device or x.device.type!='npu'):
        raise ValueError(f'unexpected grouped O LoRA A operands: x={x.shape}/{x.dtype}/{x.device}, '
                         f'weight={w.shape}/{w.dtype}/{w.device}')
    def project(flat,weight):
        sizes=torch.tensor([len(flat)],device=flat.device,dtype=torch.int64)
        y,=torch_npu.npu_grouped_matmul(x=[flat.contiguous()],weight=[weight.T.contiguous().unsqueeze(0)],
            split_item=2,group_list_type=1,group_type=0,group_list=sizes)
        return y
    class Trainable(torch.autograd.Function):
        @staticmethod
        def forward(ctx,flat,weight):
            ctx.save_for_backward(flat,weight)
            return project(flat,weight)
        @staticmethod
        def backward(ctx,dy):
            flat,weight=ctx.saved_tensors
            return (dy @ weight if ctx.needs_input_grad[0] else None,
                    dy.T @ flat if ctx.needs_input_grad[1] else None)
    flat=x.reshape(-1,w.shape[1])
    y=Trainable.apply(flat,w) if torch.is_grad_enabled() else project(flat,w)
    return y.reshape(*x.shape[:-1],w.shape[0])


def attach_hf(text,spec):
    from collections import Counter
    from contextlib import ExitStack
    import torch
    validate(spec)
    if spec.get('algorithm')!='grouped_bf16':raise ValueError('expected paired grouped HF control')
    def unwrap(mod):
        while type(mod).__name__ in ('CheckpointWrapper','FullyShardedDataParallel'):
            mod=mod.module if type(mod).__name__=='FullyShardedDataParallel' else mod._checkpoint_wrapped_module
        return mod
    sites={'o_lora_a':'o','qkv_lora_a':'qkv','qkvo_lora_a':'qkvo'}[spec['site']]
    counts=Counter();operand_modes=Counter();stack=ExitStack()
    try:
        for i in spec['layers']:
            attn=unwrap(unwrap(text.layers[i]).self_attn)
            for site in sites:
                leaf=unwrap(unwrap(getattr(attn,site+'_proj')).lora_A['default'])
                if not isinstance(leaf,torch.nn.Linear) or leaf.bias is not None or leaf.out_features!=spec.get('adapter_rank',8):
                    raise ValueError('unexpected PEFT O LoRA A leaf')
                if 'forward' in vars(leaf):stack.callback(setattr,leaf,'forward',leaf.forward)
                else:stack.callback(delattr,leaf,'forward')
                def forward(x,n=f'{i}.{site}',leaf=leaf):
                    weight=leaf.weight
                    device=x.device.type
                    autocast=torch.is_autocast_enabled(device)
                    mode=(str(x.dtype),str(weight.dtype),autocast)
                    if autocast:
                        # PEFT casts x to the adapter dtype before native linear
                        # dispatches under actor autocast. Preserve that boundary.
                        if (torch.get_autocast_dtype(device)!=torch.bfloat16 or
                                any(t.dtype not in (torch.float32,torch.bfloat16) for t in (x,weight))):
                            raise ValueError(f'unsupported O adapter autocast: {mode}, target={torch.get_autocast_dtype(device)}')
                        with torch.autocast(device_type=device,enabled=False):
                            y=grouped_linear(x.bfloat16(),weight.bfloat16())
                    else:
                        y=grouped_linear(x,weight)
                    counts[n]+=1
                    operand_modes[mode]+=1
                    return y
                leaf.forward=forward
    except BaseException:
        stack.close()
        raise
    def close(*,check_coverage=True):
        stack.close()
        if check_coverage and set(counts)!={f'{i}.{s}' for i in spec['layers'] for s in sites}:raise ValueError('incomplete HF LoRA A consumption')
        return dict(algorithm=spec['algorithm'],calls_by_layer_and_branch=dict(counts),operand_modes=[
            dict(input_dtype=x,weight_dtype=w,autocast=active,calls=n)
            for (x,w,active),n in operand_modes.items()])
    return close
