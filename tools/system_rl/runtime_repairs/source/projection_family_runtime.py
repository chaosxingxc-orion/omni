"""BF16 TP4 QKV/O projection contracts using live shards and adapter mappings.

Full batch M is retained for each adapter; gathering masked rows would change
GEMM shape and can change BF16 results. Inactive rows receive no adapter delta.
"""
import hashlib
import inspect
from pathlib import Path


def _require(ok,message):
    if not ok:raise ValueError(message)


def _inputs(x,weight,rank):
    import torch
    _require(not torch.is_grad_enabled(),'inference projection only')
    _require(type(rank) is int and rank in range(4),'TP4 rank required')
    _require(x.ndim==weight.ndim==2 and x.dtype==weight.dtype==torch.bfloat16 and x.device==weight.device,
             'BF16 input and weight on same device')
    _require(not torch.is_autocast_enabled(x.device.type),'explicit BF16 arithmetic required')


def _slots(x,aa,bb,indices,scales):
    import torch,math
    _require(len(aa)==len(bb) and len(aa)>0,'paired adapter branches')
    slots=aa[0].shape[0]
    _require(len(scales)==slots and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>0 for v in scales),
             'positive explicit scale per adapter slot')
    for a,b in zip(aa,bb):
        _require(a.ndim==b.ndim==4 and a.shape[:2]==b.shape[:2]==(slots,1)
                 and a.shape[2]==b.shape[3] and a.shape[2]>=8,'rank8 or padded adapter layout')
        _require(a.dtype==b.dtype==torch.float32 and a.device==b.device==x.device,'FP32 stored adapters on input device')
        if a.shape[2]>8:
            _require(not bool(torch.count_nonzero(a[:,:,8:,:])) and
                     not bool(torch.count_nonzero(b[:,:,:,8:])),
                     'nonzero rank padding; only actual rank8 adapters supported')
    _require(indices.ndim==1 and indices.numel()==len(x) and indices.dtype in (torch.int32,torch.int64)
             and indices.device==x.device,'one integer adapter index per row')
    _require(bool(((indices>=-1)&(indices<slots)).all()),'adapter index out of range')
    return slots


def _rank8(aa,bb):
    # Preserve the validated rank8 GEMM dimensions despite server capacity16.
    return [a[:,:,:8,:].contiguous() for a in aa],[b[:,:,:,:8].contiguous() for b in bb]


def _gather(t,dim,rank,gather):
    import torch
    full=gather(t.contiguous(),dim=dim)
    shape=list(t.shape);shape[dim]*=4
    _require(list(full.shape)==shape and full.dtype==t.dtype and full.device==t.device,'TP gather shape/dtype/device')
    _require(torch.equal(full.narrow(dim,rank*t.shape[dim],t.shape[dim]),t),'TP gather rank ordering')
    return full


def qkv(x,weight,aa,bb,indices,scales,*,rank,gather):
    import torch
    from torch.nn.functional import linear
    _inputs(x,weight,rank)
    _require(x.shape[1]==2048 and weight.shape==(1280,2048),'packed Omni QKV layout')
    _require(len(aa)==len(bb)==3,'three QKV adapter branches')
    slots=_slots(x,aa,bb,indices,scales);aa,bb=_rank8(aa,bb);out=[];offset=0
    for width,a,b in zip((1024,128,128),aa,bb):
        _require(a.shape[3]==2048 and b.shape[2]==width,'QKV adapter dimensions')
        full_weight=_gather(weight[offset:offset+width],0,rank,gather)
        base=linear(x,full_weight)[:,rank*width:(rank+1)*width].contiguous()
        # Full B output width avoids introducing another column-shard GEMM.
        full_b=_gather(b,2,rank,gather)
        for slot in range(slots):
            mask=(indices==slot)[:,None]
            u=linear(x,a[slot,0].bfloat16())
            z=linear(u,(full_b[slot,0]/scales[slot]).bfloat16())
            delta=z[:,rank*width:(rank+1)*width]*scales[slot]
            base=torch.where(mask,base+delta,base)
        out.append(base);offset+=width
    return torch.cat(out,-1)


def row(x,weight,aa,bb,indices,scales,*,rank,gather):
    import torch
    from torch.nn.functional import linear
    _inputs(x,weight,rank)
    _require(x.shape[1]==1024 and weight.shape==(2048,1024),'Omni TP4 row layout')
    _require(len(aa)==len(bb)==1,'one row adapter branch')
    slots=_slots(x,aa,bb,indices,scales);aa,bb=_rank8(aa,bb);a,b=aa[0],bb[0]
    _require(a.shape[3]==1024 and b.shape[2]==2048,'row adapter dimensions')
    full_x=_gather(x,1,rank,gather);full_w=_gather(weight,1,rank,gather);full_a=_gather(a,3,rank,gather)
    result=linear(full_x,full_w)
    for slot in range(slots):
        u=linear(full_x,full_a[slot,0].bfloat16())
        z=linear(u,(b[slot,0]/scales[slot]).bfloat16())
        result=torch.where((indices==slot)[:,None],result+z*scales[slot],result)
    return result


def attach(module,*,kind,source_sha256,adapter_scales,preserve_original=True):
    """Explicit synchronous adapter ABI; source and per-slot scales are configured."""
    import torch
    if kind not in ('qkv','row'):raise ValueError('known projection kind required')
    name='_apply_lora_to_output' if kind=='qkv' else 'forward'
    old=getattr(module,name);source=Path(inspect.getfile(getattr(old,'__func__',old)))
    _require(hashlib.sha256(source.read_bytes()).hexdigest()==source_sha256,'projection source identity changed')
    _require(module.tp_size==4 and not module._enable_aux_cuda_stream,'synchronous TP4 required')
    _require(getattr(module.base_layer,'bias',None) is None,'bias-free projection required')
    present=name in vars(module);raw=vars(module).get(name)
    audit=dict(calls=0,original_calls=0,restored=False,kind=kind,production_acceptance=False)
    def operands(x):
        from vllm.distributed import get_tp_group,tensor_model_parallel_all_gather
        group=get_tp_group();_require(group.world_size==4 and group.rank_in_group==module.tp_rank,'same live TP group')
        p=module.punica_wrapper
        if p.no_lora:ids=torch.full((len(x),),-1,dtype=torch.int64,device=x.device)
        else:
            _require(p._token_lora_indices.numel()>=len(x),'complete adapter mapping')
            ids=p._token_lora_indices[:len(x)]
        return ids,tensor_model_parallel_all_gather
    def forward(x,*args,**kwargs):
        _require(not kwargs,'known synchronous projection ABI')
        ids,gather=operands(x)
        if kind=='qkv':
            _require(len(args)==1 and args[0].shape==(len(x),1280),'base-to-adapter QKV ABI')
            result=qkv(x,module.base_layer.weight,module.lora_a_stacked,module.lora_b_stacked,ids,adapter_scales,
                       rank=module.tp_rank,gather=gather)
            if preserve_original:
                control=old(x,args[0]);_require(isinstance(control,torch.Tensor) and control.shape==result.shape,'QKV return ABI')
                audit['original_calls']+=1
            args[0].copy_(result);answer=args[0]
        else:
            _require(not args and module.base_layer.input_is_parallel and module.base_layer.reduce_results
                     and module.base_layer.return_bias,'row reduction/return ABI')
            result=row(x,module.base_layer.weight,module.lora_a_stacked,module.lora_b_stacked,ids,adapter_scales,
                       rank=module.tp_rank,gather=gather)
            if preserve_original:
                control=old(x);_require(isinstance(control,tuple) and len(control)==2 and control[1] is None,'row return ABI')
                audit['original_calls']+=1
            answer=result,None
        audit['calls']+=1
        return answer
    setattr(module,name,forward)
    def restore():
        if audit['restored']:return
        _require(vars(module).get(name) is forward,'projection adapter ownership changed')
        if present:setattr(module,name,raw)
        else:delattr(module,name)
        audit['restored']=True
    return audit,restore
