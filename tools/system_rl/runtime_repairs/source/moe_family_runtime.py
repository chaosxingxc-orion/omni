"""One shared MoE hook set, scoped by selected live layer execution.

Uses the proven TP gate/down, activation and grouped-combine primitives. Global
implementation hooks are installed once, not once per decoder layer.
"""
import hashlib
import inspect
from pathlib import Path


def require_shared_counts(counts,device,gather):
    """Agree on TP4 communication branches before omitting empty chunks."""
    import torch
    local=torch.tensor(counts,dtype=torch.int64,device=device)
    full=gather(local,dim=0)
    if full.numel()!=4*len(counts):raise ValueError('TP4 expert count agreement shape')
    if not bool((full.reshape(4,len(counts))==local[None]).all()):
        raise ValueError('TP expert counts disagree')


def require_no_active_expert_lora(context):
    """Loaded slots are not batch activity; obey the live Punica row mapping."""
    import torch
    if context is None or context.punica_wrapper.no_lora:
        return
    ids=context.punica_wrapper.token_lora_indices
    enabled=context.adapter_enabled
    if ids.ndim!=1 or ids.dtype not in (torch.int32,torch.int64) or enabled.ndim!=1:
        raise ValueError('explicit live expert adapter mapping required')
    if bool(((ids < -1) | (ids >= context.max_loras)).any()):
        raise ValueError('expert adapter mapping out of range')
    active=ids[ids>=0].long()
    if active.numel() and bool(torch.count_nonzero(enabled[active])):
        raise ValueError('expert LoRA not enabled in qkvo baseline')


def attach(mlps,*,source_pins,preserve_original=True,expert_chunk=4,
           down_weight_layout='contiguous',skip_empty_chunks=False,packed_active_limit=0):
    import torch
    import vllm_ascend.ops
    from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod as Method
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather as Dispatcher
    from vllm_ascend.ops.fused_moe import moe_mlp
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.distributed import get_tp_group,tensor_model_parallel_all_gather
    from expert_gate_repair import project as gate,dispatch_order
    from expert_down_projection import project as down
    from expert_arithmetic import activation
    from expert_grouped_combine import combine
    if type(packed_active_limit) is not int or not 0 <= packed_active_limit <= 32 or (packed_active_limit and not skip_empty_chunks):
        raise ValueError('bounded packed experts require TP count agreement')
    if expert_chunk!=4:raise ValueError('validated four-expert transport chunk required')
    if type(skip_empty_chunks) is not bool:raise ValueError('boolean empty expert chunk policy required')
    if down_weight_layout not in ('contiguous','hf_transposed'):raise ValueError('unknown down layout')
    targets=[(Method,'apply_gmm1','expert'),(Method,'apply_gmm2','expert'),
             (moe_mlp,'_unified_apply_activation','activation'),(Dispatcher,'token_combine','combine')]
    for obj,name,pin in targets:
        fn=getattr(obj,name)
        if hashlib.sha256(Path(inspect.getfile(fn)).read_bytes()).hexdigest()!=source_pins[pin]:raise ValueError('MoE source changed: '+name)
        if getattr(fn,'_omni_family_owned',False):raise ValueError('MoE family already installed')
    state={};undo=[];audit=dict(layers={},restored=False,production_acceptance=False,
                              down_weight_layout=down_weight_layout,skip_empty_chunks=skip_empty_chunks)
    def install(obj,name,wrapper):
        present=name in vars(obj);raw=vars(obj).get(name);setattr(wrapper,'_omni_family_owned',True)
        setattr(obj,name,wrapper);undo.append((obj,name,present,raw,wrapper))
    def restore():
        if audit['restored']:return
        for obj,name,present,raw,wrapper in reversed(undo):
            if vars(obj).get(name) is not wrapper:raise ValueError('MoE wrapper owner changed: '+name)
            if present:setattr(obj,name,raw)
            else:delattr(obj,name)
        audit['restored']=True
    def contract(method,m):
        group=get_tp_group()
        if torch.is_grad_enabled() or group.world_size!=4 or method.moe.has_bias or m.need_trans or m.topk_scales is not None:
            raise ValueError('inference TP4 unquantized bias-free expert contract')
        if m.hidden_states.dtype!=torch.bfloat16:raise ValueError('BF16 experts required')
        require_no_active_expert_lora(m.lora_context)
        raw=m.group_list.cpu().long().tolist()
        if m.group_list_type not in (0,1):raise ValueError('unknown group offsets')
        counts=[v-(raw[i-1] if i else 0) for i,v in enumerate(raw)] if m.group_list_type==0 else raw
        if len(counts)!=128 or min(counts)<0 or sum(counts)!=len(m.hidden_states):raise ValueError('complete128 expert groups')
        if skip_empty_chunks:
            if state['phase']=='gate':
                # All TP ranks must skip identical collectives. Verify before
                # the first data-dependent branch, on every invocation.
                require_shared_counts(counts,m.hidden_states.device,tensor_model_parallel_all_gather)
                state['counts']=counts
            elif counts!=state.get('counts'):
                raise ValueError('expert counts changed between gate and down')
        return group,counts,dispatch_order(m.topk_ids,m.expanded_row_idx)
    def get_weight(method,m,index):
        weight=method.get_mlp_weights(m.layer)[index]
        if isinstance(weight,list):
            if len(weight)!=1:raise ValueError('one packed expert weight')
            weight=weight[0]
        return weight
    def select_weight(method,m,index,counts):
        weight=get_weight(method,m,index)
        active=[i for i,n in enumerate(counts) if n]
        if not packed_active_limit or not 0 < len(active) <= packed_active_limit:
            return weight,counts,expert_chunk
        # Drop only groups of zero rows. CPU counts have already been
        # checked on all ranks; dispatched rows retain their original order.
        idx=torch.tensor(active,device=weight.device,dtype=torch.long)
        return weight.index_select(0,idx),[counts[i] for i in active],len(active)
    old_gate=Method.apply_gmm1;old_down=Method.apply_gmm2;old_act=moe_mlp._unified_apply_activation;old_combine=Dispatcher.token_combine
    def gate_call(method,mlp_compute_input):
        m=mlp_compute_input
        if not state:return old_gate(method,m)
        group,counts,order=contract(method,m)
        if state['phase']!='gate':raise ValueError('MoE gate order')
        if preserve_original:old_gate(method,m)
        w,used,chunk=select_weight(method,m,0,counts)
        result=gate(m.hidden_states,w,used,group.rank_in_group,tensor_model_parallel_all_gather,
                    expert_chunk=chunk,order=order,skip_empty_chunks=skip_empty_chunks)
        state['phase']='activation';return result
    def act_call(mlp_compute_input,hidden_states,quant_method):
        m,hidden,method=mlp_compute_input,hidden_states,quant_method
        if not state:return old_act(m,hidden,method)
        if state['phase']!='activation' or m.activation!=MoEActivation.SILU or m.swiglu_limit>0 or hidden.shape[-1]!=384:
            raise ValueError('MoE activation order/contract')
        if preserve_original:old_act(m,hidden,method)
        result=activation(hidden);state['phase']='down';return result
    def down_call(method,mlp_compute_input,hidden_states,act_out_scale):
        m,hidden,scale=mlp_compute_input,hidden_states,act_out_scale
        if not state:return old_down(method,m,hidden,scale)
        group,counts,order=contract(method,m)
        if state['phase']!='down' or scale is not None:raise ValueError('MoE down order/unquantized contract')
        if preserve_original:old_down(method,m,hidden,scale)
        w,used,chunk=select_weight(method,m,1,counts)
        result=down(hidden,w,used,tensor_model_parallel_all_gather,
                    expert_chunk=chunk,order=order,weight_layout=down_weight_layout,
                    skip_empty_chunks=skip_empty_chunks)
        state['phase']='combine'
        return result if group.rank_in_group==0 else torch.zeros_like(result)
    def combine_call(dispatcher,hidden_states,combine_metadata,bias=None):
        hidden,metadata=hidden_states,combine_metadata
        if not state:return old_combine(dispatcher,hidden,metadata,bias)
        if state['phase']!='combine' or bias is not None:raise ValueError('MoE combine order/bias')
        if preserve_original:old_combine(dispatcher,hidden,metadata,bias)
        result=combine(hidden,metadata.expanded_row_idx,metadata.topk_weights.to(hidden.dtype))
        if len(metadata.restore_shape)==3:result=result.view(metadata.restore_shape)
        state['phase']='done';return result
    try:
        for obj,name,fn in [(Method,'apply_gmm1',gate_call),(Method,'apply_gmm2',down_call),
                            (moe_mlp,'_unified_apply_activation',act_call),(Dispatcher,'token_combine',combine_call)]:install(obj,name,fn)
        for layer,mlp in mlps.items():
            entry=dict(calls=0);audit['layers'][str(layer)]=entry
            def wrapper(old,entry,layer):
                def forward(*args,**kwargs):
                    if state:raise ValueError('overlapping MoE execution')
                    state.update(layer=layer,phase='gate')
                    try:
                        result=old(*args,**kwargs)
                        if state['phase']!='done':raise ValueError('MoE family not fully consumed')
                        entry['calls']+=1;return result
                    finally:state.clear()
                return forward
            install(mlp,'forward',wrapper(mlp.forward,entry,layer))
    except BaseException:
        restore();raise
    return audit,restore
