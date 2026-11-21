"""Use full-width Q/K/V/O operands prepared at the verified adapter boundary."""
from functools import wraps


def attach(worker, family):
    import torch
    import torch_npu
    import projection_family_runtime as runtime
    from local_activation import is_decode
    from vllm.distributed import tensor_model_parallel_all_gather as gather

    views, bases = worker._frozen_projection_cache
    rank = worker.rank
    scales = family['adapter_scales']
    if len(scales) != 1 or scales[0] <= 0:
        raise ValueError('one active adapter slot with a positive scale required')
    scale = scales[0]
    text = worker.model_runner.get_model().language_model.model
    modules = {kind: [getattr(layer.self_attn, attr) for layer in text.layers]
               for kind, attr in [('qkv','qkv_proj'), ('row','o_proj')]}
    operands = {}
    sizes = {}
    policy = None

    def make_ready(receipt):
        nonlocal policy, operands
        identity = receipt['adapter_id'], receipt['sha256'], receipt['active_slot']
        if policy == identity:
            return
        if receipt['active_slot'] != 0 or receipt['compared_original_keys'] != 384:
            raise ValueError('expected a verified Q/K/V/O adapter in slot zero')
        ready = {}
        with torch.inference_mode():
            for kind, group in modules.items():
                for i, module in enumerate(group):
                    a, b = runtime._rank8(module.lora_a_stacked, module.lora_b_stacked)
                    pieces = []
                    for j, (src, dest) in enumerate(zip(a, b, strict=True)):
                        view = views[i*4+(3 if kind=='row' else j)][0]
                        key = (view.data_ptr(), tuple(view.shape), view.stride(), view.dtype,
                               view.device, 1 if kind=='row' else 0)
                        if kind=='row':
                            src = gather(src.contiguous(),dim=3)
                        else:
                            dest = gather(dest.contiguous(),dim=2)
                        pieces.append((bases[key].T.contiguous().unsqueeze(0),
                                       src[0,0].bfloat16(), (dest[0,0]/scale).bfloat16()))
                    ready[module.base_layer.weight.data_ptr()] = pieces
        if len(ready) != 96:
            raise ValueError('incomplete Q/K/V/O operand preparation')
        if hasattr(worker,'_clear_decode_graphs'):
            worker._clear_decode_graphs()
        operands, policy = ready, identity

    def wrap(kind, old):
        @wraps(old)
        def forward(x, weight, aa, bb, indices, scales, *, rank, gather):
            if not is_decode(len(x)):
                return old(x, weight, aa, bb, indices, scales, rank=rank, gather=gather)
            if policy is None or x.dtype != torch.bfloat16 or torch.is_grad_enabled():
                raise ValueError('decode requires a verified live BF16 policy')
            pieces = operands[weight.data_ptr()]
            if len(x) not in sizes:
                sizes[len(x)] = torch.tensor([len(x)],dtype=torch.long,device=x.device)
            if kind=='row':
                x = gather(x.contiguous(),dim=1)
            mask = (indices==0)[:,None]
            outputs = []
            for packed, a, b in pieces:
                base, = torch_npu.npu_grouped_matmul(x=[x.contiguous()],weight=[packed],
                    split_item=2,group_list_type=1,group_type=0,group_list=sizes[len(x)])
                # The installed LoRA geometry intercepts A and checks its call count.
                u = torch.nn.functional.linear(x,a)
                delta = torch.nn.functional.linear(u,b)
                if kind=='qkv':
                    width=base.shape[-1]//4
                    base=base[:,rank*width:(rank+1)*width]
                    delta=delta[:,rank*width:(rank+1)*width]
                outputs.append(torch.where(mask,base+delta*scale,base))
            return outputs[0] if kind=='row' else torch.cat(outputs,dim=-1)
        return forward
    for kind in modules:
        setattr(runtime,kind,wrap(kind,getattr(runtime,kind)))
    worker._current_projection_operands=lambda: (policy,operands)
    worker._decode_projection_scale=scale
    worker._projection_adapter_verified = make_ready
