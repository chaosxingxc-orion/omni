"""Full-width frozen experts on their owner ranks during pure decode."""
from functools import wraps


def validate(settings):
    if set(settings)-{'static_routes'} != {'reserve_bytes', 'budget_bytes', 'transfer_experts'}:
        raise ValueError('explicit expert bank capacity and packing chunk required')
    if type(settings.get('static_routes',False)) is not bool or any(type(v) is not int or v <= 0 for k,v in settings.items() if k!='static_routes'):
        raise ValueError('positive integer expert bank limits required')
    if 32 % settings['transfer_experts']:
        raise ValueError('expert transfer chunk must divide the 32 experts on each owner')


def split_weights(packed, down):
    # Rank-major [gate_shard, up_shard] -> full gate followed by full up.
    size = packed.shape[0]
    gate_up = packed.reshape(size, 2048, 4, 2, 192).permute(0, 1, 3, 2, 4).reshape(size, 2048, 1536)
    return gate_up, down


def grouped(x, weights, counts):
    import torch_npu
    result, = torch_npu.npu_grouped_matmul(
        x=[x.contiguous()], weight=[weights], group_list=counts,
        split_item=2, group_list_type=1, group_type=0)
    return result


def local_routes(x, ids, weights, gate_up, down, rank, reduce):
    import torch
    # Use the same source-row order inside each expert as the precision path.
    expert, order = torch.sort(ids.reshape(-1))
    low = 32 * rank
    belongs = (expert >= low) & (expert < low + 32)
    route = order[belongs]
    supplied = x.new_zeros((order.numel(), 2048))
    if route.numel():
        local = (expert[belongs] - low).long()
        counts = torch.bincount(local, minlength=32).long()
        first, second = grouped(x[route // 8], gate_up, counts).chunk(2, dim=-1)
        result = grouped(torch.nn.functional.silu(first) * second, down, counts)
        supplied.index_copy_(0, route.long(), result)
    global_rows = reduce(supplied)
    return (global_rows.reshape(len(x), 8, 2048) * weights[:, :, None]).sum(dim=1)


def static_routes(x, ids, weights, gate_up, down, rank, reduce, slots):
    """Route non-owned entries to a separate dummy expert, preserving real groups."""
    import torch
    flat = ids.flatten()
    owner = (flat >= 32*rank) & (flat < 32*(rank+1))
    local = torch.where(owner, flat - 32*rank, 32)
    # Small integer IDs are exactly representable in FP32. Sort output slots
    # with the routes, so token/expert pairs stay associated after permutation.
    order = torch.argsort(local.float())
    counts = (local[:,None] == slots).sum(dim=0, dtype=torch.int64)
    g,u = grouped(x[order//8],gate_up,counts).chunk(2,-1)
    y = grouped(torch.nn.functional.silu(g)*u,down,counts)
    routed = torch.empty_like(y).index_copy_(0,order,y)
    rows = reduce(torch.where(owner[:,None],routed,0))
    return (rows.reshape(len(x),8,2048)*weights[:,:,None]).sum(dim=1)


def attach(worker, config):
    import torch
    from local_activation import is_decode
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather as gather
    from vllm.distributed import tensor_model_parallel_all_reduce as reduce
    from router_arithmetic import select
    from moe_family_runtime import require_no_active_expert_lora

    validate(config)
    text = worker.model_runner.get_model().language_model.model
    tp = get_tp_group()
    rank = tp.rank_in_group
    if len(text.layers) != 48 or tp.world_size != 4 or rank != worker.rank:
        raise ValueError('full-owner expert bank requires expected 48-layer TP4 model')
    chunk = config['transfer_experts']
    per_expert = 2 * (2048 * 1536 + 768 * 2048)
    static = config.get('static_routes',False)
    capacity = 33 if static else 32
    required = len(text.layers) * capacity * per_expert
    scratch = chunk * per_expert * 4
    free, _ = torch.npu.mem_get_info()
    acceptable = required <= config['budget_bytes'] and free >= required + scratch + config['reserve_bytes']
    device = next(text.parameters()).device
    if not bool(gather(torch.tensor([int(acceptable)], device=device), dim=0).all()):
        raise ValueError('insufficient expert bank budget or free HBM on at least one rank')
    prepared = []
    with torch.inference_mode():
        for layer in text.layers:
            mlp = layer.mlp
            if mlp.shared_expert is not None or mlp.enable_eplb or mlp.is_sequence_parallel:
                raise ValueError('unexpected additional experts or sequence sharding')
            expert, = [m for m in mlp.modules() if type(m).__name__ == 'AscendRoutedExperts']
            runner, = [m for m in mlp.modules() if type(m).__name__ == 'AscendMoERunner']
            wrappers = [m for m in mlp.modules() if callable(getattr(type(m), '_build_lora_context', None))]
            if len(wrappers)>1 or not wrappers and getattr(expert.quant_method,'lora_context',None) is not None:
                raise ValueError('unexpected expert LoRA context')
            a, b = expert.w13_weight, expert.w2_weight
            if a.shape != (128,2048,384) or b.shape != (128,192,2048) or a.dtype != torch.bfloat16 or b.dtype != a.dtype:
                raise ValueError('unexpected live TP4 BF16 expert shard layout')
            gu = torch.empty((capacity,2048,1536), device=device, dtype=a.dtype)
            down = torch.empty((capacity,768,2048), device=device, dtype=b.dtype)
            if static:
                gu[-1].zero_()
                down[-1].zero_()
            for start in range(0,128,chunk):
                full_a = gather(a[start:start+chunk].contiguous(), dim=2)
                full_b = gather(b[start:start+chunk].contiguous(), dim=1)
                if start // 32 == rank:
                    offset = start % 32
                    part, tail = split_weights(full_a,full_b)
                    gu[offset:offset+chunk].copy_(part)
                    down[offset:offset+chunk].copy_(tail)
                    del part, tail
                del full_a, full_b
            prepared.append((mlp,runner,wrappers,gu,down))
    slots=torch.arange(33,device=device,dtype=torch.long) if static else None
    def wrap(mlp, runner, wrappers, gu, down):
        previous = mlp.forward
        @wraps(previous)
        def forward(hidden_states):
            if hidden_states.ndim != 2 or not is_decode(len(hidden_states)):
                return previous(hidden_states)
            if hidden_states.shape[1] != 2048 or hidden_states.dtype != torch.bfloat16 or torch.is_grad_enabled():
                raise ValueError('expected BF16 decode hidden states')
            if wrappers:
                require_no_active_expert_lora(wrappers[0]._build_lora_context())
            logits = runner._apply_router_linear(hidden_states.float())
            w, ids = select(logits, 8, True)
            if static:
                return static_routes(hidden_states, ids, w, gu, down, rank, reduce, slots)
            return local_routes(hidden_states, ids, w, gu, down, rank, reduce)
        return forward
    worker._decode_owner_weights=[(gu,down,runner) for mlp,runner,wrappers,gu,down in prepared]
    for mlp, runner, wrappers, gu, down in prepared:
        mlp.forward = wrap(mlp,runner,wrappers,gu,down)
    return {'bytes': required, 'experts_per_rank_per_layer': 32, 'layers':len(prepared)}
