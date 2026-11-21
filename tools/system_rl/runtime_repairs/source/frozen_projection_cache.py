"""Prepack the four immutable base projections per layer in the activation window."""


def validate(config):
    if set(config) != {'reserve_bytes', 'budget_bytes'} or any(
            type(value) is not int or value <= 0 for value in config.values()):
        raise ValueError('positive explicit projection cache budget and reserve required')


def attach(worker, config):
    import torch
    import projection_family_runtime as projection
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather as gather

    validate(config)
    model = worker.model_runner.get_model().language_model.model
    group = get_tp_group()
    rank = group.rank_in_group
    if len(model.layers) != 48 or group.world_size != 4 or worker.rank != rank:
        raise ValueError('frozen projection cache requires the accepted TP4 topology')
    views = []
    for layer in model.layers:
        qkv = layer.self_attn.qkv_proj.base_layer.weight
        row = layer.self_attn.o_proj.base_layer.weight
        if qkv.shape != (1280, 2048) or row.shape != (2048, 1024):
            raise ValueError('unexpected base projection layout')
        for view in qkv.split((1024, 128, 128), dim=0):
            views.append((view, 0))
        views.append((row, 1))
    if any(t.dtype != torch.bfloat16 or t.device.type != 'npu' for t, _ in views):
        raise ValueError('live BF16 base projections required')
    required = sum(t.numel() * t.element_size() * group.world_size for t, _ in views)
    # Hold the accumulated final matrices plus one temporary gather/transposition.
    scratch = max(t.numel() * t.element_size() for t, _ in views) * group.world_size * 3
    free, _ = torch.npu.mem_get_info()
    admissible = required <= config['budget_bytes'] and free >= required + scratch + config['reserve_bytes']
    if not bool(gather(torch.tensor([int(admissible)], device=views[0][0].device), dim=0).all()):
        raise ValueError('insufficient projection cache headroom on at least one rank')
    def key(t, dim):
        return (t.data_ptr(), tuple(t.shape), t.stride(), t.dtype, t.device, dim)

    original = projection._gather
    entries = {}
    with torch.inference_mode():
        for view, dim in views:
            identity = key(view, dim)
            if identity in entries:
                raise ValueError('overlapping frozen projection identities')
            full = original(view, dim, rank, gather)
            entries[identity] = full.T.contiguous().T
            del full
    def lookup(t, dim, current_rank, live_gather):
        if rank != current_rank:
            raise ValueError('projection cache rank changed')
        cached = entries.get(key(t, dim))
        if cached is not None:
            return cached
        return original(t, dim, current_rank, live_gather)

    # Keep the base storages alive for the entire worker lifetime. Hot LoRA
    # tensors and request activations always go through the live gather.
    worker._frozen_projection_cache = (views, entries)
    projection._gather = lookup
    return {'matrices': len(entries), 'bytes': required}
