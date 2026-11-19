"""Diagnostic-only gate/up replacement; use real live BF16 full-width operands."""

from collections import Counter


def project(input, weight):
    """The same NPU BF16 forward operation for a single complete expert."""
    import torch
    import torch_npu

    if (input.ndim != 2 or weight.shape != (2048, 1536) or input.shape[1] != 2048
            or input.dtype != torch.bfloat16 or weight.dtype != input.dtype
            or input.device != weight.device or input.device.type != "npu" or not len(input)):
        raise ValueError("candidate expects nonempty full-width BF16 gate operands")
    rows = torch.tensor([len(input)], dtype=torch.int64, device=input.device)
    result, = torch_npu.npu_grouped_matmul(
        x=[input.contiguous()], weight=[weight.contiguous().unsqueeze(0)],
        split_item=2, group_list_type=1, group_type=0, group_list=rows,
    )
    if result.shape != (len(input), 1536) or result.dtype != input.dtype:
        raise ValueError("grouped gate result changed shape or dtype")
    return result


def attach_hf(text, layers, unwrap):
    import torch
    from transformers.integrations import moe

    old = moe._grouped_mm
    active, handles = [], []
    counts = Counter()
    if not layers or sorted(set(layers)) != layers or any(type(i) is not int or i not in range(len(text.layers)) for i in layers):
        raise ValueError("explicit unique HF gate candidate layers required")

    def grouped(input, weight, offs):
        if not active or tuple(weight.shape[1:]) != (2048, 1536):
            return old(input, weight, offs)
        if torch.is_grad_enabled():
            raise ValueError("diagnostic only; gate candidate backward has not been validated")
        ends = offs.cpu().tolist()
        if len(ends) != len(weight) or ends[-1] != len(input) or any(a > b for a, b in zip([0]+ends, ends)):
            raise ValueError("HF expert offsets do not cover complete routes")
        outputs = [project(input[lo:hi], weight[e])
                   for e, (lo, hi) in enumerate(zip([0]+ends, ends)) if lo < hi]
        counts[active[-1]] += len(outputs)
        return torch.cat(outputs)

    def post(i):
        def end(*_):
            if active != [i]:
                raise ValueError("HF gate candidate scope changed")
            active.pop()
        return end

    for i in layers:
        mlp = unwrap(unwrap(text.layers[i]).mlp)
        handles.extend((mlp.register_forward_pre_hook(lambda m, args, i=i: active.append(i)),
                        mlp.register_forward_hook(post(i), always_call=True)))
    moe._grouped_mm = grouped

    def close():
        for handle in handles:
            handle.remove()
        if moe._grouped_mm is not grouped:
            raise ValueError("HF gate candidate ownership changed")
        moe._grouped_mm = old
        if active or set(counts) != set(layers):
            raise ValueError("incomplete HF gate candidate execution")
        return dict(arithmetic="full_width_bf16_grouped_matmul", layers=dict(counts))

    return close


def attach(model, layers):
    import torch
    from torch.overrides import TorchFunctionMode
    from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod as Method

    known = model.language_model.model.layers
    if not layers or sorted(set(layers)) != layers or not all(type(i) is int and 0 <= i < len(known) for i in layers):
        raise ValueError("explicit unique gate candidate layers required")
    active = []
    counts = Counter()
    handles = []
    old = Method.apply_gmm1

    class GateProjection(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kw = kwargs or {}
            if func is not torch.mm or len(args) != 2 or args[1].shape != (2048, 1536):
                return func(*args, **kw)
            x, w = args
            out = kw["out"]
            result = project(x, w)
            if result.shape != out.shape or result.dtype != out.dtype:
                raise ValueError("grouped gate result changed shape or dtype")
            out.copy_(result)
            counts[active[-1]] += 1
            return out

    def gate(self, m):
        if not active:
            return old(self, m)
        prior = counts[active[-1]]
        with GateProjection():
            result = old(self, m)
        if counts[active[-1]] == prior:
            raise ValueError("candidate did not replace a live gate projection")
        return result

    def post(i):
        def end(*_):
            if active != [i]:
                raise ValueError("gate candidate scope changed")
            active.pop()
        return end

    for i in layers:
        mlp = known[i].mlp
        handles.extend((mlp.register_forward_pre_hook(lambda m, args, i=i: active.append(i)),
                        mlp.register_forward_hook(post(i), always_call=True)))
    Method.apply_gmm1 = gate

    def close():
        for handle in handles:
            handle.remove()
        if Method.apply_gmm1 is not gate:
            raise ValueError("gate candidate ownership changed")
        Method.apply_gmm1 = old
        if active or set(counts) != set(layers):
            raise ValueError("incomplete scoped gate candidate execution")
        return dict(arithmetic="full_width_bf16_grouped_matmul", layers=dict(counts))

    return close
