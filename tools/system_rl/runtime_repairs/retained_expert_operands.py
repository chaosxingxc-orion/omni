"""Capture the actual per-expert BF16 mm and replay row-shape controls."""

import bisect
import hashlib
import itertools

from expert_down_capture import Capture


def digest(t):
    import torch
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy()).hexdigest()


def diff(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("incompatible expert operands")
    return dict(unequal=int((a != b).sum()), max_abs=float((a.float() - b.float()).abs().max()))


class ExpertOperands:
    def __init__(self, session, stages=('gate', 'down')):
        self.session = session
        self.wanted = tuple(stages)
        self.stages = {}

    def observe(self, stage, m, positions):
        import torch
        from torch.overrides import TorchFunctionMode
        from contextlib import nullcontext
        from expert_gate_repair import dispatch_order

        if stage not in self.wanted:
            return nullcontext()
        counts = m.group_list.cpu().long().tolist()
        if m.group_list_type == 0:
            counts = [n - prev for n, prev in zip(counts, [0] + counts[:-1])]
        elif m.group_list_type != 1:
            raise ValueError("unknown expert group encoding")
        ends = list(itertools.accumulate(counts))
        if len(counts) != 128 or min(counts) < 0 or ends[-1] != len(m.hidden_states):
            raise ValueError("incomplete expert groups")
        _, inverse = dispatch_order(m.topk_ids, m.expanded_row_idx)
        wanted = inverse[positions.to(inverse.device)].tolist()
        target = {}
        for slot, row in enumerate(wanted):
            expert = bisect.bisect_right(ends, row)
            if expert in target:
                raise ValueError("duplicate selected expert")
            target[expert] = (slot, row - (ends[expert - 1] if expert else 0))

        # Reuse the projection's bounded capture format. Select all eight live
        # routes, and keep this broadened budget confined to the explicit probe.
        cap = Capture(
            self.session, sorted(target), max_bytes=160 * 1024**2,
            max_experts=8, namespace="expert_mm." + stage,
        )
        shape = {"gate": (2048, 1536), "down": (768, 2048)}[stage]
        experts = iter(i for i, c in enumerate(counts) if c)

        class ObserveMM(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kw = kwargs or {}
                if func is not torch.mm or len(args) != 2 or tuple(args[1].shape) != shape:
                    return func(*args, **kw)
                e = next(experts)
                x, w = args
                y = kw["out"]
                if len(x) != counts[e] or y.shape != (counts[e], shape[1]):
                    raise ValueError("live expert GEMM order changed")
                start = ends[e] - counts[e]
                if e in target:
                    cap.record(e, "before", start, x, w, y)
                result = func(*args, **kw)
                if e in target:
                    cap.record(e, "after", start, x, w, result)
                return result

        self.stages[stage] = (cap, target, experts)
        return ObserveMM()

    def close(self):
        result = {}
        for stage, (cap, target, remaining) in self.stages.items():
            if next(remaining, None) is not None:
                raise ValueError("expert GEMM coverage changed")
            cap.validate()
            result[stage] = dict(calls=cap.audit["calls"], target=target)
        if result.keys() != set(self.wanted):
            raise ValueError("missing live expert projection")
        return result


def arithmetic_controls(xa, xb, wa, ia, ib, device):
    """Compare candidate operators under a pure row swap and actual groups."""
    import time
    import torch
    import torch_npu
    w = wa.to(device)
    inputs = [x.to(device) for x in (xa, xb)]
    ref = (xa[ia].double() @ wa.double()).bfloat16()
    checks = {}
    variants = {
        'bf16_mm_contiguous': lambda x: torch.mm(x, w),
        'bf16_mm_transposed': lambda x: torch.mm(x, wt),
        'fp32_mm_transposed': lambda x: torch.mm(x.float(), wf).bfloat16(),
        'bf16_grouped_matmul': lambda x: torch_npu.npu_grouped_matmul(
            x=[x], weight=[w.unsqueeze(0)], split_item=2, group_list_type=1,
            group_type=0, group_list=torch.tensor([len(x)], dtype=torch.int64, device=device),
        )[0],
    }
    wt, wf = w.T.contiguous().T, w.T.contiguous().float().T
    for name, op in variants.items():
        checks[name] = {}
        started = time.monotonic()
        try:
            pair = [op(x)[i].cpu() for x, i in zip(inputs, (ia, ib))]
            # Preserve the complete set of background rows, changing only
            # target position (including tile tail and interior offsets).
            locations = sorted(set((0, ia, ib, len(xa)-1, 1, 15, 16, 31, 32, 127, 128)))
            locations = [i for i in locations if 0 <= i < len(xa)]
            rows = []
            for pos in locations:
                moved = inputs[0].clone()
                if pos != ia:
                    moved[pos], moved[ia] = inputs[0][ia], inputs[0][pos]
                rows.append(op(moved)[pos].cpu())
            checks[name].update(
                actual_pair=diff(*pair),
                compared_offsets=locations,
                row_permutation=[diff(pair[0], row) for row in rows],
                vs_rounded_cpu_fp64=[diff(row, ref) for row in pair],
                wall_seconds=time.monotonic()-started,
            )
        except (RuntimeError, NotImplementedError) as exc:
            checks[name]['error'] = str(exc).splitlines()[0]
    return checks


def compare(a, b, device, *, controls=False):
    import torch
    aa, bb = a["operands"], b["operands"]
    report = {}
    if aa.keys() != bb.keys():
        raise ValueError("different observed expert stages")
    for stage in aa:
        ta, tb = aa[stage]["target"], bb[stage]["target"]
        if {e: s for e, (s, _) in ta.items()} != {e: s for e, (s, _) in tb.items()}:
            raise ValueError("expert identities or slot order changed")
        for e, (slot, ia) in ta.items():
            ib = tb[e][1]
            key = f"expert_mm.{stage}.{e}."

            def get(snapshot, name):
                return snapshot["capture"][key + name]

            ca, cb = aa[stage]["calls"][str(e)], bb[stage]["calls"][str(e)]
            xa, xb = (get(s, "input") for s in (a, b))
            ya, yb = (get(s, "output") for s in (a, b))
            wa, wb = (get(s, "weight") for s in (a, b))
            if not torch.equal(wa, wb) or ca["weight"]["stride"] != cb["weight"]["stride"]:
                raise ValueError("live expert weights must agree")
            w = wa.to(device)
            if ca["weight"]["stride"][0] == 1:
                w = w.T.contiguous().T
            if list(w.stride()) != ca["weight"]["stride"]:
                raise ValueError("failed to restore live expert weight layout")
            def mm(x):
                x = x.to(device).contiguous()
                out = torch.empty((len(x), w.shape[1]), device=device, dtype=x.dtype)
                torch.mm(x, w, out=out)
                return out.cpu()

            n, m = len(xa), len(xb)
            anchor = xa[ia]
            # Keep identical operands at the target row and use the same
            # background rows, changing only M or the anchor's row offset.
            base = xa if n >= m else xb
            def control(rows, idx):
                x = base[:rows].clone()
                x[idx] = anchor
                return mm(x)[idx]
            ma, mb = mm(xa), mm(xb)
            sa, sb = control(n, 0), control(m, 0)
            record = dict(
                slot=slot, rows=[n, m], selected_offsets=[ia, ib],
                weight_sha256=digest(wa), live_layouts=[ca, cb],
                selected_input=diff(xa[ia], xb[ib]),
                selected_output=diff(ya[ia], yb[ib]),
                original_group_replay=[diff(ya[ia], ma[ia]), diff(yb[ib], mb[ib])],
                same_row_offset_vary_m=diff(sa, sb),
                same_m_vary_row_offset=[diff(sa, control(n, ia)), diff(sb, control(m, ib))],
            )
            if stage == "gate":
                def shard(v, i):
                    s = 192*i
                    return torch.cat((v[s:s+192], v[768+s:768+s+192]))
                record["live_output_rank_differences"] = [
                    diff(shard(ya[ia], i), shard(yb[ib], i)) for i in range(4)
                ]
            if (controls and record["selected_output"]["unequal"]
                    and not record["selected_input"]["unequal"]):
                record["candidate_arithmetic"] = arithmetic_controls(xa, xb, wa, ia, ib, device)
            report.setdefault(stage, {})[str(e)] = record
    return report
