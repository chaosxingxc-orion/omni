"""Record intervals on the current NPU stream and read them on later forwards."""
import time

pending = []
observed = set()
current = None


def configure(layers):
    global observed
    observed = set(layers)


def begin(step):
    global current
    from vllm import stage_clock
    remaining = []
    for entry in pending:
        first, last, key, layer, phase, host_ms = entry
        if last.query():
            stage_clock.emit('prefill_interval', step=key, layer=layer, phase=phase,
                             device_ms=first.elapsed_time(last), host_ms=host_ms)
        else:
            remaining.append(entry)
    pending[:] = remaining
    current = step


def start(layer):
    if current is None or layer not in observed:
        return None
    import torch
    first = torch.npu.Event(enable_timing=True)
    first.record()
    return first, time.perf_counter_ns()


def end(layer, phase, mark):
    if mark is None:
        return
    import torch
    first, start_ns = mark
    host_ms = (time.perf_counter_ns() - start_ns) / 1e6
    last = torch.npu.Event(enable_timing=True)
    last.record()
    pending.append((first, last, current, layer, phase, host_ms))
