"""Bounded Python call timing, included only in a decode profiling worker."""
import cProfile
from pathlib import Path
import pstats
import time


def attach_decode_profile(worker, settings):
    from io_utils import save
    from vllm.forward_context import get_forward_context

    if worker.rank != settings['rank']:
        return
    layers = worker.model_runner.get_model().language_model.model.layers
    state = {'steps': 0}
    hooks = []

    def begin(_module, _args):
        meta = get_forward_context().attn_metadata
        groups = meta if isinstance(meta, list) else [meta]
        entries = [value for group in groups if isinstance(group, dict) for value in group.values()]
        if not entries:
            return
        entry = entries[0]
        lengths = getattr(entry, 'actual_seq_lengths_q', None)
        contexts = getattr(entry, 'seq_lens_list', None)
        phase = str(getattr(entry, 'attn_state', '')).split('.')[-1]
        if (phase != 'DecodeOnly' or not isinstance(lengths, (list, tuple)) or
                list(lengths) != list(range(1, len(lengths) + 1)) or not lengths):
            return
        state['context'] = list(contexts) if contexts is not None else []
        state['phase'] = phase
        state['start'] = time.perf_counter()
        state['profiler'] = cProfile.Profile()
        state['profiler'].enable()

    def end(_module, _args, _result):
        profiler = state.pop('profiler', None)
        if profiler is None:
            return
        profiler.disable()
        elapsed = time.perf_counter() - state['start']
        state['steps'] += 1
        rows = []
        for (file, line, func), (_, calls, own, total, _) in pstats.Stats(profiler).stats.items():
            rows.append(dict(file=Path(file).name, line=line, function=func,
                             calls=calls, self_seconds=own, inclusive_seconds=total))
        save(Path('/runtime/state/decode-profile') / f"step-{state['steps']}.json",
             dict(rank=worker.rank, step=state['steps'], phase=state['phase'],
                  seq_lens=state['context'], wall_seconds=elapsed, layers=len(layers),
                  functions=rows))
        if state['steps'] == settings['steps']:
            for hook in hooks:
                hook.remove()

    hooks.append(layers[0].register_forward_pre_hook(begin))
    hooks.append(layers[-1].register_forward_hook(end, always_call=True))
