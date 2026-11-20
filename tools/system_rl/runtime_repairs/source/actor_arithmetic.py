"""Attach the calibrated BF16 arithmetic at actor construction, including VJPs."""
from contextlib import ExitStack

PROFILES = {
    4: 'omni-bf16-tp4-fsdp4-v1',
    8: 'omni-bf16-tp4-fsdp8-candidate-v1',
}

def install(adapted, profile, world):
    if PROFILES.get(world) != profile or profile is None:
        raise ValueError('unknown actor arithmetic profile or unverified topology')
    from retained_expert_candidate import attach_hf as experts
    from retained_attention_candidate import attach_hf as attention
    from retained_router_candidate import attach as router
    from retained_kv_candidate import attach as kv
    text = adapted.get_base_model().model
    if len(text.layers) != 48:
        raise ValueError('expected 48 Omni text layers')
    layers = list(range(48))

    def unwrap(m):
        while type(m).__name__ in ('CheckpointWrapper', 'FullyShardedDataParallel'):
            m = m.module if type(m).__name__ == 'FullyShardedDataParallel' else m._checkpoint_wrapped_module
        return m

    receipts = {}
    stack = ExitStack()
    def bind(key, closer):
        def collect():
            receipts[key] = closer()
        stack.callback(collect)
    try:
        bind('gate', experts(text, layers, unwrap, frozen_vjp=True))
        bind('attention', attention(text, layers, unwrap, training_vjp=True, tiled_layers=layers))
        bind('down', experts(text, layers, unwrap, stage='down', frozen_vjp=True))
        bind('router', router(text, layers, side='hf', unwrap=unwrap, training_vjp=True))
        bind('kv_projection', kv(text, layers, side='hf', unwrap=unwrap, training_vjp=True))
    except BaseException:
        stack.close()
        raise
    def close():
        stack.close()
        return receipts.copy()
    return close
