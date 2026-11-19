"""Per-route observation of the live Ascend MoE combine, named token rows only.

The other backend computes each expert on a TP slice of the intermediate
dimension, so one route's value is the sum of four local partials and the final
output is their all-reduce. This observer records, for the named rows, the real
combine operands (per-rank partial down outputs), the routing weights and the
combine result, so an offline check can decide whether the observed output
difference already exists in the per-route values or only in the reduction.

`expanded_row_idx` from the actual init-routing is indexed by the original flat
(token, k) route and gives where that route landed in the expert-sorted array
(the semantics the installed LoRA routing recovery documents and verifies); the
dispatching route is therefore `expanded_row_idx[t*top_k+s]`, not the reverse.
Nothing is replaced and the original output is returned unchanged.

With `stages=True` the same window also records the rank-local operands of the
two upstream expert stages - the gate/up projection output and the activation
output (the down projection's input) - at the identical route positions. This
answers which stage inside the MoE first differs from the reference without any
new arithmetic: the full-width operand is the rank-ordered concatenation the
production gather itself performs.
"""
import hashlib
import inspect
from pathlib import Path

import torch

NAMES = ('combine.input', 'combine.weights', 'combine.route', 'combine.output')
STAGE_NAMES = ('gate_up.input', 'gate_up.output', 'activation.input',
               'activation.output', 'down.input', 'down.output')


def attach(session, text, *, layer, rows, max_bytes=8*1024**2, tokens=4998, stages=False,
           full_down_experts=(), dispatcher_class=False, operands=None):
    from contextlib import nullcontext
    from boundary_storage import require
    mlp = text.layers[layer].mlp
    runners = [m for m in mlp.modules() if type(m).__name__ == 'AscendMoERunner']
    experts = [m for m in mlp.modules() if type(m).__name__ == 'AscendRoutedExperts']
    require(len(runners) == len(experts) == 1, 'unique live Ascend runner and routed experts')
    runner, expert = runners[0], experts[0]
    top_k = getattr(expert, 'top_k', None) or getattr(runner, 'top_k', None)
    require(type(top_k) is int and top_k > 0, 'actual vLLM routing width')
    require(isinstance(rows, (list, tuple)) and rows and list(rows) == sorted(set(rows))
            and all(isinstance(r, int) and 0 <= r < tokens for r in rows),
            'ordered unique observed token rows')
    routes = [t*top_k+s for t in rows for s in range(top_k)]
    routes_t = torch.tensor(routes, dtype=torch.long)
    rows_t = torch.tensor(list(rows), dtype=torch.long)
    config = getattr(runner, 'moe_config', None)
    audit = dict(layer=layer, backend='vllm', rows=list(rows), top_k=top_k, routes=routes,
                 calls={}, bytes=0, max_bytes=max_bytes, dispatches=0, active=False,
                 installed=False, dispatches_observed=0, routing_weight_applied_inside=False,
                 tp_size=getattr(config, 'tp_size', None),
                 ep_size=getattr(config, 'ep_size', None),
                 num_local_experts=getattr(config, 'num_local_experts', None),
                 original_output_used=True, extra_collectives=0,
                 stages=bool(stages), names=list(NAMES)+(list(STAGE_NAMES) if stages else []),
                 production_acceptance=False,
                 scope='per-rank partial combine operands of the live Ascend MoE, named rows only')
    session.audit['expert_window'] = audit
    state = dict(active=False)
    full = None
    if full_down_experts:
        import expert_down_projection
        from expert_down_capture import Capture
        full = Capture(session, list(full_down_experts))
        def notify_builder(old):
            def notify(phase, expert, start, x, weight, output):
                old(phase, expert, start, x, weight, output)
                if state.get('down_active'):
                    full.record(expert, phase, start, x, weight, output)
            return notify
        session.method(expert_down_projection, 'notify', notify_builder)
    wanted = tuple(audit['names'])

    def record(name, value):
        require(name in wanted and name not in audit['calls'], 'single window relation '+name)
        value = value.detach().contiguous()
        require(value.numel() > 0, 'nonempty window relation '+name)
        size = value.numel()*value.element_size()
        require(audit['bytes']+size <= audit['max_bytes'], 'bounded expert window capture')
        session.tensor('expert_window.'+name, value)
        audit['bytes'] += size
        audit['calls'][name] = dict(shape=list(value.shape), dtype=str(value.dtype))

    def layout(name, value):
        # Capture before indexing/contiguous changes the actual operand layout.
        audit.setdefault('operand_layouts', {})[name] = dict(
            shape=list(value.shape), stride=list(value.stride()),
            storage_offset=value.storage_offset(), dtype=str(value.dtype),
            device=str(value.device))

    def positions_of(expanded):
        """Route -> expert-sorted row, from the live init-routing permutation."""
        flat = torch.abs(expanded).reshape(-1).to(torch.int64)
        require(flat.numel() == tokens*top_k,
                'init-routing covers every (token, slot) route')
        require(torch.equal(torch.sort(flat).values.to(torch.int64),
                            torch.arange(tokens*top_k, dtype=torch.int64,
                                         device=flat.device)),
                'init-routing positions are a complete permutation')
        return flat[routes_t.to(flat.device)]

    def dispatch_builder(old):
        def dispatch(token_dispatch_input):
            result = old(token_dispatch_input)
            if state['active']:
                # The offline comparison assumes the routing weight is applied
                # once, at the combine. Refuse a path that already fused it.
                require(getattr(result, 'topk_scales', None) is None,
                        'routing weight not applied inside the expert path')
                state['dispatched'] = True
            return result
        return dispatch

    def stage_positions(m):
        """The same positions the combine hook will observe, computed early."""
        if state.get('positions') is None:
            expanded = getattr(m, 'expanded_row_idx', None)
            require(expanded is not None, 'live init-routing positions')
            positions = positions_of(expanded)
            state['positions'] = positions
        return state['positions']

    def gmm1_builder(old):
        def call(method, mlp_compute_input):
            if state['active']:
                require(not state.get('gate_up'), 'single observed gate/up projection')
                positions = stage_positions(mlp_compute_input)
                layout('gate_up.input', mlp_compute_input.hidden_states)
                record('gate_up.input',
                       mlp_compute_input.hidden_states[positions.to(
                           mlp_compute_input.hidden_states.device)])
            active = state['active'] and operands is not None
            with operands.observe('gate', mlp_compute_input, positions) if active else nullcontext():
                result = old(method, mlp_compute_input)
            if state['active']:
                layout('gate_up.output', result)
                record('gate_up.output', result[positions.to(result.device)])
                state['gate_up'] = True
            return result
        return call

    def activation_builder(old):
        def call(mlp_compute_input, hidden_states, quant_method):
            if state['active']:
                require(state.get('gate_up') and not state.get('activation'),
                        'ordered activation after the observed gate/up projection')
                positions = stage_positions(mlp_compute_input)
                layout('activation.input', hidden_states)
                record('activation.input', hidden_states[positions.to(hidden_states.device)])
            result = old(mlp_compute_input, hidden_states, quant_method)
            if state['active']:
                layout('activation.output', result)
                record('activation.output', result[positions.to(result.device)])
                state['activation'] = True
            return result
        return call

    def gmm2_builder(old):
        def call(method, mlp_compute_input, hidden_states, act_out_scale):
            if state['active']:
                require(state.get('activation') and not state.get('down'),
                        'ordered down projection after the observed activation')
                positions = stage_positions(mlp_compute_input)
                layout('down.input', hidden_states)
                record('down.input', hidden_states[positions.to(hidden_states.device)])
                state['down'] = True
            state['down_active'] = state['active']
            try:
                active = state['active'] and operands is not None
                with operands.observe('down', mlp_compute_input, positions) if active else nullcontext():
                    result = old(method, mlp_compute_input, hidden_states, act_out_scale)
            finally:
                state['down_active'] = False
            if state['active']:
                layout('down.output', result)
                record('down.output', result[positions.to(result.device)])
            return result
        return call

    def combine_builder(old):
        def combine(hidden_states, combine_metadata, bias=None):
            if not state['active']:
                return old(hidden_states, combine_metadata, bias)
            require(state.get('dispatched'), 'combine follows the observed dispatch')
            # Route -> expert-sorted position, as the installed LoRA routing
            # recovery reads the same tensor.
            positions = positions_of(combine_metadata.expanded_row_idx)
            if state.get('positions') is not None:
                require(torch.equal(state['positions'].to(positions.device),
                                    positions.to(state['positions'].device)),
                        'expert-sorted positions agree across observed stages')
            require(hidden_states.shape[0] == tokens*top_k,
                    'every route holds one expert-sorted row')
            positions = positions.to(hidden_states.device)
            weights = combine_metadata.topk_weights
            require(list(weights.shape) == [tokens, top_k], 'routing weights cover every token')
            require(audit['dispatches'] == 0, 'single observed combine')
            audit['dispatches'] = 1
            record('combine.route', routes_t)
            layout('combine.input', hidden_states)
            record('combine.input', hidden_states[positions])
            record('combine.weights', weights.reshape(-1)[routes_t.to(weights.device)])
            result = old(hidden_states, combine_metadata, bias)
            record('combine.output', result[rows_t.to(result.device)])
            return result
        return combine

    def before(module, args, kwargs):
        require(not state['active'], 'single observed language prefill')
        if not audit['installed']:
            if dispatcher_class:
                # The eager decoder enters the Ascend communication scope
                # inside the runner, after the MLP prehook. Bind the exact
                # dispatcher class used by the installed family overlay.
                from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
                owner = TokenDispatcherWithAllGather
                def unbound(builder):
                    def wrap(old):
                        def call(dispatcher, *args, **kwargs):
                            return builder(old.__get__(dispatcher, type(dispatcher)))(*args, **kwargs)
                        return call
                    return wrap
                session.method(owner, 'token_dispatch', unbound(dispatch_builder))
                session.method(owner, 'token_combine', unbound(combine_builder))
                dispatcher_name = owner.__module__ + '.' + owner.__qualname__
            else:
                from vllm_ascend.ascend_forward_context import _EXTRA_CTX
                comm = getattr(_EXTRA_CTX, 'moe_comm_method', None)
                require(comm is not None and getattr(comm, 'token_dispatcher', None) is not None,
                        'live MoE communication method and dispatcher')
                owner = comm.token_dispatcher
                session.method(owner, 'token_dispatch', dispatch_builder)
                session.method(owner, 'token_combine', combine_builder)
                dispatcher_name = type(owner).__module__+'.'+type(owner).__qualname__
            if stages:
                from vllm_ascend.ops.fused_moe.routed_experts import (
                    AscendUnquantizedFusedMoEMethod as Method)
                from vllm_ascend.ops.fused_moe import moe_mlp
                def identity(fn):
                    path = Path(inspect.getfile(fn))
                    return dict(module=fn.__module__, qualname=fn.__qualname__,
                                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                audit['stage_sources'] = dict(
                    apply_gmm1=identity(Method.apply_gmm1),
                    activation=identity(moe_mlp._unified_apply_activation),
                    apply_gmm2=identity(Method.apply_gmm2))
                session.method(Method, 'apply_gmm1', gmm1_builder)
                session.method(moe_mlp, '_unified_apply_activation', activation_builder)
                session.method(Method, 'apply_gmm2', gmm2_builder)
            audit['installed'] = True
            audit['dispatcher'] = dispatcher_name
        state['active'] = True
        for key in ('dispatched', 'positions', 'gate_up', 'activation', 'down'):
            state.pop(key, None)
        audit['active'] = True

    def after(module, args, kwargs, result):
        if full is not None:
            full.validate()
        state['active'] = False
        for key in ('dispatched', 'positions', 'gate_up', 'activation', 'down'):
            state.pop(key, None)
        audit['active'] = False

    session.pre(mlp, before)
    session.post(mlp, after)
    return audit


def validate(session):
    from boundary_storage import require
    a = session.audit['expert_window']
    require(a.get('backend') == 'vllm' and a.get('installed') and not a.get('active'),
            'closed vLLM window observation')
    require(a['dispatches'] == 1, 'exactly one observed combine')
    require(set(a['calls']) == set(a.get('names') or NAMES), 'complete per-rank window')
    expected = len(a['routes'])
    for name in ('combine.route', 'combine.input', 'combine.weights'):
        require(a['calls'][name]['shape'][0] == expected, 'complete window rows '+name)
    require(a['calls']['combine.output']['shape'][0] == len(a['rows']), 'combine row coverage')
    require(a['calls']['combine.route']['dtype'] in ('torch.int64', 'torch.int32'),
            'route identity recorded')
    for name in ('combine.input', 'combine.weights', 'combine.output'):
        require(a['calls'][name]['dtype'] == 'torch.bfloat16', 'BF16 combine operand '+name)
    if a.get('stages'):
        for name in STAGE_NAMES:
            require(name in a['calls'] and a['calls'][name]['shape'][0] == expected,
                    'complete observed stage rows '+name)
            require(a['calls'][name]['dtype'] == 'torch.bfloat16',
                    'BF16 observed stage operand '+name)
        require(a['calls']['gate_up.output']['shape'][1]
                == 2*a['calls']['activation.output']['shape'][1],
                'packed gate/up width is twice the activation width')
        require(a['calls']['down.input']['shape'] == a['calls']['activation.output']['shape'],
                'down operand is the observed activation output')
    return a
