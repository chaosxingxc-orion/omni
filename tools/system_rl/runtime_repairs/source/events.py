"""DP event transport into the original verl actor; no backward/optimizer loop.

Only real owned rows contribute. Tail slots replay a declared existing input
with zero VJP, never a fabricated action or a denominator member. Every rank
has the same sequence of policy/NLL/pair forward+backward collective slots.
"""
import copy
from dependencies import require


def partition(plan, fields, world):
    require(type(world) is int and world > 0, 'positive configured world size')
    plan.assert_intact()
    require(plan.base.rows > 0, 'all-empty cohort skips optimizer and clock')
    require(set(plan.base.routes) == {'default'}, 'first release supports shared default LoRA only')
    # Greedy deterministic length balancing, with exactly one row owner.
    lengths = fields['tensors']['attention_mask'].sum(-1).tolist()
    loads = [0]*world; owners = [-1]*plan.base.rows
    for row in sorted(range(plan.base.rows), key=lambda i: (-lengths[i], i)):
        rank = min(range(world), key=lambda r: (loads[r], r))
        owners[row] = rank; loads[rank] += int(lengths[row])
    lanes = {kind: [[] for _ in range(world)] for kind in ('policy', 'nll', 'pair')}
    for row, owner in enumerate(owners):
        lanes['policy'][owner].append(dict(kind='policy', row=row, owned=True))
        for event in plan.rows[row]:
            coefficient = plan.coefficients[event['term']]
            if coefficient == 0:
                continue
            require(plan.counts[event['term']] > 0, 'positive original observed denominator')
            kind = 'pair' if event['kind'] == 'ranking' else 'nll'
            lanes[kind][owner].append(dict(kind=kind, row=row, event=copy.deepcopy(event),
                weight=coefficient/plan.counts[event['term']], owned=True))
    schedules = [[] for _ in range(world)]
    for kind, by_rank in lanes.items():
        count = max(map(len, by_rank))
        if not count:
            continue
        donor = next(e for items in by_rank for e in items)
        for index in range(count):
            for rank in range(world):
                event = copy.deepcopy(by_rank[rank][index] if index < len(by_rank[rank]) else donor)
                event['owned'] = index < len(by_rank[rank])
                event['slot'] = len(schedules[rank])
                if kind == 'pair':
                    event['pair_id'] = index
                    for side in ('preferred', 'dispreferred'):
                        schedules[rank].append(dict(event, side=side, slot=len(schedules[rank])))
                else:
                    schedules[rank].append(event)
    return dict(owners=owners, loads=loads, schedules=schedules,
                real_rows=plan.base.rows, slots_per_rank=len(schedules[0]),
                global_episode_denominator=plan.base.denominator,
                task_denominators=plan.counts, full_sample_replicated=False)


def slice_fields(fields, row):
    return dict(tensors={k:v[row:row+1] for k,v in fields['tensors'].items()},
                non_tensors={k:v[row:row+1] for k,v in fields['non_tensors'].items()},
                meta_info=copy.deepcopy(fields['meta_info']))


def transport(fields, schedule):
    """Variable shape pieces carry source row + event outside online model kwargs."""
    from verl import DataProto
    class Events(DataProto):
        def select(self, batch_keys=None, non_tensor_batch_keys=None, **kwargs):
            keys = list(dict.fromkeys(list(non_tensor_batch_keys or [])+['record_identity','adapter_route']))
            return super().select(batch_keys=batch_keys, non_tensor_batch_keys=keys, **kwargs)
        def split(self, size):
            if not getattr(self, '_inside', False):
                require(size == len(schedule), 'one optimizer minibatch of events')
                mini = self[:]; mini._inside = True
                return [mini]
            require(size == 1, 'one graph per event')
            return [self[i:i+1] for i in range(len(schedule))]
    # Transport rows retain the original sampled IDs, old logprobs and masks.
    # Teacher targets are built offline in the forward attachment, not admitted
    # as sampled trajectories, not used in the global episode denominator.
    import torch
    order = [e['row'] for e in schedule]
    return Events.from_dict(tensors={k:v[torch.tensor(order)] for k,v in fields['tensors'].items()},
        non_tensors={k:v[order] for k,v in fields['non_tensors'].items()},
        meta_info=dict(fields['meta_info'], temperature=1., use_dynamic_bsz=False))


def run_existing_actor(actor, fields, plan, scorer, policy_forward, *, actor_sha256, world, rank,
                       before_step=None, forward_api="tuple", after_probe=None):
    """Called by both CPU verification and the configured real worker entry."""
    import inspect
    import torch
    import torch.distributed as dist
    from dependencies import sha, upstream
    from multimodal_jev.verl_native_loss import _register
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY
    task = upstream()
    require(sha(inspect.getfile(type(actor))) == actor_sha256, 'installed actor pin differs')
    require(dist.get_world_size() == world and dist.get_rank() == rank, 'actual world/rank differs')
    require(actor.scaler is None, 'BF16/FP32 event callback only')
    require(not any(key in fields['tensors'] for key in ('rollout_log_probs','rollout_is_weights')),
            'event scalars cannot feed token logprob diagnostics or importance correction')
    require(not any(isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.p for m in actor.actor_module.modules()),
            'deterministic dropout0 recomputation required')
    if plan.base.rows == 0:
        return dict(status='no_gradient',optimizer_step=False,advance_successful_step=False)
    layout = partition(plan, fields, world)
    indices = [i for i, owner in enumerate(layout['owners']) if owner == rank]
    task.distributed_contract(plan, indices, world, rank)
    peers = [None]*world
    dist.all_gather_object(peers, layout)
    require(all(p == layout for p in peers), 'global event schedule mismatch')
    schedule = layout['schedules'][rank]
    data = transport(fields, schedule)
    pairs = {}; live = None; cursor = 0; backward_complete = True
    trace = []; probes = 0; gradient_signal = None
    # Detached true objective values, already normalized by ORIGINAL global
    # denominators. One owner per row; pair softplus once on preferred side.
    loss_names=['policy','kl','policy_total',*sorted(plan.counts)]
    loss_sums={name:0. for name in loss_names}
    def observe(name,value,event):
        if event['owned']:loss_sums[name]+=float(value.detach())
    def on_device(inputs):
        device = next(actor.actor_module.parameters()).device
        converted = {k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in inputs.items()}
        converted['multi_modal_inputs'] = [{k:v.to(device) for k,v in mm.items()} for mm in inputs['multi_modal_inputs']]
        return converted
    # Probe pair scores without retaining any graph. Parameters never change
    # until the original actor has consumed all event backwards.
    for event in schedule:
        if event['kind'] != 'pair' or event['side'] != 'preferred':
            continue
        source = slice_fields(fields, event['row'])
        inputs = on_device({**source['tensors'], **source['non_tensors']})
        with torch.no_grad():
            a = scorer.probe(inputs, 0, event['event']['preferred'])[0]
            if after_probe: after_probe()
            b = scorer.probe(inputs, 0, event['event']['dispreferred'])[0]
            if after_probe: after_probe()
            require(torch.isfinite(a) and torch.isfinite(b), 'nonfinite pair scalar probe')
            c = event['weight']*torch.sigmoid(b-a)
            pairs[event['pair_id']] = dict(c=c.detach(), a=a.detach(), b=b.detach(),
                value=(event['weight']*torch.nn.functional.softplus(b-a)).detach())
            probes += 2
    saved_config = actor.config
    had_forward = '_forward_micro_batch' in vars(actor)
    saved_forward = vars(actor).get('_forward_micro_batch')
    had_step = '_optimizer_step' in vars(actor)
    saved_step = vars(actor).get('_optimizer_step')
    original_step = actor._optimizer_step
    config = copy.deepcopy(saved_config)
    for key, value in dict(ppo_epochs=1, use_dynamic_bsz=False, use_kl_loss=False,
            entropy_coeff=0., calculate_entropy=False, use_rollout_log_probs=True,
            global_batch_info={}, ppo_mini_batch_size=len(schedule), ppo_micro_batch_size_per_gpu=1,
            loss_agg_mode='unified-token-sum', policy_loss={'loss_mode':'omni_jev_dp8_events_v1'}).items():
        setattr(config, key, value)
    name = 'omni_jev_dp8_events_v1'
    def forward(inputs, temperature, calculate_entropy=False, **kwargs):
        nonlocal live, backward_complete
        require(live is None and backward_complete and not kwargs and temperature == 1. and not calculate_entropy,
                'new graph before prior backward finalized')
        inputs = on_device(inputs)
        event = schedule[cursor]; row = event['row']
        require(list(inputs['record_identity']) == plan.base.identities[row:row+1], 'event source identity changed')
        require(list(inputs['adapter_route']) == ['default'], 'shared route changed')
        if event['kind'] == 'policy':
            logp = policy_forward(inputs)
            value, components = plan.base.loss(logp, row, row+1)
            observe('policy',components['rl'],event)
            observe('kl',components['kl'],event)
            observe('policy_total',value,event)
        else:
            target = event['event'][event['side']] if event['kind'] == 'pair' else event['event']['text']
            scores = task.score_target(scorer, inputs, 0, target)
            if event['kind'] == 'nll':
                value = -event['weight']*scores[1]
                observe(event['event']['term'],value,event)
            else:
                pair = pairs[event['pair_id']]
                key = 'a' if event['side'] == 'preferred' else 'b'
                if event['side']=='preferred':observe(event['event']['term'],pair['value'],event)
                require(torch.allclose(scores[0].detach(), pair[key], atol=1e-5, rtol=1e-6),
                        'probe/recompute changed: dropout or model state drift')
                value = scores[0]*pair['c']*(-1 if key == 'a' else 1)
        # Return the actual differentiable event scalar as the callback tensor.
        # Sampled fields remain intact for the original actor transport checks.
        live = value.reshape(1, 1)
        require(live.requires_grad and live.grad_fn is not None, 'frozen base lost LoRA graph')
        backward_complete = False
        def started(grad):
            def finished():
                nonlocal backward_complete
                backward_complete = True
            torch.autograd.Variable._execution_engine.queue_callback(finished)
            return grad
        live.register_hook(started)
        trace.append(dict(slot=cursor, kind=event['kind'], row=row, owned=event['owned']))
        require(forward_api in ('tuple','dict'), 'explicit pinned actor forward ABI')
        return {'log_probs':live} if forward_api == 'dict' else (None, live)
    def loss(old_log_prob, log_prob, advantages, response_mask, **kwargs):
        nonlocal cursor, live
        require(log_prob is live and live is not None, 'exact event graph must be consumed')
        row = schedule[cursor]['row']
        for key, actual in [('old_log_probs', old_log_prob), ('advantages', advantages), ('response_mask', response_mask)]:
            require(torch.equal(actual.cpu(), fields['tensors'][key][row:row+1].cpu()), 'changed original transport: '+key)
        value = live.sum()*int(schedule[cursor]['owned'])
        cursor += 1; live = None
        # Native loop divides by S, FSDP/DDP averages by W. Compensate ONCE.
        return value*len(schedule)*world, {'dp8/owned_event':int(schedule[cursor-1]['owned'])}
    def optimizer_step():
        nonlocal gradient_signal
        require(cursor == len(schedule) and live is None and backward_complete, 'incomplete event backwards')
        plan.assert_intact()
        require(all(p.grad is None for p in actor.actor_module.parameters() if not p.requires_grad), 'frozen parameter got gradient')
        if before_step:
            before_step(actor)
        # Inspect the actual reduced objective gradient, including auxiliary
        # and KL terms. Zero advantages alone say nothing about that gradient.
        # FSDP shards can be empty/zero on one rank; all ranks share the decision.
        device = next(actor.actor_module.parameters()).device
        flags = torch.zeros(2, dtype=torch.int32, device=device)
        for parameter in actor.actor_module.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                grad = parameter.grad
                flags[0] = torch.maximum(flags[0], (~torch.isfinite(grad).all()).to(torch.int32))
                flags[1] = torch.maximum(flags[1], (grad != 0).any().to(torch.int32))
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        require(flags[0].item() == 0, 'nonfinite objective gradient; no optimizer step')
        gradient_signal = bool(flags[1].item())
        if not gradient_signal:
            # Calling Adam with zero non-None gradients advances its moments
            # and can move parameters using old momentum. Do not call it.
            return torch.zeros((), device=device)
        return original_step()
    require(not any(key in fields['tensors'] for key in ('rollout_log_probs','rollout_is_weights')),
            'token diagnostics are incompatible with explicit event scalar callback')
    _register(name, loss)
    actor.config = config; actor._forward_micro_batch = forward; actor._optimizer_step = optimizer_step
    try:
        metrics = actor.update_policy(data)
        require(cursor == len(schedule) and backward_complete, 'incomplete native loop')
        require(gradient_signal is not None, 'native actor never reached optimizer seam')
        # HCCL cannot all-reduce doubles on Ascend. The detached scalar totals
        # are already CPU doubles; sum the tiny per-rank report in stable order.
        import math
        reports=[None]*world
        dist.all_gather_object(reports,[loss_sums[n] for n in loss_names])
        objective={name:math.fsum(row[i] for row in reports) for i,name in enumerate(loss_names)}
        objective['total']=objective['policy_total']+sum(objective[n] for n in plan.counts)
        observation=dict(schema='owned-global-true-objective-v1',components=objective,
            global_episode_denominator=plan.base.denominator,task_denominators=plan.counts,
            component_semantics=dict(policy='original rl surrogate; actual objective uses minus rl, included in policy_total',kl='unweighted original KL; beta is in snapshot',task='coefficient times original loss divided by observed term count'),
            snapshot=copy.deepcopy(plan.base.snapshot),snapshot_sha256=fields['meta_info'].get('snapshot_sha256'),
            sidecar_sha256=plan.sidecar_sha256,owned_rows=plan.base.rows,padding_included=False,
            pair_value='weighted_original_softplus_once_per_owned_preferred',extra_forward=0,
            reduction='sum_ranks_already_global_normalized',optimizer_step=gradient_signal)
        return dict(objective_observation=observation,metrics=metrics, layout=layout, local_owned_rows=indices, events=trace,
                    scalar_probes=probes, pair_loss_values={str(k):float(v['value']) for k,v in pairs.items()},
                    native_update_policy=True, native_optimizer_step=gradient_signal,
                    status='updated' if gradient_signal else 'no_gradient',
                    optimizer_step=gradient_signal, advance_successful_step=gradient_signal,
                    gradient_decision='collective_actual_objective_gradient_nonzero')
    finally:
        actor.config = saved_config
        if had_forward: actor._forward_micro_batch = saved_forward
        else: del actor._forward_micro_batch
        if had_step: actor._optimizer_step = saved_step
        else: del actor._optimizer_step
        if POLICY_LOSS_REGISTRY.get(name) is loss:
            del POLICY_LOSS_REGISTRY[name]
