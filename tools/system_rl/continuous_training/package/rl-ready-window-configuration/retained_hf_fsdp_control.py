"""Use the real training worker for fixed-action, no-update BF16 parity."""
import argparse
import hashlib
import json
import os
import pwd
import grp
import sys
import time
from datetime import timedelta
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_bytes())


def main(spec, inputs_root, out):
    import torch
    import torch_npu
    import torch.distributed as dist
    from dependencies import ROOT, require
    from storage import check_running, commit
    from policy_identity import tensor_hash

    sys.path.insert(0, str(ROOT/'native-model-integration'))
    sys.path.insert(0, str(ROOT/'native-training-successor'))
    from validate_npu import inputs_for
    from forward import forward
    from worker_dp8 import construct

    def sealed(name):
        p = inputs_root/name
        require(hashlib.sha256(p.read_bytes()).hexdigest() == spec['files'][name], f'input drift: {name}')
        return p

    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('npu', rank)
    torch.npu.set_device(device)
    torch.set_num_threads(2)
    # Use the training image's actual format policy; record it before inference.
    fmt = torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')
    check_running(out, spec['storage_budget_bytes'])
    assets = sealed('BASE-ASSETS.json')
    cfg = dict(spec['training'], base_assets_ref=dict(path=str(assets), sha256=spec['files'][assets.name]))
    # HCCP TLS resolves the container UID/GID through the host identity mount.
    uid = pwd.getpwuid(os.getuid())
    grp.getgrgid(os.getgid())
    dist.init_process_group('hccl', timeout=timedelta(seconds=900))
    try:
        witness = torch.tensor([rank + 1], dtype=torch.int32, device=device)
        dist.all_reduce(witness)
        world = dist.get_world_size()
        require(int(witness) == world*(world+1)//2, 'first HCCL collective mismatch')
        commit(out/f'rank-{rank}'/'COLLECTIVE-READY.json',
               dict(world_size=world, uid=uid.pw_uid, collective_sum=int(witness)))
        if 'operator_vjp' in spec:
            for name in ('retained_operator_vjp.py','retained_expert_candidate.py','retained_attention_candidate.py'):
                sealed(name)
            from retained_operator_vjp import measure
            probe = spec['operator_vjp']
            commit(out/f'rank-{rank}'/'RESULT.json',measure(device,rank,probe['lengths'],probe['seed'],**{k:True for k in ('router_only','tiled_kv') if probe.get(k)}))
            return
        if 'batch_probe' in spec:
            sealed('probe_actor_microbatch.py')
            from probe_actor_microbatch import measure
            return measure(spec,out,rank,world,cfg,sealed)
        if 'accepted_fields' in spec:
            for name in ('bounded_actor_step.py', 'actor_arithmetic.py', 'retained_kv_candidate.py',
                         'retained_router_candidate.py', 'retained_expert_candidate.py',
                         'retained_attention_candidate.py', 'policy_identity.py'):
                sealed(name)
            from bounded_actor_step import prepare, run
            # Preserve the original verl-before-task-adapter import order.
            worker, adapted, _ = construct(cfg, out)
            if 'generation' in spec:
                from bounded_actor_step import finalize_and_run
                if 'continuous_actor' in spec:
                    from resident import serve_formal
                    sealed('lora_row_precision.py')
                    from lora_row_precision import attach_hf
                    close_lora=attach_hf(adapted.get_base_model().model,spec['lora_row_geometry'])
                    try:
                        return serve_formal(worker,adapted,spec,out,rank,world,sealed,finalize_and_run)
                    finally:
                        try:close_lora(check_coverage=False)
                        finally:worker.close_actor_arithmetic()
                if 'lora_row_geometry' in spec:
                    sealed('lora_row_precision.py')
                    from lora_row_precision import attach_hf
                    close_lora=attach_hf(adapted.get_base_model().model,spec['lora_row_geometry'])
                    try:
                        result=finalize_and_run(worker,adapted,spec,out,rank,world,sealed)
                    except BaseException:
                        close_lora(check_coverage=False)
                        raise
                    commit(out/f'rank-{rank}/LORA-ARITHMETIC.json',close_lora())
                    return result
                return finalize_and_run(worker,adapted,spec,out,rank,world,sealed)
            if 'parity_restore_step' in spec:
                from bounded_actor_step import restore_updated_probe
                from updated_http_parity import evaluate
                sealed('updated_http_parity.py')
                sealed('retained_group_parity.py')
                ref,restored=restore_updated_probe(worker,spec)
                return evaluate(worker,adapted,spec,sealed,out,rank,world,ref,restored)
            batch = prepare(spec, sealed, world)
            if 'efficiency' in spec:
                sealed('actor_efficiency.py')
                from bounded_actor_step import recover
                recover(worker,spec,batch)
                from actor_efficiency import evaluate
                return evaluate(worker,adapted,spec,out,rank,world,batch)
            if 'updated_http_parity' in spec:
                sealed('updated_http_parity.py')
                sealed('retained_group_parity.py')
                if spec.get('trace_q_operands') or spec.get('projection_family_shadow'):
                    sealed('q_projection_trace.py')
                from bounded_actor_step import recover
                ref, restored = recover(worker,spec,batch)
                from updated_http_parity import evaluate
                return evaluate(worker,adapted,spec,sealed,out,rank,world,ref,restored)
            return run(worker, spec, out, rank, world, batch)
        worker, adapted, _ = construct(cfg, out)
        actor = worker.actor
        if 'actor_arithmetic_profile' in cfg:
            fresh = spec['fresh_group']
            for name in ('retained_group_parity.py', 'actor_arithmetic.py', 'retained_kv_candidate.py',
                         'retained_router_candidate.py', 'retained_expert_candidate.py', 'retained_attention_candidate.py'):
                sealed(name)
            from retained_group_parity import hf_scores
            actor.actor_module.eval()
            rs = hf_scores(worker, adapted, read(sealed(fresh['manifest'])),
                           read(sealed(fresh['result']))['fresh_group'], spec['files'][fresh['manifest']],
                           rank, world, actor_forward=True, frozen_reference=fresh.get('frozen_reference',False))
            commit(out/f'rank-{rank}'/'RESULT.json', dict(rank=rank, records=rs, updates=0,
                   actor_arithmetic_profile=cfg['actor_arithmetic_profile'], **worker.close_actor_arithmetic()))
            return
        case = next(c for c in read(sealed('retained-actions.json'))['cases'] if c['node'] == 'P')
        rec = read(sealed('P-BEHAVIOR.json'))['records'][case['index']]
        prep = torch.load(sealed('P-prepared.pt'), map_location='cpu', weights_only=False)
        require(prep['input_identity'] == rec['input_identity'] and
                prep['encoded']['input_ids'][0].tolist() == rec['prompt_token_ids'], 'retained input drift')
        tokens = rec['token_ids']
        require(1 <= spec['action_ordinal'] <= len(tokens), 'target outside retained actions')
        pos = len(rec['prompt_token_ids']) + spec['action_ordinal'] - 2
        from diagnostic_logits import observe
        sample = inputs_for(prep, tokens, adapted, device)
        body = adapted.get_base_model()
        text = body.model
        rows, hooks = {}, []
        def keep(key, x):
            x = x[0, pos].detach().clone()
            require(key not in rows and x.shape == (2048,) and x.dtype == torch.bfloat16,
                    'invalid actual predictor boundary: '+key)
            rows[key] = x
        def inner(module):
            # Register inside FSDP/checkpoint wrappers, observing original outputs.
            while type(module).__name__ in ('CheckpointWrapper', 'FullyShardedDataParallel'):
                module = (module._checkpoint_wrapped_module
                          if type(module).__name__ == 'CheckpointWrapper' else module.module)
            return module
        hooks.append(inner(text.layers[0]).register_forward_pre_hook(
            lambda m, a, kw: keep('decoder.input', a[0] if a else kw['hidden_states']), with_kwargs=True))
        for i, wrapped in enumerate(text.layers):
            layer = inner(wrapped)
            for site, mod in [('input_norm', layer.input_layernorm), ('post_norm', layer.post_attention_layernorm)]:
                hooks.append(mod.register_forward_hook(lambda m, a, v, k=f'{i}.{site}': keep(k, v)))
        hooks.append(text.norm.register_forward_hook(lambda m, a, v: keep('final_norm', v)))
        close_candidate = None
        do_vjp = spec.get('actor_vjp',False)
        if 'gate_candidate_layers' in spec:
            sealed('retained_expert_candidate.py')
            from retained_expert_candidate import attach_hf
            close_candidate = attach_hf(text, spec['gate_candidate_layers'], inner, **(dict(frozen_vjp=True) if do_vjp else {}))
        tile_set=spec.get('fresh_group',spec).get('tiled_layers')
        close_attention = None
        if 'attention_candidate_layers' in spec:
            sealed('retained_attention_candidate.py')
            from retained_attention_candidate import attach_hf as attach_attention
            close_attention = attach_attention(text, spec['attention_candidate_layers'], inner, **(dict(tiled_layers=tile_set) if tile_set else {}), **(dict(training_vjp=True) if do_vjp else {}))
        close_down = None
        if 'down_candidate_layers' in spec:
            sealed('retained_expert_candidate.py')
            from retained_expert_candidate import attach_hf as attach_expert
            close_down = attach_expert(text, spec['down_candidate_layers'], inner, stage='down', **(dict(frozen_vjp=True) if do_vjp else {}))
        actor.actor_module.eval()
        if 'fresh_group' in spec:
            for hook in hooks: hook.remove()
            sealed('retained_group_parity.py')
            from retained_group_parity import hf_scores
            fresh=spec['fresh_group']
            router=None
            if 'router_layers' in fresh:
                sealed('retained_router_candidate.py')
                from retained_router_candidate import attach as attach_router
                router=attach_router(text,fresh['router_layers'],side='hf',unwrap=inner)
            kv=None
            if "kv_layers" in fresh:
                sealed("retained_kv_candidate.py")
                from retained_kv_candidate import attach as attach_kv
                kv=attach_kv(text,fresh["kv_layers"],side="hf",unwrap=inner)
            rs=hf_scores(worker,adapted,read(sealed(fresh['manifest'])),read(sealed(fresh['result']))['fresh_group'],
                         spec['files'][fresh['manifest']],rank,world,trace_group=fresh.get('trace_group'),origin_indices=fresh.get('origin_indices'),
                         router_shadow=fresh.get('router_shadow',False))
            commit(out/f'rank-{rank}'/'RESULT.json',dict(rank=rank,records=rs,updates=0,kv_projection=kv() if kv else None,
                   down=close_down(),gate=close_candidate(),attention=close_attention(),router=router() if router else None))
            return
        close_kv=None
        if "kv_layers" in spec:
            sealed("retained_kv_candidate.py")
            from retained_kv_candidate import attach as attach_kv
            close_kv=attach_kv(text,spec["kv_layers"],side="hf",unwrap=inner,training_vjp=do_vjp)
        close_router=None
        if 'router_layers' in spec:
            sealed('retained_router_candidate.py')
            from retained_router_candidate import attach as attach_router
            close_router=attach_router(text,spec['router_layers'],side='hf',unwrap=inner,training_vjp=do_vjp)
        start = time.monotonic()
        policy = dict(schema='omni-response-logits-v1', vocabulary=spec['vocabulary'], max_bytes=16*1024**2)
        try:
            with torch.no_grad(), observe(actor.actor_module, sample, policy) as capture:
                _, logp = forward(actor, sample, temperature=1., feature_dtype=torch.bfloat16)
        finally:
            for h in hooks:
                h.remove()
        require(len(rows) == 98, 'incomplete actual model trace')
        gradient = {}
        if do_vjp:
            actor.actor_module.train()
            with torch.enable_grad():
                _, training_scores = forward(actor,sample,temperature=1.,feature_dtype=torch.bfloat16)
                objective = -training_scores.mean()
                objective.backward()
            loss=float(objective.detach())
            total, sq, missing = 0, 0., []
            for name,p in actor.actor_module.named_parameters():
                if not p.requires_grad or not p.numel():
                    continue
                if p.grad is None:
                    missing.append(name)
                    continue
                g=p.grad.detach().float()
                require(bool(g.isfinite().all()),'nonfinite adapter gradient: '+name)
                total += g.numel()
                sq += float(torch.linalg.vector_norm(g))**2
            counts=torch.tensor([total,len(missing)],device=device,dtype=torch.int64)
            dist.all_reduce(counts)
            sums=torch.tensor([sq],device=device,dtype=torch.float32)
            dist.all_reduce(sums)
            require(int(counts[0])==6684672 and bool(sums.isfinite().all()) and float(sums[0])>0 and counts[1]==0,'incomplete adapter gradient coverage')
            gradient=dict(objective='mean fixed-action negative log probability (diagnostic)',
                loss=loss,live_scores=training_scores.detach().float().cpu()[0].tolist(),
                forward_max_abs=float((logp-training_scores.detach()).abs().max()),
                globally_covered=int(counts[0]),global_norm=float(sums[0])**.5, optimizer_steps=0)
        down_candidate = close_down() if close_down else None
        candidate = close_candidate() if close_candidate else None
        attn_candidate = close_attention() if close_attention else None
        names = list(rows)
        packed = torch.stack(list(rows.values())).cpu()
        require(bool(packed.isfinite().all()), 'nonfinite layer boundary')
        row_hashes = {key: tensor_hash(row) for key, row in zip(names, packed)}
        result = dict(world_size=dist.get_world_size(), rank=rank, hardware=torch.npu.get_device_name(rank),
                      input_pins=spec['files'], asset_sha256=worker.base_identity['assets_sha256'],
                      internal_format=str(fmt), full_vocabulary=capture['metadata'],
                      argmax=capture['tensor'].argmax(dim=-1).tolist(), scores=logp[0].float().cpu().tolist(),
                      predictor_position=pos, rows=row_hashes, updates=0,
                      gate_candidate=candidate, attention_candidate=attn_candidate,
                      down_candidate=down_candidate,
                      kv_projection=close_kv() if close_kv else None,
                      router=close_router() if close_router else None,
                      actor_gradient=gradient,elapsed_seconds=time.monotonic()-start)
        commit(out/f'rank-{rank}'/'RESULT.json', result)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(read(a.config), a.input, a.output)
