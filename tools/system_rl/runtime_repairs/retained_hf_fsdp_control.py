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
    from diagnostic_logits import observe
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
    case = next(c for c in read(sealed('retained-actions.json'))['cases'] if c['node'] == 'P')
    rec = read(sealed('P-BEHAVIOR.json'))['records'][case['index']]
    prep = torch.load(sealed('P-prepared.pt'), map_location='cpu', weights_only=False)
    require(prep['input_identity'] == rec['input_identity'] and
            prep['encoded']['input_ids'][0].tolist() == rec['prompt_token_ids'], 'retained input drift')
    tokens = rec['token_ids']
    require(1 <= spec['action_ordinal'] <= len(tokens), 'target outside retained actions')
    pos = len(rec['prompt_token_ids']) + spec['action_ordinal'] - 2
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
        worker, adapted, _ = construct(cfg, out)
        actor = worker.actor
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
        if 'gate_candidate_layers' in spec:
            sealed('retained_expert_candidate.py')
            from retained_expert_candidate import attach_hf
            close_candidate = attach_hf(text, spec['gate_candidate_layers'], inner)
        close_attention = None
        if 'attention_candidate_layers' in spec:
            sealed('retained_attention_candidate.py')
            from retained_attention_candidate import attach_hf as attach_attention
            close_attention = attach_attention(text, spec['attention_candidate_layers'], inner)
        actor.actor_module.eval()
        start = time.monotonic()
        policy = dict(schema='omni-response-logits-v1', vocabulary=spec['vocabulary'], max_bytes=16*1024**2)
        try:
            with torch.no_grad(), observe(actor.actor_module, sample, policy) as capture:
                _, logp = forward(actor, sample, temperature=1., feature_dtype=torch.bfloat16)
        finally:
            for h in hooks:
                h.remove()
        require(len(rows) == 98, 'incomplete actual model trace')
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
                      elapsed_seconds=time.monotonic()-start)
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
