"""Resident FSDP identity and separate observed endpoint custody; no JSON opt-in."""
import hashlib
import json
import re
from pathlib import Path
from dependencies import require, ROOT, load
IDENTITY_GAP = 'actual resident/checkpoint/behavior evidence required'
KEY = re.compile(r'.*?model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.lora_([AB])(?:\.default)?\.weight$')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def canonical(name):
    name = name.replace('_fsdp_wrapped_module.', '').replace('_checkpoint_wrapped_module.', '')
    match = KEY.fullmatch(name)
    require(match is not None, 'unexpected LoRA key: '+name)
    return 'base_model.model.model.layers.%s.self_attn.%s.lora_%s.weight' % match.groups()


def tensor_hash(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def describe(weights, *, alpha=16, rank=8):
    """Effective BA norm via rank-by-rank Gram matrices, without dense BA allocation."""
    import torch
    require(len({canonical(k) for k in weights}) == len(weights), 'duplicate canonical keys')
    weights = {canonical(k):v.detach().cpu().contiguous() for k,v in weights.items()}
    expected = {'base_model.model.model.layers.%d.self_attn.%s.lora_%s.weight' % (i,p,s)
                for i in range(48) for p in ('q_proj','k_proj','v_proj','o_proj') for s in ('A','B')}
    require(set(weights) == expected and sum(v.numel() for v in weights.values()) == 6684672,
            '48-layer 384-key 6684672-element adapter required')
    tensors = {}; layers = {}
    for key,value in sorted(weights.items()):
        require(value.ndim == 2 and torch.isfinite(value).all(), 'invalid resident adapter tensor')
        tensors[key] = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_hash(value))
    for layer in range(48):
        for projection in ('q_proj','k_proj','v_proj','o_proj'):
            prefix = f'base_model.model.model.layers.{layer}.self_attn.{projection}'
            a,b = (weights[prefix+f'.lora_{s}.weight'].double() for s in ('A','B'))
            out,inp={'q_proj':(4096,2048),'k_proj':(512,2048),'v_proj':(512,2048),'o_proj':(2048,4096)}[projection]
            require(tuple(a.shape)==(rank,inp) and tuple(b.shape)==(out,rank), 'LoRA projection shape differs')
            squared = float(((a@a.T) * (b.T@b)).sum()) * (alpha/rank)**2
            require(squared >= -1e-12, 'invalid effective delta norm')
            layers[f'{layer}.{projection}'] = max(0., squared)**.5
    semantics = dict(tensors=tensors, rank=rank, alpha=alpha, base='qwen3-omni-30b-a3b-instruct', thinking=False)
    return dict(schema='omni-resident-lora-identity-v1', **semantics, identity_sha256=digest(semantics),
                effective_delta_norms=layers, nonzero_effective_delta=any(v > 0 for v in layers.values()))


def resident_lora_weights(worker):
    """Collect only small real LoRA FSDP units on every rank in identical order."""
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    # Child-only summon performs lazy initialization. Establish the actual
    # root first, so a leaf adapter can never register itself as another root.
    from torch.distributed.fsdp._runtime_utils import _lazy_init
    root=worker.actor.actor_module
    require(isinstance(root,FSDP),'resident actor must own the FSDP root')
    _lazy_init(root,root)
    weights = {}
    for name,module in root.named_modules():
        clean = name.replace('_fsdp_wrapped_module.', '').replace('_checkpoint_wrapped_module.', '')
        if isinstance(module, FSDP) and re.search(r'\.lora_[AB]\.default$', clean):
            with FSDP.summon_full_params(module, recurse=False, writeback=False, rank0_only=False):
                for key,value in module.module.named_parameters(recurse=False):
                    full = canonical(clean+'.'+key)
                    require(full not in weights, 'duplicate FSDP adapter ownership')
                    weights[full] = value.detach().cpu().clone()
    require(weights, 'resident FSDP LoRA units absent')
    return weights


def resident(worker):
    import torch
    import torch.distributed as dist
    weights=resident_lora_weights(worker)
    require(getattr(worker,'base_identity',None) is not None, 'actual loader base asset identity absent')
    local = dict(base=worker.base_identity,master=describe(weights), publication=describe({k:v.to(torch.bfloat16) for k,v in weights.items()}))
    peers = [None]*dist.get_world_size(); dist.all_gather_object(peers, digest(local))
    require(len(set(peers)) == 1, 'resident FSDP adapter differs across ranks')
    return dict(**local, world_size=dist.get_world_size(), rank_digests=peers)


def artifact(path):
    from safetensors.torch import load_file
    return describe(load_file(str(path), device='cpu'))


def behavior_binding(identity, behavior):
    """Recompute existing HTTP witness, including device readback and response bytes."""
    from entry import pinned
    ref = behavior['endpoint_witness_ref']
    path = ROOT/'policy-iteration-successor'
    import sys
    old_path = sys.path[:]
    saved = {n:sys.modules.pop(n,None) for n in ('io_utils','protocol','lineage')}
    try:
        sys.path.insert(0,str(path))
        protocol = load(path/'protocol.py','dp8_policy_protocol')
        bind_readonly_protocol(protocol, sys.modules['io_utils'])
        from publication import validate_publication
        import types
        sys.modules['lineage']=types.SimpleNamespace(publication=lambda ref:validate_publication(ref,verify_parity=False,verify_checkpoint_shards=False))
        witness = protocol.verify(ref)
    finally:
        sys.path[:] = old_path
        for name,value in saved.items():
            sys.modules.pop(name,None)
            if value is not None: sys.modules[name]=value
    activation = json.loads(pinned(witness['activation_ref']).read_text())
    if activation['base']:
        require(not identity['publication']['nonzero_effective_delta'], 'base behavior with nonzero actor')
        policy=json.loads(pinned(witness['policy_ref']).read_text())
        seal=json.loads(pinned(policy['asset_manifest_ref']).read_text())
        require(seal['status']=='CLOSED_IMMUTABLE' and seal['assets']==policy['assets'], 'base immutable asset seal differs')
        require(policy['version']=='base-sha256:'+digest(policy['assets']), 'base identity/version differs')
        require(identity['base']['asset_manifest_ref']==policy['asset_manifest_ref'], 'same model name with different base weights')
    else:
        loaded = artifact(pinned(activation['weights_ref']))
        from publication import validate_publication
        published=validate_publication(json.loads(pinned(witness['policy_ref']).read_text())['publication_ref'],verify_parity=False,verify_checkpoint_shards=False)
        require(published['resident_identity']['base']==identity['base'], 'updated behavior base weights differ')
        require(loaded['identity_sha256'] == identity['publication']['identity_sha256'],
                'resident/publication/service tensor identity mismatch')
        require(all(row['compared_original_keys'] == 384 for row in activation['workers']), 'incomplete readback')
    return dict(policy_identity_sha256=identity['publication']['identity_sha256'],
                endpoint_witness_ref=ref, policy_version=witness['policy_version'])


def bind_readonly_protocol(protocol, io):
    """Replay a closed witness through the authenticated location map.

    Keep logical references unchanged. Verification must never create remote
    paths on the client, nor write into the closed content-addressed objects.
    """
    from dependencies import local_path
    original_pin = io.pin

    def pin(path):
        result = original_pin(local_path(path))
        return dict(result, path=str(Path(path).resolve()))

    def raw(ref):
        require(pin(ref['path']) == {k: ref[k] for k in ('path','sha256','bytes')},
                'PIN_DRIFT:' + ref['path'])
        payload = local_path(ref['path']).read_bytes()
        require(len(payload) == ref['bytes'] and hashlib.sha256(payload).hexdigest() == ref['sha256'],
                'INPUT_CHANGED:' + ref['path'])
        return payload

    def save(path, value):
        ref = pin(path)
        require(json.loads(raw(ref)) == value, 'IMMUTABLE_COLLISION:' + str(path))
        return ref

    io.pin, io.raw, io.save = pin, raw, save
    protocol.pin, protocol.raw, protocol.save = pin, raw, save
    original = protocol.original
    def mapped_capture():
        capture = original()
        def read_ref(ref):
            value = json.loads(raw(ref))
            for key in ref.get('pointer', []):value = value[key]
            return value
        capture.read_ref = read_ref
        return capture
    protocol.original = mapped_capture


def require_update_identity(*, config, window=None, worker=None, identity=None):
    require(window is not None and identity is not None, 'live identity and sealed window required')
    require(identity['world_size'] == 8, 'DP8 resident identity required')
    binding = behavior_binding(identity, window)
    require(binding['policy_version'] == window['policy_version'], 'behavior window policy differs')
    mode = config.get('update_mode','bounded_first_update')
    require(mode in ('bounded_first_update','train'), 'unknown update mode')
    if mode == 'bounded_first_update':
        require((config['max_optimizer_updates'] == 1 or config.get('resident',{}).get('enabled')) and config['resume_ref'] is None
                and window['policy_step']==0,
                'bootstrap limited to first single update')
    else:
        require(identity['publication']['nonzero_effective_delta'], 'train requires real updated delta')
    return binding


def base_assets(model_path, manifest_ref):
    """Reuse closed weight hashes; inspect actual loader files, hash small config.

    Ceph is readonly in the fixed container. No repeated full-model rescan.
    The closed seal is the provenance authority, observed file facts prevent a
    same-name path from silently selecting a different configuration/assets.
    """
    from entry import pinned
    path=Path(model_path).resolve();seal=json.loads(pinned(manifest_ref).read_text())
    require(seal['status']=='CLOSED_IMMUTABLE' and seal['assets'],'base asset seal absent')
    observed=[]
    for ref in seal['assets']:
        asset=Path(ref['path']).resolve()
        require(asset.is_relative_to(path) and asset.is_file() and asset.stat().st_size==ref['bytes'],'loader asset path/size differs')
        stat=asset.stat();observed.append(dict(path=str(asset),sha256=ref['sha256'],bytes=stat.st_size))
    config=path/'config.json'
    require(any(Path(r['path']).resolve()==config and r['sha256']==__import__('hashlib').sha256(config.read_bytes()).hexdigest() for r in seal['assets']),'actual base config hash differs')
    require(any(Path(r['path']).suffix=='.safetensors' for r in seal['assets']),'actual weight artifacts not in base seal')
    return dict(asset_manifest_ref=manifest_ref,assets_sha256=digest(seal['assets']),model_path=str(path),observed_assets=observed,readonly_seal_reused=True)
