"""Rank-dependent counts and deterministic enlargement of a published PEFT adapter."""
import math


def validate(profile):
    t=profile['training'];r=t['lora_rank'];alpha=t['lora_alpha'];drop=t['lora_dropout']
    if (profile['schema']!='p-lora-profile-v1' or type(r) is not int or r<=0 or
            type(alpha) not in (int,float) or not math.isfinite(alpha) or alpha<=0 or
            type(drop) not in (int,float) or not 0<=drop<1 or
            type(profile['inference']['max_lora_rank']) is not int or profile['inference']['max_lora_rank']<r):
        raise ValueError('invalid paired LoRA training/service profile')
    return t


def load_initial_publication(adapted, directory, config):
    """Load the exact expanded BF16 adapter shared with the inference service."""
    import json
    import torch
    from pathlib import Path
    from safetensors.torch import load_file
    from policy_identity import canonical
    directory=Path(directory)
    saved=json.loads((directory/'adapter_config.json').read_bytes())
    if (saved['r'],saved['lora_alpha'],saved['lora_dropout'])!=tuple(config[k] for k in
            ('lora_rank','lora_alpha','lora_dropout')):
        raise ValueError('published LoRA profile differs from actor')
    weights=load_file(str(directory/'adapter_model.safetensors'))
    with torch.no_grad():
        for name,parameter in adapted.named_parameters():
            if parameter.requires_grad:
                weight=weights.pop(canonical(name))
                if weight.dtype!=torch.bfloat16 or parameter.shape!=weight.shape or parameter.is_meta:
                    raise ValueError('published LoRA tensor differs from actor: '+name)
                parameter.copy_(weight)
    if weights:raise ValueError('unconsumed initial adapter tensors')
