"""Frozen audio-to-Thinker boundary; standalone preparation, not a trainer plugin.

Reuse the model's audio encoder and RoPE before training the language adapter.
Only audio/text is supported here. Image/video require their DeepStack path.
"""
from __future__ import annotations

import torch


def collate_audio_features(examples):
    """Pad feature time axes before verl's generic dim-0 multimodal concatenation.

Each example carries processor output [audios, mel_bins, frames] and a binary
feature_attention_mask [audios, frames]. No feature is dropped for unequal length.
"""
    if not examples:
        raise ValueError('nonempty processor outputs required')
    mel_bins, feature_dtype, feature_device = None, None, None
    longest = 0
    for example in examples:
        if set(example) != {'input_features', 'feature_attention_mask'}:
            raise ValueError('exact audio processor feature and mask fields required')
        features, mask = example['input_features'], example['feature_attention_mask']
        if features.ndim != 3 or mask.ndim != 2 or features.shape[::2] != mask.shape:
            raise ValueError('audio feature/mask shapes differ')
        if not features.is_floating_point() or mask.device != features.device:
            raise ValueError('floating features and colocated mask required')
        if not torch.all((mask == 0) | (mask == 1)) or torch.any(mask.sum(-1) == 0):
            raise ValueError('nonempty binary audio mask required')
        if torch.any(mask[:, 1:] > mask[:, :-1]):
            raise ValueError('processor audio masks must be right padded')
        signature = features.shape[1], features.dtype, features.device
        if mel_bins is None:
            mel_bins, feature_dtype, feature_device = signature
        elif signature != (mel_bins, feature_dtype, feature_device):
            raise ValueError('audio feature batch dtype/device/mel bins differ')
        longest = max(longest, features.shape[-1])
    result = []
    for example in examples:
        padding = longest - example['input_features'].shape[-1]
        result.append({key: torch.nn.functional.pad(value, (0, padding))
                       for key, value in example.items()})
    return result


def frozen_audio_inputs(thinker, *, input_ids, attention_mask, input_features,
                        feature_attention_mask, response_mask):
    """Produce detached merged embeddings and exact Omni positions for one batch.

Return full-sequence tensors, not a prompt KV cache. A new adapter changes the
decoder states, so decoder KV values must never be cached across adapter updates.
All tensors must be reconstructed per rollout sequence before actor packing.
"""
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or response_mask.shape != input_ids.shape:
        raise ValueError('full input, attention and response masks must align')
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError('binary text attention mask required')
    if not torch.all((response_mask == 0) | (response_mask == 1)) or torch.any(response_mask > attention_mask):
        raise ValueError('response-only mask must select valid tokens')
    if torch.any((input_ids == thinker.config.image_token_id) | (input_ids == thinker.config.video_token_id)):
        raise ValueError('image/video require their DeepStack multimodal path')
    for module in (thinker.audio_tower, thinker.get_input_embeddings()):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError('audio tower and token embeddings must be frozen')
    if thinker.audio_tower.training:
        raise ValueError('frozen audio tower must remain in eval mode')
    collate_audio_features([{'input_features': input_features,
                            'feature_attention_mask': feature_attention_mask}])
    audio_mask = input_ids == thinker.config.audio_token_id
    if torch.any(response_mask[audio_mask] != 0) or torch.any(attention_mask[audio_mask] != 1):
        raise ValueError('audio placeholders must be valid prompt-only tokens')
    with torch.no_grad():
        features = thinker.get_audio_features(input_features=input_features,
            feature_attention_mask=feature_attention_mask, return_dict=True).last_hidden_state
        embeds = thinker.get_input_embeddings()(input_ids)
        if features.ndim != 2 or features.shape != (int(audio_mask.sum()), embeds.shape[-1]):
            raise ValueError('encoded audio length and placeholder count differ')
        if not torch.isfinite(features).all():
            raise ValueError('audio features must be finite')
        embeds = embeds.masked_scatter(audio_mask.unsqueeze(-1), features.to(embeds))
        positions, rope_deltas = thinker.get_rope_index(input_ids=input_ids,
            attention_mask=attention_mask, use_audio_in_video=False,
            audio_seqlens=feature_attention_mask.sum(-1))
    if positions.shape != (3, *input_ids.shape):
        raise ValueError('expected exact three-axis Omni RoPE')
    return {'input_ids': input_ids.detach().clone(), 'inputs_embeds': embeds.detach(),
            'attention_mask': attention_mask.detach().clone(), 'position_ids': positions.detach(),
            'loss_mask': response_mask.detach().clone(), 'rope_deltas': rope_deltas.detach()}


def forward_inputs(prepared):
    """Arguments for the Thinker forward, excluding loss masks used by verl."""
    return {key: prepared[key] for key in ('input_ids', 'inputs_embeds', 'attention_mask', 'position_ids')}
