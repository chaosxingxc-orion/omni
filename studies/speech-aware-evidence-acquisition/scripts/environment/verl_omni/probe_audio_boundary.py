"""CPU random-tiny Omni forward equivalence; no real weights or training."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, Qwen3OmniMoeThinkerForConditionalGeneration

from audio_feature_boundary import frozen_audio_inputs, forward_inputs


def probe(model_path):
    torch.manual_seed(20261101)
    torch.set_num_threads(4)
    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True).thinker_config
    for name, value in {'vocab_size':64, 'hidden_size':32, 'intermediate_size':64,
                        'num_hidden_layers':2, 'num_attention_heads':4, 'num_key_value_heads':2,
                        'head_dim':8, 'num_experts':4, 'num_experts_per_tok':2,
                        'moe_intermediate_size':16}.items():
        setattr(cfg.text_config, name, value)
    cfg.text_config.rope_parameters = {'rope_type':'default', 'rope_theta':1000000.0,
                                       'mrope_section':[1, 1, 2], 'mrope_interleaved':True}
    for name, value in {'encoder_layers':1, 'encoder_attention_heads':4, 'encoder_ffn_dim':64,
                        'd_model':32, 'output_dim':32, 'downsample_hidden_size':4}.items():
        setattr(cfg.audio_config, name, value)
    for name, value in {'depth':1, 'hidden_size':32, 'intermediate_size':64, 'num_heads':4,
                        'out_hidden_size':32, 'deepstack_visual_indexes':[]}.items():
        setattr(cfg.vision_config, name, value)
    for name, value in {'audio_token_id':5, 'image_token_id':6, 'video_token_id':7,
                        'audio_start_token_id':8, 'audio_end_token_id':9,
                        'vision_start_token_id':10, 'vision_end_token_id':11}.items():
        setattr(cfg, name, value)
    cfg._attn_implementation = 'sdpa'
    cfg.text_config._attn_implementation = 'sdpa'
    cfg.audio_config._attn_implementation = 'sdpa'
    cfg.vision_config._attn_implementation = 'sdpa'
    model = Qwen3OmniMoeThinkerForConditionalGeneration(cfg).float().eval().requires_grad_(False)
    # 16 audio feature frames produce two audio tokens in the native encoder.
    batch = {'input_ids':torch.tensor([[1, 8, 5, 5, 9, 2, 3]]),
             'attention_mask':torch.ones(1, 7, dtype=torch.long),
             'input_features':torch.randn(1, 128, 16),
             'feature_attention_mask':torch.ones(1, 16, dtype=torch.long),
             'response_mask':torch.tensor([[0, 0, 0, 0, 0, 1, 1]])}
    with torch.no_grad():
        original = model(**{k:v for k,v in batch.items() if k != 'response_mask'}, use_cache=False).logits
        prepared = frozen_audio_inputs(model, **batch)
        cached = model(**forward_inputs(prepared), use_cache=False).logits
    error = float((original-cached).abs().max())
    if not torch.allclose(original, cached, rtol=1e-5, atol=1e-6):
        raise ValueError(f'cached audio boundary changes model logits: {error}')
    return {'schema':'omni-frozen-audio-boundary-probe-v1', 'equivalent':True,
            'max_abs_logit_error':error, 'device':'cpu', 'real_weights_loaded':False,
            'random_tiny_model':True, 'forward_calls':2, 'backward_calls':0,
            'optimizer_steps':0, 'npu_work':False, 'production_training_ready':False,
            'scope':'Native tiny audio encoder/Omni RoPE/Thinker raw-vs-cached forward equivalence only.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('fresh result required')
    result = probe(args.model)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
