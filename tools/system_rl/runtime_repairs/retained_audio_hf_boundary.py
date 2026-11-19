"""Replay the complete frozen HF audio tower and stop at the actual decoder call."""
import hashlib
import inspect
import json
import os
from pathlib import Path


def run(root, model, out):
    import torch
    import torch_npu
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import initialization
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeThinkerConfig
    spec = json.loads((root/'REPLAY.json').read_bytes())
    def checked(key):
        r = spec[key]
        p = root/r['file']
        if p.parent != root or hashlib.sha256(p.read_bytes()).hexdigest() != r['sha256']:
            raise ValueError('retained replay identity differs: '+key)
        return p
    sample = torch.load(checked('prepared'), weights_only=False, map_location='cpu')
    action = json.loads(checked('behavior').read_bytes())['records'][spec['action_index']]
    enc = sample['encoded']
    if sample['input_identity'] != action['input_identity'] or enc['input_ids'][0].tolist() != action['prompt_token_ids']:
        raise ValueError('input and behavior are not paired')
    if hashlib.sha256(Path(inspect.getfile(hf)).read_bytes()).hexdigest() != spec['hf_module_sha256']:
        raise ValueError('HF implementation drift')
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    limit = spec['max_device_bytes']
    free, total = torch.npu.mem_get_info(0)
    if free < limit + 1024**3: raise ValueError('insufficient device headroom')
    torch.npu.set_per_process_memory_fraction(limit/total, 0)
    cfg = Qwen3OmniMoeThinkerConfig(**json.loads((model/'config.json').read_bytes())['thinker_config'])
    cfg._attn_implementation = 'sdpa'
    with init_empty_weights(include_buffers=False), initialization.no_init_weights():
        tower = hf.Qwen3OmniMoeAudioEncoder(cfg.audio_config)
    index = json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map']
    def tensor(key):
        with safe_open(str(model/index[key]), framework='pt', device='cpu') as f:
            t = f.get_tensor(key)
            if t.dtype != torch.bfloat16: raise ValueError('expected frozen BF16 weights')
            return t
    names = list(tower.state_dict())
    tower.load_state_dict({k:tensor('thinker.audio_tower.'+k) for k in names}, strict=True, assign=True)
    tower.eval().requires_grad_(False).to('npu:0')
    class Stop(Exception): pass
    observations = {}
    def record(name, v):
        if name in observations: raise ValueError('duplicate forward boundary')
        observations[name] = v.detach().cpu().contiguous()
    class Entry(torch.nn.Module):
        def get_input_embeddings(self): return self.embedding
        def forward(self, *, inputs_embeds, **kwargs):
            record('decoder_input', inputs_embeds)
            raise Stop
    thinker = hf.Qwen3OmniMoeThinkerForConditionalGeneration.__new__(hf.Qwen3OmniMoeThinkerForConditionalGeneration)
    torch.nn.Module.__init__(thinker)
    thinker.config, thinker.audio_tower, thinker.model = cfg, tower, Entry()
    thinker.spatial_merge_size = cfg.vision_config.spatial_merge_size
    thinker.rope_deltas = None
    embed = tensor('thinker.model.embed_tokens.weight')
    thinker.model.embedding = torch.nn.Embedding.from_pretrained(embed, freeze=True).to('npu:0')
    thinker.eval()
    for i, layer in enumerate(tower.layers):
        layer.register_forward_hook(lambda m, args, out, i=i: record(f'layer.{i}.out', out[0]))
    tower.layers[0].register_forward_pre_hook(lambda m,args:record('layer.0.in', args[0]))
    tower.register_forward_hook(lambda m,args,out:record('audio_out',out.last_hidden_state))
    ids = torch.tensor([action['prompt_token_ids']+action['token_ids']],device='npu:0')
    mask = enc['feature_attention_mask'].to('npu:0')
    try:
        with torch.no_grad(), torch.autocast('npu',dtype=torch.bfloat16):
            thinker(input_ids=ids, input_features=enc['input_features'].to('npu:0',torch.bfloat16),
                    feature_attention_mask=mask, audio_feature_lengths=mask.sum(-1),
                    attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
    except Stop: pass
    audio = observations['audio_out']
    h0 = observations['decoder_input']
    dest = (ids.cpu()[0] == cfg.audio_token_id).nonzero().flatten()
    assert len(observations)==len(tower.layers)+3 and torch.equal(h0[0,dest],audio)
    assert all(torch.isfinite(t).all() for t in observations.values())
    from safetensors.torch import save_file
    target = out/'HF.safetensors'
    if target.exists() or (out/'RESULT.json').exists(): raise FileExistsError('immutable replay output')
    save_file(observations, str(target))
    with target.open('rb') as f: os.fsync(f.fileno())
    result = dict(schema='hf-retained-audio-boundaries-v1',inputs=spec, weights=len(names), audio_attention=tower.config._attn_implementation,
                  layers=len(tower.layers), audio_shape=list(audio.shape), decoder_shape=list(h0.shape),
                  placement_count=len(dest), placement_exact=True, peak_bytes=torch.npu.max_memory_allocated(),
                  tensors_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  fsdp_worker=False, language_layers_executed=0, prompt_tokens=len(action['prompt_token_ids']))
    with (out/'RESULT.json').open('x') as f:
        json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(out,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    print(json.dumps(result))
if __name__=='__main__':
    run(Path('/e'),Path('/model'),Path('/hot'))
