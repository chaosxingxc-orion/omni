"""Identical, no-update HF text path on both hosts, using a sealed decoder input."""
import hashlib
import inspect
import json
import os
from pathlib import Path


def score(spec, root, model_path, output):
    import torch
    import torch_npu
    from safetensors import safe_open
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
    from omni_fsdp_loading import load_cpu_thinker
    from dispatch_retained_hf_boundary import save
    def checked(name):
        f=root/name
        if hashlib.sha256(f.read_bytes()).hexdigest()!=spec['files'][name]:raise ValueError('changed input '+name)
        return f
    if hashlib.sha256(Path(inspect.getfile(hf)).read_bytes()).hexdigest()!=spec['hf_module_sha256']:
        raise ValueError('different HF implementation')
    if sum(int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))*1024<128*1024**3:
        raise ValueError('insufficient host memory for CPU model')
    torch.set_num_threads(4);torch.npu.set_device(0);torch.npu.config.allow_internal_format=False
    free,total=torch.npu.mem_get_info(0);limit=spec['device_budget']
    if free<limit+1024**3:raise ValueError('guard did not drain enough device memory')
    torch.npu.set_per_process_memory_fraction(limit/total,0)
    c=next(c for c in json.loads(checked('retained-actions.json').read_bytes())['cases'] if c['node']=='P')
    rec=json.loads(checked('P-BEHAVIOR.json').read_bytes())['records'][c['index']]
    prepared=torch.load(checked('P-prepared.pt'),map_location='cpu',weights_only=False)
    assert prepared['input_identity']==rec['input_identity'] and prepared['encoded']['input_ids'][0].tolist()==rec['prompt_token_ids']
    with safe_open(str(checked('HF.safetensors')),framework='pt',device='cpu') as f: emb=f.get_tensor('decoder_input')
    ids=torch.tensor([rec['prompt_token_ids']+rec['token_ids']]);mask=torch.ones_like(ids)
    assert list(emb.shape)==[1,ids.shape[1],2048] and emb.dtype==torch.bfloat16
    model,_=load_cpu_thinker(model_path);text=model.model
    pos,_=model.get_rope_index(input_ids=ids,attention_mask=mask,
            audio_seqlens=prepared['encoded']['feature_attention_mask'].sum(-1),use_audio_in_video=False)
    target=len(rec['prompt_token_ids'])+spec['action_ordinal']-2
    assert pos.shape==(3,1,ids.shape[-1]) and pos[:,0,target].eq(target).all()
    rows={}
    def record(key,x):
        x=x[0, target].detach().cpu().contiguous()
        if x.shape!=(2048,) or x.dtype!=torch.bfloat16 or key in rows or not x.isfinite().all():raise ValueError('invalid row '+key)
        rows[key]=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()
    def before(layer,args):layer.to(device='npu:0')
    def after(layer,args,value):layer.to(device='cpu')
    hooks=[]
    for n,layer in enumerate(text.layers):
        hooks.append(layer.register_forward_pre_hook(before))
        for label,mod in [('input_norm',layer.input_layernorm),('post_norm',layer.post_attention_layernorm)]:
            hooks.append(mod.register_forward_hook(lambda m,a,v,k=f'{n}.{label}':record(k,v)))
        hooks.append(layer.register_forward_hook(after))
    record('decoder.input',emb)
    text.rotary_emb.to('npu:0');text.norm.to('npu:0');model.lm_head.to('npu:0')
    with torch.inference_mode(),torch.autocast(device_type='npu',dtype=torch.bfloat16):
        hidden=text(inputs_embeds=emb.to('npu:0'),attention_mask=mask.to('npu:0'),
                    position_ids=pos.long().to('npu:0'),use_cache=False,return_dict=True).last_hidden_state
        record('final_norm',hidden)
        # Preserve the full-row projection shape used by the parent model.
        logits=model.lm_head(hidden)
    for h in hooks:h.remove()
    assert len(rows)==98
    r=len(rec['token_ids']);scores=logits[:,-r-1:-1,:].float().log_softmax(-1).gather(
            -1,ids[:,-r:].to('npu:0').unsqueeze(-1)).flatten().cpu().tolist()
    if not all(__import__('math').isfinite(x) for x in scores):raise ValueError('nonfinite original action score')
    save(output,dict(input_pins=spec['files'],hf_source=spec['hf_module_sha256'],scores=scores,
        parent_actor_scores=c['hf_logprobs'],token_ids=rec['token_ids'],predictor_position=target,
        rows=rows,peak_bytes=torch.npu.max_memory_allocated(),hardware=torch.npu.get_device_name(0),
        mode='HF BF16 full prefill, sequential weight placement',fsdp=False,updates=0))
if __name__=='__main__':
    import sys
    cfg=json.loads(Path(sys.argv[1]).read_bytes())
    score(cfg,Path('/e'),sys.argv[2],Path(sys.argv[3]))
