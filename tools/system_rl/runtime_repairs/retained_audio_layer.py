"""Read-only observation of one audio block, including active TP weight shards."""
import hashlib
import json


def observe(layer, backend, number, rank, root):
    import torch
    from safetensors.torch import save_file
    from dispatch_retained_hf_boundary import save
    assert backend in ('hf','vl') and rank in range(4)
    seen, parameters, handles = {}, {}, []
    blocks = dict(norm=layer.self_attn_layer_norm, out=layer.self_attn.out_proj,
                  norm2=layer.final_layer_norm, fc1=layer.fc1, fc2=layer.fc2)
    if backend == 'hf':
        blocks.update({n:getattr(layer.self_attn,n+'_proj') for n in ('q','k','v')})
    else:
        blocks['qkv']=layer.self_attn.qkv_proj
    def keep(key,x):
        if key in seen: raise ValueError('repeated block observation: '+key)
        seen[key]=x.detach().to(device='cpu',copy=True).contiguous()
    def pin(x):
        y=x.detach().cpu().contiguous()
        return dict(shape=list(y.shape),dtype=str(y.dtype),sha256=hashlib.sha256(y.view(torch.uint8).numpy()).hexdigest())
    def param(name, key, value):
        # HF digest for the same logical shard the real vLLM rank loaded.
        branches = zip(('q','k','v'),value.chunk(3,0)) if name=='qkv' else [(name,value)]
        for branch,v in branches:
            axis = 0 if branch in ('q','k','v','fc1') else 1 if branch in ('out','fc2') and key=='weight' else None
            split=v.chunk(4,axis) if backend=='hf' and axis is not None else [v]*4
            for r in (range(4) if backend=='hf' else [rank]):
                parameters[f'{branch}.{key}.rank{r}']=pin(split[r])
    def install(name,mod):
        def pre(m,args):
            keep(name+'.in',args[0])
            for key in ('weight','bias'): param(name,key,getattr(m,key))
        def post(m,args,out):
            keep(name+'.out',out[0] if isinstance(out,tuple) else out)
        handles.extend([mod.register_forward_pre_hook(pre),mod.register_forward_hook(post)])
    for n,m in blocks.items():install(n,m)
    def lens(m,args,kwargs):
        value=kwargs.get('cu_seqlens') if 'cu_seqlens' in kwargs else args[1]
        keep('cu_seqlens',value)
    handles.append(layer.register_forward_pre_hook(lens,with_kwargs=True))
    def finish():
        for h in handles:h.remove()
        assert len(seen)==2*len(blocks)+1
        path=root/'LAYER.safetensors'
        if path.exists(): raise FileExistsError(path)
        save_file(seen,str(path))
        import os
        with path.open('rb') as f:os.fsync(f.fileno())
        doc=dict(layer=number,backend=backend,rank=rank,weights=parameters,
                 norm_eps=[blocks[n].eps for n in ('norm','norm2')],
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        save(root/'LAYER.json',doc)
        return doc['sha256']
    return finish
