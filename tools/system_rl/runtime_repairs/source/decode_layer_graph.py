"""Static-shape pure decode compute segments around the real paged KV backend."""
from functools import wraps


def validate(config):
    shapes = config.get('batch_sizes')
    if set(config) != {'batch_sizes','reserve_bytes'} or type(config['reserve_bytes']) is not int or config['reserve_bytes'] <= 0:
        raise ValueError('explicit graph batch sizes and free HBM reserve required')
    if not isinstance(shapes,list) or not shapes or any(type(n) is not int or not 1 <= n <= 8 for n in shapes) or sorted(set(shapes)) != shapes:
        raise ValueError('ordered unique decode batch sizes required')


def attach(worker, config):
    import torch
    import torch_npu
    from vllm.distributed import tensor_model_parallel_all_reduce as reduce
    from vllm.distributed import tensor_model_parallel_all_gather as gather
    from rope_table_compute import inverse_frequencies, tables
    from local_activation import is_decode
    from decode_expert_owners import grouped

    validate(config)
    text = worker.model_runner.get_model().language_model.model
    rank = worker.rank
    owners = worker._decode_owner_weights
    scale = worker._decode_projection_scale
    device = next(text.parameters()).device
    stream = torch.npu.Stream()
    inv = inverse_frequencies(1000000.,128,device='cpu')
    slots = torch.arange(33,device=device,dtype=torch.long)
    graphs = {}
    positions_key = None
    trig = None
    def norm(x,w,eps=1e-6):
        f=x.float()
        return w*(f*torch.rsqrt(f.square().mean(-1,keepdim=True)+eps)).bfloat16()

    def compute_projection(x, mask, pieces, qkv, length):
        outs=[]
        for base, a, b in pieces:
            raw,=torch_npu.npu_grouped_matmul(x=[x.contiguous()],weight=[base],
                split_item=2,group_list_type=1,group_type=0,group_list=length)
            u,=torch_npu.npu_grouped_matmul(x=[x.contiguous()],weight=[a.T.contiguous().unsqueeze(0)],
                split_item=2,group_list_type=1,group_type=0,group_list=length)
            delta=torch.nn.functional.linear(u,b)
            if qkv:
                width=raw.shape[-1]//4
                raw=raw[:,rank*width:(rank+1)*width]
                delta=delta[:,rank*width:(rank+1)*width]
            outs.append(torch.where(mask,raw+delta*scale,raw))
        return torch.cat(outs,-1) if qkv else outs[0]

    def capture(callback, inputs):
        graph=torch.npu.NPUGraph()
        inputs=tuple(v.clone() for v in inputs)
        # Warm actual kernels before graph capture on its own stream.
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            callback(*inputs)
        stream.synchronize()
        with torch.npu.graph(graph,stream=stream):
            output=callback(*inputs)
        def run(*values):
            for buf,value in zip(inputs,values,strict=True):
                buf.copy_(value)
            graph.replay()
            return output
        return run

    def build(i,n,hidden,residual,cos,sin,mask):
        ref=text.layers[i]
        attention=ref.self_attn
        gu,down,runner=owners[i]
        if gu.shape[0]!=33 or down.shape[0]!=33:
            raise ValueError('graph requires a distinct dummy expert')
        identity,base=worker._current_projection_operands()
        if identity is None:
            raise ValueError('decode graph requires verified LoRA operands')
        qparts=base[attention.qkv_proj.base_layer.weight.data_ptr()]
        oparts=base[attention.o_proj.base_layer.weight.data_ptr()]
        router=runner.gate.weight_fp32.bfloat16().T.contiguous().unsqueeze(0)
        length=torch.tensor([n],device=device,dtype=torch.int64)
        def before(x,res,c,s,mask):
            total=x if i==0 else x+res
            src=norm(total,ref.input_layernorm.weight)
            q,k,v=compute_projection(src,mask,qparts,True,length).split([1024,128,128],dim=-1)
            def rotate(y,w):
                normalized=norm(y.reshape(n,-1,128),w)
                rot=torch.cat([-normalized[...,64:],normalized[...,:64]],dim=-1)
                return (normalized*c[:,None]+rot*s[:,None]).reshape(n,-1)
            return rotate(q,attention.q_norm.weight),rotate(k,attention.k_norm.weight),v,total
        def after(full_attn,res,mask):
            total=compute_projection(full_attn,mask,oparts,False,length)+res
            h=norm(total,ref.post_attention_layernorm.weight)
            logits,=torch_npu.npu_grouped_matmul(x=[h.contiguous()],weight=[router],
                split_item=2,group_list_type=1,group_type=0,group_list=length)
            w,ids=torch.topk(torch.nn.functional.softmax(logits,dtype=torch.float32,dim=-1),8,dim=-1)
            w=(w/w.sum(dim=-1,keepdim=True)).bfloat16()
            flat=ids.flatten()
            own=(flat>=32*rank)&(flat<32*(rank+1))
            local=torch.where(own,flat-32*rank,32)
            order=torch.argsort(local.float())
            counts=(local[:,None]==slots).sum(dim=0,dtype=torch.int64)
            first,second=grouped(h[order//8],gu,counts).chunk(2,dim=-1)
            y=grouped(torch.nn.functional.silu(first)*second,down,counts)
            rows=torch.empty_like(y).index_copy_(0,order,y)
            return torch.where(own[:,None],rows,0),w,total
        free,_=torch.npu.mem_get_info()
        if free < config['reserve_bytes']:
            raise ValueError('decode graph free HBM reserve exceeded')
        one=capture(before,(hidden,residual,cos,sin,mask))
        dummy=hidden.new_zeros((n,4096))
        two=capture(after,(dummy,residual,mask))
        return one,two

    def wrap(i,layer):
        old=layer.forward
        @wraps(old)
        def forward(positions,hidden_states,residual):
            nonlocal trig,positions_key
            n=len(hidden_states)
            if n not in config['batch_sizes'] or not is_decode(n) or (residual is None)!=(i==0):
                return old(positions,hidden_states,residual)
            if i==0:
                c,s=tables(positions,inv,sections=[24,20,20],output_dtype=hidden_states.dtype,compute_device=str(device))
                trig=c[0],s[0]
                positions_key=(positions.data_ptr(),tuple(positions.shape))
            if positions_key!=(positions.data_ptr(),tuple(positions.shape)):
                raise ValueError('position identity changed inside a decode batch')
            p=layer.self_attn.qkv_proj.punica_wrapper
            mask=(p._token_lora_indices[:n]==0)[:,None] if not p.no_lora else torch.zeros((n,1),device=device,dtype=torch.bool)
            rs=hidden_states if residual is None else residual
            if (i,n) not in graphs:
                graphs[i,n]=build(i,n,hidden_states,rs,*trig,mask)
            one,two=graphs[i,n]
            q,k,v,total=one(hidden_states,rs,*trig,mask)
            # The existing backend owns writes, block tables, prefix hits and tiled FIA.
            attention_output=layer.self_attn.attn(q,k,v)
            full=gather(attention_output.contiguous(),dim=1)
            local,w,total=two(full,total,mask)
            rows=reduce(local)
            out=(rows.reshape(n,8,2048)*w[:,:,None]).sum(dim=1)
            return out,total
        return forward

    for i,layer in enumerate(text.layers):
        layer.forward=wrap(i,layer)
    worker._clear_decode_graphs=graphs.clear
