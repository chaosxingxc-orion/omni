"""Static-shape pure decode compute segments around the real paged KV backend."""
from functools import wraps


def validate(config):
    shapes = config.get('batch_sizes')
    if set(config)-{'pool_mode','capture_segments','prefill_tile_rows','tiled_attention_batch','page_gather'} != {'batch_sizes','reserve_bytes'} or type(config['reserve_bytes']) is not int or config['reserve_bytes'] <= 0:
        raise ValueError('explicit graph batch sizes and free HBM reserve required')
    if config.get('pool_mode','private') not in ('private','per_batch'):
        raise ValueError('unknown graph pool mode')
    if config.get('capture_segments','both') not in ('both','before','after'):
        raise ValueError('unknown graph segments')
    if 'tiled_attention_batch' in config and (type(config['tiled_attention_batch']) is not int or not 1<=config['tiled_attention_batch']<=128):
        raise ValueError('explicit tiled decode attention batch limit required')
    if 'page_gather' in config and ('tiled_attention_batch' not in config or config['page_gather'] not in ('interleaved','separate')):
        raise ValueError('unsupported page gather configuration')
    tile=config.get('prefill_tile_rows')
    if 'prefill_tile_rows' in config and (type(tile) is not int or not 16<=tile<=512 or tile%16):
        raise ValueError('prefill tile must be a multiple of 16 in [16,512]')
    if not isinstance(shapes,list) or not shapes or any(type(n) is not int or not 1 <= n <= 128 for n in shapes) or sorted(set(shapes)) != shapes:
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
    if 'tiled_attention_batch' in config:
        from decode_tiled_attention import install
        install(text, config['tiled_attention_batch'],config.get('page_gather','separate'))
    rank = worker.rank
    owners = worker._decode_owner_weights
    scale = worker._decode_projection_scale
    device = next(text.parameters()).device
    stream = torch.npu.Stream()
    inv = inverse_frequencies(1000000.,128,device='cpu')
    slots = torch.arange(33,device=device,dtype=torch.long)
    graphs = {}
    pools = {}
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

    def capture(callback, inputs, pool):
        graph=torch.npu.NPUGraph()
        inputs=tuple(v.clone() for v in inputs)
        # Warm actual kernels before graph capture on its own stream.
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            callback(*inputs)
        stream.synchronize()
        with torch.npu.graph(graph,stream=stream,pool=pool):
            output=callback(*inputs)
        # Captured kernels retain raw device pointers, not Python owners.
        # Keep callbacks (and closed-over router/length tensors) alive as
        # long as the graphs, including between asynchronous replays.
        def run(*values, _owner=callback):
            for buf,value in zip(inputs,values,strict=True):
                buf.copy_(value)
            graph.replay()
            return output
        return run

    def build(i,n,hidden,residual,cos,sin,mask,*,eager=False):
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
        if eager:
            return before,after
        free,_=torch.npu.mem_get_info()
        if free < config['reserve_bytes']:
            raise ValueError('decode graph free HBM reserve exceeded')
        # These segments replay serially in capture order for a fixed shape.
        # Separate shapes may interleave; they must not borrow the same pool.
        if config.get('pool_mode')=='per_batch' and n not in pools:
            pools[n]=torch.npu.graph_pool_handle()
        pool=pools.get(n)
        segments=config.get('capture_segments','both')
        one=capture(before,(hidden,residual,cos,sin,mask),pool) if segments!='after' else before
        dummy=hidden.new_zeros((n,4096))
        two=capture(after,(dummy,residual,mask),pool) if segments!='before' else after
        return one,two

    def wrap(i,layer):
        old=layer.forward
        @wraps(old)
        def forward(positions,hidden_states,residual):
            nonlocal trig,positions_key
            n=len(hidden_states)
            pure = is_decode(n) if n in config['batch_sizes'] else False
            tile = config.get('prefill_tile_rows')
            if not pure and tile:
                from vllm.forward_context import get_forward_context
                meta=get_forward_context().attn_metadata
                entry=meta.get(layer.self_attn.attn.layer_name) if isinstance(meta,dict) else None
                state=str(getattr(entry,'attn_state','')).split('.')[-1]
                tiled=(state in ('PrefillNoCache','PrefillCacheHit','ChunkedPrefill','DecodeOnly')
                       and entry.num_actual_tokens==n and n>0)
            else:
                tiled=False
            if (not pure and not tiled) or (residual is None)!=(i==0):
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
            if tiled:
                # Split only positionwise arithmetic; attention still receives the original
                # complete batch, block tables and positions. Reuse grouped BF16 operands.
                # Bound closure/cache size: fixed full tiles plus a per-call tail.
                def helpers(lo,hi):
                    size=hi-lo
                    key=('tile',i,size)
                    if key in graphs:
                        return graphs[key]
                    pair=build(i,size,hidden_states[lo:hi],rs[lo:hi],trig[0][lo:hi],
                               trig[1][lo:hi],mask[lo:hi],eager=True)
                    if size==tile:
                        graphs[key]=pair
                    return pair
                pieces=[]
                totals=[]
                afters=[]
                for lo in range(0,n,tile):
                    hi=min(lo+tile,n)
                    one,two=helpers(lo,hi)
                    q,k,v,total=one(hidden_states[lo:hi],rs[lo:hi],trig[0][lo:hi],trig[1][lo:hi],mask[lo:hi])
                    pieces.append((q,k,v))
                    totals.append(total)
                    afters.append(two)
                q,k,v=(torch.cat(items,dim=0) for items in zip(*pieces,strict=True))
                full=gather(layer.self_attn.attn(q,k,v).contiguous(),dim=1)
                routed=[];weights=[];new_residual=[]
                for j,lo in enumerate(range(0,n,tile)):
                    hi=min(lo+tile,n)
                    local,w,total=afters[j](full[lo:hi],totals[j],mask[lo:hi])
                    routed.append(local);weights.append(w);new_residual.append(total)
                rows=reduce(torch.cat(routed,dim=0)).reshape(n,8,2048)
                return (rows*torch.cat(weights,dim=0)[:,:,None]).sum(dim=1),torch.cat(new_residual,dim=0)
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
    def clear():
        graphs.clear()
        pools.clear()
    worker._clear_decode_graphs=clear
