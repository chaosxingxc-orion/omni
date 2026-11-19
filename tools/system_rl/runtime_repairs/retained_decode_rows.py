"""Observe one original predictor position during actual cached decode and prefill."""
import hashlib


def begin(worker, mode, target, attention_layer=None, prefix_origin=None, moe_origin=False,
          expert_operands=False, prefix_scan=None, origin_layers=2, attention_shape_shadow=False):
    import torch
    from family_precision import ATTR
    model = worker.model_runner.get_model()
    text = model.language_model.model
    if len(text.layers)!=48 or type(target)!=int or target<1 or mode not in ('decode','prefill'):
        raise ValueError('unsupported retained language trace')
    if hasattr(worker,'_retained_predictor_trace'):
        raise ValueError('previous trace still active')
    if mode=='prefill' and not hasattr(worker,'_retained_decode_first'):
        raise ValueError('missing same-worker natural predictor trace')
    handles=[]; rows={};state={'absolute':target};device_rows={}
    attn_rows={};undo=None
    if attention_layer is not None:
        if type(attention_layer)!=int or attention_layer not in range(48):raise ValueError('invalid attention layer')
        from retained_attention_operands import observe
        undo=observe(text.layers[attention_layer].self_attn.attn.impl,state,mode,attn_rows,
                     shape_shadow=attention_shape_shadow)
    close_prefix=None
    close_moe=None
    close_scan=None
    if prefix_scan:
        from retained_prefix_scan import observe as scan
        close_scan=scan(text,prefix_scan['rows'],prefix_scan['layers'])
    if prefix_origin is not None:
        from retained_prefix_origin import attach
        if not 0<=prefix_origin<target:
            raise ValueError('invalid history position')
        close_prefix=attach(text,prefix_origin,target,origin_layers)
    if moe_origin:
        from retained_moe_origin import hook
        if prefix_origin is None:
            raise ValueError('MoE observation needs a history position')
        close_moe=hook(text,prefix_origin,expert_operands)
    family=getattr(model,ATTR)['audit']['modules']
    def attn_states():
        return {str(i):dict(family[f'layer.{i}.attention']['states']) for i in range(48)}
    start=attn_states()
    def on_entry(module,args):
        pos, hidden, _ = args
        if device_rows: raise ValueError('overlapping predictor traces')
        mask=pos.eq(target)
        if mask.ndim==2: mask=mask.all(dim=0)
        ids=torch.where(mask)[0].tolist()
        if not ids:
            state.pop('at',None)
            return
        if len(ids)!=1 or rows:
            raise ValueError('target row ambiguous across language forwards')
        state.update(at=ids[0],forward_rows=hidden.shape[0])
        keep('decoder.input',hidden)
    def keep(key, tensor):
        if 'at' not in state:return
        tensor=tensor[0] if isinstance(tensor,tuple) else tensor
        if key in device_rows or tensor.ndim!=2 or tensor.dtype!=torch.bfloat16 or tensor.shape[-1]!=2048:
            raise ValueError('unexpected language trace boundary '+key)
        device_rows[key]=tensor[state['at']].detach().clone()
    def end_step(m,args,value):
        keep('final_norm',value)
        if not device_rows:return
        keys=list(device_rows); stacked=torch.stack(list(device_rows.values())).cpu()
        rows.update(zip(keys,stacked.unbind(0)))
        device_rows.clear();state.pop('at',None)
    handles.append(text.layers[0].register_forward_pre_hook(on_entry))
    for n,layer in enumerate(text.layers):
        for name,module in [('input_norm',layer.input_layernorm),('post_norm',layer.post_attention_layernorm)]:
            key=f'{n}.{name}'
            handles.append(module.register_forward_hook(lambda m,a,v,k=key:keep(k,v)))
    handles.append(text.norm.register_forward_hook(end_step))
    worker._retained_predictor_trace=(handles,rows,state,start,attn_states,mode,target,attention_layer,attn_rows,undo,close_prefix,close_moe,close_scan)
    return worker.rank


def finish(worker):
    import torch
    handles,rows,state,start,states,mode,target,attention_layer,attn_rows,undo,close_prefix,close_moe,close_scan=worker._retained_predictor_trace
    prefix_rows=close_prefix() if close_prefix else None
    moe_rows=close_moe() if close_moe else None
    scan_rows=close_scan() if close_scan else None
    if undo:undo()
    for h in handles:h.remove()
    del worker._retained_predictor_trace
    if len(rows)!=98:raise ValueError(f'incomplete retained language trace: {len(rows)}/98')
    counts={k:{s:n-start[k].get(s,0) for s,n in v.items() if n!=start[k].get(s,0)}
            for k,v in states().items()}
    result=dict(rank=worker.rank,predictor_position=target,forward_rows=state['forward_rows'],
                corrected_attention_states=counts,mode=mode,observed_boundaries=len(rows))
    if mode=='decode':
        worker._retained_decode_first=(target,rows,attn_rows,prefix_rows,moe_rows,scan_rows)
    else:
        earlier,first,attn_first,prefix_first,moe_first,scan_first=worker._retained_decode_first
        del worker._retained_decode_first
        if target!=earlier or list(rows)!=list(first):raise ValueError('unaligned language signatures')
        if attention_layer is not None:
            from retained_attention_operands import compare, delta
            result['attention_layer']=attention_layer
            result['attention_operands']=compare(attn_first,attn_rows)
            if attention_layer is not None and 'shape_shadow' in attn_rows:
                shadow=attn_rows['shape_shadow']
                shadow['live_operands_equal']=all(result['attention_operands'][k]['unequal']==0
                                                 for k in ('query','key_prefix','value_prefix'))
                if shadow['live_operands_equal']:
                    shadow['actual_cached_vs_shadow']=delta(attn_first['attention_out'],attn_rows['shape_shadow_single_out'])
                result['attention_shape_shadow']=shadow
        if close_prefix:
            from retained_prefix_origin import compare
            result['prefix_origin']=compare(prefix_first,prefix_rows)
        if close_moe:
            from retained_moe_origin import compare
            result['moe_origin']=compare(moe_first,moe_rows,worker.rank)
        if close_scan:
            from retained_prefix_scan import compare as compare_scan
            result['prefix_scan']=compare_scan(scan_first,scan_rows)
        diff={}
        for k,x in rows.items():
            y=first[k]
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('nonfinite language trace')
            diff[k]=dict(unequal=int(torch.count_nonzero(x!=y)),max_abs=float((x.float()-y.float()).abs().max()),
                         prefill_sha256=hashlib.sha256(x.view(torch.uint8).numpy()).hexdigest(),
                         decode_sha256=hashlib.sha256(y.view(torch.uint8).numpy()).hexdigest())
        result['boundaries']=diff
        result['first_observed_difference']=next((k for k,v in diff.items() if v['unequal']),None)
    return result
