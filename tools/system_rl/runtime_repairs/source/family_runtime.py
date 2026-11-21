"""One config-driven attachment lifecycle for validated numerical families.

Explicitly opt in per family. This is an integration API, not an automatic claim
that an entire decoder is aligned. Frozen legacy diagnostic packages are untouched.
"""


def validate_config(config):
    if config.get('schema')!='omni-family-runtime-v1':raise ValueError('unknown runtime schema')
    supported={'schema','layers','attention','norm','final_norm','source_pins','preserve_original',
               'qkv','row','qk_rope','router','router_selection','moe','adapter_scales','down_weight_layout',
               'skip_empty_expert_chunks','packed_active_limit'}
    if set(config)-supported:raise ValueError('unknown runtime configuration keys')
    layers=config['layers']
    if not layers or any(type(i) is not int or not 0<=i<48 for i in layers) or len(set(layers))!=len(layers):
        raise ValueError('unique existing decoder layers required')
    families=('attention','norm','final_norm','qkv','row','qk_rope','router','router_selection','moe')
    for name in (*families,'preserve_original','skip_empty_expert_chunks'):
        if type(config.get(name,False)) is not bool:raise ValueError('boolean family switches required')
    if not any(config.get(k,False) for k in families):raise ValueError('no selected family')
    limit = config.get('packed_active_limit',0)
    if type(limit) is not int or not 0 <= limit <= 32 or (limit and not config.get('skip_empty_expert_chunks')):
        raise ValueError('bounded packed experts require TP count agreement')


def attach(text,config):
    from attention_family_runtime import attach as attach_attention
    from norm_family_runtime import attach as attach_norm
    validate_config(config)
    layers=config['layers'];pins=config['source_pins']
    if max(layers)>=len(text.layers):raise ValueError('configured layers exceed model depth')
    undo=[];audits={};closed=False
    def install(name,fn,module,pin,**kwargs):
        audit,restore=fn(module,source_sha256=pins[pin],preserve_original=config.get('preserve_original',True),**kwargs)
        audits[name]=audit;undo.append(restore)
    def restore():
        nonlocal closed
        if closed:return
        errors=[]
        for fn in reversed(undo):
            try:fn()
            except Exception as error:errors.append(str(error))
        if errors:raise ValueError('family restoration failed: '+'; '.join(errors))
        closed=True
    try:
        for i in layers:
            layer=text.layers[i]
            if config.get('norm',False):
                for name in ('input_layernorm','post_attention_layernorm'):
                    install(f'layer.{i}.{name}',attach_norm,getattr(layer,name),'norm')
            if config.get('attention',False):
                install(f'layer.{i}.attention',attach_attention,layer.self_attn.attn.impl,'attention')
            if config.get('qkv',False) or config.get('row',False):
                from projection_family_runtime import attach as attach_projection
                for kind,module in [('qkv',layer.self_attn.qkv_proj),('row',layer.self_attn.o_proj)]:
                    if config.get(kind,False):install(f'layer.{i}.{kind}',attach_projection,module,kind,
                        kind=kind,adapter_scales=config['adapter_scales'])
            if config.get('qk_rope',False):
                from qk_family_runtime import attach as attach_qk
                install(f'layer.{i}.qk_rope',attach_qk,layer.self_attn,'qk_rope')
            if config.get('router',False):
                from router_family_runtime import attach as attach_router
                install(f'layer.{i}.router',attach_router,layer.mlp,'router')
            if config.get('router_selection',False):
                from router_selection_runtime import attach as attach_selection
                install(f'layer.{i}.router_selection',attach_selection,layer.mlp,'router_selection')
        if config.get('final_norm',False):install('final_norm',attach_norm,text.norm,'norm')
        if config.get('moe',False):
            from moe_family_runtime import attach as attach_moe
            audit,cleanup=attach_moe({i:text.layers[i].mlp for i in layers},source_pins=pins['moe'],
                preserve_original=config.get('preserve_original',True),
                down_weight_layout=config.get('down_weight_layout','contiguous'),
                skip_empty_chunks=config.get('skip_empty_expert_chunks',False),
                packed_active_limit=config.get('packed_active_limit',0))
            audits['moe']=audit;undo.append(cleanup)
    except BaseException:
        restore();raise
    return dict(schema='omni-family-runtime-audit-v1',modules=audits,production_acceptance=False),restore
