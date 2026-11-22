"""Run a single mixed-prefill FIA directly on the resident, paged KV."""
import inspect
import sys
from pathlib import Path

from decode_lineage import digest, snapshot, changes
from project_prefill_tile import build as tile
from project_screen_lanes import read, save


def page_queries(ql, kl):
    """Left padding makes each request begin at its absolute query page."""
    if len(ql) != len(kl) or not ql:
        raise ValueError('unpaired spans')
    if any(type(q) != int or type(k) != int or not 0 < q <= k <= 24576
           for q, k in zip(ql, kl)):
        raise ValueError('invalid trailing queries')
    return [(k - q) % 128 for q, k in zip(ql, kl)]


def paged_call(impl, query, metadata, output, kv_cache, mask, plan):
    import torch
    import torch_npu
    from itertools import accumulate

    endpoints = metadata.actual_seq_lengths_q
    ql = [b-a for a,b in zip([0,*endpoints[:-1]],endpoints)]
    kl = list(metadata.seq_lens_list)
    leads = page_queries(ql, kl)
    real = endpoints[-1]
    if not metadata.causal or metadata.num_actual_tokens != real or real > len(query):
        raise ValueError('prefill query metadata differs')
    if (query.dtype != torch.bfloat16 or output.shape != query.shape or query.shape[1:] != (8,128)
            or output.device != query.device or output.dtype != query.dtype or torch.is_grad_enabled()):
        raise ValueError('expected BF16 inference with real TP4 heads')
    key, value = impl.key_cache, impl.value_cache
    if key is None or value is None:
        key, value = kv_cache
    table = metadata.block_tables[:len(kl)]
    if (key.shape[1:] != (128,1,128) or value.shape != key.shape or key.dtype != query.dtype
            or value.dtype != key.dtype or not query.device == key.device == value.device == table.device
            or table.dtype != torch.int32 or table.ndim != 2 or table.shape[0]!=len(kl)):
        raise ValueError('unexpected page geometry')
    # Validate every active physical ID once for this table within the forward.
    ident = (table.data_ptr(), tuple(table.shape), tuple(table.stride()), tuple(ql), tuple(kl), len(key))
    if ident not in plan:
        counts = [(k+127)//128 for k in kl]
        if max(counts) > table.shape[1]:
            raise ValueError('short block table')
        ids = torch.cat([t[:c] for t,c in zip(table,counts)])
        if bool(((ids < 0) | (ids >= len(key))).any()):
            raise ValueError('invalid physical page')
        ends = list(accumulate(q+l for q,l in zip(ql,leads)))
        slots = torch.cat([torch.arange(end-q,end,device=query.device) for q,end in zip(ql,ends)])
        plan[ident] = table, ends, slots
    _, ends, slots = plan[ident]
    packed=query.new_zeros((ends[-1],8,128)).index_copy_(0,slots,query[:real])
    # With paged TND, Q ends are cumulative and KV lengths are per sequence.
    # Keeping physical KV at the actual length avoids reading right-hand padding.
    result=torch_npu.npu_fused_infer_attention_score(
        query=packed, key=key.flatten(2), value=value.flatten(2),
        block_table=table, block_size=128, atten_mask=mask, input_layout='TND', sparse_mode=3,
        actual_seq_lengths=ends, actual_seq_lengths_kv=kl,
        num_heads=8, num_key_value_heads=1, scale=impl.scale)[0]
    output[:real].copy_(result.reshape_as(packed).index_select(0,slots))
    return output


def build(spec):
    if spec['attention_layout'] != 'paged_absolute_queries':
        raise ValueError('unknown cached prefill layout')
    tile(spec)
    root=Path(spec['projection']['target'])
    parent=Path(spec['measured_parent'])
    before=snapshot(parent)
    if snapshot(root)!=before:
        raise ValueError('prefill parent differs')
    rel='parents/jev-system-rl/implementation/decode_tiled_attention.py'
    source=root/'bundle'/rel
    rp=root/'bundle/ROWS.json'
    rows=read(rp)
    if rows['decode_attention_sha256']!=digest(source):
        raise ValueError('unexpected attention source')
    anchor='            n = len(query)\n'
    src=source.read_text()
    if src.count(anchor)!=1:
        raise ValueError('unsupported worker hook')
    extra="""            state=str(attn_metadata.attn_state).split('.')[-1]
            if state in ('ChunkedPrefill','PrefillCacheHit'):
                context=get_forward_context()
                if index == 0 or step_owner is not context:
                    cached.clear()
                    step_owner=context
                if causal_mask is None:
                    causal_mask=torch.ones((2048,2048),device=query.device,dtype=torch.bool).triu_(1)
                return paged_call(impl,query,attn_metadata,output,kv_cache,causal_mask,cached)
"""
    src=src.replace(anchor,extra+anchor)+'\n'+inspect.getsource(page_queries)+'\n'+inspect.getsource(paged_call)
    compile(src,rel,'exec')
    source.write_text(src)
    rows['decode_attention_sha256']=digest(source)
    save(rp,rows)
    for lane in spec['projection']['lanes']:
        file=root/'bundle'/('SERVER-'+lane+'.json')
        c=read(file)
        c['local_activation']['rows_ref'].update(sha256=digest(rp),bytes=rp.stat().st_size)
        save(file,c)
    delta=changes(before,snapshot(root))
    if delta['engine'] or set(delta['files'])!={rel} or set(delta['rows'])!={'decode_attention_sha256'}:
        raise ValueError('unrelated change')
    save(root/'PAGED-PREFILL.json',dict(changes=delta,parent_sha256=digest(parent/'CANDIDATE.json')))


if __name__=='__main__':
    build(read(Path(sys.argv[1])))
