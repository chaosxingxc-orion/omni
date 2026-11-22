"""Gather KV pages in batch and keep the dense FIA absolute tile geometry."""
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


def operands(impl, query, metadata, output, kv_cache):
    import torch
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
    return ql, kl, leads, real, key, value, table


def dense_plan(table, ql, kl, leads, blocks):
    import torch
    from itertools import accumulate

    counts = [(k + 127) // 128 for k in kl]
    if max(counts) > table.shape[1]:
        raise ValueError('short block table')
    ids = torch.cat([row[:count] for row, count in zip(table, counts, strict=True)]).long()
    if bool(((ids < 0) | (ids >= blocks)).any()):
        raise ValueError('invalid active physical page')
    kv_ends = list(accumulate(c * 128 for c in counts))
    query_sizes = [c * 128 - (k - q - lead)
                   for c, k, q, lead in zip(counts, kl, ql, leads, strict=True)]
    query_ends = list(accumulate(query_sizes))
    slots = torch.cat([torch.arange(end - size + lead, end - size + lead + q, device=table.device)
                       for end, size, lead, q in zip(query_ends, query_sizes, leads, ql, strict=True)])
    valid = torch.cat([torch.arange(c * 128, device=table.device) < k
                       for c, k in zip(counts, kl, strict=True)])[:, None, None]
    return ids, valid, slots, query_ends, kv_ends


def dense_call(impl, query, metadata, output, kv_cache, mask, plan):
    import torch
    import torch_npu

    ql, kl, leads, real, key, value, table = operands(impl, query, metadata, output, kv_cache)
    ident = (table.data_ptr(), tuple(table.shape), tuple(table.stride()), tuple(ql), tuple(kl), len(key))
    if ident not in plan:
        plan[ident] = dense_plan(table, ql, kl, leads, len(key))
    ids, valid, slots, q_ends, kv_ends = plan[ident]
    packed = query.new_zeros((q_ends[-1], 8, 128)).index_copy_(0, slots, query[:real])
    def pack(cache):
        return torch.where(valid, cache.index_select(0, ids).flatten(0, 1), 0)
    result = torch_npu.npu_fused_infer_attention_score(
        query=packed, key=pack(key), value=pack(value), atten_mask=mask, input_layout='TND',
        sparse_mode=3, actual_seq_lengths=q_ends, actual_seq_lengths_kv=kv_ends,
        num_heads=8, num_key_value_heads=1, scale=impl.scale)[0]
    output[:real].copy_(result.reshape_as(packed).index_select(0, slots))
    return output


def build(spec):
    mode = spec['attention_layout']
    if mode != 'dense_planned_pages':
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
                return dense_call(impl,query,attn_metadata,output,kv_cache,causal_mask,cached)
"""
    helpers = [page_queries, operands, dense_call, dense_plan]
    src = src.replace(anchor, extra + anchor) + '\n' + '\n'.join(inspect.getsource(fn) for fn in helpers)
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
    save(root/'PAGED-PREFILL.json',dict(layout=mode,changes=delta,parent_sha256=digest(parent/'CANDIDATE.json')))


if __name__=='__main__':
    build(read(Path(sys.argv[1])))
