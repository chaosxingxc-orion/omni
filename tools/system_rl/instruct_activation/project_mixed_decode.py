"""Batch verified one-token decode rows in mixed steps of the tiled BF16 engine."""
from pathlib import Path
import inspect
import sys


def mixed_shape(meta, limit):
    if str(meta.attn_state).split('.')[-1] != 'ChunkedPrefill':
        return None
    d = meta.num_decodes
    if not (type(d) is int and 2 <= d <= limit and meta.num_decode_tokens == d and meta.num_prefills > 0):
        return None
    ends, lengths = list(meta.actual_seq_lengths_q), list(meta.seq_lens_list)
    # Do not reinterpret short extends, dummy/padded requests, or other layouts.
    if (len(ends) != d+meta.num_prefills or len(lengths) != len(ends)
            or ends[:d] != list(range(1,d+1)) or ends[-1] != meta.num_actual_tokens):
        return None
    sizes = [b-a for a,b in zip([0]+ends[:-1],ends)]
    if not all(type(n) is int and type(k) is int and 0 < n <= k <= 24576
               for n,k in zip(sizes,lengths)):
        return None
    return d, ends[-1], sizes[d:], lengths[d:]


def edit(src):
    def change(a,b):
        nonlocal src
        if src.count(a)!=1: raise ValueError('unexpected tiled parent: '+a[:50])
        src = src.replace(a,b)
    change('    cached = {}', '''    from retained_attention_candidate import causal_attention
    from paged_attention_contract import logical_kv
    tail_compute = causal_attention(absolute_tile=128)
    seen_mixed = False
    cached = {}''')
    lo=src.index('        def call(query, key, value, attn_metadata, output, kv_cache=None):')
    hi=src.index('            if (not attn_metadata.causal',lo)
    dispatch=src[lo:hi]
    if 'return original' not in dispatch or 'DecodeOnly' not in dispatch:
        raise ValueError('unrecognized original routing')
    change(dispatch,'''        def batched(query, attn_metadata, output, kv_cache, lengths):
            nonlocal causal_mask, step_owner
            n = len(query)
''')
    change('attn_metadata.num_actual_tokens != n or query.dtype','query.dtype')
    change('            lengths = tuple(attn_metadata.seq_lens_list)\n','')
    pure_guard=dispatch[dispatch.index('            if not ('):dispatch.index('                return original')]
    change('        return call', '''        def call(query, key, value, attn_metadata, output, kv_cache=None):
            nonlocal seen_mixed
            n = len(query)
            split = mixed_shape(attn_metadata, max_batch)
            if split is not None:
                d, end, ql, kl = split
                if end > n or output.shape != query.shape or output.dtype != query.dtype or output.device != query.device:
                    raise ValueError('invalid parent output ABI')
                # The backend already wrote all current K/V before this read-only FIA boundary.
                batched(query[:d], attn_metadata, output[:d], kv_cache, tuple(attn_metadata.seq_lens_list[:d]))
                kc,vc=impl.key_cache,impl.value_cache
                if kc is None or vc is None:
                    kc,vc=kv_cache
                kk,vv=logical_kv(kc,vc,attn_metadata.block_tables[d:d+len(kl)],kl)
                output[d:end].copy_(tail_compute(query[d:end],kk,vv,ql,kl,scale=impl.scale))
                if not seen_mixed:
                    import logging
                    logging.getLogger('vllm').info('mixed single-token FIA active: layer=%d decode=%d prefill=%d',index,d,len(ql))
                    seen_mixed=True
                return output
'''+pure_guard+'''                return original(query, key, value, attn_metadata, output, kv_cache=kv_cache)
            if attn_metadata.num_actual_tokens != n:
                raise ValueError('actual decode length changed')
            return batched(query, attn_metadata, output, kv_cache, tuple(attn_metadata.seq_lens_list))
        return call''')
    src += '\n\n' + inspect.getsource(mixed_shape)
    compile(src,'decode_tiled_attention.py','exec')
    return src


def build(spec):
    from project_prefill_tile import build as tile
    from project_screen_lanes import read,save
    from decode_lineage import changes,digest,snapshot
    if spec['batch_scope'] != 'mixed_single_token_prefix': raise ValueError('unsupported scope')
    tile(spec)
    parent=Path(spec['measured_parent']); root=Path(spec['projection']['target']); before=snapshot(parent)
    if snapshot(root)!=before: raise ValueError('measured parent differs')
    rel='parents/jev-system-rl/implementation/decode_tiled_attention.py'
    f=root/'bundle'/rel; rp=root/'bundle/ROWS.json'; rows=read(rp)
    if rows['tiled_layers']!=list(range(48)) or rows['decode_attention_sha256']!=digest(f):
        raise ValueError('unexpected arithmetic scope')
    f.write_text(edit(f.read_text())); rows['decode_attention_sha256']=digest(f); save(rp,rows)
    for name in spec['projection']['lanes']:
        p=root/'bundle'/('SERVER-'+name+'.json'); s=read(p)
        s['local_activation']['rows_ref'].update(sha256=digest(rp),bytes=rp.stat().st_size); save(p,s)
    diff=changes(before,snapshot(root))
    if diff['engine'] or set(diff['files'])!={rel} or set(diff['rows'])!={'decode_attention_sha256'}:
        raise ValueError('unexpected runtime changes')
    save(root/'MIXED-DECODE.json',dict(parent_sha256=digest(parent/'CANDIDATE.json'),changes=diff))

if __name__=='__main__':
    from project_screen_lanes import read
    build(read(Path(sys.argv[1])))
