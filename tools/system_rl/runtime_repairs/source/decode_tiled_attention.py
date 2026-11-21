"""Batched BF16 FIA for single-token decode with absolute 128-row query tiles."""
from itertools import accumulate


def page_plan(table, lengths, blocks, block_size):
    import torch
    if (table.ndim != 2 or table.shape[0] != len(lengths)
            or table.dtype not in (torch.int32, torch.int64) or block_size != 128
            or not lengths or any(type(n) is not int or not 0 < n <= 24576 for n in lengths)):
        raise ValueError('expected validated TP4 length and page geometry')
    counts = [(n + 127) // 128 for n in lengths]
    if max(counts) > table.shape[1]:
        raise ValueError('short active block table')
    ids = torch.cat([row[:count] for row, count in zip(table, counts, strict=True)]).long()
    if bool(((ids < 0) | (ids >= blocks)).any()):
        raise ValueError('invalid active cache block')
    ends = list(accumulate(n * 128 for n in counts))
    # Prefix position and requested query row are independent for each request.
    device = table.device
    valid = torch.cat([torch.arange(count * 128, device=device) < n
                       for n, count in zip(lengths, counts, strict=True)])
    valid = valid[:, None, None]
    query_rows = torch.tensor([i * 128 + (n - 1) % 128 for i, n in enumerate(lengths)],
                              dtype=torch.long, device=device)
    return ids, valid, query_rows, ends


def install(text, max_batch):
    import torch
    import torch_npu
    from vllm.forward_context import get_forward_context

    cached = {}
    causal_mask = None
    step_owner = None

    def wrap(impl, index):
        original = impl.forward_fused_infer_attention
        if ((impl.num_heads, impl.num_kv_heads, impl.head_size) != (8, 1, 128)
                or impl.scale != 128 ** -.5 or impl.sliding_window is not None
                or impl.sinks is not None or getattr(impl, 'alibi_slopes', None) is not None
                or any(getattr(impl, mode, False) for mode in (
                    'enable_c8_quant', 'pcp_enabled', 'is_pcp_decode_sharded', 'use_bnsd_kv_cache'))):
            raise ValueError('unsupported attention backend for tiled batch decode')

        def call(query, key, value, attn_metadata, output, kv_cache=None):
            nonlocal causal_mask, step_owner
            n = len(query)
            if not (1 <= n <= max_batch and str(attn_metadata.attn_state).split('.')[-1] == 'DecodeOnly'
                    and list(attn_metadata.actual_seq_lengths_q) == list(range(1, n + 1))):
                return original(query, key, value, attn_metadata, output, kv_cache=kv_cache)
            if (not attn_metadata.causal or torch.is_grad_enabled() or query.shape[1:] != (8, 128)
                    or attn_metadata.num_actual_tokens != n or query.dtype != torch.bfloat16
                    or output.shape != query.shape or output.device != query.device or output.dtype != query.dtype):
                raise ValueError('decode tensor or causal contract changed')
            kc, vc = impl.key_cache, impl.value_cache
            if kc is None or vc is None:
                kc, vc = kv_cache
            if (kc.ndim != 4 or kc.shape[1:] != (128, 1, 128) or vc.shape != kc.shape
                    or kc.dtype != query.dtype or vc.dtype != query.dtype
                    or kc.device != query.device or vc.device != query.device):
                raise ValueError('expected original page cache and BF16 K/V')
            lengths = tuple(attn_metadata.seq_lens_list)
            if len(lengths) != n:
                raise ValueError('mismatched decode request count')
            table = attn_metadata.block_tables[:n]
            context = get_forward_context()
            if index == 0 or step_owner is not context:
                cached.clear()
                step_owner = context
            # The scheduler may reuse table storage on the next token. Only
            # reuse page addresses within this forward, never reuse KV values.
            identity = (table.data_ptr(), tuple(table.stride()), tuple(table.shape), lengths, len(kc))
            if identity not in cached:
                cached[identity] = page_plan(table, lengths, len(kc), kc.shape[1])
            indices, valid, query_rows, ends = cached[identity]
            def gather_pages(cache):
                packed = cache.index_select(0, indices).flatten(0, 1)
                return torch.where(valid, packed, 0)
            q = query.new_zeros((n * 128, 8, 128)).index_copy_(0, query_rows, query)
            if causal_mask is None:
                causal_mask = torch.ones((2048, 2048), dtype=torch.bool, device=q.device).triu_(1)
            result = torch_npu.npu_fused_infer_attention_score(
                query=q, key=gather_pages(kc), value=gather_pages(vc), atten_mask=causal_mask,
                input_layout='TND', sparse_mode=3, actual_seq_lengths=list(range(128, n * 128 + 1, 128)),
                actual_seq_lengths_kv=ends, num_heads=8, num_key_value_heads=1, scale=impl.scale)[0]
            output.copy_(result.reshape_as(q).index_select(0, query_rows))
            return output
        return call

    for index, layer in enumerate(text.layers):
        impl = layer.self_attn.attn.impl
        impl.forward_fused_infer_attention = wrap(impl, index)
