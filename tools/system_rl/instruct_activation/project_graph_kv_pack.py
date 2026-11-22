"""Pack the identical dense K/V pages for FIA in one BF16 copy kernel."""
import ast
from pathlib import Path
import sys
from decode_lineage import snapshot, changes, digest as file_digest
from io_utils import doc, pin, digest, require, save
from project_graph_tile_rows import build as build_tile
from project_prefill_timing import once
from project_screen_lanes import save as write_json


KERNEL = '''
import triton
import triton.language as tl

@triton.jit
def _copy_kv_pages(K, V, IDS, VALID, KD, VD, TOTAL,
                   KS0: tl.constexpr, KS1: tl.constexpr, KS3: tl.constexpr,
                   VS0: tl.constexpr, VS1: tl.constexpr, VS3: tl.constexpr,
                   BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // 128
    col = offsets % 128
    in_range = offsets < TOTAL
    valid = tl.load(VALID + row, mask=in_range, other=0).to(tl.int1)
    page = tl.load(IDS + row // 128, mask=in_range, other=0).to(tl.int64)
    within = row % 128
    key = tl.load(K + page * KS0 + within * KS1 + col * KS3, mask=in_range & valid, other=0.)
    val = tl.load(V + page * VS0 + within * VS1 + col * VS3, mask=in_range & valid, other=0.)
    tl.store(KD + offsets, key, mask=in_range)
    tl.store(VD + offsets, val, mask=in_range)


def pack_pair(key, value, ids, valid):
    rows = ids.numel() * 128
    if not (valid.is_contiguous() and ids.is_contiguous() and valid.numel() == rows):
        raise ValueError("invalid dense page map")
    packed_key = key.new_empty((rows, 1, 128))
    packed_value = value.new_empty((rows, 1, 128))
    size = rows * 128
    _copy_kv_pages[(triton.cdiv(size, _PACK_BLOCK),)](
        key, value, ids, valid, packed_key, packed_value, size,
        key.stride(0), key.stride(1), key.stride(3),
        value.stride(0), value.stride(1), value.stride(3), BLOCK=_PACK_BLOCK)
    return packed_key, packed_value
'''


def edit(source, block):
    require(type(block) is int and block in (1024, 2048, 4096, 8192), 'unsupported BF16 copy block')
    source = once(source, 'from itertools import accumulate\n\n\ndef page_plan',
                  'from itertools import accumulate\n\n_PACK_BLOCK = '+str(block)+'\n'+KERNEL+'\n\ndef page_plan')
    source = once(source, '''            def gather_pages(cache):
                packed = cache.index_select(0, indices).flatten(0, 1)
                return torch.where(valid, packed, 0)''',
                  '            packed_key, packed_value = pack_pair(kc, vc, indices, valid)')
    source = once(source, 'key=gather_pages(kc), value=gather_pages(vc)', 'key=packed_key, value=packed_value')
    source = once(source, '''    def pack(cache):
        return torch.where(valid, cache.index_select(0, ids).flatten(0, 1), 0)''',
                  '    packed_key, packed_value = pack_pair(key, value, ids, valid)')
    source = once(source, 'key=pack(key), value=pack(value)', 'key=packed_key, value=packed_value')
    compile(source, 'decode_tiled_attention.py', 'exec')
    return source


def edit_tail_clear(source):
    """Gather the exact same pages, then zero just the invalid tail rows in owned storage."""
    helper = """def tail_rows(lengths, counts, device):
    import torch
    from itertools import accumulate
    ends = accumulate(count * 128 for count in counts)
    invalid = [row for length, end in zip(lengths, ends, strict=True)
               for row in range(end - (-length % 128), end)]
    return torch.tensor(invalid, device=device, dtype=torch.long)


def gather_zero_tail(cache, ids, invalid):
    # index_select owns this allocation; never write into the resident cache.
    gathered = cache.index_select(0, ids).flatten(0, 1)
    if invalid.numel():
        gathered.index_fill_(0, invalid, 0)
    return gathered


"""
    source = once(source, 'def page_plan(', helper + 'def page_plan(')
    source = once(source, """    valid = torch.cat([torch.arange(count * 128, device=device) < n
                       for n, count in zip(lengths, counts, strict=True)])
    valid = valid[:, None, None]""", '    valid = tail_rows(lengths, counts, device)')
    source = once(source, """                packed = cache.index_select(0, indices).flatten(0, 1)
                return torch.where(valid, packed, 0)""", '                return gather_zero_tail(cache, indices, valid)')
    source = once(source, """    valid = torch.cat([torch.arange(c * 128, device=table.device) < k
                       for c, k in zip(counts, kl, strict=True)])[:, None, None]""",
                  '    valid = tail_rows(kl, counts, table.device)')
    source = once(source, 'return torch.where(valid, cache.index_select(0, ids).flatten(0, 1), 0)',
                  'return gather_zero_tail(cache, ids, valid)')
    compile(source, 'decode_tiled_attention.py', 'exec')
    return source


def edit_interleaved(source, reference, *, include_decode):
    """Use the previously measured block-major gather on both dense FIA paths."""
    original = Path(reference['path'])
    require(pin(original) == reference, 'interleaved gather source changed')
    code = original.read_text()
    helper, = [node for node in ast.parse(code).body
               if isinstance(node, ast.FunctionDef) and node.name == 'interleaved_pages']
    source = once(source, 'def page_plan(', ast.get_source_segment(code, helper) + '\n\n\ndef page_plan(')
    if include_decode:
        source = once(source, '''            def gather_pages(cache):
                packed = cache.index_select(0, indices).flatten(0, 1)
                return torch.where(valid, packed, 0)''', '''            packed_key, packed_value = (
                torch.where(valid, plane, 0) for plane in interleaved_pages(kc, vc, indices))''')
        source = once(source, 'key=gather_pages(kc), value=gather_pages(vc)', 'key=packed_key, value=packed_value')
    source = once(source, '''    def pack(cache):
        return torch.where(valid, cache.index_select(0, ids).flatten(0, 1), 0)''',
                  '''    packed_key, packed_value = (
        torch.where(valid, plane, 0) for plane in interleaved_pages(key, value, ids))''')
    source = once(source, 'key=pack(key), value=pack(value)', 'key=packed_key, value=packed_value')
    compile(source, 'decode_tiled_attention.py', 'exec')
    return source


def edit_single_query(source):
    """Pack only live KV rows. Each decode query attends its entire live prefix."""
    lo = source.index('    ends = list(accumulate(n * 128 for n in counts))')
    hi = source.index('    return ids, valid, query_rows, ends', lo) + len('    return ids, valid, query_rows, ends')
    source = once(source, source[lo:hi], '''    positions = torch.cat([torch.arange(length, device=table.device) for length in lengths])
    pages = torch.cat([ids[start:start + count].repeat_interleave(128)[:length]
                       for start, count, length in zip([0, *accumulate(counts[:-1])], counts, lengths, strict=True)])
    return pages, positions.remainder(128), list(accumulate(lengths))''')
    lo = source.index('            indices, valid, query_rows, ends = cached[identity]')
    hi = source.index('            return output', lo)
    source = once(source, source[lo:hi], '''            pages, offsets, ends = cached[identity]
            result = torch_npu.npu_fused_infer_attention_score(
                query=query.contiguous(), key=kc[pages, offsets], value=vc[pages, offsets],
                input_layout='TND', sparse_mode=0, actual_seq_lengths=list(range(1, n + 1)),
                actual_seq_lengths_kv=ends, num_heads=8, num_key_value_heads=1, scale=impl.scale)[0]
            output.copy_(result.reshape_as(query))
''')
    compile(source, 'decode_tiled_attention.py', 'exec')
    return source


def build(config):
    mode = config.get('kv_pack_mode', 'triton_pair')
    require(mode in ('interleaved_pages', 'interleaved_prefill', 'zero_tail_rows', 'single_query', 'triton_pair'), 'unknown KV packing mode')
    if mode != 'triton_pair':
        from project_packed_prefill_qkv import build as packed
        packed(config)
    else:
        build_tile(config)
    target = Path(config['projection']['target'])
    parent = snapshot(Path(config['pack_parent']))
    require(snapshot(target) == parent and digest(parent) == doc(config['pack_parent_proof'])['runtime'],
            'KV packing parent differs from the precision-reviewed candidate')
    relative = 'parents/jev-system-rl/implementation/decode_tiled_attention.py'
    source = target/'bundle'/relative
    rows_path = target/'bundle/ROWS.json'
    rows = doc(pin(rows_path))
    require(rows['decode_attention_sha256'] == file_digest(source), 'incorrect parent attention code')
    original = source.read_text()
    if mode in ('interleaved_pages', 'interleaved_prefill'):
        result = edit_interleaved(original, config['interleaved_reference'], include_decode=mode=='interleaved_pages')
    elif mode == 'single_query':
        result = edit_single_query(original)
    elif mode == 'zero_tail_rows':
        result = edit_tail_clear(original)
    else:
        result = edit(original, config['pack_block_elements'])
    source.write_text(result)
    rows['decode_attention_sha256'] = file_digest(source)
    write_json(rows_path, rows)
    for lane in config['projection']['lanes']:
        server = target/'bundle'/('SERVER-'+lane+'.json')
        data = doc(pin(server))
        data['local_activation']['rows_ref'].update(bytes=rows_path.stat().st_size,sha256=file_digest(rows_path))
        write_json(server, data)
    delta = changes(parent,snapshot(target))
    require(not delta['engine'] and set(delta['files']) == {relative} and
            set(delta['rows']) == {'decode_attention_sha256'}, 'unexpected runtime changes')
    save(target/'KV-PACK.json',dict(parent=digest(parent),changes=delta,block=config.get('pack_block_elements'),mode=config.get('kv_pack_mode','triton_pair')))


if __name__ == '__main__':
    build(doc(pin(sys.argv[1])))
