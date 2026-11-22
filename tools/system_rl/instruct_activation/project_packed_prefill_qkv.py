"""Compute local Q/K/V base columns and all three LoRA A projections in one GMM."""
from pathlib import Path
import sys

from decode_lineage import changes, digest as file_digest, snapshot
from io_utils import digest, doc, pin, require, save
from project_graph_tile_rows import build as build_tiles
from project_prefill_timing import once
from project_screen_lanes import save as write_json


def edit(s):
    s = once(s, '    pools = {}', '    pools = {}\n    packed_prefill_weights = {}')
    s = once(s, '        pools.clear()', '        pools.clear()\n        packed_prefill_weights.clear()')
    anchor = '        oparts=base[attention.o_proj.base_layer.weight.data_ptr()]'
    s = once(s, anchor, '''        if eager and i not in packed_prefill_weights:
            # Cache only weights of the currently verified adapter. Its load hook
            # clears both graphs and packed weights before swapping policies.
            cols, aa, bb, widths, ranks = [], [], [], [], []
            for base_w, a, b in qparts:
                width, remainder = divmod(base_w.shape[-1], 4)
                if remainder or b.shape != (4*width, a.shape[0]) or a.shape[1] != base_w.shape[1]:
                    raise ValueError('incompatible QKV LoRA operands')
                cols.append(base_w.narrow(-1, rank*width, width))
                aa.append(a.T.unsqueeze(0))
                bb.append(b.narrow(0, rank*width, width))
                widths.append(width)
                ranks.append(a.shape[0])
            packed_prefill_weights[i] = (torch.cat(cols+aa, dim=-1), tuple(bb), tuple(widths+ranks))
        packed_prefill = packed_prefill_weights[i] if eager else None
''' + anchor)
    anchor = '            q,k,v=compute_projection(src,mask,qparts,True,length).split([1024,128,128],dim=-1)'
    s = once(s, anchor, '''            if packed_prefill is None:
                q,k,v=compute_projection(src,mask,qparts,True,length).split([1024,128,128],dim=-1)
            else:
                packed_w, local_b, sizes = packed_prefill
                projected, = torch_npu.npu_grouped_matmul(x=[src.contiguous()], weight=[packed_w],
                    split_item=2, group_list_type=1, group_type=0, group_list=length)
                parts = projected.split(sizes, dim=-1)
                q,k,v = [torch.where(mask, base_part + torch.nn.functional.linear(u.contiguous(), b)*scale,
                                     base_part)
                         for base_part,u,b in zip(parts[:3],parts[3:],local_b,strict=True)]''')
    compile(s, 'decode_layer_graph.py', 'exec')
    return s


def build(config):
    build_tiles(config)
    root = Path(config['projection']['target'])
    parent = snapshot(Path(config['qkv_parent']))
    proof = doc(config['qkv_parent_proof'])
    require(proof['passed'] and digest(parent) == proof['runtime'], 'training-reviewed parent required')
    before_packing = snapshot(root)
    if before_packing != parent:
        from copy import deepcopy
        expected = deepcopy(parent)
        expected['rows']['decode_layer_graph']['prefill_tile_rows'] = config['graph_tile_rows']
        require(before_packing == expected and config['graph_tile_rows'] == config['combine_with_tile_rows'],
                'combined candidate may change only the configured tile rows before packing')
    rel = 'parents/jev-system-rl/implementation/decode_layer_graph.py'
    src, rp = root/'bundle'/rel, root/'bundle/ROWS.json'
    rows = doc(pin(rp))
    require(file_digest(src) == rows['decode_layer_graph']['source_sha256'], 'graph source drift')
    packed_source=edit(src.read_text())
    src.write_text(packed_source)
    rows['decode_layer_graph']['source_sha256'] = file_digest(src)
    write_json(rp,rows)
    for lane in config['projection']['lanes']:
        path = root/'bundle'/('SERVER-'+lane+'.json')
        spec = doc(pin(path))
        spec['local_activation']['rows_ref'].update(bytes=rp.stat().st_size,sha256=file_digest(rp))
        write_json(path,spec)
    delta = changes(parent,snapshot(root))
    require(not delta['engine'] and set(delta['files']) == {rel} and set(delta['rows']) == {'decode_layer_graph'},
            'unrelated runtime change')
    save(root/'PACKED-PREFILL-QKV.json',dict(parent=digest(parent),changes=delta))
    if 'extend_to_decode_parent' in config:
        # Isolate the small-batch decode branch. Packing has already been
        # measured for prefill; the decode shapes need their own parity check.
        immediate=snapshot(Path(config['extend_to_decode_parent']))
        require(snapshot(root)==immediate,'decode trial must start at the exact packed runtime')
        packed_source=once(packed_source,'if eager and i not in packed_prefill_weights:',
                           'if i not in packed_prefill_weights:')
        packed_source=once(packed_source,'packed_prefill_weights[i] if eager else None','packed_prefill_weights[i]')
        compile(packed_source,rel,'exec')
        src.write_text(packed_source)
        rows['decode_layer_graph']['source_sha256']=file_digest(src)
        write_json(rp,rows)
        for lane in config['projection']['lanes']:
            path=root/'bundle'/('SERVER-'+lane+'.json')
            spec=doc(pin(path))
            spec['local_activation']['rows_ref'].update(bytes=rp.stat().st_size,sha256=file_digest(rp))
            write_json(path,spec)
        child=changes(immediate,snapshot(root))
        require(not child['engine'] and set(child['files'])=={rel} and set(child['rows'])=={'decode_layer_graph'},
                'unrelated decode trial change')
        save(root/'PACKED-DECODE-QKV.json',dict(immediate_parent=digest(immediate),changes=child))


if __name__ == '__main__':
    build(doc(pin(sys.argv[1])))
