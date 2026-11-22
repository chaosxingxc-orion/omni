"""Run only owner expert GEMMs inside the measured full-tile post-attention graph."""
from pathlib import Path
import sys

from decode_lineage import changes, digest as file_digest, snapshot
from io_utils import digest, doc, pin, require, save
from project_graph_tile_rows import build as build_tiles
from project_prefill_timing import once
from project_screen_lanes import save as write_json


def edit(src):
    # Same dropless gather semantics as the source-matched eager routing trial.
    # Use its real-range counts, not the baseline's padded 33rd expert. The
    # native operator supports graph capture and fixed maximum output shape.
    marker = '            flat=ids.flatten()'
    src = once(src, marker, '''            if eager and n == config['prefill_tile_rows']:
                low = 32 * rank
                packed, inverse, counts, _ = torch_npu.npu_moe_init_routing_v2(
                    h, ids.int(), active_num=n*8, expert_num=128, expert_tokens_num_type=1,
                    expert_tokens_num_flag=True, quant_mode=-1, active_expert_range=[low,low+32],
                    row_idx_type=0)
                first,second=grouped(packed,gu[:32],counts).chunk(2,dim=-1)
                y=grouped(torch.nn.functional.silu(first)*second,down[:32],counts)
                inverse=inverse.reshape(-1).long()
                rows=y.index_select(0,inverse.clamp(min=0))
                return torch.where((inverse>=0)[:,None],rows,0),w,total
''' + marker)
    compile(src, 'decode_layer_graph.py', 'exec')
    return src


def prepare_decode_widths(src):
    anchor = '            if (i,n) not in graphs:'
    src = once(src, anchor, '''            if ('prepared_widths', i) not in graphs:
                for width in config['batch_sizes']:
                    if (i, width) in graphs:
                        continue
                    # The capture only sees positionwise operands. It does not access
                    # the KV cache, advance a request or emit a token.
                    sample = [x[:1].expand(width, *x.shape[1:]).contiguous()
                              for x in (hidden_states, rs, *trig, mask)]
                    graphs[i, width] = build(i, width, *sample)
                graphs['prepared_widths', i] = True
''' + anchor)
    compile(src, 'decode_layer_graph.py', 'exec')
    return src


def build(config):
    if 'packed_route_parent' in config:
        from project_packed_prefill_qkv import build as pack_qkv
        pack_qkv(config)
    else:
        build_tiles(config)
    root = Path(config['projection']['target'])
    parent = snapshot(Path(config['route_parent']))
    proof = doc(config['route_parent_proof'])
    require(proof['passed'] and digest(parent) == proof['runtime'], 'training-reviewed parent required')
    previous = snapshot(root)
    if 'packed_route_parent' in config:
        candidate = Path(config['packed_route_parent'])
        packed = doc(pin(candidate/'PACKED-PREFILL-QKV.json'))
        require(packed['parent'] == digest(parent) and packed['changes'] == changes(parent, previous)
                and previous == snapshot(candidate), 'different packed QKV parent')
    else:
        require(previous == parent, 'unexpected parent implementation')
    rel = 'parents/jev-system-rl/implementation/decode_layer_graph.py'
    src, rp = root/'bundle'/rel, root/'bundle/ROWS.json'
    rows = doc(pin(rp))
    require(file_digest(src) == rows['decode_layer_graph']['source_sha256'], 'graph source drift')
    source = edit(src.read_text())
    if config.get('prepare_decode_widths'):
        require(config['prepare_decode_widths'] is True and
                rows['decode_layer_graph'].get('pool_mode') == 'per_batch',
                'decode preparation requires isolated graph pools for each batch width')
        source = prepare_decode_widths(source)
    src.write_text(source)
    rows['decode_layer_graph']['source_sha256'] = file_digest(src)
    write_json(rp,rows)
    for lane in config['projection']['lanes']:
        path = root/'bundle'/('SERVER-'+lane+'.json')
        spec = doc(pin(path))
        spec['local_activation']['rows_ref'].update(bytes=rp.stat().st_size,sha256=file_digest(rp))
        write_json(path,spec)
    if 'prefill_page_component' in config:
        from shutil import copyfile
        ref = config['prefill_page_component']
        component = Path(ref['path']).parent
        proof = doc(ref)
        code = 'parents/jev-system-rl/implementation/decode_tiled_attention.py'
        key = 'decode_attention_sha256'
        component_delta = changes(previous, snapshot(component))
        require(proof['mode'] == 'interleaved_prefill' and proof['parent'] == digest(previous)
                and proof['changes'] == component_delta and not component_delta['engine']
                and set(component_delta['files']) == {code} and set(component_delta['rows']) == {key},
                'unexpected prefill page component')
        graph_delta = changes(previous, snapshot(root))
        require(not graph_delta['engine'] and set(graph_delta['files']) == {rel}
                and set(graph_delta['rows']) == {'decode_layer_graph'}, 'unexpected routing component')
        copyfile(component/'bundle'/code, root/'bundle'/code)
        rows[key] = file_digest(root/'bundle'/code)
        write_json(rp, rows)
        for lane in config['projection']['lanes']:
            path = root/'bundle'/('SERVER-'+lane+'.json')
            spec = doc(pin(path))
            spec['local_activation']['rows_ref'].update(bytes=rp.stat().st_size,sha256=file_digest(rp))
            write_json(path, spec)
        expected = {kind: graph_delta[kind] | component_delta[kind] for kind in graph_delta}
        require(changes(previous,snapshot(root)) == expected, 'composition differs from its inputs')
        save(root/'COMPONENTS.json', dict(parent=digest(previous), components=[graph_delta,component_delta],
                                          changes=expected))
        return
    delta = changes(parent,snapshot(root))
    require(not delta['engine'] and set(delta['files']) == {rel} and set(delta['rows']) == {'decode_layer_graph'},
            'unrelated runtime change')
    save(root/'GRAPH-OWNED-ROUTES.json',dict(parent=digest(parent),immediate_parent=digest(previous),changes=delta))


if __name__ == '__main__':
    build(doc(pin(sys.argv[1])))
