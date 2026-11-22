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


def build(config):
    build_tiles(config)
    root = Path(config['projection']['target'])
    parent = snapshot(Path(config['route_parent']))
    proof = doc(config['route_parent_proof'])
    require(proof['passed'] and digest(parent) == proof['runtime'] and snapshot(root) == parent,
            'training-reviewed parent required')
    rel = 'parents/jev-system-rl/implementation/decode_layer_graph.py'
    src, rp = root/'bundle'/rel, root/'bundle/ROWS.json'
    rows = doc(pin(rp))
    require(file_digest(src) == rows['decode_layer_graph']['source_sha256'], 'graph source drift')
    src.write_text(edit(src.read_text()))
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
    save(root/'GRAPH-OWNED-ROUTES.json',dict(parent=digest(parent),changes=delta))


if __name__ == '__main__':
    build(doc(pin(sys.argv[1])))
