"""Change the length of a captured prefill tile on a verified graph runtime."""
import sys
from pathlib import Path
from decode_lineage import changes, digest, snapshot
from io_utils import digest as digest_state
from project_dense_tile_graph import build as after_graph
from project_dense_graph_segments import build as both_graphs
from project_prefill_timing import once
from project_screen_lanes import read, save


def build(spec):
    mode = spec['tile_parent_segments']
    rows_per_tile = spec['graph_tile_rows']
    if type(rows_per_tile) is not int or rows_per_tile not in (1024, 2048):
        raise ValueError('expected a supported larger prefill tile')
    if mode == ['after']:
        after_graph(spec)
    elif mode == ['before', 'after']:
        both_graphs(spec)
    else:
        raise ValueError('unrecognized graph segments')
    root = Path(spec['projection']['target'])
    parent = snapshot(Path(spec['tile_parent_packet']))
    if snapshot(root) != parent:
        raise ValueError('parent implementation differs')
    rel = 'parents/jev-system-rl/implementation/decode_layer_graph.py'
    code = root / 'bundle' / rel
    rp = root / 'bundle/ROWS.json'
    rows = read(rp)
    graph = rows['decode_layer_graph']
    if graph['source_sha256'] != digest(code) or graph['prefill_tile_rows'] != 512:
        raise ValueError('expected verified 512-row graph')
    source = once(code.read_text(), '16<=tile<=512', '16<=tile<=2048')
    source = once(source, 'in [16,512]', 'in [16,2048]')
    compile(source, rel, 'exec')
    code.write_text(source)
    graph.update(source_sha256=digest(code), prefill_tile_rows=rows_per_tile)
    save(rp, rows)
    for lane in spec['projection']['lanes']:
        server = root / 'bundle' / ('SERVER-' + lane + '.json')
        cfg = read(server)
        cfg['local_activation']['rows_ref'].update(sha256=digest(rp), bytes=rp.stat().st_size)
        save(server, cfg)
    delta = changes(parent, snapshot(root))
    if delta['engine'] or set(delta['files']) != {rel} or set(delta['rows']) != {'decode_layer_graph'}:
        raise ValueError('unrelated runtime difference')
    save(root / 'GRAPH-TILE-ROWS.json', dict(parent_sha256=digest_state(parent), changes=delta))


if __name__ == '__main__':
    build(read(Path(sys.argv[1])))
