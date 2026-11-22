"""Evaluate skipping foreign expert routes during eager prefill tiles only."""
import sys
from pathlib import Path
from project_prefill_tile import build as tile
from project_screen_lanes import read, save
from decode_lineage import changes, digest, snapshot


def edit(source):
    anchor = '            first,second=grouped(h[order//8],gu,counts).chunk(2,dim=-1)'
    if source.count(anchor) != 1:
        raise ValueError('unsupported routing source')
    # Keep relative row order inside each real expert identical to the parent.
    # Dynamic shape selection can synchronize: measure this in wall clock.
    added = """            if eager:
                routes = order[own[order]]
                rows = h.new_zeros((flat.numel(), h.shape[-1]))
                if routes.numel():
                    first,second=grouped(h[routes//8],gu[:-1],counts[:-1]).chunk(2,dim=-1)
                    y=grouped(torch.nn.functional.silu(first)*second,down[:-1],counts[:-1])
                    rows.index_copy_(0,routes,y)
                return rows,w,total
"""
    result = source.replace(anchor, added + anchor)
    compile(result, 'decode_layer_graph.py', 'exec')
    return result


def build(spec):
    if spec['routing'] != 'owned_eager_rows':
        raise ValueError('unknown routing mode')
    tile(spec)
    root = Path(spec['projection']['target'])
    parent = Path(spec['measured_parent'])
    before = snapshot(parent)
    if snapshot(root) != before:
        raise ValueError('measured parent differs')
    rel = 'parents/jev-system-rl/implementation/decode_layer_graph.py'
    source = root/'bundle'/rel
    rp = root/'bundle/ROWS.json'
    rows = read(rp)
    if digest(source) != rows['decode_layer_graph']['source_sha256']:
        raise ValueError('unrecognized graph source')
    source.write_text(edit(source.read_text()))
    rows['decode_layer_graph']['source_sha256'] = digest(source)
    save(rp, rows)
    for name in spec['projection']['lanes']:
        server = root/'bundle'/('SERVER-'+name+'.json')
        cfg = read(server)
        cfg['local_activation']['rows_ref'].update(bytes=rp.stat().st_size, sha256=digest(rp))
        save(server,cfg)
    diff = changes(before,snapshot(root))
    if diff['engine'] or set(diff['files']) != {rel} or set(diff['rows']) != {'decode_layer_graph'}:
        raise ValueError('unexpected runtime change')
    save(root/'ROUTING.json', dict(routing=spec['routing'],parent_sha256=digest(parent/'CANDIDATE.json'),changes=diff))


if __name__ == '__main__':
    build(read(Path(sys.argv[1])))
