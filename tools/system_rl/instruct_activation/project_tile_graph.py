"""Capture full-size post-attention tiles and copy outputs before replay."""
import sys
from pathlib import Path
from project_prefill_tile import build as tile
from project_screen_lanes import read, save
from decode_lineage import changes, digest, snapshot


def edit(source):
    anchor = '                    local,w,total=afters[j](full[lo:hi],totals[j],mask[lo:hi])'
    if source.count(anchor)!=1:raise ValueError('unexpected tiled forward')
    replacement="""                    args=(full[lo:hi],totals[j],mask[lo:hi])
                    if hi-lo!=tile:
                        local,w,total=afters[j](*args)
                    else:
                        key=('full_tile_after',i,tile)
                        if key not in graphs:
                            # Tile counts differ across forwards. Keep a private pool per graph;
                            # graph-owned outputs are copied before replaying it for another tile.
                            free,_=torch.npu.mem_get_info()
                            if free < config['reserve_bytes'] + config['full_tile_capture_budget']:
                                raise ValueError('insufficient free HBM for tile capture')
                            graphs[key]=capture(afters[j],args,None)
                        local,w,total=(part.clone() for part in graphs[key](*args))
"""
    source=source.replace(anchor,replacement.rstrip())
    anchor="set(config)-{'pool_mode'"
    if source.count(anchor)!=1:raise ValueError('unexpected graph configuration')
    source=source.replace(anchor,"set(config)-{'full_tile_capture_budget','pool_mode'")
    compile(source,'decode_layer_graph.py','exec')
    return source


def build(spec):
    budget=spec['capture_budget_bytes']
    if type(budget)!=int or budget<256*1024**2:raise ValueError('explicit capture budget required')
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
    rows['decode_layer_graph'].update(source_sha256=digest(source),full_tile_capture_budget=budget)
    save(rp, rows)
    for name in spec['projection']['lanes']:
        server = root/'bundle'/('SERVER-'+name+'.json')
        cfg = read(server)
        cfg['local_activation']['rows_ref'].update(bytes=rp.stat().st_size, sha256=digest(rp))
        save(server,cfg)
    diff = changes(before,snapshot(root))
    if diff['engine'] or set(diff['files']) != {rel} or set(diff['rows']) != {'decode_layer_graph'}:
        raise ValueError('unexpected runtime change')
    save(root/'TILE-GRAPH.json', dict(capture_budget_bytes=budget,parent_sha256=digest(parent/'CANDIDATE.json'),changes=diff))


if __name__ == '__main__':
    build(read(Path(sys.argv[1])))
