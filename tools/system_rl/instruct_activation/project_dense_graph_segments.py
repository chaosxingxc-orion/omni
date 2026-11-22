"""Add pre-attention graph replay to the measured after-attention graph."""
from pathlib import Path
import sys

from decode_lineage import changes, digest, snapshot
from io_utils import digest as digest_state
from project_dense_tile_graph import build as after_graph
from project_prefill_timing import once
from project_screen_lanes import read, save


def build(spec):
    if spec['prefill_segments'] != ['before', 'after']:
        raise ValueError('expected full before/after study')
    after_graph(spec)
    root=Path(spec['projection']['target'])
    parent=Path(spec['after_graph_parent'])
    before=snapshot(parent)
    if snapshot(root)!=before:
        raise ValueError('after-graph runtime changed')
    rel='parents/jev-system-rl/implementation/decode_layer_graph.py'
    source=root/'bundle'/rel
    rp=root/'bundle/ROWS.json'
    rows=read(rp)
    graph=rows['decode_layer_graph']
    if digest(source)!=graph['source_sha256']:
        raise ValueError('graph source mismatch')
    src=once(source.read_text(), '                    q,k,v,total=one(hidden_states[lo:hi],rs[lo:hi],trig[0][lo:hi],trig[1][lo:hi],mask[lo:hi])',
             '''                    args = (hidden_states[lo:hi], rs[lo:hi], trig[0][lo:hi], trig[1][lo:hi], mask[lo:hi])
                    if hi-lo == tile:
                        q,k,v,total = tile_replay(i, 'before', one, args)
                    else:
                        q,k,v,total = one(*args)''')
    compile(src, rel, 'exec')
    source.write_text(src)
    graph.update(source_sha256=digest(source))
    graph['prefill_graph']['segments']=spec['prefill_segments']
    save(rp,rows)
    for lane in spec['projection']['lanes']:
        file=root/'bundle'/('SERVER-'+lane+'.json')
        c=read(file)
        c['local_activation']['rows_ref'].update(sha256=digest(rp),bytes=rp.stat().st_size)
        save(file,c)
    delta=changes(before,snapshot(root))
    if delta['engine'] or set(delta['files'])!={rel} or set(delta['rows'])!={'decode_layer_graph'}:
        raise ValueError('unrelated graph change')
    save(root/'GRAPH-SEGMENTS.json',dict(parent_sha256=digest_state(before),changes=delta))

if __name__=='__main__':
    build(read(Path(sys.argv[1])))
