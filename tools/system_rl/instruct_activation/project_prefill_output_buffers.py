"""Keep graph outputs live only until they are copied into their final contiguous slice."""
from pathlib import Path
import sys
from decode_lineage import snapshot, changes, digest as file_digest
from io_utils import doc, pin, digest, require, save
from project_graph_tile_rows import build as build_tile
from project_prefill_timing import once
from project_screen_lanes import save as write_json


def edit(s):
    s = once(s, '        return tuple(x.clone() for x in graphs[key](*args))',
             '        return graphs[key](*args)  # consumed below before another replay')
    s = once(s, '                routed=[];weights=[];new_residual=[]', '''                # Private destinations, outside the graph pool; all operations use
                # the caller stream. Never hand a graph-owned buffer to a later layer.
                routed = hidden_states.new_empty((n * 8, 2048))
                weights = hidden_states.new_empty((n, 8))
                new_residual = torch.empty_like(hidden_states)''')
    s = once(s, '''                    routed.append(local);weights.append(w);new_residual.append(total)
                rows=reduce(torch.cat(routed,dim=0)).reshape(n,8,2048)
                return (rows*torch.cat(weights,dim=0)[:,:,None]).sum(dim=1),torch.cat(new_residual,dim=0)''',
             '''                    routed[lo*8:hi*8].copy_(local)
                    weights[lo:hi].copy_(w)
                    new_residual[lo:hi].copy_(total)
                rows=reduce(routed).reshape(n,8,2048)
                return (rows*weights[:,:,None]).sum(dim=1),new_residual''')
    compile(s, 'decode_layer_graph.py', 'exec')
    return s


def build(config):
    single = config.get('buffer_mode','copy_to_slices')
    require(single in ('copy_to_slices','single_piece'), 'unknown buffer mode')
    if single=='single_piece':
        from project_packed_prefill_qkv import build as pack
        pack(config)
    else:
        build_tile(config)
    root = Path(config['projection']['target'])
    parent = snapshot(Path(config['buffer_parent']))
    proof = doc(config['buffer_parent_proof'])
    require(proof['passed'] and digest(parent) == proof['runtime'] and snapshot(root) == parent,
            'unexpected training-reviewed parent')
    rel = 'parents/jev-system-rl/implementation/decode_layer_graph.py'
    src, rp = root/'bundle'/rel, root/'bundle/ROWS.json'
    rows = doc(pin(rp))
    require(file_digest(src) == rows['decode_layer_graph']['source_sha256'], 'graph source drift')
    original=src.read_text()
    if single=='single_piece':
        # tile_replay still clones outputs out of its shared graph pool. Keep
        # contiguous operands for the backend and preserve every reduction.
        original=once(original,'                q,k,v=(torch.cat(items,dim=0) for items in zip(*pieces,strict=True))',
                      '''                if len(pieces)==1:
                    q,k,v=(x.contiguous() for x in pieces[0])
                else:
                    q,k,v=(torch.cat(items,dim=0) for items in zip(*pieces,strict=True))''')
        k='                rows=reduce(torch.cat(routed,dim=0)).reshape(n,8,2048)'
        original=once(original,k,'''                if len(routed)==1:
                    rows=reduce(routed[0].contiguous()).reshape(n,8,2048)
                    return (rows*weights[0][:,:,None]).sum(dim=1),new_residual[0]
'''+k)
        compile(original,rel,'exec')
    else:
        original=edit(original)
    src.write_text(original)
    rows['decode_layer_graph']['source_sha256'] = file_digest(src)
    write_json(rp, rows)
    for lane in config['projection']['lanes']:
        path = root/'bundle'/('SERVER-'+lane+'.json')
        spec = doc(pin(path))
        spec['local_activation']['rows_ref'].update(bytes=rp.stat().st_size, sha256=file_digest(rp))
        write_json(path, spec)
    delta = changes(parent, snapshot(root))
    require(not delta['engine'] and set(delta['files']) == {rel} and set(delta['rows']) == {'decode_layer_graph'},
            'unrelated runtime changes')
    save(root/'OUTPUT-BUFFERS.json',dict(parent=digest(parent), changes=delta))


if __name__ == '__main__':
    build(doc(pin(sys.argv[1])))
