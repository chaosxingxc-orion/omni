"""Use only local output columns of the frozen Q/K/V base and adapter B on prefill."""
from pathlib import Path
import sys
from project_prefill_tile import build as tile
from project_screen_lanes import read,save
from decode_lineage import snapshot,changes,digest

def edit(source):
    def replace(old,new):
        nonlocal source
        if source.count(old)!=1:raise ValueError('unexpected parent implementation')
        source=source.replace(old,new)
    replace('    pools = {}', '    pools = {}\n    local_qkv = {}')
    replace('        oparts=base[attention.o_proj.base_layer.weight.data_ptr()]', """        if eager:
            if i not in local_qkv:
                packed=[]
                for matrix,a,b in qparts:
                    width=matrix.shape[-1]//4
                    if matrix.shape[-1] != 4*width or b.shape[0]!=4*width:
                        raise ValueError('QKV packing shape mismatch')
                    packed.append((matrix.narrow(-1,rank*width,width).contiguous(),a,
                                   b.narrow(0,rank*width,width).contiguous()))
                local_qkv[i]=packed
            qparts=local_qkv[i]
        oparts=base[attention.o_proj.base_layer.weight.data_ptr()]""")
    replace('def compute_projection(x, mask, pieces, qkv, length):','def compute_projection(x, mask, pieces, qkv, length, local=False):')
    replace('            if qkv:', '            if qkv and not local:')
    replace('compute_projection(src,mask,qparts,True,length)', 'compute_projection(src,mask,qparts,True,length,local=eager)')
    replace('        pools.clear()', '        pools.clear()\n        local_qkv.clear()')
    compile(source,'decode_layer_graph.py','exec')
    return source

def build(spec):
    if spec["projection_columns"] != "local_prefill":raise ValueError("unknown projection mode")
    tile(spec)
    root=Path(spec['projection']['target']); parent=Path(spec['measured_parent'])
    baseline=snapshot(parent)
    if snapshot(root)!=baseline:raise ValueError('runtime differs from measured tile parent')
    rel='parents/jev-system-rl/implementation/decode_layer_graph.py'
    path=root/'bundle'/rel
    rows=read(root/'bundle/ROWS.json')
    if digest(path)!=rows['decode_layer_graph']['source_sha256']:raise ValueError('source mismatch')
    path.write_text(edit(path.read_text()))
    rows['decode_layer_graph'].update(source_sha256=digest(path))
    rp=root/'bundle/ROWS.json';save(rp,rows)
    for name in spec['projection']['lanes']:
        server=root/'bundle'/('SERVER-'+name+'.json')
        record=read(server); record['local_activation']['rows_ref'].update(bytes=rp.stat().st_size,sha256=digest(rp))
        save(server,record)
    delta=changes(baseline,snapshot(root))
    if delta['engine'] or set(delta['files'])!={rel} or set(delta['rows'])!={'decode_layer_graph'}:
        raise ValueError('unrelated runtime changed')
    save(root/'QKV-COLUMNS.json',dict(parent_sha256=digest(parent/'CANDIDATE.json'),changes=delta))

if __name__=='__main__':build(read(Path(sys.argv[1])))
