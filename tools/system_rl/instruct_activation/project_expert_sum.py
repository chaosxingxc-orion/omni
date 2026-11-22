"""Compare owner-local MoE sums against the exact same tiled parent packet."""
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
    replace("set(config)-{'pool_mode'", "set(config)-{'expert_sum','pool_mode'")
    replace("    shapes = config.get('batch_sizes')","""    if config.get('expert_sum','rows') not in ('rows','owner_fp32'):
        raise ValueError('unknown expert sum')
    shapes = config.get('batch_sizes')""")
    replace('            return torch.where(own[:,None],rows,0),w,total','''            rows=torch.where(own[:,None],rows,0)
            if config.get('expert_sum')=='owner_fp32':
                weighted=rows.reshape(n,8,2048)*w[:,:,None]
                return weighted.sum(dim=1,dtype=torch.float32),w,total
            return rows,w,total''')
    replace('                rows=reduce(torch.cat(routed,dim=0)).reshape(n,8,2048)','''                if config.get('expert_sum')=='owner_fp32':
                    return reduce(torch.cat(routed,dim=0)).bfloat16(),torch.cat(new_residual,dim=0)
                rows=reduce(torch.cat(routed,dim=0)).reshape(n,8,2048)''')
    replace('            rows=reduce(local)','''            if config.get('expert_sum')=='owner_fp32':
                return reduce(local).bfloat16(),total
            rows=reduce(local)''')
    compile(source,'decode_layer_graph.py','exec')
    return source

def build(spec):
    tile(spec)
    root=Path(spec['projection']['target']); parent=Path(spec['measured_parent'])
    baseline=snapshot(parent)
    if snapshot(root)!=baseline:raise ValueError('runtime differs from measured tile parent')
    rel='parents/jev-system-rl/implementation/decode_layer_graph.py'
    path=root/'bundle'/rel
    rows=read(root/'bundle/ROWS.json')
    if digest(path)!=rows['decode_layer_graph']['source_sha256']:raise ValueError('source mismatch')
    path.write_text(edit(path.read_text()))
    rows['decode_layer_graph'].update(source_sha256=digest(path),expert_sum='owner_fp32')
    rp=root/'bundle/ROWS.json';save(rp,rows)
    for name in spec['projection']['lanes']:
        server=root/'bundle'/('SERVER-'+name+'.json')
        record=read(server); record['local_activation']['rows_ref'].update(bytes=rp.stat().st_size,sha256=digest(rp))
        save(server,record)
    delta=changes(baseline,snapshot(root))
    if delta['engine'] or set(delta['files'])!={rel} or set(delta['rows'])!={'decode_layer_graph'}:
        raise ValueError('unrelated runtime changed')
    save(root/'EXPERT-SUM.json',dict(parent_sha256=digest(parent/'CANDIDATE.json'),changes=delta))

if __name__=='__main__':build(read(Path(sys.argv[1])))
