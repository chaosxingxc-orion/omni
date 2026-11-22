"""Partition full routed tokens before summing, then distribute the smaller result."""
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
    replace('    from vllm.distributed import tensor_model_parallel_all_reduce as reduce',
            '    from vllm.distributed import tensor_model_parallel_reduce_scatter as scatter_rows\n'
            '    from vllm.distributed import tensor_model_parallel_all_reduce as reduce')
    replace('                rows=reduce(torch.cat(routed,dim=0)).reshape(n,8,2048)',"""                if n % 4 == 0:
                    # Row partition keeps the parent's eight products and summation
                    # order together. Only one rank owns each expert contribution.
                    rows=scatter_rows(torch.cat(routed,dim=0),dim=0).reshape(n//4,8,2048)
                    weights_local=torch.cat(weights,dim=0).narrow(0, rank*(n//4), n//4)
                    out=(rows*weights_local[:,:,None]).sum(dim=1)
                    return gather(out.contiguous(),dim=0),torch.cat(new_residual,dim=0)
                rows=reduce(torch.cat(routed,dim=0)).reshape(n,8,2048)""")
    compile(source,'decode_layer_graph.py','exec')
    return source

def build(spec):
    if spec["collective"] != "scatter_complete_tokens":raise ValueError("unknown collective")
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
    save(root/'TOKEN-SCATTER.json',dict(parent_sha256=digest(parent/'CANDIDATE.json'),changes=delta))

if __name__=='__main__':build(read(Path(sys.argv[1])))
