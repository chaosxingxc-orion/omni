"""Use native BF16 dropless routing for local prefill experts and retain route slots."""
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
    replace('    graphs = {}','    graphs = {}\n    route_layout_checked = False')
    replace('        def after(full_attn,res,mask):','        def after(full_attn,res,mask):\n            nonlocal route_layout_checked')
    replace('            flat=ids.flatten()',"""            if eager:
                lo=32*rank
                packed,inverse,counts,_=torch_npu.npu_moe_init_routing_v2(
                    h,ids.int(),active_num=n*8,expert_num=128,expert_tokens_num_type=1,
                    expert_tokens_num_flag=True,quant_mode=-1,active_expert_range=[lo,lo+32],row_idx_type=0)
                inverse=inverse.reshape(-1).long()
                local=(ids>=lo)&(ids<lo+32)
                lookup=inverse.clamp(min=0)
                if not route_layout_checked:
                    # Check both request/token order and the actual packed expert groups
                    # before using this image's fused permutation for real responses.
                    route_blocks=torch.arange(32,device=h.device).repeat_interleave(counts)
                    valid=local.flatten()
                    if not torch.equal(inverse>=0,valid) or not torch.equal(
                            packed.index_select(0,lookup[valid]),
                            h.index_select(0,torch.arange(n,device=h.device).repeat_interleave(8)[valid])) or not torch.equal(
                            route_blocks.index_select(0,lookup[valid]),ids.flatten()[valid]-lo):
                        raise ValueError('native routing input, group or output slot mismatch')
                    route_layout_checked=True
                g,u=grouped(packed,gu[:-1],counts).chunk(2,dim=-1)
                y=grouped(torch.nn.functional.silu(g)*u,down[:-1],counts)
                rows=y.index_select(0,lookup)
                return torch.where(local.reshape(-1,1),rows,0),w,total
            flat=ids.flatten()""")
    compile(source,'decode_layer_graph.py','exec')
    return source

def build(spec):
    if spec["routing"] != "native_local_gather":raise ValueError("unknown routing")
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
    save(root/'NATIVE-ROUTING.json',dict(parent_sha256=digest(parent/'CANDIDATE.json'),changes=delta))

if __name__=='__main__':build(read(Path(sys.argv[1])))
