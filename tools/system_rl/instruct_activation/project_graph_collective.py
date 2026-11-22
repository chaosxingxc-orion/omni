"""Replay the entire post-attention update for a frequent scheduled shape."""
from pathlib import Path
import sys

from decode_lineage import changes, digest, snapshot
from io_utils import digest as digest_state
from project_dense_tile_graph import build as build_graph
from project_prefill_timing import once
from project_screen_lanes import read, save


def edit(source):
    anchor = '                full=gather(layer.self_attn.attn(q,k,v).contiguous(),dim=1)'
    code = '''                attended = layer.self_attn.attn(q,k,v).contiguous()
                if n == config['prefill_graph']['collective_rows']:
                    if n % tile or len(afters) != n // tile:
                        raise ValueError('full step must consist entirely of equal-size tiles')
                    apply_tile = afters[0]
                    if any(fn is not apply_tile for fn in afters):
                        raise ValueError('full tiles use different projection or routing closures')

                    def update(local_attention, all_totals, adapter_mask):
                        full_attention = gather(local_attention, dim=1)
                        routed_tiles = []
                        weight_tiles = []
                        residual_tiles = []
                        for start in range(0, n, tile):
                            stop = start + tile
                            expert_rows, weights, next_residual = apply_tile(
                                full_attention[start:stop], all_totals[start:stop], adapter_mask[start:stop])
                            routed_tiles.append(expert_rows)
                            weight_tiles.append(weights)
                            residual_tiles.append(next_residual)
                        rows = reduce(torch.cat(routed_tiles, dim=0)).reshape(n,8,2048)
                        output = (rows * torch.cat(weight_tiles,dim=0)[:,:,None]).sum(dim=1)
                        return output, torch.cat(residual_tiles,dim=0)

                    key = ('collective_graph', i, n)
                    inputs = attended, torch.cat(totals, dim=0), mask
                    if key not in graphs:
                        free, _ = torch.npu.mem_get_info()
                        capacity = config['reserve_bytes'] + config['prefill_graph']['collective_capture_bytes']
                        ready = torch.tensor([int(free >= capacity)], device=device, dtype=torch.int32)
                        if int(reduce(ready).item()) != 4:
                            raise ValueError('all four ranks require HBM for collective capture')
                        if 'collective_graph' not in pools:
                            pools['collective_graph'] = torch.npu.graph_pool_handle()
                        graphs[key] = capture(update, inputs, pools['collective_graph'])
                    # The next layer may replay from the same scratch pool.
                    return tuple(t.clone() for t in graphs[key](*inputs))
                full = gather(attended, dim=1)'''
    source = once(source, anchor, code)
    compile(source, 'decode_layer_graph.py', 'exec')
    return source


def build(spec):
    setting=spec['collective_graph']
    if set(setting)!={'rows','capture_budget_bytes'}:
        raise ValueError('expected capture shape and budget')
    n, budget = setting['rows'], setting['capture_budget_bytes']
    if type(n)!=int or n<=0 or n>2048 or n%spec['prefill_tile_rows'] or type(budget)!=int or budget<=0:
        raise ValueError('unsupported capture geometry')
    build_graph(spec)
    root=Path(spec['projection']['target'])
    before=snapshot(Path(spec['collective_parent']))
    if digest_state(before)!=read(Path(spec['parent_lock']))['runtime_sha256'] or snapshot(root)!=before:
        raise ValueError('collective parent differs from performance lock')
    rel='parents/jev-system-rl/implementation/decode_layer_graph.py'
    source=root/'bundle'/rel
    rp=root/'bundle/ROWS.json'
    rows=read(rp)
    graph=rows['decode_layer_graph']
    if digest(source)!=graph['source_sha256']:
        raise ValueError('graph source mismatch')
    source.write_text(edit(source.read_text()))
    graph['prefill_graph'].update(collective_rows=n,collective_capture_bytes=budget)
    graph.update(source_sha256=digest(source))
    save(rp,rows)
    for lane in spec['projection']['lanes']:
        p=root/'bundle'/('SERVER-'+lane+'.json')
        cfg=read(p)
        cfg['local_activation']['rows_ref'].update(sha256=digest(rp),bytes=rp.stat().st_size)
        save(p,cfg)
    delta=changes(before,snapshot(root))
    if delta['engine'] or set(delta['files'])!={rel} or set(delta['rows'])!={'decode_layer_graph'}:
        raise ValueError('unrelated runtime changed')
    save(root/'COLLECTIVE-GRAPH.json',dict(parent_sha256=digest_state(before),changes=delta))


if __name__=='__main__':
    build(read(Path(sys.argv[1])))
