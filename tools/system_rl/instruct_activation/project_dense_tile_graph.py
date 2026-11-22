"""Capture independent full prefill tiles on one reusable pool per rank."""
import sys
from pathlib import Path

from decode_lineage import changes, digest, snapshot
from io_utils import digest as digest_state
from project_prefill_dense_pages import build as dense_parent
from project_prefill_timing import once
from project_screen_lanes import read, save


def patch(source):
    source = once(source, "set(config)-{'pool_mode'", "set(config)-{'prefill_graph','pool_mode'")
    anchor = '    def build(i,n,hidden,residual,cos,sin,mask,*,eager=False):'
    source = once(source, anchor, '''    def tile_replay(i, side, callback, args):
        key = ('prefill_graph', i, side)
        if key not in graphs:
            free, _ = torch.npu.mem_get_info()
            if free < config['reserve_bytes'] + config['prefill_graph']['capture_budget_bytes']:
                raise ValueError('insufficient free HBM for prefill graph capture')
            if 'prefill_graph' not in pools:
                pools['prefill_graph'] = torch.npu.graph_pool_handle()
            graphs[key] = capture(callback, args, pools['prefill_graph'])
        # Inputs live outside the graph pool. No captured output crosses into
        # the next replay: clone even when an output aliases a static input.
        return tuple(x.clone() for x in graphs[key](*args))

''' + anchor)
    line = '                    local,w,total=afters[j](full[lo:hi],totals[j],mask[lo:hi])'
    source = once(source, line, '''                    args = (full[lo:hi], totals[j], mask[lo:hi])
                    if hi - lo == tile:
                        local, w, total = tile_replay(i, 'after', afters[j], args)
                    else:
                        local, w, total = afters[j](*args)''')
    compile(source, 'decode_layer_graph.py', 'exec')
    return source


def build(spec):
    mode = spec['prefill_graph']
    if (set(mode) != {'capture_budget_bytes','expandable_segments'}
            or type(mode['capture_budget_bytes']) != int or mode['capture_budget_bytes'] <= 0
            or type(mode['expandable_segments']) != bool):
        raise ValueError('expected capture budget and allocator setting')
    dense_parent(spec)
    root = Path(spec['projection']['target'])
    before = snapshot(Path(spec['baseline_packet']))
    if digest_state(before) != read(Path(spec['performance_lock']))['runtime_sha256'] or snapshot(root) != before:
        raise ValueError('measured dense prefill baseline changed')
    rel = 'parents/jev-system-rl/implementation/decode_layer_graph.py'
    path = root / 'bundle' / rel
    rp = root / 'bundle/ROWS.json'
    rows = read(rp)
    if digest(path) != rows['decode_layer_graph']['source_sha256']:
        raise ValueError('graph source mismatch')
    path.write_text(patch(path.read_text()))
    rows['decode_layer_graph'].update(source_sha256=digest(path), prefill_graph=mode)
    save(rp, rows)
    for name in spec['projection']['lanes']:
        server = root / 'bundle' / ('SERVER-' + name + '.json')
        c = read(server)
        c['local_activation']['rows_ref'].update(sha256=digest(rp), bytes=rp.stat().st_size)
        save(server,c)
        code = root / ('lane-' + name)
        launcher = code / 'launch.py'
        cfg_path = code / 'config.json'
        cfg = read(cfg_path)
        cfg['expandable_segments'] = mode['expandable_segments']
        save(cfg_path, cfg)
        anchor = "    for k, v in env.items():"
        launcher.write_text(once(launcher.read_text(), anchor,
                            "    env['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:' + str(C['expandable_segments'])\n" + anchor))
        pins = read(code / 'PINS.json')
        for f in (launcher, cfg_path):
            pins[f.name] = digest(f)
        save(code / 'PINS.json', pins)
    delta = changes(before,snapshot(root))
    if set(delta['engine']) != {'launch.py'} or set(delta['files']) != {rel} or set(delta['rows']) != {'decode_layer_graph'}:
        raise ValueError('unrelated runtime differences')
    save(root / 'PREFILL-GRAPH.json', dict(baseline_sha256=digest_state(before), settings=mode, changes=delta))


if __name__ == '__main__':
    build(read(Path(sys.argv[1])))
