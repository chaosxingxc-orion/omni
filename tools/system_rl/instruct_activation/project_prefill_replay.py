"""Build a fixed-action diagnostic sibling of an already measured prefill runtime."""
from pathlib import Path
import shutil
import sys

from decode_lineage import changes, snapshot
from io_utils import digest, doc, pin, require, save
from project_graph_tile_rows import build as build_tiles
from project_screen_lanes import read, save as write_config


def build(c):
    kind = c.get('replay_runtime', 'captured_tiles')
    if kind == 'captured_tiles':
        build_tiles(c)
    elif kind == 'packed_qkv':
        from project_packed_prefill_qkv import build as build_qkv
        build_qkv(c)
    elif kind == 'owner_routes':
        from project_graph_owned_routes import build as build_routes
        build_routes(c)
    elif kind == 'packed_schedule':
        from project_schedule_budget import build as build_schedule
        require('packed_schedule_parent' in c, 'expected training-aligned scheduling parent')
        build_schedule(c)
    elif kind == 'decode_collective':
        from project_graph_collective import build as build_collective
        require(c['collective_scope'] == 'pure_decode', 'unexpected collective scope')
        build_collective(c)
    elif kind == 'single_piece':
        from project_prefill_output_buffers import build as build_buffers
        require(c['buffer_mode']=='single_piece','unexpected buffer implementation')
        build_buffers(c)
    else:
        raise ValueError('unsupported measured replay runtime')
    root = Path(c['projection']['target'])
    original = snapshot(Path(c['measured_candidate']))
    require(original == snapshot(root), 'replay would observe different arithmetic')
    lane, = c['projection']['lanes']
    source = doc(pin(c['replay_source']))
    rel = 'parents/jev-system-rl/implementation/decode_action_processor.py'
    processor = Path(c['verified_replay_packet']) / 'bundle' / rel
    require(pin(processor)['sha256'] == source['changes']['files'][rel]['new'],
            'replay processor differs from audited implementation')
    shutil.copyfile(processor, root / 'bundle' / rel)
    cfg = root / ('lane-' + lane) / 'config.json'
    server = root / 'bundle' / ('SERVER-' + lane + '.json')
    opts = read(cfg)
    opts['serve_args'] += ['--logits-processors', 'decode_action_processor:ReplayActions']
    write_config(cfg, opts)
    write_config(server, dict(read(server), release_kind='decode_replay'))
    pins = cfg.with_name('PINS.json')
    write_config(pins, dict(read(pins), **{'config.json': pin(cfg)['sha256']}))
    delta = changes(original, snapshot(root))
    require(not delta['rows'] and set(delta['files']) == {rel} and
            set(delta['engine']) == {'serve_args', 'release_kind'}, 'unexpected replay runtime delta')
    save(root/'REPLAY-DELTA.json', dict(parent_sha256=digest(original), changes=delta))


if __name__ == '__main__':
    build(read(Path(sys.argv[1])))
