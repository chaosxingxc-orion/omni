"""Time page planning, packing, FIA and output on an isolated dense-prefill copy."""
from pathlib import Path
import sys

from decode_lineage import digest
from project_prefill_dense_pages import build as build_dense
from project_prefill_timing import once
from project_screen_lanes import read, save
from project_stage_trace import bind


def instrument(source, layers):
    source = once(source, '    cached = {}',
                  '    from vllm import prefill_timing as timing\n'
                  f'    timing.configure({layers!r})\n    cached = {{}}')
    source = once(source, 'causal_mask,cached)', 'causal_mask,cached,index)')
    source = once(source, 'output, kv_cache, mask, plan):',
                  'output, kv_cache, mask, plan, index):\n    from vllm import prefill_timing as timing')
    anchors = [('    ql, kl, leads, real, key, value, table = operands', None),
               ('    packed = query.new_zeros', 'metadata_and_plan'),
               ('    def pack(cache):', 'query_pack'),
               ('    result = torch_npu.npu_fused_infer_attention_score(\n        query=packed', 'kv_pack'),
               ('    output[:real].copy_', 'dense_fia'),
               ('    return output\n\ndef dense_plan', 'output_copy')]
    source = once(source, 'query=packed, key=pack(key), value=pack(value)',
                  'query=packed, key=packed_key, value=packed_value')
    for anchor, preceding in anchors:
        code = ''
        if preceding == 'kv_pack':
            code += '    packed_key, packed_value = pack(key), pack(value)\n'
        if preceding:
            code += f'    timing.end(index, {preceding!r}, stamp)\n'
        if preceding != 'output_copy':
            code += '    stamp = timing.start(index)\n'
        source = once(source, anchor, code + anchor)
    compile(source, 'decode_tiled_attention.py', 'exec')
    return source


def build(spec):
    layers = spec['timed_layers']
    if not layers or layers != sorted(set(layers)) or any(type(i) != int or not 0 <= i < 48 for i in layers):
        raise ValueError('invalid observed layers')
    build_dense(spec)
    root = Path(spec['projection']['target'])
    bind(spec, root, Path(spec['timing_parent']))
    lane, = spec['projection']['lanes']
    code = root / ('lane-' + lane)
    source = root / 'bundle/parents/jev-system-rl/implementation/decode_tiled_attention.py'
    rp = root / 'bundle/ROWS.json'
    rows = read(rp)
    if rows['decode_attention_sha256'] != digest(source):
        raise ValueError('attention source drift')
    source.write_text(instrument(source.read_text(), layers))
    rows['decode_attention_sha256'] = digest(source)
    save(rp, rows)
    server = root / 'bundle' / ('SERVER-' + lane + '.json')
    conf = read(server)
    conf['local_activation']['rows_ref'].update(bytes=rp.stat().st_size, sha256=digest(rp))
    save(server, conf)
    helper = code / 'probes/prefill_timing.py'
    helper.write_bytes(Path(__file__).with_name(helper.name).read_bytes())
    runner = code / 'probes/model_runner_v1.py'
    runner.write_text(once(runner.read_text(), '            _tfwd=sc.now()',
                           '            from vllm import prefill_timing\n            _tfwd=sc.now()\n'
                           '            prefill_timing.begin(_tfwd)'))
    launcher = code / 'launch.py'
    anchor = '    for src, dst, ro in mounts:'
    launcher.write_text(once(launcher.read_text(), anchor,
                        "    mounts.append((str(CODE/'probes/prefill_timing.py'),"
                        "'/vllm-workspace/vllm/vllm/prefill_timing.py',True))\n" + anchor))
    pins = read(code / 'PINS.json')
    for file in (helper, runner, launcher):
        pins[str(file.relative_to(code))] = digest(file)
    save(code / 'PINS.json', pins)
    abi = read(root / 'TRACE-ABI.json')
    abi.update(timed_layers=layers, attention_sha256=digest(source), patched=pins)
    save(root / 'TRACE-ABI.json', abi)


if __name__ == '__main__':
    build(read(Path(sys.argv[1])))
