"""Measure complete segments outside the captured graphs on the pinned TP4 candidate."""
from pathlib import Path
import sys

from decode_lineage import digest, snapshot
from io_utils import digest as digest_state
from project_dense_tile_graph import build as make_graph
from project_prefill_timing import once
from project_stage_trace import bind
from project_screen_lanes import read, save


def surround_block(source, begin, end, phase):
    """Include begin up to, but excluding, the statement at end."""
    first = '                ' + begin
    following = '                ' + end
    if source.index(first) >= source.index(following):
        raise ValueError('reordered prefill segments')
    source = once(source, first, '                stamp = timing.start(i)\n' + first)
    return once(source, following, f"                timing.end(i, '{phase}', stamp)\n" + following)


def timed(source, layers):
    source = once(source, '    owners = worker._decode_owner_weights',
                  '    from vllm import prefill_timing as timing\n'
                  f'    timing.configure({layers!r})\n    owners = worker._decode_owner_weights')
    source = once(source, '                full=gather(layer.self_attn.attn(q,k,v).contiguous(),dim=1)',
                  '                attended = layer.self_attn.attn(q,k,v).contiguous()\n'
                  '                full = gather(attended, dim=1)')
    source = once(source, '                rows=reduce(torch.cat(routed,dim=0)).reshape(n,8,2048)',
                  '                flat = torch.cat(routed,dim=0)\n'
                  '                rows = reduce(flat).reshape(n,8,2048)')
    source = once(source, '                return (rows*torch.cat(weights,dim=0)[:,:,None]).sum(dim=1),torch.cat(new_residual,dim=0)',
                  '                answer = (rows*torch.cat(weights,dim=0)[:,:,None]).sum(dim=1),torch.cat(new_residual,dim=0)\n'
                  '                return answer')
    boundaries = ['pieces=[]','attended =','full = gather','routed=[]','flat = torch.cat','rows = reduce','answer =','return answer']
    phases = ['before_attention','attention_kv','attention_gather','post_attention','expert_cat','expert_reduce','weighted_sum']
    for start, stop, phase in zip(boundaries[:-1], boundaries[1:], phases, strict=True):
        source = surround_block(source, start, stop, phase)
    decode = [('            q,k,v,total=one(hidden_states,rs,*trig,mask)', 'decode_before'),
              ('            attention_output=layer.self_attn.attn(q,k,v)', 'decode_kv'),
              ('            full=gather(attention_output.contiguous(),dim=1)', 'decode_gather'),
              ('            local,w,total=two(full,total,mask)', 'decode_after'),
              ('            rows=reduce(local)', 'decode_reduce'),
              ('            out=(rows.reshape(n,8,2048)*w[:,:,None]).sum(dim=1)', 'decode_sum')]
    for line,phase in decode:
        source=once(source,line+'\n','            stamp = timing.start(i)\n'+line+'\n'
                    +f"            timing.end(i, '{phase}', stamp)\n")
    compile(source,'decode_layer_graph.py','exec')
    return source


def build(spec):
    layers = spec['timed_layers']
    if not layers or layers != sorted(set(layers)) or any(type(i)!=int or i not in range(48) for i in layers):
        raise ValueError('invalid observed layers')
    make_graph(spec)
    root=Path(spec['projection']['target'])
    parent=Path(spec['timing_parent'])
    if digest_state(snapshot(parent)) != read(Path(spec['graph_lock']))['runtime_sha256']:
        raise ValueError('expected measured graph baseline')
    bind(spec,root,parent)
    lane,=spec['projection']['lanes']
    code=root/('lane-'+lane)
    target=root/'bundle/parents/jev-system-rl/implementation/decode_layer_graph.py'
    rp=root/'bundle/ROWS.json'
    rows=read(rp)
    if digest(target)!=rows['decode_layer_graph']['source_sha256']:
        raise ValueError('layer source differs')
    target.write_text(timed(target.read_text(),layers))
    rows['decode_layer_graph']['source_sha256']=digest(target)
    save(rp,rows)
    server=root/'bundle'/('SERVER-'+lane+'.json')
    cfg=read(server)
    cfg['local_activation']['rows_ref'].update(sha256=digest(rp),bytes=rp.stat().st_size)
    save(server,cfg)
    helper=code/'probes/prefill_timing.py'
    helper.write_bytes(Path(__file__).with_name(helper.name).read_bytes())
    runner=code/'probes/model_runner_v1.py'
    runner.write_text(once(runner.read_text(),'            _tfwd=sc.now()',
                           '            from vllm import prefill_timing\n            _tfwd=sc.now()\n'
                           '            prefill_timing.begin(_tfwd)'))
    launcher=code/'launch.py'
    anchor='    for src, dst, ro in mounts:'
    launcher.write_text(once(launcher.read_text(),anchor,
                            "    mounts.append((str(CODE/'probes/prefill_timing.py'),"
                            "'/vllm-workspace/vllm/vllm/prefill_timing.py',True))\n"+anchor))
    pins=read(code/'PINS.json')
    for file in (helper,runner,launcher):
        pins[str(file.relative_to(code))]=digest(file)
    save(code/'PINS.json',pins)
    abi=read(root/'TRACE-ABI.json')
    abi.update(timed_layers=layers,patched=pins,layer_source_sha256=digest(target))
    save(root/'TRACE-ABI.json',abi)


if __name__=='__main__':
    build(read(Path(sys.argv[1])))
