"""Measure disjoint intervals inside real prefill tiles on a diagnostic lane."""
from pathlib import Path
import sys
from project_stage_trace import build as trace_parent
from decode_lineage import digest
from project_screen_lanes import read,save

def once(source, old, new):
    if source.count(old)!=1:
        raise ValueError('changed prefill instrumentation site: '+old)
    return source.replace(old,new)

def patch(source, layers):
    source=once(source,'    owners = worker._decode_owner_weights',
                '    from vllm import prefill_timing as timing\n    timing.configure('+repr(layers)+')\n    owners = worker._decode_owner_weights')
    def mark(first, following, phase, conditional=True):
        nonlocal source
        line=next(x for x in source.splitlines() if x.lstrip().startswith(first))
        last=next(x for x in source.splitlines() if x.lstrip().startswith(following))
        indent=line[:len(line)-len(line.lstrip())]
        start='stamp=timing.start(i)'+(' if eager else None' if conditional else '')
        source=once(source,line+'\n',indent+start+'\n'+line+'\n')
        source=once(source,last+'\n',indent+"timing.end(i,'"+phase+"',stamp)\n"+last+'\n')
    # Take explicit end stamps before returns or the start of the next phase.
    mark('total=x if i==0','q,k,v=compute_projection','input_norm')
    mark('q,k,v=compute_projection','def rotate(','qkv_projection')
    source=once(source,'            return rotate(q,attention.q_norm.weight),rotate(k,attention.k_norm.weight),v,total',
                """            stamp=timing.start(i) if eager else None
            result=rotate(q,attention.q_norm.weight),rotate(k,attention.k_norm.weight),v,total
            timing.end(i,'qk_norm_rope',stamp)
            return result""")
    mark('total=compute_projection(full_attn','h=norm(total','output_projection')
    mark('h=norm(total','flat=ids.flatten()','norm_router_topk')
    mark('flat=ids.flatten()','first,second=grouped','expert_routing')
    mark('first,second=grouped','rows=torch.empty_like(y)','expert_matmuls_activation')
    source=once(source,'            return torch.where(own[:,None],rows,0),w,total',
                """            routed_rows=torch.where(own[:,None],rows,0)
            timing.end(i,'expert_restore',stamp)
            return routed_rows,w,total""")
    source=once(source,'            rows=torch.empty_like(y)', '            stamp=timing.start(i) if eager else None\n            rows=torch.empty_like(y)')
    source=once(source,'                full=gather(layer.self_attn.attn(q,k,v).contiguous(),dim=1)',
                """                stamp=timing.start(i)
                attention=layer.self_attn.attn(q,k,v).contiguous()
                timing.end(i,'attention_kv',stamp)
                stamp=timing.start(i)
                full=gather(attention,dim=1)
                timing.end(i,'attention_gather',stamp)""")
    source=once(source,'                rows=reduce(torch.cat(routed,dim=0)).reshape(n,8,2048)',
                """                stamp=timing.start(i)
                rows=reduce(torch.cat(routed,dim=0)).reshape(n,8,2048)
                timing.end(i,'expert_collective_and_cat',stamp)""")
    source=once(source,'                return (rows*torch.cat(weights,dim=0)[:,:,None]).sum(dim=1),torch.cat(new_residual,dim=0)',
                """                stamp=timing.start(i)
                result=(rows*torch.cat(weights,dim=0)[:,:,None]).sum(dim=1),torch.cat(new_residual,dim=0)
                timing.end(i,'weighted_sum',stamp)
                return result""")
    compile(source,'decode_layer_graph.py','exec')
    return source

def build(spec):
    layers=spec['timed_layers']
    if not layers or layers!=sorted(set(layers)) or any(type(i)!=int or not 0<=i<48 for i in layers):
        raise ValueError('expected unique layer IDs from the measured model')
    root=Path(spec['projection']['target'])
    if 'performance_lock' in spec:
        from project_prefill_dense_pages import build as build_dense
        from project_stage_trace import bind
        from decode_lineage import snapshot
        from io_utils import digest as digest_state
        parent=Path(spec['timing_parent'])
        expected=read(Path(spec['performance_lock']))['runtime_sha256']
        if digest_state(snapshot(parent)) != expected:
            raise ValueError('profiling parent differs from the performance baseline')
        build_dense(spec)
        bind(spec,root,parent)
    else:
        trace_parent(spec)
    code=root/('lane-'+next(iter(spec['projection']['lanes'])))
    proj=code/'probes'; target=root/'bundle/parents/jev-system-rl/implementation/decode_layer_graph.py'
    rp=root/'bundle/ROWS.json';rows=read(rp)
    if digest(target)!=rows['decode_layer_graph']['source_sha256']:raise ValueError('layer graph differs')
    target.write_text(patch(target.read_text(),layers))
    rows['decode_layer_graph']['source_sha256']=digest(target);save(rp,rows)
    server=root/'bundle'/('SERVER-'+next(iter(spec['projection']['lanes']))+'.json')
    conf=read(server);conf['local_activation']['rows_ref'].update(sha256=digest(rp),bytes=rp.stat().st_size);save(server,conf)
    helper=proj/'prefill_timing.py';helper.write_bytes(Path(__file__).with_name(helper.name).read_bytes())
    runner=proj/'model_runner_v1.py';s=runner.read_text();s=once(s,'            _tfwd=sc.now()','            from vllm import prefill_timing\n            _tfwd=sc.now()\n            prefill_timing.begin(_tfwd)')
    compile(s,str(runner),'exec');runner.write_text(s)
    launcher=code/'launch.py';s=launcher.read_text();a='    for src, dst, ro in mounts:'
    launcher.write_text(once(s,a,"    mounts.append((str(CODE/'probes/prefill_timing.py'),'/vllm-workspace/vllm/vllm/prefill_timing.py',True))\n"+a))
    pins=read(code/'PINS.json')
    for f in (launcher,runner,helper):pins[str(f.relative_to(code))]=digest(f)
    save(code/'PINS.json',pins)
    abi=read(root/'TRACE-ABI.json');abi.update(patched=pins,timed_layers=layers,layer_source_sha256=digest(target));save(root/'TRACE-ABI.json',abi)

if __name__=='__main__':build(read(Path(sys.argv[1])))
