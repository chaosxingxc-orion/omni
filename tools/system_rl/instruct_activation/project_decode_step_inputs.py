"""Share decode graph inputs that are constant across all layers of one forward."""
from pathlib import Path
import sys

from decode_lineage import changes, digest as file_digest, snapshot
from io_utils import digest, doc, pin, require, save
from project_packed_prefill_qkv import build as packed
from project_prefill_timing import once
from project_screen_lanes import save as write_config


def edit(s):
    s = once(s, '    def capture(callback, inputs, pool):', '''    step_buffers = {}
    step_mapping = None

    def capture(callback, inputs, pool, shared=()):''')
    s = once(s, '        inputs=tuple(v.clone() for v in inputs)',
             '        inputs=tuple(v if j in shared else v.clone() for j,v in enumerate(inputs))')
    s = once(s, '            for buf,value in zip(inputs,values,strict=True):\n                buf.copy_(value)',
             '''            for j,(buf,value) in enumerate(zip(inputs,values,strict=True)):
                if j not in shared:
                    buf.copy_(value)
                elif buf is not value:
                    raise ValueError('decode graph step input changed storage')''')
    s = once(s, 'one=capture(before,(hidden,residual,cos,sin,mask),pool)',
             'one=capture(before,(hidden,residual,cos,sin,mask),pool,shared=(2,3,4))')
    s = once(s, 'two=capture(after,(dummy,residual,mask),pool)',
             'two=capture(after,(dummy,residual,mask),pool,shared=(2,))')
    s = once(s, '            nonlocal trig,positions_key', '            nonlocal trig,positions_key,step_mapping')
    old='            mask=(p._token_lora_indices[:n]==0)[:,None] if not p.no_lora else torch.zeros((n,1),device=device,dtype=torch.bool)'
    s = once(s, old, '''            if pure:
                ids = p._token_lora_indices
                mapping = (p.no_lora, ids.data_ptr(), ids.stride(), n)
                if i == 0:
                    if n not in step_buffers:
                        step_buffers[n] = (torch.empty_like(trig[0]),torch.empty_like(trig[1]),
                                           torch.empty((n,1),device=device,dtype=torch.bool))
                    cbuf,sbuf,mbuf = step_buffers[n]
                    cbuf.copy_(trig[0]); sbuf.copy_(trig[1])
                    if p.no_lora:
                        mbuf.zero_()
                    else:
                        torch.eq(ids[:n],0,out=mbuf[:,0])
                    step_mapping = mapping
                elif step_mapping != mapping:
                    raise ValueError('different adapter mappings inside one decode forward')
                cbuf,sbuf,mask = step_buffers[n]
            else:
    ''' + old)
    s = once(s, 'build(i,n,hidden_states,rs,*trig,mask)', 'build(i,n,hidden_states,rs,cbuf,sbuf,mask)')
    s = once(s, 'q,k,v,total=one(hidden_states,rs,*trig,mask)', 'q,k,v,total=one(hidden_states,rs,cbuf,sbuf,mask)')
    s = once(s, '        pools.clear()', '        pools.clear()\n        step_buffers.clear()')
    compile(s, 'decode_layer_graph.py', 'exec')
    return s


def build(c):
    packed(c)
    root=Path(c['projection']['target'])
    before=snapshot(Path(c['step_inputs_parent']))
    proof=doc(c['step_inputs_proof'])
    require(proof['passed'] and proof['runtime']==digest(before) and snapshot(root)==before,
            'step inputs require the training-reviewed packed runtime')
    rel='parents/jev-system-rl/implementation/decode_layer_graph.py'
    src,rp=root/'bundle'/rel,root/'bundle/ROWS.json'
    rows=doc(pin(rp)); graph=rows['decode_layer_graph']
    require(file_digest(src)==graph['source_sha256'] and graph['capture_segments']=='both','unexpected graph')
    src.write_text(edit(src.read_text())); graph['source_sha256']=file_digest(src)
    write_config(rp,rows)
    for lane in c['projection']['lanes']:
        srv=root/'bundle'/('SERVER-'+lane+'.json'); spec=doc(pin(srv))
        spec['local_activation']['rows_ref'].update(bytes=rp.stat().st_size,sha256=file_digest(rp))
        write_config(srv,spec)
    delta=changes(before,snapshot(root))
    require(not delta['engine'] and set(delta['rows'])=={'decode_layer_graph'}
            and set(delta['files'])=={rel}, 'unrelated runtime difference')
    save(root/'STEP-INPUTS.json',dict(immediate_parent=digest(before),changes=delta))


if __name__=='__main__':
    build(doc(pin(sys.argv[1])))
