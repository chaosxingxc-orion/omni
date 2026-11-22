"""Use the pinned stock Ascend attention implementation inside the measured tiled engine."""
import sys
from pathlib import Path
from project_prefill_tile import build as tile
from project_screen_lanes import read,save
from decode_lineage import changes,digest,snapshot

def build(spec):
    if spec['attention_scope'] not in ('decode','all'):
        raise ValueError('unknown attention scope')
    tile(spec)
    root=Path(spec['projection']['target']);parent=Path(spec['measured_parent'])
    base=snapshot(parent)
    if snapshot(root)!=base:raise ValueError('runtime differs from measured tile parent')
    bundle=root/'bundle';rel='parents/jev-system-rl/implementation/decode_tiled_attention.py'
    src=bundle/rel; rows=read(bundle/'ROWS.json')
    if rows['decode_attention_sha256']!=digest(src):raise ValueError('attention source mismatch')
    family=read(bundle/'parents/jev-system-rl/production-family93/FAMILY-RUNTIME.json')
    backend=family['source_pins']['attention']
    old='        original = impl.forward_fused_infer_attention'
    new='''        import hashlib,inspect
        from pathlib import Path
        default=type(impl).forward_fused_infer_attention
        if hashlib.sha256(Path(inspect.getfile(default)).read_bytes()).hexdigest() != BACKEND_PIN:
            raise ValueError('stock attention source changed')
        stock=default.__get__(impl,type(impl))
'''
    if spec['attention_scope']=='all':
        new+='        return stock'
    else:
        new+='''        original=impl.forward_fused_infer_attention
        def route(query,key,value,attn_metadata,output,kv_cache=None):
            n=len(query)
            pure=(1<=n<=max_batch and str(attn_metadata.attn_state).split('.')[-1]=='DecodeOnly'
                  and list(attn_metadata.actual_seq_lengths_q)==list(range(1,n+1)))
            return (stock if pure else original)(query,key,value,attn_metadata,output,kv_cache=kv_cache)
        return route'''
    before=src.read_text()
    lo=before.index(old);hi=before.index('    for index, layer in enumerate(text.layers):',lo)
    if before.count(old)!=1:raise ValueError('unexpected wrapper structure')
    candidate=before[:lo]+new.replace('BACKEND_PIN',repr(backend))+'\n\n'+before[hi:]
    compile(candidate,rel,'exec');src.write_text(candidate)
    rows['decode_attention_sha256']=digest(src)
    rp=bundle/'ROWS.json';save(rp,rows)
    for name in spec['projection']['lanes']:
        path=bundle/('SERVER-'+name+'.json'); server=read(path)
        server['local_activation']['rows_ref'].update(bytes=rp.stat().st_size,sha256=digest(rp))
        save(path,server)
    diff=changes(base,snapshot(root))
    if diff['engine'] or set(diff['files'])!={rel} or set(diff['rows'])!={'decode_attention_sha256'}:
        raise ValueError('unrelated runtime changed')
    save(root/'ATTENTION.json',dict(parent_sha256=digest(parent/'CANDIDATE.json'),scope=spec['attention_scope'],backend=backend,changes=diff))

if __name__=='__main__':build(read(Path(sys.argv[1])))
