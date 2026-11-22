"""Reuse only the mixed/prefill KV address plan within one forward across layers."""
from pathlib import Path
import inspect,sys
from project_prefill_tile import build as tile
from project_screen_lanes import read,save
from decode_lineage import snapshot,changes,digest


def make_reader(context):
    import torch
    owner=None
    plans={}
    def read_kv(k,v,table,lengths):
        nonlocal owner
        if (k.ndim!=4 or k.shape!=v.shape or k.dtype!=torch.bfloat16 or v.dtype!=k.dtype
                or table.ndim!=2 or table.shape[0]!=len(lengths) or table.dtype not in (torch.int32,torch.int64)
                or not (table.device==k.device==v.device) or not lengths):
            raise ValueError('invalid logical KV buffers')
        size=k.shape[1]
        if size!=128 or any(type(n)!=int or not 0<n<=size*table.shape[1] for n in lengths):
            raise ValueError('unsupported block geometry')
        now=context()
        if now is None:raise ValueError('missing forward context')
        if owner is not now:
            plans.clear()
            owner=now
        key=(table.device,table.dtype,table.data_ptr(),tuple(table.stride()),tuple(table.shape),tuple(lengths),len(k))
        if key not in plans:
            counts=[(n+size-1)//size for n in lengths]
            ids=torch.cat([table[i,:b] for i,b in enumerate(counts)]).long()
            if bool(((ids<0)|(ids>=len(k))).any()):raise ValueError('invalid active block')
            # Selected pages are packed in sequence order. Strip interior padding,
            # independently of the physical layout/strides of each layer cache.
            spans=[];off=0
            for n,count in zip(lengths,counts):
                spans.append((off,n))
                off+=count*size
            plans[key]=(table,ids,spans)
        _,ids,spans=plans[key]
        def gather(cache):
            pages=cache.index_select(0,ids).flatten(0,1)
            if len(spans)==1:return pages[:spans[0][1]]
            return torch.cat([pages[start:start+n] for start,n in spans])
        return gather(k),gather(v)
    return read_kv


def install_mixed_kv():
    import paged_attention_contract as paged
    from vllm.forward_context import get_forward_context
    paged.logical_kv=make_reader(get_forward_context)


def build(spec):
    if spec['reuse']!='forward_address_plan':raise ValueError('unknown reuse scope')
    tile(spec)
    root=Path(spec['projection']['target']);parent=Path(spec['measured_parent'])
    before=snapshot(parent)
    if snapshot(root)!=before:raise ValueError('tile parent differs')
    rel='parents/jev-system-rl/implementation/decode_tiled_attention.py'
    path=root/'bundle'/rel;rp=root/'bundle/ROWS.json';rows=read(rp)
    if rows['decode_attention_sha256']!=digest(path):raise ValueError('attention identity differs')
    src=path.read_text();anchor='    cached = {}\n'
    if src.count(anchor)!=1:raise ValueError('unexpected install')
    src=src.replace(anchor,'    install_mixed_kv()\n'+anchor)+'\n\n'+inspect.getsource(make_reader)+'\n\n'+inspect.getsource(install_mixed_kv)
    compile(src,rel,'exec');path.write_text(src)
    rows['decode_attention_sha256']=digest(path);save(rp,rows)
    for name in spec['projection']['lanes']:
        p=root/'bundle'/('SERVER-'+name+'.json');r=read(p)
        r['local_activation']['rows_ref'].update(sha256=digest(rp),bytes=rp.stat().st_size);save(p,r)
    d=changes(before,snapshot(root))
    if d['engine'] or set(d['files'])!={rel} or set(d['rows'])!={'decode_attention_sha256'}:
        raise ValueError('unrelated runtime change')
    save(root/'MIXED-KV.json',dict(parent_sha256=digest(parent/'CANDIDATE.json'),changes=d))

if __name__=='__main__':build(read(Path(sys.argv[1])))
