"""Batch independent absolute-position prefill tiles inside the isolated measured packet."""
import sys
from pathlib import Path
from project_prefill_tile import build as tile
from project_screen_lanes import read,save
from decode_lineage import snapshot, changes, digest


def install_packed_tiles(limit):
    import torch
    import torch_npu
    from itertools import accumulate
    import retained_attention_candidate as ref

    single = ref.tiled
    masks = {}

    def packed(fia,q,k,v,tile):
        n,m=len(q),len(k)
        start=m-n
        if (tile != 128 or not 0 < n <= m <= 24576 or len(v)!=m
                or q.shape[1:]!=(8,128) or k.shape[1:]!=(1,128) or k.shape!=v.shape
                or q.dtype!=torch.bfloat16 or k.dtype!=q.dtype or v.dtype!=q.dtype
                or not (q.device==k.device==v.device) or torch.is_grad_enabled()):
            raise ValueError('unexpected packed inference tile operands')
        lo0=start//tile*tile
        if m-lo0<=tile:
            return single(fia,q,k,v,tile)
        if q.device not in masks:
            masks[q.device]=torch.ones((2048,2048),dtype=torch.bool,device=q.device).triu_(1)
        end=((m+tile-1)//tile)*tile
        # Reuse one padded logical K/V tail. All earlier prefixes are views.
        def extend(x):
            return torch.cat((x,x.new_zeros((end-m,*x.shape[1:])))) if end!=m else x
        kfull,vfull=extend(k),extend(v)
        qfull=q.new_zeros((end-lo0,*q.shape[1:]))
        qfull[start-lo0:m-lo0].copy_(q)
        out=[]
        for first in range(lo0,end,tile*limit):
            last=min(first+tile*limit,end)
            lens=list(range(first+tile,last+1,tile))
            qq=qfull[first-lo0:last-lo0]
            if len(lens)==1:
                out.append(fia(qq,kfull[:last],vfull[:last]))
            else:
                kk=torch.cat([kfull[:l] for l in lens])
                vv=torch.cat([vfull[:l] for l in lens])
                result=torch_npu.npu_fused_infer_attention_score(
                    query=qq,key=kk,value=vv,atten_mask=masks[q.device],input_layout='TND',
                    sparse_mode=3,actual_seq_lengths=list(range(tile,len(qq)+1,tile)),
                    actual_seq_lengths_kv=list(accumulate(lens)),num_heads=8,
                    num_key_value_heads=1,scale=128**-.5)[0]
                out.append(result.reshape_as(qq))
        return torch.cat(out)[start-lo0:m-lo0]
    ref.tiled=packed


def build(spec):
    size=spec['tiles_per_call']
    if type(size)!=int or not 2<=size<=8:
        raise ValueError('tile batch outside measured scope')
    tile(spec)
    root=Path(spec['projection']['target']);parent=Path(spec['measured_parent'])
    before=snapshot(parent)
    if snapshot(root)!=before:raise ValueError('parent differs')
    rel='parents/jev-system-rl/implementation/decode_tiled_attention.py'
    path=root/'bundle'/rel;rp=root/'bundle/ROWS.json';rows=read(rp)
    if list(range(48))!=rows['tiled_layers'] or rows['decode_attention_sha256']!=digest(path):
        raise ValueError('tiled attention identity differs')
    import inspect
    src=path.read_text();anchor='    cached = {}\n'
    if src.count(anchor)!=1:raise ValueError('unexpected attention install')
    src=src.replace(anchor,'    install_packed_tiles('+str(size)+')\n'+anchor)+ '\n\n'+inspect.getsource(install_packed_tiles)
    compile(src,rel,'exec');path.write_text(src);rows['decode_attention_sha256']=digest(path);save(rp,rows)
    for name in spec['projection']['lanes']:
        p=root/'bundle'/('SERVER-'+name+'.json');r=read(p)
        r['local_activation']['rows_ref'].update(sha256=digest(rp),bytes=rp.stat().st_size);save(p,r)
    d=changes(before,snapshot(root))
    if d['engine'] or set(d['files'])!={rel} or set(d['rows'])!={'decode_attention_sha256'}:
        raise ValueError('unexpected changes')
    save(root/'PACKED-TILES.json',dict(tiles_per_call=size,parent_sha256=digest(parent/'CANDIDATE.json'),changes=d))

if __name__=='__main__':build(read(Path(sys.argv[1])))
