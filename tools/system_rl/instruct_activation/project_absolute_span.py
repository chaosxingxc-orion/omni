"""Keep absolute BF16 attention boundaries but combine adjacent causal query tiles."""
import inspect, sys
from pathlib import Path
from project_prefill_tile import build as tile_parent
from project_screen_lanes import read,save
from decode_lineage import snapshot,changes,digest


def aligned_span(fia,q,k,v,tile):
    import torch
    n,m=len(q),len(k)
    if tile!=128 or not 0<n<=m<=24576 or len(v)!=m:
        raise ValueError('unexpected causal tile shape')
    start=m-n
    lo=start//tile*tile
    end=(m+tile-1)//tile*tile
    # With bottom-right sparse_mode=3, row r is allowed keys [0,lo+r].
    # Adding rows ahead of the real query and padding the shared KV tail keeps
    # this boundary at the same absolute position even for a mid-page resume.
    leading=start-lo
    qq=q if not leading and m==end else q.new_zeros((end-lo,*q.shape[1:]))
    if qq is not q:qq[leading:leading+n].copy_(q)
    def right(x):return x if end==m else torch.cat((x,x.new_zeros((end-m,*x.shape[1:]))))
    return fia(qq,right(k),right(v))[leading:leading+n]


def install_span():
    import retained_attention_candidate as ref
    prior=ref.tiled
    def dispatch(fia,q,k,v,tile):
        if (len(k)-len(q))//tile == (len(k)-1)//tile:
            return prior(fia,q,k,v,tile)
        return aligned_span(fia,q,k,v,tile)
    ref.tiled=dispatch


def build(spec):
    if spec['query_layout']!='one_absolute_span':raise ValueError('unknown layout')
    tile_parent(spec)
    root=Path(spec['projection']['target']);parent=Path(spec['measured_parent'])
    before=snapshot(parent)
    if snapshot(root)!=before:raise ValueError('parent differs')
    rel='parents/jev-system-rl/implementation/decode_tiled_attention.py'
    path=root/'bundle'/rel;rp=root/'bundle/ROWS.json';rows=read(rp)
    if list(range(48))!=rows['tiled_layers'] or rows['decode_attention_sha256']!=digest(path):
        raise ValueError('tiled attention identity differs')
    src=path.read_text();anchor='    cached = {}\n'
    if src.count(anchor)!=1:raise ValueError('unexpected attention install')
    src=src.replace(anchor,'    install_span()\n'+anchor)+'\n\n'+inspect.getsource(aligned_span)+'\n\n'+inspect.getsource(install_span)
    compile(src,rel,'exec');path.write_text(src);rows['decode_attention_sha256']=digest(path);save(rp,rows)
    for name in spec['projection']['lanes']:
        p=root/'bundle'/('SERVER-'+name+'.json');r=read(p)
        r['local_activation']['rows_ref'].update(sha256=digest(rp),bytes=rp.stat().st_size);save(p,r)
    d=changes(before,snapshot(root))
    if d['engine'] or set(d['files'])!={rel} or set(d['rows'])!={'decode_attention_sha256'}:raise ValueError('unexpected changes')
    save(root/'ABSOLUTE-SPAN.json',dict(parent_sha256=digest(parent/'CANDIDATE.json'),changes=d,layout=spec['query_layout']))

if __name__=='__main__':build(read(Path(sys.argv[1])))
