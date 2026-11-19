"""Replay competing bias-rounding formulas on retained operands (CPU controls only)."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_retained_layer import read_at


def weight(model,index,name):
    with (model/index[name]).open('rb') as f:
        size=int.from_bytes(f.read(8),'little')
        d=json.loads(f.read(size))[name]
        lo,hi=d['data_offsets'];f.seek(8+size+lo)
        assert d['dtype']=='BF16'
        return np.frombuffer(f.read(hi-lo),dtype='<u2').reshape(d['shape'])


def bits(x):
    u=x.astype('<f4').view('<u4')
    return ((u+0x7fff+((u>>16)&1))>>16).astype('<u2')


def fp(u):return (u.astype('<u4')<<16).view('<f4')


def controls(x,w,b,actual):
    # Float64 accumulation is a mathematical control, not an Ascend replay.
    prod=fp(bits(x)).astype('float64')@fp(w).astype('float64').T
    bias=fp(b)
    variants={'one_round':bits(prod+bias),'round_before_bias':bits(fp(bits(prod))+bias),'no_bias':bits(prod)}
    return {k:int(np.count_nonzero(v!=bits(actual))) for k,v in variants.items()}


def analyze(root,model):
    idx=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map']
    def inputs(folder,row,side,rank):return read_at(folder,row['layer_sha256'],side,rank)
    hd,h=inputs(root/'hf',json.loads((root/'hf/RESULT.json').read_bytes()),'hf',0)
    ranks=json.loads((root/'vl/RESULT.json').read_bytes())['ranks']
    rows=[]
    for row in ranks:
        r=row['rank'];vd,v=inputs(root/f'vl/rank-{r}',row,'vl',r)
        assert hd['layer']==vd['layer']
        for i,name in enumerate(('q','k','v','fc1')):
            head=f"thinker.audio_tower.layers.{hd['layer']}."+('fc1' if name=='fc1' else 'self_attn.'+name+'_proj')
            w,b=(np.split(weight(model,idx,head+'.'+p),4)[r] for p in ('weight','bias'))
            for val,param in [(w,'weight'),(b,'bias')]:
                key=f'{name}.{param}.rank{r}'
                assert hashlib.sha256(val.tobytes()).hexdigest()==hd['weights'][key]['sha256']==vd['weights'][key]['sha256']
            xin=h[name+'.in'][0];target=np.split(h[name+'.out'][0],4,axis=-1)[r]
            if name=='fc1':vx,vy=v['fc1.in'][0],v['fc1.out'][0]
            else:vx,vy=v['qkv.in'][0],np.split(v['qkv.out'][0],3,axis=-1)[i]
            rows.append(dict(rank=r,projection=name,shape=list(target.shape),
                             input_unequal_bf16=int(np.count_nonzero(bits(xin)!=bits(vx))),
                             hf=controls(xin,w,b,target),vl=controls(vx,w,b,vy)))
    row_controls=[]
    for name,leaf in [('out','self_attn.out_proj'),('fc2','fc2')]:
        head=f"thinker.audio_tower.layers.{hd['layer']}."+leaf
        w,b=(weight(model,idx,head+'.'+p) for p in ('weight','bias'))
        parts=[];observed=[]
        for spec in sorted(ranks,key=lambda z:z['rank']):
            r=spec['rank'];vd,v=inputs(root/f'vl/rank-{r}',spec,'vl',r)
            for val,param in [(np.split(w,4,axis=1)[r],'weight'),(b,'bias')]:
                key=f'{name}.{param}.rank{r}'
                assert hashlib.sha256(val.tobytes()).hexdigest()==hd['weights'][key]['sha256']==vd['weights'][key]['sha256']
            parts.append(v[name+'.in'][0]);observed.append(bits(v[name+'.out'][0]))
        assert all(np.array_equal(observed[0],o) for o in observed)
        products=[x.astype('float64')@fp(ws).astype('float64').T for x,ws in zip(parts,np.split(w,4,axis=1))]
        bias=fp(b);total=sum(products)
        partials=[fp(bits(x)) for x in products]
        forms={'full':bits(total+bias),'full_round_then_bias':bits(fp(bits(total))+bias),
               'partial_round_then_sum_bias':bits(sum(partials)+bias),
               'partial_round_then_sum_round_then_bias':bits(fp(bits(sum(partials)))+bias),
               'bias_in_rank0_before_partial_round':bits(fp(bits(products[0]+bias))+sum(partials[1:]))}
        # HCCL may reduce the rounded partials in BF16. Keep plausible tree
        # orders explicit; a matching formula alone is not a kernel trace.
        acc=[fp(bits(products[0]+bias)),*partials[1:]]
        for order in [(0,1,2,3),(0,2,1,3),(0,3,1,2)]:
            a0,a1,a2,a3=(acc[i] for i in order)
            forms['bf16_tree_'+''.join(map(str,order))]=bits(fp(bits(a0+a1))+fp(bits(a2+a3)))
        def counts(target):return {k:int(np.count_nonzero(target!=y)) for k,y in forms.items()}
        row_controls.append(dict(name=name,input_unequal=int(np.count_nonzero(np.concatenate(parts,-1)!=h[name+'.in'][0])),
            vl=counts(observed[0]),hf_with_vl_inputs=counts(bits(h[name+'.out'][0]))))
    return dict(layer=hd['layer'],row_controls=row_controls,rows=rows,device='CPU',method='FP64 product, BF16 operands, compare BF16 outputs',
                measured_kernel_cause=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.config.read_bytes());report=analyze(Path(c['input']),Path(c['model']))
    with Path(c['output']).open('x') as f:json.dump(report,f,indent=2)
