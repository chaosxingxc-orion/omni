"""Compare actual closed BF16 audio and merged prompt operands on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def tensors(path, expected):
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected: raise ValueError('tensor digest mismatch')
    end=8+int.from_bytes(data[:8],'little')
    header=json.loads(data[8:end])
    result={}
    for name,spec in header.items():
        if spec['dtype']!='BF16':raise ValueError('unexpected boundary dtype '+name)
        lo,hi=spec['data_offsets']
        result[name]=np.frombuffer(data,dtype='<u2',count=(hi-lo)//2,offset=end+lo).reshape(spec['shape'])
    return result


def diff(left,right):
    if left.shape!=right.shape:raise ValueError('boundary shape mismatch')
    f=lambda x:(x.astype('<u4')<<16).view('<f4')
    a,b=f(left),f(right)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite boundary')
    return dict(shape=list(a.shape),unequal=int(np.count_nonzero(left!=right)),max_abs=float(np.max(np.abs(a-b))))


def run(root):
    hf=json.loads((root/'hf/RESULT.json').read_bytes());vl=json.loads((root/'vl/RESULT.json').read_bytes())
    assert hf['layers']==32 and vl['actual_tp']==4 and vl['sampled_actions']==0
    assert sorted(x['rank'] for x in vl['ranks'])==[0,1,2,3]
    ref=tensors(root/'hf/HF.safetensors',hf['tensors_sha256'])
    names=['layer.0.in',*[f'layer.{i}.out' for i in range(32)],'audio_out']
    results=[]
    for rank in vl['ranks']:
        v=tensors(root/f"vl/rank-{rank['rank']}/VL.safetensors",rank['sha256'])
        rows={name:diff(ref[name],v[name]) for name in names}
        count=hf['prompt_tokens']
        rows['merged_prompt']=diff(ref['decoder_input'][0,:count],v['merged_prompt'])
        results.append(dict(rank=rank['rank'],boundaries=rows,first_difference=next((k for k,d in rows.items() if d['unequal']),None)))
    return dict(layers=32, ranks=results, confirmed_operator_root_cause=False, optimizer_updates=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=run(a.input)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
