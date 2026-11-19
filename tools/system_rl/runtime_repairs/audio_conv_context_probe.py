"""Replay actual earliest audio operands under explicit execution contexts.

No model construction or output replacement. The two retained outputs remain
independent references; a matching operator does not certify model parity.
"""
import base64
import hashlib
import json
from pathlib import Path
import time


def analyze(c, device):
    import torch
    from torch.nn import functional as F
    start=time.monotonic();target=f'npu:{device}'
    if 'allow_internal_format' in c:
        assert type(c['allow_internal_format']) is bool
        torch.npu.config.allow_internal_format=c['allow_internal_format']
    bundle=json.loads(Path(c['operand_bundle']).read_text())
    def tensor(name):
        row=bundle[name];raw=base64.b64decode(row['data'],validate=True)
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        value=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(row['shape'])
        return value
    references={name:tensor(name) for name in ('hf','vllm')}
    x=tensor('input').to(target);weight=tensor('weight').to(target);bias=tensor('bias').to(target)
    rows=[]
    # Native archived input is channels-last; compare both physical layouts.
    for layout in ('channels_last','contiguous'):
        operand=x.contiguous(memory_format=torch.channels_last if layout=='channels_last' else torch.contiguous_format)
        for inference in (False,True):
            for autocast in (False,True):
                with torch.inference_mode(inference),torch.no_grad(),torch.autocast('npu',dtype=torch.bfloat16,enabled=autocast):
                    value=F.conv2d(operand,weight,bias,stride=2,padding=1).detach().cpu()
                comparisons={}
                for name,reference in references.items():
                    delta=(value.float()-reference.float()).abs()
                    comparisons[name]=dict(elements=value.numel(),unequal=int((value!=reference).sum()),max_abs=float(delta.max()))
                rows.append(dict(layout=layout,stride=list(operand.stride()),inference_mode=inference,
                    autocast=autocast,comparisons=comparisons,sha256=hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
                assert torch.npu.max_memory_allocated()<=c['max_device_bytes']
    return dict(schema='actual-audio-conv-context-v1',device=device,records=rows,
                allow_internal_format_requested=c.get('allow_internal_format'),
                operand_pins={k:v['sha256'] for k,v in bundle.items()},seconds=time.monotonic()-start,
                peak_allocated_bytes=torch.npu.max_memory_allocated(),optimizer_steps=0,production_acceptance=False)
