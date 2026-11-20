"""Replay the two witnessed O LoRA A calls on one device, varying only row placement."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_tensor(root, name):
    meta = json.loads((root / (name + '.json')).read_bytes())
    raw = (root / (name + '.bin')).read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta['sha256'] or meta['dtype'] != 'torch.bfloat16':
        raise ValueError('operand differs from its sealed metadata')
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(meta['shape']).to('npu:0')


@torch.no_grad()
def run(config):
    import torch_npu
    torch.npu.set_device(0)
    cuts, batches, outputs = {}, {}, {}
    for lane, path in config['operands'].items():
        path = Path(path)
        records = json.loads((path.parent.parent / 'COMPLETE.json').read_bytes())['workers'][0]['prefix_trace']
        calls = records['gemm']
        cuts[lane] = calls[0]['prior']
        if any(c['rows'] != c['end'] or c['begin'] != 0 for c in calls):
            raise ValueError('unexpected padding, replay requires complete live rows')
        batches[lane] = [load_tensor(path, f'operand-{i}-x') for i in range(len(calls))]
        outputs[lane] = torch.cat([load_tensor(path, f'operand-{i}-u') for i in range(len(calls))])
        current = load_tensor(path, 'operand-0-a')
        if lane != next(iter(config['operands'])) and not torch.equal(weight, current):
            raise ValueError('live weights differ')
        weight = current
    cut = cuts['warm']
    full = torch.cat(batches['cold'])
    if cuts['cold'] or not torch.equal(full[cut:], torch.cat(batches['warm'])):
        raise ValueError('input mismatch')
    hf_shape_reference = F.linear(full, weight)
    def count(a, b):
        return int((a != b).sum())
    def compare(project):
        results = {lane:torch.cat([project(x) for x in rows]) for lane, rows in batches.items()}
        return dict(cold_to_actual=count(results['cold'],outputs['cold']),
                    warm_to_actual=count(results['warm'],outputs['warm']),
                    cold_to_warm=count(results['cold'][cut:],results['warm']),
                    cold_to_full_linear=count(results['cold'], hf_shape_reference),
                    warm_to_full_linear=count(results['warm'], hf_shape_reference[cut:]))
    records = {'linear':compare(lambda x:F.linear(x,weight))}
    for width in config['output_widths']:
        padded = F.pad(weight, (0,0,0,width-len(weight)))
        records['output_width_'+str(width)] = compare(lambda x:F.linear(x,padded)[:,:len(weight)])
    def grouped(x):
        sizes = torch.tensor([len(x)],dtype=torch.int64,device=x.device)
        out, = torch_npu.npu_grouped_matmul(x=[x],weight=[weight.T.contiguous().unsqueeze(0)],
                        split_item=2,group_list_type=1,group_type=0,group_list=sizes)
        return out
    records['grouped'] = compare(grouped)
    # The same complete matrix on the same device; only permute M and undo it.
    x=batches['cold'][1]
    shift=cut-cuts['cold']-len(batches['cold'][0])
    same=F.linear(x,weight)
    perm=F.linear(x.roll(-shift,0),weight).roll(shift,0)
    return dict(results=records,reference='single full-prompt BF16 linear on the same NPU',
                shift=shift,rows=len(x),permutation_differences=count(perm,same),
                repeated_call_differences=count(F.linear(x,weight),same))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_bytes())),indent=2))
