"""Reassemble real FSDP LoRA shards and compare two first-step observations."""
import argparse
import json
import math
import re
from pathlib import Path
import torch


def read_arm(root):
    exit_record = json.loads((root/'EXIT.json').read_bytes())
    world = exit_record['complete_ranks']
    if exit_record['exit_code'] != 0 or world not in (4, 8):
        raise ValueError('incomplete FSDP observation')
    pieces = {}
    results = []
    for rank in range(world):
        loc = root/f'rank-{rank}'
        result = json.loads((loc/'RESULT.json').read_bytes())
        assert result['step'] == 1 and result['rank'] == rank
        results.append(result)
        before, after = [torch.load(loc/f'UPDATE-{phase}.pt', map_location='cpu', weights_only=True)
                         for phase in ('before', 'after')]
        assert set(before) == set(after) and len(before) == 384
        for key, row in before.items():
            name = key.replace('_fsdp_wrapped_module.', '').replace('_checkpoint_wrapped_module.', '')
            a, b = row['parameter'], after[key]['parameter']
            if a.numel():
                assert a.dtype == b.dtype == torch.float32 and after[key]['step'].item() == 1
                data = dict(initial=a, grad=row['gradient'], delta=b-a,
                            m=after[key]['exp_avg'], v=after[key]['exp_avg_sq'], updated=b)
                group = pieces.setdefault(name, {})
                for field, tensor in data.items():
                    assert tensor.shape == a.shape and torch.isfinite(tensor).all()
                    group.setdefault(field, []).append(tensor.flatten())
    rows = sorted(row['row'] for result in results for row in result['actions'])
    assert rows == list(range(8)) and sum(row['tokens'] for result in results for row in result['actions']) == 86
    return {n:{k:torch.cat(v) for k,v in group.items()} for n,group in pieces.items()}, results[0]['objective']


def compare(a, b):
    left, lobj = read_arm(a)
    right, robj = read_arm(b)
    assert lobj == robj and left.keys() == right.keys() and len(left) == 384
    table = {}
    for name in left:
        kind = re.search(r'lora_([AB])\.', name).group(1)
        for field, x in left[name].items():
            y = right[name][field]
            assert x.shape == y.shape
            group = table.setdefault(kind, {}).setdefault(field, dict(elements=0, different=0,
                max_abs=0., worst=None, squared_difference=0., squared_reference=0.))
            diff = x.double()-y.double()
            peak = diff.abs().max().item()
            if peak > group['max_abs']:
                group.update(max_abs=peak,worst=name)
            group['elements'] += x.numel()
            group['different'] += diff.count_nonzero().item()
            group['squared_difference'] += diff.square().sum().item()
            group['squared_reference'] += x.double().square().sum().item()
    for g in table.values():
        for row in g.values():
            norm = math.sqrt(row.pop('squared_reference'))
            err = math.sqrt(row.pop('squared_difference'))
            row['relative_l2'] = err/norm if norm else None
    return dict(rows=8,tokens=86,loss=lobj['components']['total'],left=a.name,right=b.name,comparison=table)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('left','right','output'):p.add_argument('--'+k, required=True, type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    with args.output.open('x') as out:
        json.dump(compare(args.left,args.right),out,indent=2,allow_nan=False)
