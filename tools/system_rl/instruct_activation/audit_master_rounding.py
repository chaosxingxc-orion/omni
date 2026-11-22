"""Replay one observed update with only BF16 parameter writeback changed; CPU only."""
import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import torch


def check(ok, message):
    if not ok:
        raise ValueError(message)


def shard(directory, commit, rank):
    name = f'jev-master-rank-{rank}.pt'
    row, = (r for r in commit['ranks'] if r['rank'] == rank)
    data = (directory / name).read_bytes()
    check(hashlib.sha256(data).hexdigest() == row['files'][name], 'shard digest mismatch')
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        s = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    check((s['rank'], s['world_size'], s['completed_updates']) ==
          (rank, commit['world_size'], commit['step']), 'shard identity mismatch')
    return s


def audit(config, destination):
    restore = json.loads(config.read_bytes())['restore_checkpoint']
    src = Path(restore['directory'])
    previous = (src / 'COMMITTED.json').read_bytes()
    current = (destination / 'COMMITTED.json').read_bytes()
    check(hashlib.sha256(previous).hexdigest() == restore['sha256'], 'restore parent changed')
    m, n = map(json.loads, (previous, current))
    check(m['step'] + 1 == n['step'] and m['world_size'] == n['world_size'] and
          all(x['recoverable'] and x['kind']=='lora_training_state' for x in (m,n)), 'not adjacent recoverable steps')
    totals = {'A':Counter(), 'B':Counter()}
    torch.set_num_threads(1)
    for rank in range(n['world_size']):
        a, b = shard(src, m, rank), shard(destination, n, rank)
        check(a['names'] == b['names'] and a['parameters'].keys() == b['parameters'].keys(), 'shard key mismatch')
        for k, v in a['parameters'].items():
            w = b['parameters'][k]
            check(v.shape == w.shape and v.dtype == w.dtype == torch.float32 and
                  v.isfinite().all() and w.isfinite().all(), 'unexpected master')
            kind, = [j for j in totals if f'.lora_{j}.' in k]
            q = v.bfloat16().float()
            d = w - v
            changed = d != 0
            simulated = (q + d).bfloat16().float()
            totals[kind].update(elements=v.numel(), fp32_changed=int(changed.sum()),
                    no_change_from_bf16_start=int((changed & (simulated == q)).sum()),
                    same_published_value=int((changed & (w.bfloat16().float() == q)).sum()))
    return dict(parent=restore['sha256'], child=hashlib.sha256(current).hexdigest(), steps=[m['step'],n['step']],
                world_size=n['world_size'], counts=totals,
                scope='one fixed FP32 delta; deterministic nearest BF16 writeback, no gradient or optimizer rerun')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('training_config',type=Path)
    p.add_argument('updated_checkpoint',type=Path)
    a=p.parse_args()
    print(json.dumps(audit(a.training_config,a.updated_checkpoint),indent=2))
