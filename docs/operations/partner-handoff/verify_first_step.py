"""Verify the pinned first-step run on CPU, without importing model code."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(receipt, root):
    doc = json.loads(receipt.read_bytes())

    def check(ok, msg):
        if not ok:
            raise ValueError(msg)

    def match(rel, digest, parse=True):
        data = (root / rel).read_bytes()
        check(hashlib.sha256(data).hexdigest() == digest, f'digest mismatch: {rel}')
        return json.loads(data) if parse else len(data)

    check(doc['schema'] == 'p-first-fsdp4-optimizer-update-v1', 'wrong receipt')
    cfg = match('TRAIN.local.json', doc['config_digest'])
    end = match('hot/EXIT.json', doc['exit_digest'])
    check(cfg['training']['world_size'] == end['complete_ranks'] == 4 and end['exit_code'] == 0 and not end['failure'], 'incomplete run')
    rows = []
    for rank in range(4):
        res = match(f'hot/rank-{rank}/RESULT.json', doc['rank_results'][rank])
        loss = match(f'hot/rank-{rank}/LOSS.json', doc['rank_losses'][rank])
        check(loss['rank'] == res['rank'] == rank and loss['step'] == res['step'] == doc['optimizer_step'], 'wrong step/rank')
        obj = res['objective']
        check(obj == loss['result']['objective_observation'] and obj['components'] == doc['components']
              and obj['global_episode_denominator'] == doc['denominator'] and loss['result']['native_optimizer_step'], 'objective mismatch')
        check(res['adapter_export'] and res['publication_sha256'] == doc['publication_identity']
              and res['metrics']['actor/grad_norm'] == [doc['gradient_norm']], 'export or gradient mismatch')
        check(res['actions'] == loss['actions'], 'inconsistent action receipts')
        rows.extend(res['actions'])
    check(sorted(x['row'] for x in rows) == list(range(doc['training_rows']))
          and sum(x['tokens'] for x in rows) == doc['action_tokens'], 'row/token coverage mismatch')
    check(max(x['max_abs'] for x in rows) == doc['max_token_difference'] and
          max(x['sequence_abs'] for x in rows) == doc['max_sequence_difference'], 'parity mismatch')
    size = match('hot/step-1/lora_adapter/adapter_model.safetensors', doc['adapter_sha256'], False)
    check(size == doc['adapter_bytes'], 'adapter size mismatch')
    print(f"PASS: ranks=4 step={doc['optimizer_step']} rows={len(rows)} tokens={doc['action_tokens']} "
          f"loss={doc['components']['total']:.8f} grad_norm={doc['gradient_norm']:.8f} adapter_bytes={size}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    a = p.parse_args()
    verify(a.receipt, a.run_root)
