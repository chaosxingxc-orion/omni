"""Bind a diagnostic replay to original, closed rollout journals without inventing top-1."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(ref):
    raw = Path(ref['path']).read_bytes()
    if sha(raw) != ref['sha256'] or len(raw) != ref['bytes']:
        raise ValueError('immutable input changed: ' + ref['path'])
    return json.loads(raw)


def pack(cfg):
    inv = checked(cfg['inventory'])
    root = Path(inv['cohort_root'])
    if not inv['closed'] or not checked(inv['generation_closed_ref'])['groups']:
        raise ValueError('rollout not closed')
    files = inv['files']
    rows = []
    for name in sorted(files):
        if not Path(name).match('groups/*/episode-*/GENERATED.json'):
            continue
        ep = root / Path(name).parent
        lane = checked(files[str((ep/'LANE.json').relative_to(root))])
        act = checked(checked(lane['activation'])['activation_ref'])
        for item in checked(files[name])['calls']:
            packet = item['packet']
            env = item['envelope']
            def evidence(key):
                ref = env[key]
                if files[str(Path(ref['path']).relative_to(root))] != ref:
                    raise ValueError('journal differs from inventory')
                return checked(ref)
            wire, intent, journal = map(evidence, ('wire_ref', 'intent_ref', 'response_ref'))
            raw = base64.b64decode(journal['response_b64'], validate=True)
            body = json.loads(raw)
            choice, = body['choices']
            scores = choice['logprobs']['content']
            if not (wire == packet['wire'] and wire['model'] == body['model']
                    and intent['payload']['wire_sha256'] == packet['wire_sha256'] == env['wire_ref']['sha256']
                    and intent['request_id'] == packet['request_id'] == journal['request_id']
                    and journal['intent_sha256'] == env['intent_ref']['sha256']
                    and sha(raw) == journal['response_sha256'] == item['reply']['receipt']['response_sha256']
                    and body['prompt_token_ids'] == item['prompt_token_ids']
                    and choice['token_ids'] == item['token_ids']
                    and [s['token'] for s in scores] == [f'token_id:{t}' for t in choice['token_ids']]
                    and [s['logprob'] for s in scores] == item['behavior_logprobs']):
                raise ValueError('original HTTP behavior does not bind to training row')
            rows.append(dict(lane=lane['lane'], packet=packet, prompt=body['prompt_token_ids'],
                             tokens=choice['token_ids'], logprobs=item['behavior_logprobs'], raw_top1=None,
                             response=sha(raw), adapter_sha=act['weights_ref']['sha256']))
    if not rows or len({r['packet']['request_id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate requests')
    return dict(schema='closed-rollout-probability-diagnostic-v1', training_use=False,
                inventory_sha256=cfg['inventory']['sha256'], rows=rows, limits=cfg['limits'],
                projection_controls=cfg.get('projection_controls', []))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = pack(json.loads(a.config.read_bytes()))
    with a.output.open('x') as f:
        json.dump(result, f, allow_nan=False)
