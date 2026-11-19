"""Compare a closed natural replay against its sealed-audio intervention."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from collect_retained_boundaries import store


def load(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def summarize(config):
    root = Path(config['root'])
    paths = {k:root/v for k,v in config['inputs'].items()}
    docs, pins = {}, {}
    for k,p in paths.items():
        docs[k], pins[k] = load(p)
    base, control = docs['base'], docs['control']
    b, c = docs['base_lease'], docs['control_lease']
    for key in ('parent_sha256','audio_row_sites','action_index'):
        require(b[key]==c[key], 'input configuration differs: '+key)
    for name in ('features.json','S1-packet.json','P-packet.json','S1-BEHAVIOR.json','P-BEHAVIOR.json','retained-actions.json'):
        require(b['input_files'][name]==c['input_files'][name], 'different replay input: '+name)
    require(pins['actions']==c['input_files']['retained-actions.json'], 'HF action input differs')
    require('audio_control' not in b and 'audio_control' in c, 'missing independent audio intervention')
    target = c['input_files'][c['audio_control']['file']]
    require(base['model']==control['model'] and base['actual_tp']==control['actual_tp']==4,'backend mismatch')
    before = {r['rank']:r for r in base['ranks']}
    after = {r['rank']:r for r in control['ranks']}
    evidence = {r['rank']:r['intervention'] for r in control['decode_row_receipts']}
    ranks=set(range(4))
    require(set(before)==set(after)==set(evidence)==ranks,'rank coverage')
    for r in ranks:
        require(before[r]['precision']==after[r]['precision'] and
                before[r]['sha256']==after[r]['sha256'], 'precision or original audio tensors differ')
        require(evidence[r]['reference']==target and evidence[r]['observations'] and
                evidence[r]['observations']==evidence[0]['observations'], 'unverified audio replacement')
    limits={x['node']:x['limit'] for x in docs['actions']['cases']}
    rows={}
    for a,z in zip(base['retained_decode'],control['retained_decode'],strict=True):
        node=a['node']
        require(z['node']==node and node not in rows and z['original_tokens']==a['original_tokens'],'case identity')
        row={}
        for name,arm in [('native_audio',a),('hf_audio',z)]:
            d=arm['deltas'];n=arm['same_action_prefix']
            require(len(d)==n and 0<=n<=arm['original_tokens'] and all(math.isfinite(x) and x>=0 for x in d),'invalid prefix metric')
            row[name]=dict(original=arm['original_tokens'],matched=n,
                           positions_above_limit=[i+1 for i,v in enumerate(d) if v>limits[node]],
                           delta_by_matched_position=d)
        rows[node]=row
    require(set(rows)==set(limits),'missing cases')
    return dict(schema='retained-audio-intervention-comparison-v1',inputs=pins,
                same_native_ranks=4,confirmed_reference=target,per_rank_observations=evidence[0]['observations'],
                logprob_limits=limits,cases=rows,training_admission=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    spec,_=load(parser.parse_args().config)
    store(Path(spec['out']),json.dumps(summarize(spec),indent=2).encode())
