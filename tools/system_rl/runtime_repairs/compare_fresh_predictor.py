"""Pair the two vLLM paths with the sealed same-action HF scores."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def compare(current, original, spec, hf):
    def load(path):
        return json.loads(path.read_bytes())

    cur, ref, run = map(load, (current, original, spec))
    cfg = run['fresh_group']
    if hashlib.sha256(original.read_bytes()).hexdigest() != run['files'][cfg['result']]:
        raise ValueError('original rollout not bound to the reference actor')
    now, old = (v['fresh_group'] for v in (cur, ref))
    def arithmetic(t):
        receipts=t['decode_row_receipts']
        if sorted(r['rank'] for r in receipts)!=list(range(4)) or t['actual_tp']!=4:
            raise ValueError('incomplete four-rank execution')
        return sorted((r['rank'], k, v['arithmetic'], sorted(v['layers']))
                      for r in receipts for k,v in r.items()
                      if isinstance(v,dict) and 'arithmetic' in v)
    if arithmetic(cur)!=arithmetic(ref):
        raise ValueError('HF reuse invalid: different observed arithmetic')
    if now['manifest_sha256'] != old['manifest_sha256'] or now['schema'] != 'fresh-train-p-predictor-trace-v1':
        raise ValueError('unpaired request manifest')
    row, = [r for r in now['records'] if 'predictor_trace' in r]
    trace = row['predictor_trace']
    idx = trace['config']['request_index']
    prev = old['records'][idx]
    for key in ('request_id','wire_sha256','source_sha256','group','replicate','seed','prompt_tokens','token_ids'):
        if row[key] != prev[key]:
            raise ValueError('HF reuse invalid: different ' + key)
    done = load(hf/'EXIT.json')
    if done['exit_code'] or done['failure'] or done['complete_ranks'] != 4:
        raise ValueError('incomplete HF control')
    rec, = [v for v in load(hf/f'rank-{idx%4}'/'RESULT.json')['records'] if v['index']==idx]
    if rec['request_id'] != row['request_id'] or row['cached_tokens'] != 0:
        raise ValueError('different request or not cold')
    scores = {}
    for k in ('behavior_logprobs','forced_scores'):
        d = [x-y for x,y in zip(row[k],rec['scores'],strict=True)]
        if not d or not all(map(math.isfinite,d)):
            raise ValueError('invalid action scores')
        scores[k] = dict(max_abs=max(map(abs,d)), sequence_abs=abs(math.fsum(d)),
                         unchanged=row[k]==prev[k], above_limit=[i+1 for i,x in enumerate(d) if abs(x)>.01])
    layers=[]
    for mode in ('decode','prefill'):
        if sorted(r['rank'] for r in trace[mode])!=list(range(4)):
            raise ValueError('incomplete rank population')
    for r in sorted(trace['prefill'],key=lambda x:x['rank']):
        if trace['config'].get('batch'):
            cold,=[v for v in trace['decode'] if v['rank']==r['rank']]
            for part in (cold,r):
                sched=part['scheduler']
                if sched['seed']!=row['seed'] or not 0<=sched['slot']<sched['batch'] or not sched['lo']<sched['hi']:
                    raise ValueError('wrong scheduler request')
                if part['observed_boundaries']!=98:
                    raise ValueError('incomplete batch trace')
            layers.append(dict(rank=r['rank'],target_first=r['first_observed_difference'],
                               natural=cold['scheduler'],forced=r['scheduler']))
            continue
        scan=r['prefix_scan']
        if r['observed_boundaries']!=98 or len(r['boundaries'])!=98 or scan['rows'][0]!=row['prompt_tokens']:
            raise ValueError('incomplete predictor or prefill coverage')
        layers.append(dict(rank=r['rank'], target_first=r['first_observed_difference'], prefix_first=scan['first'],
                           changed_prefix={k:v for k,v in scan['boundaries'].items() if v['unequal']}))
    return dict(index=idx, actions=len(row['token_ids']), action_ordinal=trace['config']['action_ordinal'],
                scores=scores,layers=layers,optimizer_steps=0,production_acceptance=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('current','original','spec','hf','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    result=compare(a.current,a.original,a.spec,a.hf)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
