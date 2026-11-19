"""Compare closed scores on original action IDs across three execution paths."""
from pathlib import Path
import argparse
import json
import math
from compare_retained_decode import load, require
from collect_retained_boundaries import store


def metrics(a, b, limit):
    delta=[x-y for x,y in zip(a,b,strict=True)]
    return dict(count=len(delta),max_abs=max(map(abs,delta),default=None),sequence_abs=abs(math.fsum(delta)),
                above_limit=[i for i,v in enumerate(delta,1) if abs(v)>limit])


def compare(cfg):
    root=Path(cfg['root'])
    lease,lpin=load(root/cfg['lease'])
    actions,apin=load(root/cfg['actions'])
    rs,rpin=load(root/cfg['result'])
    require(lease['input_files']['retained-actions.json']==apin and lease['score_retained_prefill'] is True,'action set identity')
    require(rs['actual_tp']==4 and rs['production_acceptance'] is False,'expected diagnostic TP4 engine')
    target=lease['input_files'][lease['audio_control']['file']]
    receipts=rs['decode_row_receipts']
    require(sorted(x['rank'] for x in receipts)==list(range(4)) and all(
        x['intervention']['reference']==target and x['intervention']['observations'] for x in receipts),'audio control missing')
    cases={}
    for c,z in zip(actions['cases'],rs['retained_decode'],strict=True):
        node=c['node']; limit=c['limit']; ref=c['hf_logprobs']; forced=z['forced_prefill']
        require(node==z['node'] and node not in cases,'case mismatch')
        behavior,bpin=load(root/c['behavior']['file'])
        require(bpin==lease['input_files'][c['behavior']['file']]==c['behavior']['sha256'],'behavior identity')
        ids=behavior['records'][c['index']]['token_ids']; natural=z['matching_prefix_scores']; scores=forced['scores']
        require(len(ref)==len(ids)==len(scores)==forced['actions_scored']==z['original_tokens'] and
                len(natural)==z['same_action_prefix']<=len(ids),'response alignment')
        require(all(math.isfinite(x) for arm in (natural,scores,ref) for x in arm),'nonfinite score')
        n=len(natural)
        rows=[dict(action=i+1,token=token,hf=ref[i],vl_prefill=scores[i],
                   vl_natural=natural[i] if i<n else None) for i,token in enumerate(ids)]
        a=metrics(scores,ref,limit); b=metrics(natural,ref[:n],limit); d=metrics(natural,scores[:n],limit)
        cases[node]=dict(actions=rows,limit=limit,hf_vs_prefill=a,hf_vs_natural_matched=b,
                         prefill_vs_natural_matched=d,natural_complete=n==len(ids),
                         prefix_cache_read_skipped_for_prompt_scores=forced['prefix_cache_read_skipped_for_prompt_scores'])
    return dict(inputs=dict(lease=lpin,actions=apin,result=rpin),cases=cases,training_admission=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path)
    cfg,_=load(p.parse_args().config)
    store(Path(cfg['out']),json.dumps(compare(cfg),indent=2).encode())
