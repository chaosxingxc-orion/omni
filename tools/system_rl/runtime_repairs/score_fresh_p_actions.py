"""Bind naturally sampled P actions to reserved groups; use the original rails and reward."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

def original_parser(cfg,parents):
    from rails import bind_p
    from cohort_io import load as module
    original=module(parents/'full-system-collection-adapter/adapter.py','original_p_adapter')
    parser_ref,=[r for r in json.loads(Path(cfg['collection_config']).read_bytes())['code_pins'] if Path(r['path']).name=='ranked_candidate_indices.py']
    parser=Path(cfg['workspace'])/parser_ref['path']
    if hashlib.sha256(parser.read_bytes()).hexdigest()!=parser_ref['sha256']:raise ValueError('P parser changed')
    adapter=original.Adapter.__new__(original.Adapter)
    adapter.parse_indices=module(parser,'original_p_parse').parse
    bind_p(adapter)
    return adapter

def score(cfg):
    load=lambda p:json.loads(Path(p).read_bytes())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    root=Path(cfg['prepared']);parents=Path(cfg['release'])/'tree/parents'
    sys.path[:0]=[str(parents/'fresh-cohort-successor'),str(parents/'unified-learning')]
    from cohort_io import pin,read
    from cohort_capture import normalized
    from scoring import score_component
    from kernel import advantages
    adapter=original_parser(cfg,parents)
    pending=load(root/'PENDING-CAPTURE.json');plan=read(pending['selected_cohort_ref'])
    manifest=load(cfg['requests']);tp=load(cfg['result']);done=load(cfg['exit'])
    if (done['exit_code'] or not done['result_exists'] or tp['actual_tp']!=4 or
        tp['fresh_group']['manifest_sha256']!=sha(cfg['requests']) or
        manifest['selected_sha256']!=pending['selected_cohort_ref']['sha256'] or
        manifest['clock_sha256']!=sha(root/'RESERVED-CLOCK.json') or tp['fresh_group']['optimizer_steps']!=0):
        raise ValueError('unverified action source')
    learning=read(read(pending['input_ref'])['learning_config_ref'])
    if learning['scorer']['metric_weights']['P']!={'position_reward_v1':1.}:
        raise ValueError('unexpected reward')
    by_group={i:[] for i in range(len(plan['draws']))};seen=set()
    for row,act in zip(manifest['rows'],tp['fresh_group']['records'],strict=True):
        p=row['packet'];i,j=row['group'],row['replicate'];draw=plan['draws'][i]
        if ((i,j) in seen or (act['group'],act['replicate'])!=(i,j) or
            act['request_id']!=p['request_id'] or act['wire_sha256']!=p['wire_sha256'] or
            act['source_sha256']!=row['source_sha256'] or p['fresh_capture_identity']['group_id']!=draw['group_id'] or
            draw['record']['sample_id']!=p['sample_id'] or len(act['token_ids'])!=len(act['behavior_logprobs']) or not act['token_ids']):
            raise ValueError('missing or misattributed action')
        seen.add((i,j))
        reply={k:act[k] for k in ('text','finish_reason','request_id','wire_sha256')}
        parsed=adapter.parse_p(p,reply)
        online,out=normalized(p,parsed,reply,[])
        gold=draw['record']['offline_supervision']['labels']['gold_intent_id']
        if gold not in online['retrieved_candidates']:raise ValueError('unadmitted TRAIN sample')
        value=score_component('P',online,out,dict(intent=gold))
        if not value['mask']:raise ValueError('missing label')
        reward=(learning['scorer']['failure_policy']['component_values']['P'] if value.get('observed_model_failure') else value['metrics']['position_reward_v1'])
        by_group[i].append(dict(replicate=j,request_id=p['request_id'],tokens=len(act['token_ids']),terminal=out['terminal'],reward=reward))
    G=learning['kernel']['G']
    if seen!={(i,j) for i in by_group for j in range(G)}:raise ValueError('incomplete configured cohort')
    result=[]
    for i,items in by_group.items():
        items.sort(key=lambda item:item['replicate'])
        adv,mean,std=advantages([v['reward'] for v in items],learning['kernel']['algorithm'],learning['kernel']['epsilon'])
        result.append(dict(group_id=plan['group_ids'][i],observations=items,mean=mean,std=std,advantages=adv))
    return dict(schema='bounded-p-natural-action-rewards-v1',groups=result,behavior=sha(cfg['result']),
                requests=sha(cfg['requests']),learning=pin(Path(read(pending['input_ref'])['learning_config_ref']['path']))['sha256'],
                selected=manifest['selected_sha256'],optimizer_steps=0,training_admission=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_bytes());x=score(cfg)
    with Path(cfg['output']).open('x') as f:json.dump(x,f,indent=2)
