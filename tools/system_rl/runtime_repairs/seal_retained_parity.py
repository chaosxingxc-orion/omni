"""Derive a scoped acceptance record from original TP/HF paired runs."""
import argparse
import hashlib
import json
from pathlib import Path
from compare_fresh_group import compare


def lock(root):
    def load(p):return json.loads(p.read_bytes())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    runs={};identities=[]
    for mode,tag in [('cold','fresh-tiled-all-v1'),('shared','fresh-tiled-shared-v1')]:
        run=root/tag
        tpfile=run/'hot/RESULT.json'; spec=run/'hf/HF-GROUP.local.json'; tpspec=root/('tp4-'+tag+'.local.json')
        hot=run/'hf/hot';out=compare(hot,spec,tpfile)
        tp=load(tpfile);hf=load(spec);ts=load(tpspec);rows=tp['fresh_group']['records']
        done=load(run/'EXIT.json')
        if done['exit_code'] or not done['result_exists'] or out['passed_requests']!=8 or out['requests']!=8 or out['max_abs'] or out['max_sequence_abs'] or any(r['strict_flips'] for r in out['rows']):
            raise ValueError('four-card exact result not established')
        if hf['devices']!=ts['devices'] or hf['training']['world_size']!=4 or set(hf['devices'])!={0,1,2,3}:raise ValueError('different topology')
        for k in ['router_layers','kv_layers','tiled_layers']:
            if ts[k]!=list(range(48)) or hf['fresh_group'][k]!=ts[k]:raise ValueError('scope mismatch')
        pins=dict(tp_source={n:h for n,h in ts['input_files'].items() if n.endswith('.py')},
                  hf_source={n:h for n,h in {**hf['files'],**hf['source_pins']}.items() if n.endswith('.py')},
                  tp_image=ts['image'],hf_image=hf['image'],manifest=tp['fresh_group']['manifest_sha256'])
        identities.append(pins)
        hits=[r['cached_tokens'] for r in rows]
        if not(hits==[0]*8 if mode=='cold' else any(hits) and 0 in hits):raise ValueError('cache condition not observed')
        runs[mode]=dict(requests=8,tokens=out['tokens'],max_token_delta=0,max_sequence_delta=0,raw_top1_flips=0,
                        cached_tokens=hits,report=sha(tpfile),hf_results=[sha(hot/f'rank-{i}/RESULT.json') for i in range(4)],
                        tp_config=sha(tpspec),hf_config=sha(spec),tp_exit=sha(run/'EXIT.json'),hf_exit=sha(hot/'EXIT.json'))
    if identities[0]!=identities[1]:raise ValueError('arithmetic changed between cache modes')
    actor=root/'tiled-kv-actor-v1';results=[load(actor/f'rank-{i}/RESULT.json') for i in range(4)]
    exit=load(actor/'EXIT.json');g=results[0]['actor_gradient']
    if exit['exit_code'] or exit['complete_ranks']!=4 or g['globally_covered']!=6684672 or not g['global_norm']>0 or g['optimizer_steps'] or g['forward_max_abs'] or any(r['scores']!=g['live_scores'] or r['actor_gradient']!=g or len(r['rows'])!=98 for r in results):
        raise ValueError('actor gradient evidence incomplete')
    return dict(schema='same-host-frozen-p-parity-v1',model='Qwen3-Omni-30B-A3B-Instruct',precision='BF16',thinking=False,
                scope='P; two authentic TRAIN recordings, four seeds per recording per cache mode; frozen policy',
                topology=dict(inference='vLLM TP4',training_reference='HF FSDP4',same_devices=True),
                runs=runs,pins=identities[0],actor=dict(fixed_actions=len(g['live_scores']),covered_lora_parameters=g['globally_covered'],
                  global_gradient_norm=g['global_norm'],optimizer_steps=0,config=sha(root/'tiled-kv-actor-v1.local.json'),
                  rank_receipts=[sha(actor/f'rank-{i}/RESULT.json') for i in range(4)]),
                excluded=['updated adapter export/reload','production throughput','training reward','other nodes and populations'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();value=lock(a.root)
    with a.output.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
