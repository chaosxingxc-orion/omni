"""Build new bounded TRAIN requests; reuse prompt IDs only for byte-identical original packets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def build(spec):
    cfg=json.loads(spec.read_bytes())
    parents=Path(cfg['release'])/'tree/parents'
    sys.path.insert(0,str(parents/'fresh-cohort-successor'))
    from cohort_io import read,pin,digest,load
    from cohort_capture import sample_packet
    from material import selected
    protocol=load(parents/'rollout-capture-contract/capture.py','original_capture_protocol')
    root=Path(cfg['prepared'])
    pending=read(pin(root/'PENDING-CAPTURE.json'))
    plan=read(pending['selected_cohort_ref']);config=read(pending['input_ref'])
    clock=read(pin(root/'RESERVED-CLOCK.json'));consumer=read(pin(root/'CONSUMER.json'))
    if (plan['prepare_mode']!='bounded_train_probe' or plan.get('diagnostic_evidence_only')
            or plan['synthetic_fixture'] or consumer['clock_state_ref']!=pin(root/'RESERVED-CLOCK.json')
            or set(clock['groups'])!=set(plan['group_ids']) or any(g['consumed'] for g in clock['groups'].values())):
        raise ValueError('expected original reserved train cohort')
    profile=read(config['capture']['profile_ref']);protocol.validate_profile(profile)
    source=read(cfg['prompt_cache'])
    by_id={r['packet']['sample_id']:r for r in source['rows']}
    old_ids={r['packet']['request_id'] for r in source['rows']}
    g=read(config['learning_config_ref'])['kernel']['G'];rows=[]
    for d in plan['draws']:
        record=d['record']
        packet,=[p for p in selected(d)[0] if p['stage']=='P']
        cached=by_id[record['sample_id']]
        if (packet!=cached['packet']['original_packet'] or record['split']!='TRAIN'
                or record['online_gold_injected'] or g!=4):raise ValueError('prompt identity or selection drift')
        for i in range(g):
            eid=d['group_id']+':e'+str(i)
            seed=int(digest(dict(seed=config['capture']['seed'],draw=d['draw_index'],replicate=i,
                                 node='P',branch=packet.get('branch')))[:15],16)
            identity=dict(group_id=d['group_id'],episode_id=eid,node='P',fold=None,attempt_id=eid,
                replicate=i,seed=seed,parents=[],run_id=plan['version'],boot_id=cfg['boot_id'],
                query_id=record['sample_id'],split='TRAIN')
            req=sample_packet(packet,profile,identity,protocol)
            if req['request_id'] in old_ids or req['wire']['messages']!=cached['packet']['wire']['messages']:
                raise ValueError('sampling identity collision or changed prompt')
            old_ids.add(req['request_id'])
            rows.append(dict(group=d['draw_index'],replicate=i,packet=req,prompt_token_ids=cached['prompt_token_ids'],
                             source_sha256=hashlib.sha256(json.dumps(req,sort_keys=True,separators=(',',':')).encode()).hexdigest()))
    result=dict(schema='bounded-train-natural-requests-v1',rows=rows,group_size=g,groups=list(range(len(plan['draws']))),
                selected_sha256=pending['selected_cohort_ref']['sha256'],clock_sha256=consumer['clock_state_ref']['sha256'])
    with Path(cfg['output']).open('x') as f:json.dump(result,f)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);build(p.parse_args().config)
