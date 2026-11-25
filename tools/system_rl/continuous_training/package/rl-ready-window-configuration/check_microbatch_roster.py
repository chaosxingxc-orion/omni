"""Compare pre-score row planning with an original closed eight-rank actor step."""
import argparse,json,sys,hashlib,shlex,subprocess
from pathlib import Path

def checked(ref):
    data=Path(ref['path']).read_bytes()
    if len(data)!=ref['bytes'] or hashlib.sha256(data).hexdigest()!=ref['sha256']:
        raise ValueError('roster input changed')
    return json.loads(data)

def run(work):
    from microbatch_roster import roster
    from report_training_updates import closed_metadata
    read=lambda p:json.loads(p.read_bytes())
    returned,member=closed_metadata(work/'returned')
    result=member(returned['result_ref']);req=checked(read(work/'SUBMIT-INTENT.json')['training_ref'])
    inv=checked(result['generation_inventory_ref']);refs=inv['files']
    if not inv['closed'] or req['generation']['generation_inventory_ref']!=result['generation_inventory_ref']:
        raise ValueError('not one closed rollout')
    plan=checked(refs['SELECTED.json']);cfg=checked(refs['INPUT.json']);clock=checked(refs['RESERVED-CLOCK.json'])
    learning=checked(cfg['learning_config_ref']);mask=clock['groups'][plan['group_ids'][0]]['snapshot']['trainable_mask']
    if any(clock['groups'][k]['snapshot']['trainable_mask']!=mask for k in plan['group_ids']):
        raise ValueError('changed mask within generation')
    ds=plan['draws']; groups=(checked(refs['groups/'+str(d['draw_index'])+'/GENERATED.json']) for d in ds)
    geom=req['actor_batch']; pred=roster(ds,groups,world=len(req['devices']),micro=geom['microbatch_per_card'],
        accumulation=geom['gradient_accumulation_steps'],trainable_mask=mask,group_size=learning['kernel']['G'],policy=plan['policy'])
    # Actual event ownership in LOSS and the mappings emitted by the accepted consumer.
    probe="""import json,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1]);rd=lambda path:json.loads(path.read_bytes())
assert hashlib.sha256((p/'CLOSED.json').read_bytes()).hexdigest()==sys.argv[2]
result=rd(p/'RESULT.json');c=Path(result['trajectory_ref']['path']).parent
ms=rd(c/'CONSUMER-REPORT.json')['all_call_mapping']
layout=rd(p/'rank-0/LOSS.json')['result']['layout']
rows=[(m['group_id'],m['episode_id'],m['request_identity']['request_id']) for m in ms if m['actor_row'] is not None]
s=[]
for i,sched in enumerate(layout['schedules']):
 actual=rd(p/f'rank-{i}'/'LOSS.json')['result']['events']
 assert [(e['row'],e['owned']) for e in actual]==[(e['row'],e['owned']) for e in sched]
 s.append([(e['row'],e['owned']) for e in actual])
print(json.dumps(dict(rows=rows,schedules=s,denominator=layout['global_episode_denominator'],loads=layout['loads'])))"""
    c=read(work/'returned/CLOSED-COMMIT.json')['closure']
    cmd=['python3','-B','-c',probe,str(Path(c['closed_path']).parent),c['closed_sha256']]
    actual=json.loads(subprocess.check_output(['ssh','-oBatchMode=yes',req['host'],shlex.join(cmd)],timeout=45))
    if (actual['rows']!=[list(r[:3]) for r in pred['rows']] or actual['denominator']!=pred['global_episode_denominator']
            or actual['loads']!=pred['loads'] or any(actual['schedules'][r]!=[list(pair) for wave in pred['waves'] for pair in wave[r]]
                for r in range(len(req['devices'])))):raise ValueError('predicted microbatches differ from executed events')
    groups_per_wave=[{pred['rows'][i][0] for rank in w for i,owned in rank if owned} for w in pred['waves']]
    return dict(step=member(result['checkpoint_ref'])['step'],rows=len(pred['rows']),group_counts=list(map(len,groups_per_wave)),
                shared_groups=len(set.intersection(*groups_per_wave)) if groups_per_wave else 0,exact=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--update',required=True,type=Path,action='append');a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
    print(json.dumps([run(w) for w in a.update],indent=2))
