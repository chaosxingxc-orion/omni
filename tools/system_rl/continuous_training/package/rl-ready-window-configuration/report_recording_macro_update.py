"""Read one committed production update, its actor timings and the exact rollout roster."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from report_training_updates import closed_metadata


def read(path):
    return json.loads(Path(path).read_bytes())


def checked(ref):
    raw=Path(ref['path']).read_bytes()
    if len(raw)!=ref['bytes'] or hashlib.sha256(raw).hexdigest()!=ref['sha256']:
        raise ValueError('recorded input changed')
    return json.loads(raw)


def check_owned_waves(rows, micro, accumulation, episodes):
    # Tail participation is required for collectives, but carries zero weight.
    if (not rows or not rows[0] or len(rows[0])>accumulation or episodes<=0
            or any(len(rank)!=len(rows[0]) or any(type(n) is not int or not 0<=n<=micro for n in rank)
                   or any(b and a!=micro for a,b in zip(rank,rank[1:])) for rank in rows)
            or not any(rank[-1] for rank in rows) or sum(map(sum,rows))!=episodes):
        raise ValueError('invalid per-rank ownership or tail padding')


def report(run, number, ledger):
    update=run/f'step-{number:06d}'/'update'
    returned=update/'returned';actual_update=update
    if returned.is_symlink():
        note=checked(read(update/'IDLE-RECOVERY.json')['binding']);actual_update=returned.resolve(strict=True).parent
        if (note['directory']!=str(update) or actual_update!=update/'idle-successor'/'update'
                or Path(note['original']['path'])!=update/'SUBMIT-INTENT.json'
                or checked(note['original'])['command']['generation_inventory_ref']!=note['generation']
                or read(actual_update/'TRAIN-REQUEST.json')['generation']['generation_inventory_ref']!=note['generation']
                or read(update/'IDLE-RECOVERY.json')['result']!=read(returned/'RETURN.json')):
            raise ValueError('different recovered update')
    ret,member=closed_metadata(returned)
    result=member(ret['result_ref']);cp=member(ret['checkpoint_ref']);selected=member(result['selected_cohort_ref'])
    obj=result['result']['objective_observation']; cfg=read(actual_update/'TRAIN-REQUEST.json')
    inv_ref=cfg['generation']['generation_inventory_ref'];inv=checked(inv_ref);cohort=Path(inv['cohort_root'])
    receipts=read(cohort.parent/'DRAINED.json');actual=checked(inv['files']['INPUT.json'])
    if Path(inv['files']['INPUT.json']['path'])!=cohort/'INPUT.json':raise ValueError('wrong rollout input')
    batch=cfg['actor_batch'];ranks=[member(ref) for ref in result['rank_refs']]
    rows=[r['metrics']['dp8/owned_event'] for r in ranks];episodes=sum(map(sum,rows))
    ids=[draw['record']['recording_ref_id'] for draw in selected['draws']]
    if (not result['formal_rl_enabled'] or ret['optimizer_steps']!=1 or result['optimizer_steps']!=1
            or result['checkpoint_ref']!=ret['checkpoint_ref'] or not cp['recoverable']
            or not (math.isfinite(obj['components']['total']) and math.isfinite(obj['gradient_norm']) and obj['gradient_norm']>0)
            or cp['step']!=obj['successful_step_after'] or obj['successful_step_before']!=cp['step']-1
            or cp['clock']!=result['clock'] or cp['world_size']!=len(cfg['devices'])
            or sorted(r['rank'] for r in ranks)!=cfg['devices'] or not all(r['step']==cp['step'] for r in ranks)
            or any(r['objective']['components']!=obj['components'] or not r['recoverable_checkpoint'] for r in ranks)
            or result['group_ids']!=selected['group_ids'] or cp['training_lineage']['group_ids']!=selected['group_ids']
            or len(set(ids))!=len(ids) or obj['groups']!=len(ids) or episodes!=obj['global_episode_denominator']
            or result['generation_inventory_ref']!=inv_ref or not inv['closed']
            or not receipts['terminal_verified'] or receipts['unknown_requests']!=0
            or receipts['generation_closed_ref']!=inv['generation_closed_ref'] or receipts['usage']['unknown']!=0
            or ledger['sequence']!=read(actual_update/'RESIDENT-CALL.json')['sequence']
            or ledger['run']!=cfg['continuous_actor']['run_id'] or ledger['generation_sha256']!=inv_ref['sha256']
            or ledger['latest']!='command_closed' or ledger['ranks']!=cp['world_size']
            or ledger['layout']!=actual['capture']['training_input_layout'] or ledger['layout']!='group_refs_v1'
            or ledger['indexed_groups']!=len(ids) or ledger['published_episodes']!=episodes):
        raise ValueError('actor, optimizer, producer and resident receipt do not describe one complete indexed update')
    check_owned_waves(rows,batch['microbatch_per_card'],batch['gradient_accumulation_steps'],episodes)
    sent={};returned={}
    for ext,table in [('dispatch',sent),('response',returned)]:
        for path in cohort.glob('groups/*/episode-*/native-journal/*.'+ext+'.json'):
            journal=read(path);key=journal['request_id']
            if key in table:raise ValueError('duplicate rollout request')
            table[key]=(journal['request_sha256'],journal['intent_sha256'],journal['identity'])
    if len(sent)!=episodes or sent!=returned:raise ValueError('unreconciled rollout journals')
    macro=checked(actual['capture']['recording_macros_ref'])
    window=checked(actual['sampled_window_ref']);width=macro['width'];start,end=window['start'],window['end']
    if (window['dataset_ref']!=macro['dataset_ref'] or not 0<=start<end<=macro['count']
            or [d['dataset_draw_index'] for d in selected['draws']]!=list(range(start,end))):
        raise ValueError('optimizer window does not belong to static dataset')
    pages=list(range(start//width,(end-1)//width+1));roster=[]
    for i in pages:
        page=checked(macro['pages'][i]);offset=i*width
        if page['start']!=offset or len(page['rows'])!=min(width,macro['count']-offset):
            raise ValueError('incorrect static page')
        roster.extend(r['id'] for r in page['rows'][max(0,start-offset):end-offset])
    if roster!=ids:raise ValueError('optimizer recordings differ from prepared page order')
    return dict(step=cp['step'],checkpoint_sha256=ret['checkpoint_ref']['sha256'],recordings_per_macro=macro['width'],
                recording_window=[start,end],page_indices=pages,dataset_count=macro['count'],
                recordings=len(ids),trajectories=episodes,unknown=0,loss=obj['components']['total'],
                owned_rows_by_rank=rows,gradient_norm=obj['gradient_norm'],online_bytes=ledger['online_bytes'],
                consumer_publication_seconds=ledger['intervals_s']['all_groups_to_consumer_published'],
                actor_seconds=ledger['intervals_s'],generation_sha256=inv_ref['sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--actor-ledger',type=Path,required=True)
    a=p.parse_args();print(json.dumps(report(a.run,a.step,read(a.actor_ledger)),indent=2))
