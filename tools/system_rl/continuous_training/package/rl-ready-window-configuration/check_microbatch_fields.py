"""Repack the rows in every consumed microbatch from closed production groups on CPU."""
import argparse,hashlib,json,time
from pathlib import Path
from microbatch_fields import micro_fields,overlap_microbatches
from dependencies import require

def checked(ref):
    raw=Path(ref['path']).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==ref['sha256'] and len(raw)==ref['bytes'],'changed sealed member')
    return json.loads(raw)

def replay(root,source,pin):
    import torch
    from check_accepted_actor_packet import equal
    read=lambda p:json.loads(p.read_bytes())
    cfg=read(root/'LOCAL-CONSUMER.json');report=read(root/'CONSUMER-REPORT.json')
    files=checked(read(root/'COMPLETE.json')['files_ref'])
    index=checked(files['refs']['online']); groups={}
    for ref in index['groups']:
        group=checked(ref); table=group.get('retained_static_v1',{})
        for g in group['online']['behavior_bundle']['groups']:
            for e in g['episodes']:
                for c in e['calls']:
                    f=c['training_fields']
                    f.update({k:table[key] for k,key in f.pop('retained_static_refs',{}).items()})
                    groups[g['group_id'],e['episode_id'],c['call_id']]=c
    advantages={(g['group_id'],e['episode_id']):e['advantage'] for g in report['group_results'] for e in g['episodes']}
    mappings=[m for m in report['all_call_mapping'] if m['actor_row'] is not None]
    rows=[(groups[m['group_id'],m['episode_id'],m['call_id']],m,advantages[m['group_id'],m['episode_id']]) for m in mappings]
    geom=tuple(max(len(c['training_fields'][k]) for c,m,a in rows) for k in ('prompt_token_ids','action_token_ids'))
    rec=read(root/'actor-packet/RECEIPT.json');file=root/'actor-packet/fields.pt'
    with file.open('rb') as f: require(hashlib.file_digest(f,'sha256').hexdigest()==rec['payload']['sha256'],'different original actor packet')
    packet=torch.load(file,mmap=True,weights_only=True,map_location='cpu')
    require(rec['dataset']==packet['dataset'] and rec['report']['sha256']==hashlib.sha256((root/'CONSUMER-REPORT.json').read_bytes()).hexdigest(),'different packet report')
    checked_rows=0;build_seconds=0.;micros=0
    for path in sorted(root.parent.glob('rank-*/LOSS.json')):
        result=read(path)['result'];events=result['events']
        # Width derived from actual receipts for the full cohorts used in this replay.
        waves=[]
        for n in result['metrics']['dp8/owned_event']:
            es,events=events[:n],events[n:];require(n>0 and len(es)==n and all(e['owned'] for e in es),'expected full micro')
            waves.append([e['row'] for e in es])
        def prepare(order):
            start=time.monotonic()
            tensors,non=micro_fields([rows[i] for i in order],geom,source=source,consumer_sha256=pin,
                snapshot=report['snapshot'],pad=cfg['pad_token_id'],context_limit=cfg['context_limit'])
            return order,tensors,non,time.monotonic()-start
        for expected,(order,tensors,non,duration) in zip(waves,overlap_microbatches(waves,prepare),strict=True):
            require(order==expected,'wave arrived out of order');build_seconds+=duration
            equal(tensors,{k:v[order] for k,v in packet['tensors'].items()})
            equal({k:list(v) for k,v in non.items()},{k:[v[i] for i in order] for k,v in packet['non_tensors'].items()})
            checked_rows+=len(order);micros+=1
        require(not events,'unconsumed events')
    require(checked_rows==report['actor_rows'],'missing rows')
    return dict(rows=checked_rows,microbatches=micros,all_fields_equal=True,pack_seconds=build_seconds)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','source'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--sha256',required=True);a=p.parse_args()
    print(json.dumps(replay(a.root,a.source,a.sha256)))
