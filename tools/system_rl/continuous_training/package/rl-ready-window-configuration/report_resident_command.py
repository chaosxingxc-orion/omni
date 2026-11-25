"""Read-only timing ledger from one real resident actor command on its own clock."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def observe(hot, sequence):
    exchange=hot/'resident'
    owner=read(exchange/'OWNER.json')
    root=exchange/'results'/f'{sequence:06d}'
    intent=read(root/'COMMAND-INTENT.json')
    cmd=intent['command']
    if (cmd['run_id']!=owner['run_id'] or cmd['sequence']!=sequence or
            len(owner['objects'])!=8 or len({row[0] for row in owner['objects']})!=8):
        raise ValueError('unexpected actor command or rank owners')
    stamps=[]
    def add(name, paths):
        if paths and all(p.is_file() for p in paths):
            stamps.append((name,max(p.stat().st_mtime_ns for p in paths)))
    def rank(base, name):
        return [base/f'rank-{i}'/(name+'.json') for i in range(8)]
    if not sequence:
        add('collective_ready',rank(hot,'COLLECTIVE-READY'))
        add('load_intent',rank(hot,'LOAD-INTENT'))
        add('worker_built',rank(hot,'WORKER-BUILT'))
        add('resident_ready',[exchange/'READY.json'])
    add('command',[root/'COMMAND-INTENT.json'])
    cfg=read(hot.parent/'TRAIN.local.json')
    folder=root/cfg['generation']['runtime']['cohort_dir']
    batch=list(folder.glob('REFERENCE-BATCH-*.json'))
    if len(batch)>1:raise ValueError('ambiguous reference roster')
    add('reference_ready',batch)
    selected=folder/'SELECTED.json'
    if selected.is_file():
        groups=[folder/'groups'/str(row['draw_index'])/'GROUP.json' for row in read(selected)['draws']]
        add('all_groups',groups)
    add('consumer_published',[folder/'COMPLETE.json'])
    for event,marker in [('finalized','FINALIZE-RESULT'),('accepted','ACCEPTED-COHORT'),
                         ('update_intent','UPDATE-INTENT'),('backwards','BACKWARDS'),
                         ('loss','LOSS'),('rank_result','RESULT')]:
        add(event,rank(root,marker))
    add('command_closed',[root/'CLOSED.json'])
    timings={}
    for (a,x),(b,y) in zip(stamps,stamps[1:]):
        if y<x:raise ValueError('actor receipt times reversed: '+a+' -> '+b)
        timings[a+'_to_'+b]=(y-x)/1e9
    extra={}
    if (folder/'COMPLETE.json').is_file():
        complete=read(folder/'COMPLETE.json');files=read(folder/'FILES.json')
        if files.get('layout')=='group_refs_v1':
            import hashlib
            def local_member(path,ref):
                raw=path.read_bytes()
                if ref['path']!=str(path) or ref['bytes']!=len(raw) or hashlib.sha256(raw).hexdigest()!=ref['sha256']:
                    raise ValueError('consumer member differs: '+path.name)
                return json.loads(raw)
            files=local_member(folder/'FILES.json',complete['files_ref'])
            online=local_member(folder/'online.json',files['refs']['online'])
            if online['schema']!='group-online-index-v1' or not complete['training_ready']:
                raise ValueError('unaccepted indexed consumer')
            extra=dict(layout=files['layout'],online_bytes=files['refs']['online']['bytes'],
                       indexed_groups=len(online['groups']),published_episodes=complete['complete_episodes'])
    return dict(**extra,metric='single_actor_host_marker_intervals',run=cmd['run_id'],sequence=sequence,
                generation_sha256=cmd['generation_inventory_ref']['sha256'],ranks=len(owner['objects']),
                latest=stamps[-1][0],intervals_s=timings,stamps_ns=dict(stamps),
                reference=read(batch[0]) if batch else None)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--hot',type=Path,required=True);p.add_argument('--sequence',type=int,required=True)
    a=p.parse_args();print(json.dumps(observe(a.hot,a.sequence),indent=2))
