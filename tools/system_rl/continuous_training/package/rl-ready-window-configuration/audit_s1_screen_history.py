"""Count exact S1 panel overlaps in a declared closed TRAIN collection, without model calls."""
import argparse
import json
from collections import Counter
from pathlib import Path
from prepare_s1_recordings import checked, read, needed
from build import pin,save


def audit(inputs,seal_path,out):
    seal=read(seal_path)
    needed(seal['immutable'] and seal['marker_scope']=='collection_analysis','closed collection required')
    version=read(checked(seal['version']))
    # Individual exports are indexes; accept a matched row only after checking the closed aggregate.
    aggregate=version['online_trajectories'];checked(aggregate)
    panel=read(inputs); wanted={}
    for row in panel['rows']:
        packet=read(checked(row['packet']))
        wanted[packet['sample_id'],row['fold']]=packet['wire_sha256']
    c=Counter();overlaps=[]
    needed(not version['synthetic'] and version['unresolved_requests']==0,'unsealed execution history')
    for episode in read(aggregate['path']):
        for s in episode['stages']['S1'].get('output') or []:
            if not isinstance(s,dict) or s.get('stage')!='S1':continue
            c['historical_S1']+=1
            key=episode['sample_id'],s['branch']
            if key not in wanted:continue
            c['matched_recording_fold']+=1
            if s['wire_sha256']!=wanted[key]:continue
            c['matched_wire']+=1
            overlaps.append(dict(sample_id=key[0],fold=key[1],wire_sha256=s['wire_sha256'],
                                 response_receipt=s.get('receipt')))
    return save(out,dict(inputs=pin(inputs),seal=pin(seal_path),population=len(wanted),
        counts=dict(c),exact_wire_candidates=overlaps,accepted_responses=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inputs');p.add_argument('seal');p.add_argument('output')
    a=p.parse_args();print(json.dumps(audit(a.inputs,a.seal,a.output)))
