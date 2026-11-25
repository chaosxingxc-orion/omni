"""Replay native S1 responses against the official metadata without reissuing requests."""
import argparse
from collections import Counter
import json
from pathlib import Path

from s1_panel_selection import classify,packet_parser,needed,read,checked,pin,save


def check(seal_path,dataset_path,out):
    seal=read(seal_path);v=read(checked(seal['version']));dataset=read(dataset_path)
    needed(seal['immutable'] and v['mode']=='live' and not v['synthetic'] and dataset['split']=='TRAIN','real TRAIN required')
    domains={d['domain']:[json.loads(line) for line in checked(d['metadata']).open()] for d in dataset['domains']}
    parse=packet_parser();counts=Counter();negatives=0
    for row in read(checked(v['receipts'])):
        if row['stage']!='S1' or row['status']!='known':continue
        p=row['packet'];reply=row['receipt'];b=p['source_binding'];domain,line,i=b['reference_id'].split(':')
        raw=domains[domain][int(line)-1];rec=Path(raw['recordings'][int(i)]['file'])
        needed(b['source_id']==domain+':'+str(raw['slurp_id']) and domain==b['domain']
               and p['sample_id'].endswith(':'+rec.stem),'official recording differs')
        allowed=p['allowed_types']; gold={e['type'] for e in raw['entities']}
        labels=dict(annotated_types=allowed,present_types=[t for t in allowed if t in gold])
        kind,_,_=classify(p,labels,reply,parse);counts[kind]+=1
        try:classify(dict(p,branch='changed-fold'),labels,reply,parse)
        except ValueError:negatives+=1
        else:raise AssertionError('wrong fold accepted')
    needed(sum(counts.values())==negatives>0,'missing verified replies')
    return save(out,dict(seal=pin(seal_path),dataset=pin(dataset_path),historical_replay=dict(counts),
                    changed_fold_rejected=negatives,model_calls=0,formal_training_set=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('seal','dataset','out'):p.add_argument(k,type=Path)
    a=p.parse_args();print(json.dumps(check(a.seal,a.dataset,a.out)))
