"""Exercise the inclusion, holdout and incomplete-panel rules on the locked panel."""
import argparse
import json
from pathlib import Path
from build import pin,save
from s1_panel_selection import read,sample,needed


def check(partition,config,out):
    split=read(partition);cfg=read(config)
    checks={r['component'] for r in split['rows'] if r['split']=='check'}
    train={r['component'] for r in split['rows'] if r['split']=='selection'}
    needed(not checks&train,'source component leakage')
    outcomes=['miss','extra','mixed','correct_empty','correct_positive','protocol']
    fixture={r['reference_id']:dict(outcome=outcomes[i%len(outcomes)]) for i,r in enumerate(split['rows'])}
    fold=split['rows'][0]['fold'];cfg=dict(cfg,training_folds=[fold])
    a=sample(cfg,split,fixture)
    ids=[r['reference_id'] for r in a['selected']]
    needed(a['panel_complete'] and ids and len(ids)==len(set(ids)) and
           all(r['component'] not in checks and r['fold']==fold and r['outcome']!='protocol'
               and 0<r['outcome_conditional_inclusion']<=1 for r in a['selected']) and
           len(a['check'])==split['counts']['check'] and
           sample(cfg,split,fixture)==a,'holdout,fold,probability or replay failure')
    fixture.pop(next(iter(fixture)))
    b=sample(cfg,split,fixture);needed(not b['selected'] and not b['panel_complete'],'partial panel selected')
    return save(out,dict(partition=pin(partition),configuration=pin(config),source_components={'selection':len(train),'check':len(checks),'overlap':0},
             checked=['fold filter','protocol isolation','inclusion','replay','incomplete panel','holdout isolation'],
             synthetic_outcomes=True,model_calls=0,formal_training_set=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('partition','config','out'):p.add_argument(k,type=Path)
    a=p.parse_args();print(json.dumps(check(a.partition,a.config,a.out)))
