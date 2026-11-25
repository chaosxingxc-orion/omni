"""Produce reproducible TRAIN recording/fold panels using versioned recording pages."""
import argparse
import collections
import json
import random
from pathlib import Path

from build import save
from prepare_s1_recordings import read, checked, needed
from source.config_profile import stage_config, bind_environment
import os


def select(cfg):
    needed(cfg['schema']=='s1-fold-panel-v1' and type(cfg['seed']) is int,'panel schema')
    pages=read(checked(cfg['pages']));folds=pages['folds']
    strata=cfg['per_fold_domain'];active=cfg['folds'] if cfg['folds']!='all' else list(folds)
    needed(active and len(active)==len(set(active)) and set(active)<=folds.keys(),'fold selection')
    needed(set(strata)=={'present','absent'} and all(type(n) is int and n>=0 for n in strata.values()) and sum(strata.values())>0,'fold quota')
    admitted={}
    for part in cfg['admission']:
        seal=read(checked(part['receipt']))
        needed(seal['knowledge_manifest_sha256']==cfg['kb_sha256'] and
               seal['recall_results_sha256']==cfg['recall_results_sha256'],'different intent KB/gate')
        ref=dict(path=part['roster'],sha256=seal['queue_sha256'])
        with checked(ref).open() as f:
            for n,line in enumerate(f,1):
                v=json.loads(line);rid=v['reference_id']
                needed(rid not in admitted and v['rank']>=1 and v['labels'][v['rank']-1]==v['gold'],'overlapping or invalid gate')
                admitted[rid]=(v['audio_sha256'],v['component'],v['recording'],str(v['source_id']))
            needed(n==seal['rows'],'gate count')
    needed(len(admitted)==cfg['expected_admissions'],'incomplete full-population gate')
    counts=collections.Counter();population=collections.Counter();reservoir={};used=set()
    rng=random.Random(cfg['seed']);outside=collections.Counter();universe=set().union(*map(set,folds.values()))
    # A random fold assignment (before reading the label) makes each recording
    # eligible in exactly one cell. Reservoir sampling is uniform within it.
    for page_no,ref in enumerate(pages['pages']):
        for i,row in enumerate(read(checked(ref))):
            rid=row['reference_id'];domain=row['domain']
            if rid not in admitted:continue
            needed(admitted[rid]==(row['audio_sha256'],row['source_component'],row['recording_file'],row['source_id'].split(':',1)[1])
                   and rid not in used,'recording/gate identity mismatch')
            used.add(rid);gold=set(row['offline']['present_types'])
            outside.update(gold-universe)  # scorer supervises only the supplied fold's types
            def cell(f):
                return domain+'/'+f+'/'+('present' if gold.intersection(folds[f]) else 'absent')
            for f in active:population[cell(f)]+=1
            fold=rng.choice(active);key=cell(fold);counts[key]+=1
            cap=strata[key.split('/')[-1]];bucket=reservoir.setdefault(key,[])
            j=rng.randrange(counts[key]);entry=dict(reference_id=rid,page=page_no,row=i,fold=fold,stratum=key)
            if len(bucket)<cap:bucket.append(entry)
            elif j<cap:bucket[j]=entry
    draws=[]
    for key,bucket in sorted(reservoir.items()):
        for e in bucket:draws.append(dict(e,conditional_inclusion=len(bucket)/counts[key],fold_assignment_probability=1/len(active)))
    rng.shuffle(draws)
    needed(draws and len({d['reference_id'] for d in draws})==len(draws),'duplicate/empty fold panel')
    return dict(schema=cfg['schema'],purpose='paired_baseline_screen',sampling=cfg,eligible_recordings=len(used),
                outside_fold_type_counts=dict(outside),population_by_stratum=dict(population),assigned_by_stratum=dict(counts),
                selected_by_stratum={k:len(v) for k,v in reservoir.items()},draws=draws,
                selection_uses_model_outcomes=False,formal_training_set=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');p.add_argument('--profile',required=True)
    a=p.parse_args();profile=read(a.profile)
    if profile['schema']=='omni-local-profile-v2':profile,_=bind_environment(profile,os.environ)
    cfg=stage_config(read(a.config),profile,'s1-fold-panel-v1');print(json.dumps(save(cfg['output'],select(cfg))))
