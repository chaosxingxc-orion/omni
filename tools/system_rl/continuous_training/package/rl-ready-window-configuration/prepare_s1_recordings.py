"""Prepare S1 recording pages from sealed TRAIN rankings; no model or retrieval."""
import argparse
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

from build import save
from source.config_profile import stage_config, bind_environment, digest


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checked(ref):
    p = Path(ref['path'])
    if sha(p) != ref['sha256']:
        raise ValueError('changed source: '+str(p))
    return p


def needed(ok, message):
    if not ok:
        raise ValueError(message)


def labels(dataset):
    needed(dataset['split']=='TRAIN' and dataset['synthetic'] is False, 'real TRAIN manifest required')
    by_id = {}
    for d in dataset['domains']:
        domain = d['domain']
        needed(domain in ('train_real','train_synth'), 'non-TRAIN domain')
        with checked(d['metadata']).open() as f:
            for line, raw in enumerate(f,1):
                v=json.loads(raw)
                types=sorted({e['type'] for e in v['entities']}) if isinstance(v.get('entities'),list) and isinstance(v.get('tokens'),list) else None
                for i,rec in enumerate(v['recordings']):
                    rid=f'{domain}:{line:06d}:{i:04d}'
                    needed(rid not in by_id and rec['file']==Path(rec['file']).name,'duplicate or invalid recording')
                    by_id[rid]=(str(v['slurp_id']),rec['file'],types,digest(v),line)
    needed(len(by_id)==dataset['population'],'incomplete TRAIN metadata')
    return by_id


def build(cfg,out,space):
    needed(cfg['schema']=='s1-recording-pages-v1' and type(cfg['recordings_per_macro']) is int
           and cfg['recordings_per_macro']>0,'invalid page config')
    recall=read(checked(cfg['recall'])); manifest=read(checked(cfg['dataset']))
    index=read(checked(cfg['shards'])); seal=read(checked(cfg['knowledge_release']))
    assets={Path(ref['path']).name:ref for ref in seal['artifacts']}
    knowledge,cache,runtime=(read(checked(assets[n])) for n in ('knowledge.private.json','retrieval.private.json','s1.runtime.json'))
    needed(cache['knowledge_id']==knowledge['knowledge_id'] and runtime['s1_consumption']['cache_id']==cache['cache_id'],'different S1 policy/bank')
    if 'max_ranked_neighbors' in cfg:
        depth=cfg['max_ranked_neighbors']
    else:
        # Compatibility with the original stage profile. New experiments supply depth explicitly.
        depth=read(Path(__file__).with_name('S1-RECORDINGS.template.json'))['max_ranked_neighbors']
    needed(type(depth) is int and depth>0,'invalid neighbor depth')
    folds={v['fold_id']:v['types'] for v in knowledge['folds']}
    bank={v['source_id']:v['source_component'] for v in cache['bank']}
    needed(len(bank)==len(cache['bank']) and folds,'duplicate bank or empty folds')
    offline=labels(manifest)
    features={}
    with checked(recall['feature_index']).open() as f:
        for line in f:
            e=json.loads(line); rid=e['reference_id']
            needed(rid not in features,'duplicate feature')
            features[rid]=e['audio_sha256']
    ranks={}
    for name,receipt in recall['worker_receipts'].items():
        f=Path(cfg['recall']['path']).parent/name/'ranked.jsonl'
        with checked(dict(path=str(f),sha256=receipt['sha256'])).open() as stream:
            for n,line in enumerate(stream,1):
                v=json.loads(line);rid=v['reference_id']
                needed(rid not in ranks and v['audio_sha256']==features[rid],'rank/recording identity drift')
                ranks[rid]=v['s1']
            needed(n==receipt['rows'],'rank count drift')
    needed(features.keys()==ranks.keys()==offline.keys() and len(ranks)==recall['population']==index['reference_rows'],'population differs')
    pages=[];rows=[];domains=collections.Counter();decisions=collections.Counter();eligible=collections.Counter()
    def flush():
        if rows:
            space()
            pages.append(save(out/'pages'/('%06d.json'%len(pages)),list(rows)));rows.clear()
    for shard in index['shards']:
        if shard['kind']!='references':continue
        with checked(shard['artifact']).open() as stream:
            for line in stream:
                v=json.loads(line);rid=v['reference_id'];domain=v['domain'];uid,file,types,record_sha,metadata_line=offline.pop(rid)
                needed(domain==shard['domain'] and v['source_id']==domain+':'+uid and
                       v['recording_file']==file and v['source_record_sha256']==record_sha,'label/source mismatch')
                ranked=ranks.pop(rid);keys=[(-score,str(sid)) for sid,score in ranked]
                needed(keys==sorted(keys) and len(ranked)<=depth and len({i for i,_ in ranked})==len(ranked)
                       and all(math.isfinite(s) and i in bank and bank[i]!=v['component_id'] for i,s in ranked),'illegal S1 neighbor')
                domains[domain]+=1;decisions[v['source_decision']]+=1
                if v['source_decision']!='allowed' or types is None:continue
                eligible[domain]+=1
                rows.append(dict(reference_id=rid,domain=domain,source_id=v['source_id'],source_component=v['component_id'],
                    source_record_sha256=record_sha,recording_file=file,audio_sha256=features[rid],neighbors=ranked,
                    offline=dict(metadata_line=metadata_line,present_types=types)))
                if len(rows)==cfg['recordings_per_macro']:flush()
    flush()
    needed(not offline and not ranks and pages,'missing reference rows')
    return save(out/'PAGES.json',dict(schema=cfg['schema'],configuration=cfg,folds=folds,pages=pages,
        population=sum(domains.values()),domains=dict(domains),source_decisions=dict(decisions),
        eligible=dict(eligible),recordings=sum(eligible.values()),model_calls=0,scope='static_inputs_only'))


def prepare(template,site):
    from multimodal_jev.local_runtime_storage import check_paths
    profile=read(site)
    if profile['schema']=='omni-local-profile-v2':profile,_=bind_environment(profile,os.environ)
    cfg=stage_config(read(template),profile,'s1-recording-pages-v1');out=Path(cfg['output'])
    def space():
        checks=check_paths({'s1_pages':out},min_free_fraction=.10)
        needed(all(v['free_bytes']>=20*1024**3+cfg['storage_budget_bytes'] for v in checks),'storage reserve')
    space();out.mkdir(mode=0o700,parents=True,exist_ok=True)
    with (out/'prepare.guard').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(out/'BINDING.json',dict(template=read(template),deployment=profile,resolved=cfg))
        return build(cfg,out,space)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config');p.add_argument('--profile',required=True)
    a=p.parse_args(); print(json.dumps(prepare(a.config,a.profile)))
