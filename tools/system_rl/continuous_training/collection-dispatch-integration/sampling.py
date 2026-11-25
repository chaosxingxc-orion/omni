"""Reuse the sampler's selection algorithm; validate preserved runtime identities."""
from pathlib import Path
import collections
from base import PARENT, require, save, canonical, digest, checked, official_domains
from stream_pool import Pool, Sequence, sampler_module
from consumer import Consumer
from content import jsonl


def check_compatible_pool(pool,c,base='.'):
    require(pool['schema']=='sealed-training-pool-v1' and pool['sealed'], 'UNSEALED_POOL')
    require(pool['official_train_scope']=='entire_official_TRAIN_real_and_synthetic' and pool['population_complete'] is True and pool['synthetic_fixture'] is False, 'incomplete pool; no ready dataset')
    require(pool['version']==c['pool_version'] and pool['scientific_identity']==c['system_version'],'POOL_SCIENCE_VERSION')
    require(pool['unknown_count']==0 and not pool['cross_project_exclusions'],'UNKNOWN_POOL')
    require(all(pool['lineage'].get(k) for k in ('raw_dataset','system_release','knowledge','retrieval','prompt','rails','scorer')),'INCOMPLETE_LINEAGE')
    # SQLite dedup/official roster and all stream hashes were verified by union.
    # On reload revalidate every stream with the same reader, without audio I/O.
    expected=official_domains(checked(pool['lineage']['raw_dataset'])); total=sum(expected.values())
    roster={r['recording_ref_id']:r for r in pool['population']}
    require(len(roster)==len(pool['population'])==pool['official_train_count']==total and all(r['split']=='TRAIN' for r in roster.values()) and collections.Counter(r['domain'] for r in roster.values())==expected, 'OFFICIAL_ROSTER')
    h=__import__('hashlib').sha256();h.update(b'[')
    for i,r in enumerate(pool['population']):
        if i:h.update(b',')
        h.update(canonical(r))
    h.update(b']');require(h.hexdigest()==pool['population_sha256'],'ROSTER_HASH')
    outcomes={};records=[];seen=set();represented=set();excluded=collections.Counter()
    for o in pool['outcomes']:
        rid=o['recording_ref_id'];require(rid not in outcomes and rid in roster,'DUPLICATE_OUTCOME')
        require(o['status'] in ('success','error','ineligible') and all(o.get(k)==v for k,v in roster[rid].items()),'UNKNOWN_OR_FOREIGN_OUTCOME')
        outcomes[rid]=o
    require(set(outcomes)==set(roster),'OUTCOME_ACCOUNTING')
    for r in pool['records']:
        rid=r['recording_ref_id'];sid=r['sample_id'];require(sid not in seen and rid not in represented,'DUPLICATE_RECORD');seen.add(sid);represented.add(rid)
        o=outcomes[rid]
        require(o['status'] in ('success','error') and o['gold_in_actual_pre_P_dynamic_retrieval'] and r['outcome']==o['status'],'RECORD_ADMISSION')
        require(r['system_version'] in pool['compatible_runtime_systems'] and pool['compatible_runtime_systems'][r['system_version']]['identity']==c['system_version'],'RECORD_SCIENCE')
        require(r['rollout_scope']==c['scope'] and r['online_gold_injected'] is False and all(r.get(k)==v for k,v in roster[rid].items()),'RECORD_SCOPE_SOURCE')
        records.append(r)
    require(represented=={rid for rid,o in outcomes.items() if o['status'] in ('success','error')},'ADMITTED_EXPORT_ACCOUNTING')
    return sorted(records,key=lambda r:r['sample_id']),dict(excluded),outcomes


def publish(manifest, config, root, readonly_roots):
    require(config.get('purpose')=='screen' and config.get('split')=='TRAIN','TRAIN_SCREEN_REQUIRED')
    pool=Pool(manifest)
    # Bind the original sample function to a compatibility validator. No record
    # system_version is normalized or rewritten, and complete official TRAIN remains required.
    sampler=sampler_module(PARENT.parent/'unified-learning')
    import types
    namespace=dict(sampler.__dict__,check_pool=check_compatible_pool)
    sample=types.FunctionType(sampler.sample.__code__,namespace)
    draws,stats=sample(pool,config,'.')
    loaded=[]
    for d in draws:
        r=d['record']; consumer=Consumer(r.get('content_root',Path(root)/'empty-cas'),readonly_roots)
        loaded.append(consumer.load(r,offline=True))
    root=Path(root)
    refs=dict(draws=jsonl(root/'draws.jsonl',draws),consumer=jsonl(root/'consumer.jsonl',loaded))
    result=dict(schema='compatible-screen-dataset-v1',version=digest(canonical(dict(config=config,pool=manifest['version'],refs=refs))),
        config=config,statistics=stats,files=refs,consumer_loaded=len(loaded),purpose='screen',training_ready=False,
        model_training_ready=False,reason='Historical screen trajectories do not contain fresh behavior/token/probability groups')
    return save(root/'COMPLETE.json',result)
