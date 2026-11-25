"""Offline replay: configured TRAIN scope through the two sampler entry points."""
import argparse
from collections import Counter
from pathlib import Path
import copy
from base import checked, read, pin, save, digest, canonical, official_domains, PARENT
from content import jsonl
from stream_pool import Pool, sampler_module, iter_jsonl
from sampling import publish


def exercise(config, official_ref, output):
    output.mkdir(exist_ok=False,parents=True)
    full=official_domains(checked(official_ref))
    sampler=sampler_module(PARENT.parent/'unified-learning')
    # Isolate all fixture artifacts from model/production paths.
    roster=[]; records=[]; outcomes=[]; domains=[]
    online=save(output/'online.json',[]);offline=save(output/'offline.json',{})
    for key,domain in [('train_real','real'),('train_synth','synthetic')]:
        metadata=output/(key+'.jsonl')
        metadata.write_bytes(canonical(dict(recordings=[dict(file=f'{i}.wav') for i in range(8)]))+b'\n')
        meta=dict(pin(metadata),recording_refs=8)
        domains.append(dict(domain=key,metadata=meta))
        for i in range(8):
            rid=f'{key}:000001:{i:04d}'
            source=dict(recording_ref_id=rid,split='TRAIN',domain=domain,source='fixture',source_id=key)
            status='success' if i%2 else 'error'
            roster.append(source)
            outcomes.append(dict(source,status=status,gold_in_actual_pre_P_dynamic_retrieval=True))
            records.append(dict(source,sample_id=rid,system_version='unit',outcome=status,rollout_scope='system',
                online_gold_injected=False,online=dict(trajectory_export=online),offline_supervision=dict(export=offline,labels={})))
    ds=dict(schema='automatic-dataset-manifest-v1',split='TRAIN',population=len(roster),synthetic=False,
            domains=domains,inputs=[d['metadata'] for d in domains])
    raw=save(output/'dataset.json',ds)
    streams={k:jsonl(output/(k+'.jsonl'),v) for k,v in [('population',roster),('outcomes',outcomes),('records',records)]}
    manifest=dict(schema='sealed-training-pool-v1',sealed=True,version='unit',system_version='unit',scientific_identity='unit',
        lineage=dict(raw_dataset=raw,**dict.fromkeys(('system_release','knowledge','retrieval','prompt','rails','scorer'),'unit')),
        compatible_runtime_systems={'unit':{'identity':'unit'}},official_train_scope='entire_official_TRAIN_real_and_synthetic',
        population_complete=True,official_train_count=len(roster),unknown_count=0,synthetic_fixture=False,
        population_sha256=digest(canonical(roster)),cross_project_exclusions=False,streams=streams)
    base=dict(config,system_version='unit',pool_version='unit',sample_count=8,domain_mix=dict(real=.5,synthetic=.5))
    checks=[]
    for i,ratio in enumerate([.25,.75]):
        cfg=dict(base,error_success=dict(error=ratio,success=1-ratio))
        direct,stats=sampler.sample(Pool(manifest),cfg)
        result=checked(publish(manifest,cfg,output/f'draw-{i}',[output]))
        exported=list(iter_jsonl(result['files']['draws']))
        assert Counter(x['record']['outcome'] for x in exported)==stats['outcomes']=={'error':int(8*ratio),'success':int(8*(1-ratio))}
        assert exported==direct and result['consumer_loaded']==8 and result['model_training_ready'] is False
        checks.append(stats['outcomes'])
    bad=[]
    def rejects(name,action,expected):
        try:action()
        except ValueError as e:
            assert expected in str(e),str(e)
            bad.append(name)
        else:raise AssertionError(name)
    cfg=dict(base,error_success={'error':.8,'success':.5})
    rejects('mix',lambda:publish(manifest,cfg,output/'invalid',[output]),'sum to one')
    for name,alter in [('wrong population',dict(ds,population=15)),
                       ('conflicting metadata pin',dict(ds,inputs=[dict(ds['inputs'][0],sha256='0'*64),ds['inputs'][1]])),
                       ('bounded manifest',dict(ds,bounded_selection=True))]:
        alt=dict(manifest,lineage=dict(manifest['lineage'],raw_dataset=save(output/(name+'.json'),alter)))
        rejects(name,lambda:sampler.sample(Pool(alt),cfg),'OFFICIAL_FULL_TRAIN_REQUIRED')
    rejects('incomplete pool',lambda:publish(dict(manifest,population_complete=False),cfg,output/'invalid',[output]),'incomplete pool')
    return save(output/'RESULT.json',dict(official_domains=full,official_population=sum(full.values()),fixture_only=True,
                two_config_draw_outcomes=checks,consumer_loaded_each=8,negative_cases=bad,model_calls=0))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('union_config',type=Path);p.add_argument('sample_config',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    print(exercise(read(args.sample_config)['pool']['sampling'],read(args.union_config)['dataset'],args.output))
