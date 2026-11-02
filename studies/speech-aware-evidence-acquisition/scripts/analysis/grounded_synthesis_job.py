"""Prepare/register/admit finite grounded shard artifacts through shared flight."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
import tfgrpo_learning_job as base
import tfgrpo_final_job as final
from speech_aware_evidence_acquisition.contracts import ExecutionPlan, assert_execution_environment
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash
from speech_aware_evidence_acquisition.rails.grounded_synthesis_runtime import build_registration, load_runtime_source, SOURCE_SCHEMA
from speech_aware_evidence_acquisition.rails.grounded_synthesis_source import load_source
from speech_aware_evidence_acquisition.rails.grounded_synthesis_plan import load_token_counter, verify_plan
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha

read,raw,pin,file_sha=base.read,base.raw,base.pin,base.file_sha


def prepare(repo,data,*,shard_plan,source_spec,source_roster,tokenizer_relpath,
            runtime_identity,out,job_name,run_id,service_session):
    assert_execution_environment()
    if not re.fullmatch('[a-z0-9][a-z0-9-]*',job_name):raise ValueError('safe job name required')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*',service_session):raise ValueError('safe service session required')
    if out.exists() or not out.resolve().is_relative_to((data/'work/codex').resolve()):
        raise ValueError('fresh owned job directory required')
    plan=read(shard_plan);spec=read(source_spec);identity=read(runtime_identity)
    custody=load_source(spec,data_root=data)
    bb=(data/spec['source_batch']['relpath']).read_bytes()
    ob=(data/spec['operations']['relpath']).read_bytes()
    count,tokenizer_id=load_token_counter(data/tokenizer_relpath)
    verify_plan(plan,bb,ob,count_tokens=count,tokenizer_identity=tokenizer_id)
    batch=json.loads(bb);model=custody['model_id']
    if base.deployment_identity(repo,model,identity)!=identity:raise ValueError('exact deployment identity required')
    ids=[batch['groups'][s['group_indices'][0]]['query_id'] for s in plan['shards']]
    old_roster=read(source_roster);items={r['sample_id']:r for r in old_roster['items']}
    if len(items)!=len(old_roster['items']) or not set(ids)<=items.keys():raise ValueError('anchors absent from source roster')
    roster={'schema':'saea-roster-receipt-v1','roster_id':run_id,'roster_role':'construction',
        'carrier_lock_key':'slurp','split_role':'discovery','count':len(ids),
        'items':[items[sid] for sid in ids],
        'split_identity_hash':split_identity_hash(prefixed('slurp',ids)),
        'selection':'First original TRAIN group per deterministic whole-group shard; no resampling.',
        'source_queries_must_not_be_reported_as_held_out':True}
    execution=ExecutionPlan(run_id=run_id,attempt_id='learning-v1',
        execution_profile='bounded-discovery-probe',carrier_lock_key='slurp',split_role='discovery',
        split_identity_hash=roster['split_identity_hash'],planned_model_calls=len(ids),
        planned_gpu_hours=8.0,planned_speech_audio_seconds=1,protocol_hash='0'*64)
    source={'schema':'saea-grounded-runtime-source-v1','plan':plan,'source':spec,
        'tokenizer_relpath':tokenizer_relpath,'max_operations':128,'concurrency':min(2,len(ids))}
    registration,binding=build_registration(execution,shard_plan=plan,custody=custody,model_id=model,
        decoding_params=base.learning_decoding(read(repo/f'configs/model/{model}.json')),
        max_context_bytes=2000000,runtime_identity=identity,
        roster_sha256=hashlib.sha256(raw(roster)).hexdigest(),source_batch=batch)
    out.mkdir(parents=True)
    artifacts={'construction-roster.json':roster,'source.json':source,
               'registration.json':registration,'runtime-identity.json':identity}
    for name,doc in artifacts.items():publish_new(out/name,raw(doc))
    complete={'schema':'saea-grounded-shards-job-v1','job':job_name,'run_id':run_id,
        'groups':plan['groups'],'runtime_rows':len(ids),'generation_calls':len(ids),'fit_calls':len(ids),
        'batch_sha256':sha(binding),'runtime_identity_sha256':sha(identity),
        'historical_custody_sha256':sha(custody),'service_session':service_session,
        'inputs':{'source_roster':pin(source_roster,data,'data')},
        'artifacts':{name:file_sha(out/name) for name in artifacts},
        'candidate_publishable':False,'readiness':'prepared-not-admitted',
        'new_model_calls':0,'new_tokenizer_calls':0}
    publish_new(out/'complete.json',raw(complete))
    verify_job(repo,data,out)
    return complete


def verify_job(repo,data,job):
    from speech_aware_evidence_acquisition.rails.tfgrpo_source import read_pin
    complete=read(job/'complete.json')
    if complete.get('schema')=='saea-learning-strategy-job-v1':
        from learning_strategy_job import verify_job as verify_strategy
        return verify_strategy(repo,data,job)
    if complete.get('schema')=='saea-grounded-fusion-job-v1':
        from grounded_fusion_job import verify_job as verify_fusion
        return verify_fusion(repo,data,job)
    if complete.get('schema')!='saea-grounded-shards-job-v1':raise ValueError('grounded job required')
    names={'construction-roster.json','source.json','registration.json','runtime-identity.json'}
    if set(complete['artifacts'])!=names:raise ValueError('exact grounded artifacts required')
    for name in names:
        if file_sha(job/name)!=complete['artifacts'][name]:raise ValueError('grounded artifact changed')
    reg=read(job/'registration.json');roster=read(job/'construction-roster.json');identity=read(job/'runtime-identity.json')
    if (base.deployment_identity(repo,reg['model_id'],identity)!=identity
            or sha(identity)!=reg['runtime_identity_sha256'] or sha(identity)!=complete['runtime_identity_sha256']):
        raise ValueError('grounded deployment identity differs')
    execution=ExecutionPlan(**{**reg['plan'],'protocol_hash':'0'*64}).validate()
    loaded,_=load_runtime_source({'tfgrpo_learning_registration':reg,'prepared_training_feedback':{
        'schema':SOURCE_SCHEMA,'source':pin(job/'source.json',data,'data')}},plan=execution,
        repo_root=repo,data_root=data,sample_ids=tuple(r['sample_id'] for r in roster['items']),
        roster_sha256=file_sha(job/'construction-roster.json'))
    original=read_pin(complete['inputs']['source_roster'],repo_root=repo,data_root=data)
    index={r['sample_id']:r for r in original['items']}
    expected=[index[c['sample_id']] for c in reg['contacts']]
    if (roster['items']!=expected or roster['count']!=len(expected)
            or roster['split_identity_hash']!=split_identity_hash(prefixed('slurp',[r['sample_id'] for r in expected]))
            or execution.split_identity_hash!=roster['split_identity_hash']
            or complete['batch_sha256']!=reg['batch_sha256']
            or complete['historical_custody_sha256']!=sha(loaded['binding']['source_custody'])
            or complete['groups']!=loaded['binding']['plan']['groups']
            or any(complete[k]!=len(expected) for k in ('runtime_rows','generation_calls','fit_calls'))
            or complete['run_id']!=execution.run_id or complete['candidate_publishable'] is not False):
        raise ValueError('grounded job derivation differs')
    return complete,reg,roster


def run(repo,data,job,out,after,predecessor_run):
    import os
    import time
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.campaign import load_campaign_cells,declared_readiness,run_campaign
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from run_real_engine_candidate import assert_campaign_success, generation_release_document
    assert_execution_environment();head=require_clean_study_repo(repo)
    complete,reg,_=verify_job(repo,data,job)
    base.require_learning_deployment(repo,reg['model_id'])
    base.verify_release(after,data,expected_run=predecessor_run)
    if out.exists() or not out.resolve().is_relative_to((data/'work/codex').resolve()):raise ValueError('fresh owned run output required')
    cells=load_campaign_cells(repo,complete['job']);cell,=cells
    if declared_readiness(cell,data).state!='ready':raise ValueError('grounded job held')
    for rid,aid in ((complete['run_id'],'learning-v1'),(complete['run_id']+'-FIT','learning-fit-v1')):
        if (data/'attempts'/rid/(aid+'.json')).exists():raise ValueError('existing spend; registered recovery required')
    out.mkdir(parents=True)
    publish_new(out/'started.json',raw({'study_commit':head,'job':complete['job'],'predecessor_release_sha256':file_sha(after)}))
    try:
        runtime=build_flight_runtime(repo,data,model=reg['model_id'],dataset=cell.document['dataset_fragment'],
            baseline=complete['job'],experiment=complete['job'],run_id=complete['run_id'],attempt_id='learning-v1',
            runtime_root=data/'work/scratch/runtime-root',lock_path=repo.parents[1]/'docs/datasets.lock.json')
        started=time.monotonic()
        def released(flight):
            event=generation_release_document(flight,candidate=complete['job'],
                spec={'run_id':complete['run_id'],'attempt_id':'learning-v1',
                      'profile':os.environ['SAEA_DEPLOYMENT_PROFILE'],
                      'service_session':complete['service_session']},
                commit=head,rows=complete['runtime_rows'],elapsed=time.monotonic()-started)
            publish_new(out/'generation-released.json',raw(event))
        result=run_campaign([runtime],cells=cells,repo_root=repo,data_root=data,
            owner='Codex-grounded-shards',expected_session_id=complete['service_session'],
            on_generation_released=released)
        publish_new(out/'campaign.json',raw(result));assert_campaign_success(result,complete['run_id'])
        publish_new(out/'complete.json',raw({'result':result,'candidate_publishable':False}))
        return {'completed':True,'candidate_publishable':False}
    except BaseException as error:
        publish_new(out/'failed.json',raw({'type':type(error).__name__,'error':str(error),'automatic_retry':False}))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','verify','register','admit','run'])
    for name in ('repo','data','job','out','shard-plan','source-spec','source-roster','runtime-identity','after'):
        parser.add_argument('--'+name,type=Path,required=name in ('repo','data'))
    for name in ('job-name','run-id','service-session','tokenizer-relpath','predecessor-run'):
        parser.add_argument('--'+name)
    args=vars(parser.parse_args());mode=args.pop('mode')
    if mode=='prepare':result=prepare(**{k:args[k] for k in ('repo','data','shard_plan','source_spec','source_roster','runtime_identity','out','job_name','run_id','service_session','tokenizer_relpath')})
    elif mode=='verify':result=verify_job(args['repo'],args['data'],args['job'])[0]
    elif mode=='run':result=run(**{k:args[k] for k in ('repo','data','job','out','after','predecessor_run')})
    else:result=getattr(final,mode)(args['repo'],args['data'],args['job'])
    print(json.dumps(result,sort_keys=True))
