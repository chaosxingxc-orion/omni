"""Prepare, register and run one source-bound final fusion through shared flight."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
import tfgrpo_learning_job as base
import tfgrpo_final_job as final
import grounded_synthesis_job as shards
from speech_aware_evidence_acquisition.contracts import ExecutionPlan, assert_execution_environment
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash
from speech_aware_evidence_acquisition.rails.grounded_fusion_runtime import (
    build_registration, load_runtime_source, SOURCE_SCHEMA,
    build_corrective_registration, load_failed_fusion, CORRECTIVE_STRATEGY,
)
from speech_aware_evidence_acquisition.rails.grounded_corrective_fusion import (
    build_corrective_input, corrective_messages,
)
from speech_aware_evidence_acquisition.rails.grounded_synthesis_fusion import (
    load_finalized_shards, build_fusion_input, fusion_messages,
)
from speech_aware_evidence_acquisition.rails.grounded_synthesis_plan import load_token_counter
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha

read,raw,pin,file_sha=base.read,base.raw,base.pin,base.file_sha


def prepare(repo,data,*,source_spec,source_roster,tokenizer_relpath,runtime_identity,
            out,job_name,run_id,service_session, strategy='fusion-v1', failed_fusion_spec=None):
    assert_execution_environment()
    if not re.fullmatch('[a-z0-9][a-z0-9-]*',job_name):raise ValueError('safe fusion job required')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*',service_session):raise ValueError('safe service session required')
    if out.exists() or not out.resolve().is_relative_to((data/'work/codex').resolve()):
        raise ValueError('fresh owned fusion directory required')
    rel=Path(tokenizer_relpath)
    if rel.is_absolute() or '..' in rel.parts or not (data/rel).resolve().is_relative_to(data.resolve()):
        raise ValueError('confined tokenizer required')
    spec,identity=read(source_spec),read(runtime_identity)
    count,tokenizer=load_token_counter(data/rel)
    source=load_finalized_shards(spec,data_root=data,count_tokens=count,tokenizer_identity=tokenizer)
    document=build_fusion_input(source)
    if strategy == CORRECTIVE_STRATEGY:
        if failed_fusion_spec is None:raise ValueError('corrective strategy requires pinned failed source')
        failed_spec=read(failed_fusion_spec)
        failure,terminal=load_failed_fusion(failed_spec,data_root=data,source=source,document=document)
        corrective=build_corrective_input(document,failure)
        measured=corrective_messages(corrective,count_tokens=count,reserve_tokens=2048)
    elif strategy == 'fusion-v1' and failed_fusion_spec is None:
        measured=fusion_messages(document,count_tokens=count,reserve_tokens=2048)
    else:raise ValueError('known strategy and matching source options required')
    model=source['model_id']
    if base.deployment_identity(repo,model,identity)!=identity:raise ValueError('exact fusion deployment identity required')
    qid=source['batch']['groups'][0]['query_id']
    original=read(source_roster);items={r['sample_id']:r for r in original['items']}
    if len(items)!=len(original['items']) or qid not in items:raise ValueError('fusion anchor absent from source roster')
    roster={'schema':'saea-roster-receipt-v1','roster_id':run_id,'roster_role':'construction',
        'carrier_lock_key':'slurp','split_role':'discovery','count':1,'items':[items[qid]],
        'split_identity_hash':split_identity_hash(prefixed('slurp',[qid])),
        'selection':'First original TRAIN anchor; all source groups bound to final fusion.',
        'source_queries_must_not_be_reported_as_held_out':True}
    execution=ExecutionPlan(run_id=run_id,attempt_id='learning-v1',
        execution_profile='bounded-discovery-probe',carrier_lock_key='slurp',split_role='discovery',
        split_identity_hash=roster['split_identity_hash'],planned_model_calls=1,
        planned_gpu_hours=8.0,planned_speech_audio_seconds=1,protocol_hash='0'*64)
    registration_args=dict(model_id=model,
        decoding_params=base.learning_decoding(read(repo/f'configs/model/{model}.json')),
        max_context_bytes=2000000,runtime_identity_sha256=sha(identity),
        roster_sha256=hashlib.sha256(raw(roster)).hexdigest())
    if strategy == CORRECTIVE_STRATEGY:
        registration,binding,custody=build_corrective_registration(execution,source=source,
            corrective=corrective,terminal=terminal,measured=measured,**registration_args)
    else:
        registration,binding=build_registration(execution,source=source,document=document,
            measured=measured,**registration_args)
        custody=source
    runtime_source={'schema':'saea-grounded-fusion-runtime-source-v1','source':spec,
        'tokenizer_relpath':tokenizer_relpath,'measurement':measured,'input_sha256':document['input_sha256']}
    if strategy == CORRECTIVE_STRATEGY:
        runtime_source.update(strategy=strategy,failed_fusion=failed_spec,
                              corrective_input_sha256=corrective['corrective_input_sha256'])
    out.mkdir(parents=True)
    artifacts={'construction-roster.json':roster,'source.json':runtime_source,
               'registration.json':registration,'runtime-identity.json':identity}
    for name,doc in artifacts.items():publish_new(out/name,raw(doc))
    complete={'schema':'saea-grounded-fusion-job-v1','job':job_name,'run_id':run_id,
        'groups':len(source['batch']['groups']),'runtime_rows':1,'generation_calls':1,'fit_calls':1,
        'batch_sha256':sha(binding),'runtime_identity_sha256':sha(identity),
        'historical_custody_sha256':sha(custody),'service_session':service_session,
        'inputs':{'source_roster':pin(source_roster,data,'data')},
        'artifacts':{n:file_sha(out/n) for n in artifacts},'semantic_review_required':True,
        'candidate_deployment_accepted':False,'readiness':'prepared-not-admitted',
        'new_model_calls':0,'new_tokenizer_calls':0}
    if strategy == CORRECTIVE_STRATEGY:
        complete.update(strategy=strategy,correction_round=1,
                        historical_usage=custody['original_total_usage'])
    publish_new(out/'complete.json',raw(complete))
    verify_job(repo,data,out)
    return complete


def verify_job(repo,data,job):
    from speech_aware_evidence_acquisition.rails.tfgrpo_source import read_pin
    complete=read(job/'complete.json')
    if complete.get('schema')!='saea-grounded-fusion-job-v1':raise ValueError('fusion job required')
    names={'construction-roster.json','source.json','registration.json','runtime-identity.json'}
    if set(complete['artifacts'])!=names:raise ValueError('exact fusion artifacts required')
    for n in names:
        if file_sha(job/n)!=complete['artifacts'][n]:raise ValueError('fusion artifact changed')
    reg,roster,identity=[read(job/n) for n in ('registration.json','construction-roster.json','runtime-identity.json')]
    if (base.deployment_identity(repo,reg['model_id'],identity)!=identity
            or sha(identity)!=reg['runtime_identity_sha256']
            or sha(identity)!=complete['runtime_identity_sha256']):raise ValueError('fusion runtime identity differs')
    execution=ExecutionPlan(**{**reg['plan'],'protocol_hash':'0'*64}).validate()
    loaded,_=load_runtime_source({'tfgrpo_learning_registration':reg,'prepared_training_feedback':{
        'schema':SOURCE_SCHEMA,'source':pin(job/'source.json',data,'data')}},plan=execution,
        repo_root=repo,data_root=data,sample_ids=tuple(x['sample_id'] for x in roster['items']),
        roster_sha256=file_sha(job/'construction-roster.json'))
    original=read_pin(complete['inputs']['source_roster'],repo_root=repo,data_root=data)
    index={r['sample_id']:r for r in original['items']}
    expected=[index[reg['contacts'][0]['sample_id']]]
    if (roster['items']!=expected or roster['count']!=1
            or roster['split_identity_hash']!=split_identity_hash(prefixed('slurp',[expected[0]['sample_id']]))
            or execution.split_identity_hash!=roster['split_identity_hash']
            or complete['run_id']!=execution.run_id
            or complete['batch_sha256']!=sha(loaded['binding'])
            or complete['historical_custody_sha256']!=sha(loaded.get('custody',loaded['source']))
            or complete['groups']!=len(loaded['source']['batch']['groups'])
            or any(complete[k]!=1 for k in ('runtime_rows','generation_calls','fit_calls'))
            or complete['semantic_review_required'] is not True
            or complete['candidate_deployment_accepted'] is not False):
        raise ValueError('fusion job derivation differs')
    if loaded.get('strategy') == CORRECTIVE_STRATEGY:
        if (complete.get('strategy') != CORRECTIVE_STRATEGY or complete.get('correction_round') != 1
                or complete.get('historical_usage') != loaded['custody']['original_total_usage']):
            raise ValueError('corrective job lineage or cost differs')
    elif any(k in complete for k in ('strategy','correction_round','historical_usage')):
        raise ValueError('plain fusion job cannot carry corrective metadata')
    return complete,reg,roster


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','verify','register','admit','run'])
    for name in ('repo','data','job','out','source-spec','source-roster','runtime-identity','after','failed-fusion-spec'):
        parser.add_argument('--'+name,type=Path,required=name in ('repo','data'))
    for name in ('job-name','run-id','service-session','tokenizer-relpath','predecessor-run'):
        parser.add_argument('--'+name)
    parser.add_argument('--strategy',choices=['fusion-v1',CORRECTIVE_STRATEGY],default='fusion-v1')
    args=vars(parser.parse_args());mode=args.pop('mode')
    if mode=='prepare':result=prepare(**{k:args[k] for k in ('repo','data','source_spec','source_roster',
        'runtime_identity','out','job_name','run_id','service_session','tokenizer_relpath',
        'strategy','failed_fusion_spec')})
    elif mode=='verify':result=verify_job(args['repo'],args['data'],args['job'])[0]
    elif mode=='run':result=shards.run(**{k:args[k] for k in ('repo','data','job','out','after','predecessor_run')})
    else:result=getattr(final,mode)(args['repo'],args['data'],args['job'])
    print(__import__('json').dumps(result,sort_keys=True))
