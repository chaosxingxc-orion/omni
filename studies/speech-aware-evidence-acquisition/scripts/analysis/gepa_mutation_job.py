"""Configuration-only preparation, registration and execution of one GEPA mutation."""
import argparse
import copy
import json
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
import tfgrpo_learning_job as base
from speech_aware_evidence_acquisition.contracts import ExecutionPlan,assert_execution_environment
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.data.splits import prefixed,split_identity_hash
from speech_aware_evidence_acquisition.rails import gepa_mutation_source as evidence
from speech_aware_evidence_acquisition.rails import gepa_mutation_runtime as runtime
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
from speech_aware_evidence_acquisition.rails.tfgrpo_source import read_pin
read,raw,pin,file_sha=base.read,base.raw,base.pin,base.file_sha
CONFIG_SCHEMA='saea-gepa-mutation-job-config-v1'
JOB_SCHEMA='saea-gepa-mutation-job-v1'


def config(value):
    fields={'schema','name','run_id','model_fragment','service_session','runtime_identity','source','bounds'}
    if not isinstance(value,dict) or set(value)!=fields or value['schema']!=CONFIG_SCHEMA:
        raise ValueError('exact GEPA mutation job config required')
    for key in ('name','model_fragment','service_session'):
        if not isinstance(value[key],str) or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*',value[key]) is None:
            raise ValueError('safe mutation job identity required')
    if not re.fullmatch('[A-Z0-9][A-Z0-9-]*',value['run_id']):raise ValueError('safe mutation run identity required')
    bounds=value['bounds']
    if (set(bounds)!={'resident_context','reserve_tokens','max_context_bytes'}
        or any(type(bounds[k]) is not int or bounds[k]<=0 for k in ('resident_context','max_context_bytes'))
        or type(bounds['reserve_tokens']) is not int or bounds['reserve_tokens']<0):
        raise ValueError('explicit finite development context budget required')
    return copy.deepcopy(value)


def documents(repo,data,cfg):
    cfg=config(cfg);source=evidence.load_source(cfg['source'],repo=repo,data=data)
    identity=read_pin(cfg['runtime_identity'],repo_root=repo,data_root=data)
    model_id=cfg['model_fragment'];model=read(repo/f'configs/model/{model_id}.json')
    if base.deployment_identity(repo,model_id,identity)!=identity:raise ValueError('current mutation deployment differs')
    seed=evidence.seed_contract(cfg['source']['seed_material'],repo=repo,data=data)
    if model.get('model_lock_key')!=seed['model'].get('model_lock_key'):
        raise ValueError('mutation cannot change frozen model weights')
    decoding=base.learning_decoding(model)
    decoding['response_format']={'type':'json_object'}
    ids=source['feedback_query_ids'];anchor=ids[0]
    roster={'schema':'saea-roster-receipt-v1','carrier_lock_key':'slurp','roster_id':cfg['run_id'],
        'roster_role':'construction','split_role':'discovery','count':1,
        'items':[{'sample_id':anchor,'stratum':'gepa-feedback-anchor','audio_seconds':0}],
        'split_identity_hash':split_identity_hash(prefixed('slurp',[anchor])),
        'selection':'One original TRAIN anchor for reflection; full selected feedback is bound in source.'}
    plan=ExecutionPlan(run_id=cfg['run_id'],attempt_id='learning-v1',execution_profile='bounded-discovery-probe',
        carrier_lock_key='slurp',split_role='discovery',split_identity_hash=roster['split_identity_hash'],
        planned_model_calls=1,planned_gpu_hours=1.0,planned_speech_audio_seconds=1,protocol_hash='0'*64)
    reg,binding=runtime.build_registration(plan,source=source,model_id=model_id,decoding_params=decoding,
        max_context_bytes=cfg['bounds']['max_context_bytes'],runtime_identity_sha256=sha(identity),
        roster_sha256=__import__('hashlib').sha256(raw(roster)).hexdigest(),
        resident_context=cfg['bounds']['resident_context'],reserve_tokens=cfg['bounds']['reserve_tokens'])
    doc={'schema':runtime.RUNTIME_SOURCE_SCHEMA,'source':cfg['source'],'source_custody_sha256':sha(source)}
    return {'construction-roster.json':roster,'source.json':doc,'registration.json':reg,
        'runtime-identity.json':identity,'job-config.json':cfg,'execution.json':binding},source,plan


def prepare(repo,data,cfg,out):
    assert_execution_environment()
    if out.exists() or not out.resolve().is_relative_to((data/'work/codex').resolve()):
        raise ValueError('fresh owned mutation preparation directory required')
    artifacts,source,_=documents(repo,data,cfg);cfg=artifacts['job-config.json'];reg=artifacts['registration.json']
    out.mkdir(parents=True)
    for name,doc in artifacts.items():publish_new(out/name,raw(doc))
    complete={'schema':JOB_SCHEMA,'job':cfg['name'],'run_id':cfg['run_id'],'strategy':'gepa_solver_prompt_search_v1',
        'groups':len(source['feedback_query_ids']),'runtime_rows':1,'generation_calls':1,'fit_calls':1,
        'new_model_calls':0,'tokenizer_work':'CPU source verification; calls not counted','batch_sha256':reg['batch_sha256'],
        'runtime_identity_sha256':reg['runtime_identity_sha256'],'source_custody_sha256':sha(source),
        'historical_usage':source['historical_usage'],'service_session':cfg['service_session'],
        'artifacts':{n:file_sha(out/n) for n in artifacts},'readiness':'prepared-not-admitted',
        'candidate_accepted':False,'focus':cfg['source']['focus']}
    publish_new(out/'complete.json',raw(complete));return complete


def verify_job(repo,data,job):
    complete=read(job/'complete.json')
    if complete.get('schema')!=JOB_SCHEMA:raise ValueError('exact prepared mutation job required')
    artifacts,source,plan=documents(repo,data,read(job/'job-config.json'))
    if set(complete['artifacts'])!=set(artifacts):raise ValueError('complete mutation artifact inventory required')
    for name,want in artifacts.items():
        if file_sha(job/name)!=complete['artifacts'][name] or read(job/name)!=want:
            raise ValueError('prepared mutation artifact changed')
    cfg=artifacts['job-config.json'];reg=artifacts['registration.json'];roster=artifacts['construction-roster.json']
    expected={'job':cfg['name'],'run_id':cfg['run_id'],'groups':len(source['feedback_query_ids']),
        'runtime_rows':1,'generation_calls':1,'fit_calls':1,'batch_sha256':reg['batch_sha256'],
        'runtime_identity_sha256':reg['runtime_identity_sha256'],'source_custody_sha256':sha(source),
        'historical_usage':source['historical_usage'],'service_session':cfg['service_session'],
        'focus':cfg['source']['focus'],'candidate_accepted':False}
    if any(complete.get(k)!=v for k,v in expected.items()):raise ValueError('mutation job counts/source differ')
    runtime.load_runtime_source({'prepared_training_feedback':{'schema':runtime.SOURCE_SCHEMA,
        'source':pin(job/'source.json',data,'data')},'tfgrpo_learning_registration':reg},
        plan=plan,repo_root=repo,data_root=data,sample_ids=tuple(r['sample_id'] for r in roster['items']),
        roster_sha256=file_sha(job/'construction-roster.json'))
    return complete,reg,roster


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','verify','register','admit','run'])
    for key in ('repo','data'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('config','job','out','after'):p.add_argument('--'+key,type=Path)
    p.add_argument('--predecessor-run');a=p.parse_args()
    if a.mode=='prepare':result=prepare(a.repo,a.data,read(a.config),a.out)
    elif a.mode=='verify':result=verify_job(a.repo,a.data,a.job)[0]
    elif a.mode in ('register','admit'):
        import tfgrpo_final_job
        result=getattr(tfgrpo_final_job,a.mode)(a.repo,a.data,a.job)
    else:
        from grounded_synthesis_job import run
        result=run(a.repo,a.data,a.job,a.out,a.after,a.predecessor_run)
    print(json.dumps(result,sort_keys=True))
