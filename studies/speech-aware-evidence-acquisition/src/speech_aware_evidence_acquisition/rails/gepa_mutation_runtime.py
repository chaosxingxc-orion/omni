"""GEPA mutation on registered Flight/FIT/native journals, with no new transport."""
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from . import gepa_mutation_execution as execution
from . import gepa_mutation_source as evidence
from . import solver_prompt_candidate as prompt
from .tfgrpo_learning import ExperienceLearner, canonical, sha
from .tfgrpo_runtime import learning_configuration, learning_source_plan, bind_runtime, contact_key
from .tfgrpo_source import read_pin
from .tokenizer_runtime import load_text_token_counter
from ..core.atomic_file import publish_new

RUNTIME_SCHEMA='saea-gepa-mutation-runtime-binding-v1'
SOURCE_SCHEMA='saea-prepared-gepa-mutation-v1'
RUNTIME_SOURCE_SCHEMA='saea-gepa-mutation-runtime-source-v1'
require=execution.require


def validate_registration(reg,*,plan,model_id):
    fields={'schema','plan','model_id','batch_sha256','execution_sha256','construction_roster_sha256',
        'contacts','decoding_params','max_context_bytes','learning_config_sha256','runtime_identity_sha256',
        'completion_cap','resident_context','reserve_tokens'}
    require(isinstance(reg,dict) and set(reg)==fields and reg['schema']==RUNTIME_SCHEMA,
        'exact GEPA mutation registration required')
    require(reg['plan']=={k:v for k,v in asdict(plan).items() if k!='protocol_hash'}
        and plan.planned_model_calls==1 and plan.split_role=='discovery'
        and plan.execution_profile=='bounded-discovery-probe' and reg['model_id']==model_id,
        'one exact TRAIN mutation contact required')
    for key in ('batch_sha256','execution_sha256','construction_roster_sha256','learning_config_sha256','runtime_identity_sha256'):
        require(isinstance(reg[key],str) and re.fullmatch('[a-f0-9]{64}',reg[key]),'pinned mutation digests required')
    require(reg['batch_sha256']==reg['execution_sha256'] and reg['learning_config_sha256']==sha(learning_configuration(reg)),
        'mutation execution/config identity changed')
    rows=reg['contacts'];require(isinstance(rows,list) and len(rows)==1,'single mutation roster required')
    row=rows[0];require(isinstance(row,dict) and set(row)=={'stage','identity','sample_id','messages_sha256'}
        and row['stage']==execution.STAGE and isinstance(row['sample_id'],str)
        and row['sample_id'].startswith('slurp:train:')
        and all(isinstance(row[k],str) and re.fullmatch('[a-f0-9]{64}',row[k]) for k in ('identity','messages_sha256')),
        'exact mutation prompt and TRAIN anchor required')
    require(all(type(reg[k]) is int and reg[k]>0 for k in ('max_context_bytes','completion_cap','resident_context'))
        and type(reg['reserve_tokens']) is int and reg['reserve_tokens']>=0
        and reg['completion_cap']+reg['reserve_tokens']<reg['resident_context'], 'finite development budget required')
    decoding=reg['decoding_params'];require(isinstance(decoding,dict) and 'id_slot' not in decoding
        and decoding.get('max_tokens')==reg['completion_cap'] and type(decoding.get('max_tokens')) is int,
        'exact mutation completion budget required')
    return {contact_key(row['stage'],row['identity']):row}


def build_registration(plan,*,source,model_id,decoding_params,max_context_bytes,
        runtime_identity_sha256,roster_sha256,resident_context,reserve_tokens):
    config={'model_id':model_id,'decoding_params':decoding_params,'max_context_bytes':max_context_bytes,
        'runtime_identity_sha256':runtime_identity_sha256}
    binding={'schema':execution.SCHEMA,'model_id':model_id,'learning_config_sha256':sha(learning_configuration(config)),
        'source_custody_sha256':sha(source),'identity':sha(source),'messages':evidence.messages(source)}
    reg={'schema':RUNTIME_SCHEMA,'plan':{k:v for k,v in asdict(plan).items() if k!='protocol_hash'},
        **config,'batch_sha256':sha(binding),'execution_sha256':sha(binding),
        'construction_roster_sha256':roster_sha256,'learning_config_sha256':binding['learning_config_sha256'],
        'completion_cap':decoding_params['max_tokens'],'resident_context':resident_context,'reserve_tokens':reserve_tokens,
        'contacts':[{'stage':execution.STAGE,'identity':binding['identity'],
            'sample_id':source['feedback_query_ids'][0],'messages_sha256':sha(binding['messages'])}]}
    validate_registration(reg,plan=plan,model_id=model_id)
    return reg,binding


def load_runtime_source(values,*,plan,repo_root,data_root,sample_ids,roster_sha256,replay=None):
    spec=values.get('prepared_training_feedback')
    require(isinstance(spec,dict) and set(spec)=={'schema','source'} and spec['schema']==SOURCE_SCHEMA,
        'exact prepared mutation source required')
    doc=read_pin(spec['source'],repo_root=repo_root,data_root=data_root)
    require(isinstance(doc,dict) and set(doc)=={'schema','source','source_custody_sha256'}
        and doc['schema']==RUNTIME_SOURCE_SCHEMA,'exact runtime mutation source required')
    source=evidence.load_source(doc['source'],repo=repo_root,data=data_root)
    require(sha(source)==doc['source_custody_sha256'],'mutation source reconstruction changed')
    reg=values['tfgrpo_learning_registration']
    expected,binding=build_registration(learning_source_plan(plan,replay=replay,data_root=data_root),
        source=source,model_id=reg['model_id'],decoding_params=reg['decoding_params'],
        max_context_bytes=reg['max_context_bytes'],runtime_identity_sha256=reg['runtime_identity_sha256'],
        roster_sha256=roster_sha256,resident_context=reg['resident_context'],reserve_tokens=reg['reserve_tokens'])
    require(reg==expected and tuple(sample_ids)==(reg['contacts'][0]['sample_id'],),
        'mutation registration or runtime roster differs')
    return {'source':source,'binding':binding,'spec':doc}, {'schema':'saea-gepa-mutation-source-receipt-v1',
        'source':spec['source'],'batch_sha256':sha(binding),'contacts':1,'rows':1,
        'construction_roster_sha256':roster_sha256,'dynamic_states':True,'semantic_review_required':True}


def request_validator(binding):
    expected=execution.request(binding)
    def validate(request,caller):
        require(request==expected,'mutation request differs from reconstructed source and hypothesis')
    return validate


def enforce_native_budget(caller,reg):
    original=caller.request_fit
    def fit(prepared):
        result=original(prepared);tokens=result.get('measured_prompt_tokens')
        require(type(tokens) is int and tokens>0
            and tokens+reg['completion_cap']+reg['reserve_tokens']<=reg['resident_context'],
            'wrapped mutation context exceeds development budget; do not truncate')
        return result
    caller.request_fit=fit


def verify_recovery(replay,*,data_root,plan,reg,binding):
    from ..core.contact_journal import comparable_request
    from ..core.preparation import load_artifact
    from .tfgrpo_consumption import checked_pin
    replay.assert_target(data_root,plan,'prepared-context-text');replay.assert_all_contacts_observed()
    require(len(replay.request_ids)<=1,'mutation recovery contains excess contacts')
    root=Path(data_root)/'runs'/replay.source_plan.run_id/replay.source_plan.attempt_id/'tfgrpo'
    require(json.loads((root/'contacts/registration.json').read_bytes())==reg
        and json.loads((root/'learner/batch.json').read_bytes())==binding,'mutation recovery source changed')
    request=execution.request(binding);key=contact_key(request['stage'],request['identity'])
    expected={key} if replay.request_ids else set()
    for directory in (root/'contacts',root/'learner/contacts'):
        intents={p.name.removesuffix('.intent.json') for p in directory.glob('*.intent.json')}
        responses={p.name.removesuffix('.response.json') for p in directory.glob('*.response.json')}
        require(intents==expected and responses<=expected,'mutation recovery unknown or foreign local contact')
    for rid in replay.request_ids:
        native=replay.source.inspect(rid);payload=native['intent']['payload']
        old=json.loads((root/'contacts'/(key+'.intent.json')).read_bytes())
        require(native['intent']['agent_contact_id']=='tfgrpo-'+key and old['request']==request
            and json.loads((root/'learner/contacts'/(key+'.intent.json')).read_bytes())==request
            and comparable_request(old['payload'])==comparable_request(payload)
            and json.loads(payload['history'][0]['content'])['source_messages']==binding['messages'],
            'mutation recovery actual native request differs')
        fit=json.loads((root/'contacts'/(key+'.fit.json')).read_bytes())
        path=checked_pin(Path(data_root),{'relpath':fit['artifact_relpath'],'sha256':fit['artifact_sha256']})
        artifact=json.loads(path.read_bytes());load_artifact(path,expected_sha256=fit['artifact_sha256'],kind='request-fit',inputs=artifact['inputs'])
        require(fit['payload_sha256']==sha(old['payload'])==artifact['inputs']['payload_sha256']
            and artifact['inputs']['registration_sha256']==sha(reg) and fit['every_request_fits'] is True
            and artifact['output']['every_request_fits'] is True
            and fit['measured_prompt_tokens']==artifact['output']['rows'][0]['measured_prompt_tokens']
            and fit['measured_prompt_tokens']+reg['completion_cap']+reg['reserve_tokens']<=reg['resident_context'],
            'mutation recovery original native FIT differs')


def run_registered_mutation(runtime):
    reg=runtime._composed.values['tfgrpo_learning_registration'];root=runtime._run_dir()/'tfgrpo'
    def reload():
        return load_runtime_source(runtime._composed.values,plan=runtime.plan,repo_root=runtime._repo_root,
            data_root=runtime._data_root,sample_ids=tuple(s.sample_id for s in runtime._samples),
            roster_sha256=reg['construction_roster_sha256'],replay=getattr(runtime,'_native_contact_replay',None))[0]
    loaded=reload();replay=getattr(runtime,'_native_contact_replay',None)
    if replay is not None:verify_recovery(replay,data_root=runtime._data_root,plan=runtime.plan,reg=reg,binding=loaded['binding'])
    caller=getattr(runtime,'_tfgrpo_caller',None)
    if caller is None:
        require(not any((root/'contacts').glob('*.intent.json')) and not any((root/'learner/contacts').glob('*.intent.json')),
            'prior mutation requires registered native recovery; no resend')
        caller=bind_runtime(runtime,receipt_root=root/'contacts',request_validator=request_validator(loaded['binding']))
        enforce_native_budget(caller,reg);runtime._client,runtime._tfgrpo_caller=caller.client,caller
    else:
        require(caller.root.resolve()==(root/'contacts').resolve() and caller.registration==reg
            and caller.client is runtime._client and caller.plan==runtime.plan and not caller.client._closed,
            'live mutation client changed')
    learner=ExperienceLearner(root/'learner',model_id=reg['model_id'],learning_config_sha256=reg['learning_config_sha256'],
        metered_call=caller,preflight=caller.preflight,reconcile=caller.reconcile,max_calls=1,concurrency=1)
    count,tokenizer=load_text_token_counter(Path(loaded['source']['tokenizer_path']))
    try:
        require(tokenizer==loaded['source']['tokenizer_identity'],'mutation tokenizer changed')
        return execution.execute(learner,loaded['binding'],verify_source=lambda:reload()['source'],count_tokens=count)
    finally:
        if hasattr(count,'close'):count.close()


def dispatch_registered_mutation(runtime):
    started=getattr(runtime,'_tfgrpo_started',None)
    if started is None:started=runtime._clock();runtime._tfgrpo_started=started
    candidate=run_registered_mutation(runtime)
    from .tfgrpo_runtime import seal_registered_candidate
    return seal_registered_candidate(runtime,candidate,started=started,ended=runtime._clock(),candidate_kind='solver_prompt')


def validate_result(candidate):
    require(candidate.get('schema')==execution.RESULT_SCHEMA and candidate.get('candidate_kind')=='solver_prompt'
        and candidate.get('sha256')==sha({k:v for k,v in candidate.items() if k!='sha256'})
        and candidate.get('deployment_accepted') is False and candidate.get('semantic_review_required') is True
        and candidate.get('accounting',{}).get('unknown_contacts')==0
        and candidate.get('accounting',{}).get('observed_contacts')==1
        and candidate.get('learning_usage',{}).get('calls')==1, 'complete unadopted prompt mutation result required')


def verify_mutation_bundle(bundle,*,repo,data):
    from ..core.config import compose
    from ..contracts import ExecutionPlan
    require(bundle.attempt_state=='FINALIZED','finalized native mutation required')
    proposals=[r['org_gepa_mutation_proposal'] for r in bundle.trace_records.values() if 'org_gepa_mutation_proposal' in r]
    require(len(proposals)==1,'one sealed prompt mutation proposal required');proposal=proposals[0]
    candidate=read_pin({k:v for k,v in proposal['candidate'].items() if k in ('root','relpath','sha256')},repo_root=repo,data_root=data)
    validate_result(candidate);fragments=bundle.manifest['config']['fragments']
    cfg=compose(Path(repo),*(fragments[k] for k in ('model','dataset','baseline','experiment')))
    require(cfg.config_hash==bundle.manifest['config']['config_hash'],'mutation source configuration changed')
    reg=cfg.values['tfgrpo_learning_registration']
    loaded,_=load_runtime_source(cfg.values,plan=ExecutionPlan(**{**reg['plan'],'protocol_hash':bundle.manifest['plan']['protocol_hash']}),
        repo_root=repo,data_root=data,sample_ids=tuple(bundle.split_ids),roster_sha256=reg['construction_roster_sha256'])
    source=loaded['source'];contacts=[(key.removesuffix('/response'),r) for key,r in bundle.trace_records.items()
        if key.endswith('/response') and 'response_text_sha256' in r]
    require(len(contacts)==1,'one native response required');rid,response=contacts[0]
    payload=bundle.trace_records[rid+'/request']['payload'];context=json.loads(payload['history'][0]['content'])
    require(context['source_messages']==loaded['binding']['messages'] and context['stage']==execution.STAGE
        and payload['sample_id']==reg['contacts'][0]['sample_id'],'sealed native mutation prompt differs')
    count,tokenizer=load_text_token_counter(Path(source['tokenizer_path']))
    try:
        require(tokenizer==source['tokenizer_identity'],'sealed mutation tokenizer changed')
        child=prompt.derive_candidate(response['text'],parent=source['parent'],expected_parent_sha256=source['parent']['candidate_sha256'],
            contract=source['contract'],expected_contract_sha256=source['contract']['contract_sha256'],count_tokens=count)
    finally:
        if hasattr(count,'close'):count.close()
    cost=bundle.manifest['cost'];usage={'calls':cost['calls_used'],'prompt_tokens':cost['prompt_tokens_total'],'completion_tokens':cost['completion_tokens_total']}
    require(candidate['child']==child and candidate['source_custody_sha256']==sha(source)
        and candidate['batch_sha256']==sha(loaded['binding']) and candidate['learning_usage']==usage==proposal['usage']
        and candidate['contact_response_sha256']==proposal['contact_response_sha256']
        and proposal['candidate']['candidate_sha256']==candidate['sha256'],'native mutation seal or usage changed')
    return {'mutation_text':response['text'],'response_sha256':response['response_sha256'],
        'contract_sha256':source['contract']['contract_sha256'],'parent_sha256':source['parent']['candidate_sha256'],
        'child_sha256':child['candidate_sha256'],'usage':usage,'historical_usage':source['historical_usage'],
        'candidate_accepted':False,'source_manifest_sha256':hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest()}
