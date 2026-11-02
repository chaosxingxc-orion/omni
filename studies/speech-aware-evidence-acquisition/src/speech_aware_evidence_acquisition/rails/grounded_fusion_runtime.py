"""Finite grounded fusion through the shared registered native FIT/client."""
from dataclasses import asdict
from pathlib import Path

from .grounded_synthesis_plan import load_token_counter
from .grounded_synthesis_fusion import (
    load_finalized_shards, build_fusion_input, fusion_messages, validate_fusion_output,
)
from .grounded_fusion_execution import SCHEMA, STAGE, execute_fusion, require
from .tfgrpo_source import read_pin
from .tfgrpo_learning import ExperienceLearner, sha, DENSE_REWARD
from .tfgrpo_runtime import (
    FUSION_RUNTIME_SCHEMA, learning_configuration, validate_registration,
    learning_source_plan, bind_runtime, seal_registered_candidate,
)

SOURCE_SCHEMA = 'saea-prepared-grounded-fusion-v1'


def build_registration(plan, *, source, document, measured, model_id, decoding_params,
                       max_context_bytes, runtime_identity_sha256, roster_sha256):
    require(source['model_id'] == model_id and plan.planned_model_calls == 1,
            'same-model single fusion contact required')
    require(document == build_fusion_input(source), 'fusion document differs from verified evidence')
    require(measured['completion_cap'] == decoding_params['max_tokens'],
            'fusion output budget differs')
    config = {'model_id': model_id, 'decoding_params': decoding_params,
              'max_context_bytes': max_context_bytes, 'runtime_identity_sha256':runtime_identity_sha256}
    config_sha = sha(learning_configuration(config))
    batch = source['batch']
    ids = [g['query_id'] for g in batch['groups']]
    binding = {'schema':SCHEMA, 'model_id':model_id, 'learning_config_sha256':config_sha,
        'source_custody_sha256':sha(source), 'input_sha256':document['input_sha256'],
        'parent':batch['parent'], 'learned_query_ids':ids,
        'rollout_kind':batch['rollout_kind'], 'reward_mode':batch['common'].get('reward_mode',DENSE_REWARD),
        'identity':document['input_sha256'], 'messages':measured['messages'],
        'bounds':{'max_items':measured['max_items'], 'max_characters':measured['max_characters'],
                  'max_operations':measured['max_items']}}
    reg = {'schema':FUSION_RUNTIME_SCHEMA,
        'plan':{k:v for k,v in asdict(plan).items() if k!='protocol_hash'}, **config,
        'batch_sha256':sha(binding), 'execution_sha256':sha(binding),
        'construction_roster_sha256':roster_sha256, 'learning_config_sha256':config_sha,
        'completion_cap':measured['completion_cap'],
        'contacts':[{'stage':STAGE,'identity':binding['identity'],'sample_id':ids[0],
                     'messages_sha256':sha(measured['messages'])}]}
    validate_registration(reg,plan=plan,model_id=model_id)
    return reg,binding


def load_runtime_source(values, *, plan, repo_root, data_root, sample_ids,
                        roster_sha256, replay=None):
    spec=values.get('prepared_training_feedback')
    require(isinstance(spec,dict) and set(spec)=={'schema','source'}
            and spec['schema']==SOURCE_SCHEMA,'exact fusion dynamic source required')
    doc=read_pin(spec['source'],repo_root=repo_root,data_root=data_root)
    require(isinstance(doc,dict) and set(doc)=={'schema','source','tokenizer_relpath',
        'measurement','input_sha256'} and doc['schema']=='saea-grounded-fusion-runtime-source-v1',
        'exact fusion source document required')
    rel=doc['tokenizer_relpath']
    require(isinstance(rel,str) and rel and not Path(rel).is_absolute()
            and '..' not in Path(rel).parts,'confined fusion tokenizer required')
    path=(Path(data_root)/rel).resolve()
    require(path.is_relative_to(Path(data_root).resolve()),'tokenizer escapes root')
    count,identity=load_token_counter(path)
    source=load_finalized_shards(doc['source'],data_root=data_root,
                                 count_tokens=count,tokenizer_identity=identity)
    document=build_fusion_input(source)
    require(document['input_sha256']==doc['input_sha256'],'fusion evidence digest differs')
    old=doc['measurement']
    measured=fusion_messages(document,count_tokens=count,
        **{k:old[k] for k in ('resident_context','completion_cap','reserve_tokens','max_items','max_characters')})
    require(measured==old,'fusion measured request changed')
    reg=values['tfgrpo_learning_registration']
    scientific=learning_source_plan(plan,replay=replay,data_root=data_root)
    expected,binding=build_registration(scientific,source=source,document=document,measured=measured,
        model_id=source['model_id'],decoding_params=reg['decoding_params'],
        max_context_bytes=reg['max_context_bytes'],runtime_identity_sha256=reg['runtime_identity_sha256'],
        roster_sha256=roster_sha256)
    require(reg==expected and tuple(sample_ids)==(reg['contacts'][0]['sample_id'],),
            'fusion registration or roster differs')
    return {'source':source,'document':document,'measurement':measured,'binding':binding,
            'spec':doc,'count_tokens':count,'tokenizer_identity':identity}, {
        'schema':'saea-grounded-fusion-source-receipt-v1','source':spec['source'],
        'batch_sha256':sha(binding),'contacts':1,'rows':1,
        'construction_roster_sha256':roster_sha256,'dynamic_states':True,
        'semantic_review_required':True}


def run_registered_fusion(runtime):
    reg=runtime._composed.values['tfgrpo_learning_registration']
    loaded,_=load_runtime_source(runtime._composed.values,plan=runtime.plan,
        repo_root=runtime._repo_root,data_root=runtime._data_root,
        sample_ids=tuple(s.sample_id for s in runtime._samples),
        roster_sha256=reg['construction_roster_sha256'],
        replay=getattr(runtime,'_native_contact_replay',None))
    replay=getattr(runtime,'_native_contact_replay',None)
    if replay is not None:
        replay.assert_target(runtime._data_root,runtime.plan,'prepared-context-text')
        replay.assert_all_contacts_observed()
    root=runtime._run_dir()/'tfgrpo'
    require(root.resolve().is_relative_to(Path(runtime._data_root).resolve()),'owned fusion artifacts required')
    caller=getattr(runtime,'_tfgrpo_caller',None)
    if caller is None:
        require(not any((root/'contacts').glob('*.intent.json')),
                'prior fusion contacts require registered recovery')
        caller=bind_runtime(runtime,receipt_root=root/'contacts')
        runtime._client,runtime._tfgrpo_caller=caller.client,caller
    else:
        require(caller.root.resolve()==(root/'contacts').resolve()
                and caller.registration==reg and caller.plan==runtime.plan
                and caller.client is getattr(runtime,'_client',None) and not caller.client._closed,
                'live fusion client changed')
    learner=ExperienceLearner(root/'learner',model_id=caller.model_id,metered_call=caller,
        learning_config_sha256=reg['learning_config_sha256'],preflight=caller.preflight,
        reconcile=caller.reconcile,max_calls=1,concurrency=1)
    return execute_fusion(learner,loaded['binding'],
        verify_source=lambda:load_finalized_shards(loaded['spec']['source'],
            data_root=runtime._data_root,count_tokens=loaded['count_tokens'],
            tokenizer_identity=loaded['tokenizer_identity']),
        validate_output=lambda text:validate_fusion_output(text,loaded['document'],
            max_items=loaded['measurement']['max_items'],
            max_characters=loaded['measurement']['max_characters']))


def dispatch_registered_fusion(runtime):
    started=getattr(runtime,'_tfgrpo_started',None)
    if started is None:
        started=runtime._clock()
        runtime._tfgrpo_started=started
    candidate=run_registered_fusion(runtime)
    return seal_registered_candidate(runtime,candidate,started=started,ended=runtime._clock())
