"""Finite grounded fusion through the shared registered native FIT/client."""
import json
from dataclasses import asdict
from pathlib import Path

from .grounded_synthesis_plan import load_token_counter
from .grounded_synthesis_fusion import (
    load_finalized_shards, build_fusion_input, fusion_messages, validate_fusion_output,
)
from .grounded_corrective_fusion import (
    build_corrective_input, corrective_messages, execute_corrective_fusion,
    bind_failed_response, build_corrective_execution,
)
from .grounded_fusion_execution import SCHEMA, STAGE, execute_fusion, require
from .tfgrpo_source import read_pin
from .tfgrpo_learning import ExperienceLearner, sha, DENSE_REWARD
from .tfgrpo_runtime import (
    FUSION_RUNTIME_SCHEMA, learning_configuration, validate_registration,
    learning_source_plan, bind_runtime, seal_registered_candidate,
)

SOURCE_SCHEMA = 'saea-prepared-grounded-fusion-v1'
CORRECTIVE_STRATEGY = 'corrective-v1'
FAILED_SOURCE_SCHEMA = 'saea-grounded-failed-fusion-source-v1'


def load_failed_fusion(spec, *, data_root, source, document):
    """Re-derive known ABORTED custody from pinned on-disk native records.

    No caller-supplied verified flags are trusted. The first version deliberately
    refuses recovery ancestry instead of silently dropping its possible costs.
    """
    from ..contracts import ExecutionPlan
    from ..core.contact_journal import ContactJournal, comparable_request
    from ..core.preparation import load_artifact
    from .tfgrpo_consumption import checked_pin
    require(isinstance(spec, dict) and set(spec) == {
        'schema', 'source_plan', 'attempt', 'execution_batch', 'registration',
        'learner_files', 'bridge_files', 'journal_files', 'fit_artifact', 'failure_reason'}
        and spec['schema'] == FAILED_SOURCE_SCHEMA, 'exact failed fusion source required')
    root = Path(data_root).resolve()
    plan = ExecutionPlan(**spec['source_plan']).validate()
    require(plan.planned_model_calls == 1 and plan.split_role == 'discovery',
            'one original TRAIN fusion contact required')
    base = root/'runs'/plan.run_id/plan.attempt_id
    require(not (base/'resume-declaration.json').exists()
            and not (base/'tfgrpo/learner/candidate.json').exists(),
            'failed corrective source cannot import recovery ancestry or candidate')
    def read(pin, expected=None):
        path = checked_pin(root, pin)
        require(expected is None or path == expected.resolve(), 'failed source location differs')
        return path, json.loads(path.read_bytes())
    _, attempt = read(spec['attempt'], root/'attempts'/plan.run_id/(plan.attempt_id+'.json'))
    require(attempt.get('state') == 'ABORTED' and attempt.get('contact_kind') == 'prepared-text'
            and attempt.get('run_id') == plan.run_id and attempt.get('attempt_id') == plan.attempt_id
            and attempt.get('carrier_lock_key') == plan.carrier_lock_key
            and attempt.get('actual') == {'calls': 1, 'audio_seconds': 0.0}
            and attempt.get('reserved') == {'calls': 1, 'audio_seconds': float(plan.planned_speech_audio_seconds)},
            'failed fusion must be terminal ABORTED with exact one-call spend')
    _, binding = read(spec['execution_batch'], base/'tfgrpo/learner/batch.json')
    _, reg = read(spec['registration'], base/'tfgrpo/contacts/registration.json')
    validate_registration(reg, plan=plan, model_id=source['model_id'])
    require(binding['source_custody_sha256'] == sha(source)
            and binding['input_sha256'] == document['input_sha256']
            and binding['identity'] == document['input_sha256']
            and reg['batch_sha256'] == reg['execution_sha256'] == sha(binding)
            and binding['learning_config_sha256'] == reg['learning_config_sha256']
            and binding['parent'] == source['batch']['parent']
            and binding['rollout_kind'] == source['batch']['rollout_kind']
            and binding['reward_mode'] == source['batch']['common'].get('reward_mode', DENSE_REWARD)
            and binding['learned_query_ids'] == [g['query_id'] for g in source['batch']['groups']]
            and reg['contacts'] == [{'stage': STAGE, 'identity': binding['identity'],
                'sample_id': binding['learned_query_ids'][0], 'messages_sha256': sha(binding['messages'])}],
            'failed fusion registration/source identity differs')
    key = sha({'stage': STAGE, 'identity': binding['identity']})
    def inventory(pins, directory, suffixes, expected_count):
        require(isinstance(pins, list) and len(pins) == expected_count, 'complete failed inventory required')
        paths = [read(pin)[0] for pin in pins]
        actual = {p for suffix in suffixes for p in directory.glob('*'+suffix)}
        require(len(set(paths)) == len(paths) and set(paths) == actual,
                'failed native/bridge inventory differs, including unknown or orphan contacts')
    local, bridge = base/'tfgrpo/learner/contacts', base/'tfgrpo/contacts'
    inventory(spec['learner_files'], local, ('.intent.json', '.response.json'), 2)
    inventory(spec['bridge_files'], bridge, ('.intent.json', '.response.json', '.fit.json'), 3)
    def local_read(directory, suffix):
        path = directory/(key+suffix)
        require(path.exists(), 'failed contact filename differs')
        return json.loads(path.read_bytes())
    request = local_read(local, '.intent.json')
    envelope = local_read(local, '.response.json')
    bridge_intent = local_read(bridge, '.intent.json')
    bridge_response = local_read(bridge, '.response.json')
    fit = local_read(bridge, '.fit.json')
    receipt = envelope['response']
    prefix = f'{plan.run_id}/{plan.attempt_id}/'
    require(receipt['receipt_id'].startswith(prefix), 'failed receipt attempt differs')
    rid = receipt['receipt_id'][len(prefix):]
    journal = ContactJournal(root, plan, 'prepared-context-text')
    inventory(spec['journal_files'], journal.root, ('.intent.json', '.response.json'), 2)
    native = journal.inspect(rid)
    require(native['state'] == 'observed'
            and set(journal.root.glob('*.intent.json')) == {journal._path(rid, 'intent')}
            and set(journal.root.glob('*.response.json')) == {journal._path(rid, 'response')},
            'exact one observed native failed contact required')
    payload = native['intent']['payload']
    require(bridge_intent.get('request') == request
            and comparable_request(bridge_intent['payload']) == comparable_request(payload)
            and payload.get('sample_id') == binding['learned_query_ids'][0]
            and payload.get('prepared_text') == 'saea-prepared-text-v1'
            and native['intent'].get('agent_contact_id') == 'tfgrpo-'+key
            and bridge_response == {'request_sha256': sha(request), 'response': receipt},
            'failed bridge/native response or TRAIN anchor differs')
    require(receipt.get('fit_receipt_sha256') == sha(fit)
            and fit.get('payload_sha256') == sha(bridge_intent['payload'])
            and fit.get('every_request_fits') is True
            and spec['fit_artifact'] == {'relpath': fit['artifact_relpath'], 'sha256': fit['artifact_sha256']},
            'failed native FIT receipt differs')
    fit_path, fit_doc = read(spec['fit_artifact'])
    load_artifact(fit_path, expected_sha256=spec['fit_artifact']['sha256'],
                  kind='request-fit', inputs=fit_doc['inputs'])
    require(fit_doc['inputs'].get('payload_sha256') == sha(bridge_intent['payload'])
            and fit_doc['output'].get('every_request_fits') is True, 'failed native FIT artifact differs')
    failure = bind_failed_response(binding, request, envelope, native['intent'], native['response'],
        historical_usage=source['original_total_usage'], failure_reason=spec['failure_reason'])
    terminal = {'schema': 'saea-grounded-failed-fusion-custody-v1',
        'source_sha256': sha(spec), 'failure_sha256': failure['failure_sha256'],
        'attempt_state': 'ABORTED', 'observed_contacts': 1, 'unknown_contacts': 0,
        'full_inventory_verified': True, 'native_fit_verified': True}
    return failure, terminal


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
    if isinstance(doc, dict) and doc.get('strategy') == CORRECTIVE_STRATEGY:
        return load_corrective_runtime_source(values, plan=plan, repo_root=repo_root,
            data_root=data_root, sample_ids=sample_ids, roster_sha256=roster_sha256, replay=replay)
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


def build_corrective_registration(plan, *, source, corrective, terminal, measured, model_id,
                                  decoding_params, max_context_bytes,
                                  runtime_identity_sha256, roster_sha256):
    require(source['model_id'] == model_id and plan.planned_model_calls == 1
            and measured['completion_cap'] == decoding_params['max_tokens'],
            'same-model single corrective contact required')
    config = {'model_id': model_id, 'decoding_params': decoding_params,
              'max_context_bytes': max_context_bytes,
              'runtime_identity_sha256': runtime_identity_sha256}
    config_sha = sha(learning_configuration(config))
    binding, custody = build_corrective_execution(corrective, measured, source=source,
        terminal=terminal, learning_config_sha256=config_sha)
    ids = binding['learned_query_ids']
    reg = {'schema': FUSION_RUNTIME_SCHEMA,
        'plan': {k:v for k,v in asdict(plan).items() if k != 'protocol_hash'}, **config,
        'batch_sha256': sha(binding), 'execution_sha256': sha(binding),
        'construction_roster_sha256': roster_sha256,
        'learning_config_sha256': config_sha, 'completion_cap': measured['completion_cap'],
        'contacts': [{'stage': STAGE, 'identity': binding['identity'],
                      'sample_id': ids[0], 'messages_sha256': sha(measured['messages'])}]}
    validate_registration(reg, plan=plan, model_id=model_id)
    return reg, binding, custody


def load_corrective_runtime_source(values, *, plan, repo_root, data_root,
                                   sample_ids, roster_sha256, replay=None):
    spec = values.get('prepared_training_feedback')
    require(isinstance(spec, dict) and set(spec) == {'schema', 'source'}
            and spec['schema'] == SOURCE_SCHEMA, 'exact fusion dynamic source required')
    doc = read_pin(spec['source'], repo_root=repo_root, data_root=data_root)
    require(isinstance(doc, dict) and set(doc) == {
        'schema', 'strategy', 'source', 'failed_fusion', 'measurement',
        'input_sha256', 'corrective_input_sha256', 'tokenizer_relpath'}
        and doc['schema'] == 'saea-grounded-fusion-runtime-source-v1'
        and doc['strategy'] == CORRECTIVE_STRATEGY, 'exact corrective source document required')
    rel = doc['tokenizer_relpath']
    require(isinstance(rel, str) and rel and not Path(rel).is_absolute()
            and '..' not in Path(rel).parts, 'confined corrective tokenizer required')
    path = (Path(data_root) / rel).resolve()
    require(path.is_relative_to(Path(data_root).resolve()), 'tokenizer escapes root')
    count, identity = load_token_counter(path)
    source = load_finalized_shards(doc['source'], data_root=data_root,
                                   count_tokens=count, tokenizer_identity=identity)
    document = build_fusion_input(source)
    failure, terminal = load_failed_fusion(doc['failed_fusion'], data_root=data_root,
                                           source=source, document=document)
    corrective = build_corrective_input(document, failure)
    require(document['input_sha256'] == doc['input_sha256']
            and corrective['corrective_input_sha256'] == doc['corrective_input_sha256'],
            'corrective input derivation differs')
    old = doc['measurement']
    measured = corrective_messages(corrective, count_tokens=count,
        **{k:old[k] for k in ('resident_context','completion_cap','reserve_tokens',
                              'max_items','max_characters')})
    require(measured == old, 'corrective measured request changed')
    reg = values['tfgrpo_learning_registration']
    scientific = learning_source_plan(plan, replay=replay, data_root=data_root)
    expected, binding, custody = build_corrective_registration(scientific, source=source,
        corrective=corrective, terminal=terminal, measured=measured, model_id=source['model_id'],
        decoding_params=reg['decoding_params'], max_context_bytes=reg['max_context_bytes'],
        runtime_identity_sha256=reg['runtime_identity_sha256'], roster_sha256=roster_sha256)
    require(reg == expected and tuple(sample_ids) == (reg['contacts'][0]['sample_id'],),
            'corrective registration or roster differs')
    return {'source': source, 'document': document, 'corrective': corrective,
            'measurement': measured, 'binding': binding, 'spec': doc,
            'count_tokens': count, 'tokenizer_identity': identity,
            'failed_attempt': terminal, 'custody': custody, 'strategy': CORRECTIVE_STRATEGY}, {
        'schema': 'saea-grounded-fusion-source-receipt-v1', 'source': spec['source'],
        'batch_sha256': sha(binding), 'contacts': 1, 'rows': 1,
        'construction_roster_sha256': roster_sha256, 'dynamic_states': True,
        'semantic_review_required': True, 'strategy': CORRECTIVE_STRATEGY}


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
    if loaded.get('strategy') == CORRECTIVE_STRATEGY:
        def verify_failed():
            failure, terminal = load_failed_fusion(loaded['spec']['failed_fusion'],
                data_root=runtime._data_root, source=loaded['source'], document=loaded['document'])
            require(failure == loaded['corrective']['failure'], 'failed fusion changed before contact')
            return terminal
        return execute_corrective_fusion(learner, loaded['corrective'], loaded['measurement'],
            count_tokens=loaded['count_tokens'],
            verify_source=lambda: load_finalized_shards(loaded['spec']['source'],
                data_root=runtime._data_root, count_tokens=loaded['count_tokens'],
                tokenizer_identity=loaded['tokenizer_identity']),
            verify_failed_attempt=verify_failed)
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
