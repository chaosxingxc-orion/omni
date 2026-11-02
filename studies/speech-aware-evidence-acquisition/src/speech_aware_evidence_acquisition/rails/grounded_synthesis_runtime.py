"""Grounded shard artifacts on the existing metered FlightRuntime path."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .grounded_synthesis_execution import (
    SCHEMA, STAGE, contact_identity, execute_shards, registration_binding,
    validate_custody, require,
)
from .grounded_synthesis_plan import load_token_counter, verify_plan
from .grounded_synthesis_source import load_source
from .tfgrpo_learning import ExperienceLearner, LearningError, sha
from .tfgrpo_source import read_pin
from .tfgrpo_runtime import (
    GROUNDED_RUNTIME_SCHEMA, LearningStageSummary, bind_runtime,
    contact_key, learning_configuration, learning_source_plan,
    validate_registration,
)

SOURCE_SCHEMA = 'saea-prepared-grounded-shards-v1'


def build_registration(plan, *, shard_plan, custody, model_id, decoding_params,
                       max_context_bytes, runtime_identity, roster_sha256,
                       source_batch, max_operations=128):
    """CPU preparation only; resulting config still needs ordinary admission."""
    require(isinstance(source_batch, dict) and source_batch.get('model_id') == model_id,
            'grounded registration source model differs')
    groups = source_batch.get('groups')
    require(isinstance(groups, list) and groups, 'grounded source groups required')
    for shard in shard_plan['shards']:
        indices = shard.get('group_indices')
        require(isinstance(indices, list) and indices
                and all(type(i) is int and 0 <= i < len(groups) for i in indices),
                'grounded shard group index outside source batch')
        require(all(isinstance(groups[i], dict)
                    and isinstance(groups[i].get('query_id'), str)
                    and groups[i]['query_id'].startswith('slurp:train:') for i in indices),
                'grounded shard groups require TRAIN anchors')
    config = {'model_id': model_id, 'decoding_params': decoding_params,
              'max_context_bytes': max_context_bytes,
              'runtime_identity_sha256': sha(runtime_identity)}
    config_sha = sha(learning_configuration(config))
    execution = registration_binding(shard_plan, model_id=model_id,
        learning_config_sha256=config_sha, custody_sha256=sha(custody),
        max_operations=max_operations, max_calls=len(shard_plan['shards']))
    binding = {'schema':SCHEMA, 'registration':execution, 'plan':shard_plan,
               'source_custody':custody}
    contacts = [{'stage':STAGE, 'identity':contact_identity(shard_plan, shard),
                 'sample_id':source_batch['groups'][shard['group_indices'][0]]['query_id'],
                 'messages_sha256':shard['messages_sha256']} for shard in shard_plan['shards']]
    registration = {'schema':GROUNDED_RUNTIME_SCHEMA,
        'plan':{k:v for k,v in asdict(plan).items() if k!='protocol_hash'},
        **config, 'batch_sha256':sha(binding), 'construction_roster_sha256':roster_sha256,
        'contacts':contacts, 'learning_config_sha256':config_sha,
        'execution_sha256':sha(execution), 'completion_cap':shard_plan['completion_cap']}
    validate_registration(registration, plan=plan, model_id=model_id)
    validate_custody(custody, shard_plan, model_id)
    return registration, binding


def load_runtime_source(values, *, plan, repo_root, data_root, sample_ids,
                        roster_sha256, replay=None):
    """Reconstruct immutable source, tokenizer, execution and contact bindings."""
    spec = values.get('prepared_training_feedback')
    require(isinstance(spec, dict) and set(spec)=={'schema','source'}
            and spec['schema']==SOURCE_SCHEMA, 'exact grounded dynamic source required')
    source = read_pin(spec['source'], repo_root=repo_root, data_root=data_root)
    require(isinstance(source, dict) and set(source)=={
        'schema','plan','source','tokenizer_relpath','max_operations','concurrency'}
        and source['schema']=='saea-grounded-runtime-source-v1',
        'exact grounded source document required')
    rel = source['tokenizer_relpath']
    require(isinstance(rel,str) and rel and not Path(rel).is_absolute()
        and '..' not in Path(rel).parts, 'confined tokenizer path required')
    tokenizer = (Path(data_root)/rel).resolve()
    require(tokenizer.is_relative_to(Path(data_root).resolve()), 'tokenizer escapes data root')
    count_tokens, tokenizer_identity = load_token_counter(tokenizer)
    custody = load_source(source['source'], data_root=data_root)
    paths = source['source']
    bb = (Path(data_root)/paths['source_batch']['relpath']).read_bytes()
    ob = (Path(data_root)/paths['operations']['relpath']).read_bytes()
    shard_plan = source['plan']
    verify_plan(shard_plan, bb, ob, count_tokens=count_tokens, tokenizer_identity=tokenizer_identity)
    registration = values['tfgrpo_learning_registration']
    scientific = learning_source_plan(plan, replay=replay, data_root=data_root)
    # Runtime identity is an independently validated digest, not a re-created object.
    validate_registration(registration, plan=scientific, model_id=custody['model_id'])
    execution = registration_binding(shard_plan, model_id=custody['model_id'],
        learning_config_sha256=registration['learning_config_sha256'],
        custody_sha256=sha(custody), max_operations=source['max_operations'],
        max_calls=len(shard_plan['shards']))
    binding = {'schema':SCHEMA, 'registration':execution, 'plan':shard_plan,
               'source_custody':custody}
    batch = json.loads(bb)
    expected_contacts = [{'stage':STAGE, 'identity':contact_identity(shard_plan,s),
        'sample_id':batch['groups'][s['group_indices'][0]]['query_id'],
        'messages_sha256':s['messages_sha256']} for s in shard_plan['shards']]
    require(registration['schema']==GROUNDED_RUNTIME_SCHEMA
        and registration['batch_sha256']==sha(binding)
        and registration['execution_sha256']==sha(execution)
        and registration['construction_roster_sha256']==roster_sha256
        and registration['completion_cap']==shard_plan['completion_cap']
        and registration['contacts']==expected_contacts
        and set(sample_ids)=={row['sample_id'] for row in expected_contacts},
        'grounded source differs from admitted runtime binding')
    require(type(source['concurrency']) is int and 0 < source['concurrency'] <= len(expected_contacts),
            'bounded shard concurrency required')
    validate_custody(custody, shard_plan, custody['model_id'])
    loaded = {'source':source, 'binding':binding, 'batch_bytes':bb, 'operation_bytes':ob,
              'count_tokens':count_tokens, 'tokenizer_identity':tokenizer_identity}
    receipt = {'schema':'saea-grounded-runtime-source-receipt-v1', 'source':spec['source'],
        'batch_sha256':sha(binding), 'contacts':len(expected_contacts), 'rows':len(sample_ids),
        'construction_roster_sha256':roster_sha256, 'dynamic_states':True,
        'candidate_publishable':False}
    return loaded, receipt


def run_registered_shards(runtime):
    """Bind actual gate/client and retain that client for ordinary artifact seal."""
    values = runtime._composed.values
    registration = values['tfgrpo_learning_registration']
    loaded, _ = load_runtime_source(values, plan=runtime.plan,
        repo_root=runtime._repo_root, data_root=runtime._data_root,
        sample_ids=tuple(s.sample_id for s in runtime._samples),
        roster_sha256=registration['construction_roster_sha256'],
        replay=getattr(runtime,'_native_contact_replay',None))
    replay = getattr(runtime, '_native_contact_replay', None)
    if replay is not None:
        # New attempts have no local learner intents: inspect the entire native
        # predecessor before constructing a caller or dispatching parallel FITs.
        replay.assert_target(runtime._data_root, runtime.plan, 'prepared-context-text')
        replay.assert_all_contacts_observed()
    root = runtime._run_dir()/'tfgrpo'
    require(root.resolve().is_relative_to(Path(runtime._data_root).resolve()),
            'grounded artifacts must remain under runtime data root')
    caller = getattr(runtime, '_tfgrpo_caller', None)
    if caller is None:
        require(not any((root/'contacts').glob('*.intent.json')),
                'prior grounded contacts require registered recovery; refusing a fresh client')
        caller = bind_runtime(runtime, receipt_root=root/'contacts')
        runtime._client, runtime._tfgrpo_caller = caller.client, caller
    else:
        require(caller.root.resolve()==(root/'contacts').resolve()
            and caller.registration==registration and caller.plan==runtime.plan
            and caller.client is getattr(runtime,'_client',None) and not caller.client._closed,
            'live grounded recovery client or registration changed')
    binding = loaded['binding']
    learner = ExperienceLearner(root/'learner', model_id=caller.model_id,
        metered_call=caller, learning_config_sha256=registration['learning_config_sha256'],
        preflight=caller.preflight, reconcile=caller.reconcile,
        max_calls=len(registration['contacts']), max_operations=loaded['source']['max_operations'],
        concurrency=loaded['source']['concurrency'])
    result = execute_shards(learner, binding['plan'], loaded['batch_bytes'], loaded['operation_bytes'],
        registration=binding['registration'], count_tokens=loaded['count_tokens'],
        tokenizer_identity=loaded['tokenizer_identity'],
        verify_source_custody=lambda: load_source(loaded['source']['source'],data_root=runtime._data_root))
    return result


def dispatch_registered_shards(runtime):
    """Seal intermediate shard output separately from an experience candidate."""
    from ..core.tracing import TraceBuffer
    started = getattr(runtime, '_tfgrpo_started', None)
    if started is None:
        started = runtime._clock()
        runtime._tfgrpo_started = started
    result = run_registered_shards(runtime)
    ended = runtime._clock()
    runtime._contact_window = (started, ended)
    caller = runtime._tfgrpo_caller
    observed, rows = {}, {}
    for contact in caller.registration['contacts']:
        key = contact_key(contact['stage'],contact['identity'])
        request = json.loads((caller.root/(key+'.intent.json')).read_bytes())['request']
        response = caller.observed(request)
        rid = response['receipt_id'].removeprefix(f'{caller.plan.run_id}/{caller.plan.attempt_id}/')
        require(rid not in observed, 'duplicate grounded metering receipt')
        observed[rid] = (contact,response)
    timings = caller.client.timings
    require({t.request_id for t in timings}==set(observed), 'grounded timing custody differs')
    for timing in timings:
        contact,response = observed[timing.request_id]
        require(timing.sample_id==contact['sample_id'] and timing.response_sha256==response['response_sha256'],
                'grounded response timing differs')
        rows[timing.sample_id] = {'sample_id':timing.sample_id,'response_text':response['text']}
    require(set(rows)=={s.sample_id for s in runtime._samples}, 'grounded output roster differs')
    usage = {k:sum(response['usage'][k] for _,response in observed.values())
             for k in ('calls','prompt_tokens','completion_tokens')}
    require(usage==result['current_accounting']['known_usage'], 'grounded usage differs from native receipts')
    path = runtime._run_dir()/'tfgrpo/learner/shards-complete.json'
    require(json.loads(path.read_bytes())==result, 'grounded artifact bytes differ')
    artifact = {'root':'data','relpath':str(path.resolve().relative_to(Path(runtime._data_root).resolve())),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'candidate_publishable':False}
    runtime._output_rows = [rows[s.sample_id] for s in runtime._samples]
    runtime._phase_summaries = (LearningStageSummary({'schema':'saea-grounded-shards-summary-v1',
        'phase':STAGE,'result_kind':'artifact','contacts':len(observed),'rows':len(rows),
        'artifact':artifact,'usage':usage}),)
    trace_binding = {'schema':'saea-grounded-shards-flight-artifact-v1', 'artifact':artifact,
        'batch_sha256':caller.registration['batch_sha256'],'usage':usage,
        'historical_usage':result['historical_usage'],'candidate_publishable':False}
    buffer = TraceBuffer(caller.plan.run_id)
    buffer.emit('ORG',caller.plan.run_id+'/grounded-shards',{'org_grounded_shards':trace_binding})
    caller.client.flush_trace_buffer(buffer)
    wall = max(ended-started,1e-9)
    payload = {'dispatch':'registered-grounded-shards','policy':runtime._composed.values['policy'],
        'result_kind':'artifact','artifact_result':dict(runtime._artifact_result_binding),
        'contact_count':len(observed),'outputs_row_count':len(rows),'wall_seconds':wall,
        'contacts_per_second':len(observed)/wall,'tokens_per_second':usage['completion_tokens']/wall,
        'tokens_per_second_basis':'actual_metered_completion_tokens',
        'average_latency_seconds':sum(t.latency_seconds for t in timings)/len(timings),
        'average_seconds_per_dataset_row':wall/len(rows),'processed_dataset_rows':len(rows),
        'total_dataset_rows':len(runtime._samples),'remaining_dataset_rows':0,'estimated_remaining_seconds':0,
        'input_mode':'text-only-zero-audio','contact_window':{'from':started,'to':ended},
        'learning':trace_binding,'candidate_publishable':False,
        'phases':[{'phase':STAGE,'contacts':len(observed)}]}
    from ..core.flight.receipts import require_mandatory_throughput_fields
    require_mandatory_throughput_fields('run_phases',payload,('contacts_per_second','tokens_per_second'))
    return runtime.receipts.record('run_phases',payload)
