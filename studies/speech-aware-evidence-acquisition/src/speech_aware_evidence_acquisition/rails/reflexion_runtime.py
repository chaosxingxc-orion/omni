"""Bounded Reflexion dynamic stages on the shared learner, native FIT and Flight.

Only stage identities/dependencies are pre-registered. Later prompt bytes do
not exist until predecessors are durably observed and validated.
"""
import copy
import json
import re
from dataclasses import asdict
from pathlib import Path

from . import reflexion_offline as method
from .grounded_synthesis_fusion import load_finalized_shards
from .grounded_synthesis_plan import load_token_counter
from .tfgrpo_learning import ExperienceLearner, UnknownLearningContact, sha
from .tfgrpo_runtime import (REFLEXION_RUNTIME_SCHEMA, bind_runtime, contact_key,
    learning_configuration, learning_source_plan, seal_registered_candidate)
from .tfgrpo_source import read_pin

SOURCE_SCHEMA = 'saea-prepared-reflexion-dynamic-v1'
RUNTIME_SOURCE_SCHEMA = 'saea-reflexion-runtime-source-v1'
require = method.require


def contact_roster(document):
    rows = []
    for index, group in enumerate(document['groups']):
        rows.append({'stage': method.REFLECT, 'identity': f'{document["input_sha256"]}:group-{index}',
            'sample_id': group['query_id'], 'group_index': index,
            'depends_on': contact_key(rows[-1]['stage'], rows[-1]['identity']) if rows else None})
    return rows


def validate_registration(reg, *, plan, model_id):
    fields = {'schema', 'plan', 'model_id', 'batch_sha256', 'execution_sha256',
        'input_sha256', 'construction_roster_sha256', 'contacts', 'decoding_params',
        'max_context_bytes', 'learning_config_sha256', 'runtime_identity_sha256',
        'completion_cap', 'bounds', 'prompt_derivation'}
    require(isinstance(reg, dict) and set(reg) == fields and reg['schema'] == REFLEXION_RUNTIME_SCHEMA,
            'exact Reflexion runtime registration required')
    require(reg['plan'] == {k:v for k,v in asdict(plan).items() if k != 'protocol_hash'}
        and plan.split_role == 'discovery' and plan.execution_profile == 'bounded-discovery-probe'
        and reg['model_id'] == model_id, 'Reflexion requires exact TRAIN discovery plan/model')
    for name in ('batch_sha256', 'execution_sha256', 'input_sha256',
                 'construction_roster_sha256', 'runtime_identity_sha256', 'learning_config_sha256'):
        require(isinstance(reg[name], str) and re.fullmatch('[a-f0-9]{64}', reg[name]), 'pinned Reflexion digests required')
    require(reg['execution_sha256'] == reg['batch_sha256']
        and reg['prompt_derivation'] == 'observed-prefix-only-v1', 'Reflexion deterministic derivation required')
    bounds = reg['bounds']
    bound_fields = {'resident_context', 'completion_cap',
        'reserve_tokens', 'max_groups', 'max_items', 'max_characters', 'max_entry_characters',
        'max_prompt_tokens_total', 'max_completion_tokens_total', 'max_calls'}
    require(isinstance(bounds, dict) and set(bounds) in (bound_fields, bound_fields | {'reflector_contract'})
        and ('reflector_contract' not in bounds or bounds['reflector_contract'] in
             (method.COMPACT_CONTRACT, method.ERROR_FEEDBACK_CONTRACT))
        and all(type(bounds[k]) is int and bounds[k] > 0 for k in bound_fields if k != 'reserve_tokens')
        and type(bounds['reserve_tokens']) is int and bounds['reserve_tokens'] >= 0, 'exact finite Reflexion bounds required')
    decoding = reg['decoding_params']
    require(isinstance(decoding, dict) and 'id_slot' not in decoding
        and type(decoding.get('max_tokens')) is int and decoding['max_tokens'] > 0
        and reg['completion_cap'] == bounds['completion_cap'] == decoding['max_tokens']
        and type(reg['max_context_bytes']) is int and reg['max_context_bytes'] > 0, 'Reflexion output/context bound differs')
    rows = reg['contacts']; indexed = {}; seen = set()
    require(isinstance(rows, list) and len(rows) == plan.planned_model_calls == bounds['max_calls']
        and 0 < len(rows) <= bounds['max_groups']
        and len(rows)*bounds['completion_cap'] <= bounds['max_completion_tokens_total'], 'Reflexion finite one-stage allocation differs')
    for position, row in enumerate(rows):
        stage = method.REFLECT; index = position
        require(isinstance(row, dict) and set(row) == {'stage','identity','sample_id','group_index','depends_on'}
            and row['stage'] == stage and type(row['group_index']) is int and row['group_index'] == index
            and row['identity'] == f'{reg["input_sha256"]}:group-{index}'
            and isinstance(row['sample_id'], str) and row['sample_id'].startswith('slurp:train:')
            and row['depends_on'] == (contact_key(rows[position-1]['stage'], rows[position-1]['identity']) if position else None),
            'Reflexion ordered TRAIN dependency differs')
        require(row['sample_id'] not in seen, 'duplicate Reflexion TRAIN group')
        seen.add(row['sample_id'])
        indexed[contact_key(stage, row['identity'])] = row
    require(reg['learning_config_sha256'] == sha(learning_configuration(reg)), 'Reflexion generation configuration changed')
    return indexed


def build_registration(plan, *, source, document, decoding_params, max_context_bytes,
                       runtime_identity_sha256, roster_sha256):
    method.sealed(document, 'input_sha256')
    # Cross-parent costs need a separate native ancestry adapter; do not claim them here.
    require(document['parent'] == source['batch']['parent'], 'Reflexion runtime requires original verified parent ancestry')
    config = {'model_id': source['model_id'], 'decoding_params': decoding_params,
        'max_context_bytes': max_context_bytes, 'runtime_identity_sha256': runtime_identity_sha256}
    require(document == method.build_input(source, source['batch']['parent'],
        learning_config_sha256=sha(learning_configuration(config)),
        **{k:v for k,v in document['bounds'].items() if k != 'max_calls'}), 'Reflexion source/document/config derivation differs')
    reg = {'schema': REFLEXION_RUNTIME_SCHEMA, 'plan': {k:v for k,v in asdict(plan).items() if k != 'protocol_hash'},
        **config, 'batch_sha256': sha(document), 'execution_sha256': sha(document),
        'input_sha256': document['input_sha256'], 'construction_roster_sha256': roster_sha256,
        'learning_config_sha256': document['learning_config_sha256'],
        'completion_cap': document['bounds']['completion_cap'], 'bounds': copy.deepcopy(document['bounds']),
        'prompt_derivation': 'observed-prefix-only-v1', 'contacts': contact_roster(document)}
    validate_registration(reg, plan=plan, model_id=source['model_id'])
    return reg


def load_runtime_source(values, *, plan, repo_root, data_root, sample_ids, roster_sha256, replay=None):
    scientific = learning_source_plan(plan,replay=replay,data_root=data_root)
    spec = values.get('prepared_training_feedback')
    require(isinstance(spec, dict) and set(spec) == {'schema','source'} and spec['schema'] == SOURCE_SCHEMA,
            'exact Reflexion source pin required')
    doc = read_pin(spec['source'], repo_root=repo_root, data_root=data_root)
    require(isinstance(doc, dict) and set(doc) == {'schema','strategy','source','tokenizer_relpath',
        'tokenizer_identity','document','initial_measurement','phase_contract'}
        and doc['schema'] == RUNTIME_SOURCE_SCHEMA and doc['strategy'] == method.STRATEGY
        and doc['phase_contract'] == 'unknown-prompts-await-observed-predecessor', 'exact Reflexion runtime source required')
    rel = doc['tokenizer_relpath']; root = Path(data_root).resolve()
    require(isinstance(rel,str) and rel and not Path(rel).is_absolute() and '..' not in Path(rel).parts
        and (root/rel).resolve().is_relative_to(root), 'confined Reflexion tokenizer required')
    count, identity = load_token_counter(root/rel)
    require(identity == doc['tokenizer_identity'], 'Reflexion tokenizer identity changed')
    source = load_finalized_shards(doc['source'], data_root=root, count_tokens=count, tokenizer_identity=identity)
    document = doc['document']; reg = values['tfgrpo_learning_registration']
    expected = build_registration(scientific, source=source, document=document, decoding_params=reg['decoding_params'],
        max_context_bytes=reg['max_context_bytes'], runtime_identity_sha256=reg['runtime_identity_sha256'],
        roster_sha256=roster_sha256)
    require(reg == expected and set(sample_ids) == {g['query_id'] for g in document['groups']}
        and len(sample_ids) == len(document['groups']), 'Reflexion registration/roster differs')
    initial = method.reflect_messages(document, 0, method.empty_memory(), count)
    require(initial == doc['initial_measurement'], 'Reflexion first prompt measurement changed')
    return {'source':source, 'document':document, 'count_tokens':count, 'tokenizer_identity':identity, 'spec':doc}, {
        'schema':'saea-reflexion-source-receipt-v1', 'source':spec['source'], 'batch_sha256':sha(document),
        'contacts':len(reg['contacts']), 'rows':len(sample_ids), 'construction_roster_sha256':roster_sha256,
        'dynamic_states':True, 'native_fit_completed':False,
        'first_phase':'AWAITING_NATIVE_FIT', 'subsequent_phases':'AWAITING_OBSERVED_PREDECESSOR'}


def _request(document, row, measured):
    return {'schema':'saea-tfgrpo-learning-request-v1', 'batch_sha256':sha(document),
        'model_id':document['model_id'], 'learning_config_sha256':document['learning_config_sha256'],
        'stage':row['stage'], 'identity':row['identity'], 'messages':measured['messages']}


def request_validator(document, count_tokens):
    """Rebuild each realized prompt from exact observed bridge predecessors."""
    rows = contact_roster(document)
    counts = {}
    def count_once(messages):
        digest = sha(messages)
        if digest not in counts: counts[digest] = count_tokens(messages)
        return counts[digest]
    def validate(request, caller):
        target = contact_key(request.get('stage'), request.get('identity'))
        keys = [contact_key(row['stage'],row['identity']) for row in rows]
        require(target in keys, 'Reflexion request is outside finite roster')
        intents = {p.name.removesuffix('.intent.json') for p in caller.root.glob('*.intent.json')}
        responses = {p.name.removesuffix('.response.json') for p in caller.root.glob('*.response.json')}
        require(intents == set(keys[:len(intents)]) and responses <= intents, 'Reflexion bridge history is foreign or nonsequential')
        memory = method.empty_memory()
        for row in rows:
            index = row['group_index']; group = document['groups'][index]
            key = contact_key(row['stage'],row['identity'])
            measured = method.reflect_messages(document,index,memory,count_once)
            expected = _request(document,row,measured)
            if key == target:
                require(request == expected, 'Reflexion prompt differs from observed predecessor derivation')
                return
            if key not in responses:
                raise UnknownLearningContact('Reflexion predecessor unobserved; no later prompt FIT or generation')
            observed = caller.observed(expected)
            reflection = method.validate_reflection(observed['text'],group,document['bounds']['max_entry_characters'],
                document['bounds'].get('reflector_contract'))
            memory, _ = method.advance_memory(memory,reflection,group,index,document['bounds'])
    return validate


def enforce_native_budget(caller, document):
    """Check real wrapped prompt tokens before any generation intent exists."""
    original = caller.request_fit; bounds = document['bounds']
    def gate(prepared):
        receipt = original(prepared)
        tokens = receipt.get('measured_prompt_tokens')
        require(type(tokens) is int and tokens > 0, 'Reflexion requires measured native wrapped prompt tokens')
        require(tokens+bounds['completion_cap']+bounds['reserve_tokens'] <= bounds['resident_context'],
            'Reflexion native wrapped prompt exceeds reserved context')
        totals = dict.fromkeys(method.USAGE,0)
        for path in caller.root.glob('*.intent.json'):
            key = path.name.removesuffix('.intent.json')
            if not (caller.root/(key+'.response.json')).exists():
                raise UnknownLearningContact('Reflexion bridge has unknown spend; hold next contact')
            observed = caller.observed(json.loads(path.read_bytes())['request'])
            require(observed['usage']['completion_tokens'] <= bounds['completion_cap'], 'Reflexion prior output cap exceeded')
            for field in totals: totals[field] += observed['usage'][field]
        require(totals['calls'] < bounds['max_calls']
            and totals['prompt_tokens']+tokens <= bounds['max_prompt_tokens_total']
            and totals['completion_tokens']+bounds['completion_cap'] <= bounds['max_completion_tokens_total'],
            'Reflexion native total token budget exhausted before generation')
        return receipt
    caller.request_fit = gate


def verify_recovery_prefix(replay, *, data_root, plan, registration, document, count_tokens):
    """Audit known native prefix and original FIT before opening a fresh client.

    A durable learner/bridge intent without a native observation is held. This
    never creates a receipt or repairs unknown contacts by resending them.
    """
    from types import SimpleNamespace
    from ..core.contact_journal import comparable_request
    from ..core.preparation import load_artifact
    from .tfgrpo_consumption import checked_pin
    replay.assert_target(data_root,plan,'prepared-context-text')
    replay.assert_all_contacts_observed()
    base=Path(data_root)/'runs'/replay.source_plan.run_id/replay.source_plan.attempt_id/'tfgrpo'
    bridge=base/'contacts'; learner=base/'learner'
    require(json.loads((bridge/'registration.json').read_bytes()) == registration
        and json.loads((learner/'batch.json').read_bytes()) == document, 'Reflexion recovery registration/batch changed')
    rows=contact_roster(document); keys=[contact_key(r['stage'],r['identity']) for r in rows]
    native={}
    for rid in replay.request_ids:
        record=replay.source.inspect(rid); payload=record['intent']['payload']
        key=record['intent']['agent_contact_id'].removeprefix('tfgrpo-')
        require(key in keys and key not in native and record['intent']['agent_contact_id']=='tfgrpo-'+key,
                'Reflexion recovery contains foreign/duplicate native stage')
        native[key]=record
    prefix=set(keys[:len(native)])
    require(set(native)==prefix,'Reflexion recovery native stages are not a sequential prefix')
    for directory in (bridge,learner/'contacts'):
        intents={p.name.removesuffix('.intent.json') for p in directory.glob('*.intent.json')}
        responses={p.name.removesuffix('.response.json') for p in directory.glob('*.response.json')}
        require(intents==prefix and responses<=prefix,'Reflexion recovery has unknown, foreign or orphan local contact')
    require({p.name.removesuffix('.fit.json') for p in bridge.glob('*.fit.json')}==prefix,
            'Reflexion recovery requires exact original FIT inventory')
    usage=dict.fromkeys(method.USAGE,0)
    def observed(request):
        key=contact_key(request['stage'],request['identity']); record=native[key]
        payload=record['intent']['payload']; body=json.loads(record['response_bytes'])
        old=json.loads((bridge/(key+'.intent.json')).read_bytes())
        require(old['request']==request and comparable_request(old['payload'])==comparable_request(payload)
            and json.loads((learner/'contacts'/(key+'.intent.json')).read_bytes())==request,
            'Reflexion recovery learner/bridge/native request changed')
        context=json.loads(payload['history'][0]['content'])
        row=rows[keys.index(key)]
        require(payload['sample_id']==row['sample_id'] and context['source_messages']==request['messages']
            and context['stage']==row['stage'] and body.get('finish_reason')=='stop',
            'Reflexion recovery native prompt/anchor/terminal response differs')
        fit=json.loads((bridge/(key+'.fit.json')).read_bytes())
        path=checked_pin(Path(data_root),{'relpath':fit['artifact_relpath'],'sha256':fit['artifact_sha256']})
        fitted=json.loads(path.read_bytes())
        load_artifact(path,expected_sha256=fit['artifact_sha256'],kind='request-fit',inputs=fitted['inputs'])
        require(fit['payload_sha256']==sha(old['payload'])==fitted['inputs']['payload_sha256']
            and fit['every_request_fits'] is True and fitted['output']['every_request_fits'] is True
            and fitted['inputs']['registration_sha256']==sha(registration)
            and fit['measured_prompt_tokens']==fitted['output']['rows'][0]['measured_prompt_tokens'],
            'Reflexion recovery original FIT custody differs')
        cap=document['bounds']['completion_cap']; tokens=fit['measured_prompt_tokens']
        require(type(tokens) is int and tokens>0 and tokens+cap+document['bounds']['reserve_tokens']
            <=document['bounds']['resident_context'],'Reflexion recovery original context exceeds bound')
        actual={'calls':1,**{k:body['usage'][k] for k in ('prompt_tokens','completion_tokens')}}
        require(all(type(v) is int and v>=0 for v in actual.values())
            and actual['completion_tokens']<=cap,'Reflexion recovery exact bounded native usage required')
        for directory in (bridge,learner/'contacts'):
            response_path=directory/(key+'.response.json')
            if response_path.exists():
                envelope=json.loads(response_path.read_bytes()); response=envelope['response']
                require(envelope['request_sha256']==sha(request) and response['model_id']==document['model_id']
                    and response['text']==body['text'] and response['usage']==actual
                    and response['response_sha256']==record['response']['response_sha256']
                    and response['fit_receipt_sha256']==sha(fit),'Reflexion recovery observed receipt changed')
                if directory != bridge:
                    require(envelope['response_sha256']==sha(response),'Reflexion recovery learner envelope changed')
        return {'text':body['text'],'usage':actual}
    validate=request_validator(document,count_tokens); caller=SimpleNamespace(root=bridge,observed=observed)
    # Native can be observed before bridge response persistence. The validator
    # reads only through this explicit native-backed view of the full prefix.
    class PrefixView:
        def glob(self,pattern):
            suffix='.intent.json' if pattern=='*.intent.json' else '.response.json'
            return [Path(key+suffix) for key in prefix]
    caller.root=PrefixView()
    for key in keys[:len(native)]:
        request=json.loads((bridge/(key+'.intent.json')).read_bytes())['request']
        validate(request,caller); result=observed(request)
        payload=json.loads(request['messages'][1]['content'])
        method.validate_reflection(result['text'],payload['original_group'],
            document['bounds']['max_entry_characters'], document['bounds'].get('reflector_contract'))
        for field in usage:usage[field]+=result['usage'][field]
    require(usage['prompt_tokens']<=document['bounds']['max_prompt_tokens_total']
        and usage['completion_tokens']<=document['bounds']['max_completion_tokens_total'],
        'Reflexion recovery prefix exceeds total budget')
    return {'schema':'saea-reflexion-known-prefix-v1','source':replay.source_binding(),
        'observed_prefix':keys[:len(native)],'known_usage':usage,'new_model_calls':0}


def run_registered_reflexion(runtime):
    reg = runtime._composed.values['tfgrpo_learning_registration']
    def reload_source():
        return load_runtime_source(runtime._composed.values, plan=runtime.plan, repo_root=runtime._repo_root,
            data_root=runtime._data_root, sample_ids=tuple(s.sample_id for s in runtime._samples),
            roster_sha256=reg['construction_roster_sha256'], replay=getattr(runtime,'_native_contact_replay',None))[0]
    loaded = reload_source(); root = runtime._run_dir()/'tfgrpo'
    require(root.resolve().is_relative_to(Path(runtime._data_root).resolve()), 'owned Reflexion artifacts required')
    replay=getattr(runtime,'_native_contact_replay',None)
    if replay is not None:
        verify_recovery_prefix(replay,data_root=runtime._data_root,plan=runtime.plan,
            registration=reg,document=loaded['document'],count_tokens=loaded['count_tokens'])
    caller = getattr(runtime,'_tfgrpo_caller',None)
    if caller is None:
        require(not any((root/'contacts').glob('*.intent.json'))
            and not any((root/'learner/contacts').glob('*.intent.json')),
            'prior Reflexion spend requires verified recovery; no fresh-client replay')
        caller = bind_runtime(runtime, receipt_root=root/'contacts',
            request_validator=request_validator(loaded['document'],loaded['count_tokens']))
        enforce_native_budget(caller,loaded['document'])
        runtime._client, runtime._tfgrpo_caller = caller.client, caller
    else:
        require(caller.root.resolve() == (root/'contacts').resolve() and caller.registration == reg
            and caller.plan == runtime.plan and caller.client is getattr(runtime,'_client',None)
            and not caller.client._closed, 'Reflexion live client/registration changed')
    learner = ExperienceLearner(root/'learner',model_id=reg['model_id'],
        learning_config_sha256=reg['learning_config_sha256'],metered_call=caller,
        preflight=caller.preflight,reconcile=caller.reconcile,max_calls=reg['bounds']['max_calls'],concurrency=1)
    return method.execute(learner,loaded['document'],count_tokens=loaded['count_tokens'],
                       verify_source=lambda:reload_source()['source'])


def dispatch_registered_reflexion(runtime):
    started = getattr(runtime,'_tfgrpo_started',None)
    if started is None:
        started = runtime._clock(); runtime._tfgrpo_started = started
    candidate = run_registered_reflexion(runtime)
    return seal_registered_candidate(runtime,candidate,started=started,ended=runtime._clock())
