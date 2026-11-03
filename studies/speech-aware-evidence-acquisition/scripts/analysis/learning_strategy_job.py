"""Prepare bounded learning strategies; reuse shared register/admit/run.

This is a source-building API, not a launcher. No generation or tokenizer
endpoint is contacted during preparation; dynamic prompts are fitted later.
"""
import hashlib
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis')]
import tfgrpo_learning_job as base
from speech_aware_evidence_acquisition.contracts import ExecutionPlan, assert_execution_environment
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash
from speech_aware_evidence_acquisition.rails import ace_incremental as ace
from speech_aware_evidence_acquisition.rails import ace_runtime as runtime
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
from speech_aware_evidence_acquisition.rails.tfgrpo_runtime import learning_configuration
from speech_aware_evidence_acquisition.rails.tfgrpo_source import read_pin

SCHEMA = 'saea-learning-strategy-job-v1'
read, raw, pin, file_sha = base.read, base.raw, base.pin, base.file_sha


def strategy_modules(strategy):
    """Only implemented strategies can reach shared registration or dispatch."""
    if strategy == ace.STRATEGY:
        return ace, runtime
    from speech_aware_evidence_acquisition.rails import reflexion_offline, reflexion_runtime
    ace.require(strategy == reflexion_offline.STRATEGY, 'implemented learning strategy required; no silent fallback')
    return reflexion_offline, reflexion_runtime


def initial_measurement(method, document, count):
    if method.STRATEGY == ace.STRATEGY:
        return method.reflect_messages(document, 0, document['parent']['items'], count)
    return method.reflect_messages(document, 0, method.empty_memory(), count)


def static_capacity_audit(method, document, count, *, extra_reserve_tokens=0):
    """Fit every fixed input with initial state; dynamic phases remain unproven."""
    ace.require(type(extra_reserve_tokens) is int and extra_reserve_tokens >= 0,
                'nonnegative static capacity reserve required')
    limits = document['bounds']; rows = []
    for index, group in enumerate(document['groups']):
        state = document['parent']['items'] if method.STRATEGY == ace.STRATEGY else method.empty_memory()
        measured = method.reflect_messages(document, index, state, count)
        required = measured['input_tokens'] + limits['completion_cap'] + limits['reserve_tokens'] + extra_reserve_tokens
        ace.require(required <= limits['resident_context'],
                    'static capacity reserve exceeded at group ' + str(index))
        rows.append({'index': index, 'query_id': group['query_id'],
                     'messages_sha256': measured['messages_sha256'],
                     'input_tokens': measured['input_tokens'], 'required_tokens': required})
    return {'schema': 'saea-learning-static-capacity-v1', 'groups': len(rows), 'rows': rows,
        'extra_reserve_tokens': extra_reserve_tokens,
        'max_input_tokens': max(row['input_tokens'] for row in rows),
        'minimum_headroom_tokens': min(limits['resident_context'] - row['required_tokens'] for row in rows),
        'scope': 'all-fixed-reflection-inputs-with-initial-state',
        'dynamic_library_or_memory_growth_verified': False, 'native_fit_completed': False}


def prepare(repo, data, *, strategy, source_spec, source_roster, tokenizer_relpath,
            runtime_identity, bounds, out, job_name, run_id, service_session,
            planned_gpu_hours, max_context_bytes=2000000, response_format=None, development_spec=None,
            capacity_reserve_tokens=0):
    assert_execution_environment()
    require = ace.require
    method, selected_runtime = strategy_modules(strategy)
    require(re.fullmatch('[a-z0-9][a-z0-9-]*',job_name)
        and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*',service_session), 'safe strategy job/session required')
    require(not out.exists() and out.resolve().is_relative_to((data/'work/codex').resolve()), 'fresh owned strategy directory required')
    rel = Path(tokenizer_relpath)
    require(not rel.is_absolute() and '..' not in rel.parts and (data/rel).resolve().is_relative_to(data.resolve()), 'confined tokenizer required')
    identity, spec = read(runtime_identity), read(source_spec)
    count, tokenizer = selected_runtime.load_token_counter(data/rel)
    source = selected_runtime.load_finalized_shards(spec,data_root=data,count_tokens=count,tokenizer_identity=tokenizer)
    model = source['model_id']
    require(base.deployment_identity(repo,model,identity) == identity, 'exact learning deployment identity required')
    decoding = base.learning_decoding(read(repo/f'configs/model/{model}.json'))
    if response_format is not None:
        require(response_format == {'type': 'json_object'},
                'only exact json_object response_format is supported')
        decoding['response_format'] = dict(response_format)
    config = {'model_id':model,'decoding_params':decoding,
        'max_context_bytes':max_context_bytes,'runtime_identity_sha256':sha(identity)}
    development = {}
    if development_spec is not None:
        require(strategy == ace.STRATEGY, 'development evidence currently requires ACE')
        from speech_aware_evidence_acquisition.rails.construction_development import load_material
        development['development_material'] = load_material(read(development_spec), source,
            repo_root=repo, data_root=data)
    document = method.build_input(source,source['batch']['parent'],
        learning_config_sha256=sha(learning_configuration(config)),**bounds,**development)
    initial = initial_measurement(method, document, count)
    capacity = static_capacity_audit(method, document, count, extra_reserve_tokens=capacity_reserve_tokens)
    ids = [g['query_id'] for g in document['groups']]
    original = read(source_roster); index = {r['sample_id']:r for r in original['items']}
    require(len(index) == len(original['items']) and set(ids) <= set(index), 'complete unique original TRAIN roster required')
    roster = {'schema':'saea-roster-receipt-v1','roster_id':run_id,'roster_role':'construction',
        'carrier_lock_key':'slurp','split_role':'discovery','count':len(ids),'items':[index[i] for i in ids],
        'split_identity_hash':split_identity_hash(prefixed('slurp',ids)),
        'selection':('All original TRAIN groups; sequential local reflection and curation.'
            if strategy == ace.STRATEGY else 'All original TRAIN groups; single reflection and bounded FIFO memory.'),
        'source_queries_must_not_be_reported_as_held_out':True}
    plan = ExecutionPlan(run_id=run_id,attempt_id='learning-v1',
        execution_profile='bounded-discovery-probe',carrier_lock_key='slurp',split_role='discovery',
        split_identity_hash=roster['split_identity_hash'],planned_model_calls=document['bounds']['max_calls'],
        planned_gpu_hours=planned_gpu_hours,planned_speech_audio_seconds=1,protocol_hash='0'*64).validate()
    reg = selected_runtime.build_registration(plan,source=source,document=document,
        decoding_params=config['decoding_params'],max_context_bytes=max_context_bytes,
        runtime_identity_sha256=sha(identity),roster_sha256=hashlib.sha256(raw(roster)).hexdigest())
    source_doc = {'schema':selected_runtime.RUNTIME_SOURCE_SCHEMA,'strategy':strategy,'source':spec,
        'tokenizer_relpath':str(rel),'tokenizer_identity':tokenizer,'document':document,
        'initial_measurement':initial,'phase_contract':'unknown-prompts-await-observed-predecessor'}
    artifacts = {'construction-roster.json':roster,'source.json':source_doc,
        'registration.json':reg,'runtime-identity.json':identity}
    out.mkdir(parents=True)
    for name,value in artifacts.items():publish_new(out/name,raw(value))
    complete = {'schema':SCHEMA,'strategy':strategy,'job':job_name,'run_id':run_id,
        'groups':len(ids),'runtime_rows':len(ids),'generation_calls':len(reg['contacts']),
        'fit_calls':len(reg['contacts']),'batch_sha256':sha(document),
        'runtime_identity_sha256':sha(identity),'historical_custody_sha256':sha(source),
        'historical_usage':source['original_total_usage'],'service_session':service_session,
        'inputs':{'source_roster':pin(source_roster,data,'data')},
        'artifacts':{n:file_sha(out/n) for n in artifacts},'semantic_review_required':True,
        'candidate_deployment_accepted':False,'readiness':'prepared-not-admitted',
        'phase_readiness':{'first':'AWAITING_NATIVE_FIT','later':'AWAITING_OBSERVED_PREDECESSOR'},
        'new_model_calls':0,'new_tokenizer_calls':0,'static_capacity_audit':capacity}
    if response_format is not None:
        complete['response_format'] = dict(response_format)
    publish_new(out/'complete.json',raw(complete))
    verify_job(repo,data,out)
    return complete


def verify_job(repo,data,job):
    complete = read(job/'complete.json'); require = ace.require
    require(complete.get('schema') == SCHEMA and isinstance(complete.get('strategy'), str),
        'known learning strategy job required')
    method, selected_runtime = strategy_modules(complete['strategy'])
    names = {'construction-roster.json','source.json','registration.json','runtime-identity.json'}
    require(set(complete['artifacts']) == names
        and all(file_sha(job/n) == complete['artifacts'][n] for n in names), 'learning strategy artifacts changed')
    reg, roster, identity = [read(job/n) for n in ('registration.json','construction-roster.json','runtime-identity.json')]
    require(base.deployment_identity(repo,reg['model_id'],identity) == identity
        and sha(identity) == reg['runtime_identity_sha256'] == complete['runtime_identity_sha256'], 'strategy deployment identity differs')
    expected_decoding = base.learning_decoding(read(repo/f"configs/model/{reg['model_id']}.json"))
    if 'response_format' in complete:
        require(complete['response_format'] == {'type': 'json_object'},
                'only exact json_object response_format is supported')
        expected_decoding['response_format'] = dict(complete['response_format'])
    require(reg['decoding_params'] == expected_decoding,
            'strategy request decoding differs from configured response format')
    plan = ExecutionPlan(**{**reg['plan'],'protocol_hash':'0'*64}).validate()
    loaded,_ = selected_runtime.load_runtime_source({'tfgrpo_learning_registration':reg,'prepared_training_feedback':{
        'schema':selected_runtime.SOURCE_SCHEMA,'source':pin(job/'source.json',data,'data')}},plan=plan,
        repo_root=repo,data_root=data,sample_ids=tuple(r['sample_id'] for r in roster['items']),
        roster_sha256=file_sha(job/'construction-roster.json'))
    if 'static_capacity_audit' in complete:
        recorded = complete['static_capacity_audit']
        require(recorded == static_capacity_audit(method, loaded['document'], loaded['count_tokens'],
            extra_reserve_tokens=recorded['extra_reserve_tokens']), 'static capacity audit changed')
    original = read_pin(complete['inputs']['source_roster'],repo_root=repo,data_root=data)
    index = {r['sample_id']:r for r in original['items']}
    ids = [g['query_id'] for g in loaded['document']['groups']]
    require(len(index) == len(original['items']) and set(ids) <= set(index), 'strategy original roster differs')
    require(roster['items'] == [index[i] for i in ids] and roster['count'] == len(ids)
        and roster['split_identity_hash'] == split_identity_hash(prefixed('slurp',ids)) == plan.split_identity_hash
        and complete['groups'] == complete['runtime_rows'] == len(ids)
        and complete['generation_calls'] == complete['fit_calls'] == len(reg['contacts'])
        and complete['run_id'] == plan.run_id and complete['batch_sha256'] == sha(loaded['document'])
        and complete['historical_custody_sha256'] == sha(loaded['source'])
        and complete['historical_usage'] == loaded['source']['original_total_usage']
        and complete['semantic_review_required'] is True and complete['candidate_deployment_accepted'] is False
        and complete['new_model_calls'] == complete['new_tokenizer_calls'] == 0
        and complete['phase_readiness'] == {'first':'AWAITING_NATIVE_FIT','later':'AWAITING_OBSERVED_PREDECESSOR'},
        'strategy job derivation differs')
    return complete,reg,roster


def register(repo,data,job):
    from tfgrpo_final_job import register as shared
    return shared(repo,data,job)


def admit(repo,data,job):
    from tfgrpo_final_job import admit as shared
    return shared(repo,data,job)


def run(repo,data,job,out,after,predecessor_run):
    from grounded_synthesis_job import run as shared
    return shared(repo,data,job,out,after,predecessor_run)


CONFIG_SCHEMA = 'saea-learning-strategy-command-v1'


def execute_config(path):
    """One reusable JSON command schema for every implemented strategy job."""
    config=read(Path(path)); require=ace.require
    require(isinstance(config,dict) and set(config)=={'schema','mode','repo','data','arguments'}
        and config['schema']==CONFIG_SCHEMA,'exact learning strategy command config required')
    mode=config['mode']; args=config['arguments']
    fields={
        'prepare':{'strategy','source_spec','source_roster','tokenizer_relpath','runtime_identity',
            'bounds','out','job_name','run_id','service_session','planned_gpu_hours','max_context_bytes'},
        'verify':{'job'}, 'register':{'job'}, 'admit':{'job'},
        'run':{'job','out','after','predecessor_run'},
    }
    # ``response_format`` was added as an optional configuration input; old
    # command files remain valid and therefore use the exact legacy field set.
    allowed_prepare = fields['prepare'] | {'response_format', 'development_spec', 'capacity_reserve_tokens'}
    require(mode in fields and isinstance(args,dict)
        and (set(args) == fields[mode] or (mode == 'prepare' and set(args) <= allowed_prepare
                                           and fields['prepare'] <= set(args))),
            'exact implemented learning strategy mode arguments required')
    paths={'repo','data','source_spec','source_roster','runtime_identity','job','out','after','development_spec'}
    def path_value(value):
        require(isinstance(value,str) and value.startswith('/') and '..' not in Path(value).parts
            and not any(c in value for c in ('\x00','\n')), 'absolute canonical command path required')
        return Path(value)
    repo,data=path_value(config['repo']),path_value(config['data'])
    kwargs={key:path_value(value) if key in paths else value for key,value in args.items()}
    if mode=='prepare':return prepare(repo,data,**kwargs)
    if mode=='verify':return verify_job(repo,data,**kwargs)[0]
    return {'register':register,'admit':admit,'run':run}[mode](repo,data,**kwargs)


if __name__=='__main__':
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    print(json.dumps(execute_config(parser.parse_args().config),sort_keys=True))
