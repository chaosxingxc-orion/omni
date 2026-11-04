"""Reflexion-specific, wire-replayed language-memory proposals with explicit development release."""
import json
from pathlib import Path

from . import external_exploration as external
from .tfgrpo_learning import sha
from .tfgrpo_external_release import require, read_pin, verify_reviewed_development

KIND = 'external-reflexion-v1'
STRATEGY = 'reflexion-external-memory-transfer-offline-v1'
SCHEMA = 'saea-reflexion-external-release-v1'
REVIEW_SCHEMA = 'saea-reflexion-external-review-v1'
SEPARATOR = '\n\nReflexion reviewed language memory (conditions and exceptions are instructions, not a programmatic gate):\n'


def validate_context(context, source_pin):
    require(context.get('schema') == 'saea-reflexion-offline-transfer-context-v1', 'explicit Reflexion transfer context required')
    memory = context.get('memory')
    require(isinstance(memory, dict) and set(memory) == {'version', 'items'}
            and isinstance(memory['version'], str) and memory['version'].strip()
            and memory['items'] == [], 'empty versioned memory required')
    frozen = context.get('frozen_knowledge', {})
    require(frozen.get('source_sha256') == source_pin['sha256'] and frozen.get('editable') is False,
            'frozen actual supplied knowledge source required')
    require(context.get('required_transfer_scope') == 'cross_task_adaptation'
            and context.get('retry_history', {}).get('available') is False
            and isinstance(context.get('actor_model_id'), str) and context['actor_model_id'],
            'explicit offline transfer without fabricated retry history required')
    return memory


def transfer_scope(proposal, context):
    declared = proposal.get('transfer_scope')
    require(declared is None or declared == 'cross_task_adaptation', 'only offline cross-task transfer supported')
    require(context.get('required_transfer_scope') == 'cross_task_adaptation', 'pinned transfer scope required')
    return {'value': 'cross_task_adaptation',
            'authority': 'teacher-and-prepared-context' if declared is not None else 'derived-from-prepared-context',
            'teacher_field_present': 'transfer_scope' in proposal}


def validated_items(proposal, context, evidence_ids):
    fields = {'id', 'evidence_query_ids', 'hypothesis', 'memory', 'retry_protocol', 'validation'}
    require(fields <= set(proposal) <= fields | {'transfer_scope'}, 'exact Reflexion proposal required')
    transfer_scope(proposal, context)
    refs = proposal['evidence_query_ids']
    require(isinstance(refs, list) and refs and len(refs) == len(set(refs)) and set(refs) <= set(evidence_ids),
            'memory must cite supplied TRAIN groups')
    for name in ('id', 'hypothesis', 'retry_protocol', 'validation'):
        require(isinstance(proposal[name], str) and proposal[name].strip(), 'complete reasoning and retry protocol required')
    memory = proposal['memory']
    require(isinstance(memory, dict) and set(memory) == {'trigger', 'action', 'exception'}, 'complete runtime memory required')
    for value in memory.values():
        require(isinstance(value, str) and value.strip() and len(value) <= 4000
                and 'slurp:' not in value, 'bounded generalizable memory required')
    key = 'reflexion-' + sha(memory)
    text = 'Trigger: ' + memory['trigger'] + '\nAction: ' + memory['action'] + '\nException: ' + memory['exception']
    require(len(text) <= 10000, 'bounded language memory required')
    return {key: text}, {key: ['memory']}


def compile_preview(spec):
    require(isinstance(spec, dict) and set(spec) == {'schema', 'prepared', 'collected', 'candidate_index'}
            and spec['schema'] == 'saea-reflexion-external-compile-v1', 'exact Reflexion compile specification required')
    pp, cp = external.checked(spec['prepared']), external.checked(spec['collected'])
    require(pp.name == 'prepared.json' and cp == pp.parent/'collected.json', 'original prepared/collected required')
    root, manifest, caller, request = external.load_prepared(pp.parent)
    source_path = external.checked(manifest['source']); source = external.read(source_path)
    context = external.read(root/'context.json'); memory = validate_context(context, manifest['source'])
    require(source['document'].get('model_id') == context['actor_model_id'], 'actor differs from original source model')
    payload = external.request_payload(request); profile = external.read(root/'profile.json'); record = external.read(cp)
    require(payload['algorithm'] == profile['algorithm'] == 'reflexion' and payload['context'] == context
            and payload['source_sha256'] == manifest['source']['sha256'], 'Reflexion source/context identity differs')
    original = {g['query_id']: g for g in source['document']['groups']}
    require(len(original) == len(source['document']['groups']), 'unique original source groups required')
    evidence = {}
    exposure = {}
    for group in payload['groups']:
        sid = group['query_id']
        require(sid.startswith('slurp:train:') and sid not in evidence and sid in original, 'unique original TRAIN groups required')
        old = original[sid]; encoded = old['evidence']; table = encoded.get('development_evidence_dictionary', {})
        require(not table or sha(table) == encoded['development_evidence_dictionary_sha256'], 'source dictionary differs')
        full = external.expand_evidence({k:v for k,v in encoded.items() if not k.startswith('development_evidence_')}, table)
        require(group == {'query_id': sid, 'group_sha256': old['group_sha256'], 'evidence': full}, 'actual exposure differs from source')
        evidence[sid] = old['group_sha256']
        exposure[sid] = {'supplied_evidence_sha256': sha(full['query']['supplied_evidence']),
                        'request_sha256': [sha(r['source_trajectory']['request']) for r in full['rollouts']],
                        'rollout_count': len(full['rollouts'])}
    require(evidence, 'nonempty Reflexion exposure required')
    observed = caller.reconcile(request)
    require(record.get('schema') == 'saea-external-exploration-collected-v1'
            and record.get('prepared_sha256') == spec['prepared']['sha256']
            and observed is not None and observed == record.get('response')
            and sha(request) == record.get('request_sha256'), 'verified Reflexion provider custody required')
    require(observed['response']['finish_reason'] == 'stop', 'complete Reflexion response required')
    proposals = external.validate_proposals(observed['response']['text'], profile, context, list(evidence))
    require(proposals == record.get('proposals') and record.get('candidate_accepted') is False, 'Reflexion collected response differs')
    index = spec['candidate_index']
    require(type(index) is int and 0 <= index < len(proposals['candidates']), 'valid Reflexion candidate index required')
    proposal = proposals['candidates'][index]; items, provenance = validated_items(proposal, context, evidence)
    body = {'schema': 'saea-reflexion-external-preview-v1', 'strategy': STRATEGY,
            'model_id': context['actor_model_id'], 'parent_version': memory['version'], 'parent_sha256': sha(memory),
            'items': items, 'item_source_fields': provenance, 'proposal': proposal, 'transfer_scope': transfer_scope(proposal, context),
            'evidence_groups': evidence, 'exposure': exposure, 'learned_query_ids': sorted(evidence),
            'source': manifest['source'], 'prepared': spec['prepared'], 'collected': spec['collected'],
            'request_sha256': sha(request), 'wire_response_sha256': observed['response']['wire_response_sha256'],
            'teacher_model_id': observed['response']['model_id'], 'provider_usage': observed['response']['provider_usage'],
            'learning_usage': observed['response']['usage'], 'usage_scope': 'one request shared by all candidate previews',
            'semantic_validation': 'pending', 'deployment_accepted': False, 'native_learning_seal': False,
            'candidate_publishable': False, 'programmatic_applicability_gate': False,
            'implementation': 'Reflexion-inspired offline cross-task language-memory proposal; not native online Reflexion retries', 'new_model_calls': 0}
    return {**body, 'sha256': sha(body)}


def render(candidate, base, consumer_ids):
    require(candidate.get('schema') == 'saea-reflexion-external-preview-v1' and candidate.get('strategy') == STRATEGY
            and candidate.get('sha256') == sha({k:v for k,v in candidate.items() if k != 'sha256'})
            and all(candidate.get(k) is False for k in ('native_learning_seal', 'deployment_accepted', 'candidate_publishable', 'programmatic_applicability_gate'))
            and candidate.get('semantic_validation') == 'pending', 'unaltered pending Reflexion preview required')
    require(not set(consumer_ids) & set(candidate['learned_query_ids']), 'learned queries cannot consume Reflexion candidate')
    require(candidate['items'], 'nonempty Reflexion additions required')
    return base + SEPARATOR + '\n'.join('['+key+']\n'+value for key,value in sorted(candidate['items'].items())) + '\n'


def verify_source(params, *, repo, data, candidate):
    require(compile_preview(read_pin(data, params['source_manifest'])) == candidate, 'Reflexion candidate differs from provider replay')
    require(candidate['model_id'] == params['model_id'], 'Reflexion frozen actor differs')
    return verify_reviewed_development(params, repo=repo, data=data, candidate=candidate,
        release_schema=SCHEMA, review_schema=REVIEW_SCHEMA, evidence=candidate['evidence_groups'])


def prepare(*, candidate_path, source_manifest, candidate_release, repo, data, base_template,
            template, consumer_ids, model_id, runtime_model_id=None, deployment_binding=None):
    from .tfgrpo_consumption import digest, checked_pin, validate_params, verify
    from ..core.atomic_file import publish_new
    import hashlib
    repo=Path(repo).resolve();data=Path(data).resolve()
    def pin(path):
        path=Path(path).resolve();require(path.is_relative_to(data), 'owned Reflexion artifact required')
        return {'relpath':str(path.relative_to(data)), 'sha256':digest(path)}
    cp=pin(candidate_path); candidate=read_pin(data,cp)
    bp={'relpath':base_template,'sha256':digest(repo/'configs'/base_template)}
    text=render(candidate,checked_pin(repo/'configs',bp).read_text(),consumer_ids)
    params={'source_kind':KIND,'candidate':cp,'source_manifest':pin(source_manifest), 'candidate_release':pin(candidate_release),
            'model_id':model_id,'base_template':bp,'template':{'relpath':template,'sha256':hashlib.sha256(text.encode()).hexdigest()},
            'consumer_ids':list(consumer_ids),'purpose':'development-evaluation'}
    if runtime_model_id is not None or deployment_binding is not None:
        require(runtime_model_id is not None and deployment_binding is not None, 'complete runtime binding required')
        params.update(runtime_model_id=runtime_model_id,deployment_binding=pin(deployment_binding))
    validate_params(params); verify_source(params,repo=repo,data=data,candidate=candidate)
    target=(repo/'configs'/template).resolve();require(target.is_relative_to(repo/'configs'), 'confined Reflexion template required')
    target.parent.mkdir(parents=True,exist_ok=True);publish_new(target,text.encode())
    verify(params,repo=repo,data=data,model_id=runtime_model_id or model_id,consumer_ids=consumer_ids)
    return params
