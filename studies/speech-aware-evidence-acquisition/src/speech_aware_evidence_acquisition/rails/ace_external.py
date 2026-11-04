"""ACE-specific, wire-replayed additive proposals with explicit development release."""
import json
from pathlib import Path

from . import external_exploration as external
from .tfgrpo_learning import sha
from .tfgrpo_external_release import require, read_pin, verify_reviewed_development

KIND = 'external-ace-v1'
STRATEGY = 'ace-external-additive-offline-v1'
SCHEMA = 'saea-ace-external-release-v1'
REVIEW_SCHEMA = 'saea-ace-external-review-v1'
SEPARATOR = '\n\nACE reviewed additive knowledge (conditions and exceptions are instructions, not a programmatic gate):\n'


def validate_context(context, source_pin):
    require(context.get('schema') == 'saea-ace-offline-additive-context-v1', 'explicit ACE additive context required')
    memory = context.get('editable_memory')
    require(isinstance(memory, dict) and set(memory) == {'version', 'items', 'prior_updates'}
            and isinstance(memory['version'], str) and memory['version'].strip()
            and memory['items'] == {} and memory['prior_updates'] == [], 'empty versioned additive layer required')
    frozen = context.get('frozen_knowledge', {})
    require(frozen.get('source_sha256') == source_pin['sha256'] and frozen.get('editable') is False,
            'frozen actual supplied knowledge source required')
    require(context.get('operations_allowed') == ['ADD'] and isinstance(context.get('actor_model_id'), str)
            and context['actor_model_id'], 'ACE ADD-only frozen actor required')
    bounds = context.get('bounds', {})
    require(type(bounds.get('max_operations_per_candidate')) is int
            and 1 <= bounds['max_operations_per_candidate'] <= 2, 'bounded ACE additions required')
    return memory


def validated_items(proposal, context, evidence_ids):
    require(set(proposal) == {'id', 'reflection', 'operations', 'validation'}, 'exact ACE proposal required')
    reflection = proposal['reflection']
    require(isinstance(reflection, dict) and set(reflection) == {'hypothesis', 'evidence_query_ids', 'counterexample'},
            'complete reflector evidence required')
    refs = reflection['evidence_query_ids']
    require(isinstance(refs, list) and refs and len(refs) == len(set(refs)) and set(refs) <= set(evidence_ids),
            'reflection must cite supplied TRAIN groups')
    for value in (proposal['id'], proposal['validation'], reflection['hypothesis'], reflection['counterexample']):
        require(isinstance(value, str) and value.strip(), 'nonempty reflection and validation required')
    ops = proposal['operations']
    require(isinstance(ops, list) and 1 <= len(ops) <= context['bounds']['max_operations_per_candidate'],
            'bounded nonempty ACE operations required')
    items, provenance = {}, {}
    for index, op in enumerate(ops):
        require(isinstance(op, dict) and set(op) == {'operation', 'target_id', 'condition', 'content', 'exception'}
                and op['operation'] == 'ADD' and op['target_id'] is None, 'empty ACE layer permits ADD only')
        for name in ('condition', 'content', 'exception'):
            require(isinstance(op[name], str) and op[name].strip() and len(op[name]) <= 4000
                    and 'slurp:' not in op[name], 'bounded generalizable condition/content/exception required')
        text = 'Condition: ' + op['condition'] + '\nKnowledge: ' + op['content'] + '\nException: ' + op['exception']
        key = 'ace-' + sha({k: op[k] for k in ('condition', 'content', 'exception')})
        # Exact structural de-duplication only; retain every source operation index.
        items[key] = text
        provenance.setdefault(key, []).append(index)
    require(sum(map(len, items.values())) <= 10000, 'bounded additive knowledge size required')
    return dict(sorted(items.items())), dict(sorted(provenance.items()))


def compile_preview(spec):
    require(isinstance(spec, dict) and set(spec) == {'schema', 'prepared', 'collected', 'candidate_index'}
            and spec['schema'] == 'saea-ace-external-compile-v1', 'exact ACE compile specification required')
    pp, cp = external.checked(spec['prepared']), external.checked(spec['collected'])
    require(pp.name == 'prepared.json' and cp == pp.parent/'collected.json', 'original prepared/collected required')
    root, manifest, caller, request = external.load_prepared(pp.parent)
    source_path = external.checked(manifest['source']); source = external.read(source_path)
    context = external.read(root/'context.json'); memory = validate_context(context, manifest['source'])
    require(source['document'].get('model_id') == context['actor_model_id'], 'actor differs from original source model')
    payload = external.request_payload(request); profile = external.read(root/'profile.json'); record = external.read(cp)
    require(payload['algorithm'] == profile['algorithm'] == 'ace' and payload['context'] == context
            and payload['source_sha256'] == manifest['source']['sha256'], 'ACE source/context identity differs')
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
    require(evidence, 'nonempty ACE exposure required')
    observed = caller.reconcile(request)
    require(record.get('schema') == 'saea-external-exploration-collected-v1'
            and record.get('prepared_sha256') == spec['prepared']['sha256']
            and observed is not None and observed == record.get('response')
            and sha(request) == record.get('request_sha256'), 'verified ACE provider custody required')
    require(observed['response']['finish_reason'] == 'stop', 'complete ACE response required')
    proposals = external.validate_proposals(observed['response']['text'], profile, context, list(evidence))
    require(proposals == record.get('proposals') and record.get('candidate_accepted') is False, 'ACE collected response differs')
    index = spec['candidate_index']
    require(type(index) is int and 0 <= index < len(proposals['candidates']), 'valid ACE candidate index required')
    proposal = proposals['candidates'][index]; items, provenance = validated_items(proposal, context, evidence)
    body = {'schema': 'saea-ace-external-preview-v1', 'strategy': STRATEGY,
            'model_id': context['actor_model_id'], 'parent_version': memory['version'], 'parent_sha256': sha(memory),
            'items': items, 'item_operation_indices': provenance, 'proposal': proposal,
            'evidence_groups': evidence, 'exposure': exposure, 'learned_query_ids': sorted(evidence),
            'source': manifest['source'], 'prepared': spec['prepared'], 'collected': spec['collected'],
            'request_sha256': sha(request), 'wire_response_sha256': observed['response']['wire_response_sha256'],
            'teacher_model_id': observed['response']['model_id'], 'provider_usage': observed['response']['provider_usage'],
            'learning_usage': observed['response']['usage'], 'usage_scope': 'one request shared by all candidate previews',
            'semantic_validation': 'pending', 'deployment_accepted': False, 'native_learning_seal': False,
            'candidate_publishable': False, 'programmatic_applicability_gate': False,
            'implementation': 'ACE-inspired offline Reflector/Curator additive proposal; not native online ACE', 'new_model_calls': 0}
    return {**body, 'sha256': sha(body)}


def render(candidate, base, consumer_ids):
    require(candidate.get('schema') == 'saea-ace-external-preview-v1' and candidate.get('strategy') == STRATEGY
            and candidate.get('sha256') == sha({k:v for k,v in candidate.items() if k != 'sha256'})
            and all(candidate.get(k) is False for k in ('native_learning_seal', 'deployment_accepted', 'candidate_publishable', 'programmatic_applicability_gate'))
            and candidate.get('semantic_validation') == 'pending', 'unaltered pending ACE preview required')
    require(not set(consumer_ids) & set(candidate['learned_query_ids']), 'learned queries cannot consume ACE candidate')
    require(candidate['items'], 'nonempty ACE additions required')
    return base + SEPARATOR + '\n'.join('['+key+']\n'+value for key,value in sorted(candidate['items'].items())) + '\n'


def verify_source(params, *, repo, data, candidate):
    require(compile_preview(read_pin(data, params['source_manifest'])) == candidate, 'ACE candidate differs from provider replay')
    require(candidate['model_id'] == params['model_id'], 'ACE frozen actor differs')
    return verify_reviewed_development(params, repo=repo, data=data, candidate=candidate,
        release_schema=SCHEMA, review_schema=REVIEW_SCHEMA, evidence=candidate['evidence_groups'])


def prepare(*, candidate_path, source_manifest, candidate_release, repo, data, base_template,
            template, consumer_ids, model_id, runtime_model_id=None, deployment_binding=None):
    from .tfgrpo_consumption import digest, checked_pin, validate_params, verify
    from ..core.atomic_file import publish_new
    import hashlib
    repo=Path(repo).resolve();data=Path(data).resolve()
    def pin(path):
        path=Path(path).resolve();require(path.is_relative_to(data), 'owned ACE artifact required')
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
    target=(repo/'configs'/template).resolve();require(target.is_relative_to(repo/'configs'), 'confined ACE template required')
    target.parent.mkdir(parents=True,exist_ok=True);publish_new(target,text.encode())
    verify(params,repo=repo,data=data,model_id=runtime_model_id or model_id,consumer_ids=consumer_ids)
    return params
