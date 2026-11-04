"""Explicit reviewed external TF-GRPO development consumption; never a native seal."""
import json
from pathlib import Path

from .tfgrpo_learning import sha
from .tfgrpo_external_compile import compile_preview
from .tfgrpo_external_proposals import STRATEGY

KIND = 'external-tfgrpo-v1'
SCHEMA = 'saea-tfgrpo-external-release-v1'
REVIEW_SCHEMA = 'saea-tfgrpo-external-review-v1'
ROSTER = 'docs/receipts/rosters/slurp-stage3-construction-r01c-h2.json'
DATASET = 'configs/dataset/slurp-speech-native-stage3-construction-r01c.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_pin(root, pin):
    from .tfgrpo_consumption import checked_pin
    return json.loads(checked_pin(root, pin).read_bytes())


def render(candidate, base, consumer_ids):
    from .tfgrpo_consumption import SEPARATOR
    require(candidate.get('schema') == 'saea-tfgrpo-external-preview-v1'
            and candidate.get('strategy') == STRATEGY
            and candidate.get('sha256') == sha({k: v for k, v in candidate.items() if k != 'sha256'})
            and candidate.get('native_learning_seal') is False
            and candidate.get('deployment_accepted') is False
            and candidate.get('candidate_publishable') is False
            and candidate.get('semantic_validation') == 'pending', 'unaltered external preview required')
    items = candidate.get('items')
    require(isinstance(items, dict) and items and all(isinstance(k, str) and k and isinstance(v, str)
            and v.strip() for k, v in items.items()), 'nonempty external experience required')
    require(not set(consumer_ids) & set(candidate['learned_query_ids']), 'learning queries cannot consume candidate')
    return base + SEPARATOR + '\n'.join(f'[{k}]. {v}' for k, v in sorted(items.items())) + '\n'


def verify_deployment_binding(params, *, repo, data):
    """Keep the learned actor identity while checking a separately named deployment."""
    import re
    require(all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', params[k])
                for k in ('model_id', 'runtime_model_id')), 'safe model fragment names required')
    binding = read_pin(data, params['deployment_binding'])
    require(set(binding) == {'schema', 'actor_model', 'runtime_model'}
            and binding['schema'] == 'saea-external-actor-deployment-v1', 'explicit deployment equivalence required')
    for key, name in [('actor_model', params['model_id']), ('runtime_model', params['runtime_model_id'])]:
        require(binding[key]['relpath'] == 'configs/model/' + name + '.json', 'exact actor/runtime fragment pin required')
    actor = read_pin(repo, binding['actor_model']); runtime = read_pin(repo, binding['runtime_model'])
    operational = {'base_url', 'runtime_receipt', 'geometry_basis', 'timeout_seconds', 'answer_reserve_tokens'}
    def science(value):
        value = json.loads(json.dumps(value))
        for key in operational: value.pop(key, None)
        capacity = value['decoding_params'].pop('max_tokens')
        require(type(capacity) is int and capacity >= 4096, 'complete frozen node output capacity required')
        args = value['server_args']
        require(args.count('--max-model-len') == 1 and not any('rope' in str(a).lower() for a in args), 'native context geometry required')
        idx = args.index('--max-model-len') + 1
        require(4096 <= int(args[idx]) <= 65536, 'native bounded context required')
        args[idx] = '<native-context-capacity>'
        return value
    require(science(actor) == science(runtime), 'runtime changes frozen actor weights, tokenizer or decoding')
    return binding


def verify_source(params, *, repo, data, candidate):
    """Offline replay of original request/response; no ExternalTeacher dispatch."""
    spec = read_pin(data, params['source_manifest'])
    require(compile_preview(spec) == candidate, 'external candidate differs from original wire replay')
    require(candidate['model_id'] == params['model_id'], 'frozen actor model differs')
    return verify_reviewed_development(params, repo=repo, data=data, candidate=candidate,
        release_schema=SCHEMA, review_schema=REVIEW_SCHEMA,
        evidence={g['query_id']: g['group_sha256'] for g in candidate['relative_evidence']['groups']})


def verify_reviewed_development(params, *, repo, data, candidate, release_schema, review_schema, evidence):
    """Shared split/audio/review custody; algorithm replay stays with its adapter."""
    release = read_pin(data, params['candidate_release'])
    fields = {'schema', 'candidate', 'source_manifest', 'consumer_roster', 'learning_dataset',
              'consumer_ids', 'selection_ids', 'review'}
    if 'runtime_model_id' in params:
        verify_deployment_binding(params, repo=repo, data=data)
        fields |= {'runtime_model_id', 'deployment_binding'}
        require(all(release.get(k) == params[k] for k in ('runtime_model_id', 'deployment_binding')), 'release runtime binding differs')
    require(isinstance(release, dict) and set(release) == fields and release['schema'] == release_schema,
            'explicit external development release required')
    require(all(release[k] == params[k] for k in ('candidate', 'source_manifest', 'consumer_ids')),
            'release candidate, source or ordered consumers differ')
    require(release['consumer_roster']['relpath'] == ROSTER and release['learning_dataset']['relpath'] == DATASET,
            'fixed original H2 roster and TRAIN625 audio view required')
    roster = read_pin(repo, release['consumer_roster']); dataset = read_pin(repo, release['learning_dataset'])
    roster_ids = [r['sample_id'] for r in roster['items']]
    ids = params['consumer_ids']; selection = release['selection_ids']
    require(len(roster_ids) == len(set(roster_ids)) == 312 and set(ids) <= set(roster_ids),
            'consumers must be a fixed H2 subset')
    require(isinstance(selection, list) and all(isinstance(s, str) for s in selection)
            and selection and len(selection) == len(set(selection)), 'explicit unique selection identities required')
    index = {r['sample_id']: r for r in dataset['items']}
    learned = set(candidate['learned_query_ids']); consumer = set(ids); selected = set(selection)
    require(dataset.get('count') == len(index) == len(dataset['items']) == 625
            and learned | consumer | selected <= set(index), 'complete TRAIN625 audio custody required')
    def identities(sample_ids):
        utterances, audios = set(), set()
        for sid in sample_ids:
            require(sid.startswith('slurp:train:') and sid.count(':') >= 3, 'TRAIN-only original utterance identity required')
            utterances.add(':'.join(sid.split(':')[:3]))
            digest = index[sid]['audio_sha256']
            require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
                    'immutable audio digest required')
            audios.add(digest)
        return utterances, audios
    learn_u, learn_a = identities(learned); use_u, use_a = identities(consumer); select_u, select_a = identities(selected)
    require(not learned & consumer and not learn_u & use_u and not learn_a & use_a,
            'learning overlaps development query, utterance or audio')
    require(not selected & (learned | consumer) and not select_u & (learn_u | use_u)
            and not select_a & (learn_a | use_a), 'selection overlaps learning or evaluation')
    review = read_pin(data, release['review'])
    bindings = {k: v for k, v in release.items() if k not in ('schema', 'review')}
    require(isinstance(review, dict) and set(review) == {'schema', 'decision', 'reviewer', 'procedure', 'bindings', 'items'}
            and review['schema'] == review_schema and review['decision'] == 'eligible-for-development-evaluation'
            and review['bindings'] == bindings, 'independent review must bind candidate, splits and sources')
    require(all(isinstance(review[k], str) and review[k].strip() for k in ('reviewer', 'procedure')),
            'named reviewer and procedure required')
    require(isinstance(review['items'], dict) and set(review['items']) == set(candidate['items']),
            'every retained experience requires independent semantic review')
    for key, content in candidate['items'].items():
        item = review['items'][key]
        require(isinstance(item, dict) and set(item) == {'content_sha256', 'decision', 'evidence', 'reason'}
                and item['content_sha256'] == sha(content) and item['decision'] == 'supported'
                and isinstance(item['reason'], str) and item['reason'].strip()
                and isinstance(item['evidence'], list) and item['evidence'], 'supported exact item required')
        for ref in item['evidence']:
            require(isinstance(ref, dict) and set(ref) == {'query_id', 'group_sha256'}
                    and ref['query_id'] in evidence and evidence[ref['query_id']] == ref['group_sha256'],
                    'review must cite supplied TRAIN evidence')
    return release


def prepare(*, candidate_path, source_manifest, candidate_release, repo, data, base_template,
            template, consumer_ids, model_id, runtime_model_id=None, deployment_binding=None):
    return prepare_reviewed(kind=KIND, renderer=render, source_verifier=verify_source,
        candidate_path=candidate_path, source_manifest=source_manifest, candidate_release=candidate_release,
        repo=repo, data=data, base_template=base_template, template=template, consumer_ids=consumer_ids,
        model_id=model_id, runtime_model_id=runtime_model_id, deployment_binding=deployment_binding)


def prepare_reviewed(*, kind, renderer, source_verifier, candidate_path, source_manifest, candidate_release,
                     repo, data, base_template, template, consumer_ids, model_id,
                     runtime_model_id=None, deployment_binding=None):
    from .tfgrpo_consumption import digest, validate_params, verify, checked_pin
    from ..core.atomic_file import publish_new
    import hashlib
    repo = Path(repo).resolve(); data = Path(data).resolve()
    def pin(path):
        path = Path(path).resolve()
        require(path.is_relative_to(data), 'owned external artifact required')
        return {'relpath': str(path.relative_to(data)), 'sha256': digest(path)}
    cp = pin(candidate_path); candidate = read_pin(data, cp)
    base_pin = {'relpath': base_template, 'sha256': digest(repo / 'configs' / base_template)}
    base = checked_pin(repo / 'configs', base_pin).read_text()
    text = renderer(candidate, base, consumer_ids)
    params = {'source_kind': kind, 'candidate': cp, 'source_manifest': pin(source_manifest),
              'candidate_release': pin(candidate_release), 'model_id': model_id, 'base_template': base_pin,
              'template': {'relpath': template, 'sha256': hashlib.sha256(text.encode()).hexdigest()},
              'consumer_ids': list(consumer_ids), 'purpose': 'development-evaluation'}
    if runtime_model_id is not None or deployment_binding is not None:
        require(runtime_model_id is not None and deployment_binding is not None, 'complete runtime alias required')
        params.update(runtime_model_id=runtime_model_id, deployment_binding=pin(deployment_binding))
    validate_params(params); source_verifier(params, repo=repo, data=data, candidate=candidate)
    target = (repo / 'configs' / template).resolve()
    require(target.is_relative_to(repo / 'configs'), 'template escaped repository')
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_new(target, text.encode())
    verify(params, repo=repo, data=data, model_id=runtime_model_id or model_id, consumer_ids=consumer_ids)
    return params
