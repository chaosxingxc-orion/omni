"""Read-only multistage reference replay and reviewed development consumption."""
from pathlib import Path
import shutil
import tempfile

from . import external_exploration as external
from . import tfgrpo_external_reference as reference
from .tfgrpo_external_release import require, read_pin, verify_reviewed_development
from .tfgrpo_learning import DENSE_REWARD, ExperienceLearner, sha

KIND = 'external-tfgrpo-reference-v1'
STRATEGY = 'tfgrpo-reference-multistage-external-dense-offline-v1'
SCHEMA = 'saea-tfgrpo-reference-release-v1'
REVIEW_SCHEMA = 'saea-tfgrpo-reference-review-v1'
PREVIEW_SCHEMA = 'saea-tfgrpo-reference-preview-v1'


def compile_preview(spec):
    """Recreate the final candidate from complete provider-backed stages, no writes to source."""
    require(isinstance(spec, dict) and set(spec) == {'schema', 'prepared', 'collected'}
            and spec['schema'] == 'saea-tfgrpo-reference-compile-v1', 'exact reference compile specification required')
    pp, cp = external.checked(spec['prepared']), external.checked(spec['collected'])
    require(pp.name == 'prepared.json' and cp == pp.parent/'collected.json', 'original reference preparation and collection required')
    root, manifest, config, groups, parent, batch, caller = reference.load(pp.parent)
    require(manifest['algorithm'] == STRATEGY and manifest['native_learning_seal'] is False
            and manifest['deployment_accepted'] is False, 'original multistage reference algorithm required')
    record = external.read(cp)
    require(external.checked(record['prepared']) == pp
            and external.checked(record['candidate']) == root/'learner/candidate.json', 'original completed reference candidate required')
    candidate = external.read(root/'learner/candidate.json')
    receipts = []
    contacts = root/'learner/contacts'
    # Validate every intent before copying; missing/unknown responses are never reconciled into the source.
    for path in sorted(contacts.glob('*.intent.json')):
        request = external.read(path)
        response_path = path.with_name(path.name.replace('.intent.json', '.response.json'))
        require(response_path.is_file(), 'incomplete reference contact')
        observed = caller.reconcile(request)
        require(observed is not None and observed == external.read(response_path), 'reference contact differs from provider wire or is unknown')
        require(observed['response']['finish_reason'] == 'stop', 'incomplete reference provider response')
        receipts.append({'stage': request['stage'], 'identity': request['identity'],
                         'request_sha256': sha(request), 'wire_response_sha256': observed['response']['wire_response_sha256']})
    require(len(receipts) == manifest['plan']['planned_calls'] > 0, 'complete nonempty reference stage plan required')
    def no_dispatch(*args):
        raise ValueError('reference replay cannot dispatch missing contacts')
    # Excluding candidate.json forces actual template/operation replay instead of the learner's cached-result path.
    with tempfile.TemporaryDirectory(prefix='saea-reference-replay-') as directory:
        replay = Path(directory)
        shutil.copytree(contacts, replay/'contacts')
        learner = ExperienceLearner(replay, model_id=caller.model_id, metered_call=no_dispatch,
            preflight=no_dispatch, reconcile=no_dispatch, learning_config_sha256=caller.learning_config_sha256,
            **{key: config['bounds'][key] for key in ('max_calls', 'concurrency', 'max_operations', 'max_items', 'max_characters')})
        actual = learner.learn(groups, parent, **config['objectives'], actor_model_id=config['actor_model_id'],
            policy_sha256=batch['policy_sha256'], rollout_kind=batch['rollout_kind'],
            num_experiences=config['bounds']['num_experiences'], reward_mode=DENSE_REWARD)
        require(actual == candidate, 'reference candidate differs from full stage replay')
        accounting = learner.accounting()
    require(accounting['unknown_contacts'] == 0 and accounting['observed_contacts'] == len(receipts)
            and candidate['learning_usage'] == accounting['known_usage'], 'complete reference learning accounting required')
    expected = {'schema': 'saea-tfgrpo-external-reference-collected-v1',
        'prepared': external.pin(pp), 'candidate': external.pin(root/'learner/candidate.json'),
        'algorithm': STRATEGY, 'actor_model_id': config['actor_model_id'], 'learner_model_id': caller.model_id,
        'contacts': receipts, 'accounting': accounting, 'candidate_sha256': candidate['sha256'],
        'native_learning_seal': False, 'semantic_validation': 'pending', 'deployment_accepted': False}
    require(record == expected, 'reference collection differs from complete stage replay')
    evidence = {g['query_id']: sha(g) for g in groups}
    exposure = {g['query_id']: {'group_sha256': sha(g), 'rollout_count': len(g['rollouts']),
                'trajectory_ids': [r['trajectory_id'] for r in g['rollouts']],
                'request_sha256': [sha(external_request(r)) for r in g['rollouts']]}
                for g in groups}
    body = {'schema': PREVIEW_SCHEMA, 'strategy': STRATEGY, 'model_id': config['actor_model_id'],
        'teacher_model_id': caller.model_id, 'items': candidate['items'],
        'learned_query_ids': sorted(set(config['query_ids']) | set(parent['learned_query_ids'])),
        'evidence_groups': evidence, 'exposure': exposure, 'parent_sha256': parent['sha256'],
        'rollout_policy_sha256': batch['policy_sha256'], 'rollout_kind': batch['rollout_kind'],
        'reward_mode': DENSE_REWARD, 'adaptations': ['continuous intent-slot reward', 'offline rollout reuse'],
        'source': config['source'], 'source_batch': config['source_batch'],
        'prepared': spec['prepared'], 'collected': spec['collected'], 'reference_candidate': record['candidate'],
        'reference_candidate_sha256': candidate['sha256'], 'contacts': receipts, 'accounting': accounting,
        'learning_usage': candidate['learning_usage'], 'plan': manifest['plan'],
        'usage_identity': sha({'prepared': spec['prepared'], 'reference_candidate_sha256': candidate['sha256']}),
        'usage_scope': 'one complete reference batch shared by all releases and evaluations; count once by usage_identity',
        'semantic_validation': 'pending', 'deployment_accepted': False, 'candidate_publishable': False,
        'native_learning_seal': False, 'new_model_calls': 0}
    if 'psd_target' in config:
        body['psd_target'] = config['psd_target']
    return {**body, 'sha256': sha(body)}


def external_request(row):
    import json
    return json.loads(row['trajectory'])['request']


def render(candidate, base, consumer_ids):
    from .tfgrpo_consumption import SEPARATOR
    require(candidate.get('schema') == PREVIEW_SCHEMA and candidate.get('strategy') == STRATEGY
            and candidate.get('sha256') == sha({k: v for k, v in candidate.items() if k != 'sha256'})
            and candidate.get('semantic_validation') == 'pending'
            and all(candidate.get(k) is False for k in ('native_learning_seal', 'deployment_accepted', 'candidate_publishable')),
            'unaltered pending multistage reference preview required')
    items = candidate.get('items')
    require(isinstance(items, dict) and items and all(isinstance(k, str) and k and isinstance(v, str)
            and v.strip() for k, v in items.items()), 'nonempty reference experiences required')
    require(not set(consumer_ids) & set(candidate['learned_query_ids']), 'reference learning exposure cannot consume candidate')
    return base + SEPARATOR + '\n'.join(f'[{k}]. {v}' for k, v in sorted(items.items())) + '\n'


def verify_source(params, *, repo, data, candidate):
    require(compile_preview(read_pin(data, params['source_manifest'])) == candidate, 'reference preview differs from source replay')
    require(candidate['model_id'] == params['model_id'], 'reference frozen actor differs')
    return verify_reviewed_development(params, repo=repo, data=data, candidate=candidate,
        release_schema=SCHEMA, review_schema=REVIEW_SCHEMA, evidence=candidate['evidence_groups'])


def prepare(**kwargs):
    from .tfgrpo_external_release import prepare_reviewed
    return prepare_reviewed(kind=KIND, renderer=render, source_verifier=verify_source, **kwargs)
