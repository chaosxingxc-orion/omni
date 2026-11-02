"""One durable, metered final update; semantic review and deployment stay separate."""
from pathlib import Path
import hashlib
import json

from ..core.atomic_file import publish_new
from .tfgrpo_learning import (
    ExperienceLearner, LearningError, canonical, sha, validate_experience,
    apply_operations, parse_operations,
)

STAGE = 'grounded_final_update'
SCHEMA = 'saea-grounded-fusion-execution-v1'


def require(ok, message):
    if not ok:
        raise LearningError(message)


def execute_fusion(learner, binding, *, verify_source, validate_output):
    """Use the shared learner journal, preserving observed invalid responses.

    Callbacks verify pinned finalized sources and structure/support references.
    They do not score a proposed candidate's generalization.
    """
    require(isinstance(learner, ExperienceLearner) and callable(learner.preflight),
            'shared learner and native FIT required')
    require(callable(verify_source) and callable(validate_output), 'source and output validators required')
    require(binding.get('schema') == SCHEMA and learner.max_calls == 1,
            'one registered fusion contact required')
    require(binding['model_id'] == learner.model_id
            and binding['learning_config_sha256'] == learner.learning_config_sha256,
            'fusion generation identity differs')
    custody = verify_source()
    require(sha(custody) == binding['source_custody_sha256']
            and custody['unknown_contacts'] == 0, 'fusion source custody differs')
    historical = custody['original_total_usage']
    require(all(type(historical.get(k)) is int and historical[k] >= 0
                for k in ('calls', 'prompt_tokens', 'completion_tokens')),
            'complete historical fusion cost required')
    parent = binding['parent']
    validate_experience(parent, learner.model_id)
    ids = binding['learned_query_ids']
    require(isinstance(ids, list) and ids and len(set(ids)) == len(ids)
            and all(isinstance(q, str) and q.startswith('slurp:train:') for q in ids),
            'exact unique TRAIN evidence identities required')
    bounds = binding['bounds']
    require(set(bounds) == {'max_items', 'max_characters', 'max_operations'}
            and all(type(v) is int and v > 0 for v in bounds.values()),
            'finite fusion result bounds required')
    digest = sha(binding)
    contact_key = sha({'stage': STAGE, 'identity': binding['identity']})
    with learner._exclusive_batch():
        path = learner.root/'batch.json'
        if path.exists():
            require(path.read_bytes() == canonical(binding), 'fusion batch changed')
        else:
            publish_new(path, canonical(binding))
        learner._batch = digest
        learner.last_invocation = {'new_calls': 0, 'reused_calls': 0}
        expected = {'schema': 'saea-tfgrpo-learning-request-v1', 'batch_sha256': digest,
                    'model_id': learner.model_id, 'learning_config_sha256': learner.learning_config_sha256,
                    'stage': STAGE, 'identity': binding['identity'], 'messages': binding['messages']}
        for old in (learner.root/'contacts').glob('*.intent.json'):
            require(old.name == contact_key+'.intent.json' and old.read_bytes() == canonical(expected),
                    'foreign or changed final fusion contact')
        for old in (learner.root/'contacts').glob('*.response.json'):
            require(old.name == contact_key+'.response.json'
                    and (learner.root/'contacts'/(contact_key+'.intent.json')).exists(),
                    'orphan fusion response')
        text = learner._call(STAGE, binding['identity'], binding['messages'])
        checked = validate_output(text)
        require(isinstance(checked, dict) and isinstance(checked.get('operations'), list),
                'validated final operations required')
        operations = parse_operations(json.dumps(checked['operations']), parent['items'],
                                      max_operations=bounds['max_operations'])
        items = apply_operations(parent['items'], operations, max_items=bounds['max_items'],
                                 max_characters=bounds['max_characters'])
        accounting = learner.accounting()
        require(accounting['unknown_contacts'] == 0 and accounting['observed_contacts'] == 1,
                'one fully observed final fusion response required')
        body = {
            'schema': 'saea-tfgrpo-experience-v1', 'model_id': learner.model_id,
            'items': items, 'learned_query_ids': sorted(set(ids) | set(parent['learned_query_ids'])),
            'parent_sha256': parent['sha256'], 'batch_sha256': digest,
            'status': 'candidate', 'deployment_accepted': False,
            'rollout_kind': binding['rollout_kind'], 'reward_mode': binding['reward_mode'],
            'planned_learning_calls': 1, 'operations': operations,
            'learning_usage': accounting['known_usage'], 'accounting': accounting,
            'historical_fusion_custody': custody,
            'fusion_validation': checked,
            'semantic_review_required': True,
            'empty_candidate': not bool(items),
            'contact_response_sha256': {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((learner.root/'contacts').glob('*.response.json'))},
        }
        candidate = {**body, 'sha256': sha(body)}
        validate_experience(candidate, learner.model_id)
        result = learner.root/'candidate.json'
        if result.exists():
            require(result.read_bytes() == canonical(candidate), 'fusion candidate changed')
        else:
            publish_new(result, canonical(candidate))
        return candidate
