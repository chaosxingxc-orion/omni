"""Deterministic relative evidence and bounded external experience operations.

A fused, offline TF-GRPO-inspired proposal operator, not the native multistage
reference implementation. Never creates a native seal or deployment approval.
"""
import copy
import math
import re

from .tfgrpo_learning import (DENSE_REWARD, LearningError, apply_operations,
                             canonical, intent_slot_reward, parse_operations,
                             sha, validate_experience, validate_reward)

CONTEXT_SCHEMA = 'saea-tfgrpo-external-context-v1'
STRATEGY = 'tfgrpo-external-relative-operations-v1'


def require(ok, message):
    if not ok:
        raise LearningError(message)


def relative_evidence(groups):
    """Keep every sampled trajectory, including identical predictions/rewards."""
    result = []
    seen_queries, seen_trajectories = set(), set()
    for group in groups:
        qid = group['query_id']
        require(isinstance(qid, str) and qid.startswith('slurp:train:') and qid not in seen_queries,
                'unique TRAIN groups required')
        seen_queries.add(qid)
        rows = group['evidence']['rollouts']
        require(isinstance(rows, list) and len(rows) >= 2, 'multiple complete rollouts required')
        values = []
        for row in rows:
            tid = row['trajectory_id']
            require(isinstance(tid, str) and tid and tid not in seen_trajectories,
                    'unique trajectory identities required; duplicate predictions remain separate')
            seen_trajectories.add(tid)
            value = row['reward']
            validate_reward(value, DENSE_REWARD)
            feedback = row['feedback']
            require(feedback.get('reward_mode') == DENSE_REWARD
                    and intent_slot_reward(feedback) == value and feedback.get('reward') == value,
                    'reward differs from fixed intent/slot feedback')
            values.append(float(value))
        mean = math.fsum(values) / len(values)
        std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))
        contrast = max(values) - min(values) > 1e-12
        result.append({'query_id': qid, 'group_sha256': group['group_sha256'],
                       'count': len(values), 'mean_reward': mean, 'population_std': std,
                       'has_contrast': contrast,
                       'rollouts': [{'trajectory_id': row['trajectory_id'], 'reward': value,
                                     'sample_weight': 1 / len(rows),
                                     'centered_reward': value - mean,
                                     'standardized_reward': (value - mean) / std if contrast else 0.0}
                                    for row, value in zip(rows, values)]})
    require(result, 'nonempty evidence required')
    return {'schema': 'saea-tfgrpo-relative-evidence-v1', 'reward_mode': DENSE_REWARD,
            'duplicate_predictions_retained': True, 'groups': result}


def validate_context(context):
    fields = {'schema', 'model_id', 'parent', 'reward_mode', 'rollout_policy_sha256',
              'rollout_kind', 'offline_stale_policy', 'bounds'}
    require(isinstance(context, dict) and set(context) in (fields, fields | {'psd'})
            and context['schema'] == CONTEXT_SCHEMA, 'exact TF-GRPO context required')
    if 'psd' in context:
        require(isinstance(context['psd'], dict) and set(context['psd']) in ({'target','source_batch'}, {'target','source_batch','parent_binding'}) and context['psd']['target'] in ('slot','decision'), 'explicit PSD target and batch required')
    require(context['reward_mode'] == DENSE_REWARD, 'explicit continuous reward required')
    validate_experience(context['parent'], context['model_id'])
    require(isinstance(context['rollout_policy_sha256'], str)
            and re.fullmatch('[a-f0-9]{64}', context['rollout_policy_sha256']), 'rollout policy pin required')
    require(context['rollout_kind'] in ('shared-perception-adaptation', 'full-same-policy')
            and type(context['offline_stale_policy']) is bool, 'explicit rollout semantics required')
    bounds = context['bounds']
    require(isinstance(bounds, dict) and set(bounds) == {'max_operations', 'max_items', 'max_characters'}
            and all(type(v) is int and v > 0 for v in bounds.values()), 'positive operation/library bounds required')
    return context


def prepare_context(context, source_document):
    validate_context(context)
    for key in ('model_id', 'parent', 'reward_mode', 'rollout_policy_sha256', 'rollout_kind', 'offline_stale_policy'):
        require(context[key] == source_document[key], 'TF-GRPO context differs from pinned source: ' + key)
    return copy.deepcopy(context)


def validate_proposal(candidate, context, ledger):
    validate_context(context)
    fields = {'id', 'target', 'parent_sha256', 'evidence_query_ids', 'relative_feedback',
              'hypothesis', 'operations', 'counterexample', 'validation'}
    require(isinstance(candidate, dict) and set(candidate) == fields, 'exact TF-GRPO operation proposal required')
    target = context.get('psd', {}).get('target')
    require(candidate['target'] == (target + '_experience_library' if target else 'solver_experience_library'), 'exact experience target required')
    require(candidate['parent_sha256'] == context['parent']['sha256'], 'external experience parent differs')
    for field in ('id', 'hypothesis', 'counterexample', 'validation'):
        require(isinstance(candidate[field], str) and candidate[field].strip(), 'nonempty ' + field + ' required')
    refs = candidate['evidence_query_ids']
    require(isinstance(refs, list) and refs and all(isinstance(q, str) for q in refs)
            and refs == sorted(set(refs)), 'sorted distinct evidence groups required')
    groups = {g['query_id']: g for g in ledger['groups']}
    require(set(refs) <= set(groups), 'proposal cites unsupplied evidence')
    contrasts = candidate['relative_feedback']
    require(isinstance(contrasts, list) and len(contrasts) == len(refs), 'one explicit contrast per cited group required')
    seen = set()
    for contrast in contrasts:
        require(isinstance(contrast, dict) and set(contrast) ==
                {'query_id', 'higher_trajectory_id', 'lower_trajectory_id', 'explanation'}, 'exact contrast reference required')
        qid = contrast['query_id']
        require(qid in refs and qid not in seen, 'unique supplied contrast group required')
        seen.add(qid)
        group = groups[qid]
        require(group['has_contrast'], 'tied groups cannot provide relative learning signal')
        rows = {r['trajectory_id']: r for r in group['rollouts']}
        high, low = contrast['higher_trajectory_id'], contrast['lower_trajectory_id']
        require(high in rows and low in rows and rows[high]['reward'] > rows[low]['reward'] + 1e-12,
                'claimed high/low ordering contradicts observed reward')
        require(isinstance(contrast['explanation'], str) and contrast['explanation'].strip(), 'contrast explanation required')
    operations = parse_operations(canonical(candidate['operations']).decode(), context['parent']['items'],
                                  max_operations=context['bounds']['max_operations'])
    require(operations and all(o['operation'] != 'NONE' for o in operations), 'nonempty actual library edits required')
    for operation in operations:
        require(operation['operation'] != 'DELETE' or operation['content'] == '',
                'DELETE content must be empty')
        require(not re.search(r'slurp:(?:train|test|devel):', operation['content']),
                'runtime experience cannot contain sample identity')
    items = apply_operations(context['parent']['items'], operations,
                             max_items=context['bounds']['max_items'], max_characters=context['bounds']['max_characters'])
    require(items != context['parent']['items'], 'proposal must change the library')
    return items
