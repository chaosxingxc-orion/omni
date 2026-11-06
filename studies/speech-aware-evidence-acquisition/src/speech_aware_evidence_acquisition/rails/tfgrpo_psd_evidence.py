"""Additional PSD fused-learning provenance, based on the full original batch."""
import json
from . import external_exploration as external
from .tfgrpo_learning import sha, DENSE_REWARD, intent_slot_reward
from .grounded_synthesis_fusion import project_group_evidence


def verify(context, document, selected):
    batch = external.read(external.checked(context['psd']['source_batch']))
    if (document.get('source_batch_sha256') != sha(batch) or batch['model_id'] != context['model_id']
        or batch['parent'] != context['parent'] or batch['policy_sha256'] != context['rollout_policy_sha256']
        or batch['rollout_kind'] != context['rollout_kind'] or batch['common']['reward_mode'] != DENSE_REWARD):
        raise ValueError('PSD source batch provenance differs')
    from .tfgrpo_psd_parent import verify as verify_parent
    verify_parent(context['parent'], context['psd']['target'], context['model_id'], context['psd'].get('parent_binding'))
    originals = {g['query_id']: g for g in batch['groups']}
    if len(originals) != len(batch['groups']): raise ValueError('duplicate source groups')
    source_groups = {g['query_id']: g for g in document['groups']}
    seen = set()
    for g in selected:
        qid = g['query_id']; original = originals[qid]; source = source_groups[qid]
        dictionary = source['evidence'].get('development_evidence_dictionary', {})
        expanded = external.expand_evidence({k:v for k,v in source['evidence'].items() if not k.startswith('development_evidence_')}, dictionary)
        if (sha(original) != source['group_sha256'] or g['group_sha256'] != source['group_sha256']
            or source['evidence'] != project_group_evidence(original, evidence_view='full-query-v1') or g['evidence'] != expanded):
            raise ValueError('PSD original group or projection differs')
        if len(original['rollouts']) < 2: raise ValueError('same-policy multiple rollouts required')
        for row in original['rollouts']:
            if (row['trajectory_id'] in seen or row['model_id'] != batch['model_id'] or row['policy_sha256'] != batch['policy_sha256'] or row['experience_sha256'] != batch['parent']['sha256']):
                raise ValueError('rollout actor/policy/parent differs')
            seen.add(row['trajectory_id'])
            traj=json.loads(row['trajectory']); feedback=json.loads(row['critique'])
            import hashlib
            if (traj['request']['sample_id'] != qid or not isinstance(traj['response_text'],str)
                or feedback['reward_mode'] != DENSE_REWARD or intent_slot_reward(feedback) != row['reward']
                or feedback['reward'] != row['reward'] or hashlib.sha256(traj['scored_answer_after_fixed_normalization'].encode()).hexdigest() != feedback['scored_answer_sha256']):
                raise ValueError('PSD trajectory or reward differs')
