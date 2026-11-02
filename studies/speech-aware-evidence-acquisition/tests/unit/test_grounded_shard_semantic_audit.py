import copy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / 'scripts/analysis/grounded_shard_semantic_audit.py'
spec = importlib.util.spec_from_file_location('grounded_shard_semantic_audit', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    gold = {'scenario': 'calendar', 'action': 'set',
            'entities': [{'type': 'time', 'filler': 'five'}, {'type': 'timeofday', 'filler': 'morning'}]}
    pred = {'scenario': 'calendar', 'action': 'set',
            'entities': [{'type': 'time', 'filler': 'five'}, {'type': 'time', 'filler': 'five'}]}
    text = json.dumps(pred)
    parent = {'model_id': 'model', 'items': {}}
    parent['sha256'] = m.raw_sha(m.canonical(parent))
    critique = {'scored_answer_sha256': m.raw_sha(text.encode()),
                'scenario_action_exact': 1, 'slot_f1_exact_tp': 1,
                'slot_f1_exact_fp': 1, 'slot_f1_exact_fn': 1,
                'reward': 0.75, 'reward_mode': 'intent-exact-slot-f1-v1', 'strict_task_success': 0}
    rollout = {'trajectory_id': 'r1', 'model_id': 'model', 'policy_sha256': 'policy',
               'experience_sha256': parent['sha256'], 'reward': 0.75,
               'critique': json.dumps(critique),
               'trajectory': json.dumps({'request': {'private': 'omitted'},
                    'response_text': 'raw response differs from normalized',
                    'scored_answer_after_fixed_normalization': text})}
    query = {'sample_id': 'slurp:train:q', 'history': [], 'audio_available_to_reflector': False,
             'supplied_evidence': 'retained at source but omitted here'}
    return {'model_id': 'model', 'policy_sha256': 'policy', 'parent': parent,
            'groups': [{'query_id': query['sample_id'], 'query': json.dumps(query),
                        'answer': json.dumps(gold), 'rollouts': [rollout]}]}


def export(batch):
    blob = m.canonical(batch)
    return m.export_evidence(blob, expected_sha256=m.raw_sha(blob))


def test_duplicate_entities_preserve_multiset_error_and_raw_scored_distinction():
    result = export(fixture())
    variant = result['groups'][0]['prediction_variants'][0]
    assert variant['difference']['extra_entities'] == [{'type': 'time', 'filler': 'five'}]
    assert variant['difference']['missing_entities'] == [{'type': 'timeofday', 'filler': 'morning'}]
    assert variant['rollout_sources'][0]['raw_response_sha256'] != variant['rollout_sources'][0]['scored_answer_sha256']
    assert result['native_custody_verified_here'] is False
    assert result['candidate_publishable'] is False
    assert result['groups'][0]['omitted_query_fields'] == ['supplied_evidence']


def test_equal_predictions_deduplicate_content_but_preserve_each_trajectory():
    batch = fixture()
    other = copy.deepcopy(batch['groups'][0]['rollouts'][0])
    other['trajectory_id'] = 'r2'
    batch['groups'][0]['rollouts'].append(other)
    result = export(batch)
    assert result['rollout_count'] == 2
    assert result['distinct_prediction_variants'] == 1
    assert len(result['groups'][0]['prediction_variants'][0]['rollout_sources']) == 2
    assert not result['groups'][0]['group_reward_has_contrast']


@pytest.mark.parametrize('fault', ['batch_hash', 'scored_hash', 'counts', 'reward', 'query', 'test_group',
                                   'policy', 'parent', 'duplicate_trajectory'])
def test_rejects_changed_identity_scores_or_source(fault):
    batch = fixture()
    group = batch['groups'][0]
    rollout = group['rollouts'][0]
    if fault == 'batch_hash':
        with pytest.raises(ValueError, match='batch bytes'):
            m.export_evidence(m.canonical(batch), expected_sha256='0' * 64)
        return
    if fault in ('scored_hash', 'counts', 'reward'):
        critique = json.loads(rollout['critique'])
        if fault == 'scored_hash':
            critique['scored_answer_sha256'] = '0' * 64
        elif fault == 'counts':
            critique['slot_f1_exact_fp'] = 0
        else:
            rollout['reward'] = critique['reward'] = 1
        rollout['critique'] = json.dumps(critique)
    elif fault == 'query':
        query = json.loads(group['query'])
        query['sample_id'] = 'slurp:train:foreign'
        group['query'] = json.dumps(query)
    elif fault == 'test_group':
        group['query_id'] = 'slurp:test:q'
    elif fault == 'policy':
        rollout['policy_sha256'] = 'foreign'
    elif fault == 'parent':
        batch['parent']['items'] = {'changed': True}
    else:
        group['rollouts'].append(copy.deepcopy(rollout))
    with pytest.raises(ValueError):
        export(batch)
