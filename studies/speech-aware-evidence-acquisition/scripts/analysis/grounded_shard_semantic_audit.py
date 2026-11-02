"""Export exact TRAIN prediction/reference differences for grounded fusion.

This CPU projection verifies supplied batch bytes and scored-answer hashes.
Native source/seal verification remains the caller's responsibility. It neither
generates a candidate nor certifies that a natural-language rule is entailed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def raw_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def answer(value, label):
    require(isinstance(value, str), label + ' must retain original text')
    result = json.loads(value)
    require(isinstance(result, dict) and
            all(isinstance(result.get(k), str) and result[k] for k in ('scenario', 'action')) and
            isinstance(result.get('entities'), list), label + ' has invalid task fields')
    for entity in result['entities']:
        require(isinstance(entity, dict) and
                all(isinstance(entity.get(k), str) for k in ('type', 'filler')),
                label + ' has invalid entity')
    return {k: result[k] for k in ('scenario', 'action', 'entities')}


def entity_counter(value):
    return Counter((e['type'], e['filler']) for e in value['entities'])


def entities(counter):
    return [{'type': key[0], 'filler': key[1]} for key in sorted(counter)
            for _ in range(counter[key])]


def task_difference(prediction, reference):
    predicted, expected = entity_counter(prediction), entity_counter(reference)
    return {
        'intent_changed': (prediction['scenario'], prediction['action']) !=
                          (reference['scenario'], reference['action']),
        'missing_entities': entities(expected - predicted),
        'extra_entities': entities(predicted - expected),
        'matched_entities': entities(predicted & expected),
    }


def export_evidence(batch_bytes: bytes, *, expected_sha256: str):
    require(raw_sha(batch_bytes) == expected_sha256, 'source batch bytes changed')
    batch = json.loads(batch_bytes)
    require(isinstance(batch.get('groups'), list) and batch['groups'], 'groups required')
    parent = batch['parent']
    require(parent['sha256'] == raw_sha(canonical({k: v for k, v in parent.items() if k != 'sha256'})),
            'parent experience hash differs')
    groups, seen, trajectory_ids = [], set(), set()
    for index, group in enumerate(batch['groups']):
        qid = group['query_id']
        require(isinstance(qid, str) and qid.startswith('slurp:train:') and qid not in seen,
                'unique TRAIN group required')
        seen.add(qid)
        query = json.loads(group['query'])
        require(query.get('sample_id') == qid and query.get('audio_available_to_reflector') is False
                and isinstance(query.get('history'), list), 'query identity or observation differs')
        reference = answer(group['answer'], 'reference')
        variants, predictions = {}, []
        for rollout in group['rollouts']:
            tid = rollout['trajectory_id']
            require(isinstance(tid, str) and tid not in trajectory_ids, 'unique trajectory required')
            trajectory_ids.add(tid)
            require(rollout['model_id'] == batch['model_id'] and
                    rollout['policy_sha256'] == batch['policy_sha256'] and
                    rollout['experience_sha256'] == parent['sha256'], 'rollout policy differs')
            trajectory, critique = json.loads(rollout['trajectory']), json.loads(rollout['critique'])
            scored_text = trajectory['scored_answer_after_fixed_normalization']
            require(isinstance(scored_text, str) and raw_sha(scored_text.encode()) ==
                    critique['scored_answer_sha256'], 'scored answer hash differs')
            prediction = answer(scored_text, 'scored prediction')
            difference = task_difference(prediction, reference)
            tp, fp, fn = (len(difference[k]) for k in
                          ('matched_entities', 'extra_entities', 'missing_entities'))
            require(all(type(critique.get(k)) in (int, float) and math.isfinite(critique[k])
                        and critique[k] == count for k, count in
                        [('slot_f1_exact_tp', tp), ('slot_f1_exact_fp', fp), ('slot_f1_exact_fn', fn)]),
                    'scored entity counts differ')
            intent = int(not difference['intent_changed'])
            require(critique['scenario_action_exact'] == intent, 'scored intent differs')
            dense = 0.5 * intent + 0.5 * (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1)
            require(type(rollout['reward']) in (int, float) and math.isfinite(rollout['reward'])
                    and abs(rollout['reward'] - dense) < 1e-12
                    and critique['reward_mode'] == 'intent-exact-slot-f1-v1'
                    and critique['reward'] == rollout['reward'], 'fixed dense reward differs')
            require(critique['strict_task_success'] == int(intent == 1 and fp == 0 and fn == 0),
                    'strict task label differs')
            key = raw_sha(canonical(prediction))
            if key not in variants:
                variants[key] = len(predictions)
                predictions.append({'prediction_sha256': key, 'prediction': prediction,
                                    'difference': difference, 'reward': rollout['reward'],
                                    'strict_task_success': critique['strict_task_success'],
                                    'rollout_sources': []})
            predictions[variants[key]]['rollout_sources'].append({
                'trajectory_id': tid,
                'trajectory_sha256': raw_sha(rollout['trajectory'].encode()),
                'raw_response_sha256': raw_sha(trajectory['response_text'].encode()),
                'scored_answer_sha256': critique['scored_answer_sha256'],
                'critique_sha256': raw_sha(rollout['critique'].encode()),
            })
        require(predictions, 'group requires observed rollouts')
        groups.append({
            'group_index': index, 'query_id': qid,
            'reference': reference, 'reference_text_sha256': raw_sha(group['answer'].encode()),
            'query_text_sha256': raw_sha(group['query'].encode()),
            'perception_history': query['history'],
            'audio_available_to_reflector': False,
            'omitted_query_fields': sorted(set(query) - {'sample_id', 'history', 'audio_available_to_reflector'}),
            'prediction_variants': predictions,
            'rollout_count': len(group['rollouts']),
            'group_reward_has_contrast': len({p['reward'] for p in predictions}) > 1,
            'causal_error_attribution': 'not_established_by_reward_or_shared_perception',
        })
    body = {
        'schema': 'saea-grounded-semantic-evidence-v1',
        'source_batch_sha256': expected_sha256, 'model_id': batch['model_id'],
        'policy_sha256': batch['policy_sha256'], 'parent_sha256': parent['sha256'],
        'native_custody_verified_here': False, 'new_model_calls': 0,
        'groups': groups, 'group_count': len(groups),
        'rollout_count': sum(g['rollout_count'] for g in groups),
        'distinct_prediction_variants': sum(len(g['prediction_variants']) for g in groups),
        'candidate_publishable': False,
    }
    return {**body, 'sha256': raw_sha(canonical(body))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = export_evidence(args.batch.read_bytes(), expected_sha256=args.expected_sha256)
    with args.out.open('xb') as stream:
        stream.write(canonical(result) + b'\n')
    print(json.dumps({k: result[k] for k in ('schema', 'sha256', 'group_count',
                    'rollout_count', 'distinct_prediction_variants', 'new_model_calls')}))
