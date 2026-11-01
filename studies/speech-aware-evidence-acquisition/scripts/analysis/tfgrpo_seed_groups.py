"""Export verified seed-only K4 groups for shared-P TF-GRPO learning.

No generation, tokenizer or scorer calls. Gold TRAIN labels are learning-only.
Half outputs stay separate; candidates learned on one half are evaluated on the
other. This text projection cannot judge the original audio independently.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis'), str(ROOT/'scripts/flights/nx')]
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import (canonical, sha, empty_experience,
    BINARY_REWARD, DENSE_REWARD, validate_reward_mode, intent_slot_reward, group_has_contrast)
from tfrl_recovered_consumers import load
from tfrl_current_half_result import score_table, ARMS
from tfrl_uncertain_credit import checked
from tfrl_seed_request_audit import verify


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_projection(payload):
    """Preserve exact text conditioning; replace media bytes with hash custody."""
    result = copy.deepcopy(payload)
    for field in ('raw_speech_query', 'speech_feature_parts'):
        if field in result:
            result[field] = {'omitted_for_text_learning': True,
                             'original_sha256': sha(result[field]),
                             'reason': 'Audio unavailable to text reflector; no independent listening claim.'}
    return result


def group_from_row(row, trace, scores, gold, *, model_id, policy_sha256, experience_sha256, namespace,
                   reward_mode=BINARY_REWARD, scored_answers=None):
    validate_reward_mode(reward_mode)
    sid = row['sample_id']
    visits = [v for v in row['native_episode_receipt']['role_visits'] if v.get('stage') == 'solver']
    if len(visits) != 1 or len(visits[0]['execution_receipt']['contacts']) != 6:
        raise ValueError('one sealed six-arm Solver visit required')
    contacts = visits[0]['execution_receipt']['contacts']
    rollouts, custody = [], []
    query = None
    for arm, contact in zip(ARMS[:4], contacts[:4]):
        rid = contact['request_id']
        request, response = trace[rid+'/request'], trace[rid+'/response']
        payload, answer = request['payload'], response['text']
        # The bundle's own payload hash uses canonical JSON from the shared rails.
        from speech_aware_evidence_acquisition.rails.hashing import content_hash
        if (payload['sample_id'] != sid or content_hash(payload) != request['payload_sha256']
                or request['payload_sha256'] != contact['request_sha256']
                or hashlib.sha256(answer.encode()).hexdigest() != contact['response_text_sha256']
                or response['response_text_sha256'] != contact['response_text_sha256']):
            raise ValueError('exact sealed request/response custody required')
        score = scores[arm][sid]
        strict_reward = int(not checked(score)['task_error'])
        reward = strict_reward if reward_mode == BINARY_REWARD else intent_slot_reward(score)
        critique = {k:score[k] for k in ('scenario_action_exact', 'slot_f1_exact_tp',
                                        'slot_f1_exact_fp', 'slot_f1_exact_fn')}
        critique.update(reward=reward, reward_rule='Intent exact AND exact-slot FP=FN=0',
                        limitation='Shared P; text reflection cannot verify audio or infer unobserved reasoning.')
        if reward_mode == DENSE_REWARD:
            critique.update(reward_rule='0.5 * IntentExact + 0.5 * ExactSlotF1; empty slots => 1',
                strict_task_success=strict_reward, reward_mode=reward_mode,
                reward_scope='Official scoring of the source arm after its fixed delivery normalization; raw response alone is not scored.',
                partial_credit_is_not_success=True)
        projected = text_projection(payload)
        if query is None:
            query = json.dumps({'task_instruction': payload['task_instruction'],
                'history': payload['history'][:-1], 'supplied_evidence': payload['supplied_evidence'],
                'sample_id': sid, 'audio_available_to_reflector': False}, ensure_ascii=False, sort_keys=True)
        trajectory = {'request':projected, 'response_text':answer}
        if reward_mode == DENSE_REWARD:
            if scored_answers is None or not isinstance(scored_answers[arm][sid],str):
                raise ValueError('dense reflection requires exact normalized scored answer')
            trajectory['scored_answer_after_fixed_normalization'] = scored_answers[arm][sid]
            critique['scored_answer_sha256'] = hashlib.sha256(scored_answers[arm][sid].encode()).hexdigest()
        trajectory_id = sha({'namespace':namespace, 'request_id':rid,
                             'request_sha256':request['payload_sha256']})
        rollouts.append({'trajectory_id':trajectory_id,
            'trajectory':json.dumps(trajectory, ensure_ascii=False, sort_keys=True),
            'critique':json.dumps(critique,sort_keys=True), 'reward':reward, 'model_id':model_id,
            'policy_sha256':policy_sha256, 'experience_sha256':experience_sha256})
        custody.append({'trajectory_id':trajectory_id,'arm':arm,'namespace':namespace,
                        'request_id':rid,'request_sha256':request['payload_sha256'],
                        'response_text_sha256':contact['response_text_sha256'],
                        'official_score_sha256':sha(score),
                        **({'scored_answer_sha256':critique['scored_answer_sha256']} if reward_mode==DENSE_REWARD else {})})
    if not group_has_contrast(rollouts, reward_mode):
        raise ValueError('export only binary mixed groups' if reward_mode == BINARY_REWARD else 'export only reward-contrast groups')
    return {'query_id':sid, 'query':query, 'answer':json.dumps(gold,ensure_ascii=False,sort_keys=True),
            'rollouts':rollouts}, custody


def gold_train(lock, data, wanted):
    from speech_aware_evidence_acquisition.data.slurp import load_slurp_train_material, _governed_roots
    _, _, metadata = _governed_roots(lock, data)
    path = metadata/'train.jsonl'; digest = file_sha(path)
    result = {}
    for row in load_slurp_train_material(lock, data):
        for recording in row.recordings:
            if recording.sample_id in wanted:
                result[recording.sample_id] = {'scenario':row.scenario, 'action':row.action,
                    'entities':[{'type':e.type,'filler':e.filler} for e in row.entities]}
    if set(result) != wanted or file_sha(path) != digest:
        raise ValueError('complete immutable TRAIN labels required')
    return result, {'relpath':str(path.relative_to(data)), 'sha256':digest,
                    'reference_transcripts_exported':False, 'runtime_use_forbidden':True}


def export(source, data, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.data.lock import load_lock, umbrella_lock_path
    assert_execution_environment()
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned output required')
    recipe, roots, halves = load(source, data)
    ids = {sid for part in halves.values() for sid in (r['row_id'] for r in part['plan']['row_plan'])}
    if len(ids) != 625: raise ValueError('exact isolated625 source required')
    lock_path = umbrella_lock_path(); gold, gold_pin = gold_train(load_lock(lock_path), data, ids)
    documents, report = {}, {'schema':'saea-tfgrpo-seed-groups-v1', 'rollout_kind':'shared-perception-adaptation',
        'sources':{h:p['source'] for h,p in halves.items()}, 'gold':gold_pin,
        'datasets_lock_sha256':file_sha(lock_path), 'halves':{}, 'new_model_calls':0,
        'new_tokenizer_calls':0, 'new_scoring_jobs':0,
        'limitation':'Text projection of fixed-P Solver trajectories, not independent full P/S rollouts.'}
    model_ids = set()
    for half, part in sorted(halves.items()):
        own = [r['row_id'] for r in part['plan']['row_plan']]
        tables = {a:score_table(part['results'][a],own) for a in ARMS[:4]}
        mixed = {sid for sid in own if len({int(not checked(tables[a][sid])['task_error']) for a in ARMS[:4]}) == 2}
        model_pin = recipe['sources'][half]['model']; model_id = Path(model_pin['relpath']).stem
        model_ids.add(model_id)
        values = part['config'].values
        if part['plan']['iteration'] != 1 or part['plan']['banks']['bank_receipt_sha256'] is not None:
            raise ValueError('unbanked initial experience required')
        # Keep actual source policy declaration, rather than assigning an arbitrary common hash.
        policy = {'model':model_pin, 'policy':values['policy'], 'native_episode':values['native_episode'],
                  'knowledge_stores':values['knowledge_stores'], 'model_decoding':values['decoding_params'],
                  'task_instruction_template':values['task_instruction_template'],
                  'rollout_kind':'shared-perception-adaptation'}
        policy_hash = sha(policy); parent = empty_experience(model_id)
        groups, custody, seen = [], [], set()
        segments = part.get('segments',[{'namespace':half,'plan':part['plan'],'rows':part.get('bundle')}])
        for segment in segments:
            bundle = segment['rows']; verify(segment['plan'],bundle.outputs_rows,bundle.trace_records)
            for row in bundle.outputs_rows:
                sid = row['sample_id']
                if sid not in mixed: continue
                if sid in seen: raise ValueError('source union repeats mixed sample')
                seen.add(sid)
                group, pins = group_from_row(row,bundle.trace_records,tables,gold[sid],model_id=model_id,
                    policy_sha256=policy_hash,experience_sha256=parent['sha256'],namespace=segment['namespace'])
                groups.append(group); custody.extend(pins)
        if seen != mixed: raise ValueError('missing mixed source rows')
        groups.sort(key=lambda g:g['query_id'])
        document = {'schema':'saea-tfgrpo-seed-half-v1','source_half':half,
            'evaluation_half':'h2' if half=='h1' else 'h1', 'model_id':model_id,
            'rollout_kind':'shared-perception-adaptation','policy':policy,'policy_sha256':policy_hash,
            'parent':parent,'groups':groups,'source':part['source'],'gold':gold_pin,'custody':custody}
        documents[half+'.json'] = document
        report['halves'][half] = {'source_rows':len(own),'mixed_groups':len(groups),
            'rollout_contacts':4*len(groups),'planned_learning_calls':6*len(groups)+1,
            'model_id':model_id,'policy_sha256':policy_hash,'experience_sha256':parent['sha256'],
            'group_query_ids_sha256':sha([g['query_id'] for g in groups])}
    if len(model_ids) != 1: raise ValueError('same frozen model required')
    report['artifacts'] = {name:hashlib.sha256(canonical(doc)).hexdigest() for name,doc in documents.items()}
    out.mkdir(parents=True)
    for name,doc in documents.items(): publish_new(out/name,canonical(doc))
    publish_new(out/'complete.json',canonical(report))
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','data','out'):p.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(export(**vars(p.parse_args())),sort_keys=True))
