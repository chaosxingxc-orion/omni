"""External TF-GRPO proposals preserve statistics and cannot impersonate native learning."""
import copy
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts/analysis'))
import external_exploration as external
import tfgrpo_external_compile as compiler
from speech_aware_evidence_acquisition.rails.tfgrpo_external_proposals import (
    relative_evidence, validate_proposal, prepare_context, CONTEXT_SCHEMA)
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import DENSE_REWARD, empty_experience, validate_experience
from test_external_exploration import job, write
from test_external_teacher import network


def fixture():
    rows = []
    # Two identical low-score outputs are distinct samples, not deduplicated.
    for i, (reward, intent) in enumerate([(0.5, 0), (0.5, 0), (1.0, 1)]):
        feedback = {'reward_mode': DENSE_REWARD, 'reward': reward, 'scenario_action_exact': intent,
                    'slot_f1_exact_tp': 1, 'slot_f1_exact_fp': 0, 'slot_f1_exact_fn': 0}
        rows.append({'trajectory_id': 'trajectory-' + str(i), 'reward': reward, 'feedback': feedback,
                     'source_trajectory': {'request': {'history': []}}, 'scored_prediction': {'same': intent}})
    groups = [{'query_id': 'slurp:train:one', 'group_sha256': 'a' * 64,
               'evidence': {'query': {'supplied_evidence': [], 'task_instruction': 'task'}, 'rollouts': rows}}]
    context = {'schema': CONTEXT_SCHEMA, 'model_id': 'frozen-actor', 'parent': empty_experience('frozen-actor'),
               'reward_mode': DENSE_REWARD, 'rollout_policy_sha256': 'b' * 64,
               'rollout_kind': 'shared-perception-adaptation', 'offline_stale_policy': False,
               'bounds': {'max_operations': 2, 'max_items': 4, 'max_characters': 2000}}
    candidate = {'id': 'conditional-rule', 'target': 'solver_experience_library',
                 'parent_sha256': context['parent']['sha256'], 'evidence_query_ids': ['slurp:train:one'],
                 'relative_feedback': [{'query_id': 'slurp:train:one', 'higher_trajectory_id': 'trajectory-2',
                                        'lower_trajectory_id': 'trajectory-0', 'explanation': 'higher reward association, not causality'}],
                 'hypothesis': 'Use explicit evidence', 'operations': [{'operation': 'ADD', 'id': None,
                 'content': 'If evidence is contradictory, compare the action cue; retain uncertainty when absent.'}],
                 'counterexample': 'Ambiguous action', 'validation': 'Paired repairs and breaks'}
    return groups, context, candidate


def test_relative_weights_preserve_duplicate_samples():
    groups, context, candidate = fixture()
    ledger = relative_evidence(groups); g = ledger['groups'][0]
    assert g['count'] == 3 and g['mean_reward'] == pytest.approx(2/3)
    assert len(g['rollouts']) == 3 and [r['sample_weight'] for r in g['rollouts']] == [1/3]*3
    assert sum(r['centered_reward'] for r in g['rollouts']) == pytest.approx(0)
    assert len(validate_proposal(candidate, context, ledger)) == 1


@pytest.mark.parametrize('fault', ['order', 'foreign', 'tie', 'parent', 'bounds', 'sample_id', 'reward', 'duplicate_identity', 'bool_reward', 'legacy_prose'])
def test_reject_false_relative_feedback_and_unbounded_operations(fault):
    groups, context, c = fixture()
    if fault == 'order': c['relative_feedback'][0]['higher_trajectory_id'] = 'trajectory-1'
    elif fault == 'foreign': c['relative_feedback'][0]['lower_trajectory_id'] = 'foreign'
    elif fault == 'tie':
        groups[0]['evidence']['rollouts'][2]['reward'] = 0.5
        groups[0]['evidence']['rollouts'][2]['feedback'].update(reward=0.5, scenario_action_exact=0)
    elif fault == 'parent': c['parent_sha256'] = 'f' * 64
    elif fault == 'bounds': context['bounds']['max_characters'] = 1
    elif fault == 'sample_id': c['operations'][0]['content'] = 'Memorize slurp:train:one'
    elif fault == 'reward': groups[0]['evidence']['rollouts'][0]['reward'] = 0.7
    elif fault == 'duplicate_identity': groups[0]['evidence']['rollouts'][1]['trajectory_id'] = 'trajectory-0'
    elif fault == 'bool_reward': groups[0]['evidence']['rollouts'][0]['reward'] = True
    elif fault == 'legacy_prose': c['intervention'] = c.pop('operations')
    with pytest.raises(ValueError): validate_proposal(c, context, relative_evidence(groups))


def prepared_material(tmp_path, monkeypatch):
    groups, context, candidate = fixture()
    path, root = job(tmp_path); cfg = external.read(path)
    source_doc = {k: context[k] for k in ('model_id','parent','reward_mode','rollout_policy_sha256','rollout_kind','offline_stale_policy')}
    source_doc['groups'] = groups
    cfg['source'] = write(tmp_path/'source.json', {'document': source_doc})
    cfg['context'] = write(tmp_path/'context-input.json', context)
    cfg['profile'] = external.pin(external.ROOT/'configs/learners/tfrl/tfgrpo.json')
    write(path,cfg); external.prepare(path)
    second = copy.deepcopy(candidate); second['id'] = 'second'; second['operations'][0]['content'] = 'When an explicit cue is absent, retain uncertainty.'
    response = {'id': 'tfgrpo-external', 'model': 'external-model', 'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps({'algorithm': 'tfgrpo','candidates': [candidate, second]})}}],
                'usage': {'prompt_tokens': 90, 'completion_tokens': 80}}
    _,_,caller,request=external.load_prepared(root);network(caller,monkeypatch,[response]);caller(request);external.collect(root)
    spec={'schema':'saea-tfgrpo-external-compile-v1','prepared':external.pin(root/'prepared.json'),
          'collected':external.pin(root/'collected.json'),'candidate_index':0}
    return spec, root


def test_real_transport_receipt_compile_and_replay_no_calls(tmp_path,monkeypatch):
    spec,root=prepared_material(tmp_path,monkeypatch)
    preview=compiler.compile_preview(spec)
    assert preview['learning_usage']['calls']==1 and preview['relative_evidence']['groups'][0]['count']==3
    assert preview['teacher_model_id']=='external-model' and preview['model_id']=='frozen-actor'
    assert preview['learned_query_ids']==['slurp:train:one']
    assert preview['new_model_calls']==0 and preview['native_learning_seal'] is False
    assert preview['candidate_publishable'] is False
    assert compiler.compile_preview(spec)==preview
    # Pending previews cannot bypass native seal/runtime release by sharing its schema.
    with pytest.raises(ValueError):validate_experience(preview,'frozen-actor')


@pytest.mark.parametrize('fault',['proposal','wire','index_bool','source'])
def test_compiler_rechecks_custody_even_if_collected_rehashed(tmp_path,monkeypatch,fault):
    spec,root=prepared_material(tmp_path,monkeypatch)
    if fault=='proposal':
        d=external.read(root/'collected.json');d['proposals']['candidates'][0]['operations'][0]['content']='Forged'
        spec['collected']=write(root/'collected.json',d)
    elif fault=='wire':
        _,_,caller,request=external.load_prepared(root);directory,_=caller._binding(request)
        (directory/'response/response.bin').write_text('{}')
    elif fault=='index_bool':spec['candidate_index']=True
    else: (tmp_path/'source.json').write_text('{}')
    with pytest.raises(Exception):compiler.compile_preview(spec)


def test_context_cannot_substitute_parent_or_rollout_policy():
    _,context,_=fixture();source=copy.deepcopy(context);context['rollout_policy_sha256']='f'*64
    with pytest.raises(ValueError,match='pinned source'):prepare_context(context,source)


def test_identical_operations_with_different_names_are_not_multiple_strategies():
    groups,context,candidate=fixture();second=copy.deepcopy(candidate);second['id']='renamed'
    profile={'algorithm':'tfgrpo','min_candidates':2,'max_candidates':8}
    with pytest.raises(ValueError,match='duplicate TF-GRPO'):
        external.validate_proposals(json.dumps({'algorithm':'tfgrpo','candidates':[candidate,second]}),
                                    profile,context,['slurp:train:one'],relative=relative_evidence(groups))


def test_tied_group_ledger_has_zero_advantage_without_claiming_signal():
    groups,_,_=fixture()
    groups[0]['evidence']['rollouts'][2]['reward']=0.5
    groups[0]['evidence']['rollouts'][2]['feedback'].update(reward=0.5,scenario_action_exact=0)
    group=relative_evidence(groups)['groups'][0]
    assert not group['has_contrast'] and group['population_std']==0
    assert [r['standardized_reward'] for r in group['rollouts']]==[0,0,0]
