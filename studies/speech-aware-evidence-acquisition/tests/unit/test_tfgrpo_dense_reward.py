"""Partial-credit update, tie handling, accounting and resume invariants."""
import copy
import json
import pytest
from test_tfgrpo_learning import Model, group, MODEL, POLICY, LEARNING_CONFIG
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import (
    ExperienceLearner, LearningError, empty_experience, intent_slot_reward,
    DENSE_REWARD, validate_reward, group_has_contrast,
)


def run(path, model, groups, **kwargs):
    learner = ExperienceLearner(path, model_id=MODEL, metered_call=model,
        learning_config_sha256=LEARNING_CONFIG, max_calls=20)
    return learner.learn(groups, empty_experience(MODEL), agent_objective='task',
        learning_objective='Compare partial correctness; no fully correct answer implied.',
        policy_sha256=POLICY, reward_mode=DENSE_REWARD, **kwargs)


def test_all_binary_failures_with_partial_difference_learn_and_resume(tmp_path):
    groups = [group(rewards=(0.25, 0.75)), group(qid='tie', rewards=(0.5, 0.5))]
    model = Model()
    result = run(tmp_path, model, groups)
    assert result['contrast_groups'] == 1
    assert result['all_below_full_success_contrast_groups'] == 1
    assert result['skipped_tied_groups'] == 1
    assert result['learned_query_ids'] == ['q1']
    assert result['learning_usage']['calls'] == 5
    assert 'mixed_groups' not in result
    assert result['reward_mode'] == DENSE_REWARD
    assert not result['deployment_accepted']
    text = '\n'.join(m['content'] for r in model.requests if r['stage']=='advantage' for m in r['messages'])
    assert 'Reward 0.25' in text and 'Reward 0.75' in text
    assert run(tmp_path, lambda _: pytest.fail('no new contacts on resume'), groups) == result
    changed = copy.deepcopy(groups); changed[0]['rollouts'][0]['reward'] = 0.2
    with pytest.raises(LearningError, match='batch binding changed'):
        run(tmp_path, lambda _: pytest.fail('changed reward must not resume'), changed)


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), -0.01, 1.01, '0.5'])
def test_invalid_partial_rewards_fail_before_any_contact(tmp_path,value):
    with pytest.raises(LearningError):
        run(tmp_path, lambda _: pytest.fail('invalid reward contact'), [group(rewards=(value, 0.5))])


def test_dense_does_not_reuse_binary_learning_stages(tmp_path):
    with pytest.raises(LearningError,match='cannot import'):
        run(tmp_path, lambda _:pytest.fail('must refuse'), [group(rewards=(.25,.75))], summary_import={'sha256':'x'})


def test_reward_counts_and_no_entity_convention():
    def score(intent,tp,fp,fn):
        return {'scenario_action_exact':intent,'slot_f1_exact_tp':tp,'slot_f1_exact_fp':fp,'slot_f1_exact_fn':fn}
    assert intent_slot_reward(score(1,0,0,0)) == 1
    assert intent_slot_reward(score(0,0,0,0)) == .5
    assert intent_slot_reward(score(1,1,1,1)) == .75
    assert intent_slot_reward(score(0,0,1,1)) == 0
    for s in [score(1,-1,0,0),score(1,float('nan'),0,0),score(True,0,0,0),score(1,.5,0,0)]:
        with pytest.raises(LearningError):intent_slot_reward(s)
    assert not group_has_contrast([{'reward':.4},{'reward':.4}],DENSE_REWARD)


def test_registered_dense_source_executes_same_binding_and_refuses_tied_contacts(tmp_path,monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from test_tfgrpo_source import setup
    from speech_aware_evidence_acquisition.rails import tfgrpo_runtime
    from speech_aware_evidence_acquisition.rails.tfgrpo_source import load_dynamic_source
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
    values,args,source=setup(tmp_path)
    batch=source['batch'];batch['common']['reward_mode']=DENSE_REWARD
    for r,v in zip(batch['groups'][0]['rollouts'],(.25,.75)):r['reward']=v
    def repin():
        path=tmp_path/'source.json';path.write_text(json.dumps(source))
        values['prepared_training_feedback']['source']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        values['tfgrpo_learning_registration']['batch_sha256']=sha(batch)
    repin();assert load_dynamic_source(values,**args)[1]['contacts']==5
    model=Model();model_result=model.__call__
    class Caller:
        client=SimpleNamespace(close=lambda:None)
        def preflight(self,request):assert request['batch_sha256']==sha(batch)
        def reconcile(self,request):return None
        def __call__(self,request):
            result=model_result(request);result['model_id']='frozen-model';return result
    monkeypatch.setattr(tfgrpo_runtime,'bind_runtime',lambda *a,**k:Caller())
    runtime=SimpleNamespace(_composed=SimpleNamespace(values=values),plan=args['plan'],_repo_root=tmp_path,
        _data_root=tmp_path,_samples=[SimpleNamespace(sample_id=args['sample_ids'][0])])
    candidate=tfgrpo_runtime.run_registered_learning(runtime,receipt_root=tmp_path/'receipts',learner_root=tmp_path/'learner')
    assert candidate['contrast_groups']==1 and len(model.requests)==5
    batch['groups'][0]['rollouts'][1]['reward']=.25;repin()
    with pytest.raises(LearningError,match='finite learning contact roster'):
        load_dynamic_source(values,**args)
