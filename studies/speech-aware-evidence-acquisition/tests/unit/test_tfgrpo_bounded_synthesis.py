import pytest
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import build_final_messages,render,LearningError
COMMON={'agent_objective':'Intent and slots','learning_objective':'Conditional evidence use','num_experiences':2}

def test_historical_prompt_bytes_are_preserved():
    material='{}'
    assert build_final_messages(COMMON,material,128,16000)==[
        {'role':'system','content':render('BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP',**COMMON)},
        {'role':'user','content':render('BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP',experiences_and_operations=material)}]

def test_new_synthesis_removes_one_per_input_constraint_and_budgets_output():
    common={**COMMON,'final_merge_policy':'bounded-synthesis-v1','final_merge_limits':{'max_items':128,'max_characters':16000}}
    messages=build_final_messages(common,'{"proposed_operations":[]}',128,16000)
    assert 'MUST exactly match the number' not in messages[0]['content']
    assert 'independent of the input count' in messages[0]['content']
    assert '10000 total Unicode' in messages[0]['content']
    assert '48 rules' in messages[0]['content']
    assert '128 rules and 16000' in messages[0]['content']
    assert 'applicability conditions and exceptions' in messages[0]['content']
    assert 'proposed_operations' in messages[1]['content']

@pytest.mark.parametrize('policy',['unknown','bounded-synthesis-v1'])
def test_synthesis_cannot_omit_exact_bounds(policy):
    with pytest.raises(LearningError):build_final_messages({**COMMON,'final_merge_policy':policy},'{}',128,16000)


def test_grounded_policy_requires_actual_train_evidence_and_preserves_old_bytes():
    from test_tfgrpo_learning import group
    g = group(qid='slurp:train:000001:audio-a')
    limits = {'max_items': 128, 'max_characters': 16000}
    old = {**COMMON, 'final_merge_policy': 'bounded-synthesis-v1', 'final_merge_limits': limits}
    assert build_final_messages(old, '{}', 128, 16000, groups=[g]) == build_final_messages(old, '{}', 128, 16000)
    new = {**old, 'final_merge_policy': 'evidence-grounded-synthesis-v1'}
    messages = build_final_messages(new, '{}', 128, 16000, groups=[g])
    assert g['answer'] in messages[1]['content']
    assert g['query'] in messages[1]['content']
    assert '"reward":0' in messages[1]['content']
    assert '"reward":1' in messages[1]['content']
    assert 'hypothesis, not ground truth' in messages[0]['content']
    with pytest.raises(LearningError, match='TRAIN groups'):
        build_final_messages(new, '{}', 128, 16000)
    for groups in ([g, g], [group(qid='slurp:test:000001:a')]):
        with pytest.raises(LearningError, match='TRAIN identities'):
            build_final_messages(new, '{}', 128, 16000, groups=groups)


def test_grounded_projection_does_not_repeat_retrieval_and_marks_omissions():
    import json
    from test_tfgrpo_learning import group
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import grounded_group_evidence
    g = group(qid='slurp:train:000001:a')
    g['query'] = json.dumps({'sample_id': g['query_id'], 'audio_available_to_reflector': False,
                            'history': [{'content': 'perception-content'}],
                            'supplied_evidence': ['large-retrieval-body'], 'task_instruction': 'task'})
    projected = grounded_group_evidence(g)
    assert projected['query']['perception_history'] == [{'content': 'perception-content'}]
    assert 'supplied_evidence' in projected['query']['omitted_fields']
    assert 'large-retrieval-body' not in json.dumps(projected)
    g['query'] = g['query'].replace('slurp:train:000001:a', 'slurp:test:000001:a')
    with pytest.raises(LearningError, match='provenance'):
        grounded_group_evidence(g)
